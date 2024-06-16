struct CameraUniforms {
    view: mat4x4<f32>,
    view_inv: mat4x4<f32>,
    proj: mat4x4<f32>,
    proj_inv: mat4x4<f32>,

    viewport: vec2<f32>,
    focal: vec2<f32>
};

struct RenderSettings {
    clipping_box_min: vec4<f32>,
    clipping_box_max: vec4<f32>,
    gaussian_scaling: f32,
    max_sh_deg: u32,
    show_env_map: u32,
    mip_spatting: u32,
    kernel_size: f32,
    walltime: f32,
    scene_extend: f32,
    current_colour_weight: f32, // could be interesting to have this as a ui parameter
    depth_smoothing_high: f32,
    colour_smoothing_high: f32,
    center: vec3<f32>,
};

struct ReprojectionData {
    vp_accu: mat4x4<f32>,
    reprojection: mat4x4<f32>,
};

@group(0) @binding(0) var<uniform> camera: CameraUniforms;

// add float32 to required features in device creation
// add sampler to bind group here
// sample with https://www.w3.org/TR/WGSL/#texturesamplelevel and mip lvl 0

// reprojection verify on the cpu

// variance calculation in two passes or with a mutex

@group(1) @binding(0) var currentFrameTexture: texture_2d<f32>;
@group(1) @binding(1) var currentFrameDepthTexture: texture_2d<f32>;

@group(1) @binding(2) var accuTexture: texture_2d<f32>;
@group(1) @binding(3) var accuDepth: texture_2d<f32>;

@group(1) @binding(4) var dstTexture: texture_storage_2d<rgba16float, read_write>;
@group(1) @binding(5) var dstDepth: texture_storage_2d<r32float, read_write>;

@group(1) @binding(6) var filter_sampler: sampler;

@group(1) @binding(7) var currentFrameDepthStatistics: texture_2d<f32>;
@group(1) @binding(8) var debug_output: texture_storage_2d<rgba32float, read_write>;

@group(2) @binding(0) var<uniform> render_settings: RenderSettings;
@group(3) @binding(0) var<uniform> reprojection_data: ReprojectionData;

const EPSILON = 1e-5;
const VARIANCE_K = 2.;

fn blend(c_col: vec4<f32>, c_depth: f32,
    a_col: vec4<f32>, a_depth: f32,
    alpha: f32) -> vec4<f32> {
    let depth_diff = abs(a_depth - c_depth);
    let colour_diff = distance(c_col.rgba, a_col.rgba);
    return vec4<f32>(depth_diff*100., colour_diff, 0., 1.);
    // if depth_diff < render_settings.depth_smoothing_high && colour_diff > render_settings.colour_smoothing_high {
        // return vec4(0.5, 1., 1., 1.);
        // return mix(a_col, c_col, render_settings.current_colour_weight);
    // } else {
        // return c_col;
    // }
}

// returns
// r => Mean
// g => Estimated Variance
// b => 0
// a => 0
fn depth_stats(raw_depth: vec4<f32>) -> vec4<f32> {
    let n = raw_depth.r;
    let n_min_1 = max(1., n - 1.);// because if it is one, (n-1) is 0 and is divided by, leading to infinity
    let sum = raw_depth.g;
    let ssum = raw_depth.b;

    let adj_mean = sum / n;
    let mean = adj_mean + (n * VARIANCE_K);
    let variance = (ssum - (sum * sum) / n) / n_min_1;
    //let variance = ((ssum / n) - (sum / n) * (sum / n)) * (n / (n_min_1));
    // ((12*12)/n-(12/n*12/n))*(1/1);
    return vec4<f32>(mean, variance, 0., 0.);
}

fn smooth_out_at(pixel_coordinate: vec2u) {
    let tex_dims_f = vec2<f32>(textureDimensions(currentFrameTexture).xy);
    let tex_dims = vec2<u32>(tex_dims_f); // assumes all texture have the same dimensions
    let current_position = pixel_coordinate;
    let current_normalized_position = vec2<f32>(current_position) / tex_dims_f;

    let current_colour = textureSampleLevel(currentFrameTexture, filter_sampler, current_normalized_position, 0.);
    let depth_stats_raw = textureSampleLevel(currentFrameDepthStatistics, filter_sampler, current_normalized_position, 0.);
    // depth is currently only a statistics vector. actually meaningful values are calculated below
    let current_depth_stats = depth_stats(depth_stats_raw);
    let current_depth_mean = current_depth_stats.r;
    let current_depth_variance = current_depth_stats.g;

    let depth_raw = textureSampleLevel(currentFrameDepthTexture, filter_sampler, current_normalized_position, 0.);
    let current_depth = depth_raw.r / depth_raw.g; // premultiplied alpha
    // end of depth calculation

    // ndc
    let current_v4_pos_ndc = vec4<f32>(
        (vec2<f32>(current_position) / tex_dims_f * 2.) - vec2<f32>(1., 1.),
        current_depth,
        1.
    );
    let current_pos_clip = current_v4_pos_ndc / current_depth;

    let reprojected_pos = reprojection_data.reprojection * current_pos_clip;

    // ndc
    let reproj_pos_ynormal = ((reprojected_pos / reprojected_pos.w).xy + vec2(1., 1.)) / 2.;
    let reproj_pos_yflipped = vec2(reproj_pos_ynormal.x, 1. - reproj_pos_ynormal.y); // flip y axis for reasons
    let reproj_pos = reproj_pos_yflipped + 1. / vec2(tex_dims_f.x / .5, tex_dims_f.y / .5); // adjust the position for an unknown, probably numeric reason

    var final_colour = current_colour;
    var dbg_colour = current_colour;
    if reproj_pos.x >= 0 && reproj_pos.x < 1. && reproj_pos.y >= 0 && reproj_pos.y < 1. {
        let accu_colour = textureSampleLevel(accuTexture, filter_sampler, reproj_pos, 0.);
        let accu_depth = textureSampleLevel(accuDepth, filter_sampler, reproj_pos, 0.).r;// here the alpha is pre-filtered out, in contrast to the current depth
        dbg_colour = blend(current_colour, current_depth, accu_colour, accu_depth, depth_raw.g);
    }
    // final_colour = vec4<f32>(vec3<f32>(sqrt(current_depth_variance)* 100), 1.);  // depth variance 
    // final_colour = vec4<f32>(vec3<f32>(current_depth_mean/100.), 1.);            // depth mean value
    // final_colour = vec4<f32>(vec3<f32>(current_depth), 1.);                      // blended depth value

    // write the texture points into the receiving buffers
    textureStore(debug_output, current_position, dbg_colour);
    textureStore(dstTexture, current_position, final_colour);
    textureStore(dstDepth, current_position, vec4<f32>(current_depth, 0., 0., 0.));
}

@compute @workgroup_size(1)
fn cs_main(@builtin(global_invocation_id) id: vec3u) {
    smooth_out_at(vec2u(id.xy * 4 + vec2<u32>(0u, 0u)));
    smooth_out_at(vec2u(id.xy * 4 + vec2<u32>(0u, 1u)));
    smooth_out_at(vec2u(id.xy * 4 + vec2<u32>(0u, 2u)));
    smooth_out_at(vec2u(id.xy * 4 + vec2<u32>(0u, 3u)));
    smooth_out_at(vec2u(id.xy * 4 + vec2<u32>(1u, 0u)));
    smooth_out_at(vec2u(id.xy * 4 + vec2<u32>(1u, 1u)));
    smooth_out_at(vec2u(id.xy * 4 + vec2<u32>(1u, 2u)));
    smooth_out_at(vec2u(id.xy * 4 + vec2<u32>(1u, 3u)));
    smooth_out_at(vec2u(id.xy * 4 + vec2<u32>(2u, 0u)));
    smooth_out_at(vec2u(id.xy * 4 + vec2<u32>(2u, 1u)));
    smooth_out_at(vec2u(id.xy * 4 + vec2<u32>(2u, 2u)));
    smooth_out_at(vec2u(id.xy * 4 + vec2<u32>(2u, 3u)));
    smooth_out_at(vec2u(id.xy * 4 + vec2<u32>(3u, 0u)));
    smooth_out_at(vec2u(id.xy * 4 + vec2<u32>(3u, 1u)));
    smooth_out_at(vec2u(id.xy * 4 + vec2<u32>(3u, 2u)));
    smooth_out_at(vec2u(id.xy * 4 + vec2<u32>(3u, 3u)));
}