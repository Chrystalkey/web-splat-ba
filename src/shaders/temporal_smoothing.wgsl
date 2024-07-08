struct CameraUniforms {
    view: mat4x4<f32>,
    view_inv: mat4x4<f32>,
    proj: mat4x4<f32>,
    proj_inv: mat4x4<f32>,

    viewport: vec2<f32>,
    focal: vec2<f32>
};

struct TSParameters {
    depth_diff_thresholds: vec2<f32>,
    colour_diff_thresholds: vec2<f32>,
    normal_diff_thresholds: vec2<f32>,
    current_frame_weight: f32,
    _pad: f32
}

struct RenderSettings {
    ts_parameters: TSParameters,
    clipping_box_min: vec4<f32>,
    clipping_box_max: vec4<f32>,
    center: vec3<f32>,
    _padding : f32,
    gaussian_scaling: f32,
    kernel_size: f32,
    walltime: f32,
    scene_extend: f32,
    max_sh_deg: u32,
    show_env_map: u32,
    mip_spatting: u32,
}

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
const BLACK = vec4<f32>(0., 0., 0., 1.);
const PI = 3.14159265359;

fn rgb2hsv(c: vec3<f32>) -> vec3<f32> {
    let K = vec4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
    let p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));
    let q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));

    let d = q.x - min(q.w, q.y);
    let e = 1.0e-10;
    return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}

// trial set dimensions
// distance() always divide by sqrt(3) because distance([0,0,0], [1,1,1] = sqrt(3) and thus it is normalized
// - euclidian or manhattan
// - rgb or hsv
// - abs(hue)
fn colour_difference(curr: vec3<f32>, accu: vec3<f32>) -> f32 {
    let ahsv = rgb2hsv(accu);
    let chsv = rgb2hsv(curr);
    return distance(curr, accu) / sqrt(3.); // distance and rgb
    // let dvec = curr-accu;
    // return (dvec.r+dvec.g+dvec.b)/3.;// manhattan and rgb, normalised to 0..1
    // return distance(ahsv, chsv) / sqrt(3.);
    // return abs(ahsv.r - chsv.r); // well, kinda useless
}

fn normvec(coord: vec2<f32>, alpha: f32) -> vec3<f32> {
    let tdim = vec2<f32>(textureDimensions(currentFrameTexture).xy);
    let coord_denorm = coord * tdim;
    let up = textureSampleLevel(currentFrameDepthTexture, filter_sampler, (coord_denorm + vec2(1., 0.)) / tdim, 0.).r;
    let dn = textureSampleLevel(currentFrameDepthTexture, filter_sampler, (coord_denorm + vec2(-1., 0.)) / tdim, 0.).r;
    let rt = textureSampleLevel(currentFrameDepthTexture, filter_sampler, (coord_denorm + vec2(0., 1.)) / tdim, 0.).r;
    let lt = textureSampleLevel(currentFrameDepthTexture, filter_sampler, (coord_denorm + vec2(0., -1.)) / tdim, 0.).r;
    let dp = textureSampleLevel(currentFrameDepthTexture, filter_sampler, coord_denorm / tdim, 0.).r; 

    // est. central difference
    return normalize(abs(vec3(up - dn, rt - lt, alpha / 4.) / alpha)); // I just guessed this four, makes the actual borders more pronounced
    // forward difference
    //return normalize(abs(vec3(up - dp, lt-dp, alpha) / alpha));
    // forward difference
    // return normalize(abs(vec3(dp - dn, dp - rt, alpha) / alpha));
}

// stands for debug-enabled colour output
struct DCO {
    colour: vec4<f32>,
    debug: vec4<f32>,
}

// useful thresholds: depth_diff ~.005, colour_diff ~.4
fn blend(c_col: vec4<f32>, c_depth: f32,
    a_col: vec4<f32>, a_depth: f32,
    surface_normal: vec3<f32>,
    alpha: f32,
    depth_variance: f32) -> DCO {
    let depth_diff = abs(a_depth - c_depth);

    let colour_diff = colour_difference(c_col.rgb, a_col.rgb);
    //return vec4<f32>(depth_diff*100., colour_diff, 0., 1.); // cool effect, though
    let normal_diff = acos(dot(surface_normal, vec3<f32>(0., 0., 1.))) / 2 * PI; // value range from 0 to 2 * PI


    var colour = c_col;
    var debug = BLACK;
    let ts_p = render_settings.ts_parameters;
    
    // let dd_coeff = 1. - smoothstep(ts_p.depth_diff_thresholds.x, ts_p.depth_diff_thresholds.y, depth_diff);
    // let cd_coeff = smoothstep(ts_p.colour_diff_thresholds.x, ts_p.colour_diff_thresholds.y, colour_diff);
    // let vr_coeff = smoothstep(0.0030, 0.0035, sqrt(depth_variance));
    // let nm_coeff = 1. - smoothstep(ts_p.normal_diff_thresholds.x, ts_p.normal_diff_thresholds.y, normal_diff); // TODO: find a useful threshold

    let dd_coeff = 1. - step(ts_p.depth_diff_thresholds.y, depth_diff);
    let cd_coeff = step(ts_p.colour_diff_thresholds.y, colour_diff);
    let vr_coeff = step(0.0035, sqrt(depth_variance));
    let nm_coeff = 1. - step(ts_p.normal_diff_thresholds.y, normal_diff); // TODO: find a useful threshold
    
    let mix_coeff = (dd_coeff * cd_coeff) * nm_coeff;


    let mix_col = mix(a_col, c_col, ts_p.current_frame_weight);
    colour = mix(c_col, mix_col, mix_coeff);
    // debug = vec4<f32>((a_depth - 10.) / 10., 0., 0., 1.);
    //debug = vec4<f32>(surface_normal, 1.); // estimated surface normals
    
    debug = vec4(mix_coeff, 0., 0., 1.);

    return DCO(colour, debug);
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

    var output = DCO(current_colour, BLACK);
    if reproj_pos.x >= 0 && reproj_pos.x < 1. && reproj_pos.y >= 0 && reproj_pos.y < 1. {
        let accu_colour = textureSampleLevel(accuTexture, filter_sampler, reproj_pos, 0.);
        let accu_depth = textureSampleLevel(accuDepth, filter_sampler, reproj_pos, 0.).r;// here the alpha is pre-filtered out, in contrast to the current depth
        let surface_normal = normvec(current_normalized_position, depth_raw.g); // estimated surface normal
        output = blend(
            current_colour, current_depth, accu_colour, accu_depth,
            surface_normal,
            depth_raw.g, current_depth_variance
        );
        //output = DCO(output.colour, vec4<f32>(output.debug.r, sqrt(current_depth_variance)*100., 0., 1.)); // premultiplied alpha
    }
    // output.debug = vec4<f32>(normvec(current_normalized_position, depth_raw.g), 1.); // normal vector debug
    // final_colour = vec4<f32>(vec3<f32>(sqrt(current_depth_variance)* 100), 1.);  // depth variance 
    // final_colour = vec4<f32>(vec3<f32>(current_depth_mean/100.), 1.);            // depth mean value
    // final_colour = vec4<f32>(vec3<f32>(current_depth), 1.);                      // blended depth value

    // write the texture points into the receiving buffers
    textureStore(debug_output, current_position, output.debug);
    textureStore(dstTexture, current_position, output.colour);
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