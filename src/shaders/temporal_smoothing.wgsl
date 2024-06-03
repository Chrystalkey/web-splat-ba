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

@group(1) @binding(2) var accuTexture: texture_storage_2d<rgba16float, read_write>;
@group(1) @binding(3) var accuDepth: texture_storage_2d<r32float, read_write>;

@group(1) @binding(4) var dstTexture: texture_storage_2d<rgba16float, read_write>;
@group(1) @binding(5) var dstDepth: texture_storage_2d<r32float, read_write>;

@group(1) @binding(6) var filter_sampler: sampler;

@group(2) @binding(0) var<uniform> render_settings: RenderSettings;
@group(3) @binding(0) var<uniform> reprojection_data: ReprojectionData;

const EPSILON = 1e-2;

fn smooth_out_at(pixel_coordinate: vec2u) {
    let tex_dims = vec2<f32>(textureDimensions(currentFrameTexture).xy); // assumes all texture have the same dimensions
    let current_position = pixel_coordinate;
    let current_normalized_position = vec2<f32>(current_position) / tex_dims;

    let current_colour = textureSampleLevel(currentFrameTexture,filter_sampler, current_normalized_position, 0.);
    let current_depth = textureSampleLevel(currentFrameDepthTexture,filter_sampler, current_normalized_position, 0.).r / (current_colour.a + EPSILON);

    //let current_depth = vec2<f32>(textureLoad(currentFrameDepthTexture, current_position, 0).r, 0.) / (current_colour.a + EPSILON);

    // ndc
    let current_v4_pos_ndc = vec4<f32>(
        (vec2<f32>(current_position) / tex_dims * 2) - vec2<f32>(1., 1.),
        current_depth,
        1.
    );
    let current_pos_clip = current_v4_pos_ndc / current_depth;

    let reprojected_pos = reprojection_data.reprojection * current_pos_clip;

    // ndc
    let reproj_pos = ((reprojected_pos / reprojected_pos.w).xy + vec2(1., 1.)) * tex_dims / 2.;
    // the reason for previous confusion was: I forgot that clip space exists :facepalm:

    var final_colour = current_colour;
    if reproj_pos.x >= 0 && reproj_pos.x < tex_dims.x && reproj_pos.y >= 0 && reproj_pos.y < tex_dims.y {
        let reproj_pos = vec2<u32>(u32(reproj_pos.x), u32(tex_dims.y - reproj_pos.y)); // flip y axis for reasons 
        let accu_colour = textureLoad(accuTexture, reproj_pos);
        let accu_depth = vec2<f32>(textureLoad(accuDepth, reproj_pos.xy).x, 0.);

        // smooth out depth difference (or check for greater than)
        if abs(accu_depth.x - current_depth) > EPSILON {
            // weigh by per-pixel information (colour diff, depth diff, depth variance)
            final_colour = current_colour * render_settings.current_colour_weight + accu_colour * (1. - render_settings.current_colour_weight);
        }
        //final_colour = vec4<f32>(.8,0,0,1);
    }
    final_colour = clamp(vec4<f32>(vec3(current_depth)/100., 1), vec4(0.), vec4(1.));

    // write the texture points into the receiving buffer
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