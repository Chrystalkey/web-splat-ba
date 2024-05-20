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
}
struct MatrixWrapper {
    matrix: mat4x4<f32>,
};

struct VertexOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) tex_coord: vec2<f32>,
};

@group(0) @binding(0) var<uniform> camera: CameraUniforms;

@group(1) @binding(0) var currentFrameTexture: texture_2d<f32>;
@group(1) @binding(1) var currentFrameDepthTexture: texture_depth_2d;

@group(1) @binding(2) var accuTexture: texture_storage_2d<rgba16float, read_write>;
@group(1) @binding(3) var accuDepth: texture_storage_2d<r32float, read_write>;

@group(1) @binding(4) var dstTexture: texture_storage_2d<rgba16float, read_write>;
@group(1) @binding(5) var dstDepth: texture_storage_2d<r32float, read_write>;

@group(1) @binding(6) var<uniform> accu_vp: MatrixWrapper;
@group(2) @binding(0) var<uniform> render_settings: RenderSettings;

const EPSILON = 1e-3;

fn reproject_position(current_position: vec4<f32>, vp_accu: mat4x4<f32>, ivp_current: mat4x4<f32>) -> vec4<f32> {
    return vp_accu * ivp_current * current_position;
}

fn smooth_out_at(pixel_coordinate: vec2u) {
    let tex_dims = textureDimensions(currentFrameTexture); // assumes all texture have the same dimensions
    let current_position = pixel_coordinate;
    let current_colour = textureLoad(currentFrameTexture, current_position, 0);
    let current_depth = vec2<f32>(textureLoad(currentFrameDepthTexture, current_position, 0), 0.) / current_colour.a;

    let current_v4_pos = vec4<f32>(vec2<f32>(current_position) / camera.viewport, 1., current_depth.x);
    let reprojected_pos = reproject_position(
        current_v4_pos,
        accu_vp.matrix,
        camera.view_inv * camera.proj_inv
    );
    let reproj_pos = vec2<u32>(reprojected_pos.xy);


    let accu_colour = textureLoad(accuTexture, reproj_pos);
    let accu_depth = vec2<f32>(textureLoad(accuDepth, reproj_pos.xy).x, 0.);

    var final_colour = current_colour;
    if abs(accu_depth.x - current_depth.x) > EPSILON {
        final_colour = current_colour * render_settings.current_colour_weight + accu_colour * (1. - render_settings.current_colour_weight);
    }
    
    // write the texture points into the receiving buffer
    textureStore(dstTexture, current_position, final_colour);
    textureStore(dstDepth, current_position, vec4<f32>(current_depth.x, 0., 0., 0.));
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