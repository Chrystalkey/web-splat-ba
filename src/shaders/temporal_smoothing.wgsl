struct CameraUniforms {
    view: mat4x4<f32>,
    view_inv: mat4x4<f32>,
    proj: mat4x4<f32>,
    proj_inv: mat4x4<f32>,

    viewport: vec2<f32>,
    focal: vec2<f32>
};

struct FrameTransformation {
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
}

struct VertexOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) tex_coord: vec2<f32>,
};

@group(0) @binding(0) var<uniform> camera: CameraUniforms;

@group(1) @binding(0) var frameSampler: sampler;

@group(1) @binding(1) var currentFrameTexture: texture_2d<f32>;
@group(1) @binding(2) var currentFrameDepthTexture: texture_depth_2d;

@group(1) @binding(3) var accuTexture: texture_storage_2d<rgba16float, read_write>;
@group(1) @binding(4) var accuDepth: texture_storage_2d<r32float, read_write>;

@group(1) @binding(5) var dstTexture: texture_storage_2d<rgba16float, read_write>;
@group(1) @binding(6) var dstDepth: texture_storage_2d<r32float, read_write>;

@group(2) @binding(0) var<uniform> accu_camera: CameraUniforms;


fn reproject_position(current_position: vec4<f32>, vp_accu: mat4x4<f32>, ivp_current: mat4x4<f32>) -> vec4<f32> {
    return vp_accu * ivp_current * current_position;
}
const EPSILON = 1e-10;

@compute @workgroup_size(1)
fn cs_main(@builtin(global_invocation_id) id: vec3u) {
    let tex_dims = textureDimensions(currentFrameTexture); // assumes all texture have the same dimensions
    let current_position = id.xy;
    let current_colour = textureLoad(currentFrameTexture, current_position, 0);
    let current_depth = vec2<f32>(textureLoad(currentFrameDepthTexture, current_position, 0), 0.);
    
    let current_v4_pos = vec4<f32>(vec2<f32>(current_position), current_depth.x, 1);
    let reprojected_pos = reproject_position(
        current_v4_pos,
        accu_camera.proj * accu_camera.view,
        camera.proj_inv * camera.view
    );
    let reproj_coordinates = vec2<u32>(reprojected_pos.xy);


    let accu_colour = textureLoad(accuTexture, reproj_coordinates);
    let accu_depth = vec2<f32>(textureLoad(accuDepth, reproj_coordinates.xy).x, 0.);

    var final_colour = (current_colour + accu_colour) / 2;
    if (abs(accu_depth.x - current_depth.x) < EPSILON) {
        final_colour = vec4<f32>(1.,1.,1.,.5);
    }
    // write the texture points into the receiving buffer
    textureStore(dstTexture, current_position, final_colour);
    textureStore(dstDepth, current_position, vec4<f32>(current_depth.x, 0., 0., 0.));
}