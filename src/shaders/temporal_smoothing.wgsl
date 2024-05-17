struct CameraUniforms {
    view: mat4x4<f32>,
    view_inv: mat4x4<f32>,
    proj: mat4x4<f32>,
    proj_inv: mat4x4<f32>,

    viewport: vec2<f32>,
    focal: vec2<f32>
};

struct FrameInformation {
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

@group(1) @binding(2) var accuTexture: texture_storage_2d<rgba16float, read_write>;

@group(1) @binding(3) var dstTexture: texture_storage_2d<rgba16float, read_write>;

@compute @workgroup_size(1)
fn cs_main(@builtin(global_invocation_id) id: vec3u){
    let tex_dims = textureDimensions(currentFrameTexture); // assumes all texture have the same dimensions
    let position = id.xy;
    let current_colour = textureLoad(currentFrameTexture, position, 0);
    let accu_colour = textureLoad(accuTexture, position);
    let final_colour = (current_colour + accu_colour)/2;
    // write the texture points into the receiving buffer
    textureStore(dstTexture, position, final_colour);
}