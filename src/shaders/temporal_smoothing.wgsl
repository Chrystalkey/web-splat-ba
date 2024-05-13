struct CameraUniforms {
    view: mat4x4<f32>,
    view_inv: mat4x4<f32>,
    proj: mat4x4<f32>,
    proj_inv: mat4x4<f32>,

    viewport: vec2<f32>,
    focal: vec2<f32>
};

struct FrameInformation{
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
}

struct VertexOut{
    @builtin(position) pos: vec4<f32>,
    @location(0) tex_coord: vec2<f32>,
};

@group(0) @binding(0) 
var<uniform> camera: CameraUniforms;
@group(1) @binding(0) 
var prevFrameTexture: texture_2d<f32>;
@group(1) @binding(1) 
var prevFrameSampler: sampler;

@vertex
fn vs_main(
    @builtin(vertex_index) in_vertex_index: u32,
) -> VertexOut {

    // creates two vertices that cover the whole screen
    let xy = vec2<f32>(
        f32(in_vertex_index % 2u == 0u),
        f32(in_vertex_index < 2u)
    );
    return VertexOut(vec4<f32>(xy * 2. - (1.), 0., 1.), vec2<f32>(xy.x, 1. - xy.y));
}

@fragment
fn fs_main(vertex_in: VertexOut) -> @location(0) vec4<f32> {
    let color = textureSample(
                    prevFrameTexture, 
                    prevFrameSampler, 
                    vertex_in.tex_coord
                    );
    return color;
}