// we cutoff at 1/255 alpha value 
const CUTOFF:f32 = 2.3539888583335364; // = sqrt(log(255))

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) screen_pos: vec2<f32>,
    @location(1) color: vec4<f32>,
    @location(2) depth: f32,
};

struct VertexInput {
    @location(0) v: vec4<f32>,
    @location(1) pos: vec4<f32>,
    @location(2) color: vec4<f32>,
};

struct Splat {
     // 4x f16 packed as u32
    v_0: u32, v_1: u32,
    // 2x f16 packed as u32
    pos: u32,
    // depth as plain f32
    depth: f32,
    // rgba packed as f16
    color_0: u32,color_1: u32,
};

@group(0) @binding(2)
var<storage, read> points_2d : array<Splat>;
@group(1) @binding(4)
var<storage, read> indices : array<u32>;

@group(2) @binding(0)
var<storage> mutex_array : array<atomic<u32>>;
@group(2) @binding(1)
var depth_output: texture_storage_2d<rgba32float, read_write>;

@vertex
fn vs_main(
    @builtin(vertex_index) in_vertex_index: u32,
    @builtin(instance_index) in_instance_index: u32
) -> VertexOutput {
    var out: VertexOutput;

    let vertex = points_2d[indices[in_instance_index] + 0u];

    // scaled eigenvectors in screen space 
    let v1 = unpack2x16float(vertex.v_0);
    let v2 = unpack2x16float(vertex.v_1);

    let v_center = unpack2x16float(vertex.pos);

    // splat rectangle with left lower corner at (-1,-1)
    // and upper right corner at (1,1)
    let x = f32(in_vertex_index % 2u == 0u) * 2. - (1.);
    let y = f32(in_vertex_index < 2u) * 2. - (1.);

    let position = vec2<f32>(x, y) * CUTOFF;

    let offset = 2. * mat2x2<f32>(v1, v2) * position;
    out.position = vec4<f32>(v_center + offset, 0., 1.);
    out.screen_pos = position;
    out.color = vec4<f32>(unpack2x16float(vertex.color_0), unpack2x16float(vertex.color_1));
    out.depth = vertex.depth;
    return out;
}


// scales screen space coordinates to the framebuffer dimensions
fn unsigned_coordinate(sp_coord: vec2<f32>)->vec2<u32>{
    let tex_dimensions = vec2<f32>(textureDimensions(depth_output).xy);
    let coord = vec2<u32>(sp_coord*tex_dimensions);
    return coord;
}

fn array_index(pixel_coordinate: vec2<f32>) -> u32{
    let tex_dimensions = textureDimensions(depth_output);
    let px_coord = unsigned_coordinate(pixel_coordinate);
    return u32(tex_dimensions.x*px_coord.x+px_coord.y);
}

// input in screen space (0,1)
fn spin_acquire_mutex(pixel_coordinate: vec2<f32>){
    let array_idx = array_index(pixel_coordinate);
    let location = &(mutex_array[array_idx]);

    var exchanged = false;
    while (!exchanged){
        let result = atomicCompareExchangeWeak(location, u32(0), u32(1));
        // result has (old_value: T, exchanged: bool)
        exchanged = result.exchanged;
    }
}

// input in screen space (0,1)
fn release_mutex(pixel_coordinate: vec2<f32>){
    let array_idx = array_index(pixel_coordinate);
    let location = &(mutex_array[array_idx]);
    let result = atomicExchange(location, u32(0));

    //atomicStore(location, u32(0)); // I for some reason cannot write, but exchange. weird.
}


// depth is a vec4
// channels are as follows: 
// r == mean
// g == variance
// b == N
// a == M_2
// the default alpha blending blends using OVER, meaning C_n+1 = C_n*C_n.a*(1.-C_in) + C_in*C_in.a
fn blend_depth(pos: vec2<f32>, new_depth: f32, alpha: f32) -> vec4<f32>{
    let curr = textureLoad(depth_output, unsigned_coordinate(pos));
    return vec4<f32>(
        (curr.r*curr.b+new_depth)/(curr.b+1), // running mean
        0.,
        curr.b+1.,
        0.
    );
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let a = dot(in.screen_pos, in.screen_pos);
    if a > 2. * CUTOFF {
        discard;
    }
    let b = min(0.99, exp(-a) * in.color.a);

    spin_acquire_mutex(in.screen_pos);
    
    textureStore(depth_output, unsigned_coordinate(in.screen_pos), blend_depth(in.screen_pos, in.depth, b));

    release_mutex(in.screen_pos);

    return vec4<f32>(in.color.rgb, 1.) * b;
}