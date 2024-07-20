// we cutoff at 1/255 alpha value 
const CUTOFF:f32 = 2.3539888583335364; // = sqrt(log(255))

struct CameraUniforms {
    view: mat4x4<f32>,
    view_inv: mat4x4<f32>,
    proj: mat4x4<f32>,
    proj_inv: mat4x4<f32>,
    
    viewport: vec2<f32>,
    focal: vec2<f32>
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) screen_pos: vec2<f32>,
    @location(1) color: vec4<f32>,
    @location(2) depth: f32,
    @location(3) splat_index: u32,
};

struct VertexInput {
    @location(0) v: vec4<f32>,
    @location(1) pos: vec4<f32>,
    @location(2) color: vec4<f32>,
};

struct Splat {
    // this times the camera origin gives the co in adjusted gaussian space
    co_transform: mat4x4<f32>,
    // things for the depth calculation
    scale_vec: vec3<f32>,
     // 4x f16 packed as u32
    v_0: u32, v_1: u32,
    // 2x f16 packed as u32
    pos: u32,
    // depth as plain f32
    depth: f32,
    // rgba packed as f16
    color_0: u32,color_1: u32,
};

struct FragmentOut {
    @location(0) color: vec4<f32>,
    @location(1) depth_stats: vec4<f32>,
    @location(2) depth_blend: vec4<f32>,
};

@group(0) @binding(2)
var<storage, read> points_2d : array<Splat>;
@group(1) @binding(4)
var<storage, read> indices : array<u32>;
@group(2) @binding(0)
var<uniform> camera : CameraUniforms;

// @group(2) @binding(0) var<uniform> camera: CameraUniforms;


// ideas how to calculate the intersection between the elipsis and the view ray:
// - i have to do it in the fragment shader of this rp, because this is the spot where rasterization happens
// - I do not have enough info to directly calculate this within the shader
// 1. somehow deproject v1,v2 into world space using depth and position(?)
// 2. while in preprocess, pass along another bit of info with which we can intersect the gaussians here
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
    out.splat_index = in_instance_index;
    return out;
}

// origin is the camera origin coordinate in world space
// pos is the camera position
// direction is the normalized ray direction
fn ray_depth(origin: vec3<f32>, direction: vec3<f32>, scale: vec3<f32>) -> f32 {
    let ror = origin;
    let dir = direction;
    let a = (dir.x * dir.x) / (scale.x * scale.x) + (dir.y * dir.y) / (scale.y * scale.y) + (dir.z * dir.z) / (scale.z * scale.z);
    let b = 2 * (ror.x * dir.x / (scale.x * scale.x) + ror.y * dir.y / (scale.y * scale.y) + ror.z * dir.z / (scale.z * scale.z));
    let c = (ror.x * ror.x) / (scale.x * scale.x) + (ror.y * ror.y) / (scale.y * scale.y) + (ror.z * ror.z) / (scale.z * scale.z) - 1;

    let disc = b * b-4 * a * c;
    if disc < 0. {
        return 0.;
    }
    let t = (-b * 0.5 / a);
    let p_mid = ror + t * dir;

    return t;
}

// TODO: somehow get the co_transform into the splat struct, accomodate the additional members on the cpu side
fn calculate_adjusted_depth(splat: Splat, screen_pos: vec2<f32>) -> f32 {
    // get screen pos into world space 
    let camera_pos = vec4<f32>(camera.view_inv[3].xyz, 1.); // in world space
    let pxl_pos = camera.view_inv * camera.proj_inv * vec4<f32>(screen_pos / camera.viewport, -1., 1.); // in world space
    // transform them into adjusted gaussian space
    let adj_co = splat.co_transform * camera_pos;
    let adj_pxl_pos = splat.co_transform * vec4<f32>(pxl_pos.xyz, 1.);
    // calculate the direction & run ray intersection
    let direction = normalize(adj_pxl_pos - adj_co);
    return ray_depth(adj_co.xyz, direction.xyz, splat.scale_vec);
}

const VARIANCE_K: f32 = 2.;
// TODO: Depth Calculation Experiments
// Ideas:
// a: use the mean depth of the gaussian splat as we do right now
// b: use the depth calculation from the paper

// 1. use alpha blending as we do now
// 2. let the depth behave like opaque objects and use min (\alpha = 1)
// 3. use mean of depth - x sd of depth

@fragment
fn fs_main(in: VertexOutput) -> FragmentOut {
    let a = dot(in.screen_pos, in.screen_pos);
    if a > 2. * CUTOFF {
        discard;
    }
    let b = min(0.99, exp(-a) * in.color.a);

    // all blend components are one and operation is add, thus
    // r is N
    // g is total sum of depth
    // b is sum of squares
    // a is sum of alphas, currently unused
    let elipsis_depth = calculate_adjusted_depth(points_2d[indices[in.splat_index] + 0u], in.screen_pos);
    let depth_adjusted = in.depth - VARIANCE_K;
    let depth_stat_return = vec4<f32>(
        1.,
        depth_adjusted,
        depth_adjusted * depth_adjusted,
        b
    );

    // premultiplied alphablending
    let depth_return = vec4<f32>(
        elipsis_depth * b,
        b,
        0.,
        b
    );
    // opaque objects
    let depth_return_opq = vec4<f32>(
        in.depth,
        b,
        0.,
        1.
    );

    return FragmentOut(vec4<f32>(in.color.rgb, 1.) * b, depth_return, depth_stat_return);
}