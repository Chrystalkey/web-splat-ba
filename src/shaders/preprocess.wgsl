const KERNEL_SIZE:f32 = 0.3;
//const MAX_SH_DEG:u32 = <injected>u;

const SH_C0:f32 = 0.28209479177387814;

const SH_C1 = 0.4886025119029199;
const SH_C2 = array<f32,5>(
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
);

const SH_C3 = array<f32,7>(
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
);

struct CameraUniforms {
    view: mat4x4<f32>,
    view_inv: mat4x4<f32>,
    proj: mat4x4<f32>,
    proj_inv: mat4x4<f32>,
    
    viewport: vec2<f32>,
    focal: vec2<f32>
};

struct Gaussian {
    pos_opacity: array<u32,2>,
    scale_rot: array<u32,4>,
}

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
    color_0: u32,color_1: u32
};

struct DrawIndirect {
    /// The number of gaussians to draw.
    vertex_count: u32,
    /// The number of instances to draw.
    instance_count: atomic<u32>,
    /// The Index of the first vertex to draw.
    base_vertex: u32,
    /// The instance ID of the first instance to draw.
    /// Has to be 0, unless [`Features::INDIRECT_FIRST_INSTANCE`](crate::Features::INDIRECT_FIRST_INSTANCE) is enabled.
    base_instance: u32,
}

struct DispatchIndirect {
    dispatch_x: atomic<u32>,
    dispatch_y: u32,
    dispatch_z: u32,
}

struct SortInfos {
    keys_size: atomic<u32>,     // essentially contains the same info as instance_count in DrawIndirect
    padded_size: u32,
    passes: u32,
    even_pass: u32,
    odd_pass: u32,
}
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
    _padding: f32,
    gaussian_scaling: f32,
    kernel_size: f32,
    walltime: f32,
    scene_extend: f32,
    max_sh_deg: u32,
    show_env_map: u32,
    mip_spatting: u32,
}

@group(0) @binding(0)
var<uniform> camera: CameraUniforms;

@group(1) @binding(0)
var<storage,read> gaussians : array<Gaussian>;
@group(1) @binding(1) 
var<storage,read> sh_coefs : array<array<u32,24>>;

@group(1) @binding(2) 
var<storage,read_write> points_2d : array<Splat>;

@group(2) @binding(0)
var<storage, read_write> sort_infos: SortInfos;
@group(2) @binding(1)
var<storage, read_write> sort_depths : array<u32>;
@group(2) @binding(2)
var<storage, read_write> sort_indices : array<u32>;
@group(2) @binding(3)
var<storage, read_write> sort_dispatch: DispatchIndirect;

@group(3) @binding(0)
var<uniform> render_settings: RenderSettings;


/// reads the ith sh coef from the vertex buffer
fn sh_coef(splat_idx: u32, c_idx: u32) -> vec3<f32> {
    let a = unpack2x16float(sh_coefs[splat_idx][(c_idx * 3u + 0u) / 2u])[(c_idx * 3u + 0u) % 2u];
    let b = unpack2x16float(sh_coefs[splat_idx][(c_idx * 3u + 1u) / 2u])[(c_idx * 3u + 1u) % 2u];
    let c = unpack2x16float(sh_coefs[splat_idx][(c_idx * 3u + 2u) / 2u])[(c_idx * 3u + 2u) % 2u];
    return vec3<f32>(
        a, b, c
    );
}

// spherical harmonics evaluation with Condon–Shortley phase
fn evaluate_sh(dir: vec3<f32>, v_idx: u32, sh_deg: u32) -> vec3<f32> {
    var result = SH_C0 * sh_coef(v_idx, 0u);

    if sh_deg > 0u {

        let x = dir.x;
        let y = dir.y;
        let z = dir.z;

        result += - SH_C1 * y * sh_coef(v_idx, 1u) + SH_C1 * z * sh_coef(v_idx, 2u) - SH_C1 * x * sh_coef(v_idx, 3u);

        if sh_deg > 1u {

            let xx = dir.x * dir.x;
            let yy = dir.y * dir.y;
            let zz = dir.z * dir.z;
            let xy = dir.x * dir.y;
            let yz = dir.y * dir.z;
            let xz = dir.x * dir.z;

            result += SH_C2[0] * xy * sh_coef(v_idx, 4u) + SH_C2[1] * yz * sh_coef(v_idx, 5u) + SH_C2[2] * (2.0 * zz - xx - yy) * sh_coef(v_idx, 6u) + SH_C2[3] * xz * sh_coef(v_idx, 7u) + SH_C2[4] * (xx - yy) * sh_coef(v_idx, 8u);

            if sh_deg > 2u {
                result += SH_C3[0] * y * (3.0 * xx - yy) * sh_coef(v_idx, 9u) + SH_C3[1] * xy * z * sh_coef(v_idx, 10u) + SH_C3[2] * y * (4.0 * zz - xx - yy) * sh_coef(v_idx, 11u) + SH_C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * sh_coef(v_idx, 12u) + SH_C3[4] * x * (4.0 * zz - xx - yy) * sh_coef(v_idx, 13u) + SH_C3[5] * z * (xx - yy) * sh_coef(v_idx, 14u) + SH_C3[6] * x * (xx - 3.0 * yy) * sh_coef(v_idx, 15u);
            }
        }
    }
    result += 0.5;

    return result;
}

fn q2mat(rot: vec4<f32>) -> mat3x3<f32> {
    let m11 = 2. * (rot.x * rot.x + rot.y * rot.y) - 1.;
    let m12 = 2. * (rot.y * rot.z - rot.x * rot.w);
    let m13 = 2. * (rot.y * rot.w + rot.x * rot.z);
    let m21 = 2. * (rot.y * rot.z + rot.x * rot.w);
    let m22 = 2. * (rot.x * rot.x + rot.z * rot.z) - 1.;
    let m23 = 2. * (rot.z * rot.w - rot.x * rot.y);
    let m31 = 2. * (rot.y * rot.w - rot.x * rot.z);
    let m32 = 2. * (rot.z * rot.w + rot.x * rot.y);
    let m33 = 2. * (rot.x * rot.x + rot.w * rot.w)-1.;
    return mat3x3<f32>(
        vec3<f32>(m11, m21, m31),
        vec3<f32>(m12, m22, m32),
        vec3<f32>(m13, m23, m33)
    );
}

// calculates covariance matrix from quaternion rotation and scale vector
fn build_cov(rot: vec4<f32>, scale: vec3<f32>) -> mat3x3<f32> {
    let rmat = q2mat(rot);
    let smat = mat3x3<f32>(
        vec3<f32>(scale.x, 0., 0.),
        vec3<f32>(0., scale.y, 0.),
        vec3<f32>(0., 0., scale.z)
    );
    let l = rmat * smat;
    let m = l * transpose(l);
    return m;
}

// builds inverse camera origin transformation matrix 
fn build_comat(rot: vec4<f32>, transl: vec3<f32>) -> mat4x4<f32> {
    let q_norm = length(rot);
    let inv_rot = vec4<f32>(-rot.x, -rot.y, -rot.z, rot.w) / q_norm;
    let inv_rmat = q2mat(inv_rot);
    let inv_transl = -transl;
    return mat4x4<f32>(
        vec4<f32>(inv_rmat[0], 0.),
        vec4<f32>(inv_rmat[1], 0.),
        vec4<f32>(inv_rmat[2], 0.),
        vec4<f32>(inv_transl, 1.)
    );
}

// fn cov_coefs(v_idx: u32) -> array<f32,6> {
//     let a = unpack2x16float(gaussians[v_idx].cov[0]);
//     let b = unpack2x16float(gaussians[v_idx].cov[1]);
//     let c = unpack2x16float(gaussians[v_idx].cov[2]);
//     return array<f32,6>(a.x, a.y, b.x, b.y, c.x, c.y);
// }

@compute @workgroup_size(256,1,1)
fn preprocess(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) wgs: vec3<u32>) {
    let idx = gid.x;
    if idx >= arrayLength(&gaussians) {
        return;
    }

    let focal = camera.focal;
    let viewport = camera.viewport;
    let vertex = gaussians[idx];
    let a = unpack2x16float(vertex.pos_opacity[0]);
    let b = unpack2x16float(vertex.pos_opacity[1]);
    let xyz = vec3<f32>(a.x, a.y, b.x);
    var opacity = b.y;

    if any(xyz < render_settings.clipping_box_min.xyz) || any(xyz > render_settings.clipping_box_max.xyz) {
        return;
    }

    var camspace = camera.view * vec4<f32>(xyz, 1.);
    let pos2d = camera.proj * camspace;
    let bounds = 1.2 * pos2d.w;
    let z = pos2d.z / pos2d.w;

    if idx == 0u {
        atomicAdd(&sort_dispatch.dispatch_x, 1u);   // safety addition to always have an unfull block at the end of the buffer
    }
    // frustum culling hack
    if z <= 0. || z >= 1. || pos2d.x < -bounds || pos2d.x > bounds || pos2d.y < -bounds || pos2d.y > bounds {
        return;
    }

    // let cov_sparse = cov_coefs(idx);

    let walltime = render_settings.walltime;
    var scale_mod = 0.;
    let dd = 5. * distance(render_settings.center, xyz) / render_settings.scene_extend;
    if walltime > dd {
        scale_mod = smoothstep(0., 1., (walltime - dd));
    }

    let scaling = render_settings.gaussian_scaling * scale_mod;
    let sr1 = unpack2x16float(vertex.scale_rot[0]);
    let sr2 = unpack2x16float(vertex.scale_rot[1]);
    let sr3 = unpack2x16float(vertex.scale_rot[2]);
    let sr4 = unpack2x16float(vertex.scale_rot[3]);

    let raw_cov = build_cov(
        vec4<f32>(sr3.x, sr3.y, sr4.x, sr4.y),
        vec3<f32>(sr1.x, sr1.y, sr2.x),
    );
    let Vrk = raw_cov * scaling * scaling;
    let J = mat3x3<f32>(
        focal.x / camspace.z,
        0.,
        -(focal.x * camspace.x) / (camspace.z * camspace.z),
        0.,
        -focal.y / camspace.z,
        (focal.y * camspace.y) / (camspace.z * camspace.z),
        0.,
        0.,
        0.
    );

    let W = transpose(mat3x3<f32>(camera.view[0].xyz, camera.view[1].xyz, camera.view[2].xyz));
    let T = W * J;
    let cov = transpose(T) * Vrk * T;

    let kernel_size = render_settings.kernel_size;
    if bool(render_settings.mip_spatting) {
        // according to Mip-Splatting by Yu et al. 2023
        let det_0 = max(1e-6, cov[0][0] * cov[1][1] - cov[0][1] * cov[0][1]);
        let det_1 = max(1e-6, (cov[0][0] + kernel_size) * (cov[1][1] + kernel_size) - cov[0][1] * cov[0][1]);
        var coef = sqrt(det_0 / (det_1 + 1e-6) + 1e-6);

        if det_0 <= 1e-6 || det_1 <= 1e-6 {
            coef = 0.0;
        }
        opacity *= coef;
    }

    let diagonal1 = cov[0][0] + kernel_size;
    let offDiagonal = cov[0][1];
    let diagonal2 = cov[1][1] + kernel_size;

    let mid = 0.5 * (diagonal1 + diagonal2);
    let radius = length(vec2<f32>((diagonal1 - diagonal2) / 2.0, offDiagonal));
    // eigenvalues of the 2D screen space splat
    let lambda1 = mid + radius;
    let lambda2 = max(mid - radius, 0.1);

    let diagonalVector = normalize(vec2<f32>(offDiagonal, lambda1 - diagonal1));
    // scaled eigenvectors in screen space 
    let v1 = sqrt(2.0 * lambda1) * diagonalVector;
    let v2 = sqrt(2.0 * lambda2) * vec2<f32>(diagonalVector.y, -diagonalVector.x);

    let v_center = pos2d.xyzw / pos2d.w;

    let camera_pos = camera.view_inv[3].xyz;
    let dir = normalize(xyz - camera_pos);
    let color = vec4<f32>(
        max(vec3<f32>(0.), evaluate_sh(dir, idx, render_settings.max_sh_deg)),
        opacity
    );

    let store_idx = atomicAdd(&sort_infos.keys_size, 1u);
    let v = vec4<f32>(v1 / viewport, v2 / viewport);
    points_2d[store_idx] = Splat(
        build_comat(vec4<f32>(sr3.x, sr3.y, sr4.x, sr4.y), xyz),
        vec3<f32>(sr1.x, sr1.y, sr2.x),
        pack2x16float(v.xy), pack2x16float(v.zw),
        pack2x16float(v_center.xy),
        z,
        pack2x16float(color.rg), pack2x16float(color.ba),
    );
    // filling the sorting buffers and the indirect sort dispatch buffer
    let znear = -camera.proj[3][2] / camera.proj[2][2];
    let zfar = -camera.proj[3][2] / (camera.proj[2][2] - (1.));
    // filling the sorting buffers and the indirect sort dispatch buffer
    sort_depths[store_idx] = u32(f32(0xffffffu) - (pos2d.z - znear) / (zfar - znear) * f32(0xffffffu));  // depth branch depth calculation
    // or
    // sort_depths[store_idx] = u32(f32(0xffffffu) - (pos2d.z - znear) / (zfar - znear) * f32(0xffffffu)); 
    // or
    // bitcast<u32>(zfar - pos2d.z);
    sort_indices[store_idx] = store_idx;

    let keys_per_wg = 256u * 15u;         // Caution: if workgroup size (256) or keys per thread (15) changes the dispatch is wrong!!
    if (store_idx % keys_per_wg) == 0u {
        atomicAdd(&sort_dispatch.dispatch_x, 1u);
    }
}