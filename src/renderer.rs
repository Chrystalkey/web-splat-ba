use crate::gpu_rs::{GPURSSorter, PointCloudSortStuff};
use crate::pointcloud::Aabb;
use crate::utils::GPUStopwatch;
use crate::{
    camera::{Camera, PerspectiveCamera, VIEWPORT_Y_FLIP},
    pointcloud::PointCloud,
    uniform::UniformBuffer,
};

use std::hash::{Hash, Hasher};
use std::num::NonZeroU64;
use std::time::Duration;

use wgpu::{include_wgsl, Extent3d, MultisampleState};

use cgmath::{EuclideanSpace, Matrix4, Point3, SquareMatrix, Vector2, Vector4};

pub struct GaussianRenderer {
    pipeline: wgpu::RenderPipeline,
    camera: UniformBuffer<CameraUniform>,

    render_settings: UniformBuffer<SplattingArgsUniform>,
    preprocess: PreprocessPipeline,

    draw_indirect_buffer: wgpu::Buffer,
    #[allow(dead_code)]
    draw_indirect: wgpu::BindGroup,
    color_format: wgpu::TextureFormat,
    sorter: GPURSSorter,
    sorter_suff: Option<PointCloudSortStuff>,
}

impl GaussianRenderer {
    pub async fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        color_format: wgpu::TextureFormat,
        sh_deg: u32,
        compressed: bool,
    ) -> Self {
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("render pipeline layout"),
            bind_group_layouts: &[
                &PointCloud::bind_group_layout_render(device), // Needed for points_2d (on binding 2)
                &GPURSSorter::bind_group_layout_rendering(device), // Needed for indices   (on binding 4)
            ],
            push_constant_ranges: &[],
        });

        let shader = device.create_shader_module(wgpu::include_wgsl!("shaders/gaussian.wgsl"));

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("render pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: color_format,
                    blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: TemporalSmoothing::IN_TEXTURE_FORMAT_DEP,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Always,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        let draw_indirect_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("indirect draw buffer"),
            size: std::mem::size_of::<wgpu::util::DrawIndirectArgs>() as u64,
            usage: wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let indirect_layout = Self::bind_group_layout(device);
        let draw_indirect = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("draw indirect buffer"),
            layout: &indirect_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: draw_indirect_buffer.as_entire_binding(),
            }],
        });

        let sorter = GPURSSorter::new(device, queue).await;

        let camera = UniformBuffer::new_default(device, Some("camera uniform buffer"));
        let preprocess = PreprocessPipeline::new(device, sh_deg, compressed);
        GaussianRenderer {
            pipeline,
            camera,
            preprocess,
            draw_indirect_buffer,
            draw_indirect,
            color_format,
            sorter,
            sorter_suff: None,
            render_settings: UniformBuffer::new_default(
                device,
                Some("render settings uniform buffer"),
            ),
        }
    }

    pub(crate) fn camera(&self) -> &UniformBuffer<CameraUniform> {
        &self.camera
    }

    fn preprocess<'a>(
        &'a mut self,
        encoder: &'a mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        pc: &'a PointCloud,
        render_settings: SplattingArgs,
    ) {
        let camera = render_settings.camera;
        let uniform = self.camera.as_mut();
        uniform.set_focal(camera.projection.focal(render_settings.viewport));
        uniform.set_viewport(render_settings.viewport.cast().unwrap());
        uniform.set_camera(camera);
        self.camera.sync(queue);

        let settings_uniform = self.render_settings.as_mut();
        *settings_uniform = SplattingArgsUniform::from_args_and_pc(render_settings, pc);
        self.render_settings.sync(queue);

        // TODO perform this in vertex buffer after draw call
        queue.write_buffer(
            &self.draw_indirect_buffer,
            0,
            wgpu::util::DrawIndirectArgs {
                vertex_count: 4,
                instance_count: 0,
                first_vertex: 0,
                first_instance: 0,
            }
            .as_bytes(),
        );
        let depth_buffer = &self.sorter_suff.as_ref().unwrap().sorter_bg_pre;
        self.preprocess.run(
            encoder,
            pc,
            &self.camera,
            &self.render_settings,
            depth_buffer,
        );
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub async fn num_visible_points(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> u32 {
        let n = {
            let (tx, rx) = futures_intrusive::channel::shared::oneshot_channel();

            wgpu::util::DownloadBuffer::read_buffer(
                device,
                queue,
                &self.draw_indirect_buffer.slice(..),
                move |b| {
                    let download = b.unwrap();
                    let data = download.as_ref();
                    let num_points = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);
                    tx.send(num_points).unwrap();
                },
            );
            device.poll(wgpu::Maintain::Wait);
            rx.receive().await.unwrap()
        };
        return n;
    }

    pub fn prepare(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        pc: &PointCloud,
        render_settings: SplattingArgs,
        stopwatch: &mut Option<GPUStopwatch>,
    ) {
        if self.sorter_suff.is_none()
            || self
                .sorter_suff
                .as_ref()
                .is_some_and(|s| s.num_points != pc.num_points() as usize)
        {
            log::debug!("created sort buffers for {:} points", pc.num_points());
            self.sorter_suff = Some(
                self.sorter
                    .create_sort_stuff(device, pc.num_points() as usize),
            );
        }

        GPURSSorter::record_reset_indirect_buffer(
            &self.sorter_suff.as_ref().unwrap().sorter_dis,
            &self.sorter_suff.as_ref().unwrap().sorter_uni,
            &queue,
        );

        // convert 3D gaussian splats to 2D gaussian splats
        if let Some(stopwatch) = stopwatch {
            stopwatch.start(encoder, "preprocess").unwrap();
        }

        self.preprocess(encoder, queue, &pc, render_settings);
        if let Some(stopwatch) = stopwatch {
            stopwatch.stop(encoder, "preprocess").unwrap();
        }
        // sort 2d splats
        if let Some(stopwatch) = stopwatch {
            stopwatch.start(encoder, "sorting").unwrap();
        }
        self.sorter.record_sort_indirect(
            &self.sorter_suff.as_ref().unwrap().sorter_bg,
            &self.sorter_suff.as_ref().unwrap().sorter_dis,
            encoder,
        );
        if let Some(stopwatch) = stopwatch {
            stopwatch.stop(encoder, "sorting").unwrap();
        }

        encoder.copy_buffer_to_buffer(
            &self.sorter_suff.as_ref().unwrap().sorter_uni,
            0,
            &self.draw_indirect_buffer,
            std::mem::size_of::<u32>() as u64,
            std::mem::size_of::<u32>() as u64,
        );
    }

    pub fn render<'rpass>(
        &'rpass self,
        render_pass: &mut wgpu::RenderPass<'rpass>,
        pc: &'rpass PointCloud,
    ) {
        render_pass.set_bind_group(0, pc.render_bind_group(), &[]);
        render_pass.set_bind_group(1, &self.sorter_suff.as_ref().unwrap().sorter_render_bg, &[]);
        render_pass.set_pipeline(&self.pipeline);

        render_pass.draw_indirect(&self.draw_indirect_buffer, 0);
    }

    pub fn bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("draw indirect"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: Some(
                        NonZeroU64::new(std::mem::size_of::<wgpu::util::DrawIndirectArgs>() as u64)
                            .unwrap(),
                    ),
                },
                count: None,
            }],
        })
    }

    pub fn color_format(&self) -> wgpu::TextureFormat {
        self.color_format
    }

    pub(crate) fn render_settings(&self) -> &UniformBuffer<SplattingArgsUniform> {
        &self.render_settings
    }
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUniform {
    /// the cameras view matrix
    pub(crate) view_matrix: Matrix4<f32>,
    /// inverse view matrix
    pub(crate) view_inv_matrix: Matrix4<f32>,

    // the cameras projection matrix
    pub(crate) proj_matrix: Matrix4<f32>,

    // inverse projection matrix
    pub(crate) proj_inv_matrix: Matrix4<f32>,

    pub(crate) viewport: Vector2<f32>,
    pub(crate) focal: Vector2<f32>,
}

impl Default for CameraUniform {
    fn default() -> Self {
        Self {
            view_matrix: Matrix4::identity(),
            view_inv_matrix: Matrix4::identity(),
            proj_matrix: Matrix4::identity(),
            proj_inv_matrix: Matrix4::identity(),
            viewport: Vector2::new(1., 1.),
            focal: Vector2::new(1., 1.),
        }
    }
}

impl CameraUniform {
    pub(crate) fn set_view_mat(&mut self, view_matrix: Matrix4<f32>) {
        self.view_matrix = view_matrix;
        self.view_inv_matrix = view_matrix.invert().unwrap();
    }

    pub(crate) fn set_proj_mat(&mut self, proj_matrix: Matrix4<f32>) {
        self.proj_matrix = VIEWPORT_Y_FLIP * proj_matrix;
        self.proj_inv_matrix = proj_matrix.invert().unwrap();
    }

    pub fn set_camera(&mut self, camera: impl Camera) {
        self.set_proj_mat(camera.proj_matrix());
        self.set_view_mat(camera.view_matrix());
    }

    pub fn set_viewport(&mut self, viewport: Vector2<f32>) {
        self.viewport = viewport;
    }
    pub fn set_focal(&mut self, focal: Vector2<f32>) {
        self.focal = focal
    }
}

struct PreprocessPipeline(wgpu::ComputePipeline);

impl PreprocessPipeline {
    fn new(device: &wgpu::Device, sh_deg: u32, compressed: bool) -> Self {
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("preprocess pipeline layout"),
            bind_group_layouts: &[
                &UniformBuffer::<CameraUniform>::bind_group_layout(device),
                &if !compressed {
                    PointCloud::bind_group_layout(device)
                } else {
                    PointCloud::bind_group_layout_compressed(device)
                },
                &GPURSSorter::bind_group_layout_preprocess(device),
                &UniformBuffer::<SplattingArgsUniform>::bind_group_layout(device),
            ],
            push_constant_ranges: &[],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("preprocess shader"),
            source: wgpu::ShaderSource::Wgsl(Self::build_shader(sh_deg, compressed).into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("preprocess pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "preprocess",
        });
        Self(pipeline)
    }

    fn build_shader(sh_deg: u32, compressed: bool) -> String {
        let shader_src: &str = if !compressed {
            include_str!("shaders/preprocess.wgsl")
        } else {
            include_str!("shaders/preprocess_compressed.wgsl")
        };
        let shader = format!(
            "
        const MAX_SH_DEG:u32 = {:}u;
        {:}",
            sh_deg, shader_src
        );
        return shader;
    }

    fn run<'a>(
        &mut self,
        encoder: &'a mut wgpu::CommandEncoder,
        pc: &PointCloud,
        camera: &UniformBuffer<CameraUniform>,
        render_settings: &UniformBuffer<SplattingArgsUniform>,
        sort_bg: &wgpu::BindGroup,
    ) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("preprocess compute pass"),
            ..Default::default()
        });
        pass.set_pipeline(&self.0);
        pass.set_bind_group(0, camera.bind_group(), &[]);
        pass.set_bind_group(1, pc.bind_group(), &[]);
        pass.set_bind_group(2, &sort_bg, &[]);
        pass.set_bind_group(3, render_settings.bind_group(), &[]);

        let wgs_x = (pc.num_points() as f32 / 256.0).ceil() as u32;
        pass.dispatch_workgroups(wgs_x, 1, 1);
    }
}


#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct MatrixWrapper {
    matrix: Matrix4<f32>,
}
impl Default for MatrixWrapper{
    fn default() -> Self {
        Self {
            matrix: Matrix4::identity(),
        }
    }

}
/// Contains everything for the temporal smoothing in-between pass. The pass starts by taking the
/// currently rendered frame without post-processing, smoothing over pixels with data from
/// previous frames and outputting it to the display texture for further processing
pub struct TemporalSmoothing {
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,

    extent: wgpu::Extent3d,
    // frame buffer for the accumulation buffer
    accu_frame: wgpu::TextureView,
    accu_depth: wgpu::TextureView,
    accu_frame_transformation: UniformBuffer<MatrixWrapper>,

    // frame buffer for the frame just finished by the main render pass and Depth Texture
    current_frame: wgpu::TextureView,
    current_depth: wgpu::TextureView,
}

impl TemporalSmoothing {
    pub const OUT_TEXTURE_FORMAT_COL: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
    pub const OUT_TEXTURE_FORMAT_DEP: wgpu::TextureFormat = wgpu::TextureFormat::R32Float;
    pub const IN_TEXTURE_FORMAT_DEP: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
    pub const PIXELS_PER_COMPUTE_AXIS: u32 = 4;
    pub const CURRENT_COLOUR_WEIGHT: f32 = 0.1;

    pub fn texture(&self) -> &wgpu::TextureView {
        &self.current_frame
    }
    pub fn depth_texture(&self) -> &wgpu::TextureView {
        &self.current_depth
    }
    fn bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("prev frame info bind group layout"),
            entries: &[
                // current frame, to be sampled
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // current frame's depth texture
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // accumulator frame
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::ReadWrite,
                        format: TemporalSmoothing::OUT_TEXTURE_FORMAT_COL,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                // accumulator frame depth map
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::ReadWrite,
                        format: TemporalSmoothing::OUT_TEXTURE_FORMAT_DEP,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                // dst frame, output texture
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::ReadWrite,
                        format: TemporalSmoothing::OUT_TEXTURE_FORMAT_COL,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                // dst frame depth map
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::ReadWrite,
                        format: TemporalSmoothing::OUT_TEXTURE_FORMAT_DEP,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                // previous frame transformation
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        })
    }

    pub fn new(
        device: &wgpu::Device,
        output_texture: &wgpu::TextureView,
        output_depth: &wgpu::TextureView,
        width: u32,
        height: u32,
    ) -> Self {
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Temporal Smoothing"),
            bind_group_layouts: &[
                &UniformBuffer::<CameraUniform>::bind_group_layout(device),
                &Self::bind_group_layout(device),
                &UniformBuffer::<SplattingArgsUniform>::bind_group_layout(device),
            ],
            push_constant_ranges: &[],
        });
        let shader = device.create_shader_module(include_wgsl!("shaders/temporal_smoothing.wgsl"));

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Temporal Smoothing"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "cs_main",
        });
        let trafo_uniform = UniformBuffer::new_default(device, Some("accu frame transformation"));
        let (current_frame, current_depth, accu_frame, accu_depth, bind_group) =
            Self::create_render_target(device, width, height, output_texture, output_depth, &trafo_uniform);

        Self {
            pipeline,
            bind_group,
            extent: Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            current_frame,
            current_depth,
            accu_frame,
            accu_depth,
            accu_frame_transformation: trafo_uniform,
        }
    }

    fn create_render_target(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        output_texture: &wgpu::TextureView,
        output_depth: &wgpu::TextureView,
        accu_trafo: &UniformBuffer<MatrixWrapper>,
    ) -> (
        wgpu::TextureView,
        wgpu::TextureView,
        wgpu::TextureView,
        wgpu::TextureView,
        wgpu::BindGroup,
    ) {
        let extent = Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };
        let current_frame = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("currently rendered image"),
            size: extent.clone(),
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: Self::OUT_TEXTURE_FORMAT_COL,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let cf_view = current_frame.create_view(&Default::default());
        let current_depth = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("currently rendered depth image"),
            size: extent.clone(),
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: Self::IN_TEXTURE_FORMAT_DEP,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let cfd_view = current_depth.create_view(&Default::default());

        let accu_frame = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("accumulator frame"),
            size: extent.clone(),
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: Self::OUT_TEXTURE_FORMAT_COL,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        });
        let ac_view = accu_frame.create_view(&Default::default());

        let accu_depth = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("accumulator depth frame"),
            size: extent.clone(),
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: Self::OUT_TEXTURE_FORMAT_DEP,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let acd_view = accu_depth.create_view(&Default::default());

        let bind_group = Self::build_bind_group(
            device,
            &cf_view,
            &cfd_view,
            &ac_view,
            &acd_view,
            output_texture,
            output_depth,
            accu_trafo,
        );
        return (cf_view, cfd_view, ac_view, acd_view, bind_group);
    }

    pub fn render(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        camera_uniform: &UniformBuffer<CameraUniform>,
        render_settings: &UniformBuffer<SplattingArgsUniform>,
    ) {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Temporal Smoothing Compute Pass"),
            ..Default::default()
        });
        compute_pass.set_bind_group(0, camera_uniform.bind_group(), &[]);
        compute_pass.set_bind_group(1, &self.bind_group, &[]);
        compute_pass.set_bind_group(2, render_settings.bind_group(), &[]);
        compute_pass.set_pipeline(&self.pipeline);
        compute_pass.dispatch_workgroups(
            self.extent.width / Self::PIXELS_PER_COMPUTE_AXIS + 1,
            self.extent.height / Self::PIXELS_PER_COMPUTE_AXIS + 1,
            1,
        );
    }

    pub fn swap_framebuffers(&mut self, display: &mut Display) {
        std::mem::swap(display.texture_mut(), &mut self.accu_frame);
        std::mem::swap(display.depth_texture_mut(), &mut self.accu_depth);
    }
    pub fn set_accu_camera(&mut self, camera: &UniformBuffer<CameraUniform>) {
        let uniform = camera.data();
        let self_uniform = self.accu_frame_transformation.as_mut();
        self_uniform.matrix = uniform.proj_matrix * uniform.view_matrix;
    }
    pub fn rewrite_bind_group(
        &mut self,
        device: &wgpu::Device,
        output_texture: &wgpu::TextureView,
        output_depth: &wgpu::TextureView,
    ) {
        self.bind_group = Self::build_bind_group(
            device,
            &self.current_frame,
            &self.current_depth,
            &self.accu_frame,
            &self.accu_depth,
            output_texture,
            output_depth,
            &self.accu_frame_transformation);
    }

    fn build_bind_group(
        device: &wgpu::Device,
        current_frame: &wgpu::TextureView,
        current_depth: &wgpu::TextureView,
        accu_frame: &wgpu::TextureView,
        accu_depth: &wgpu::TextureView,
        output_texture: &wgpu::TextureView,
        output_depth: &wgpu::TextureView,
        accu_trafo: &UniformBuffer<MatrixWrapper>,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("render target bind group of tempsmooth"),
            layout: &Self::bind_group_layout(device),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(current_frame),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(current_depth),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(accu_frame),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(accu_depth),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(output_texture),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(output_depth),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding { buffer: accu_trafo.buffer(), offset: 0, size: None }),
                },
            ],
        })
    }

    pub fn resize(
        &mut self,
        device: &wgpu::Device,
        width: u32,
        height: u32,
        output_texture: &wgpu::TextureView,
        output_depth: &wgpu::TextureView,
    ) {
        let (c, cd, a, ad, bind_group) =
            Self::create_render_target(device, width, height, output_texture, output_depth, &self.accu_frame_transformation);
        self.bind_group = bind_group;
        self.accu_frame = a;
        self.accu_depth = ad;
        self.current_frame = c;
        self.current_depth = cd;
        self.extent = Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };
    }
}

pub struct Display {
    pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
    view: wgpu::TextureView,
    depth_view: wgpu::TextureView,

    sampler: wgpu::Sampler,
    env_bg: wgpu::BindGroup,
    has_env_map: bool,
}

impl Display {
    pub fn new(
        device: &wgpu::Device,
        target_format: wgpu::TextureFormat,
        width: u32,
        height: u32,
    ) -> Self {
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("display pipeline layout"),
            bind_group_layouts: &[
                &Self::bind_group_layout(device),
                &Self::env_map_bind_group_layout(device),
                &UniformBuffer::<CameraUniform>::bind_group_layout(device),
                &UniformBuffer::<SplattingArgsUniform>::bind_group_layout(device),
            ],
            push_constant_ranges: &[],
        });
        let shader = device.create_shader_module(include_wgsl!("shaders/display.wgsl"));
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("display pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview: None,
        });
        let env_bg = Self::create_env_map_bg(device, None);
        let (view, depth_view, sampler, bind_group) =
            Self::create_render_target(device, width, height);
        Self {
            pipeline,
            view,
            depth_view,
            sampler,
            bind_group,
            env_bg,
            has_env_map: false,
        }
    }

    pub fn texture(&self) -> &wgpu::TextureView {
        &self.view
    }
    pub fn texture_mut(&mut self) -> &mut wgpu::TextureView {
        &mut self.view
    }

    pub fn depth_texture(&self) -> &wgpu::TextureView {
        &self.depth_view
    }
    pub fn depth_texture_mut(&mut self) -> &mut wgpu::TextureView {
        &mut self.depth_view
    }

    fn env_map_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("env map bind group layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        })
    }

    fn create_env_map_bg(
        device: &wgpu::Device,
        env_texture: Option<&wgpu::TextureView>,
    ) -> wgpu::BindGroup {
        let env_map_placeholder = device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some("placeholder"),
                size: Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba16Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            })
            .create_view(&Default::default());
        let env_texture_view = env_texture.unwrap_or(&env_map_placeholder);
        let env_map_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("env map sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        return device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("env map bind group"),
            layout: &Self::env_map_bind_group_layout(device),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(env_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&env_map_sampler),
                },
            ],
        });
    }

    pub fn set_env_map(&mut self, device: &wgpu::Device, env_texture: Option<&wgpu::TextureView>) {
        self.env_bg = Self::create_env_map_bg(device, env_texture);
        self.has_env_map = env_texture.is_some();
    }

    pub fn has_env_map(&self) -> bool {
        self.has_env_map
    }

    fn create_render_target(
        device: &wgpu::Device,
        width: u32,
        height: u32,
    ) -> (
        wgpu::TextureView,
        wgpu::TextureView,
        wgpu::Sampler,
        wgpu::BindGroup,
    ) {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("display render image"),
            size: Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: TemporalSmoothing::OUT_TEXTURE_FORMAT_COL,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::STORAGE_BINDING, // storage binding because tempsmoother needs it
            view_formats: &[],
        });
        let texture_view = texture.create_view(&Default::default());

        let depth_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("display render depth"),
            size: Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: TemporalSmoothing::OUT_TEXTURE_FORMAT_DEP,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::STORAGE_BINDING, // storage binding because tempsmoother needs it
            view_formats: &[],
        });
        let dt_view = depth_tex.create_view(&Default::default());

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("render target bind group"),
            layout: &Display::bind_group_layout(device),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });
        return (texture_view, dt_view, sampler, bind_group);
    }

    pub fn bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("display bind group layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        })
    }

    pub fn rewrite_bind_group(&mut self, device: &wgpu::Device) {
        self.bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("display shader bind group"),
            layout: &Display::bind_group_layout(device),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&self.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        });
    }

    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        let (view, dt_view, sampler, bind_group) =
            Self::create_render_target(device, width, height);
        self.bind_group = bind_group;
        self.view = view;
        self.depth_view = dt_view;
        self.sampler = sampler;
    }

    pub fn render(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        background_color: wgpu::Color,
        camera: &UniformBuffer<CameraUniform>,
        render_settings: &UniformBuffer<SplattingArgsUniform>,
    ) {
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("render pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(background_color),
                    store: wgpu::StoreOp::Store,
                },
            })],
            ..Default::default()
        });
        render_pass.set_bind_group(0, &self.bind_group, &[]);
        render_pass.set_bind_group(1, &self.env_bg, &[]);
        render_pass.set_bind_group(2, camera.bind_group(), &[]);
        render_pass.set_bind_group(3, render_settings.bind_group(), &[]);
        render_pass.set_pipeline(&self.pipeline);

        render_pass.draw(0..4, 0..1);
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct SplattingArgs {
    pub camera: PerspectiveCamera,
    pub viewport: Vector2<u32>,
    pub gaussian_scaling: f32,
    pub max_sh_deg: u32,
    pub show_env_map: bool,
    pub mip_splatting: Option<bool>,
    pub kernel_size: Option<f32>,
    pub clipping_box: Option<Aabb<f32>>,
    pub walltime: Duration,
    pub scene_center: Option<Point3<f32>>,
    pub scene_extend: Option<f32>,
    pub current_colour_weight: f32,
}

impl Hash for SplattingArgs {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.camera.hash(state);
        self.viewport.hash(state);
        self.max_sh_deg.hash(state);
        self.gaussian_scaling.to_bits().hash(state);
        self.show_env_map.hash(state);
        self.mip_splatting.hash(state);
        self.kernel_size.map(f32::to_bits).hash(state);
        self.walltime.hash(state);
        self.clipping_box
            .as_ref()
            .map(|b| bytemuck::bytes_of(&b.min))
            .hash(state);
        self.clipping_box
            .as_ref()
            .map(|b| bytemuck::bytes_of(&b.max))
            .hash(state);

        bytemuck::bytes_of(&self.current_colour_weight).hash(state);
    }
}

pub const DEFAULT_KERNEL_SIZE: f32 = 0.3;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SplattingArgsUniform {
    clipping_box_min: Vector4<f32>,
    clipping_box_max: Vector4<f32>,
    gaussian_scaling: f32,
    max_sh_deg: u32,
    show_env_map: u32,
    mip_splatting: u32,

    kernel_size: f32,
    walltime: f32,
    scene_extend: f32,
    
    current_coulour_weight: f32,

    scene_center: Vector4<f32>,
}

impl SplattingArgsUniform {
    /// replaces values with default values for point cloud
    pub fn from_args_and_pc(args: SplattingArgs, pc: &PointCloud) -> Self {
        Self {
            gaussian_scaling: args.gaussian_scaling,
            max_sh_deg: args.max_sh_deg,
            show_env_map: args.show_env_map as u32,
            mip_splatting: args
                .mip_splatting
                .map(|v| v as u32)
                .unwrap_or(pc.mip_splatting().unwrap_or(false) as u32),
            kernel_size: args
                .kernel_size
                .unwrap_or(pc.dilation_kernel_size().unwrap_or(DEFAULT_KERNEL_SIZE)),
            clipping_box_min: args
                .clipping_box
                .map_or(pc.bbox().min, |b| b.min)
                .to_vec()
                .extend(0.),
            clipping_box_max: args
                .clipping_box
                .map_or(pc.bbox().max, |b| b.max)
                .to_vec()
                .extend(0.),
            walltime: args.walltime.as_secs_f32(),
            scene_center: pc.center().to_vec().extend(0.),
            scene_extend: args
                .scene_extend
                .unwrap_or(pc.bbox().radius())
                .max(pc.bbox().radius()),
            current_coulour_weight: args.current_colour_weight,
            ..Default::default()
        }
    }
}

impl Default for SplattingArgsUniform {
    fn default() -> Self {
        Self {
            gaussian_scaling: 1.0,
            max_sh_deg: 3,
            show_env_map: true as u32,
            mip_splatting: false as u32,
            kernel_size: DEFAULT_KERNEL_SIZE,
            clipping_box_max: Vector4::new(f32::INFINITY, f32::INFINITY, f32::INFINITY, 0.),
            clipping_box_min: Vector4::new(
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                0.,
            ),
            walltime: 0.,
            scene_center: Vector4::new(0., 0., 0., 0.),
            scene_extend: 1.,
            current_coulour_weight: TemporalSmoothing::CURRENT_COLOUR_WEIGHT,
        }
    }
}
