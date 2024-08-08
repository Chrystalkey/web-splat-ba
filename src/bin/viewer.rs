use clap::Parser;
#[allow(unused_imports)]
use std::{fmt::Debug, fs::File, path::PathBuf};
#[allow(unused_imports)]
use web_splats::{open_window, RenderConfig};

#[derive(Debug, Parser)]
#[command(author, version, about)]
struct Opt {
    /// Input file
    input: PathBuf,

    /// Scene json file
    scene: Option<PathBuf>,

    #[arg(long, default_value_t = false)]
    no_vsync: bool,

    /// Enter Performance test mode for <N> frames, then exit
    #[arg(long)]
    perftest: Option<u32>,

    /// Output timing information to file
    #[arg(long)]
    timing_output : Option<PathBuf>,

    /// Disables UI. Reenable by pressing 'u'
    #[arg(long, short, default_value_t = false)]
    no_ui: bool,
    /// Initial window width
    #[arg(long)]
    width: Option<u32>,
    /// Initial window height
    #[arg(long)]
    height: Option<u32>,

    /// Support HDR rendering
    #[arg(long, default_value_t = false)]
    hdr: bool,

    /// Sky box image
    #[arg(long)]
    skybox: Option<PathBuf>,
}

/// check if there is a scene file in the same directory or parent directory as the input file
#[allow(unused)]
fn try_find_scene_file(input: &PathBuf, depth: u32) -> Option<PathBuf> {
    if let Some(parent) = input.parent() {
        let scene = parent.join("cameras.json");
        if scene.exists() {
            return Some(scene);
        }
        if depth == 0 {
            return None;
        }
        return try_find_scene_file(&parent.to_path_buf(), depth - 1);
    }
    return None;
}

#[cfg(not(target_arch = "wasm32"))]
#[pollster::main]
async fn main() {
    let mut opt = Opt::parse();

    if opt.scene.is_none() {
        opt.scene = try_find_scene_file(&opt.input, 2);
        log::warn!("No scene file specified, using {:?}", opt.scene);
    }
    let data_file = File::open(&opt.input).unwrap();

    let scene_file = opt.scene.as_ref().map(|p| File::open(p).unwrap());

    if opt.no_vsync {
        log::info!("V-sync disabled");
    }
    if opt.no_ui {
        log::info!("UI disabled");
    }
    if let Some(ptest) = opt.perftest {
        log::info!("Performance test for {} frames", ptest);
    } else {
        log::info!("Performance Testing disabled, normal rendering");
    }
    open_window(
        data_file,
        scene_file,
        RenderConfig {
            no_vsync: opt.no_vsync,
            skybox: opt.skybox,
            hdr: opt.hdr,
            perftest: opt.perftest,
            ui_enabled: opt.perftest.is_none() && !opt.no_ui,
            width: opt.width,
            height: opt.height,
            timing_output: opt.timing_output,
        },
        Some(opt.input),
        opt.scene,
    )
    .await;
}
#[cfg(target_arch = "wasm32")]
fn main() {
    todo!("not implemented")
}
