use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use traduzai_renderer_bridge::{render_to_png, renderer_rasterizer_for_debug, RenderRequest};

fn main() {
    if let Err(err) = run() {
        eprintln!("{err:#}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let mut request_path: Option<PathBuf> = None;
    let mut output_path: Option<PathBuf> = None;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--request" => request_path = args.next().map(PathBuf::from),
            "--output" => output_path = args.next().map(PathBuf::from),
            "--help" | "-h" => {
                println!("renderer-bridge --request request.json --output output.png");
                return Ok(());
            }
            other => bail!("unknown argument {other}"),
        }
    }

    let request_path = request_path.context("missing --request")?;
    let output_path = output_path.context("missing --output")?;
    let payload = std::fs::read_to_string(&request_path)
        .with_context(|| format!("failed to read {}", request_path.display()))?;
    let request: RenderRequest = serde_json::from_str(&payload)
        .with_context(|| format!("invalid request JSON {}", request_path.display()))?;
    render_to_png(&request, &output_path)?;
    println!(
        "{}",
        serde_json::json!({
            "status": "ok",
            "output": output_path,
            "blocks": request.blocks.len(),
            "rasterizer": renderer_rasterizer_for_debug(),
        })
    );
    Ok(())
}
