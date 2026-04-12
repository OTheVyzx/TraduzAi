use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result};
use clap::Parser;
use image::DynamicImage;
use koharu_core::{TextBlock, TextDirection};
use koharu_llm::paddleocr_vl::{PaddleOcrVl, PaddleOcrVlTask};
use koharu_llm::safe::llama_backend::LlamaBackend;
use koharu_ml::comic_text_bubble_detector::ComicTextBubbleDetector;
use koharu_ml::comic_text_detector::crop_text_block_bbox;
use koharu_runtime::{default_app_data_root, ComputePolicy, RuntimeManager};
use serde::{Deserialize, Serialize};

#[derive(Parser, Debug)]
#[command(author, version, about)]
struct Cli {
    #[arg(long, value_name = "FILE")]
    request_file: Option<PathBuf>,

    #[arg(long, default_value_t = false)]
    warmup: bool,

    #[arg(long, value_name = "DIR")]
    runtime_root: Option<PathBuf>,

    #[arg(long, default_value_t = false)]
    cpu: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
struct VisionRequest {
    image_path: String,
    #[serde(default = "default_mode")]
    mode: String,
    #[serde(default)]
    region: Option<[u32; 4]>,
    #[serde(default)]
    runtime_root: Option<String>,
    #[serde(default)]
    cpu: Option<bool>,
    #[serde(default)]
    max_new_tokens: Option<usize>,
    #[serde(default)]
    detection_threshold: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
struct VisionResponse {
    status: String,
    image_width: u32,
    image_height: u32,
    text_blocks: Vec<TextBlockOutput>,
    bubble_regions: Vec<BubbleRegionOutput>,
    timings_ms: BTreeMap<String, u128>,
    warnings: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
struct TextBlockOutput {
    bbox: [u32; 4],
    confidence: f32,
    text: String,
    detector: Option<String>,
    source_direction: Option<String>,
    line_polygons: Option<Vec<[[f32; 2]; 4]>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
struct BubbleRegionOutput {
    bbox: [u32; 4],
    confidence: f32,
}

fn default_mode() -> String {
    "page".into()
}

fn init_tracing() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .try_init();
}

#[tokio::main]
async fn main() -> Result<()> {
    init_tracing();

    let cli = Cli::parse();
    if cli.warmup {
        let runtime_root = cli
            .runtime_root
            .clone()
            .unwrap_or_else(|| default_app_data_root().into());
        warmup(runtime_root, cli.cpu).await?;
        println!(
            "{}",
            serde_json::to_string(&serde_json::json!({"status": "ok"}))?
        );
        return Ok(());
    }

    let request = read_request(cli.request_file.as_deref())?;
    let response = run_request(&request).await?;
    println!("{}", serde_json::to_string(&response)?);
    Ok(())
}

fn read_request(request_file: Option<&Path>) -> Result<VisionRequest> {
    let raw = if let Some(path) = request_file {
        fs::read_to_string(path)
            .with_context(|| format!("failed to read request file `{}`", path.display()))?
    } else {
        let mut input = String::new();
        std::io::stdin()
            .read_line(&mut input)
            .context("failed to read request from stdin")?;
        input
    };
    serde_json::from_str(&raw).context("failed to parse request JSON")
}

async fn warmup(runtime_root: PathBuf, cpu: bool) -> Result<()> {
    let runtime = build_runtime(&runtime_root, cpu)?;
    runtime.prepare().await?;
    let _detector = ComicTextBubbleDetector::load(&runtime, cpu).await?;
    koharu_llm::sys::initialize(&runtime)?;
    let backend = Arc::new(LlamaBackend::init()?);
    let _ocr = PaddleOcrVl::load(&runtime, cpu, backend).await?;
    Ok(())
}

async fn run_request(request: &VisionRequest) -> Result<VisionResponse> {
    let started = Instant::now();
    let cpu = request.cpu.unwrap_or(false);
    let runtime_root = request
        .runtime_root
        .as_ref()
        .map(PathBuf::from)
        .unwrap_or_else(|| default_app_data_root().into());
    let runtime = build_runtime(&runtime_root, cpu)?;

    let prepare_started = Instant::now();
    runtime.prepare().await?;
    koharu_llm::sys::initialize(&runtime)?;
    let backend = Arc::new(LlamaBackend::init()?);

    let mut timings_ms = BTreeMap::new();
    timings_ms.insert("prepare".into(), prepare_started.elapsed().as_millis());

    let image_path = PathBuf::from(&request.image_path);
    let source_image = image::open(&image_path)
        .with_context(|| format!("failed to open image `{}`", image_path.display()))?;
    let original_width = source_image.width();
    let original_height = source_image.height();

    let (working_image, offset_x, offset_y) = if request.mode == "region" {
        let region = request
            .region
            .with_context(|| "region mode requires `region` coordinates")?;
        crop_region(&source_image, region)
    } else {
        (source_image.clone(), 0, 0)
    };

    let detect_started = Instant::now();
    let detector = ComicTextBubbleDetector::load(&runtime, cpu).await?;
    let detection = detector
        .inference_with_threshold(&working_image, request.detection_threshold.unwrap_or(0.3))?;
    timings_ms.insert("detect".into(), detect_started.elapsed().as_millis());

    let ocr_started = Instant::now();
    let mut ocr = PaddleOcrVl::load(&runtime, cpu, backend).await?;
    let crops: Vec<DynamicImage> = detection
        .text_blocks
        .iter()
        .map(|block| crop_text_block_bbox(&working_image, block))
        .collect();
    let ocr_outputs = if crops.is_empty() {
        Vec::new()
    } else {
        ocr.inference_images(
            &crops,
            PaddleOcrVlTask::Ocr,
            request.max_new_tokens.unwrap_or(128),
        )?
    };
    timings_ms.insert("ocr".into(), ocr_started.elapsed().as_millis());

    let normalize_started = Instant::now();
    let text_blocks = detection
        .text_blocks
        .iter()
        .zip(ocr_outputs.iter())
        .map(|(block, output)| {
            normalize_text_block(block, output.text.as_str(), offset_x, offset_y)
        })
        .collect();
    let bubble_regions = detection
        .detections
        .iter()
        .filter(|region| region.is_bubble())
        .map(|region| normalize_bubble_region(region.bbox, region.score, offset_x, offset_y))
        .collect();
    timings_ms.insert("normalize".into(), normalize_started.elapsed().as_millis());
    timings_ms.insert("total".into(), started.elapsed().as_millis());

    Ok(VisionResponse {
        status: "ok".into(),
        image_width: original_width,
        image_height: original_height,
        text_blocks,
        bubble_regions,
        timings_ms,
        warnings: Vec::new(),
    })
}

fn build_runtime(runtime_root: &Path, cpu: bool) -> Result<RuntimeManager> {
    RuntimeManager::new(
        runtime_root,
        if cpu {
            ComputePolicy::CpuOnly
        } else {
            ComputePolicy::PreferGpu
        },
    )
    .with_context(|| format!("failed to create runtime at `{}`", runtime_root.display()))
}

fn crop_region(image: &DynamicImage, region: [u32; 4]) -> (DynamicImage, u32, u32) {
    let [x1, y1, x2, y2] = region;
    let x1 = x1.min(image.width());
    let y1 = y1.min(image.height());
    let x2 = x2.min(image.width()).max(x1 + 1);
    let y2 = y2.min(image.height()).max(y1 + 1);
    (
        image.crop_imm(x1, y1, x2.saturating_sub(x1), y2.saturating_sub(y1)),
        x1,
        y1,
    )
}

fn normalize_text_block(
    block: &TextBlock,
    text: &str,
    offset_x: u32,
    offset_y: u32,
) -> TextBlockOutput {
    let bbox = [
        block.x.max(0.0).floor() as u32 + offset_x,
        block.y.max(0.0).floor() as u32 + offset_y,
        (block.x + block.width).ceil().max(block.x + 1.0) as u32 + offset_x,
        (block.y + block.height).ceil().max(block.y + 1.0) as u32 + offset_y,
    ];
    TextBlockOutput {
        bbox,
        confidence: block.confidence,
        text: text.to_string(),
        detector: block.detector.clone(),
        source_direction: block.source_direction.map(text_direction_to_string),
        line_polygons: block
            .line_polygons
            .as_ref()
            .map(|lines| offset_line_polygons(lines, offset_x as f32, offset_y as f32)),
    }
}

fn normalize_bubble_region(
    bbox: [f32; 4],
    confidence: f32,
    offset_x: u32,
    offset_y: u32,
) -> BubbleRegionOutput {
    BubbleRegionOutput {
        bbox: [
            bbox[0].max(0.0).floor() as u32 + offset_x,
            bbox[1].max(0.0).floor() as u32 + offset_y,
            bbox[2].ceil().max(bbox[0] + 1.0) as u32 + offset_x,
            bbox[3].ceil().max(bbox[1] + 1.0) as u32 + offset_y,
        ],
        confidence,
    }
}

fn offset_line_polygons(
    line_polygons: &Vec<[[f32; 2]; 4]>,
    offset_x: f32,
    offset_y: f32,
) -> Vec<[[f32; 2]; 4]> {
    line_polygons
        .iter()
        .map(|line| {
            let mut shifted = *line;
            for point in &mut shifted {
                point[0] += offset_x;
                point[1] += offset_y;
            }
            shifted
        })
        .collect()
}

fn text_direction_to_string(direction: TextDirection) -> String {
    match direction {
        TextDirection::Horizontal => "horizontal".into(),
        TextDirection::Vertical => "vertical".into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vision_worker_contract_roundtrips_page_request() {
        let request = VisionRequest {
            image_path: "page.jpg".into(),
            mode: "page".into(),
            region: None,
            runtime_root: Some("D:/traduzai_data".into()),
            cpu: Some(false),
            max_new_tokens: Some(128),
            detection_threshold: Some(0.3),
        };

        let encoded = serde_json::to_string(&request).expect("request serializes");
        let decoded: VisionRequest = serde_json::from_str(&encoded).expect("request deserializes");
        assert_eq!(decoded, request);
    }

    #[test]
    fn vision_worker_contract_roundtrips_region_request() {
        let request = VisionRequest {
            image_path: "page.jpg".into(),
            mode: "region".into(),
            region: Some([20, 40, 140, 220]),
            runtime_root: Some("D:/traduzai_data".into()),
            cpu: Some(false),
            max_new_tokens: Some(64),
            detection_threshold: Some(0.4),
        };

        let encoded = serde_json::to_string(&request).expect("request serializes");
        let decoded: VisionRequest = serde_json::from_str(&encoded).expect("request deserializes");
        assert_eq!(decoded, request);
    }

    #[test]
    fn normalize_bubble_region_offsets_coordinates() {
        let region = normalize_bubble_region([10.2, 18.6, 40.4, 52.1], 0.91, 100, 200);
        assert_eq!(region.bbox, [110, 218, 141, 253]);
        assert!((region.confidence - 0.91).abs() < f32::EPSILON);
    }
}
