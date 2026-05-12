use std::collections::BTreeMap;
use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use anyhow::{anyhow, Context, Result};
use clap::Parser;
use image::DynamicImage;
use koharu_llm::paddleocr_vl::{PaddleOcrVl, PaddleOcrVlTask};
use koharu_llm::safe::llama_backend::LlamaBackend;
use koharu_ml::comic_text_bubble_detector::ComicTextBubbleDetector;
use koharu_ml::comic_text_detector::crop_text_block_bbox;
use koharu_ml::{TextDirection, TextRegion};
use koharu_runtime::{default_app_data_root, ComputePolicy, RuntimeManager};
use serde::{Deserialize, Serialize};

#[derive(Parser, Debug)]
#[command(author, version, about)]
struct Cli {
    #[arg(long, value_name = "FILE")]
    request_file: Option<PathBuf>,

    #[arg(long, value_name = "FILE")]
    batch_request_file: Option<PathBuf>,

    #[arg(long, default_value_t = false)]
    warmup: bool,

    #[arg(long, default_value_t = false)]
    stdio_server: bool,

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
    #[serde(default, alias = "knownTextBBoxes")]
    known_text_bboxes: Vec<[u32; 4]>,
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
struct VisionBatchRequest {
    requests: Vec<VisionRequest>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
struct VisionBatchItemResponse {
    index: usize,
    status: String,
    response: Option<VisionResponse>,
    error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
struct VisionBatchResponse {
    status: String,
    responses: Vec<VisionBatchItemResponse>,
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
        .with_writer(std::io::stderr)
        .try_init();
}

fn main() -> Result<()> {
    init_tracing();

    std::thread::Builder::new()
        .name("traduzai-vision".into())
        .stack_size(64 * 1024 * 1024)
        .spawn(|| {
            let result: Result<()> = (|| {
                let rt = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()?;
                rt.block_on(run_cli())
            })();
            if let Err(error) = result {
                eprintln!("Error: {error:#}");
                std::process::exit(1);
            }
            std::process::exit(0);
        })?
        .join()
        .map_err(|_| anyhow!("vision worker thread panicked"))?
}

async fn run_cli() -> Result<()> {
    let cli = Cli::parse();
    if cli.warmup {
        let runtime_root = cli
            .runtime_root
            .clone()
            .unwrap_or_else(|| default_app_data_root().into());
        warmup(runtime_root, cli.cpu).await?;
        emit_json(&serde_json::json!({"status": "ok"}))?;
        return Ok(());
    }

    if cli.stdio_server {
        run_stdio_server(cli.runtime_root.as_deref(), cli.cpu).await?;
        return Ok(());
    }

    if let Some(batch_file) = cli.batch_request_file.as_deref() {
        let request = read_batch_request(batch_file)?;
        let response = run_batch_request(&request, cli.runtime_root.as_deref(), cli.cpu).await?;
        emit_json(&response)?;
        return Ok(());
    }

    let request = read_request(cli.request_file.as_deref())?;
    let response = run_request(&request).await?;
    emit_json(&response)?;
    Ok(())
}

fn emit_json<T: Serialize>(payload: &T) -> Result<()> {
    let mut stdout = std::io::stdout().lock();
    serde_json::to_writer(&mut stdout, payload)?;
    writeln!(stdout)?;
    stdout.flush()?;
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

fn read_batch_request(request_file: &Path) -> Result<VisionBatchRequest> {
    let raw = fs::read_to_string(request_file).with_context(|| {
        format!(
            "failed to read batch request file `{}`",
            request_file.display()
        )
    })?;
    serde_json::from_str(&raw).context("failed to parse batch request JSON")
}

async fn warmup(runtime_root: PathBuf, cpu: bool) -> Result<()> {
    let _session = build_session(&runtime_root, cpu).await?;
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

    let prepare_started = Instant::now();
    let mut session = build_session(&runtime_root, cpu).await?;
    let mut timings_ms: BTreeMap<String, u128> = BTreeMap::new();
    timings_ms.insert("prepare".into(), prepare_started.elapsed().as_millis());

    let mut response = run_request_with_session(request, &mut session)?;
    response
        .timings_ms
        .insert("prepare".into(), *timings_ms.get("prepare").unwrap_or(&0));
    response
        .timings_ms
        .insert("total".into(), started.elapsed().as_millis());
    Ok(response)
}

async fn run_batch_request(
    request: &VisionBatchRequest,
    cli_runtime_root: Option<&Path>,
    cli_cpu: bool,
) -> Result<VisionBatchResponse> {
    let started = Instant::now();
    let mut timings_ms: BTreeMap<String, u128> = BTreeMap::new();
    let warnings = Vec::new();

    if request.requests.is_empty() {
        timings_ms.insert("total".into(), started.elapsed().as_millis());
        return Ok(VisionBatchResponse {
            status: "ok".into(),
            responses: Vec::new(),
            timings_ms,
            warnings,
        });
    }

    let runtime_root = request
        .requests
        .iter()
        .find_map(|item| item.runtime_root.as_ref())
        .map(PathBuf::from)
        .or_else(|| cli_runtime_root.map(PathBuf::from))
        .unwrap_or_else(|| default_app_data_root().into());
    let cpu = request
        .requests
        .iter()
        .find_map(|item| item.cpu)
        .unwrap_or(cli_cpu);

    let prepare_started = Instant::now();
    let mut session = build_session(&runtime_root, cpu).await?;
    timings_ms.insert("prepare".into(), prepare_started.elapsed().as_millis());

    let runtime_root_str = runtime_root.to_string_lossy().to_string();
    run_batch_request_with_session(
        request,
        &mut session,
        &runtime_root_str,
        cpu,
        started,
        timings_ms,
        warnings,
    )
}

async fn run_stdio_server(cli_runtime_root: Option<&Path>, cli_cpu: bool) -> Result<()> {
    let stdin = std::io::stdin();
    let mut reader = BufReader::new(stdin.lock());
    let mut line = String::new();
    let mut session_state: Option<(String, bool, VisionWorkerSession)> = None;

    loop {
        line.clear();
        let bytes = reader
            .read_line(&mut line)
            .context("failed to read stdio server request")?;
        if bytes == 0 {
            break;
        }
        if line.trim().is_empty() {
            continue;
        }

        let started = Instant::now();
        let request: VisionBatchRequest = match serde_json::from_str(line.trim()) {
            Ok(request) => request,
            Err(error) => {
                emit_json(&serde_json::json!({
                    "status": "error",
                    "error": format!("failed to parse batch request JSON: {error}"),
                }))?;
                continue;
            }
        };

        if request.requests.is_empty() {
            emit_json(&VisionBatchResponse {
                status: "ok".into(),
                responses: Vec::new(),
                timings_ms: BTreeMap::from([("total".into(), started.elapsed().as_millis())]),
                warnings: Vec::new(),
            })?;
            continue;
        }

        let runtime_root = request
            .requests
            .iter()
            .find_map(|item| item.runtime_root.as_ref())
            .map(PathBuf::from)
            .or_else(|| cli_runtime_root.map(PathBuf::from))
            .unwrap_or_else(|| default_app_data_root().into());
        let cpu = request
            .requests
            .iter()
            .find_map(|item| item.cpu)
            .unwrap_or(cli_cpu);
        let runtime_root_str = runtime_root.to_string_lossy().to_string();

        let mut timings_ms: BTreeMap<String, u128> = BTreeMap::new();
        let needs_session = session_state
            .as_ref()
            .map(|(root, state_cpu, _)| root != &runtime_root_str || *state_cpu != cpu)
            .unwrap_or(true);
        if needs_session {
            let prepare_started = Instant::now();
            match build_session(&runtime_root, cpu).await {
                Ok(session) => {
                    timings_ms.insert("prepare".into(), prepare_started.elapsed().as_millis());
                    session_state = Some((runtime_root_str.clone(), cpu, session));
                }
                Err(error) => {
                    emit_json(&serde_json::json!({
                        "status": "error",
                        "error": format!("{error:#}"),
                    }))?;
                    continue;
                }
            }
        } else {
            timings_ms.insert("prepare".into(), 0);
            timings_ms.insert("reusedSession".into(), 1);
        }

        let (_, _, session) = session_state
            .as_mut()
            .expect("stdio server session should be initialized");
        let response = run_batch_request_with_session(
            &request,
            session,
            &runtime_root_str,
            cpu,
            started,
            timings_ms,
            Vec::new(),
        )?;
        emit_json(&response)?;
    }
    Ok(())
}

fn run_batch_request_with_session(
    request: &VisionBatchRequest,
    session: &mut VisionWorkerSession,
    runtime_root_str: &str,
    cpu: bool,
    started: Instant,
    mut timings_ms: BTreeMap<String, u128>,
    mut warnings: Vec<String>,
) -> Result<VisionBatchResponse> {
    let mut responses = Vec::with_capacity(request.requests.len());
    for (index, item) in request.requests.iter().enumerate() {
        let mut effective = item.clone();
        if effective.runtime_root.is_none() {
            effective.runtime_root = Some(runtime_root_str.to_string());
        } else if effective.runtime_root.as_deref() != Some(runtime_root_str) {
            warnings.push(format!(
                "request {} uses a different runtimeRoot; batch uses `{}`",
                index, runtime_root_str
            ));
        }
        if effective.cpu.is_none() {
            effective.cpu = Some(cpu);
        } else if effective.cpu != Some(cpu) {
            warnings.push(format!(
                "request {} uses a different cpu flag; batch uses {}",
                index, cpu
            ));
        }

        let item_started = Instant::now();
        match run_request_with_session(&effective, session) {
            Ok(mut response) => {
                response
                    .timings_ms
                    .insert("batchItemTotal".into(), item_started.elapsed().as_millis());
                responses.push(VisionBatchItemResponse {
                    index,
                    status: "ok".into(),
                    response: Some(response),
                    error: None,
                });
            }
            Err(error) => responses.push(VisionBatchItemResponse {
                index,
                status: "error".into(),
                response: None,
                error: Some(format!("{error:#}")),
            }),
        }
    }

    timings_ms.insert("total".into(), started.elapsed().as_millis());
    Ok(VisionBatchResponse {
        status: "ok".into(),
        responses,
        timings_ms,
        warnings,
    })
}

struct VisionWorkerSession {
    _runtime: RuntimeManager,
    detector: ComicTextBubbleDetector,
    ocr: PaddleOcrVl,
}

async fn build_session(runtime_root: &Path, cpu: bool) -> Result<VisionWorkerSession> {
    let runtime = build_runtime(runtime_root, cpu)?;
    runtime.prepare().await?;
    let detector = ComicTextBubbleDetector::load(&runtime, cpu).await?;
    koharu_llm::sys::initialize(&runtime)?;
    let backend = Arc::new(LlamaBackend::init()?);
    koharu_llm::suppress_native_logs();
    let ocr = PaddleOcrVl::load(&runtime, cpu, backend).await?;
    Ok(VisionWorkerSession {
        _runtime: runtime,
        detector,
        ocr,
    })
}

fn run_request_with_session(
    request: &VisionRequest,
    session: &mut VisionWorkerSession,
) -> Result<VisionResponse> {
    let started = Instant::now();
    let mut timings_ms: BTreeMap<String, u128> = BTreeMap::new();

    let image_path = PathBuf::from(&request.image_path);
    let source_image = image::open(&image_path)
        .with_context(|| format!("failed to open image `{}`", image_path.display()))?;
    let original_width = source_image.width();
    let original_height = source_image.height();

    let mode = request.mode.as_str();
    let (working_image, offset_x, offset_y) =
        if mode == "region" || (mode == "ocrOnly" && request.region.is_some()) {
            let region = request
                .region
                .with_context(|| "region mode requires `region` coordinates")?;
            crop_region(&source_image, region)
        } else {
            (source_image.clone(), 0, 0)
        };

    let (text_blocks, bubble_regions) = if mode == "ocrOnly" {
        let detect_started = Instant::now();
        let known_bboxes: Vec<[u32; 4]> = request
            .known_text_bboxes
            .iter()
            .filter_map(|bbox| clamp_bbox(*bbox, working_image.width(), working_image.height()))
            .collect();
        timings_ms.insert("detect".into(), detect_started.elapsed().as_millis());
        timings_ms.insert("detectSkipped".into(), 1);
        timings_ms.insert("knownTextBboxCount".into(), known_bboxes.len() as u128);
        if known_bboxes.is_empty() {
            (Vec::new(), Vec::new())
        } else {
            let ocr_started = Instant::now();
            let crops: Vec<DynamicImage> = known_bboxes
                .iter()
                .map(|bbox| crop_bbox(&working_image, *bbox))
                .collect();
            let ocr_outputs = session.ocr.inference_images(
                &crops,
                PaddleOcrVlTask::Ocr,
                request.max_new_tokens.unwrap_or(128),
            )?;
            timings_ms.insert("ocr".into(), ocr_started.elapsed().as_millis());

            let normalize_started = Instant::now();
            let text_blocks = known_bboxes
                .iter()
                .zip(ocr_outputs.iter())
                .map(|(bbox, output)| {
                    normalize_known_text_bbox(*bbox, output.text.as_str(), offset_x, offset_y)
                })
                .collect();
            let bubble_regions = known_bboxes
                .iter()
                .map(|bbox| normalize_known_bubble_bbox(*bbox, offset_x, offset_y))
                .collect();
            timings_ms.insert("normalize".into(), normalize_started.elapsed().as_millis());
            (text_blocks, bubble_regions)
        }
    } else {
        let detect_started = Instant::now();
        let detection = session
            .detector
            .inference_with_threshold(&working_image, request.detection_threshold.unwrap_or(0.3))?;
        timings_ms.insert("detect".into(), detect_started.elapsed().as_millis());

        let ocr_started = Instant::now();
        let crops: Vec<DynamicImage> = detection
            .text_blocks
            .iter()
            .map(|block| crop_text_block_bbox(&working_image, block))
            .collect();
        let ocr_outputs = if crops.is_empty() {
            Vec::new()
        } else {
            session.ocr.inference_images(
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
        (text_blocks, bubble_regions)
    };
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

fn clamp_bbox(bbox: [u32; 4], width: u32, height: u32) -> Option<[u32; 4]> {
    let [x1, y1, x2, y2] = bbox;
    let x1 = x1.min(width);
    let y1 = y1.min(height);
    let x2 = x2.min(width).max(x1.saturating_add(1));
    let y2 = y2.min(height).max(y1.saturating_add(1));
    if x2 <= x1 || y2 <= y1 {
        return None;
    }
    Some([x1, y1, x2, y2])
}

fn crop_bbox(image: &DynamicImage, bbox: [u32; 4]) -> DynamicImage {
    let [x1, y1, x2, y2] = bbox;
    image.crop_imm(x1, y1, x2.saturating_sub(x1), y2.saturating_sub(y1))
}

fn normalize_known_text_bbox(
    bbox: [u32; 4],
    text: &str,
    offset_x: u32,
    offset_y: u32,
) -> TextBlockOutput {
    TextBlockOutput {
        bbox: [
            bbox[0] + offset_x,
            bbox[1] + offset_y,
            bbox[2] + offset_x,
            bbox[3] + offset_y,
        ],
        confidence: 0.9,
        text: text.to_string(),
        detector: Some("known-text-bbox".into()),
        source_direction: None,
        line_polygons: None,
    }
}

fn normalize_known_bubble_bbox(bbox: [u32; 4], offset_x: u32, offset_y: u32) -> BubbleRegionOutput {
    BubbleRegionOutput {
        bbox: [
            bbox[0] + offset_x,
            bbox[1] + offset_y,
            bbox[2] + offset_x,
            bbox[3] + offset_y,
        ],
        confidence: 0.9,
    }
}

fn normalize_text_block(
    block: &TextRegion,
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
            known_text_bboxes: Vec::new(),
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
            known_text_bboxes: Vec::new(),
        };

        let encoded = serde_json::to_string(&request).expect("request serializes");
        let decoded: VisionRequest = serde_json::from_str(&encoded).expect("request deserializes");
        assert_eq!(decoded, request);
    }

    #[test]
    fn vision_worker_contract_roundtrips_batch_request() {
        let request = VisionBatchRequest {
            requests: vec![
                VisionRequest {
                    image_path: "roi-01.jpg".into(),
                    mode: "region".into(),
                    region: Some([10, 20, 120, 180]),
                    runtime_root: Some("D:/traduzai_data".into()),
                    cpu: Some(false),
                    max_new_tokens: Some(96),
                    detection_threshold: Some(0.34),
                    known_text_bboxes: Vec::new(),
                },
                VisionRequest {
                    image_path: "roi-02.jpg".into(),
                    mode: "page".into(),
                    region: None,
                    runtime_root: Some("D:/traduzai_data".into()),
                    cpu: Some(false),
                    max_new_tokens: Some(128),
                    detection_threshold: Some(0.3),
                    known_text_bboxes: Vec::new(),
                },
            ],
        };

        let encoded = serde_json::to_string(&request).expect("batch request serializes");
        assert!(encoded.contains("imagePath"));
        assert!(encoded.contains("maxNewTokens"));
        let decoded: VisionBatchRequest =
            serde_json::from_str(&encoded).expect("batch request deserializes");
        assert_eq!(decoded, request);
    }

    #[test]
    fn vision_worker_contract_roundtrips_ocr_only_request() {
        let request = VisionRequest {
            image_path: "roi.jpg".into(),
            mode: "ocrOnly".into(),
            region: None,
            runtime_root: Some("D:/traduzai_data".into()),
            cpu: Some(false),
            max_new_tokens: Some(64),
            detection_threshold: None,
            known_text_bboxes: vec![[10, 20, 120, 180]],
        };

        let encoded = serde_json::to_string(&request).expect("request serializes");
        assert!(encoded.contains("knownTextBboxes"));
        let decoded: VisionRequest = serde_json::from_str(&encoded).expect("request deserializes");
        assert_eq!(decoded, request);
    }

    #[test]
    fn vision_worker_contract_preserves_batch_item_order_and_errors() {
        let mut timings_ms = BTreeMap::new();
        timings_ms.insert("total".into(), 12);
        let response = VisionBatchResponse {
            status: "ok".into(),
            responses: vec![
                VisionBatchItemResponse {
                    index: 0,
                    status: "ok".into(),
                    response: Some(VisionResponse {
                        status: "ok".into(),
                        image_width: 100,
                        image_height: 200,
                        text_blocks: Vec::new(),
                        bubble_regions: Vec::new(),
                        timings_ms: BTreeMap::new(),
                        warnings: Vec::new(),
                    }),
                    error: None,
                },
                VisionBatchItemResponse {
                    index: 1,
                    status: "error".into(),
                    response: None,
                    error: Some("failed to open image".into()),
                },
            ],
            timings_ms,
            warnings: Vec::new(),
        };

        let encoded = serde_json::to_string(&response).expect("batch response serializes");
        assert!(encoded.contains("\"responses\""));
        assert!(encoded.contains("\"imageWidth\""));
        let decoded: VisionBatchResponse =
            serde_json::from_str(&encoded).expect("batch response deserializes");
        assert_eq!(decoded.responses[0].index, 0);
        assert_eq!(decoded.responses[1].index, 1);
        assert_eq!(decoded.responses[1].status, "error");
        assert_eq!(
            decoded.responses[1].error.as_deref(),
            Some("failed to open image")
        );
    }

    #[test]
    fn normalize_bubble_region_offsets_coordinates() {
        let region = normalize_bubble_region([10.2, 18.6, 40.4, 52.1], 0.91, 100, 200);
        assert_eq!(region.bbox, [110, 218, 141, 253]);
        assert!((region.confidence - 0.91).abs() < f32::EPSILON);
    }
}
