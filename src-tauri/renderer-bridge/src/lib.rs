use anyhow::{bail, Context, Result};
use fontdue::layout::{CoordinateSystem, Layout, LayoutSettings, TextStyle};
use image::{imageops, GrayImage, Luma, RgbaImage};
use imageproc::distance_transform::{distance_transform, Norm};
use serde::{Deserialize, Serialize};
use skrifa::{
    instance::{Location, Size},
    outline::{DrawSettings, OutlinePen},
    GlyphId, MetadataProvider, OutlineGlyph,
};
use std::path::{Path, PathBuf};
use tiny_skia::{FillRule, Paint, Path as SkiaPath, PathBuilder, Pixmap, Transform};

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct RenderRequest {
    pub image_width: u32,
    pub image_height: u32,
    #[serde(default)]
    pub bubble_mask_path: Option<String>,
    pub blocks: Vec<RenderBlock>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct RenderBlock {
    pub id: String,
    pub text: String,
    #[serde(alias = "box")]
    pub bbox: [i32; 4],
    #[serde(default)]
    pub rotation_deg: f32,
    #[serde(default)]
    pub bubble_id: Option<serde_json::Value>,
    #[serde(default)]
    pub layout_lines: Vec<RenderLine>,
    #[serde(default)]
    pub style: RenderStyle,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct RenderLine {
    pub text: String,
    pub x: f32,
    pub y: f32,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct RenderStyle {
    #[serde(default)]
    pub font_family: Option<String>,
    #[serde(default)]
    pub font_file: Option<String>,
    #[serde(default)]
    pub font_size: Option<f32>,
    #[serde(default = "default_color")]
    pub color: String,
    #[serde(default)]
    pub stroke_color: Option<String>,
    #[serde(default)]
    pub stroke_width: f32,
    #[serde(default)]
    pub bold: bool,
    #[serde(default)]
    pub italic: bool,
    #[serde(default = "default_align")]
    pub align: String,
}

impl Default for RenderStyle {
    fn default() -> Self {
        Self {
            font_family: None,
            font_file: None,
            font_size: Some(24.0),
            color: default_color(),
            stroke_color: None,
            stroke_width: 0.0,
            bold: false,
            italic: false,
            align: default_align(),
        }
    }
}

fn default_color() -> String {
    "#000000".to_string()
}

fn default_align() -> String {
    "center".to_string()
}

pub fn render_to_rgba(request: &RenderRequest) -> Result<RgbaImage> {
    if request.image_width == 0 || request.image_height == 0 {
        bail!(
            "invalid image size {}x{}",
            request.image_width,
            request.image_height
        );
    }

    let mut canvas = RgbaImage::new(request.image_width, request.image_height);
    let bubble_mask = load_bubble_mask(request)?;
    let fonts = FontResolver::new();
    let renderer = GlyphRenderer;
    for block in &request.blocks {
        renderer.render_block(&mut canvas, &fonts, block, bubble_mask.as_ref())?;
    }
    Ok(canvas)
}

pub fn render_to_png(request: &RenderRequest, output_path: &std::path::Path) -> Result<()> {
    let image = render_to_rgba(request)?;
    image
        .save(output_path)
        .with_context(|| format!("failed to save {}", output_path.display()))
}

pub fn resolve_font_source_for_debug(style: &RenderStyle) -> Result<String> {
    let fonts = FontResolver::new();
    let source = fonts.resolve_font_source(style)?;
    fonts.font_source_label(&source)
}

pub fn renderer_rasterizer_for_debug() -> &'static str {
    "koharu_outline_supersampled"
}

struct FontResolver {
    db: fontdb::Database,
}

enum ResolvedFontSource {
    File { path: PathBuf, index: u32 },
    Database { id: fontdb::ID },
}

struct ResolvedFont {
    fontdue: fontdue::Font,
    data: Vec<u8>,
    index: u32,
}

impl FontResolver {
    fn new() -> Self {
        let mut db = fontdb::Database::new();
        db.load_system_fonts();
        Self { db }
    }

    fn font_for(&self, style: &RenderStyle) -> Result<ResolvedFont> {
        let source = self.resolve_font_source(style)?;
        self.load_resolved_font(&source)
    }

    fn resolve_font_source(&self, style: &RenderStyle) -> Result<ResolvedFontSource> {
        if let Some(font_file) = style.font_file.as_deref().and_then(|value| {
            let trimmed = value.trim();
            (!trimmed.is_empty()).then_some(trimmed)
        }) {
            let path = Path::new(font_file);
            if path.exists() {
                return Ok(ResolvedFontSource::File {
                    path: path.to_path_buf(),
                    index: 0,
                });
            }
            if is_bare_font_file_name(path) {
                if let Some(path) = find_project_font_file(font_file) {
                    return Ok(ResolvedFontSource::File { path, index: 0 });
                }
            }
            bail!("explicit font_file was not found: {font_file}");
        }

        let mut candidates: Vec<String> = Vec::new();
        if let Some(family) = style.font_family.as_deref().and_then(|value| {
            let trimmed = value.trim();
            (!trimmed.is_empty()).then_some(trimmed)
        }) {
            if let Some(path) = find_project_font_file(family) {
                return Ok(ResolvedFontSource::File { path, index: 0 });
            }
            candidates.push(family.to_string());
        }
        candidates.extend(
            [
                "Arial",
                "Segoe UI",
                "DejaVu Sans",
                "Liberation Sans",
                "Noto Sans",
            ]
            .iter()
            .map(|value| value.to_string()),
        );

        for family in candidates {
            let query = fontdb::Query {
                families: &[fontdb::Family::Name(&family)],
                ..Default::default()
            };
            if let Some(id) = self.db.query(&query) {
                return Ok(ResolvedFontSource::Database { id });
            }
        }

        for face in self.db.faces() {
            return Ok(ResolvedFontSource::Database { id: face.id });
        }

        bail!("no system font available")
    }

    fn load_resolved_font(&self, source: &ResolvedFontSource) -> Result<ResolvedFont> {
        match source {
            ResolvedFontSource::File { path, index } => Self::load_file(path, *index),
            ResolvedFontSource::Database { id } => self
                .load(*id)?
                .ok_or_else(|| anyhow::anyhow!("fontdb face source is unavailable")),
        }
    }

    fn font_source_label(&self, source: &ResolvedFontSource) -> Result<String> {
        match source {
            ResolvedFontSource::File { path, .. } => Ok(path.display().to_string()),
            ResolvedFontSource::Database { id } => {
                let Some((source, _)) = self.db.face_source(*id) else {
                    bail!("fontdb face source is unavailable")
                };
                match source {
                    fontdb::Source::File(path) | fontdb::Source::SharedFile(path, _) => {
                        Ok(path.display().to_string())
                    }
                    fontdb::Source::Binary(_) => Ok("<fontdb-binary>".to_string()),
                }
            }
        }
    }

    fn load_file(path: &Path, index: u32) -> Result<ResolvedFont> {
        let bytes = std::fs::read(path)
            .with_context(|| format!("failed to read font {}", path.display()))?;
        let fontdue = fontdue::Font::from_bytes(
            bytes.clone(),
            fontdue::FontSettings {
                collection_index: index,
                ..fontdue::FontSettings::default()
            },
        )
        .map_err(|err| anyhow::anyhow!("failed to load font {}: {err}", path.display()))?;
        Ok(ResolvedFont {
            fontdue,
            data: bytes,
            index,
        })
    }

    fn load(&self, id: fontdb::ID) -> Result<Option<ResolvedFont>> {
        let Some((source, index)) = self.db.face_source(id) else {
            return Ok(None);
        };
        let bytes = match source {
            fontdb::Source::File(path) => std::fs::read(&path)
                .with_context(|| format!("failed to read font {}", path.display()))?,
            fontdb::Source::Binary(data) => data.as_ref().as_ref().to_vec(),
            fontdb::Source::SharedFile(path, _) => std::fs::read(&path)
                .with_context(|| format!("failed to read font {}", path.display()))?,
        };
        let fontdue = fontdue::Font::from_bytes(
            bytes.clone(),
            fontdue::FontSettings {
                collection_index: index,
                ..fontdue::FontSettings::default()
            },
        )
        .map_err(|err| anyhow::anyhow!("failed to load font face: {err}"))?;
        Ok(Some(ResolvedFont {
            fontdue,
            data: bytes,
            index,
        }))
    }
}

fn is_bare_font_file_name(path: &Path) -> bool {
    path.file_name().is_some()
        && path
            .parent()
            .map(|parent| parent.as_os_str().is_empty())
            .unwrap_or(true)
}

fn project_font_dirs() -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    if let Some(configured) = std::env::var_os("TRADUZAI_FONTS_DIR") {
        dirs.push(PathBuf::from(configured));
    }
    dirs.push(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("fonts"),
    );
    if let Ok(exe) = std::env::current_exe() {
        for ancestor in exe.ancestors().take(6) {
            dirs.push(ancestor.join("fonts"));
            dirs.push(ancestor.join("..").join("fonts"));
        }
    }
    dirs
}

fn find_project_font_file(value: &str) -> Option<PathBuf> {
    let requested = Path::new(value);
    let requested_name = requested.file_name()?.to_string_lossy().to_lowercase();
    let requested_stem = requested
        .file_stem()
        .map(|stem| stem.to_string_lossy().to_lowercase());
    let file_like = matches!(
        requested
            .extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| ext.to_ascii_lowercase())
            .as_deref(),
        Some("ttf" | "otf" | "ttc")
    );
    for dir in project_font_dirs() {
        if !dir.exists() {
            continue;
        }
        let direct = dir.join(value);
        if direct.is_file() {
            return Some(direct);
        }
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if !path.is_file() {
                continue;
            }
            let name = path.file_name()?.to_string_lossy().to_lowercase();
            if name == requested_name {
                return Some(path);
            }
            if file_like {
                let stem = path.file_stem()?.to_string_lossy().to_lowercase();
                if requested_stem.as_deref() == Some(stem.as_str()) {
                    return Some(path);
                }
            }
        }
    }
    None
}

struct GlyphRenderer;

struct PixelClip<'a> {
    bbox: [i32; 4],
    bubble_mask: Option<&'a GrayImage>,
    bubble_id: Option<u8>,
}

impl PixelClip<'_> {
    fn allows(&self, x: i32, y: i32, canvas_w: i32, canvas_h: i32) -> bool {
        let [clip_x1, clip_y1, clip_x2, clip_y2] = self.bbox;
        if x < 0
            || y < 0
            || x >= canvas_w
            || y >= canvas_h
            || x < clip_x1
            || y < clip_y1
            || x >= clip_x2
            || y >= clip_y2
        {
            return false;
        }
        let Some(mask) = self.bubble_mask else {
            return true;
        };
        if x as u32 >= mask.width() || y as u32 >= mask.height() {
            return false;
        }
        let value = mask.get_pixel(x as u32, y as u32)[0];
        match self.bubble_id {
            Some(id) => value == id,
            None => value > 0,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct LayoutBox {
    x: f32,
    y: f32,
    width: f32,
    height: f32,
}

pub fn resolve_bubble_safe_bbox_for_debug(
    mask: &GrayImage,
    bubble_id: u8,
    seed_bbox: [i32; 4],
) -> Option<[i32; 4]> {
    resolve_bubble_safe_bbox(mask, bubble_id, seed_bbox)
}

fn resolve_bubble_safe_bbox(
    mask: &GrayImage,
    bubble_id: u8,
    _seed_bbox: [i32; 4],
) -> Option<[i32; 4]> {
    let bbox = bubble_bbox_for_id(mask, bubble_id)?;
    let safe = safe_layout_box(mask, bubble_id, bbox);
    Some(layout_box_to_bbox(safe, mask.width(), mask.height()))
}

fn bubble_bbox_for_id(mask: &GrayImage, bubble_id: u8) -> Option<LayoutBox> {
    let mut x0 = mask.width();
    let mut y0 = mask.height();
    let mut x1 = 0u32;
    let mut y1 = 0u32;
    let mut seen = false;
    for (x, y, pixel) in mask.enumerate_pixels() {
        if pixel.0[0] != bubble_id {
            continue;
        }
        seen = true;
        x0 = x0.min(x);
        y0 = y0.min(y);
        x1 = x1.max(x);
        y1 = y1.max(y);
    }
    seen.then_some(LayoutBox {
        x: x0 as f32,
        y: y0 as f32,
        width: (x1 - x0 + 1) as f32,
        height: (y1 - y0 + 1) as f32,
    })
}

fn safe_layout_box(mask: &GrayImage, bubble_id: u8, bbox: LayoutBox) -> LayoutBox {
    let desired_padding = (bbox.width.min(bbox.height) * 0.12).round().max(1.0) as u8;
    for padding in [
        desired_padding,
        desired_padding.saturating_mul(3) / 4,
        desired_padding / 2,
        desired_padding / 4,
        1,
        0,
    ] {
        if let Some(safe) = safe_layout_box_with_padding(mask, bubble_id, bbox, padding) {
            return safe;
        }
    }
    inset_layout_box(bbox)
}

fn safe_layout_box_with_padding(
    mask: &GrayImage,
    bubble_id: u8,
    bbox: LayoutBox,
    padding: u8,
) -> Option<LayoutBox> {
    let x0 = bbox.x.floor().max(0.0) as u32;
    let y0 = bbox.y.floor().max(0.0) as u32;
    let x1 = (bbox.x + bbox.width).ceil().min(mask.width() as f32) as u32;
    let y1 = (bbox.y + bbox.height).ceil().min(mask.height() as f32) as u32;
    let width = x1.checked_sub(x0)?;
    let height = y1.checked_sub(y0)?;
    if width == 0 || height == 0 {
        return None;
    }

    let mut background = GrayImage::from_pixel(width + 2, height + 2, Luma([255u8]));
    for y in 0..height {
        for x in 0..width {
            if mask.get_pixel(x0 + x, y0 + y).0[0] == bubble_id {
                background.put_pixel(x + 1, y + 1, Luma([0u8]));
            }
        }
    }
    let distance = distance_transform(&background, Norm::L2);
    let safe_threshold = padding.max(1);
    let mut count = 0f32;
    let mut sum_x = 0f32;
    let mut sum_y = 0f32;
    let mut max_dist = 0u8;
    let mut max_point = (0u32, 0u32);

    for y in 0..height {
        for x in 0..width {
            let lx = x + 1;
            let ly = y + 1;
            if background.get_pixel(lx, ly).0[0] != 0 {
                continue;
            }
            let dist = distance.get_pixel(lx, ly).0[0];
            if dist > max_dist {
                max_dist = dist;
                max_point = (x, y);
            }
            if dist >= safe_threshold {
                count += 1.0;
                sum_x += x as f32;
                sum_y += y as f32;
            }
        }
    }

    if count == 0.0 {
        return None;
    }
    let mut cx = (sum_x / count)
        .round()
        .clamp(0.0, width.saturating_sub(1) as f32) as u32;
    let mut cy = (sum_y / count)
        .round()
        .clamp(0.0, height.saturating_sub(1) as f32) as u32;
    let centroid_dist = distance.get_pixel(cx + 1, cy + 1).0[0];
    if max_dist > 0 && (centroid_dist as f32) < (max_dist as f32 * 0.70) {
        (cx, cy) = max_point;
    }
    if !is_safe_pixel(&background, &distance, cx, cy, safe_threshold) {
        (cx, cy) = nearest_safe_pixel(&background, &distance, cx, cy, safe_threshold)?;
    }

    let safe = build_safe_map(&background, &distance, width, height, safe_threshold);
    let (left, top, right, bottom) = largest_safe_rectangle(&safe, width, height, (cx, cy))?;
    Some(LayoutBox {
        x: x0 as f32 + left as f32,
        y: y0 as f32 + top as f32,
        width: (right - left) as f32,
        height: (bottom - top) as f32,
    })
}

fn build_safe_map(
    background: &GrayImage,
    distance: &GrayImage,
    width: u32,
    height: u32,
    threshold: u8,
) -> Vec<bool> {
    let mut safe = Vec::with_capacity(width as usize * height as usize);
    for y in 0..height {
        for x in 0..width {
            safe.push(is_safe_pixel(background, distance, x, y, threshold));
        }
    }
    safe
}

fn largest_safe_rectangle(
    safe: &[bool],
    width: u32,
    height: u32,
    anchor: (u32, u32),
) -> Option<(u32, u32, u32, u32)> {
    let width = width as usize;
    if width == 0 || height == 0 || safe.len() != width * height as usize {
        return None;
    }

    let mut heights = vec![0u32; width];
    let mut best: Option<(u64, u64, u32, u32, u32, u32)> = None;
    for y in 0..height {
        let row_start = y as usize * width;
        for x in 0..width {
            heights[x] = if safe[row_start + x] {
                heights[x] + 1
            } else {
                0
            };
        }

        let mut stack: Vec<usize> = Vec::with_capacity(width);
        for x in 0..=width {
            let current = if x == width { 0 } else { heights[x] };
            while let Some(&last) = stack.last() {
                if heights[last] <= current {
                    break;
                }
                let bar = stack.pop().expect("stack is non-empty");
                let rect_height = heights[bar];
                if rect_height == 0 {
                    continue;
                }
                let left = stack.last().map_or(0, |&prev| prev + 1);
                let right = x;
                let rect_width = right - left;
                if rect_width == 0 {
                    continue;
                }
                let bottom = y + 1;
                let top = bottom - rect_height;
                let left = left as u32;
                let right = right as u32;
                let area = rect_width as u64 * rect_height as u64;
                let anchor_dist2 = rectangle_anchor_distance2(left, top, right, bottom, anchor);
                let replace = match best {
                    None => true,
                    Some((best_area, best_dist2, _, _, _, _)) => {
                        area > best_area || (area == best_area && anchor_dist2 < best_dist2)
                    }
                };
                if replace {
                    best = Some((area, anchor_dist2, left, top, right, bottom));
                }
            }
            if x < width {
                stack.push(x);
            }
        }
    }
    best.map(|(_, _, left, top, right, bottom)| (left, top, right, bottom))
}

fn rectangle_anchor_distance2(
    left: u32,
    top: u32,
    right: u32,
    bottom: u32,
    anchor: (u32, u32),
) -> u64 {
    let rect_cx2 = left as i64 + right as i64;
    let rect_cy2 = top as i64 + bottom as i64;
    let anchor_cx2 = anchor.0 as i64 * 2 + 1;
    let anchor_cy2 = anchor.1 as i64 * 2 + 1;
    let dx = rect_cx2 - anchor_cx2;
    let dy = rect_cy2 - anchor_cy2;
    (dx * dx + dy * dy) as u64
}

fn is_safe_pixel(
    background: &GrayImage,
    distance: &GrayImage,
    x: u32,
    y: u32,
    threshold: u8,
) -> bool {
    let lx = x + 1;
    let ly = y + 1;
    background.get_pixel(lx, ly).0[0] == 0 && distance.get_pixel(lx, ly).0[0] >= threshold
}

fn nearest_safe_pixel(
    background: &GrayImage,
    distance: &GrayImage,
    cx: u32,
    cy: u32,
    threshold: u8,
) -> Option<(u32, u32)> {
    let width = background.width().checked_sub(2)?;
    let height = background.height().checked_sub(2)?;
    let mut best: Option<(u64, u32, u32)> = None;
    for y in 0..height {
        for x in 0..width {
            if !is_safe_pixel(background, distance, x, y, threshold) {
                continue;
            }
            let dx = x.abs_diff(cx) as u64;
            let dy = y.abs_diff(cy) as u64;
            let dist2 = dx * dx + dy * dy;
            let replace = match best {
                None => true,
                Some((best_dist2, _, _)) => dist2 < best_dist2,
            };
            if replace {
                best = Some((dist2, x, y));
            }
        }
    }
    best.map(|(_, x, y)| (x, y))
}

fn inset_layout_box(bbox: LayoutBox) -> LayoutBox {
    let inset_x = bbox.width * 0.12;
    let inset_y = bbox.height * 0.12;
    LayoutBox {
        x: bbox.x + inset_x,
        y: bbox.y + inset_y,
        width: (bbox.width - 2.0 * inset_x).max(1.0),
        height: (bbox.height - 2.0 * inset_y).max(1.0),
    }
}

fn layout_box_to_bbox(bbox: LayoutBox, width: u32, height: u32) -> [i32; 4] {
    let x1 = bbox.x.floor().max(0.0).min(width as f32) as i32;
    let y1 = bbox.y.floor().max(0.0).min(height as f32) as i32;
    let x2 = (bbox.x + bbox.width)
        .ceil()
        .max(x1 as f32 + 1.0)
        .min(width as f32) as i32;
    let y2 = (bbox.y + bbox.height)
        .ceil()
        .max(y1 as f32 + 1.0)
        .min(height as f32) as i32;
    [x1, y1, x2, y2]
}

impl GlyphRenderer {
    fn render_block(
        &self,
        canvas: &mut RgbaImage,
        fonts: &FontResolver,
        block: &RenderBlock,
        bubble_mask: Option<&GrayImage>,
    ) -> Result<()> {
        let text = block.text.trim();
        if text.is_empty() && block.layout_lines.is_empty() {
            return Ok(());
        }

        let selected_bubble_id = bubble_id_u8(block.bubble_id.as_ref());
        let render_bbox = selected_bubble_id
            .and_then(|bubble_id| {
                bubble_mask.and_then(|mask| resolve_bubble_safe_bbox(mask, bubble_id, block.bbox))
            })
            .unwrap_or(block.bbox);
        let [x1, y1, x2, y2] = render_bbox;
        if x2 <= x1 || y2 <= y1 {
            bail!("invalid block bbox for {}", block.id);
        }
        let box_w = (x2 - x1).max(1) as f32;
        let box_h = (y2 - y1).max(1) as f32;
        let font_size = block
            .style
            .font_size
            .unwrap_or(24.0)
            .clamp(6.0, box_h.max(6.0));
        let font = fonts.font_for(&block.style)?;
        let color = parse_hex_rgba(&block.style.color, [0, 0, 0, 255]);
        let stroke_color = block
            .style
            .stroke_color
            .as_deref()
            .map(|value| parse_hex_rgba(value, [255, 255, 255, 255]));
        let clip = PixelClip {
            bbox: render_bbox,
            bubble_mask,
            bubble_id: selected_bubble_id,
        };

        if !block.layout_lines.is_empty() {
            for line in &block.layout_lines {
                if line.text.trim().is_empty() {
                    continue;
                }
                self.render_line_at(
                    canvas,
                    &font,
                    line.text.trim(),
                    font_size,
                    line.x,
                    line.y,
                    color,
                    stroke_color,
                    &block.style,
                    &clip,
                );
            }
            return Ok(());
        }

        let mut layout = Layout::new(CoordinateSystem::PositiveYDown);
        layout.reset(&LayoutSettings {
            x: 0.0,
            y: 0.0,
            max_width: Some(box_w),
            max_height: Some(box_h),
            ..LayoutSettings::default()
        });
        layout.append(&[&font.fontdue], &TextStyle::new(text, font_size, 0));

        let bounds =
            layout
                .glyphs()
                .iter()
                .fold(None, |acc: Option<(f32, f32, f32, f32)>, glyph| {
                    let min_x = glyph.x;
                    let min_y = glyph.y;
                    let max_x = glyph.x + glyph.width as f32;
                    let max_y = glyph.y + glyph.height as f32;
                    Some(match acc {
                        None => (min_x, min_y, max_x, max_y),
                        Some((ax1, ay1, ax2, ay2)) => (
                            ax1.min(min_x),
                            ay1.min(min_y),
                            ax2.max(max_x),
                            ay2.max(max_y),
                        ),
                    })
                });
        let (text_w, text_h) = bounds
            .map(|(bx1, by1, bx2, by2)| ((bx2 - bx1).max(0.0), (by2 - by1).max(0.0)))
            .unwrap_or((0.0, 0.0));
        let offset_x = match block.style.align.as_str() {
            "left" => 0.0,
            "right" => (box_w - text_w).max(0.0),
            _ => ((box_w - text_w) * 0.5).max(0.0),
        };
        let offset_y = ((box_h - text_h) * 0.5).max(0.0);

        for glyph in layout.glyphs() {
            let (metrics, bitmap) = rasterize_glyph_alpha(&font, glyph.key.glyph_index, font_size);
            if metrics.width == 0 || metrics.height == 0 {
                continue;
            }
            let dst_x = x1 + (glyph.x + offset_x).round() as i32 + metrics.xmin;
            let dst_y = y1 + (glyph.y + offset_y).round() as i32 - metrics.ymin;
            blend_styled_bitmap(
                canvas,
                dst_x,
                dst_y,
                metrics.width,
                metrics.height,
                &bitmap,
                color,
                stroke_color,
                &block.style,
                &clip,
            );
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn render_line_at(
        &self,
        canvas: &mut RgbaImage,
        font: &ResolvedFont,
        text: &str,
        font_size: f32,
        x: f32,
        y: f32,
        color: [u8; 4],
        stroke_color: Option<[u8; 4]>,
        style: &RenderStyle,
        clip: &PixelClip<'_>,
    ) {
        let mut layout = Layout::new(CoordinateSystem::PositiveYDown);
        layout.reset(&LayoutSettings {
            x: 0.0,
            y: 0.0,
            max_width: None,
            max_height: None,
            ..LayoutSettings::default()
        });
        layout.append(&[&font.fontdue], &TextStyle::new(text, font_size, 0));

        for glyph in layout.glyphs() {
            let (metrics, bitmap) = rasterize_glyph_alpha(font, glyph.key.glyph_index, font_size);
            if metrics.width == 0 || metrics.height == 0 {
                continue;
            }
            let dst_x = x.round() as i32 + glyph.x.round() as i32 + metrics.xmin;
            let dst_y = y.round() as i32 + glyph.y.round() as i32 - metrics.ymin;
            blend_styled_bitmap(
                canvas,
                dst_x,
                dst_y,
                metrics.width,
                metrics.height,
                &bitmap,
                color,
                stroke_color,
                style,
                clip,
            );
        }
    }
}

fn rasterize_glyph_alpha(
    font: &ResolvedFont,
    glyph_index: u16,
    font_size: f32,
) -> (fontdue::Metrics, Vec<u8>) {
    let (metrics, fallback) = font.fontdue.rasterize_indexed(glyph_index, font_size);
    if metrics.width == 0 || metrics.height == 0 {
        return (metrics, fallback);
    }
    if let Some(alpha) = render_outline_alpha(font, glyph_index, font_size, &metrics) {
        return (metrics, alpha);
    }
    (metrics, fallback)
}

fn render_outline_alpha(
    font: &ResolvedFont,
    glyph_index: u16,
    font_size: f32,
    metrics: &fontdue::Metrics,
) -> Option<Vec<u8>> {
    const RASTER_SCALE: u32 = 2;

    let width = u32::try_from(metrics.width).ok()?;
    let height = u32::try_from(metrics.height).ok()?;
    if width == 0 || height == 0 {
        return None;
    }
    let raster_width = width.checked_mul(RASTER_SCALE)?;
    let raster_height = height.checked_mul(RASTER_SCALE)?;
    let raster_scale = RASTER_SCALE as f32;

    let font_ref = skrifa::FontRef::from_index(font.data.as_slice(), font.index).ok()?;
    let outline = font_ref
        .outline_glyphs()
        .get(GlyphId::new(glyph_index as u32))?;
    let path = outline_to_path(&outline, font_size * raster_scale)?;

    let mut surface = Pixmap::new(raster_width, raster_height)?;
    let paint = paint_from_rgba([255, 255, 255, 255]);
    let baseline_x = -(metrics.xmin as f32) * raster_scale;
    let baseline_y = (metrics.height as f32 + metrics.ymin as f32) * raster_scale;
    surface.fill_path(
        &path,
        &paint,
        FillRule::Winding,
        Transform::from_translate(baseline_x, baseline_y),
        None,
    );

    let high = RgbaImage::from_raw(raster_width, raster_height, surface.data().to_vec())?;
    let downsampled = imageops::resize(&high, width, height, imageops::FilterType::Lanczos3);
    let alpha = downsampled
        .pixels()
        .map(|pixel| pixel.0[3])
        .collect::<Vec<_>>();
    alpha.iter().any(|&value| value > 0).then_some(alpha)
}

fn parse_hex_rgba(value: &str, default: [u8; 4]) -> [u8; 4] {
    let hex = value.trim().trim_start_matches('#');
    if !(hex.len() == 6 || hex.len() == 8) {
        return default;
    }
    let parse = |range: std::ops::Range<usize>| u8::from_str_radix(&hex[range], 16).ok();
    match (parse(0..2), parse(2..4), parse(4..6)) {
        (Some(r), Some(g), Some(b)) => {
            let a = if hex.len() == 8 {
                parse(6..8).unwrap_or(255)
            } else {
                255
            };
            [r, g, b, a]
        }
        _ => default,
    }
}

fn paint_from_rgba(color: [u8; 4]) -> Paint<'static> {
    let mut paint = Paint {
        anti_alias: true,
        ..Default::default()
    };
    paint.set_color_rgba8(color[0], color[1], color[2], color[3]);
    paint
}

fn outline_to_path(outline: &OutlineGlyph<'_>, font_size: f32) -> Option<SkiaPath> {
    let mut pen = TinySkiaPathPen::new();
    let location = Location::default();
    let settings = DrawSettings::unhinted(Size::new(font_size), &location);
    outline.draw(settings, &mut pen).ok()?;
    pen.finish()
}

struct TinySkiaPathPen {
    builder: PathBuilder,
}

impl TinySkiaPathPen {
    fn new() -> Self {
        Self {
            builder: PathBuilder::new(),
        }
    }

    fn finish(self) -> Option<SkiaPath> {
        self.builder.finish()
    }
}

impl OutlinePen for TinySkiaPathPen {
    fn move_to(&mut self, x: f32, y: f32) {
        self.builder.move_to(x, -y);
    }

    fn line_to(&mut self, x: f32, y: f32) {
        self.builder.line_to(x, -y);
    }

    fn quad_to(&mut self, cx0: f32, cy0: f32, x: f32, y: f32) {
        self.builder.quad_to(cx0, -cy0, x, -y);
    }

    fn curve_to(&mut self, cx0: f32, cy0: f32, cx1: f32, cy1: f32, x: f32, y: f32) {
        self.builder.cubic_to(cx0, -cy0, cx1, -cy1, x, -y);
    }

    fn close(&mut self) {
        self.builder.close();
    }
}

fn blend_alpha_bitmap(
    canvas: &mut RgbaImage,
    x: i32,
    y: i32,
    width: usize,
    height: usize,
    alpha: &[u8],
    color: [u8; 4],
    clip: &PixelClip<'_>,
) {
    let canvas_w = canvas.width() as i32;
    let canvas_h = canvas.height() as i32;
    for row in 0..height {
        for col in 0..width {
            let cx = x + col as i32;
            let cy = y + row as i32;
            if !clip.allows(cx, cy, canvas_w, canvas_h) {
                continue;
            }
            let mask = alpha[row * width + col] as u16;
            if mask == 0 {
                continue;
            }
            let src_a = ((mask * color[3] as u16) / 255) as u8;
            let pixel = canvas.get_pixel_mut(cx as u32, cy as u32);
            let dst_a = pixel.0[3];
            let out_a = src_a.saturating_add(((dst_a as u16 * (255 - src_a) as u16) / 255) as u8);
            if out_a == 0 {
                continue;
            }
            for channel in 0..3 {
                let src = color[channel] as u32 * src_a as u32;
                let dst = pixel.0[channel] as u32 * dst_a as u32 * (255 - src_a) as u32 / 255;
                pixel.0[channel] = ((src + dst) / out_a as u32).min(255) as u8;
            }
            pixel.0[3] = out_a;
        }
    }
}

fn blend_styled_bitmap(
    canvas: &mut RgbaImage,
    x: i32,
    y: i32,
    width: usize,
    height: usize,
    alpha: &[u8],
    color: [u8; 4],
    stroke_color: Option<[u8; 4]>,
    style: &RenderStyle,
    clip: &PixelClip<'_>,
) {
    let stroke_radius = style.stroke_width.ceil().max(0.0) as usize;
    if stroke_radius > 0 {
        if let Some(outline) = stroke_color {
            let radius = stroke_radius.min(16);
            let dilated = dilate_alpha(alpha, width, height, radius);
            blend_alpha_bitmap(
                canvas,
                x - radius as i32,
                y - radius as i32,
                width + (radius * 2),
                height + (radius * 2),
                &dilated,
                outline,
                clip,
            );
        }
    }

    if style.bold {
        let radius = 1usize;
        let dilated = dilate_alpha(alpha, width, height, radius);
        blend_alpha_bitmap(
            canvas,
            x - radius as i32,
            y - radius as i32,
            width + (radius * 2),
            height + (radius * 2),
            &dilated,
            color,
            clip,
        );
    } else {
        blend_alpha_bitmap(canvas, x, y, width, height, alpha, color, clip);
    }
}

fn load_bubble_mask(request: &RenderRequest) -> Result<Option<GrayImage>> {
    let Some(raw_path) = request
        .bubble_mask_path
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
    else {
        return Ok(None);
    };
    let mask = image::open(raw_path)
        .with_context(|| format!("failed to open bubble mask {raw_path}"))?
        .to_luma8();
    if mask.width() != request.image_width || mask.height() != request.image_height {
        bail!(
            "bubble mask size {}x{} does not match request {}x{}",
            mask.width(),
            mask.height(),
            request.image_width,
            request.image_height
        );
    }
    Ok(Some(mask))
}

fn bubble_id_u8(value: Option<&serde_json::Value>) -> Option<u8> {
    match value {
        Some(serde_json::Value::Number(number)) => {
            number.as_u64().and_then(|raw| u8::try_from(raw).ok())
        }
        Some(serde_json::Value::String(text)) => text.parse::<u8>().ok(),
        _ => None,
    }
}

fn dilate_alpha(alpha: &[u8], width: usize, height: usize, radius: usize) -> Vec<u8> {
    if radius == 0 || width == 0 || height == 0 {
        return alpha.to_vec();
    }
    let out_w = width + radius * 2;
    let out_h = height + radius * 2;
    let mut output = vec![0u8; out_w * out_h];
    let radius_i = radius as isize;
    let radius_sq = radius_i * radius_i;
    for row in 0..height {
        for col in 0..width {
            let value = alpha[row * width + col];
            if value == 0 {
                continue;
            }
            let center_x = col as isize + radius_i;
            let center_y = row as isize + radius_i;
            for dy in -radius_i..=radius_i {
                for dx in -radius_i..=radius_i {
                    if dx * dx + dy * dy > radius_sq {
                        continue;
                    }
                    let out_x = center_x + dx;
                    let out_y = center_y + dy;
                    if out_x < 0 || out_y < 0 || out_x >= out_w as isize || out_y >= out_h as isize
                    {
                        continue;
                    }
                    let idx = out_y as usize * out_w + out_x as usize;
                    output[idx] = output[idx].max(value);
                }
            }
        }
    }
    output
}
