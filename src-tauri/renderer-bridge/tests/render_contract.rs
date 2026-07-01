use std::path::PathBuf;
use traduzai_renderer_bridge::{
    render_to_rgba, renderer_rasterizer_for_debug, resolve_font_source_for_debug, RenderBlock,
    RenderRequest, RenderStyle,
};

fn alpha_bbox(image: &image::RgbaImage) -> Option<(u32, u32, u32, u32)> {
    let mut x1 = image.width();
    let mut y1 = image.height();
    let mut x2 = 0;
    let mut y2 = 0;
    let mut seen = false;
    for (x, y, pixel) in image.enumerate_pixels() {
        if pixel.0[3] == 0 {
            continue;
        }
        seen = true;
        x1 = x1.min(x);
        y1 = y1.min(y);
        x2 = x2.max(x);
        y2 = y2.max(y);
    }
    seen.then_some((x1, y1, x2 + 1, y2 + 1))
}

#[test]
fn renders_simple_text_request() {
    let request = RenderRequest {
        image_width: 320,
        image_height: 240,
        bubble_mask_path: None,
        blocks: vec![RenderBlock {
            id: "a".into(),
            text: "OLA".into(),
            bbox: [40, 40, 220, 120],
            rotation_deg: 0.0,
            bubble_id: None,
            layout_lines: Vec::new(),
            style: RenderStyle::default(),
        }],
    };

    let image = render_to_rgba(&request).unwrap();
    assert_eq!(image.width(), 320);
    assert_eq!(image.height(), 240);
    assert!(image.pixels().any(|p| p.0[3] > 0));
}

#[test]
fn honors_absolute_layout_lines_when_provided() {
    let request: RenderRequest = serde_json::from_value(serde_json::json!({
        "image_width": 320,
        "image_height": 240,
        "blocks": [{
            "id": "positioned",
            "text": "OLA",
            "box": [0, 0, 320, 240],
            "layout_lines": [{"text": "OLA", "x": 230.0, "y": 172.0}],
            "style": {"font_size": 28.0, "color": "#000000"}
        }]
    }))
    .unwrap();

    let image = render_to_rgba(&request).unwrap();
    let bbox = alpha_bbox(&image).unwrap();
    assert!(
        bbox.0 >= 220,
        "expected x near positioned line, got {bbox:?}"
    );
    assert!(
        bbox.1 >= 150,
        "expected y near positioned line, got {bbox:?}"
    );
}

#[test]
fn stroke_width_expands_rendered_alpha_bounds() {
    let base: RenderRequest = serde_json::from_value(serde_json::json!({
        "image_width": 220,
        "image_height": 120,
        "blocks": [{
            "id": "plain",
            "text": "SIM",
            "box": [20, 20, 200, 100],
            "layout_lines": [{"text": "SIM", "x": 70.0, "y": 55.0}],
            "style": {"font_size": 36.0, "color": "#000000"}
        }]
    }))
    .unwrap();
    let stroked: RenderRequest = serde_json::from_value(serde_json::json!({
        "image_width": 220,
        "image_height": 120,
        "blocks": [{
            "id": "stroke",
            "text": "SIM",
            "box": [20, 20, 200, 100],
            "layout_lines": [{"text": "SIM", "x": 70.0, "y": 55.0}],
            "style": {
                "font_size": 36.0,
                "color": "#000000",
                "stroke_color": "#ffffff",
                "stroke_width": 4.0
            }
        }]
    }))
    .unwrap();

    let base_bbox = alpha_bbox(&render_to_rgba(&base).unwrap()).unwrap();
    let stroked_bbox = alpha_bbox(&render_to_rgba(&stroked).unwrap()).unwrap();
    assert!(
        stroked_bbox.0 < base_bbox.0,
        "stroke should expand left: {base_bbox:?} vs {stroked_bbox:?}"
    );
    assert!(
        stroked_bbox.1 < base_bbox.1,
        "stroke should expand up: {base_bbox:?} vs {stroked_bbox:?}"
    );
    assert!(
        stroked_bbox.2 > base_bbox.2,
        "stroke should expand right: {base_bbox:?} vs {stroked_bbox:?}"
    );
    assert!(
        stroked_bbox.3 > base_bbox.3,
        "stroke should expand down: {base_bbox:?} vs {stroked_bbox:?}"
    );
}

#[test]
fn clips_stroke_to_render_box() {
    let request: RenderRequest = serde_json::from_value(serde_json::json!({
        "image_width": 120,
        "image_height": 80,
        "blocks": [{
            "id": "clipped",
            "text": "SIM",
            "box": [44, 24, 86, 58],
            "layout_lines": [{"text": "SIM", "x": 36.0, "y": 34.0}],
            "style": {
                "font_size": 34.0,
                "color": "#000000",
                "stroke_color": "#ffffff",
                "stroke_width": 8.0
            }
        }]
    }))
    .unwrap();

    let image = render_to_rgba(&request).unwrap();
    let bbox = alpha_bbox(&image).unwrap();
    assert!(bbox.0 >= 44, "left alpha escaped clip box: {bbox:?}");
    assert!(bbox.1 >= 24, "top alpha escaped clip box: {bbox:?}");
    assert!(bbox.2 <= 86, "right alpha escaped clip box: {bbox:?}");
    assert!(bbox.3 <= 58, "bottom alpha escaped clip box: {bbox:?}");
}

#[test]
fn loads_project_font_by_file_name_family() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let font_path = manifest_dir
        .join("..")
        .join("..")
        .join("fonts")
        .join("ComicNeue-Bold.ttf");
    assert!(
        font_path.exists(),
        "missing test font {}",
        font_path.display()
    );

    let request: RenderRequest = serde_json::from_value(serde_json::json!({
        "image_width": 240,
        "image_height": 120,
        "blocks": [{
            "id": "project-font",
            "text": "ÁGUA",
            "box": [20, 20, 220, 100],
            "layout_lines": [{"text": "ÁGUA", "x": 54.0, "y": 55.0}],
            "style": {
                "font_family": "ComicNeue-Bold.ttf",
                "font_size": 38.0,
                "color": "#000000"
            }
        }]
    }))
    .unwrap();

    let image = render_to_rgba(&request).unwrap();
    assert!(image.pixels().any(|p| p.0[3] > 0));

    let source = resolve_font_source_for_debug(&RenderStyle {
        font_family: Some("ComicNeue-Bold.ttf".into()),
        ..RenderStyle::default()
    })
    .unwrap();
    assert!(
        source
            .replace('\\', "/")
            .ends_with("fonts/ComicNeue-Bold.ttf"),
        "expected project font source, got {source}"
    );
}

#[test]
fn explicit_font_file_wins_over_family() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let font_path = manifest_dir
        .join("..")
        .join("..")
        .join("fonts")
        .join("ComicNeue-Bold.ttf");
    assert!(
        font_path.exists(),
        "missing test font {}",
        font_path.display()
    );

    let source = resolve_font_source_for_debug(&RenderStyle {
        font_family: Some("Arial".into()),
        font_file: Some(font_path.display().to_string()),
        ..RenderStyle::default()
    })
    .unwrap();

    assert!(
        source
            .replace('\\', "/")
            .ends_with("fonts/ComicNeue-Bold.ttf"),
        "expected explicit font file source, got {source}"
    );
}

#[test]
fn explicit_bare_font_file_resolves_from_project_fonts() {
    let source = resolve_font_source_for_debug(&RenderStyle {
        font_family: Some("Arial".into()),
        font_file: Some("ComicNeue-Bold.ttf".into()),
        ..RenderStyle::default()
    })
    .unwrap();

    assert!(
        source
            .replace('\\', "/")
            .ends_with("fonts/ComicNeue-Bold.ttf"),
        "expected bare font_file to resolve from project fonts, got {source}"
    );
}

#[test]
fn missing_explicit_font_file_is_an_error() {
    let source = resolve_font_source_for_debug(&RenderStyle {
        font_family: Some("Arial".into()),
        font_file: Some("N:/TraduzAI/fonts/does-not-exist.ttf".into()),
        ..RenderStyle::default()
    });

    assert!(
        source.is_err(),
        "missing explicit font_file must not silently fall back to a family"
    );
}

#[test]
fn missing_absolute_font_file_does_not_fallback_to_project_font_with_same_name() {
    let source = resolve_font_source_for_debug(&RenderStyle {
        font_family: Some("Arial".into()),
        font_file: Some("N:/TraduzAI/missing/ComicNeue-Bold.ttf".into()),
        ..RenderStyle::default()
    });

    assert!(
        source.is_err(),
        "missing absolute font_file must not resolve by basename from project fonts"
    );
}

#[test]
fn renderer_reports_koharu_outline_rasterizer() {
    assert_eq!(
        renderer_rasterizer_for_debug(),
        "koharu_outline_supersampled"
    );
}
