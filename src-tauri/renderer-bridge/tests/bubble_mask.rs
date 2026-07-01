use image::{GrayImage, Luma};
use traduzai_renderer_bridge::{render_to_rgba, resolve_bubble_safe_bbox_for_debug, RenderRequest};

#[test]
fn bubble_mask_limits_alpha_to_selected_bubble_id() {
    let temp = tempfile::tempdir().unwrap();
    let mask_path = temp.path().join("bubble-mask.png");
    let mut mask = GrayImage::new(180, 120);
    for y in 36..78 {
        for x in 70..132 {
            mask.put_pixel(x, y, Luma([1]));
        }
    }
    mask.save(&mask_path).unwrap();

    let request: RenderRequest = serde_json::from_value(serde_json::json!({
        "image_width": 180,
        "image_height": 120,
        "bubble_mask_path": mask_path,
        "blocks": [{
            "id": "masked",
            "text": "FUNCIONA",
            "box": [20, 20, 160, 100],
            "bubble_id": 1,
            "layout_lines": [{"text": "FUNCIONA", "x": 24.0, "y": 54.0}],
            "style": {
                "font_size": 30.0,
                "color": "#000000",
                "stroke_color": "#ffffff",
                "stroke_width": 4.0
            }
        }]
    }))
    .unwrap();

    let image = render_to_rgba(&request).unwrap();
    assert!(image.pixels().any(|pixel| pixel.0[3] > 0));
    for (x, y, pixel) in image.enumerate_pixels() {
        if pixel.0[3] > 0 {
            assert_eq!(
                mask.get_pixel(x, y)[0],
                1,
                "alpha escaped bubble mask at {x},{y}"
            );
        }
    }
}

#[test]
fn bubble_mask_safe_bbox_ignores_thin_tail() {
    let mut mask = GrayImage::new(220, 140);
    for y in 20..100 {
        for x in 20..120 {
            mask.put_pixel(x, y, Luma([1]));
        }
    }
    for y in 55..65 {
        for x in 120..190 {
            mask.put_pixel(x, y, Luma([1]));
        }
    }

    let safe = resolve_bubble_safe_bbox_for_debug(&mask, 1, [20, 20, 190, 100])
        .expect("expected safe bubble bbox");

    assert!(safe[0] >= 20);
    assert!(safe[1] >= 20);
    assert!(
        safe[2] <= 122,
        "safe bbox should not extend into tail: {safe:?}"
    );
    assert!(safe[3] <= 100);
}
