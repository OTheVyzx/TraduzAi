pub mod descriptor;
pub mod engine_data;
pub mod packbits;
pub mod writer;

use self::descriptor::{
    bounds_descriptor, write_versioned_descriptor, DescriptorObject, DescriptorValue,
};
use self::engine_data::{encode_engine_data, TextEngineSpec, TextOrientation};
use self::packbits::{encode_image_rle, ChannelId};
use self::writer::PsdWriter;
use image::{imageops::overlay, Rgba, RgbaImage};

#[derive(Debug, Clone)]
pub struct PsdLayer {
    pub name: String,
    pub x: i32,
    pub y: i32,
    pub pixels: RgbaImage,
    pub hidden: bool,
    pub text_spec: Option<TextEngineSpec>,
}

pub fn export_psd(width: u32, height: u32, layers: &[PsdLayer]) -> Result<Vec<u8>, String> {
    let mut psd = PsdWriter::new();
    write_header(&mut psd, width, height);

    // Color Mode Data
    psd.write_u32(0);
    // Image Resources
    psd.write_u32(0);

    // Layer and Mask Information
    let layer_mask_info = build_layer_and_mask_info(layers)?;
    psd.write_u32(layer_mask_info.len() as u32);
    psd.write_bytes(&layer_mask_info);

    // Image Data (Merged composite)
    let composite = merged_composite(width, height, layers);
    write_image_data(&mut psd, &composite)?;

    Ok(psd.into_inner())
}

fn write_header(writer: &mut PsdWriter, width: u32, height: u32) {
    writer.write_signature("8BPS");
    writer.write_u16(1); // Version
    writer.write_zeroes(6); // Reserved
    writer.write_u16(4); // Channels (RGBA)
    writer.write_u32(height);
    writer.write_u32(width);
    writer.write_u16(8); // Depth
    writer.write_u16(3); // Color Mode (RGB)
}

fn merged_composite(width: u32, height: u32, layers: &[PsdLayer]) -> RgbaImage {
    let mut canvas = RgbaImage::from_pixel(width, height, Rgba([255, 255, 255, 255]));
    for layer in layers.iter().filter(|l| !l.hidden) {
        overlay(&mut canvas, &layer.pixels, layer.x as i64, layer.y as i64);
    }
    canvas
}

fn build_layer_and_mask_info(layers: &[PsdLayer]) -> Result<Vec<u8>, String> {
    let mut layer_info = PsdWriter::new();

    // Layer count (negative for absolute alpha)
    if layers.is_empty() {
        layer_info.write_i16(0);
    } else {
        layer_info.write_i16(-(layers.len() as i16));
    }

    let mut encoded_channels = Vec::new();
    let mut extra_data_blocks = Vec::new();

    // Iterate TOP TO BOTTOM for the PSD layer record list
    // (PSD stores layers in the record collection from top to bottom)
    for layer in layers.iter().rev() {
        let channels = encode_image_rle(
            &layer.pixels,
            &[
                ChannelId::Red,
                ChannelId::Green,
                ChannelId::Blue,
                ChannelId::Alpha,
            ],
            &layer.name,
        )?;

        let mut extra = PsdWriter::new();
        extra.write_u32(0); // Mask data len
        extra.write_u32(0); // Blending ranges len
        extra.write_pascal_string(&layer.name, 4);

        if let Some(spec) = &layer.text_spec {
            // UNIcode layer name
            let mut luni = PsdWriter::new();
            luni.write_unicode_string(&layer.name);
            write_additional_info_block(&mut extra, "luni", &luni.into_inner(), 4);

            // Text Tool Info (Editable text)
            let tysh = tysh_body(spec, layer.x, layer.y)?;
            write_additional_info_block(&mut extra, "TySh", &tysh, 2);
        }

        encoded_channels.push(channels);
        extra_data_blocks.push(extra.into_inner());
    }

    // Write Layer Records
    for (idx, layer) in layers.iter().rev().enumerate() {
        let channels = &encoded_channels[idx];
        let extra = &extra_data_blocks[idx];

        let right = layer.x + layer.pixels.width() as i32;
        let bottom = layer.y + layer.pixels.height() as i32;

        layer_info.write_i32(layer.y); // top
        layer_info.write_i32(layer.x); // left
        layer_info.write_i32(bottom);
        layer_info.write_i32(right);
        layer_info.write_u16(channels.len() as u16);

        for channel in channels {
            layer_info.write_i16(channel.channel_id);
            layer_info.write_u32((2 + channel.data.len()) as u32);
        }

        layer_info.write_signature("8BIM");
        layer_info.write_signature("norm"); // Blend mode
        layer_info.write_u8(255); // Opacity
        layer_info.write_u8(0); // Clipping
        layer_info.write_u8(if layer.hidden { 0x0A } else { 0x08 }); // Flags (visible/hidden)
        layer_info.write_u8(0); // Filler

        layer_info.write_u32(extra.len() as u32);
        layer_info.write_bytes(extra);
    }

    // Write Channel Image Data
    for channels in &encoded_channels {
        for channel in channels {
            layer_info.write_u16(1); // Compression: PackBits RLE
            layer_info.write_bytes(&channel.data);
        }
    }

    layer_info.pad_to_multiple(4);

    let mut full = PsdWriter::new();
    full.write_u32(layer_info.len() as u32);
    full.write_bytes(&layer_info.into_inner());
    full.write_u32(0); // Global mask info

    Ok(full.into_inner())
}

fn write_additional_info_block(writer: &mut PsdWriter, key: &str, body: &[u8], alignment: usize) {
    let padding = (alignment - (body.len() % alignment)) % alignment;
    writer.write_signature("8BIM");
    writer.write_signature(key);
    writer.write_u32((body.len() + padding) as u32);
    writer.write_bytes(body);
    writer.write_zeroes(padding);
}

fn tysh_body(spec: &TextEngineSpec, x: i32, y: i32) -> Result<Vec<u8>, String> {
    let engine_data = encode_engine_data(spec);

    let left = x as f64;
    let top = y as f64;
    let right = (x as f64) + spec.box_width;
    let bottom = (y as f64) + spec.box_height;

    let bounds = bounds_descriptor("bounds", left, top, right, bottom);
    let bounding_box = bounds_descriptor("boundingBox", left, top, right, bottom);

    let text_descriptor = DescriptorObject::new("", "TxLr")
        .with_item("Txt ", DescriptorValue::Text(spec.text.clone()))
        .with_item(
            "textGridding",
            DescriptorValue::Enum {
                type_id: "textGridding".to_string(),
                value: "None".to_string(),
            },
        )
        .with_item(
            "Ornt",
            DescriptorValue::Enum {
                type_id: "Ornt".to_string(),
                value: match spec.orientation {
                    TextOrientation::Horizontal => "Hrzn".to_string(),
                    TextOrientation::Vertical => "Vrtc".to_string(),
                },
            },
        )
        .with_item(
            "AntA",
            DescriptorValue::Enum {
                type_id: "Annt".to_string(),
                value: "antiAliasSharp".to_string(),
            },
        )
        .with_item("bounds", DescriptorValue::Object(bounds))
        .with_item("boundingBox", DescriptorValue::Object(bounding_box))
        .with_item("TextIndex", DescriptorValue::Integer(0))
        .with_item("EngineData", DescriptorValue::Raw(engine_data));

    let warp_descriptor = DescriptorObject::new("", "warp")
        .with_item(
            "warpStyle",
            DescriptorValue::Enum {
                type_id: "warpStyle".to_string(),
                value: "warpNone".to_string(),
            },
        )
        .with_item("warpValue", DescriptorValue::Double(0.0))
        .with_item("warpPerspective", DescriptorValue::Double(0.0))
        .with_item("warpPerspectiveOther", DescriptorValue::Double(0.0))
        .with_item(
            "warpRotate",
            DescriptorValue::Enum {
                type_id: "Ornt".to_string(),
                value: match spec.orientation {
                    TextOrientation::Horizontal => "Hrzn".to_string(),
                    TextOrientation::Vertical => "Vrtc".to_string(),
                },
            },
        )
        .with_item(
            "bounds",
            DescriptorValue::Object(bounds_descriptor("bounds", left, top, right, bottom)),
        );

    let mut body = PsdWriter::new();
    body.write_i16(1); // Version
                       // Transform matrix
    body.write_f64(1.0);
    body.write_f64(0.0);
    body.write_f64(0.0);
    body.write_f64(1.0);
    body.write_f64(left);
    body.write_f64(top);

    body.write_i16(50); // Text version
    write_versioned_descriptor(&mut body, &text_descriptor)?;

    body.write_i16(1); // Warp version
    write_versioned_descriptor(&mut body, &warp_descriptor)?;

    body.write_f32(left as f32);
    body.write_f32(top as f32);
    body.write_f32(right as f32);
    body.write_f32(bottom as f32);

    Ok(body.into_inner())
}

fn write_image_data(writer: &mut PsdWriter, image: &RgbaImage) -> Result<(), String> {
    writer.write_u16(1); // Compression: PackBits RLE
    let channels = encode_image_rle(
        image,
        &[
            ChannelId::Red,
            ChannelId::Green,
            ChannelId::Blue,
            ChannelId::Alpha,
        ],
        "Merged Composite",
    )?;

    let row_lengths_len = image.height() as usize * 2;
    for channel in &channels {
        writer.write_bytes(&channel.data[..row_lengths_len]);
    }
    for channel in &channels {
        writer.write_bytes(&channel.data[row_lengths_len..]);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read_be_f64(bytes: &[u8], offset: usize) -> f64 {
        let chunk: [u8; 8] = bytes[offset..offset + 8]
            .try_into()
            .expect("slice with incorrect length");
        f64::from_be_bytes(chunk)
    }

    #[test]
    fn tysh_body_uses_layer_position_in_transform_matrix() {
        let spec = TextEngineSpec {
            text: "Teste PSD".to_string(),
            font_name: "ArialMT".to_string(),
            font_size: 28.0,
            color: [0, 0, 0, 255],
            faux_bold: false,
            faux_italic: false,
            orientation: TextOrientation::Horizontal,
            justification: crate::export::psd::engine_data::TextJustification::Center,
            box_width: 320.0,
            box_height: 140.0,
        };

        let body = tysh_body(&spec, 185, 9765).expect("TySh body should be encoded");

        assert_eq!(i16::from_be_bytes([body[0], body[1]]), 1);
        assert_eq!(read_be_f64(&body, 2), 1.0);
        assert_eq!(read_be_f64(&body, 10), 0.0);
        assert_eq!(read_be_f64(&body, 18), 0.0);
        assert_eq!(read_be_f64(&body, 26), 1.0);
        assert_eq!(read_be_f64(&body, 34), 185.0);
        assert_eq!(read_be_f64(&body, 42), 9765.0);
    }
}
