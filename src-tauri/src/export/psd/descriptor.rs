use crate::export::psd::writer::PsdWriter;

#[derive(Debug, Clone)]
pub struct DescriptorObject {
    pub name: String,
    pub class_id: String,
    pub items: Vec<DescriptorItem>,
}

#[derive(Debug, Clone)]
pub struct DescriptorItem {
    pub key: String,
    pub value: DescriptorValue,
}

#[derive(Debug, Clone)]
pub enum DescriptorValue {
    Text(String),
    Enum { type_id: String, value: String },
    Integer(i32),
    Double(f64),
    UnitPixels(f64),
    Raw(Vec<u8>),
    Object(DescriptorObject),
}

impl DescriptorObject {
    pub fn new(name: impl Into<String>, class_id: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            class_id: class_id.into(),
            items: Vec::new(),
        }
    }

    pub fn with_item(mut self, key: impl Into<String>, value: DescriptorValue) -> Self {
        self.items.push(DescriptorItem {
            key: key.into(),
            value,
        });
        self
    }
}

pub fn write_versioned_descriptor(
    writer: &mut PsdWriter,
    descriptor: &DescriptorObject,
) -> Result<(), String> {
    writer.write_u32(16);
    write_descriptor_object(writer, descriptor)
}

pub fn bounds_descriptor(
    class_id: &str,
    left: f64,
    top: f64,
    right: f64,
    bottom: f64,
) -> DescriptorObject {
    DescriptorObject::new("", class_id)
        .with_item("Left", DescriptorValue::UnitPixels(left))
        .with_item("Top ", DescriptorValue::UnitPixels(top))
        .with_item("Rght", DescriptorValue::UnitPixels(right))
        .with_item("Btom", DescriptorValue::UnitPixels(bottom))
}

fn write_descriptor_object(
    writer: &mut PsdWriter,
    descriptor: &DescriptorObject,
) -> Result<(), String> {
    validate_descriptor_id(&descriptor.class_id)?;
    writer.write_unicode_string_with_padding(&descriptor.name);
    writer.write_ascii_or_class_id(&descriptor.class_id);
    writer.write_u32(descriptor.items.len() as u32);

    for item in &descriptor.items {
        validate_descriptor_key(&item.key)?;
        writer.write_ascii_or_class_id(&item.key);
        write_descriptor_value(writer, &item.value)?;
    }

    Ok(())
}

fn write_descriptor_value(writer: &mut PsdWriter, value: &DescriptorValue) -> Result<(), String> {
    match value {
        DescriptorValue::Text(text) => {
            writer.write_signature("TEXT");
            writer.write_unicode_string_with_padding(text);
        }
        DescriptorValue::Enum { type_id, value } => {
            validate_descriptor_id(type_id)?;
            validate_descriptor_id(value)?;
            writer.write_signature("enum");
            writer.write_ascii_or_class_id(type_id);
            writer.write_ascii_or_class_id(value);
        }
        DescriptorValue::Integer(number) => {
            writer.write_signature("long");
            writer.write_i32(*number);
        }
        DescriptorValue::Double(number) => {
            writer.write_signature("doub");
            writer.write_f64(*number);
        }
        DescriptorValue::UnitPixels(number) => {
            writer.write_signature("UntF");
            writer.write_signature("#Pxl");
            writer.write_f64(*number);
        }
        DescriptorValue::Raw(bytes) => {
            writer.write_signature("tdta");
            writer.write_u32(bytes.len() as u32);
            writer.write_bytes(bytes);
        }
        DescriptorValue::Object(object) => {
            writer.write_signature("Objc");
            write_descriptor_object(writer, object)?;
        }
    }

    Ok(())
}

fn validate_descriptor_id(value: &str) -> Result<(), String> {
    if value.is_empty() {
        return Err("descriptor IDs must not be empty".to_string());
    }

    if !value.is_ascii() {
        return Err(format!("descriptor IDs must be ASCII: {value:?}"));
    }

    Ok(())
}

fn validate_descriptor_key(value: &str) -> Result<(), String> {
    validate_descriptor_id(value)
}
