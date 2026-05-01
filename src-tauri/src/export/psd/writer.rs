#[derive(Debug, Default, Clone)]
pub struct PsdWriter {
    bytes: Vec<u8>,
}

impl PsdWriter {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.bytes.len()
    }

    pub fn into_inner(self) -> Vec<u8> {
        self.bytes
    }

    pub fn write_bytes(&mut self, bytes: &[u8]) {
        self.bytes.extend_from_slice(bytes);
    }

    pub fn write_zeroes(&mut self, count: usize) {
        self.bytes.resize(self.bytes.len() + count, 0);
    }

    pub fn write_u8(&mut self, value: u8) {
        self.bytes.push(value);
    }

    pub fn write_i16(&mut self, value: i16) {
        self.write_bytes(&value.to_be_bytes());
    }

    pub fn write_u16(&mut self, value: u16) {
        self.write_bytes(&value.to_be_bytes());
    }

    pub fn write_i32(&mut self, value: i32) {
        self.write_bytes(&value.to_be_bytes());
    }

    pub fn write_u32(&mut self, value: u32) {
        self.write_bytes(&value.to_be_bytes());
    }

    pub fn write_f32(&mut self, value: f32) {
        self.write_bytes(&value.to_be_bytes());
    }

    pub fn write_f64(&mut self, value: f64) {
        self.write_bytes(&value.to_be_bytes());
    }

    pub fn write_signature(&mut self, signature: &str) {
        assert_eq!(signature.len(), 4, "PSD signatures must be 4 bytes");
        self.write_bytes(signature.as_bytes());
    }

    pub fn write_ascii_or_class_id(&mut self, value: &str) {
        let treat_as_class_id =
            value.len() == 4 && !matches!(value, "warp" | "time" | "hold" | "list");
        if treat_as_class_id {
            self.write_i32(0);
            self.write_signature(value);
        } else {
            self.write_i32(value.len() as i32);
            self.write_bytes(value.as_bytes());
        }
    }

    pub fn write_pascal_string(&mut self, text: &str, pad_to: usize) {
        let ascii = ascii_legacy(text, 255);
        self.write_u8(ascii.len() as u8);
        self.write_bytes(ascii.as_bytes());

        let mut total = ascii.len() + 1;
        while total % pad_to != 0 {
            self.write_u8(0);
            total += 1;
        }
    }

    pub fn write_unicode_string(&mut self, text: &str) {
        let utf16: Vec<u16> = text.encode_utf16().collect();
        self.write_u32(utf16.len() as u32);
        for unit in utf16 {
            self.write_u16(unit);
        }
    }

    pub fn write_unicode_string_with_padding(&mut self, text: &str) {
        let utf16: Vec<u16> = text.encode_utf16().collect();
        self.write_u32((utf16.len() + 1) as u32);
        for unit in utf16 {
            self.write_u16(unit);
        }
        self.write_u16(0);
    }

    pub fn pad_to_multiple(&mut self, multiple: usize) {
        while self.bytes.len() % multiple != 0 {
            self.write_u8(0);
        }
    }
}

pub fn ascii_legacy(text: &str, max_bytes: usize) -> String {
    let mut out = String::new();
    for ch in text.chars() {
        let mapped = if ch.is_ascii() { ch } else { '?' };
        if out.len() + mapped.len_utf8() > max_bytes {
            break;
        }
        out.push(mapped);
    }
    out
}
