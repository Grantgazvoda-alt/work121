use std::collections::BTreeMap;
use std::env;
use std::fmt::Write as _;
use std::fs;
use std::io::{self, Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{self, Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

const VERSION: &str = env!("CARGO_PKG_VERSION");
const DEFAULT_ORACLE_KEY: &[u8] = b"benchmark-oracle-key-v1";
const UTF8_BOM: &[u8] = &[0xEF, 0xBB, 0xBF];

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum FramingMode {
    Raw,
    LengthPrefixed,
    Ndjson,
}

impl FramingMode {
    fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "raw" => Ok(Self::Raw),
            "length_prefixed" => Ok(Self::LengthPrefixed),
            "ndjson" => Ok(Self::Ndjson),
            other => Err(format!("Unsupported framing mode: {other}")),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Raw => "raw",
            Self::LengthPrefixed => "length_prefixed",
            Self::Ndjson => "ndjson",
        }
    }
}

#[derive(Clone, Debug)]
struct GateConfig {
    oracle_key: Vec<u8>,
    framing: FramingMode,
    require_signed_observed: bool,
    max_wire_bytes: usize,
    max_claim_chars: usize,
    max_nesting_depth: usize,
    reject_special_tokens: bool,
}

impl Default for GateConfig {
    fn default() -> Self {
        Self {
            oracle_key: DEFAULT_ORACLE_KEY.to_vec(),
            framing: FramingMode::Raw,
            require_signed_observed: true,
            max_wire_bytes: 32_768,
            max_claim_chars: 8_192,
            max_nesting_depth: 8,
            reject_special_tokens: true,
        }
    }
}

#[derive(Clone, Debug)]
struct JsonStr {
    text: String,
    has_lone_surrogate: bool,
}

#[derive(Clone, Debug)]
enum JsonValue {
    Null,
    Bool(bool),
    Int(i64),
    BigInt(String),
    Float(f64),
    Str(JsonStr),
    Array(Vec<JsonValue>),
    Object(Vec<(JsonStr, JsonValue)>),
}

// ---------------------------------------------------------------------------
// SHA-256 & HMAC-SHA256 (FIPS 180-4 / RFC 2104, zero external dependencies)
// ---------------------------------------------------------------------------

const SHA256_K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

fn sha256_bytes(input: &[u8]) -> [u8; 32] {
    let mut state: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
    ];
    let bit_len = (input.len() as u64).wrapping_mul(8);
    let mut padded = Vec::with_capacity((input.len() + 9).div_ceil(64) * 64);
    padded.extend_from_slice(input);
    padded.push(0x80);
    while (padded.len() % 64) != 56 {
        padded.push(0x00);
    }
    padded.extend_from_slice(&bit_len.to_be_bytes());

    let (blocks, _) = padded.as_chunks::<64>();
    for block in blocks {
        let mut w = [0u32; 64];
        for (i, slot) in w.iter_mut().enumerate().take(16) {
            let j = i * 4;
            *slot = u32::from_be_bytes([block[j], block[j + 1], block[j + 2], block[j + 3]]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = state;
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let temp1 = h
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(SHA256_K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = s0.wrapping_add(maj);

            h = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }
        state[0] = state[0].wrapping_add(a);
        state[1] = state[1].wrapping_add(b);
        state[2] = state[2].wrapping_add(c);
        state[3] = state[3].wrapping_add(d);
        state[4] = state[4].wrapping_add(e);
        state[5] = state[5].wrapping_add(f);
        state[6] = state[6].wrapping_add(g);
        state[7] = state[7].wrapping_add(h);
    }

    let mut out = [0u8; 32];
    for (i, word) in state.iter().enumerate() {
        out[i * 4..i * 4 + 4].copy_from_slice(&word.to_be_bytes());
    }
    out
}

fn to_hex_lower(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        let _ = write!(&mut s, "{b:02x}");
    }
    s
}

fn hmac_sha256_hex(key: &[u8], msg: &[u8]) -> String {
    let mut k_block = [0u8; 64];
    if key.len() > 64 {
        let hk = sha256_bytes(key);
        k_block[..32].copy_from_slice(&hk);
    } else {
        k_block[..key.len()].copy_from_slice(key);
    }
    let mut ipad = [0x36u8; 64];
    let mut opad = [0x5cu8; 64];
    for i in 0..64 {
        ipad[i] ^= k_block[i];
        opad[i] ^= k_block[i];
    }
    let mut inner = Vec::with_capacity(64 + msg.len());
    inner.extend_from_slice(&ipad);
    inner.extend_from_slice(msg);
    let inner_hash = sha256_bytes(&inner);

    let mut outer = Vec::with_capacity(64 + 32);
    outer.extend_from_slice(&opad);
    outer.extend_from_slice(&inner_hash);
    to_hex_lower(&sha256_bytes(&outer))
}

fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for (&x, &y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

fn decode_hex(input: &str) -> Result<Vec<u8>, String> {
    let cleaned: String = input.chars().filter(|c| !c.is_whitespace()).collect();
    if !cleaned.len().is_multiple_of(2) {
        return Err("Hex input must have even length".to_string());
    }
    let mut out = Vec::with_capacity(cleaned.len() / 2);
    let (pairs, _) = cleaned.as_bytes().as_chunks::<2>();
    for pair in pairs {
        let s = std::str::from_utf8(pair).map_err(|_| "Invalid ASCII in hex")?;
        let byte = u8::from_str_radix(s, 16).map_err(|e| format!("Invalid hex byte '{s}': {e}"))?;
        out.push(byte);
    }
    Ok(out)
}

fn format_hex_dump(raw: &[u8], max_bytes: usize) -> String {
    let view = &raw[..raw.len().min(max_bytes)];
    let mut out = String::new();
    for (row_idx, chunk) in view.chunks(16).enumerate() {
        let offset = row_idx * 16;
        let mut hex_col = String::new();
        for (i, b) in chunk.iter().enumerate() {
            if i == 8 {
                hex_col.push(' ');
            }
            if i > 0 {
                hex_col.push(' ');
            }
            let _ = write!(&mut hex_col, "{b:02x}");
        }
        let ascii: String = chunk
            .iter()
            .map(|&b| if (0x20..=0x7e).contains(&b) { b as char } else { '.' })
            .collect();
        let _ = writeln!(&mut out, "{offset:08x}  {hex_col:<48}  |{ascii}|");
    }
    out
}

// ---------------------------------------------------------------------------
// Strict RFC 8259 JSON Parser (duplicate keys, NaN/Inf, lone surrogates)
// ---------------------------------------------------------------------------

struct JsonParser<'a> {
    src: &'a [u8],
    pos: usize,
}

impl<'a> JsonParser<'a> {
    fn new(src: &'a str) -> Self {
        Self { src: src.as_bytes(), pos: 0 }
    }

    fn skip_ws(&mut self) {
        while self.pos < self.src.len() {
            match self.src[self.pos] {
                b' ' | b'\t' | b'\n' | b'\r' => self.pos += 1,
                _ => break,
            }
        }
    }

    fn parse_root(&mut self) -> Result<JsonValue, (&'static str, String)> {
        self.skip_ws();
        let v = self.parse_value()?;
        self.skip_ws();
        if self.pos != self.src.len() {
            return Err(("json_parse_error", format!("Extra data at byte offset {}", self.pos)));
        }
        Ok(v)
    }

    fn parse_value(&mut self) -> Result<JsonValue, (&'static str, String)> {
        self.skip_ws();
        if self.pos >= self.src.len() {
            return Err(("json_parse_error", "Unexpected end of JSON input".to_string()));
        }
        match self.src[self.pos] {
            b'{' => self.parse_object(),
            b'[' => self.parse_array(),
            b'"' => self.parse_string().map(JsonValue::Str),
            b't' => {
                if self.src[self.pos..].starts_with(b"true") {
                    self.pos += 4;
                    Ok(JsonValue::Bool(true))
                } else {
                    Err(("json_parse_error", "Invalid literal starting with 't'".to_string()))
                }
            }
            b'f' => {
                if self.src[self.pos..].starts_with(b"false") {
                    self.pos += 5;
                    Ok(JsonValue::Bool(false))
                } else {
                    Err(("json_parse_error", "Invalid literal starting with 'f'".to_string()))
                }
            }
            b'n' => {
                if self.src[self.pos..].starts_with(b"null") {
                    self.pos += 4;
                    Ok(JsonValue::Null)
                } else {
                    Err(("json_parse_error", "Invalid literal starting with 'n'".to_string()))
                }
            }
            b'N' if self.src[self.pos..].starts_with(b"NaN") => Err((
                "non_finite_json_constant",
                "non-finite JSON constant: NaN".to_string(),
            )),
            b'I' if self.src[self.pos..].starts_with(b"Infinity") => Err((
                "non_finite_json_constant",
                "non-finite JSON constant: Infinity".to_string(),
            )),
            b'-' if self.src[self.pos..].starts_with(b"-Infinity") => Err((
                "non_finite_json_constant",
                "non-finite JSON constant: -Infinity".to_string(),
            )),
            b'-' | b'0'..=b'9' => self.parse_number(),
            other => Err((
                "json_parse_error",
                format!("Unexpected byte 0x{:02x} at offset {}", other, self.pos),
            )),
        }
    }

    fn parse_hex4(&mut self) -> Result<u16, (&'static str, String)> {
        if self.pos + 4 > self.src.len() {
            return Err(("json_parse_error", "Truncated \\uXXXX escape".to_string()));
        }
        let hex = std::str::from_utf8(&self.src[self.pos..self.pos + 4])
            .map_err(|_| ("json_parse_error", "Invalid UTF-8 in \\uXXXX escape".to_string()))?;
        let val = u16::from_str_radix(hex, 16)
            .map_err(|_| ("json_parse_error", format!("Invalid hex escape \\u{hex}")))?;
        self.pos += 4;
        Ok(val)
    }

    fn parse_string(&mut self) -> Result<JsonStr, (&'static str, String)> {
        self.pos += 1; // '"'
        let mut text = String::new();
        let mut has_lone_surrogate = false;
        while self.pos < self.src.len() {
            let b = self.src[self.pos];
            if b == b'"' {
                self.pos += 1;
                return Ok(JsonStr { text, has_lone_surrogate });
            }
            if b < 0x20 {
                return Err(("json_parse_error", "Unescaped control char in JSON string".to_string()));
            }
            if b == b'\\' {
                self.pos += 1;
                if self.pos >= self.src.len() {
                    return Err(("json_parse_error", "Unterminated JSON escape sequence".to_string()));
                }
                let esc = self.src[self.pos];
                self.pos += 1;
                match esc {
                    b'"' => text.push('"'),
                    b'\\' => text.push('\\'),
                    b'/' => text.push('/'),
                    b'b' => text.push('\u{0008}'),
                    b'f' => text.push('\u{000c}'),
                    b'n' => text.push('\n'),
                    b'r' => text.push('\r'),
                    b't' => text.push('\t'),
                    b'u' => {
                        let code = self.parse_hex4()?;
                        if (0xD800..=0xDBFF).contains(&code) {
                            if self.pos + 2 <= self.src.len()
                                && self.src[self.pos] == b'\\'
                                && self.src[self.pos + 1] == b'u'
                            {
                                let save = self.pos;
                                self.pos += 2;
                                let low = self.parse_hex4()?;
                                if (0xDC00..=0xDFFF).contains(&low) {
                                    let cp = 0x10000
                                        + (((code as u32 - 0xD800) << 10) | (low as u32 - 0xDC00));
                                    if let Some(ch) = char::from_u32(cp) {
                                        text.push(ch);
                                        continue;
                                    }
                                }
                                self.pos = save;
                            }
                            has_lone_surrogate = true;
                        } else if (0xDC00..=0xDFFF).contains(&code) {
                            has_lone_surrogate = true;
                        } else if let Some(ch) = char::from_u32(code as u32) {
                            text.push(ch);
                        }
                    }
                    other => {
                        return Err(("json_parse_error", format!("Invalid escape \\{}", other as char)));
                    }
                }
            } else {
                let s = std::str::from_utf8(&self.src[self.pos..])
                    .map_err(|_| ("invalid_utf8_bytes", "Invalid UTF-8 inside string".to_string()))?;
                let ch = s.chars().next().unwrap();
                text.push(ch);
                self.pos += ch.len_utf8();
            }
        }
        Err(("json_parse_error", "Unterminated JSON string".to_string()))
    }

    fn parse_number(&mut self) -> Result<JsonValue, (&'static str, String)> {
        let start = self.pos;
        if self.src[self.pos] == b'-' {
            self.pos += 1;
        }
        if self.pos >= self.src.len() || !self.src[self.pos].is_ascii_digit() {
            return Err(("json_parse_error", "Invalid JSON number".to_string()));
        }
        if self.src[self.pos] == b'0' {
            self.pos += 1;
            if self.pos < self.src.len() && self.src[self.pos].is_ascii_digit() {
                return Err(("json_parse_error", "Leading zero in JSON number".to_string()));
            }
        } else {
            while self.pos < self.src.len() && self.src[self.pos].is_ascii_digit() {
                self.pos += 1;
            }
        }
        let mut is_float = false;
        if self.pos < self.src.len() && self.src[self.pos] == b'.' {
            is_float = true;
            self.pos += 1;
            if self.pos >= self.src.len() || !self.src[self.pos].is_ascii_digit() {
                return Err(("json_parse_error", "Invalid fraction in JSON number".to_string()));
            }
            while self.pos < self.src.len() && self.src[self.pos].is_ascii_digit() {
                self.pos += 1;
            }
        }
        if self.pos < self.src.len() && matches!(self.src[self.pos], b'e' | b'E') {
            is_float = true;
            self.pos += 1;
            if self.pos < self.src.len() && matches!(self.src[self.pos], b'+' | b'-') {
                self.pos += 1;
            }
            if self.pos >= self.src.len() || !self.src[self.pos].is_ascii_digit() {
                return Err(("json_parse_error", "Invalid exponent in JSON number".to_string()));
            }
            while self.pos < self.src.len() && self.src[self.pos].is_ascii_digit() {
                self.pos += 1;
            }
        }
        let raw = std::str::from_utf8(&self.src[start..self.pos]).unwrap();
        if is_float {
            let f: f64 = raw
                .parse()
                .map_err(|_| ("json_parse_error", format!("Invalid float literal '{raw}'")))?;
            if !f.is_finite() {
                return Err(("non_finite_json_constant", format!("Non-finite float '{raw}'")));
            }
            Ok(JsonValue::Float(f))
        } else if let Ok(i) = raw.parse::<i64>() {
            Ok(JsonValue::Int(i))
        } else {
            Ok(JsonValue::BigInt(raw.to_string()))
        }
    }

    fn parse_array(&mut self) -> Result<JsonValue, (&'static str, String)> {
        self.pos += 1; // '['
        let mut items = Vec::new();
        self.skip_ws();
        if self.pos < self.src.len() && self.src[self.pos] == b']' {
            self.pos += 1;
            return Ok(JsonValue::Array(items));
        }
        loop {
            items.push(self.parse_value()?);
            self.skip_ws();
            if self.pos >= self.src.len() {
                return Err(("json_parse_error", "Unterminated JSON array".to_string()));
            }
            match self.src[self.pos] {
                b',' => {
                    self.pos += 1;
                }
                b']' => {
                    self.pos += 1;
                    return Ok(JsonValue::Array(items));
                }
                other => {
                    return Err((
                        "json_parse_error",
                        format!("Expected ',' or ']' in array, got 0x{other:02x}"),
                    ))
                }
            }
        }
    }

    fn parse_object(&mut self) -> Result<JsonValue, (&'static str, String)> {
        self.pos += 1; // '{'
        let mut entries: Vec<(JsonStr, JsonValue)> = Vec::new();
        let mut seen = BTreeMap::<String, ()>::new();
        self.skip_ws();
        if self.pos < self.src.len() && self.src[self.pos] == b'}' {
            self.pos += 1;
            return Ok(JsonValue::Object(entries));
        }
        loop {
            self.skip_ws();
            if self.pos >= self.src.len() || self.src[self.pos] != b'"' {
                return Err(("json_parse_error", "Expected string key in JSON object".to_string()));
            }
            let key = self.parse_string()?;
            self.skip_ws();
            if self.pos >= self.src.len() || self.src[self.pos] != b':' {
                return Err(("json_parse_error", "Expected ':' after JSON key".to_string()));
            }
            self.pos += 1;
            let val = self.parse_value()?;
            if seen.insert(key.text.clone(), ()).is_some() {
                return Err((
                    "duplicate_json_key",
                    format!("duplicate JSON key: {:?}", key.text),
                ));
            }
            entries.push((key, val));
            self.skip_ws();
            if self.pos >= self.src.len() {
                return Err(("json_parse_error", "Unterminated JSON object".to_string()));
            }
            match self.src[self.pos] {
                b',' => {
                    self.pos += 1;
                }
                b'}' => {
                    self.pos += 1;
                    return Ok(JsonValue::Object(entries));
                }
                other => {
                    return Err((
                        "json_parse_error",
                        format!("Expected ',' or '}}' in object, got 0x{other:02x}"),
                    ))
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Python-Compatible Canonical JSON Serializer (for HMAC-SHA256 verification)
// ---------------------------------------------------------------------------

fn write_json_escaped_ascii(s: &str, out: &mut String) {
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\u{0008}' => out.push_str("\\b"),
            '\u{000c}' => out.push_str("\\f"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c if (c as u32) <= 0x7E => out.push(c),
            c if (c as u32) <= 0xFFFF => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => {
                let mut buf = [0u16; 2];
                for unit in c.encode_utf16(&mut buf) {
                    let _ = write!(out, "\\u{:04x}", *unit);
                }
            }
        }
    }
    out.push('"');
}

fn format_python_float(v: f64) -> String {
    if v == 0.0 {
        return if v.is_sign_negative() { "-0.0".to_string() } else { "0.0".to_string() };
    }
    let abs_v = v.abs();
    if abs_v < 1e-4 || abs_v >= 1e16 {
        let sign = if v < 0.0 { "-" } else { "" };
        let mut exp = abs_v.log10().floor() as i32;
        let mut mant = abs_v / 10f64.powi(exp);
        if mant >= 10.0 - 1e-15 {
            let test_pow = 10f64.powi(exp + 1);
            if abs_v >= test_pow {
                exp += 1;
                mant = abs_v / test_pow;
            }
        } else if mant < 1.0 {
            exp -= 1;
            mant = abs_v / 10f64.powi(exp);
        }
        let mut m_str = mant.to_string();
        if let Ok(parsed) = format!("{sign}{m_str}e{exp}").parse::<f64>() {
            if parsed != v {
                m_str = format!("{mant:.16}")
                    .trim_end_matches('0')
                    .trim_end_matches('.')
                    .to_string();
            }
        }
        let exp_str = if exp < 0 { format!("e-{:02}", -exp) } else { format!("e+{:02}", exp) };
        format!("{sign}{m_str}{exp_str}")
    } else {
        let s = v.to_string();
        if s.contains('.') { s } else { format!("{s}.0") }
    }
}

fn write_canonical_json(val: &JsonValue, out: &mut String) {
    match val {
        JsonValue::Null => out.push_str("null"),
        JsonValue::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        JsonValue::Int(i) => out.push_str(&i.to_string()),
        JsonValue::BigInt(s) => out.push_str(s),
        JsonValue::Float(f) => out.push_str(&format_python_float(*f)),
        JsonValue::Str(s) => write_json_escaped_ascii(&s.text, out),
        JsonValue::Array(items) => {
            out.push('[');
            for (idx, item) in items.iter().enumerate() {
                if idx > 0 { out.push(','); }
                write_canonical_json(item, out);
            }
            out.push(']');
        }
        JsonValue::Object(entries) => {
            let mut sorted: BTreeMap<&str, &JsonValue> = BTreeMap::new();
            for (k, v) in entries {
                sorted.insert(k.text.as_str(), v);
            }
            out.push('{');
            for (idx, (k, v)) in sorted.iter().enumerate() {
                if idx > 0 { out.push(','); }
                write_json_escaped_ascii(k, out);
                out.push(':');
                write_canonical_json(v, out);
            }
            out.push('}');
        }
    }
}

fn python_type_name(val: &JsonValue) -> &'static str {
    match val {
        JsonValue::Null => "NoneType",
        JsonValue::Bool(_) => "bool",
        JsonValue::Int(_) | JsonValue::BigInt(_) => "int",
        JsonValue::Float(_) => "float",
        JsonValue::Str(_) => "str",
        JsonValue::Array(_) => "list",
        JsonValue::Object(_) => "dict",
    }
}

// ---------------------------------------------------------------------------
// HardenedBoundaryGate Native Implementation
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct ParseOutcome {
    accepted: bool,
    error_code: Option<&'static str>,
    error_detail: Option<String>,
    msg_id: Option<String>,
    provenance: Option<String>,
    performative: Option<String>,
}

fn reject(code: &'static str, detail: impl Into<String>) -> ParseOutcome {
    ParseOutcome {
        accepted: false,
        error_code: Some(code),
        error_detail: Some(detail.into()),
        msg_id: None,
        provenance: None,
        performative: None,
    }
}

fn is_valid_identifier(s: &str) -> bool {
    !s.is_empty()
        && s.len() <= 128
        && s.bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'.' | b':' | b'-'))
}

fn is_lower_hex64(s: &str) -> bool {
    s.len() == 64 && s.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
}

fn has_forbidden_ctrl_or_bidi(s: &str) -> bool {
    s.chars().any(|ch| {
        let u = ch as u32;
        matches!(
            u,
            0x00..=0x08 | 0x0B | 0x0C | 0x0E..=0x1F | 0x7F | 0x202A..=0x202E | 0x2066..=0x2069
        )
    })
}

fn has_special_token_delimiter(s: &str) -> bool {
    for pat in ["[INST]", "[/INST]", "<s>", "</s>", "<<SYS>>", "<</SYS>>"] {
        if s.contains(pat) {
            return true;
        }
    }
    let bytes = s.as_bytes();
    let mut i = 0;
    while i + 3 < bytes.len() {
        if bytes[i] == b'<' && bytes[i + 1] == b'|' {
            let max_end_excl = (i + 2 + 33).min(bytes.len().saturating_sub(1));
            for j in (i + 2)..max_end_excl {
                if bytes[j] == b'|' && bytes[j + 1] == b'>' {
                    if j - (i + 2) >= 1 && j - (i + 2) <= 32 {
                        return true;
                    }
                    break;
                }
                if bytes[j] == b'|' {
                    break;
                }
            }
        }
        i += 1;
    }
    false
}

fn check_text_boundaries(text: &str, field_name: &str, reject_special: bool) -> Option<ParseOutcome> {
    if has_forbidden_ctrl_or_bidi(text) {
        return Some(reject(
            "forbidden_control_or_bidi_char",
            format!("forbidden control or bidi character in {field_name}"),
        ));
    }
    if reject_special && has_special_token_delimiter(text) {
        return Some(reject(
            "special_token_delimiter",
            format!("chat-template special token delimiter in {field_name}"),
        ));
    }
    None
}

fn validate_tree(val: &JsonValue, remaining: isize, max_depth: usize) -> Option<ParseOutcome> {
    if remaining < 0 {
        return Some(reject(
            "max_nesting_depth_exceeded",
            format!("JSON nesting exceeded limit {max_depth}"),
        ));
    }
    match val {
        JsonValue::Str(s) => {
            if s.has_lone_surrogate {
                return Some(reject(
                    "invalid_surrogate_in_payload",
                    "lone UTF-16 surrogate in JSON string",
                ));
            }
            None
        }
        JsonValue::Object(entries) => {
            for (k, v) in entries {
                if k.has_lone_surrogate {
                    return Some(reject(
                        "invalid_surrogate_in_payload",
                        "lone UTF-16 surrogate in JSON key",
                    ));
                }
                if let Some(err) = validate_tree(v, remaining - 1, max_depth) {
                    return Some(err);
                }
            }
            None
        }
        JsonValue::Array(items) => {
            for item in items {
                if let Some(err) = validate_tree(item, remaining - 1, max_depth) {
                    return Some(err);
                }
            }
            None
        }
        _ => None,
    }
}

fn parse_bytes(raw: &[u8], cfg: &GateConfig) -> ParseOutcome {
    if raw.is_empty() {
        return reject("empty_wire_bytes", "empty byte stream");
    }

    let body: &[u8] = match cfg.framing {
        FramingMode::Raw => raw,
        FramingMode::LengthPrefixed => {
            if raw.len() < 4 {
                return reject(
                    "frame_header_truncated",
                    format!("expected 4-byte length prefix, got {} bytes", raw.len()),
                );
            }
            let declared = u32::from_be_bytes([raw[0], raw[1], raw[2], raw[3]]) as usize;
            if declared == 0 {
                return reject("empty_wire_bytes", "declared frame length is 0");
            }
            if declared > cfg.max_wire_bytes {
                return reject(
                    "payload_too_large",
                    format!("declared frame length {declared} exceeds cap {}", cfg.max_wire_bytes),
                );
            }
            let b = &raw[4..];
            if b.len() != declared {
                return reject(
                    "frame_length_mismatch",
                    format!("declared {declared} bytes, actual body is {} bytes", b.len()),
                );
            }
            b
        }
        FramingMode::Ndjson => {
            if !raw.ends_with(b"\n") {
                return reject(
                    "frame_missing_delimiter",
                    "NDJSON frame must terminate with a newline byte",
                );
            }
            let b = &raw[..raw.len() - 1];
            if b.contains(&b'\n') || b.contains(&b'\r') {
                return reject(
                    "frame_embedded_newline",
                    "NDJSON frame contains embedded CR/LF delimiter",
                );
            }
            b
        }
    };

    if body.is_empty() {
        return reject("empty_wire_bytes", "empty frame body");
    }
    if body.len() > cfg.max_wire_bytes {
        return reject(
            "payload_too_large",
            format!("body size {} exceeds max_wire_bytes {}", body.len(), cfg.max_wire_bytes),
        );
    }
    if body.starts_with(UTF8_BOM) {
        return reject("utf8_bom_forbidden", "UTF-8 BOM prefix is not permitted");
    }

    let text = match std::str::from_utf8(body) {
        Ok(s) => s,
        Err(e) => return reject("invalid_utf8_bytes", e.to_string()),
    };
    if text.trim().is_empty() {
        return reject("empty_wire_text", "decoded payload is whitespace-only");
    }

    let mut parser = JsonParser::new(text);
    let doc = match parser.parse_root() {
        Ok(v) => v,
        Err((code, detail)) => return reject(code, detail),
    };

    let entries = match &doc {
        JsonValue::Object(e) => e,
        other => {
            return reject(
                "root_must_be_object",
                format!("JSON root must be an object, got {}", python_type_name(other)),
            )
        }
    };

    if let Some(err) = validate_tree(&doc, cfg.max_nesting_depth as isize, cfg.max_nesting_depth) {
        return err;
    }

    let mut fields: BTreeMap<&str, &JsonValue> = BTreeMap::new();
    for (k, v) in entries {
        fields.insert(k.text.as_str(), v);
    }

    const REQUIRED: [&str; 9] = [
        "msg_id", "sender", "recipient", "round_idx", "performative", "claim", "answer", "confidence", "provenance",
    ];
    const ALLOWED: [&str; 12] = [
        "msg_id", "sender", "recipient", "round_idx", "performative", "claim", "answer", "confidence", "provenance",
        "constraints", "evidence_refs", "oracle_sig",
    ];

    let missing: Vec<&str> = REQUIRED.iter().copied().filter(|k| !fields.contains_key(k)).collect();
    if !missing.is_empty() {
        return reject("missing_required_fields", format!("missing required fields: {missing:?}"));
    }
    let extra: Vec<&str> = fields.keys().copied().filter(|k| !ALLOWED.contains(k)).collect();
    if !extra.is_empty() {
        return reject("unexpected_fields", format!("unexpected fields: {extra:?}"));
    }

    for id_field in ["msg_id", "sender", "recipient"] {
        match fields.get(id_field) {
            Some(JsonValue::Str(s)) if is_valid_identifier(&s.text) => {}
            _ => return reject("invalid_identifier", format!("invalid identifier for {id_field}")),
        }
    }

    let round_idx = match fields.get("round_idx") {
        Some(JsonValue::Int(i)) if *i >= 0 => *i,
        _ => return reject("invalid_round_idx", "invalid round_idx"),
    };

    match fields.get("confidence") {
        Some(JsonValue::Int(i)) if (0..=1).contains(i) => {}
        Some(JsonValue::Float(f)) if f.is_finite() && (0.0..=1.0).contains(f) => {}
        _ => return reject("invalid_confidence", "invalid confidence"),
    }

    let performative = match fields.get("performative") {
        Some(JsonValue::Str(s))
            if matches!(s.text.as_str(), "propose" | "critique" | "verify" | "revise" | "abstain") =>
        {
            s.text.as_str()
        }
        _ => return reject("invalid_enum", "invalid performative enum"),
    };

    let provenance = match fields.get("provenance") {
        Some(JsonValue::Str(s)) if matches!(s.text.as_str(), "observed" | "derived" | "assumed") => {
            s.text.as_str()
        }
        _ => return reject("invalid_enum", "invalid provenance enum"),
    };

    let claim = match fields.get("claim") {
        Some(JsonValue::Str(s)) if !s.text.trim().is_empty() => s.text.as_str(),
        _ => return reject("invalid_or_empty_claim", "claim must be a non-empty string"),
    };
    if claim.chars().count() > cfg.max_claim_chars {
        return reject(
            "claim_too_long",
            format!("claim length exceeds max_claim_chars {}", cfg.max_claim_chars),
        );
    }
    if let Some(err) = check_text_boundaries(claim, "claim", cfg.reject_special_tokens) {
        return err;
    }

    let constraints_ref: Option<&Vec<JsonValue>> = match fields.get("constraints") {
        None => None,
        Some(JsonValue::Array(items)) => {
            for item in items {
                match item {
                    JsonValue::Str(s) => {
                        if let Some(err) = check_text_boundaries(&s.text, "constraint", cfg.reject_special_tokens) {
                            return err;
                        }
                    }
                    _ => return reject("invalid_constraints_type", "constraints must be list[str]"),
                }
            }
            Some(items)
        }
        Some(_) => return reject("invalid_constraints_type", "constraints must be list[str]"),
    };

    if let Some(ev_val) = fields.get("evidence_refs") {
        match ev_val {
            JsonValue::Array(items) => {
                for item in items {
                    match item {
                        JsonValue::Str(s) => {
                            if let Some(err) = check_text_boundaries(&s.text, "evidence_ref", cfg.reject_special_tokens) {
                                return err;
                            }
                        }
                        _ => return reject("invalid_evidence_refs_type", "evidence_refs must be list[str]"),
                    }
                }
            }
            _ => return reject("invalid_evidence_refs_type", "evidence_refs must be list[str]"),
        }
    }

    let oracle_sig = match fields.get("oracle_sig") {
        None | Some(JsonValue::Null) => None,
        Some(JsonValue::Str(s)) => Some(s.text.as_str()),
        Some(_) => return reject("invalid_oracle_sig_type", "oracle_sig must be str or null"),
    };

    let answer_val = *fields.get("answer").unwrap();
    if let JsonValue::Str(s) = answer_val {
        if let Some(err) = check_text_boundaries(&s.text, "answer", cfg.reject_special_tokens) {
            return err;
        }
    }

    let msg_id = match fields.get("msg_id") {
        Some(JsonValue::Str(s)) => s.text.as_str(),
        _ => unreachable!(),
    };

    if provenance == "observed" {
        if cfg.require_signed_observed || (oracle_sig.is_some() && !cfg.oracle_key.is_empty()) {
            if cfg.oracle_key.is_empty() {
                return reject("oracle_key_unconfigured", "oracle key is not configured");
            }
            let sig = match oracle_sig {
                Some(s) if !s.is_empty() => s,
                _ => return reject("missing_oracle_signature", "observed provenance requires oracle_sig"),
            };
            if !is_lower_hex64(sig) {
                return reject(
                    "malformed_oracle_signature_format",
                    "oracle_sig must be a 64-char lowercase hex SHA-256 digest",
                );
            }

            let constraints_vec = match constraints_ref {
                Some(v) => v.clone(),
                None => Vec::new(),
            };
            let canonical_obj = JsonValue::Object(vec![
                (
                    JsonStr { text: "answer".to_string(), has_lone_surrogate: false },
                    answer_val.clone(),
                ),
                (
                    JsonStr { text: "answer_type".to_string(), has_lone_surrogate: false },
                    JsonValue::Str(JsonStr {
                        text: python_type_name(answer_val).to_string(),
                        has_lone_surrogate: false,
                    }),
                ),
                (
                    JsonStr { text: "claim".to_string(), has_lone_surrogate: false },
                    JsonValue::Str(JsonStr { text: claim.to_string(), has_lone_surrogate: false }),
                ),
                (
                    JsonStr { text: "constraints".to_string(), has_lone_surrogate: false },
                    JsonValue::Array(constraints_vec),
                ),
                (
                    JsonStr { text: "domain".to_string(), has_lone_surrogate: false },
                    JsonValue::Str(JsonStr {
                        text: "agent-protocol-v1:observed".to_string(),
                        has_lone_surrogate: false,
                    }),
                ),
                (
                    JsonStr { text: "msg_id".to_string(), has_lone_surrogate: false },
                    JsonValue::Str(JsonStr { text: msg_id.to_string(), has_lone_surrogate: false }),
                ),
                (
                    JsonStr { text: "provenance".to_string(), has_lone_surrogate: false },
                    JsonValue::Str(JsonStr { text: provenance.to_string(), has_lone_surrogate: false }),
                ),
                (
                    JsonStr { text: "round_idx".to_string(), has_lone_surrogate: false },
                    JsonValue::Int(round_idx),
                ),
            ]);
            let mut canonical = String::new();
            write_canonical_json(&canonical_obj, &mut canonical);
            let expected_hex = hmac_sha256_hex(&cfg.oracle_key, canonical.as_bytes());
            if !constant_time_eq(expected_hex.as_bytes(), sig.as_bytes()) {
                return reject("invalid_oracle_signature", "HMAC signature verification failed");
            }
        }
    } else if oracle_sig.is_some() {
        return reject(
            "unexpected_signature_on_non_observed",
            format!("oracle_sig is only allowed on observed provenance, got {provenance}"),
        );
    }

    ParseOutcome {
        accepted: true,
        error_code: None,
        error_detail: None,
        msg_id: Some(msg_id.to_string()),
        provenance: Some(provenance.to_string()),
        performative: Some(performative.to_string()),
    }
}

fn parse_bool_flag(val: &str) -> Result<bool, String> {
    match val.to_ascii_lowercase().as_str() {
        "true" | "1" | "yes" => Ok(true),
        "false" | "0" | "no" => Ok(false),
        _ => Err(format!("Invalid boolean flag value: '{val}'")),
    }
}

fn outcome_to_json(raw: &[u8], cfg: &GateConfig, out: &ParseOutcome, elapsed_us: f64) -> String {
    let sha = to_hex_lower(&sha256_bytes(raw));
    let mut parts = Vec::new();
    parts.push(format!("\"accepted\":{}", out.accepted));
    match out.error_code {
        Some(c) => {
            let mut cs = String::new();
            write_json_escaped_ascii(c, &mut cs);
            parts.push(format!("\"error_code\":{cs}"));
        }
        None => parts.push("\"error_code\":null".to_string()),
    }
    match &out.error_detail {
        Some(d) => {
            let mut ds = String::new();
            write_json_escaped_ascii(d, &mut ds);
            parts.push(format!("\"error_detail\":{ds}"));
        }
        None => parts.push("\"error_detail\":null".to_string()),
    }
    parts.push(format!("\"elapsed_us\":{elapsed_us:.3}"));
    parts.push(format!("\"framing\":\"{}\"", cfg.framing.as_str()));
    parts.push(format!("\"byte_length\":{}", raw.len()));
    parts.push(format!("\"sha256\":\"{sha}\""));
    if let (Some(mid), Some(prov), Some(perf)) = (&out.msg_id, &out.provenance, &out.performative) {
        let mut ms = String::new();
        write_json_escaped_ascii(mid, &mut ms);
        parts.push(format!(
            "\"envelope\":{{\"msg_id\":{ms},\"provenance\":\"{prov}\",\"performative\":\"{perf}\"}}"
        ));
    } else {
        parts.push("\"envelope\":null".to_string());
    }
    parts.push("\"runtime\":\"rust-native\"".to_string());
    format!("{{{}}}", parts.join(","))
}

fn base64_decode(input: &str) -> Result<Vec<u8>, String> {
    let clean: Vec<u8> = input.bytes().filter(|b| !b.is_ascii_whitespace()).collect();
    if clean.is_empty() {
        return Ok(Vec::new());
    }
    if !clean.len().is_multiple_of(4) {
        return Err("Invalid base64 length".to_string());
    }
    let val = |b: u8| -> Result<u8, String> {
        match b {
            b'A'..=b'Z' => Ok(b - b'A'),
            b'a'..=b'z' => Ok(b - b'a' + 26),
            b'0'..=b'9' => Ok(b - b'0' + 52),
            b'+' => Ok(62),
            b'/' => Ok(63),
            _ => Err(format!("Invalid base64 byte: 0x{b:02x}")),
        }
    };
    let mut out = Vec::with_capacity((clean.len() / 4) * 3);
    let (chunks, _) = clean.as_chunks::<4>();
    for chunk in chunks {
        let a = val(chunk[0])?;
        let b = val(chunk[1])?;
        if chunk[2] == b'=' {
            if chunk[3] != b'=' {
                return Err("Invalid base64 padding".to_string());
            }
            out.push((a << 2) | (b >> 4));
        } else if chunk[3] == b'=' {
            let c = val(chunk[2])?;
            out.push((a << 2) | (b >> 4));
            out.push((b << 4) | (c >> 2));
        } else {
            let c = val(chunk[2])?;
            let d = val(chunk[3])?;
            out.push((a << 2) | (b >> 4));
            out.push((b << 4) | (c >> 2));
            out.push((c << 6) | d);
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Native Workbench Orchestrator, TCP Wire Probe, Hex Dumper & Microbenchmark
// ---------------------------------------------------------------------------

fn find_work121_root() -> Option<PathBuf> {
    if let Ok(p) = env::var("WORK121_HOME") {
        let pb = PathBuf::from(p);
        if pb.join("backend_server.py").is_file() {
            return Some(pb);
        }
    }
    if let Ok(cwd) = env::current_dir() {
        for cand in [cwd.clone(), cwd.join("work121")] {
            if cand.join("backend_server.py").is_file() {
                return Some(cand);
            }
        }
    }
    if let Ok(exe) = env::current_exe() {
        let mut cur = exe.parent().map(Path::to_path_buf);
        for _ in 0..6 {
            if let Some(dir) = cur {
                if dir.join("backend_server.py").is_file() {
                    return Some(dir);
                }
                cur = dir.parent().map(Path::to_path_buf);
            } else {
                break;
            }
        }
    }
    None
}

fn find_python(root: &Path) -> PathBuf {
    if let Ok(py) = env::var("WORK121_PYTHON") {
        let p = PathBuf::from(py);
        if p.exists() {
            return p;
        }
    }
    for c in [
        root.join(".venv").join("Scripts").join("python.exe"),
        root.join(".venv").join("bin").join("python"),
    ] {
        if c.is_file() {
            return c;
        }
    }
    PathBuf::from("python")
}

fn open_browser(url: &str) {
    #[cfg(target_os = "windows")]
    {
        let _ = Command::new("cmd").args(["/c", "start", "", url]).spawn();
    }
    #[cfg(target_os = "macos")]
    {
        let _ = Command::new("open").arg(url).spawn();
    }
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        let _ = Command::new("xdg-open").arg(url).spawn();
    }
}

fn raw_tcp_http_request(host: &str, port: u16, req: &str) -> Result<(f64, String), String> {
    let addr: SocketAddr = format!("{host}:{port}")
        .parse()
        .map_err(|e| format!("Invalid socket addr: {e}"))?;
    let t0 = Instant::now();
    let mut stream = TcpStream::connect_timeout(&addr, Duration::from_secs(2))
        .map_err(|e| format!("TCP connect failed to {addr}: {e}"))?;
    let _ = stream.set_read_timeout(Some(Duration::from_secs(3)));
    let _ = stream.set_write_timeout(Some(Duration::from_secs(3)));
    stream.write_all(req.as_bytes()).map_err(|e| format!("TCP write failed: {e}"))?;
    let mut resp = Vec::new();
    let _ = stream.read_to_end(&mut resp);
    let ms = t0.elapsed().as_secs_f64() * 1_000.0;
    Ok((ms, String::from_utf8_lossy(&resp).into_owned()))
}

fn parse_wire_flags(args: &[String]) -> Result<(GateConfig, Option<Vec<u8>>, bool), String> {
    let mut cfg = GateConfig::default();
    let mut use_stdin = true;
    let mut input_file: Option<String> = None;
    let mut hex_input: Option<String> = None;
    let mut json_mode = false;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--framing" => {
                i += 1;
                let v = args.get(i).ok_or("--framing requires a value")?;
                cfg.framing = FramingMode::from_str(v)?;
            }
            "--max-wire-bytes" => {
                i += 1;
                cfg.max_wire_bytes = args
                    .get(i)
                    .and_then(|s| s.parse().ok())
                    .ok_or("Invalid --max-wire-bytes")?;
            }
            "--max-claim-chars" => {
                i += 1;
                cfg.max_claim_chars = args
                    .get(i)
                    .and_then(|s| s.parse().ok())
                    .ok_or("Invalid --max-claim-chars")?;
            }
            "--max-nesting-depth" => {
                i += 1;
                cfg.max_nesting_depth = args
                    .get(i)
                    .and_then(|s| s.parse().ok())
                    .ok_or("Invalid --max-nesting-depth")?;
            }
            "--require-signed-observed" => {
                i += 1;
                let v = args.get(i).ok_or("--require-signed-observed requires true|false")?;
                cfg.require_signed_observed = parse_bool_flag(v)?;
            }
            "--reject-special-tokens" => {
                i += 1;
                let v = args.get(i).ok_or("--reject-special-tokens requires true|false")?;
                cfg.reject_special_tokens = parse_bool_flag(v)?;
            }
            "--oracle-key" => {
                i += 1;
                let v = args.get(i).ok_or("--oracle-key requires a value")?;
                cfg.oracle_key = v.as_bytes().to_vec();
            }
            "--oracle-key-hex" => {
                i += 1;
                let v = args.get(i).ok_or("--oracle-key-hex requires a hex value")?;
                cfg.oracle_key = decode_hex(v)?;
            }
            "--stdin" => use_stdin = true,
            "--file" => {
                i += 1;
                input_file = Some(args.get(i).ok_or("--file requires a path")?.clone());
                use_stdin = false;
            }
            "--hex" => {
                i += 1;
                hex_input = Some(args.get(i).ok_or("--hex requires a hex string")?.clone());
                use_stdin = false;
            }
            "--json" => json_mode = true,
            other => return Err(format!("Unknown flag: {other}")),
        }
        i += 1;
    }

    let raw = if let Some(h) = hex_input {
        Some(decode_hex(&h)?)
    } else if let Some(path) = input_file {
        Some(fs::read(&path).map_err(|e| format!("Failed to read '{path}': {e}"))?)
    } else if use_stdin {
        let mut buf = Vec::new();
        io::stdin()
            .read_to_end(&mut buf)
            .map_err(|e| format!("Failed to read stdin: {e}"))?;
        Some(buf)
    } else {
        None
    };

    Ok((cfg, raw, json_mode))
}

fn cmd_wire_check_or_batch(subcmd: &str, args: &[String]) -> i32 {
    if subcmd == "wire-batch" {
        // Parse flags without consuming stdin before reading batch lines
        let mut cfg = GateConfig::default();
        let mut i = 0;
        while i < args.len() {
            match args[i].as_str() {
                "--framing" => {
                    i += 1;
                    if let Some(v) = args.get(i) {
                        cfg.framing = FramingMode::from_str(v).unwrap_or_else(|e| { eprintln!("{e}"); process::exit(2); });
                    }
                }
                "--max-wire-bytes" => { i += 1; cfg.max_wire_bytes = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(32_768); }
                "--max-claim-chars" => { i += 1; cfg.max_claim_chars = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(8_192); }
                "--max-nesting-depth" => { i += 1; cfg.max_nesting_depth = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(8); }
                "--require-signed-observed" => { i += 1; if let Some(v) = args.get(i) { cfg.require_signed_observed = parse_bool_flag(v).unwrap_or(true); } }
                "--reject-special-tokens" => { i += 1; if let Some(v) = args.get(i) { cfg.reject_special_tokens = parse_bool_flag(v).unwrap_or(true); } }
                "--oracle-key" => { i += 1; if let Some(v) = args.get(i) { cfg.oracle_key = v.as_bytes().to_vec(); } }
                "--oracle-key-hex" => {
                    i += 1;
                    if let Some(v) = args.get(i) {
                        cfg.oracle_key = decode_hex(v).unwrap_or_else(|e| { eprintln!("{e}"); process::exit(2); });
                    }
                }
                "--stdin" | "--json" => {}
                other => { eprintln!("Unknown flag: {other}"); return 2; }
            }
            i += 1;
        }
        let mut buf = String::new();
        if let Err(e) = io::stdin().read_to_string(&mut buf) {
            eprintln!("Failed to read stdin: {e}");
            return 2;
        }
        let mut results = Vec::new();
        let trimmed_buf = buf.trim();
        if trimmed_buf.starts_with('[') {
            let val = match JsonParser::new(trimmed_buf).parse_root() {
                Ok(v) => v,
                Err((_, e)) => { eprintln!("Invalid JSON batch array: {e}"); return 2; }
            };
            if let JsonValue::Array(items) = val {
                for (idx, item) in items.into_iter().enumerate() {
                    let mut case_id_opt: Option<String> = None;
                    let mut raw_bytes_opt: Option<Vec<u8>> = None;
                    if let JsonValue::Object(map) = item {
                        for (k, v) in map {
                            if k.text == "case_id" {
                                if let JsonValue::Str(s) = v { case_id_opt = Some(s.text); }
                            } else if k.text == "raw_hex" {
                                if let JsonValue::Str(s) = v {
                                    match decode_hex(&s.text) {
                                        Ok(b) => raw_bytes_opt = Some(b),
                                        Err(e) => {
                                            eprintln!("Invalid raw_hex at batch index {idx}: {e}");
                                            return 2;
                                        }
                                    }
                                }
                            } else if k.text == "raw_b64" {
                                if let JsonValue::Str(s) = v {
                                    match base64_decode(&s.text) {
                                        Ok(b) => raw_bytes_opt = Some(b),
                                        Err(e) => {
                                            eprintln!("Invalid raw_b64 at batch index {idx}: {e}");
                                            return 2;
                                        }
                                    }
                                }
                            }
                        }
                    } else {
                        eprintln!("Batch item at index {idx} must be a JSON object");
                        return 2;
                    }
                    let Some(raw) = raw_bytes_opt else {
                        eprintln!("Batch item at index {idx} missing raw_hex or raw_b64");
                        return 2;
                    };
                    let t0 = Instant::now();
                    let outcome = parse_bytes(&raw, &cfg);
                    let elapsed_us = t0.elapsed().as_secs_f64() * 1_000_000.0;
                    let outcome_json = outcome_to_json(&raw, &cfg, &outcome, elapsed_us);
                    let full_json = if let Some(cid) = case_id_opt {
                        let mut escaped_cid = String::new();
                        write_json_escaped_ascii(&cid, &mut escaped_cid);
                        format!("{{\"case_id\":{},{}", escaped_cid, &outcome_json[1..])
                    } else {
                        outcome_json
                    };
                    results.push(full_json);
                }
            } else {
                eprintln!("JSON batch root must be an array");
                return 2;
            }
        } else {
            for line in buf.split_terminator('\n') {
                let trimmed = line.trim_end_matches('\r').trim();
                let raw = match base64_decode(trimmed) {
                    Ok(b) => b,
                    Err(e) => { eprintln!("Invalid base64 batch line: {e}"); return 2; }
                };
                let t0 = Instant::now();
                let outcome = parse_bytes(&raw, &cfg);
                let elapsed_us = t0.elapsed().as_secs_f64() * 1_000_000.0;
                results.push(outcome_to_json(&raw, &cfg, &outcome, elapsed_us));
            }
        }
        println!("[{}]", results.join(","));
        return 0;
    }

    let (cfg, raw_opt, json_mode) = match parse_wire_flags(args) {
        Ok(v) => v,
        Err(e) => { eprintln!("{e}"); return 2; }
    };
    let raw = raw_opt.unwrap_or_default();
    let t0 = Instant::now();
    let outcome = parse_bytes(&raw, &cfg);
    let elapsed_us = t0.elapsed().as_secs_f64() * 1_000_000.0;
    if json_mode {
        println!("{}", outcome_to_json(&raw, &cfg, &outcome, elapsed_us));
    } else {
        println!(
            "accepted={} error_code={} elapsed_us={:.3} detail={}",
            outcome.accepted,
            outcome.error_code.unwrap_or("none"),
            elapsed_us,
            outcome.error_detail.as_deref().unwrap_or("")
        );
    }
    if outcome.accepted { 0 } else { 1 }
}

fn cmd_wire_hex(args: &[String]) -> i32 {
    let (cfg, raw_opt, _) = match parse_wire_flags(args) {
        Ok(v) => v,
        Err(e) => { eprintln!("{e}"); return 2; }
    };
    let raw = raw_opt.unwrap_or_default();
    println!("byte_length: {}", raw.len());
    println!("sha256:      {}", to_hex_lower(&sha256_bytes(&raw)));
    if cfg.framing == FramingMode::LengthPrefixed && raw.len() >= 4 {
        let decl = u32::from_be_bytes([raw[0], raw[1], raw[2], raw[3]]);
        println!(
            "frame_hdr:   {:02x} {:02x} {:02x} {:02x} (declared={} bytes, actual_body={} bytes)",
            raw[0], raw[1], raw[2], raw[3], decl, raw.len() - 4
        );
    }
    print!("{}", format_hex_dump(&raw, 8192));
    0
}

fn cmd_wire_bench(args: &[String]) -> i32 {
    let mut iterations: usize = 5_000;
    let mut framing = FramingMode::Raw;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--iterations" => {
                i += 1;
                if let Some(v) = args.get(i) { iterations = v.parse().unwrap_or(5_000); }
            }
            "--framing" => {
                i += 1;
                if let Some(v) = args.get(i) { framing = FramingMode::from_str(v).unwrap_or(FramingMode::Raw); }
            }
            _ => {}
        }
        i += 1;
    }

    let base_json = br#"{"msg_id":"bench-001","sender":"agent_a","recipient":"agent_b","round_idx":1,"performative":"propose","claim":"Deterministic benchmark payload","answer":42,"confidence":0.9,"provenance":"derived","constraints":["int"],"evidence_refs":["ref-1"],"oracle_sig":null}"#;
    let wire_payload: Vec<u8> = match framing {
        FramingMode::Raw => base_json.to_vec(),
        FramingMode::LengthPrefixed => {
            let mut v = (base_json.len() as u32).to_be_bytes().to_vec();
            v.extend_from_slice(base_json);
            v
        }
        FramingMode::Ndjson => {
            let mut v = base_json.to_vec();
            v.push(b'\n');
            v
        }
    };

    let cfg = GateConfig { framing, ..Default::default() };
    let mut latencies_us = Vec::with_capacity(iterations);
    let t_total = Instant::now();
    let mut accepted = 0usize;
    for _ in 0..iterations {
        let t0 = Instant::now();
        if parse_bytes(&wire_payload, &cfg).accepted {
            accepted += 1;
        }
        latencies_us.push(t0.elapsed().as_secs_f64() * 1_000_000.0);
    }
    let total_secs = t_total.elapsed().as_secs_f64();
    latencies_us.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p50 = latencies_us[latencies_us.len() / 2];
    let p95 = latencies_us[(latencies_us.len() * 95) / 100];
    let max_us = *latencies_us.last().unwrap_or(&0.0);
    let throughput = iterations as f64 / total_secs.max(1e-9);

    println!(
        "{{\"framing\":\"{}\",\"iterations\":{},\"accepted\":{},\"throughput_msgs_per_sec\":{:.1},\"p50_us\":{:.2},\"p95_us\":{:.2},\"max_us\":{:.2}}}",
        framing.as_str(),
        iterations,
        accepted,
        throughput,
        p50,
        p95,
        max_us
    );
    0
}

fn cmd_wire_probe(args: &[String]) -> i32 {
    let mut host = "127.0.0.1".to_string();
    let mut port: u16 = 8775;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--host" => { i += 1; if let Some(v) = args.get(i) { host = v.clone(); } }
            "--port" => { i += 1; if let Some(v) = args.get(i) { port = v.parse().unwrap_or(8775); } }
            _ => {}
        }
        i += 1;
    }

    let get_req = format!("GET /health HTTP/1.1\r\nHost: {host}:{port}\r\nConnection: close\r\n\r\n");
    match raw_tcp_http_request(&host, port, &get_req) {
        Ok((ms, resp)) => {
            println!("[tcp] GET /health -> {} ({ms:.2} ms)", resp.lines().next().unwrap_or(""));
        }
        Err(e) => { eprintln!("[tcp] {e}"); return 1; }
    }

    let rpc_body = r#"{"jsonrpc":"2.0","id":"rust-probe","method":"tools/list"}"#;
    let post_req = format!(
        "POST /mcp HTTP/1.1\r\nHost: {host}:{port}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{rpc_body}",
        rpc_body.len()
    );
    match raw_tcp_http_request(&host, port, &post_req) {
        Ok((ms, resp)) => {
            println!("[tcp] POST /mcp tools/list -> {} ({ms:.2} ms)", resp.lines().next().unwrap_or(""));
            if let Some((_, body)) = resp.split_once("\r\n\r\n") {
                println!("{body}");
            }
            0
        }
        Err(e) => { eprintln!("[tcp] {e}"); 1 }
    }
}

fn cmd_doctor() -> i32 {
    println!("work121 native binary v{VERSION}");
    match find_work121_root() {
        Some(root) => {
            let py = find_python(&root);
            let web_ok = root.join("web").join("index.html").is_file();
            let runs_count = fs::read_dir(root.join("eval_runs"))
                .map(|it| it.filter_map(Result::ok).filter(|e| e.path().extension().and_then(|s| s.to_str()) == Some("jsonl")).count())
                .unwrap_or(0);
            let corpus_count = fs::read_dir(root.join("corpus"))
                .map(|it| it.filter_map(Result::ok).filter(|e| e.path().extension().and_then(|s| s.to_str()) == Some("md")).count())
                .unwrap_or(0);
            println!("work121_root:   {}", root.display());
            println!("python_exe:     {} (exists={})", py.display(), py.exists());
            println!("web_ui_ready:   {web_ok}");
            println!("saved_runs:     {runs_count} jsonl");
            println!("corpus_docs:    {corpus_count} md");
            let mcp_up = raw_tcp_http_request("127.0.0.1", 8775, "GET /health HTTP/1.1\r\nHost: 127.0.0.1:8775\r\nConnection: close\r\n\r\n").is_ok();
            let ollama_up = raw_tcp_http_request("127.0.0.1", 11434, "GET /api/tags HTTP/1.1\r\nHost: 127.0.0.1:11434\r\nConnection: close\r\n\r\n").is_ok();
            println!("server_8775:    {}", if mcp_up { "ONLINE" } else { "offline" });
            println!("ollama_11434:   {}", if ollama_up { "ONLINE" } else { "offline" });
            0
        }
        None => {
            eprintln!("work121_root:   NOT FOUND (set WORK121_HOME)");
            1
        }
    }
}

fn cmd_serve_or_ui(args: &[String], open_ui: bool) -> i32 {
    let root = match find_work121_root() {
        Some(r) => r,
        None => {
            eprintln!("error: could not locate work121 root (set WORK121_HOME)");
            return 2;
        }
    };
    let python = find_python(&root);
    let mut host = "127.0.0.1".to_string();
    let mut port: u16 = 8775;
    let mut no_browser = !open_ui;
    let mut pass_args: Vec<String> = Vec::new();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--host" => {
                if let Some(v) = args.get(i + 1) {
                    host = v.clone();
                    pass_args.extend(["--host".to_string(), v.clone()]);
                }
                i += 2;
            }
            "--port" => {
                if let Some(v) = args.get(i + 1) {
                    port = v.parse().unwrap_or(8775);
                    pass_args.extend(["--port".to_string(), v.clone()]);
                }
                i += 2;
            }
            "--no-browser" => {
                no_browser = true;
                i += 1;
            }
            other => {
                pass_args.push(other.to_string());
                i += 1;
            }
        }
    }

    let mut cmd = Command::new(&python);
    cmd.arg(root.join("backend_server.py"))
        .args(&pass_args)
        .current_dir(&root)
        .stdin(Stdio::inherit())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit());

    let mut child = match cmd.spawn() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("error: failed to launch backend_server.py via {}: {e}", python.display());
            return 2;
        }
    };

    if !no_browser {
        let probe_host = if host == "0.0.0.0" { "127.0.0.1".to_string() } else { host.clone() };
        let url = format!("http://{probe_host}:{port}/");
        thread::spawn(move || {
            let get_req = format!("GET /health HTTP/1.1\r\nHost: {probe_host}:{port}\r\nConnection: close\r\n\r\n");
            for _ in 0..40 {
                if let Ok((_, resp)) = raw_tcp_http_request(&probe_host, port, &get_req) {
                    if resp.contains("200 OK") {
                        open_browser(&url);
                        break;
                    }
                }
                thread::sleep(Duration::from_millis(200));
            }
        });
    }

    match child.wait() {
        Ok(st) => st.code().unwrap_or(1),
        Err(e) => { eprintln!("error waiting for server: {e}"); 1 }
    }
}

fn delegate_python(prefix: &[&str], args: &[String]) -> i32 {
    let root = match find_work121_root() {
        Some(r) => r,
        None => {
            eprintln!("error: could not locate work121 root (set WORK121_HOME)");
            return 2;
        }
    };
    let python = find_python(&root);
    let mut cmd = Command::new(&python);
    for p in prefix {
        cmd.arg(p);
    }
    cmd.args(args)
        .current_dir(&root)
        .stdin(Stdio::inherit())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit());
    match cmd.status() {
        Ok(st) => st.code().unwrap_or(1),
        Err(e) => { eprintln!("error launching Python command: {e}"); 2 }
    }
}

fn print_usage() {
    println!(
        "work121 native binary gate & workbench orchestrator (v{VERSION})

USAGE:
  work121 <COMMAND> [ARGS...]

NATIVE BINARY COMMANDS (zero Python required):
  wire-check   Validate raw/length_prefixed/ndjson bytes with native Rust HardenedBoundaryGate
  wire-batch   Batch-validate newline-delimited base64 wire frames from stdin
  wire-hex     Print canonical 16-byte hex+ASCII wire dump, SHA-256, and frame dissection
  wire-bench   Run native compiled wire-gate latency & throughput microbenchmark
  wire-probe   Probe live server over raw TCP socket (GET /health + POST /mcp tools/list)
  doctor       Inspect workspace root, Python venv, Web UI, Ollama, and server status
  version      Print native binary version JSON

UNIFIED WORKBENCH & PYTHON SUITE COMMANDS:
  ui           Start unified backend + Web UI and open browser (http://127.0.0.1:8775/)
  serve        Start unified backend + Web UI + MCP server headless
  streamlit    Launch 7-tab Streamlit workbench (dashboard.py)
  run          Execute CLI validation run (cli.py run)
  diff         Compare JSONL runs for regressions/drift (cli.py diff)
  stats        Run exact Fisher/McNemar significance tests (cli.py stats)
  audit-owasp  Generate OWASP LLM Top 10 remediation report (cli.py audit-owasp)
  live         Run live cross-model evaluation (run_live_eval.py)
  wheel        Run ModelHamsterWheel admission gauntlet (hamster_wheel.py)
  robustness   Run agent_comm_robustness wire parser suite
  mcp          Run standalone MCP HTTP server (mcp_http_server.py)"
    );
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 || matches!(args[1].as_str(), "-h" | "--help" | "help") {
        print_usage();
        process::exit(0);
    }
    if matches!(args[1].as_str(), "version" | "--version" | "-V") {
        println!("{{\"name\":\"work121-native\",\"version\":\"{VERSION}\"}}");
        return;
    }
    let subcmd = args[1].as_str();
    let rest = &args[2..];
    let code = match subcmd {
        "wire-check" | "wire-batch" => cmd_wire_check_or_batch(subcmd, rest),
        "wire-hex" => cmd_wire_hex(rest),
        "wire-bench" => cmd_wire_bench(rest),
        "wire-probe" => cmd_wire_probe(rest),
        "doctor" => cmd_doctor(),
        "ui" => cmd_serve_or_ui(rest, true),
        "serve" => cmd_serve_or_ui(rest, false),
        "streamlit" => delegate_python(&["-m", "streamlit", "run", "dashboard.py"], rest),
        "run" | "diff" | "stats" | "audit-owasp" => delegate_python(&["cli.py", subcmd], rest),
        "live" => delegate_python(&["run_live_eval.py"], rest),
        "wheel" => delegate_python(&["hamster_wheel.py"], rest),
        "robustness" => delegate_python(&["-m", "agent_comm_robustness"], rest),
        "mcp" => delegate_python(&["mcp_http_server.py"], rest),
        _ => {
            eprintln!("Unknown subcommand: {subcmd}");
            print_usage();
            2
        }
    };
    process::exit(code);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hmac_sha256_rfc4231_vector1() {
        let key = [0x0bu8; 20];
        let data = b"Hi There";
        let mac = hmac_sha256_hex(&key, data);
        assert_eq!(mac, "b0344c61d8db38535ca8afceaf0bf12b881dc200c9833da726e9376c2e32cff7");
    }

    #[test]
    fn test_rejects_duplicate_key_and_lone_surrogate() {
        let cfg = GateConfig::default();
        let dup = br#"{"msg_id":"m1","sender":"a","recipient":"b","round_idx":0,"performative":"propose","claim":"ok","answer":1,"confidence":0.5,"provenance":"assumed","provenance":"observed"}"#;
        let out_dup = parse_bytes(dup, &cfg);
        assert!(!out_dup.accepted);
        assert_eq!(out_dup.error_code, Some("duplicate_json_key"));

        let surr = br#"{"msg_id":"m1","sender":"a","recipient":"b","round_idx":0,"performative":"propose","claim":"bad\ud800s","answer":1,"confidence":0.5,"provenance":"derived"}"#;
        let out_surr = parse_bytes(surr, &cfg);
        assert!(!out_surr.accepted);
        assert_eq!(out_surr.error_code, Some("invalid_surrogate_in_payload"));
    }

    #[test]
    fn test_length_prefixed_positive_derived() {
        let cfg = GateConfig {
            framing: FramingMode::LengthPrefixed,
            ..Default::default()
        };
        let body = br#"{"answer":42,"claim":"Derived result is 42","confidence":0.75,"constraints":["integer_only"],"evidence_refs":["step-1"],"msg_id":"msg-drv-001","oracle_sig":null,"performative":"propose","provenance":"derived","recipient":"model_b","round_idx":0,"sender":"model_a"}"#;
        let mut framed = (body.len() as u32).to_be_bytes().to_vec();
        framed.extend_from_slice(body);
        let out = parse_bytes(&framed, &cfg);
        assert!(out.accepted, "unexpected failure: {:?}", out);
    }

    #[test]
    fn test_base64_decode_roundtrip() {
        assert_eq!(base64_decode("").unwrap(), b"");
        assert_eq!(base64_decode("SGVsbG8h").unwrap(), b"Hello!");
        assert_eq!(base64_decode("Zm9v").unwrap(), b"foo");
        assert_eq!(base64_decode("Zm8=").unwrap(), b"fo");
        assert_eq!(base64_decode("Zg==").unwrap(), b"f");
    }
}
