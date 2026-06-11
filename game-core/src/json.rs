//! Minimal JSON string helpers for the game-private payloads of
//! [`crate::GameUi::view_data`] / [`crate::GameUi::transition_data`].
//!
//! Games build those payloads with `format!` — fine for numbers and fixed
//! keys, but any interpolated *string* must go through [`string`] or the
//! first apostrophe-free assumption that breaks corrupts the payload. This
//! is intentionally not a serializer; reach for serde the day a game's
//! schema outgrows hand-assembly.

/// `s` as a JSON string literal: quoted, with `"`, `\`, and control
/// characters escaped.
pub fn string(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// `Some(s)` as a JSON string literal, `None` as `null`.
pub fn string_or_null(s: Option<impl AsRef<str>>) -> String {
    match s {
        Some(s) => string(s.as_ref()),
        None => "null".into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn escapes_quotes_backslashes_and_controls() {
        assert_eq!(string("a\"b\\c\nd"), r#""a\"b\\c\nd""#);
        assert_eq!(string("\u{1}"), "\"\\u0001\"");
        assert_eq!(string_or_null(None::<&str>), "null");
    }
}
