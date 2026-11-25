//! Evidence writing: JSON metadata plus raw little-endian f64 binaries.
//!
//! All tensors are written as raw `f64` arrays (full precision) with their
//! dimensions recorded in the accompanying JSON manifest.

use std::fs;
use std::io;
use std::path::Path;

use serde_json::Value;

pub fn ensure_dir(path: &Path) -> io::Result<()> {
    fs::create_dir_all(path)
}

pub fn write_json(path: &Path, value: &Value) -> io::Result<()> {
    fs::write(path, serde_json::to_string_pretty(value).unwrap())
}

pub fn write_f64_bin(path: &Path, data: &[f64]) -> io::Result<()> {
    let mut bytes = Vec::with_capacity(data.len() * 8);
    for x in data {
        bytes.extend_from_slice(&x.to_le_bytes());
    }
    fs::write(path, bytes)
}

pub fn read_f64_bin(path: &Path) -> io::Result<Vec<f64>> {
    let bytes = fs::read(path)?;
    if bytes.len() % 8 != 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("{} is not a whole number of f64 values", path.display()),
        ));
    }
    Ok(bytes
        .chunks_exact(8)
        .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
        .collect())
}

pub fn read_json(path: &Path) -> io::Result<Value> {
    Ok(serde_json::from_str(&fs::read_to_string(path)?)?)
}

/// UTC timestamp like `2026-11-11T12:34:56Z`.
pub fn utc_timestamp() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    // civil-from-days algorithm (Howard Hinnant)
    let days = (secs / 86400) as i64;
    let rem = secs % 86400;
    let (h, mi, s) = (rem / 3600, (rem % 3600) / 60, rem % 60);
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!("{y:04}-{m:02}-{d:02}T{h:02}:{mi:02}:{s:02}Z")
}
