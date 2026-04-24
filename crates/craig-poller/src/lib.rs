//! craig-poller — watches a configured inbox file for Craig download URLs,
//! fetches the zip, extracts Matt's per-user flac, and emits the flac path
//! so the listenerd orchestrator can kick off whisper on it.
//!
//! The "configured inbox file" is a flat text file that a lightweight Discord
//! bot or Matt himself appends Craig URLs to. We do NOT implement voice
//! receive ourselves — that's py-cord territory and we decided to stay out
//! of it for v1. Craig bot handles capture; this crate handles ingest.
//!
//! Inbox format (one per line):
//!   2026-04-23T22:01:00Z  https://craig.horse/rec/{id}?key={key}  {craig_user_id_for_matt}
//!
//! The craig_user_id lets us pick Matt's track out of the multi-user zip
//! without guessing.

use anyhow::{anyhow, bail, Context, Result};
use regex::Regex;
use std::io::Read;
use std::path::{Path, PathBuf};
use tracing::{info, warn};
use url::Url;

/// Only these hosts may be fetched. The inbox file is in Matt's trust
/// boundary, but adopting an allowlist prevents a compromised inbox file
/// from pivoting this daemon into an SSRF probe of internal services.
const ALLOWED_HOSTS: &[&str] = &["craig.horse", "craig.chat"];

#[derive(Debug, Clone)]
pub struct CraigJob {
    pub url: String,
    pub matt_craig_user_id: String,
    pub captured_at: String,
}

/// Parse one craig-inbox line → job. Returns None for blank/comment lines.
pub fn parse_line(line: &str) -> Option<CraigJob> {
    let line = line.trim();
    if line.is_empty() || line.starts_with('#') {
        return None;
    }
    let mut parts = line.split_whitespace();
    let captured_at = parts.next()?.to_string();
    let url = parts.next()?.to_string();
    let matt_craig_user_id = parts.next()?.to_string();
    Some(CraigJob {
        url,
        matt_craig_user_id,
        captured_at,
    })
}

/// Download a Craig zip to `dest_dir` and return the zip path.
///
/// Hardening:
/// - URL host is allowlist-checked (ALLOWED_HOSTS) → no SSRF pivot via a
///   tampered inbox file.
/// - Output filename is derived from the parsed rec-id regex, not from the
///   URL path directly → no way for a crafted URL to escape `dest_dir`.
pub async fn download(job: &CraigJob, dest_dir: &Path) -> Result<PathBuf> {
    std::fs::create_dir_all(dest_dir)?;

    let url = Url::parse(&job.url).context("parse craig url")?;
    let host = url.host_str().ok_or_else(|| anyhow!("url has no host"))?;
    if !ALLOWED_HOSTS.contains(&host) {
        bail!(
            "refusing to fetch from {host}: not in ALLOWED_HOSTS ({:?})",
            ALLOWED_HOSTS
        );
    }
    if url.scheme() != "https" {
        bail!("refusing non-https craig url: scheme={}", url.scheme());
    }

    let id = extract_rec_id(&job.url).context("parse craig url for rec id")?;
    let out_path = dest_dir.join(format!("craig_{id}.zip"));
    info!(url = %job.url, ?out_path, "downloading craig zip");

    let resp = reqwest::get(&job.url).await?.error_for_status()?;
    let bytes = resp.bytes().await?;
    std::fs::write(&out_path, &bytes)?;
    info!(?out_path, bytes = bytes.len(), "craig zip saved");
    Ok(out_path)
}

fn extract_rec_id(url: &str) -> Result<String> {
    // Character class is [A-Za-z0-9_-] → alphanumeric + underscore + hyphen.
    // No path separators permitted, so the captured id is safe to use in
    // a filename component.
    let re = Regex::new(r"/rec/([A-Za-z0-9_-]+)")?;
    let caps = re
        .captures(url)
        .ok_or_else(|| anyhow!("no /rec/{{id}} in craig url"))?;
    Ok(caps[1].to_string())
}

/// Defense-in-depth path validation: confirms `candidate` is a descendant of
/// `base` after normalization, rejecting any `..` traversal attempts even if
/// earlier layers fail. Returns the validated absolute path.
fn ensure_within(base: &Path, candidate: &Path) -> Result<PathBuf> {
    let base_abs = base
        .canonicalize()
        .with_context(|| format!("canonicalize base {base:?}"))?;
    // The candidate may not exist yet (we're about to create it); canonicalize
    // the parent and re-join to validate.
    let (parent, file) = (
        candidate
            .parent()
            .ok_or_else(|| anyhow!("candidate has no parent"))?,
        candidate
            .file_name()
            .ok_or_else(|| anyhow!("candidate has no file_name"))?,
    );
    let parent_abs = parent
        .canonicalize()
        .with_context(|| format!("canonicalize parent {parent:?}"))?;
    if !parent_abs.starts_with(&base_abs) {
        bail!(
            "path traversal refused: {parent_abs:?} escapes base {base_abs:?}"
        );
    }
    Ok(parent_abs.join(file))
}

/// Extract Matt's track from a Craig zip. Craig puts per-user audio as
/// `{N}-{user-id}.flac` inside the zip. Returns the extracted flac path.
///
/// Zip-slip hardening (belt-and-suspenders):
///   1. Reject zip entries whose name contains a path separator or `..`
///      anywhere — refuse to even consider them.
///   2. Strip to `file_name()` only (drops any directory components).
///   3. Canonicalize the resulting path and assert it descends from
///      `extract_dir` via `ensure_within`.
pub fn extract_matt_track(zip_path: &Path, matt_id: &str, extract_dir: &Path) -> Result<PathBuf> {
    std::fs::create_dir_all(extract_dir)?;

    // matt_id comes from the craig-inbox line Matt controls, but validate
    // anyway — any future path where it flows in from Discord metadata
    // gets the check for free.
    if matt_id.is_empty()
        || matt_id
            .chars()
            .any(|c| !(c.is_ascii_alphanumeric() || c == '-' || c == '_'))
    {
        bail!("invalid matt_id {matt_id:?} — must be [A-Za-z0-9_-]");
    }

    let f = std::fs::File::open(zip_path)?;
    let mut archive = zip::ZipArchive::new(f)?;
    for i in 0..archive.len() {
        let mut entry = archive.by_index(i)?;
        let name = entry.name().to_string();

        // Reject anything suspicious outright — don't even try to sanitize.
        if name.contains("..") || name.contains('\0') {
            warn!(name, "zip entry rejected: suspicious characters");
            continue;
        }
        if !name.contains(matt_id) || !name.ends_with(".flac") {
            continue;
        }

        // Strip to bare filename, then re-validate via canonicalization.
        let bare = Path::new(&name)
            .file_name()
            .ok_or_else(|| anyhow!("zip entry has no file_name: {name}"))?;
        let candidate = extract_dir.join(bare);
        let out_path = ensure_within(extract_dir, &candidate)
            .context("zip entry failed path containment check")?;

        let mut buf = Vec::with_capacity(entry.size() as usize);
        entry.read_to_end(&mut buf)?;
        std::fs::write(&out_path, &buf)?;
        info!(?out_path, "extracted matt track");
        return Ok(out_path);
    }
    warn!(matt_id, zip = ?zip_path, "no matching flac in zip");
    Err(anyhow!("no flac matching matt_id={matt_id} in zip"))
}
