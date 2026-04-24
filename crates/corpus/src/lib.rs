//! corpus — JSONL staging + promote + dedupe for matt-voice training data.
//!
//! Schema mirrors the Python `2_build_corpus.py` so new voice samples can
//! merge directly with the existing Discord corpus:
//!
//! { "context": "...", "matt": "...", "source": "voice-solo" | "voice-group"
//!                                             | "discord", "sha": "<hex>" }
//!
//! - `source` lets the training pipeline upweight voice samples later.
//! - `sha` is sha256(matt + "\x00" + context[:256]) — identical input = same hash.

use anyhow::{Context, Result};
use chrono::Utc;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use tracing::{info, warn};
use whisper_local::Utterance;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingPair {
    pub context: String,
    pub matt: String,
    pub source: String,
    pub sha: String,
    pub avg_logprob: Option<f32>,
    pub captured_at: Option<String>,
}

impl TrainingPair {
    pub fn new(context: String, matt: String, source: impl Into<String>) -> Self {
        let sha = hash(&matt, &context);
        Self {
            context,
            matt,
            source: source.into(),
            sha,
            avg_logprob: None,
            captured_at: Some(Utc::now().to_rfc3339()),
        }
    }
}

fn hash(matt: &str, context: &str) -> String {
    let mut h = Sha256::new();
    h.update(matt.as_bytes());
    h.update(b"\x00");
    let ctx_head: String = context.chars().take(256).collect();
    h.update(ctx_head.as_bytes());
    hex::encode(h.finalize())
}

/// Convert a stream of whisper utterances into training pairs by using each
/// utterance's previous N utterances as context.
pub fn utterances_to_pairs(
    utts: &[Utterance],
    source: &str,
    context_window: usize,
) -> Vec<TrainingPair> {
    let mut pairs = Vec::with_capacity(utts.len());
    for (i, u) in utts.iter().enumerate() {
        let start = i.saturating_sub(context_window);
        let ctx: String = utts[start..i]
            .iter()
            .map(|p| p.text.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        let mut p = TrainingPair::new(ctx, u.text.clone(), source);
        p.avg_logprob = Some(u.avg_logprob);
        pairs.push(p);
    }
    pairs
}

pub fn write_jsonl(path: impl AsRef<Path>, pairs: &[TrainingPair]) -> Result<()> {
    let path = path.as_ref();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    let f = File::create(path).with_context(|| format!("create {path:?}"))?;
    let mut w = BufWriter::new(f);
    for p in pairs {
        serde_json::to_writer(&mut w, p)?;
        w.write_all(b"\n")?;
    }
    w.flush()?;
    info!(?path, n = pairs.len(), "wrote jsonl");
    Ok(())
}

pub fn read_jsonl(path: impl AsRef<Path>) -> Result<Vec<TrainingPair>> {
    let f = File::open(path.as_ref())?;
    let r = BufReader::new(f);
    let mut out = Vec::new();
    for line in r.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let p: TrainingPair = serde_json::from_str(&line)
            .with_context(|| format!("parse line: {line}"))?;
        out.push(p);
    }
    Ok(out)
}

/// Merge `staging` into `master`, deduped by SHA. Returns count of new pairs.
pub fn promote(staging: &Path, master: &Path) -> Result<usize> {
    let new_pairs = read_jsonl(staging).with_context(|| format!("read {staging:?}"))?;
    let existing: HashSet<String> = if master.exists() {
        read_jsonl(master)?.into_iter().map(|p| p.sha).collect()
    } else {
        HashSet::new()
    };

    let mut added = 0usize;
    let mut f = OpenOptions::new()
        .create(true)
        .append(true)
        .open(master)
        .with_context(|| format!("open {master:?}"))?;
    for p in new_pairs {
        if existing.contains(&p.sha) {
            continue;
        }
        serde_json::to_writer(&mut f, &p)?;
        f.write_all(b"\n")?;
        added += 1;
    }
    info!(?staging, ?master, added, "promote done");
    Ok(added)
}

/// Default layout under /j/matt-voice (or cnc equivalent).
#[derive(Debug, Clone)]
pub struct Layout {
    pub root: PathBuf,
}

impl Layout {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }
    pub fn staging_dir(&self) -> PathBuf {
        self.root.join("training-data/staging")
    }
    pub fn master_voice(&self) -> PathBuf {
        self.root.join("training-data/matt-voice-voice.jsonl")
    }
    pub fn staging_file(&self, source: &str) -> PathBuf {
        let ts = Utc::now().format("%Y-%m-%d_%H%M%S");
        self.staging_dir()
            .join(format!("{ts}_{source}.jsonl"))
    }
}

/// Gate for the eventual auto-promote step (phase 2). Called by `corpus review`
/// for now; becomes an auto-trigger later.
pub fn auto_promote_ok(pairs: &[TrainingPair]) -> bool {
    if pairs.len() < 20 {
        warn!(n = pairs.len(), "fewer than 20 utterances, staging only");
        return false;
    }
    let avg_lp = pairs
        .iter()
        .filter_map(|p| p.avg_logprob)
        .sum::<f32>()
        / pairs.len().max(1) as f32;
    let ok = avg_lp > -0.5;
    info!(avg_logprob = avg_lp, ok, "auto-promote gate");
    ok
}
