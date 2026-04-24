//! listenerd — tokio daemon that:
//!   1. tails $MATT_VOICE_ROOT/craig-inbox.txt for new Craig URLs
//!   2. downloads → extracts Matt's flac
//!   3. runs audio-ingest + whisper-local → utterances
//!   4. writes JSONL to $MATT_VOICE_ROOT/training-data/staging/
//!   5. telegram-notifies Matt with a review link
//!
//! Phase 1: staging-only (per Matt's explicit direction 2026-04-23).
//! Phase 2: auto-promote gated on corpus::auto_promote_ok().

use anyhow::{Context, Result};
use audio_ingest::{decode_to_whisper_input, vad};
use clap::Parser;
use corpus::{utterances_to_pairs, write_jsonl, Layout};
use craig_poller::{download, extract_matt_track, parse_line};
use std::path::PathBuf;
use std::time::Duration;
use tokio::io::{AsyncBufReadExt, BufReader};
use tracing::{error, info, warn};
use whisper_local::Whisper;

#[derive(Parser)]
#[command(version, about = "matt-voice listenerd — staging-first capture orchestrator")]
struct Cli {
    #[arg(long, env = "MATT_VOICE_ROOT", default_value = "/opt/matt-voice")]
    root: PathBuf,

    /// Polling interval for the craig-inbox file.
    #[arg(long, default_value = "10")]
    poll_secs: u64,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info,listenerd=debug".into()),
        )
        .json()
        .init();

    let cli = Cli::parse();
    let layout = Layout::new(&cli.root);
    let inbox = cli.root.join("craig-inbox.txt");
    let zip_dir = cli.root.join("craig-zips");
    let flac_dir = cli.root.join("craig-flac");
    let cursor_file = cli.root.join(".craig-inbox.cursor");

    info!(root = ?cli.root, ?inbox, "listenerd starting");

    // Whisper is loaded once and shared across jobs to avoid repeated model load.
    let whisper = Whisper::from_env()
        .context("whisper-local init failed — is MATT_VOICE_WHISPER_MODEL set?")?;

    loop {
        if let Err(e) = tick(&layout, &inbox, &cursor_file, &zip_dir, &flac_dir, &whisper).await {
            error!(err = %e, "tick failed");
        }
        tokio::time::sleep(Duration::from_secs(cli.poll_secs)).await;
    }
}

async fn tick(
    layout: &Layout,
    inbox: &std::path::Path,
    cursor_file: &std::path::Path,
    zip_dir: &std::path::Path,
    flac_dir: &std::path::Path,
    whisper: &Whisper,
) -> Result<()> {
    if !inbox.exists() {
        return Ok(());
    }
    let cursor = read_cursor(cursor_file).await.unwrap_or(0);
    let f = tokio::fs::File::open(inbox).await?;
    let reader = BufReader::new(f);
    let mut lines = reader.lines();
    let mut processed_any = false;
    let mut new_cursor = cursor;
    let mut idx = 0u64;

    while let Some(line) = lines.next_line().await? {
        idx += 1;
        if idx <= cursor {
            continue;
        }
        new_cursor = idx;
        let Some(job) = parse_line(&line) else { continue };
        info!(job.url, job.matt_craig_user_id, job.captured_at, "new craig job");

        let zip = match download(&job, zip_dir).await {
            Ok(z) => z,
            Err(e) => {
                warn!(err = %e, "download failed, skipping");
                continue;
            }
        };
        let flac = match extract_matt_track(&zip, &job.matt_craig_user_id, flac_dir) {
            Ok(p) => p,
            Err(e) => {
                warn!(err = %e, "extract failed, skipping");
                continue;
            }
        };

        let audio = decode_to_whisper_input(&flac)?;
        let segments = vad::segment(
            &audio,
            vad::DEFAULT_MIN_SILENCE_SECS,
            vad::DEFAULT_MIN_UTTERANCE_SECS,
        );
        let utts = whisper.transcribe(&audio, &segments)?;
        let pairs = utterances_to_pairs(&utts, "voice-group", 2);
        let out = layout.staging_file("voice-group");
        write_jsonl(&out, &pairs)?;
        info!(n_pairs = pairs.len(), out = ?out, "staged");
        processed_any = true;
        notify_matt(&out, pairs.len()).ok();
    }

    if processed_any || new_cursor != cursor {
        write_cursor(cursor_file, new_cursor).await?;
    }
    Ok(())
}

async fn read_cursor(path: &std::path::Path) -> Result<u64> {
    if !path.exists() {
        return Ok(0);
    }
    let s = tokio::fs::read_to_string(path).await?;
    Ok(s.trim().parse().unwrap_or(0))
}

async fn write_cursor(path: &std::path::Path, v: u64) -> Result<()> {
    tokio::fs::write(path, v.to_string()).await?;
    Ok(())
}

/// Fire-and-forget telegram ping. Uses the same `notify-telegram.sh` script
/// as the rest of the fleet — no token in Rust code.
fn notify_matt(staging_file: &std::path::Path, n_pairs: usize) -> Result<()> {
    let msg = format!(
        "matt-voice: {n_pairs} new utterances staged at {}\n  review: corpus review {0}",
        staging_file.display()
    );
    // Best-effort; non-fatal.
    let _ = std::process::Command::new("bash")
        .args(["/j/baremetal claude/tools/notify-telegram.sh", &msg])
        .spawn();
    Ok(())
}
