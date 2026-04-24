//! `corpus` CLI — ingest audacity captures, review staging files, promote to master.

use anyhow::{Context, Result};
use audio_ingest::{decode_to_whisper_input, vad};
use clap::{Parser, Subcommand};
use corpus::{auto_promote_ok, promote, read_jsonl, utterances_to_pairs, write_jsonl, Layout};
use std::path::PathBuf;
use tracing::info;
use whisper_local::Whisper;

#[derive(Parser)]
#[command(version, about = "matt-voice corpus tool")]
struct Cli {
    /// Root of the matt-voice project (default: /j/matt-voice, override on cnc).
    #[arg(long, env = "MATT_VOICE_ROOT", default_value = "/j/matt-voice")]
    root: PathBuf,

    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Decode + segment + transcribe an Audacity capture, write to staging.
    Ingest {
        /// Path to wav/flac/mp3.
        input: PathBuf,
        /// "voice-solo" for scripted reads, "voice-group" for multi-speaker.
        #[arg(long, default_value = "voice-solo")]
        source: String,
    },
    /// Print a summary + random-sample of a staging file.
    Review {
        file: PathBuf,
        #[arg(long, default_value_t = 10)]
        n: usize,
    },
    /// Merge a staging file into the master voice corpus (SHA-deduped).
    Promote { file: PathBuf },
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info,audio_ingest=debug,corpus=debug".into()),
        )
        .init();

    let cli = Cli::parse();
    let layout = Layout::new(&cli.root);

    match cli.cmd {
        Cmd::Ingest { input, source } => cmd_ingest(&layout, &input, &source),
        Cmd::Review { file, n } => cmd_review(&file, n),
        Cmd::Promote { file } => cmd_promote(&layout, &file),
    }
}

fn cmd_ingest(layout: &Layout, input: &std::path::Path, source: &str) -> Result<()> {
    info!(?input, source, "ingest start");
    let audio = decode_to_whisper_input(input)?;
    info!(dur = audio.duration_secs(), "decoded");

    let segments = vad::segment(
        &audio,
        vad::DEFAULT_MIN_SILENCE_SECS,
        vad::DEFAULT_MIN_UTTERANCE_SECS,
    );
    info!(n_segments = segments.len(), "segmented");

    let whisper = Whisper::from_env().context(
        "MATT_VOICE_WHISPER_MODEL must point at ggml-large-v3.bin (or quant)",
    )?;
    let utts = whisper.transcribe(&audio, &segments)?;
    info!(n_utts = utts.len(), "transcribed");

    let pairs = utterances_to_pairs(&utts, source, 2);
    let out = layout.staging_file(source);
    write_jsonl(&out, &pairs)?;
    println!("staged {} pairs → {}", pairs.len(), out.display());
    println!("run:  corpus review {}", out.display());
    Ok(())
}

fn cmd_review(file: &std::path::Path, n: usize) -> Result<()> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let pairs = read_jsonl(file)?;
    let n = n.min(pairs.len());
    let mut idx: Vec<usize> = (0..pairs.len()).collect();
    idx.sort_by_key(|i| {
        let mut h = DefaultHasher::new();
        pairs[*i].sha.hash(&mut h);
        h.finish()
    });

    println!("== {} pairs in {} ==", pairs.len(), file.display());
    println!("auto-promote gate: {}", if auto_promote_ok(&pairs) { "PASS" } else { "FAIL" });
    let avg_lp: f32 = pairs
        .iter()
        .filter_map(|p| p.avg_logprob)
        .sum::<f32>()
        / pairs.len().max(1) as f32;
    println!("avg logprob: {avg_lp:.3}");
    println!();
    for i in idx.iter().take(n) {
        let p = &pairs[*i];
        let ctx_preview: String = p.context.chars().take(80).collect();
        println!("---");
        println!("  ctx:  {ctx_preview}");
        println!("  matt: {}", p.matt);
        if let Some(lp) = p.avg_logprob {
            println!("  lp:   {lp:.3}");
        }
    }
    Ok(())
}

fn cmd_promote(layout: &Layout, file: &std::path::Path) -> Result<()> {
    let master = layout.master_voice();
    let added = promote(file, &master)?;
    println!("promoted {added} new pairs → {}", master.display());
    Ok(())
}
