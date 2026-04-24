//! whisper-local — thin wrapper over whisper-rs.
//!
//! Default build: CPU inference. With `--features cuda` (enabled on cnc once
//! the P100 cables land), routes to CUDA. With `--features metal`, macOS.
//!
//! The model (ggml-large-v3.bin or q5_0 quant) is expected at
//! `$MATT_VOICE_WHISPER_MODEL`. We don't download it automatically — user
//! puts it on the machine once.

use anyhow::{Context, Result};
use audio_ingest::{Audio, vad::Segment, WHISPER_SR};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use tracing::{debug, info};
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Utterance {
    pub start_secs: f32,
    pub end_secs: f32,
    pub text: String,
    /// Mean per-token log-probability — used as a cheap confidence proxy for
    /// the auto-promote gate downstream.
    pub avg_logprob: f32,
}

pub struct Whisper {
    ctx: WhisperContext,
}

impl Whisper {
    pub fn from_env() -> Result<Self> {
        let p = std::env::var("MATT_VOICE_WHISPER_MODEL")
            .context("set MATT_VOICE_WHISPER_MODEL to the ggml model path")?;
        Self::new(PathBuf::from(p))
    }

    pub fn new(model_path: impl AsRef<Path>) -> Result<Self> {
        let model_path = model_path.as_ref();
        info!(?model_path, "loading whisper model");
        let params = WhisperContextParameters::default();
        let ctx = WhisperContext::new_with_params(
            model_path.to_str().context("model path must be utf-8")?,
            params,
        )
        .context("load whisper context")?;
        Ok(Self { ctx })
    }

    /// Transcribe a pre-segmented list of utterances, preserving segment
    /// start/end offsets from the source audio.
    pub fn transcribe(&self, audio: &Audio, segments: &[Segment]) -> Result<Vec<Utterance>> {
        assert_eq!(audio.sr, WHISPER_SR);
        let mut out = Vec::with_capacity(segments.len());
        for seg in segments {
            let clip = &audio.samples[seg.start_sample..seg.end_sample.min(audio.samples.len())];
            if clip.is_empty() {
                continue;
            }
            let (text, avg_logprob) = self.transcribe_clip(clip)?;
            debug!(
                start = seg.start_secs(),
                end = seg.end_secs(),
                avg_logprob,
                "segment transcribed"
            );
            out.push(Utterance {
                start_secs: seg.start_secs(),
                end_secs: seg.end_secs(),
                text,
                avg_logprob,
            });
        }
        Ok(out)
    }

    fn transcribe_clip(&self, clip: &[f32]) -> Result<(String, f32)> {
        let mut state = self.ctx.create_state().context("create whisper state")?;
        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        params.set_language(Some("en"));
        params.set_print_special(false);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);
        params.set_translate(false);
        params.set_single_segment(false);

        state.full(params, clip).context("whisper full pass")?;
        let n = state.full_n_segments().context("n_segments")?;
        let mut text_parts = Vec::new();
        let mut logprob_acc = 0.0f32;
        let mut token_count = 0u32;
        for i in 0..n {
            text_parts.push(
                state
                    .full_get_segment_text(i)
                    .context("segment text")?,
            );
            let seg_tokens = state.full_n_tokens(i).context("n_tokens")?;
            for t in 0..seg_tokens {
                let td = state.full_get_token_data(i, t).context("token data")?;
                logprob_acc += td.plog;
                token_count += 1;
            }
        }
        let avg = if token_count > 0 {
            logprob_acc / token_count as f32
        } else {
            f32::NEG_INFINITY
        };
        Ok((text_parts.join("").trim().to_string(), avg))
    }
}
