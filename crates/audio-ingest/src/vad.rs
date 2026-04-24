//! Simple energy-threshold voice-activity segmentation.
//!
//! Not webrtc-grade — good enough to chop a 60-minute Audacity capture or
//! Craig flac into utterance boundaries for whisper. For the group-call case
//! we rely on per-speaker track separation upstream (Craig does this), so
//! cross-talk isn't a failure mode for the VAD itself.

use crate::{Audio, WHISPER_SR};
use tracing::debug;

/// Minimum gap (seconds) of sub-threshold audio that splits utterances.
pub const DEFAULT_MIN_SILENCE_SECS: f32 = 1.2;
/// Minimum utterance length we keep (shorter = drop, likely breath/click).
pub const DEFAULT_MIN_UTTERANCE_SECS: f32 = 0.8;

#[derive(Debug, Clone, Copy)]
pub struct Segment {
    pub start_sample: usize,
    pub end_sample: usize,
}

impl Segment {
    pub fn start_secs(&self) -> f32 {
        self.start_sample as f32 / WHISPER_SR as f32
    }
    pub fn end_secs(&self) -> f32 {
        self.end_sample as f32 / WHISPER_SR as f32
    }
}

/// Split audio into utterance segments using energy + silence-gap heuristic.
pub fn segment(
    audio: &Audio,
    min_silence_secs: f32,
    min_utterance_secs: f32,
) -> Vec<Segment> {
    assert_eq!(audio.sr, WHISPER_SR, "audio must be resampled to WHISPER_SR");

    let frame_len = (audio.sr as f32 * 0.030) as usize; // 30ms frames
    let silence_frames = (min_silence_secs / 0.030) as usize;
    let min_utt_frames = (min_utterance_secs / 0.030) as usize;

    let rms_threshold = adaptive_rms_threshold(&audio.samples, frame_len);
    debug!(rms_threshold, "adaptive VAD threshold");

    let mut voiced: Vec<bool> = Vec::with_capacity(audio.samples.len() / frame_len + 1);
    for chunk in audio.samples.chunks(frame_len) {
        let rms = rms(chunk);
        voiced.push(rms > rms_threshold);
    }

    let mut segments = Vec::new();
    let mut cur_start: Option<usize> = None;
    let mut trailing_silence = 0usize;

    for (i, &v) in voiced.iter().enumerate() {
        match (cur_start, v) {
            (None, true) => {
                cur_start = Some(i);
                trailing_silence = 0;
            }
            (Some(_), true) => {
                trailing_silence = 0;
            }
            (Some(start), false) => {
                trailing_silence += 1;
                if trailing_silence >= silence_frames {
                    let end = i - trailing_silence;
                    if end - start >= min_utt_frames {
                        segments.push(Segment {
                            start_sample: start * frame_len,
                            end_sample: end * frame_len,
                        });
                    }
                    cur_start = None;
                    trailing_silence = 0;
                }
            }
            (None, false) => {}
        }
    }
    if let Some(start) = cur_start {
        let end = voiced.len() - trailing_silence;
        if end - start >= min_utt_frames {
            segments.push(Segment {
                start_sample: start * frame_len,
                end_sample: end * frame_len,
            });
        }
    }

    debug!(n_segments = segments.len(), "segmentation done");
    segments
}

fn rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let ss: f32 = samples.iter().map(|s| s * s).sum();
    (ss / samples.len() as f32).sqrt()
}

/// Adaptive threshold = noise-floor (10th percentile of per-frame RMS) × 2.5.
fn adaptive_rms_threshold(samples: &[f32], frame_len: usize) -> f32 {
    let mut rmses: Vec<f32> = samples.chunks(frame_len).map(rms).collect();
    if rmses.is_empty() {
        return 0.01;
    }
    rmses.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let noise_floor = rmses[rmses.len() / 10];
    (noise_floor * 2.5).max(0.005)
}
