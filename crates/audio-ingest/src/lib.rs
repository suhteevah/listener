//! audio-ingest — decode WAV/FLAC/MP3/etc. and produce 16kHz mono f32 samples
//! ready for whisper-rs. Also provides energy-based VAD segmentation so we can
//! split long recordings into per-utterance chunks before transcription.

use anyhow::{Context, Result};
use std::path::Path;
use tracing::{debug, info};

pub mod vad;

/// Target sample rate for whisper.
pub const WHISPER_SR: u32 = 16_000;

/// A mono-PCM audio buffer at [`WHISPER_SR`].
#[derive(Debug, Clone)]
pub struct Audio {
    pub samples: Vec<f32>,
    pub sr: u32,
}

impl Audio {
    pub fn duration_secs(&self) -> f32 {
        self.samples.len() as f32 / self.sr as f32
    }
}

/// Decode any supported format into 16kHz mono f32 via symphonia + rubato.
pub fn decode_to_whisper_input<P: AsRef<Path>>(path: P) -> Result<Audio> {
    let path = path.as_ref();
    info!(?path, "decoding audio file");

    let src = std::fs::File::open(path).with_context(|| format!("open {path:?}"))?;
    let mss = symphonia::core::io::MediaSourceStream::new(Box::new(src), Default::default());

    let hint = {
        let mut h = symphonia::core::probe::Hint::new();
        if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
            h.with_extension(ext);
        }
        h
    };
    let probed = symphonia::default::get_probe()
        .format(&hint, mss, &Default::default(), &Default::default())
        .context("symphonia probe")?;
    let mut format = probed.format;
    let track = format
        .default_track()
        .context("no default audio track")?
        .clone();

    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &Default::default())
        .context("codec init")?;

    let src_sr = track.codec_params.sample_rate.context("no sample rate")?;
    let channels = track
        .codec_params
        .channels
        .context("no channel map")?
        .count();

    let mut mono = Vec::<f32>::new();
    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(symphonia::core::errors::Error::IoError(e))
                if e.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                break;
            }
            Err(e) => return Err(e).context("decode"),
        };
        let decoded = decoder.decode(&packet).context("decoder.decode")?;
        append_mono_f32(&decoded, channels, &mut mono);
    }
    debug!(decoded_samples = mono.len(), src_sr, channels, "decode done");

    if src_sr == WHISPER_SR {
        return Ok(Audio {
            samples: mono,
            sr: WHISPER_SR,
        });
    }

    let resampled = resample(&mono, src_sr, WHISPER_SR)?;
    Ok(Audio {
        samples: resampled,
        sr: WHISPER_SR,
    })
}

fn append_mono_f32(
    buf: &symphonia::core::audio::AudioBufferRef,
    channels: usize,
    out: &mut Vec<f32>,
) {
    use symphonia::core::audio::{AudioBufferRef, Signal};
    macro_rules! mix_channels {
        ($buf:expr) => {{
            let spec = $buf.spec();
            let frames = $buf.frames();
            let ch = spec.channels.count();
            for f in 0..frames {
                let mut acc = 0.0f32;
                for c in 0..ch {
                    let s = $buf.chan(c)[f];
                    acc += s as f32;
                }
                out.push(acc / ch as f32);
            }
        }};
    }
    let _ = channels;
    match buf {
        AudioBufferRef::F32(b) => mix_channels!(b),
        AudioBufferRef::F64(b) => {
            let spec = b.spec();
            let frames = b.frames();
            let ch = spec.channels.count();
            for f in 0..frames {
                let mut acc = 0.0f32;
                for c in 0..ch {
                    acc += b.chan(c)[f] as f32;
                }
                out.push(acc / ch as f32);
            }
        }
        AudioBufferRef::S16(b) => {
            let spec = b.spec();
            let frames = b.frames();
            let ch = spec.channels.count();
            for f in 0..frames {
                let mut acc = 0.0f32;
                for c in 0..ch {
                    acc += (b.chan(c)[f] as f32) / i16::MAX as f32;
                }
                out.push(acc / ch as f32);
            }
        }
        AudioBufferRef::S32(b) => {
            let spec = b.spec();
            let frames = b.frames();
            let ch = spec.channels.count();
            for f in 0..frames {
                let mut acc = 0.0f32;
                for c in 0..ch {
                    acc += (b.chan(c)[f] as f32) / i32::MAX as f32;
                }
                out.push(acc / ch as f32);
            }
        }
        AudioBufferRef::U8(b) => {
            let spec = b.spec();
            let frames = b.frames();
            let ch = spec.channels.count();
            for f in 0..frames {
                let mut acc = 0.0f32;
                for c in 0..ch {
                    acc += ((b.chan(c)[f] as f32) - 128.0) / 128.0;
                }
                out.push(acc / ch as f32);
            }
        }
        _ => {
            // other bit-depths: skip-but-warn rather than panic, rare for our inputs
            tracing::warn!("unsupported symphonia sample format, skipping packet");
        }
    }
}

fn resample(samples: &[f32], from_sr: u32, to_sr: u32) -> Result<Vec<f32>> {
    use rubato::{Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction};
    let params = SincInterpolationParameters {
        sinc_len: 256,
        f_cutoff: 0.95,
        interpolation: SincInterpolationType::Linear,
        oversampling_factor: 160,
        window: WindowFunction::BlackmanHarris2,
    };
    let ratio = to_sr as f64 / from_sr as f64;
    let mut resampler =
        SincFixedIn::<f32>::new(ratio, 2.0, params, samples.len(), 1).context("build resampler")?;
    let out = resampler
        .process(&[samples.to_vec()], None)
        .context("resample process")?;
    Ok(out.into_iter().next().unwrap_or_default())
}
