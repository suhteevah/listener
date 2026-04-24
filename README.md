# matt-voice / listener

Rust cargo workspace that turns recorded audio (Audacity scripted reads, Craig multi-user Discord captures) into training-ready JSONL for the matt-voice LoRA.

Lives on cnc-server in systemd. **Staging-first** — every capture lands in `training-data/staging/` for Matt to review before it touches the master corpus. Auto-promote is a phase-2 upgrade gated on whisper confidence.

## Layout

```
listener/
├── Cargo.toml                (workspace)
└── crates/
    ├── audio-ingest/         # decode wav/flac/mp3 → 16kHz mono f32, energy-VAD segmentation
    ├── whisper-local/        # whisper-rs wrapper (CPU default, --features cuda for P100s)
    ├── craig-poller/         # craig-inbox.txt → zip download → per-user flac extract
    ├── corpus/               # JSONL staging, SHA-dedupe, promote CLI
    └── listenerd/            # tokio daemon orchestrating the pipeline on cnc
```

## Build

```bash
# dev (kokonoe, no CUDA)
cargo build --release -p corpus -p listenerd

# cnc (once P100 cables land)
cargo build --release -p corpus -p listenerd --features whisper-local/cuda
```

## Use — offline Audacity reads

```bash
export MATT_VOICE_WHISPER_MODEL=/opt/matt-voice/models/ggml-large-v3.bin

# scripted 2-3 min rants, no diarization needed
./target/release/corpus ingest /j/matt-voice/audio-raw/2026-04-23_rant-wraith.wav --source voice-solo

# eyeball the result
./target/release/corpus review /j/matt-voice/training-data/staging/2026-04-23_...jsonl

# promote into master if it looks good
./target/release/corpus promote /j/matt-voice/training-data/staging/2026-04-23_...jsonl
```

## Use — live Discord capture via Craig

1. In Discord voice with the guys, invite Craig (`/join`).
2. When the session ends, Craig DMs you a download link. Paste it into `/opt/matt-voice/craig-inbox.txt`:
   ```
   2026-04-23T22:01:00Z  https://craig.horse/rec/<id>?key=<key>  <matt-craig-user-id>
   ```
3. `listenerd` polls the file every 10s, downloads, extracts your track, runs whisper, writes staging JSONL, telegram-pings you the review command.
4. You run `corpus review <file>`, then `corpus promote <file>` if it's clean.

## Why staging-first

Whisper hallucinates on silence and crosstalk. A bad 3hr capture could poison the corpus. Cost of a manual eyeball (30 seconds) << cost of retraining on polluted data. Auto-promote unlocks in phase 2 once we've measured `avg_logprob` distributions across a few real captures and set a sane auto-gate.

## What's NOT in here (by design)

- **Training.** Training still lives in Python (`5_train_voice.py`) via unsloth. Candle-based training is a long-term contribution interest — not in scope for v1.
- **py-cord voice bot.** Discord's official bot SDK for voice-receive is Python-only; Craig handles capture, we handle ingest. Upgrade path to a native Rust Discord bot exists if Craig's manual `/join` friction becomes a problem.
- **Multi-speaker diarization.** Craig's per-user tracks make it unnecessary — each speaker is already isolated in the zip.

---

## Support This Project

If you find this project useful, consider buying me a coffee! Your support helps me keep building and sharing open-source tools.

[![Donate via PayPal](https://img.shields.io/badge/Donate-PayPal-blue.svg?logo=paypal)](https://www.paypal.me/baal_hosting)

**PayPal:** [baal_hosting@live.com](https://paypal.me/baal_hosting)

Every donation, no matter how small, is greatly appreciated and motivates continued development. Thank you!
