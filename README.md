# Hybrid Driver

**Learn each Formula 1 driver's "driving fingerprint" from real telemetry, blend two drivers into a hybrid, and listen to the result as live music.**

🌐 Live stage: https://lileinsteinbrain.github.io/hybrid-driver-demo/

## The idea
Two drivers can set similar lap times while driving very differently: when they brake, how hard they commit to the throttle, how smoothly they steer. This project learns a compact fingerprint for each driver from official qualifying telemetry, lets you mix two of them (e.g. 60% Norris + 40% Russell), and turns the blend into sound, so differences in driving style become something you can *hear* rather than read off a dashboard.

## How it works
```
 FastF1 qualifying telemetry (throttle, brake, steering, speed)
            │  resample each lap to 220 points
            ▼
 PolicyNet: predicts the driver's next action given the track phase
            each driver gets a learned embedding z  ("fingerprint")
            │
            ▼  z_mix = α·z_A + (1−α)·z_B
 Streamlit app: per-segment similarity → "this corner looks most like VER"
            │
            ▼  WebSocket bridge
 Browser live stage (Tone.js): tempo, mode, bass, lead, drum density
            driven by the mixed fingerprint
```

## What's in this repo
| Path | What it is |
|---|---|
| `app_hybrid.py` | Main Streamlit app: pick event + drivers, blend with α, explain segments, push to audio |
| `driver-fingerprint/scripts/utils_fp.py` | Model (PolicyNet), dataset building and training |
| `driver-fingerprint/scripts/export_persona.py` | Turns learned fingerprints into traceable music parameters |
| `driver-fingerprint/results/integration_Q5/` | Trained model and summary |
| `ws_bridge.py` | WebSocket relay between the app and the browser |
| `docs/` | Browser live stage (GitHub Pages) |
| `live-stage/` | Standalone local version of the stage |

## Status
Working demo, deployed (Streamlit app on Railway, stage on GitHub Pages).
- ✅ Different drivers produce clearly different rhythmic structures
- ✅ Blending is continuous; segment-level explanations show which driver each section resembles
- 🔲 Qualifying sessions only; mappings are not yet stable across tracks
- 🔲 Perceptual test of whether listeners can tell drivers apart → see [f1-abx-test](https://github.com/lileinsteinbrain/f1-abx-test)
