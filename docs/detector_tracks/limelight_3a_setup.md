# Limelight 3A (SSD)

Model files, upload flow, and decoding are documented in [yolo/README.md](https://github.com/sidhuharjas/Hive-Vision/blob/main/yolo/README.md). This page adds what that README doesn't warn about — most of it learned the hard way on a robot.

## Setup order

1. Upload `best_limelight3a_ssd_mobilenetv2_300x300.tflite` + `labels.txt` (class order must match; engine = CPU).
2. **Verify detections in the web UI before writing any FTC code.**
3. Set confidence slider **0.35–0.45** — 0.25 produces visible false positives.
4. Match the network-table name to your FTC hardware-map entry.
5. Read `getLatestResult()` / `getDetectorResults()` with the SDK-native `Limelight3A` class (no third-party helper). Filter to your alliance class and gate on confidence before acting.

## Gotchas the README won't tell you

* **Staleness is in microseconds.** `LLResult.getStaleness()` is µs — compare against \~120,000 (120 ms), not 120.
* **Poll rate ≠ detection rate.** A 100 Hz poll doesn't create 100 Hz of fresh neural frames. Always gate on staleness.
* **Confidence scale may be 0–1 or 0–100** depending on the API surface. Read it once in telemetry before trusting `MIN_CONF`.
* **You are reading the primary target only** unless you iterate the detection list. For multiple balls, use `getDetectorResults()`, not just `getTx/ty/ta`.

This page covers detection only. Robot driving that _uses_ these detections is intentionally a separate, unreleased topic.
