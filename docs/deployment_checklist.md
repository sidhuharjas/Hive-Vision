# Deployment checklist

The full artifact table and build/verify commands live in
[shipping/SHIP_README.md](https://github.com/sidhuharjas/Hive-Vision/blob/main/shipping/SHIP_README.md).
This page is the on-match-day version: what to check, in order, without
re-reading the handbook.

- [ ] Model + labels on the 3A, engine = CPU, correct pipeline
- [ ] Confidence 0.35–0.45 (0.25 = false positives)
- [ ] Network-table name matches the FTC hardware map
- [ ] FTC code filters your alliance class and gates on confidence
- [ ] Hub track (if used) confirmed across frames, not single-frame
- [ ] Recall sanity-checked on your field's lighting, not just demo clips
- [ ] **Still unverified on hardware**: the SSD `.tflite` actually loading and
      reporting on a real 3A — PC runners verify the model, not the device