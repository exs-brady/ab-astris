# Preprocessing Parameters (Table 2)

Domain-specific preprocessing parameters used in Stage 1 of the pipeline.
All subsequent stages (2-5) are identical across domains.

| Domain | Sampling Rate | Bandpass (Hz) | Envelope | Windows | Freq Range (Hz) |
|---|---|---|---|---|---|
| Astronomy | ~2 min cadence | 0.1–50 d⁻¹ | No | 4–8 sectors | 0.1–50 d⁻¹ |
| Bearing | 12 kHz | 500–5000 | Hilbert | 8 | 50–500 |
| Volcanic tremor | 100 Hz | 1–10 | Hilbert | 6 | 0.5–5 |
| Oceanography | Hourly | Constituent-specific | No | 6 | 0.01–0.10 |
| Wind turbines | 10 min SCADA | None | No | 4 | 0.01–2.0 |
| SHM (Z24) | 100 Hz | 2–20 | No | 4 | 2–15 |
| SHM (buildings) | 200 Hz | 0.5–25 | No | 6 | 0.5–25 |

## Notes

- **Hilbert envelope**: Applied for bearing and volcanic domains to extract
  amplitude modulation from carrier signals (Section 2.1, Stage 1).
- **Window count**: Varies by data volume. Astronomy uses sector boundaries
  as natural windows; other domains use 4–8 overlapping windows with 50% overlap.
- **Frequency grid**: 10,000 points (linearly spaced) for primary analysis;
  5,000 points per window in multi-window validation.
