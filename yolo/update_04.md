# Hive Vision â€” Update #4

**Vision pipeline: pick your tier.**

Decide once, at the start of the season, which detector the FTC code trusts.
All three tiers are permissive (usable this year), but they trade robustness
against cost and hardware.

## The three tiers

| Option | Strengths | Tradeoffs |
|--------|-----------|-----------|
| **Limelight SSD** | More adaptable to shape, distance, blur, and changing backgrounds; generally the stronger accuracy option | Requires a Limelight, model upload, and FTC-side result integration |
| **Control Hub OpenCV** | Lightweight, low-cost, fast to run locally â€” no model or coprocessor | More fragile to lighting, white balance, camera resolution, and colors that resemble the balls; needs field retuning |
| **Control Hub Lab** | Same cost as OpenCV, but matches on **Lab chromaticity hue with an adaptive chroma floor** â€” the classic HSV shadow failure (a shadowed ball reading as a different color) is largely gone | Slightly more compute than HSV; **deep-shadow balls are so desaturated that no color-only detector can see them** (then use Limelight SSD) |

## Recommended: Control Hub Lab (chroma-floor)

`Lab` (CIE L\*a\*b\*) separates luminance from color better than HSV/RGB, and the
**adaptive chroma floor** is the key move: instead of a fixed color threshold
that a shadow can cross, the floor rides the local saturation so a ball keeps
its label until it genuinely desaturates past visibility. That kills the worst
HSV failure â€” a shadowed ball reading as a *different* color â€” because hue
survives shadows that only darken luminance.

The honest limit: take a ball deep enough into shadow and it stops having
enough color for *any* color-only detector to see (`a\*`,`b\*` â‰ˆ chroma â‰ˆ 0).
That case is explicitly not this tier's job â€” it's the Limelight SSD's.

## Decision tree (one shot, preseason)

1. **Ball fully lit / distinct colors** â†’ **Control Hub Lab** (free, fast,
   robust). Done.
2. **Confident balls in shadow / unreliable hue** â†’ **Control Hub OpenCV** as
   a second cheap hint, but don't trust it on shadowed balls.
3. **Balls under deep shadow / night / low light** â†’ **Limelight SSD** (SSD
   detector / 3A) â€” the only tier that sees desaturated balls.

> Keyframe rule: Lab for color, Limelight SSD for shape â€” whenever color
> can't decide, shape outranks chroma.

See `yolo/README.md` for the artifact table and `UPDATE_03.md` for the SSD
read-out decode.
