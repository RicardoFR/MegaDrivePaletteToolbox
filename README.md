# MegaDrive Palette Toolbox

Mega Drive / Genesis palette quantizer — maps any image to 1–4 hardware palettes (9-bit, 3bpc) using Wu's Color Quantization and OKLab perceptual color matching. Outputs an indexed PNG ready for SGDK.

### The idea

Mega Drive games typically share palettes across multiple assets — sprites, HUD elements, and backgrounds all reuse the same 4 hardware palettes. This tool lets you bring those existing palettes into the conversion process:

- **Fixed palettes** — palettes already defined by your game (sprites, UI, etc.). The tool will use them as-is and assign tiles to them when they're a good fit.
- **Generated palettes** — new palettes the tool creates automatically from the parts of the image not well covered by the fixed ones.

This way the output image reuses your game's existing color slots wherever possible, and only generates new palette entries for what's truly needed.

## Web app

👉 **[Launch on GitHub Pages](https://ricardofr.github.io/MegaDrivePaletteToolbox/)**

## Quick start

Example palette files are included. Use them with your own input image:

```bash
npm install
node cli.js your_image.png example_PAL1.png example_PAL2.png ./
```

## CLI

```bash
npm install
node cli.js [input] [pal1.png pal2.png ...] [output_dir]
```

**Environment variables:**

| Variable | Default | Description |
|----------|---------|-------------|
| `GENERATE` | `1` | Number of palettes to generate |
| `DITHER_MODE` | `none` | Dither mode: `none`, `fs` (Floyd-Steinberg), `bayer` |
| `DITHER` | `20` | Dither strength: 0–100 for FS, pixel offset for Bayer |
| `RESIDUAL` | `3600` | Residual threshold² — pixels further than this from any fixed palette color are considered uncovered and used to build generated palettes |
| `ITERS` | `6` | Refinement iterations |
| `SMOOTH` | `1` | Tile smoothing and row-majority enforcement (0 = off) |
| `FIXED_BIAS` | `80` | Fixed palette affinity 0–100: how aggressively fixed palettes claim tiles over generated ones |
| `SEED` | `1` | RNG seed for deterministic output |

**Example:**

```bash
GENERATE=2 DITHER_MODE=fs DITHER=75 node cli.js bg.png sky.png sprites.png out/
```

## Outputs

| File | Description |
|------|-------------|
| `output.png` | Indexed PNG (8bpp, SGDK-ready) |
| `output_preview.png` | RGB preview |
| `debug.png` | Tile palette assignment map |
| `palette_report.txt` | Color usage per palette |

## How it works

1. **Residual analysis** — finds pixels not well covered by any fixed palette (using `RESIDUAL` threshold)
2. **Wu's Color Quantization** — generates palettes from residual pixels pre-snapped to the MD color grid, eliminating wasted slots from post-snap duplicates
3. **Iterative refinement** — reassigns tiles to best palette, rebuilds palette colors from assigned pixels, repeats N times until convergence
4. **Row-majority enforcement** — each tile row is locked to a single palette (matches Mega Drive scanline palette behavior), unless it significantly degrades quality
5. **Tile smoothing** — removes isolated palette islands (a tile surrounded by neighbors using a different palette)
6. **Dithering** — optional Floyd-Steinberg error diffusion (palette-boundary-aware) or 4×4 Bayer ordered dithering
7. **MD color snap** — all generated colors are quantized to valid 9-bit MD values (3 bits per channel: 0, 36, 73, 109, 146, 182, 219, 255)

Color matching throughout uses **OKLab** perceptual distance for visually accurate nearest-color selection.

## Palette format

Fixed palette images should be 16×1 (or 1×16) pixels — index 0 is ignored (transparent), indices 1–15 are the colors.

Indexed PNG output encodes pixel color as `palette_index * 16 + color_index`, matching SGDK's `PAL0`–`PAL3` layout.

## Requirements

- Node.js 18+ (for CLI)
- Any modern browser (for web app)
