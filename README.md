# MegaDrive Palette Toolbox

Mega Drive / Genesis palette quantizer — maps any image to 1–4 hardware palettes (9-bit, 3bpc) using K-means++ and 4×4 Bayer ordered dithering. Outputs an indexed PNG ready for SGDK.

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
| `DITHER` | `20` | Bayer dither strength (0 = off) |
| `RESIDUAL` | `3600` | Residual threshold² for palette seeding |
| `ITERS` | `6` | K-means refinement iterations |
| `SMOOTH` | `1` | Row-majority palette enforcement (0 = off) |

**Example:**

```bash
GENERATE=2 DITHER=30 node cli.js bg.png sky.png sprites.png out/
```

## Outputs

| File | Description |
|------|-------------|
| `output.png` | Indexed PNG (8bpp, SGDK-ready) |
| `output_preview.png` | RGB preview |
| `debug.png` | Tile palette assignment map |
| `palette_report.txt` | Color usage per palette |

## How it works

1. **Residual analysis** — finds pixels not well covered by any fixed palette
2. **K-means++** — generates N palettes from residual pixels, seeded by brightness groups
3. **Iterative refinement** — reassigns tiles to best palette, rebuilds palette colors, repeats
4. **Row-majority enforcement** — each tile row is locked to a single palette (matches Mega Drive HINT scanline behavior), unless it significantly degrades quality
5. **Bayer 4×4 dithering** — 16-level ordered dithering to smooth color transitions
6. **MD color snap** — all generated colors are quantized to valid 9-bit MD values (3 bits per channel)

## Palette format

Fixed palette images should be 16×1 (or 1×16) pixels — index 0 is ignored (transparent), indices 1–15 are the colors.

Indexed PNG output encodes pixel color as `palette_index * 16 + color_index`, matching SGDK's `PAL0`–`PAL3` layout.

## Requirements

- Node.js 18+ (for CLI)
- Any modern browser (for web app)
