#!/usr/bin/env node
'use strict';

/**
 * cli.js — Mega Drive N-palette tile mapper (command line)
 *
 * Usage:
 *   node cli.js [input] [pal1.png pal2.png ...] [output_dir]
 *
 *   The last positional argument is always the output directory.
 *   All arguments between input and output_dir are fixed palette files.
 *
 * Defaults:
 *   input      ../input.png
 *   palettes   ../PAL1.png ../PAL2.png
 *   output_dir ../
 *
 * Options (env vars):
 *   DITHER_MODE=fs   Dither mode: fs (Floyd-Steinberg, default), bayer, none
 *   DITHER=75        Dither strength: 0-100 for fs (default 75), or pixel offset for bayer (default 20)
 *   RESIDUAL=3600    Residual threshold² (default 3600)
 *   ITERS=6          Refinement iterations (default 6)
 *   SMOOTH=1         Tile smoothing (1=on, 0=off, default 1)
 *   GENERATE=1       Number of palettes to generate (default 1, 0 = only fixed palettes)
 *   FIXED_BIAS=80    Fixed palette affinity 0-100 (0=never use fixed, 100=use if marginally better, default 80)
 *   QUANT_METHOD=wuquant   Quantization: wuquant, neuquant, neuquant-float, rgbquant (default wuquant)
 *   QUANT_DISTANCE=euclidean  Color distance for quantization: euclidean, euclidean-bt709-noalpha,
 *                              cie94-graphic-arts, cie94-textiles, ciede2000, manhattan, pngquant
 */

const { Jimp }       = require('jimp');
const path           = require('path');
const fs             = require('fs');
const zlib           = require('zlib');
const { processImage } = require('./core.js');

// ── Args / options ─────────────────────────────────────────────────────────── //
const args = process.argv.slice(2);

if (args.length < 2) {
    console.error('Usage: node cli.js <input.png> [pal1.png pal2.png ...] <output_dir>');
    process.exit(1);
}

const INPUT_PATH = args[0];
const OUT_DIR    = args[args.length - 1];
const PAL_PATHS  = args.slice(1, -1);

const DITHER_MODE     = process.env.DITHER_MODE    ?? 'none';
const DITHER_STRENGTH = parseInt(process.env.DITHER      ?? '20');
const RESIDUAL_THR    = Math.sqrt(parseInt(process.env.RESIDUAL  ?? '3600'));
const MAX_ITER        = parseInt(process.env.ITERS        ?? '6');
const DO_SMOOTH       = (process.env.SMOOTH        ?? '1') !== '0';
const NUM_GENERATE    = parseInt(process.env.GENERATE     ?? '1');
const FIXED_BIAS      = parseInt(process.env.FIXED_BIAS   ?? '80') / 100;
const SEED            = parseInt(process.env.SEED         ?? '1');
const QUANT_METHOD    = process.env.QUANT_METHOD   ?? 'wuquant';
const QUANT_DISTANCE  = process.env.QUANT_DISTANCE ?? 'euclidean';

// ── deflate: Node.js zlib ─────────────────────────────────────────────────── //
function deflate(data) {
    return new Promise((resolve, reject) =>
        zlib.deflate(Buffer.from(data), (err, buf) =>
            err ? reject(err) : resolve(new Uint8Array(buf))));
}

// ── Main ──────────────────────────────────────────────────────────────────── //
async function main() {
    console.log(`Input:    ${INPUT_PATH}`);
    PAL_PATHS.forEach((p, i) => console.log(`Fixed[${i}]: ${p}`));
    console.log(`Output:   ${OUT_DIR}`);
    console.log(`Generate: ${NUM_GENERATE}  Dither: ${DITHER_MODE}${DITHER_MODE === 'bayer' ? `(${DITHER_STRENGTH})` : ''}  Residual: ${RESIDUAL_THR.toFixed(0)}  Iters: ${MAX_ITER}  Smooth: ${DO_SMOOTH}  Seed: ${SEED}`);
    console.log(`Quant: ${QUANT_METHOD}  Distance: ${QUANT_DISTANCE}`);
    console.log('');

    process.stdout.write('Loading images...');
    const [imgInput, ...palImgs] = await Promise.all([
        Jimp.read(INPUT_PATH),
        ...PAL_PATHS.map(p => Jimp.read(p)),
    ]);
    const W = imgInput.bitmap.width, H = imgInput.bitmap.height;
    console.log(` ${W}×${H}`);

    function extractPalette(img) {
        const { data, width, height } = img.bitmap;
        const horiz = width >= height;
        const colors = [];
        for (let i = 1; i < 16; i++) {
            const x = horiz ? i : 0, y = horiz ? 0 : i;
            const idx = (y * width + x) * 4;
            colors.push([data[idx], data[idx + 1], data[idx + 2]]);
        }
        return colors;
    }

    const inputData        = new Uint8ClampedArray(imgInput.bitmap.data.buffer);
    const fixedPaletteColors = palImgs.map(extractPalette);

    let lastPct = -1;
    const result = await processImage(
        { inputData, fixedPaletteColors, numGenerate: NUM_GENERATE, W, H,
          ditherMode: DITHER_MODE, ditherStrength: DITHER_STRENGTH, residualThr: RESIDUAL_THR,
          maxIter: MAX_ITER, doSmooth: DO_SMOOTH, fixedBias: FIXED_BIAS, seed: SEED,
          quantMethod: QUANT_METHOD, quantDistance: QUANT_DISTANCE },
        deflate,
        (pct, text) => {
            if (pct !== lastPct) { process.stdout.write(`\r${text.padEnd(50)}`); lastPct = pct; }
        }
    );
    console.log('\nDone.');

    // Save preview
    process.stdout.write('Saving outputs...');
    const previewImg = new Jimp({ width: W, height: H });
    for (let i = 0; i < W * H; i++) {
        previewImg.bitmap.data[i * 4]     = result.outRgb[i * 4];
        previewImg.bitmap.data[i * 4 + 1] = result.outRgb[i * 4 + 1];
        previewImg.bitmap.data[i * 4 + 2] = result.outRgb[i * 4 + 2];
        previewImg.bitmap.data[i * 4 + 3] = 255;
    }
    await previewImg.write(path.join(OUT_DIR, 'output_preview.png'));
    fs.writeFileSync(path.join(OUT_DIR, 'output.png'), Buffer.from(result.indexedPng));

    // Save debug tilemap
    const debugImg = new Jimp({ width: W, height: H });
    for (let i = 0; i < W * H; i++) {
        debugImg.bitmap.data[i * 4]     = result.debugData[i * 4];
        debugImg.bitmap.data[i * 4 + 1] = result.debugData[i * 4 + 1];
        debugImg.bitmap.data[i * 4 + 2] = result.debugData[i * 4 + 2];
        debugImg.bitmap.data[i * 4 + 3] = 255;
    }
    await debugImg.write(path.join(OUT_DIR, 'debug.png'));
    console.log(' done');

    const total = result.usage.reduce((a, b) => a + b, 0);
    const pct   = i => total ? Math.round(100 * result.usage[i] / total) : 0;

    const usageParts = [];
    for (let i = 0; i < result.numGenerate; i++) usageParts.push(`GEN${i} ${pct(i)}%`);
    for (let i = 0; i < result.numFixed;    i++) usageParts.push(`FIX${i} ${pct(result.numGenerate + i)}%`);
    console.log(`\nTile usage: ${usageParts.join('  ')}  (${total} tiles)`);

    console.log(`\nSaved → ${OUT_DIR}/output.png  output_preview.png  debug.png`);

    // Build and save palette report
    const reportLines = [];
    for (let pi = 0; pi < result.palettes.length; pi++) {
        const isGen  = pi < result.numGenerate;
        const label  = isGen ? `GEN${pi}` : `FIX${pi - result.numGenerate}`;
        const pal    = result.palettes[pi];
        const used   = result.colorUsed[pi];
        const count  = used.slice(1).reduce((s, v) => s + v, 0);
        reportLines.push(`${label} — ${count}/15 colors used  (${pct(pi)}% of tiles)`);
        for (let ci = 1; ci < 16; ci++) {
            const [r, g, b] = pal[ci];
            const tick = used[ci] ? '✓' : '·';
            reportLines.push(`  [${String(ci).padStart(2)}] ${tick}  RGB(${String(r).padStart(3)}, ${String(g).padStart(3)}, ${String(b).padStart(3)})`);
        }
        reportLines.push('');
    }
    const reportPath = path.join(OUT_DIR, 'palette_report.txt');
    fs.writeFileSync(reportPath, reportLines.join('\n'), 'utf8');
    console.log(`       → ${reportPath}`);
}

main().catch(err => { console.error('Error:', err.message); process.exit(1); });
