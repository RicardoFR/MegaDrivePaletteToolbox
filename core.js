'use strict';

// ── Constants ─────────────────────────────────────────────────────────────── //
const TILE = 8;
const N_PAL_COLORS = 15;

// 4×4 Bayer matrix — 16 dither levels vs 4 from 2×2.
// Normalized offset = (BAYER4[y%4][x%4] / 16 - 0.5) * strength
const BAYER4 = [
    [ 0,  8,  2, 10],
    [12,  4, 14,  6],
    [ 3, 11,  1,  9],
    [15,  7, 13,  5],
];

// ── Color distance: perceptual weighted RGB ───────────────────────────────── //
function dist2(a, b) {
    const dr = a[0] - b[0], dg = a[1] - b[1], db = a[2] - b[2];
    return 2 * dr * dr + 4 * dg * dg + db * db;
}

// ── Nearest palette color index (1-15, skip transparent index 0) ─────────── //
function nearestColor(palette, r, g, b) {
    let minD = Infinity, minI = 1;
    for (let i = 1; i < palette.length; i++) {
        const d = dist2(palette[i], [r, g, b]);
        if (d < minD) { minD = d; minI = i; }
    }
    return minI;
}

// ── Total distance of a tile to a palette ────────────────────────────────── //
function tileDist(palette, flat) {
    let sum = 0;
    for (let i = 0; i < flat.length; i += 3) {
        const ci = nearestColor(palette, flat[i], flat[i + 1], flat[i + 2]);
        sum += dist2(palette[ci], [flat[i], flat[i + 1], flat[i + 2]]);
    }
    return sum;
}

// ── Seeded PRNG (mulberry32) — ensures deterministic results ─────────────── //
function makePrng(seed) {
    let s = seed >>> 0;
    return function () {
        s += 0x6D2B79F5;
        let t = Math.imul(s ^ (s >>> 15), 1 | s);
        t ^= t + Math.imul(t ^ (t >>> 7), 61 | t);
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}

// ── K-means++ palette generation ─────────────────────────────────────────── //
function kmeans(samples, k, maxIter, rng) {
    if (maxIter === undefined) maxIter = 30;
    if (!rng) rng = Math.random.bind(Math);
    if (samples.length === 0) return Array.from({ length: k }, () => [0, 0, 0]);

    const centers = [samples[Math.floor(rng() * samples.length)].slice()];
    while (centers.length < k) {
        const dists = samples.map(p => {
            let minD = Infinity;
            for (const c of centers) { const d = dist2(p, c); if (d < minD) minD = d; }
            return minD;
        });
        const total = dists.reduce((a, b) => a + b, 0);
        if (total === 0) { centers.push(samples[0].slice()); continue; }
        let r = rng() * total;
        let idx = 0;
        for (; idx < dists.length - 1 && r > 0; idx++) r -= dists[idx];
        centers.push(samples[idx].slice());
    }

    for (let iter = 0; iter < maxIter; iter++) {
        const sums = Array.from({ length: k }, () => [0, 0, 0]);
        const counts = new Int32Array(k);
        for (const p of samples) {
            let minD = Infinity, minI = 0;
            for (let i = 0; i < k; i++) {
                const d = dist2(p, centers[i]);
                if (d < minD) { minD = d; minI = i; }
            }
            sums[minI][0] += p[0];
            sums[minI][1] += p[1];
            sums[minI][2] += p[2];
            counts[minI]++;
        }
        let changed = false;
        for (let i = 0; i < k; i++) {
            if (counts[i] > 0) {
                const nc = [
                    Math.round(sums[i][0] / counts[i]),
                    Math.round(sums[i][1] / counts[i]),
                    Math.round(sums[i][2] / counts[i]),
                ];
                if (nc[0] !== centers[i][0] || nc[1] !== centers[i][1] || nc[2] !== centers[i][2]) {
                    centers[i] = nc;
                    changed = true;
                }
            }
        }
        if (!changed) break;
    }
    return centers;
}

// ── Snap an RGB color to the nearest Mega Drive 9-bit color ─────────────── //
// MD has 3 bits per channel → 8 levels: round(n * 255/7) for n = 0..7
function snapToMD(c) {
    return c.map(v => Math.round(Math.round(v * 7 / 255) * 255 / 7));
}

// ── Build palette: inherit useful fixed colors, fill rest with K-means ───── //
//
// Phase 1 — Inherit: fixed palette colors that cover samples within thr2 are
//   reused in the generated palette (snapped to MD grid, deduped). This makes
//   GEN share boundary colors with FIX palettes → seamless tile transitions.
// Phase 2 — Fill: remaining slots are filled with K-means on pixels still not
//   well covered by the inherited colors.
//
function buildPal0(samples, rng, inheritCandidates, thr2) {
    const MAX_SAMPLES = 8000;
    let s = samples;
    if (samples.length > MAX_SAMPLES) {
        const step = Math.ceil(samples.length / MAX_SAMPLES);
        s = samples.filter((_, i) => i % step === 0);
    }

    const seenKeys = new Set();
    const unique   = [];

    // ── Phase 1: inherit fixed colors with coverage in this sample set ─────── //
    if (inheritCandidates && inheritCandidates.length > 0 && thr2 > 0) {
        const scored = [];
        const scoredSeen = new Set();
        for (const raw of inheritCandidates) {
            const c   = snapToMD(raw);
            const key = c[0] * 65536 + c[1] * 256 + c[2];
            if (scoredSeen.has(key)) continue;
            scoredSeen.add(key);
            let count = 0;
            for (const p of s) { if (dist2(p, c) <= thr2) count++; }
            if (count > 0) scored.push({ c, key, count });
        }
        scored.sort((a, b) => b.count - a.count);
        for (const { c, key } of scored) {
            if (unique.length >= N_PAL_COLORS) break;
            seenKeys.add(key);
            unique.push(c);
        }
    }

    // ── Phase 2: K-means on pixels not covered by inherited colors ─────────── //
    const residual = unique.length > 0
        ? s.filter(p => { for (const c of unique) { if (dist2(p, c) <= (thr2 || 0)) return false; } return true; })
        : s;

    const base = residual.length >= (N_PAL_COLORS - unique.length) ? residual : s;
    let k = N_PAL_COLORS - unique.length;
    for (let attempt = 0; attempt < 3 && unique.length < N_PAL_COLORS; attempt++) {
        if (k <= 0) break;
        for (const c of kmeans(base, k, 30, rng).map(snapToMD)) {
            const key = c[0] * 65536 + c[1] * 256 + c[2];
            if (!seenKeys.has(key)) { seenKeys.add(key); unique.push(c); }
            if (unique.length === N_PAL_COLORS) break;
        }
        k = (N_PAL_COLORS - unique.length) * 2;
    }

    // Pad with black if we couldn't fill all slots (rare edge case)
    while (unique.length < N_PAL_COLORS) unique.push([0, 0, 0]);
    return [[0, 0, 0], ...unique];
}

// ── Tile assignment ───────────────────────────────────────────────────────── //
// A fixed palette only wins a tile if it beats the best generated palette by
// at least 20%. This prevents fixed palettes from stealing tiles via marginal
// color matches (e.g. a sprite palette's white snatching bright sky tiles).
function assignTiles(imageData, palettes, W, H, numGenerate, fixedBias) {
    const TX = Math.floor(W / TILE), TY = Math.floor(H / TILE);
    const tileMap = new Uint8Array(TY * TX);
    for (let ty = 0; ty < TY; ty++) {
        for (let tx = 0; tx < TX; tx++) {
            const flat = [];
            for (let dy = 0; dy < TILE; dy++) {
                for (let dx = 0; dx < TILE; dx++) {
                    const i = ((ty * TILE + dy) * W + (tx * TILE + dx)) * 4;
                    flat.push(imageData[i], imageData[i + 1], imageData[i + 2]);
                }
            }
            let bestGenD = Infinity, bestGenPi = 0;
            for (let pi = 0; pi < numGenerate; pi++) {
                const d = tileDist(palettes[pi], flat);
                if (d < bestGenD) { bestGenD = d; bestGenPi = pi; }
            }
            let minD = bestGenD, minPi = bestGenPi;
            for (let pi = numGenerate; pi < palettes.length; pi++) {
                const d = tileDist(palettes[pi], flat);
                if (d < minD * fixedBias) { minD = d; minPi = pi; }
            }
            tileMap[ty * TX + tx] = minPi;
        }
    }
    return { tileMap, TX, TY };
}

// ── Row-majority palette enforcement ─────────────────────────────────────── //
// For each tile row, all tiles are assigned to whichever palette wins the most
// tiles in that row — but only if the winner doesn't produce significantly worse
// rendering than the current per-tile assignment. This prevents forcing a
// sky palette onto rows that also contain trees or ground.
function enforceRowMajority(tileMap, imageData, palettes, TX, TY, W) {
    const m = new Uint8Array(tileMap);

    function rowTileDist(ty, pi) {
        let sum = 0;
        for (let tx = 0; tx < TX; tx++) {
            const flat = [];
            for (let dy = 0; dy < TILE; dy++)
                for (let dx = 0; dx < TILE; dx++) {
                    const i = ((ty * TILE + dy) * W + (tx * TILE + dx)) * 4;
                    flat.push(imageData[i], imageData[i+1], imageData[i+2]);
                }
            sum += tileDist(palettes[pi], flat);
        }
        return sum;
    }

    for (let ty = 0; ty < TY; ty++) {
        const votes = {};
        for (let tx = 0; tx < TX; tx++) {
            const pi = m[ty * TX + tx];
            votes[pi] = (votes[pi] || 0) + 1;
        }
        const sorted = Object.entries(votes).sort((a, b) => b[1] - a[1]);
        const winner = parseInt(sorted[0][0]);

        // Already uniform — nothing to do.
        if (sorted.length === 1) continue;

        // Measure rendering quality: current mixed vs forced winner.
        let mixedDist = 0;
        for (let tx = 0; tx < TX; tx++) {
            const pi = m[ty * TX + tx];
            const flat = [];
            for (let dy = 0; dy < TILE; dy++)
                for (let dx = 0; dx < TILE; dx++) {
                    const i = ((ty * TILE + dy) * W + (tx * TILE + dx)) * 4;
                    flat.push(imageData[i], imageData[i+1], imageData[i+2]);
                }
            mixedDist += tileDist(palettes[pi], flat);
        }
        const winnerDist = rowTileDist(ty, winner);

        // Only enforce if:
        // 1. Winner has a clear majority (>60% of tiles in the row)
        // 2. Winner's quality is within 15% of the per-tile optimal
        const winnerVotes = votes[winner];
        const majority = winnerVotes / TX;
        if (majority > 0.6 && winnerDist <= mixedDist * 1.15) {
            for (let tx = 0; tx < TX; tx++) m[ty * TX + tx] = winner;
        }
    }
    return m;
}

// ── Smooth tile map: remove palette islands (4-neighbor) ─────────────────── //
function smoothTileMap(tileMap, imageData, palettes, TX, TY, W) {
    const m = new Uint8Array(tileMap);

    function getTileFlat(ty, tx) {
        const flat = [];
        for (let dy = 0; dy < TILE; dy++)
            for (let dx = 0; dx < TILE; dx++) {
                const i = ((ty * TILE + dy) * W + (tx * TILE + dx)) * 4;
                flat.push(imageData[i], imageData[i + 1], imageData[i + 2]);
            }
        return flat;
    }

    for (let pass = 0; pass < 8; pass++) {
        let changed = false;
        for (let ty = 0; ty < TY; ty++) {
            for (let tx = 0; tx < TX; tx++) {
                const pi = m[ty * TX + tx];

                // Collect all 4 neighbors (skip out-of-bounds)
                const neighbors = [];
                if (ty > 0)      neighbors.push(m[(ty - 1) * TX + tx]);
                if (ty < TY - 1) neighbors.push(m[(ty + 1) * TX + tx]);
                if (tx > 0)      neighbors.push(m[ty * TX + (tx - 1)]);
                if (tx < TX - 1) neighbors.push(m[ty * TX + (tx + 1)]);

                // Count how many neighbors agree on a single different palette
                const votes = {};
                for (const n of neighbors) {
                    if (n !== pi) votes[n] = (votes[n] || 0) + 1;
                }
                const best = Object.entries(votes).sort((a, b) => b[1] - a[1])[0];
                if (!best) continue;

                const candidate = parseInt(best[0]);
                const voteCount = best[1];
                // Switch if majority of neighbors use a different palette.
                // Tolerance scales with isolation: a tile surrounded on all sides
                // is forced to match (high tolerance) to eliminate palette islands.
                const tolerance = voteCount >= 4 ? 999 : voteCount >= 3 ? 3.0 : voteCount >= 2 ? 1.6 : 1.4;
                const flat = getTileFlat(ty, tx);
                if (tileDist(palettes[candidate], flat) <= tileDist(palettes[pi], flat) * tolerance) {
                    m[ty * TX + tx] = candidate;
                    changed = true;
                }
            }
        }
        if (!changed) break;
    }
    return m;
}

// ── Bayer 4×4 ordered dithering render ───────────────────────────────────── //
function renderBayer(imageData, W, H, tileMap, palettes, TX, strength) {
    const outRgb = new Uint8ClampedArray(W * H * 4);
    const outIdx = new Uint8Array(W * H);
    for (let y = 0; y < H; y++) {
        for (let x = 0; x < W; x++) {
            const pi      = tileMap[Math.floor(y / TILE) * TX + Math.floor(x / TILE)];
            const palette = palettes[pi];
            const src     = (y * W + x) * 4;
            const boff    = (BAYER4[y & 3][x & 3] / 16 - 0.5) * strength;
            const r = Math.max(0, Math.min(255, imageData[src]     + boff));
            const g = Math.max(0, Math.min(255, imageData[src + 1] + boff));
            const b = Math.max(0, Math.min(255, imageData[src + 2] + boff));
            const ci  = nearestColor(palette, r, g, b);
            const col = palette[ci];
            const dst = src;
            outRgb[dst]     = col[0];
            outRgb[dst + 1] = col[1];
            outRgb[dst + 2] = col[2];
            outRgb[dst + 3] = 255;
            outIdx[y * W + x] = pi * 16 + ci;
        }
    }
    return { outRgb, outIdx };
}

// ── PNG encoder (deflate is injected) ────────────────────────────────────── //
const crcTable = (() => {
    const t = new Uint32Array(256);
    for (let n = 0; n < 256; n++) {
        let c = n;
        for (let k = 0; k < 8; k++) c = (c & 1) ? (0xEDB88320 ^ (c >>> 1)) : (c >>> 1);
        t[n] = c;
    }
    return t;
})();

function crc32(buf) {
    let c = 0xFFFFFFFF;
    for (let i = 0; i < buf.length; i++) c = crcTable[(c ^ buf[i]) & 0xFF] ^ (c >>> 8);
    return (c ^ 0xFFFFFFFF) >>> 0;
}

function u32be(n) {
    return [(n >>> 24) & 0xFF, (n >>> 16) & 0xFF, (n >>> 8) & 0xFF, n & 0xFF];
}

function pngChunk(type, data) {
    const tb = [...type].map(c => c.charCodeAt(0));
    return new Uint8Array([...u32be(data.length), ...tb, ...data, ...u32be(crc32([...tb, ...data]))]);
}

async function encodeIndexedPNG(W, H, indices, rgbPalette, deflate) {
    const ihdr = new Uint8Array([...u32be(W), ...u32be(H), 8, 3, 0, 0, 0]);
    const plte = new Uint8Array(256 * 3);
    plte.set(rgbPalette.subarray(0, Math.min(rgbPalette.length, 768)));
    const raw = new Uint8Array(H * (1 + W));
    for (let y = 0; y < H; y++) {
        raw[y * (1 + W)] = 0;
        raw.set(indices.subarray(y * W, (y + 1) * W), y * (1 + W) + 1);
    }
    const idat = await deflate(raw);
    const sig = new Uint8Array([137, 80, 78, 71, 13, 10, 26, 10]);
    const chunks = [pngChunk('IHDR', ihdr), pngChunk('PLTE', plte), pngChunk('IDAT', idat), pngChunk('IEND', new Uint8Array(0))];
    const out = new Uint8Array(sig.length + chunks.reduce((s, c) => s + c.length, 0));
    out.set(sig);
    let off = sig.length;
    for (const c of chunks) { out.set(c, off); off += c.length; }
    return out;
}

// ── Collect pixels assigned to each generated palette from a tile map ────── //
function collectGenPixels(tileMap, inputData, TX, TY, W, numGenerate) {
    const genPixels = Array.from({ length: numGenerate }, () => []);
    for (let ty = 0; ty < TY; ty++) {
        for (let tx = 0; tx < TX; tx++) {
            const pi = tileMap[ty * TX + tx];
            if (pi >= numGenerate) continue;
            for (let dy = 0; dy < TILE; dy++)
                for (let dx = 0; dx < TILE; dx++) {
                    const i = ((ty * TILE + dy) * W + (tx * TILE + dx)) * 4;
                    genPixels[pi].push([inputData[i], inputData[i + 1], inputData[i + 2]]);
                }
        }
    }
    return genPixels;
}

// ── Rebuild generated palettes from their assigned pixels ─────────────────── //
function rebuildGenPalettes(genPalettes, genPixels, rng, allFixedColors, thr2) {
    return genPalettes.map((pal, g) =>
        genPixels[g].length >= N_PAL_COLORS
            ? buildPal0(genPixels[g], rng, allFixedColors, thr2)
            : pal
    );
}

// ── Build 256-entry PLTE byte array from palette list ─────────────────────── //
function buildRgbPalette(palettes) {
    const out = new Uint8Array(256 * 3);
    for (let pi = 0; pi < palettes.length; pi++) {
        const pal = palettes[pi];
        for (let ci = 0; ci < 16; ci++) {
            const b = (pi * 16 + ci) * 3;
            out[b] = pal[ci][0]; out[b + 1] = pal[ci][1]; out[b + 2] = pal[ci][2];
        }
    }
    return out;
}

// ── Build debug tilemap image (one hue per palette) ───────────────────────── //
function buildDebugImage(tileMap, TX, TY, W, H, totalPalettes) {
    const DBG_HUES = [0, 120, 240, 60, 180, 300, 30, 150];
    function hueToRgb(h) {
        const s = 0.7, l = 0.55;
        const c = (1 - Math.abs(2 * l - 1)) * s;
        const x = c * (1 - Math.abs((h / 60) % 2 - 1));
        const m = l - c / 2;
        let r, g, b;
        if (h < 60)       { r = c; g = x; b = 0; }
        else if (h < 120) { r = x; g = c; b = 0; }
        else if (h < 180) { r = 0; g = c; b = x; }
        else if (h < 240) { r = 0; g = x; b = c; }
        else if (h < 300) { r = x; g = 0; b = c; }
        else              { r = c; g = 0; b = x; }
        return [Math.round((r + m) * 255), Math.round((g + m) * 255), Math.round((b + m) * 255), 255];
    }
    const colors = Array.from({ length: totalPalettes }, (_, i) => hueToRgb(DBG_HUES[i % DBG_HUES.length]));
    const out = new Uint8ClampedArray(W * H * 4);
    for (let ty = 0; ty < TY; ty++)
        for (let tx = 0; tx < TX; tx++) {
            const col = colors[tileMap[ty * TX + tx]];
            for (let dy = 0; dy < TILE; dy++)
                for (let dx = 0; dx < TILE; dx++)
                    out.set(col, ((ty * TILE + dy) * W + (tx * TILE + dx)) * 4);
        }
    return out;
}

// ── Count tile and color usage per palette ────────────────────────────────── //
function buildUsageStats(tileMap, outIdx, totalPalettes) {
    const usage = new Array(totalPalettes).fill(0);
    for (let i = 0; i < tileMap.length; i++) usage[tileMap[i]]++;
    const colorUsed = Array.from({ length: totalPalettes }, () => new Uint8Array(16));
    for (let i = 0; i < outIdx.length; i++) colorUsed[outIdx[i] >> 4][outIdx[i] & 0xF] = 1;
    return { usage, colorUsed };
}

// ── Main pipeline (shared by CLI and browser) ─────────────────────────────── //
//
// fixedPaletteColors : [[r,g,b]×15][]  — array of fixed (input) palettes
// numGenerate        : number          — how many palettes to generate (default 1)
// deflate            : async (Uint8Array) => Uint8Array  — injected by caller
// onProgress         : (pct, text) => void               — injected by caller
//
// Output palette order in the indexed PNG:
//   slots 0..numGenerate-1          → generated palettes
//   slots numGenerate..total-1      → fixed palettes (same order as input)
//
async function processImage({ inputData, fixedPaletteColors, numGenerate, W, H, ditherStrength, residualThr, maxIter, doSmooth, fixedBias, seed }, deflate, onProgress) {
    if (fixedBias === undefined) fixedBias = 0.8;
    const rng = makePrng(seed !== undefined ? seed : 1);
    if (!numGenerate || numGenerate < 0) numGenerate = 0;
    const fixedPalettes = (fixedPaletteColors || []).map(cols => [[0, 0, 0], ...cols]);
    numGenerate = Math.min(numGenerate, Math.max(0, 4 - fixedPalettes.length));
    if (numGenerate === 0 && fixedPalettes.length === 0) numGenerate = 1;
    const totalPalettes = numGenerate + fixedPalettes.length;
    const thr2 = residualThr * residualThr;
    const allFixedColors = fixedPalettes.flatMap(pal => pal.slice(1));

    // ── Residual analysis ──────────────────────────────────────────────────── //
    onProgress(5, 'Analyzing coverage...');
    const allPixels = Array.from({ length: W * H }, (_, i) =>
        [inputData[i * 4], inputData[i * 4 + 1], inputData[i * 4 + 2]]);
    const residualPixels = allPixels.filter(p => {
        for (const pal of fixedPalettes)
            for (let j = 1; j < pal.length; j++) if (dist2(p, pal[j]) <= thr2) return false;
        return true;
    });
    const initSamples = residualPixels.length >= N_PAL_COLORS ? residualPixels : allPixels;

    // ── Initial palette generation ─────────────────────────────────────────── //
    let genPalettes = [];
    if (numGenerate > 0) {
        onProgress(15, `Building ${numGenerate} palette(s) (${residualPixels.length} residual px)...`);
        const sorted = initSamples.slice().sort((a, b) => (a[0] + a[1] + a[2]) - (b[0] + b[1] + b[2]));
        const groupSize = Math.ceil(sorted.length / numGenerate);
        genPalettes = Array.from({ length: numGenerate }, (_, g) => {
            const group = sorted.slice(g * groupSize, (g + 1) * groupSize);
            return buildPal0(group.length >= N_PAL_COLORS ? group : initSamples, rng, allFixedColors, thr2);
        });
    }

    // ── Iterative refinement: assign tiles → rebuild palettes ─────────────── //
    let palettes = [...genPalettes, ...fixedPalettes];
    let tileMap, TX, TY;

    for (let iter = 1; iter <= (numGenerate > 0 ? maxIter : 0); iter++) {
        onProgress(15 + (iter / maxIter) * 55, `Iteration ${iter}/${maxIter}...`);
        ({ tileMap, TX, TY } = assignTiles(inputData, palettes, W, H, numGenerate, fixedBias));
        genPalettes = rebuildGenPalettes(genPalettes, collectGenPixels(tileMap, inputData, TX, TY, W, numGenerate), rng, allFixedColors, thr2);
        palettes = [...genPalettes, ...fixedPalettes];
    }

    // ── Spatial smoothing ──────────────────────────────────────────────────── //
    if (doSmooth) {
        onProgress(73, 'Enforcing row palette bands...');
        tileMap = enforceRowMajority(tileMap, inputData, palettes, TX, TY, W);
        onProgress(77, 'Smoothing tile islands...');
        tileMap = smoothTileMap(tileMap, inputData, palettes, TX, TY, W);
        genPalettes = rebuildGenPalettes(genPalettes, collectGenPixels(tileMap, inputData, TX, TY, W, numGenerate), rng, allFixedColors, thr2);
        palettes = [...genPalettes, ...fixedPalettes];
    }

    // ── Render + encode ────────────────────────────────────────────────────── //
    onProgress(80, 'Rendering (Bayer dither)...');
    const { outRgb, outIdx } = renderBayer(inputData, W, H, tileMap, palettes, TX, ditherStrength);

    onProgress(90, 'Encoding PNG...');
    const indexedPng = await encodeIndexedPNG(W, H, outIdx, buildRgbPalette(palettes), deflate);
    const debugData  = buildDebugImage(tileMap, TX, TY, W, H, totalPalettes);
    const { usage, colorUsed } = buildUsageStats(tileMap, outIdx, totalPalettes);

    return {
        outRgb, outIdx, indexedPng, debugData,
        generatedPaletteColors: genPalettes.map(p => p.slice(1)),
        usage, colorUsed, W, H,
        numGenerate, numFixed: fixedPalettes.length,
        palettes,
    };
}

// ── Export (Node.js) — browser loads via importScripts() ─────────────────── //
if (typeof module !== 'undefined') {
    module.exports = { TILE, N_PAL_COLORS, processImage };
}
