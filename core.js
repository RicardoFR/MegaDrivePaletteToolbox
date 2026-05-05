'use strict';

// ── image-q (Node.js via require, browser via imageQ global) ─────────────── //
let _iq = null;
if (typeof require !== 'undefined') {
    try { _iq = require('image-q'); } catch (e) {}
} else if (typeof imageQ !== 'undefined') {
    _iq = imageQ;
}

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

// ── sRGB → linear lookup table (256 entries) ─────────────────────────────── //
const LINEAR = new Float32Array(256);
for (let i = 0; i < 256; i++) {
    const v = i / 255;
    LINEAR[i] = v <= 0.04045 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4);
}

// ── RGB → OKLab ───────────────────────────────────────────────────────────── //
// Inputs may be floats (e.g. after error diffusion) — round to nearest integer
// before the lookup table access (Float32Array[float] returns undefined in JS).
function rgbToOklab(r, g, b) {
    const lr = LINEAR[(r + 0.5) | 0], lg = LINEAR[(g + 0.5) | 0], lb = LINEAR[(b + 0.5) | 0];
    const l = Math.cbrt(0.4122214708 * lr + 0.5363325363 * lg + 0.0514459929 * lb);
    const m = Math.cbrt(0.2119034982 * lr + 0.6806995451 * lg + 0.1073969566 * lb);
    const s = Math.cbrt(0.0883024619 * lr + 0.2817188376 * lg + 0.6299787005 * lb);
    return [
        0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
        1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
        0.0259040371 * l + 0.4784205430 * m - 0.5043165098 * s,
    ];
}

// ── OKLab → sRGB ──────────────────────────────────────────────────────────── //
function oklabToRgb(L, a, b) {
    const l_ = L + 0.3963377774 * a + 0.2158037573 * b;
    const m_ = L - 0.1055613458 * a - 0.0638541728 * b;
    const s_ = L - 0.0894841775 * a - 1.2914855480 * b;
    const l = l_ * l_ * l_, m = m_ * m_ * m_, s = s_ * s_ * s_;
    const lr =  4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s;
    const lg = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s;
    const lb = -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s;
    function delinearize(v) {
        v = Math.max(0, Math.min(1, v));
        return Math.round(255 * (v <= 0.0031308 ? 12.92 * v : 1.055 * Math.pow(v, 1 / 2.4) - 0.055));
    }
    return [delinearize(lr), delinearize(lg), delinearize(lb)];
}

// ── Color distance: perceptual (OKLab) — used for matching and clustering ─── //
function dist2(a, b) {
    const [L1, a1, b1] = rgbToOklab(a[0], a[1], a[2]);
    const [L2, a2, b2] = rgbToOklab(b[0], b[1], b[2]);
    const dL = L1 - L2, da = a1 - a2, db = b1 - b2;
    return dL * dL + da * da + db * db;
}

// ── Coverage distance: weighted RGB — keeps RESIDUAL threshold in its original scale ─ //
function coverDist2(a, b) {
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

// ── K-means++ palette generation (works in OKLab space) ──────────────────── //
function kmeans(samples, k, maxIter, rng) {
    if (maxIter === undefined) maxIter = 30;
    if (!rng) rng = Math.random.bind(Math);
    if (samples.length === 0) return Array.from({ length: k }, () => [0, 0, 0]);

    // Convert samples to OKLab for perceptually uniform clustering
    const lab = samples.map(p => rgbToOklab(p[0], p[1], p[2]));

    function labDist2(a, b) {
        const dL = a[0] - b[0], da = a[1] - b[1], db = a[2] - b[2];
        return dL * dL + da * da + db * db;
    }

    // k-means++ seeding in OKLab space
    const centers = [lab[Math.floor(rng() * lab.length)].slice()];
    while (centers.length < k) {
        const dists = lab.map(p => {
            let minD = Infinity;
            for (const c of centers) { const d = labDist2(p, c); if (d < minD) minD = d; }
            return minD;
        });
        const total = dists.reduce((a, b) => a + b, 0);
        if (total === 0) { centers.push(lab[0].slice()); continue; }
        let r = rng() * total;
        let idx = 0;
        for (; idx < dists.length - 1 && r > 0; idx++) r -= dists[idx];
        centers.push(lab[idx].slice());
    }

    // Iterate: assign + update centroids in OKLab space
    for (let iter = 0; iter < maxIter; iter++) {
        const sums = Array.from({ length: k }, () => [0, 0, 0]);
        const counts = new Int32Array(k);
        for (const p of lab) {
            let minD = Infinity, minI = 0;
            for (let i = 0; i < k; i++) {
                const d = labDist2(p, centers[i]);
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
                    sums[i][0] / counts[i],
                    sums[i][1] / counts[i],
                    sums[i][2] / counts[i],
                ];
                if (nc[0] !== centers[i][0] || nc[1] !== centers[i][1] || nc[2] !== centers[i][2]) {
                    centers[i] = nc;
                    changed = true;
                }
            }
        }
        if (!changed) break;
    }

    // Convert OKLab centers back to RGB
    return centers.map(c => oklabToRgb(c[0], c[1], c[2]));
}

// ── Snap an RGB color to the nearest Mega Drive 9-bit color ─────────────── //
// MD has 3 bits per channel → 8 levels: round(n * 255/7) for n = 0..7
function snapToMD(c) {
    return c.map(v => Math.round(Math.round(v * 7 / 255) * 255 / 7));
}

// ── Wu's color quantization on MD-snapped pixels (Node.js) ───────────────── //
// Pre-snapping the input guarantees all output colors land on the MD grid,
// eliminating duplicates and wasted slots that post-snap causes.
// Falls back to k-means when image-q is unavailable (browser).
function wuQuantMD(samples, k, rng, quantMethod, quantDistance) {
    if (!_iq || k <= 0 || samples.length < k) {
        return kmeans(samples, k, 30, rng).map(snapToMD);
    }
    // Snap every sample to the MD grid first
    const snapped = samples.map(snapToMD);
    const buf = new Uint8Array(snapped.length * 4);
    for (let i = 0; i < snapped.length; i++) {
        buf[i * 4]     = snapped[i][0];
        buf[i * 4 + 1] = snapped[i][1];
        buf[i * 4 + 2] = snapped[i][2];
        buf[i * 4 + 3] = 255;
    }
    const container = _iq.utils.PointContainer.fromUint8Array(buf, snapped.length, 1);
    const palette   = _iq.buildPaletteSync([container], {
        colors:               k,
        paletteQuantization:  quantMethod  || 'wuquant',
        colorDistanceFormula: quantDistance || 'euclidean',
    });
    return palette.getPointContainer().getPointArray().map(p => [p.r, p.g, p.b]);
}

// ── Build partial palette: forced fixed colors + WuQuant fill ────────────── //
// reservedColors: [[r,g,b]] — always included (snapped to MD grid, in order)
// samples: pixels to generate fill colors from
// Remaining slots (up to 15) are filled with WuQuant on pixels not already
// covered by the reserved colors.
function buildPartialPal(reservedColors, samples, rng, thr2, quantMethod, quantDistance) {
    const MAX_SAMPLES = 8000;
    let s = samples;
    if (samples.length > MAX_SAMPLES) {
        const step = Math.ceil(samples.length / MAX_SAMPLES);
        s = samples.filter((_, i) => i % step === 0);
    }

    const seenKeys = new Set();
    const unique   = [];

    for (const raw of reservedColors) {
        const c   = snapToMD(raw);
        const key = c[0] * 65536 + c[1] * 256 + c[2];
        if (!seenKeys.has(key) && unique.length < N_PAL_COLORS) {
            seenKeys.add(key);
            unique.push(c);
        }
    }

    const k = N_PAL_COLORS - unique.length;
    if (k > 0 && s.length > 0) {
        const residual = thr2 > 0
            ? s.filter(p => { for (const c of unique) if (coverDist2(p, c) <= thr2) return false; return true; })
            : s;
        const base = residual.length >= k ? residual : s;
        for (const c of wuQuantMD(base, k, rng, quantMethod, quantDistance)) {
            const key = c[0] * 65536 + c[1] * 256 + c[2];
            if (!seenKeys.has(key)) { seenKeys.add(key); unique.push(c); }
            if (unique.length === N_PAL_COLORS) break;
        }
    }

    while (unique.length < N_PAL_COLORS) unique.push([0, 0, 0]);
    return [[0, 0, 0], ...unique];
}

// ── Build palette: inherit useful fixed colors, fill rest with K-means ───── //
//
// Phase 1 — Inherit: fixed palette colors that cover samples within thr2 are
//   reused in the generated palette (snapped to MD grid, deduped). This makes
//   GEN share boundary colors with FIX palettes → seamless tile transitions.
// Phase 2 — Fill: remaining slots are filled with K-means on pixels still not
//   well covered by the inherited colors.
//
function buildPal0(samples, rng, inheritCandidates, thr2, quantMethod, quantDistance) {
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
            for (const p of s) { if (coverDist2(p, c) <= thr2) count++; }
            if (count > 0) scored.push({ c, key, count });
        }
        scored.sort((a, b) => b.count - a.count);
        for (const { c, key } of scored) {
            if (unique.length >= N_PAL_COLORS) break;
            seenKeys.add(key);
            unique.push(c);
        }
    }

    // ── Phase 2: WuQuant on pixels not covered by inherited colors ────────── //
    const residual = unique.length > 0
        ? s.filter(p => { for (const c of unique) { if (coverDist2(p, c) <= (thr2 || 0)) return false; } return true; })
        : s;

    const k    = N_PAL_COLORS - unique.length;
    const base = residual.length >= k ? residual : s;
    if (k > 0) {
        for (const c of wuQuantMD(base, k, rng, quantMethod, quantDistance)) {
            const key = c[0] * 65536 + c[1] * 256 + c[2];
            if (!seenKeys.has(key)) { seenKeys.add(key); unique.push(c); }
            if (unique.length === N_PAL_COLORS) break;
        }
    }

    // Pad with black if we couldn't fill all slots (rare edge case)
    while (unique.length < N_PAL_COLORS) unique.push([0, 0, 0]);
    return [[0, 0, 0], ...unique];
}

// ── Tile assignment ───────────────────────────────────────────────────────── //
// A fixed palette only wins a tile if it beats the best no-bias palette by fixedBias.
// When tileSemanticGroups is provided, the palette matching the tile's semantic group
// gets a semanticBias discount (multiplied onto its distance), so it wins ties against
// palettes that are only marginally better in colour distance.
function assignTiles(imageData, palettes, W, H, numNoBias, fixedBias, tileSemanticGroups, semanticBias) {
    const TX = Math.floor(W / TILE), TY = Math.floor(H / TILE);
    const tileMap = new Uint8Array(TY * TX);
    const hasSeg = tileSemanticGroups && tileSemanticGroups.length === TY * TX;
    const sBias  = (hasSeg && semanticBias > 0) ? semanticBias : 0;

    for (let ty = 0; ty < TY; ty++) {
        for (let tx = 0; tx < TX; tx++) {
            const flat = [];
            for (let dy = 0; dy < TILE; dy++) {
                for (let dx = 0; dx < TILE; dx++) {
                    const i = ((ty * TILE + dy) * W + (tx * TILE + dx)) * 4;
                    flat.push(imageData[i], imageData[i + 1], imageData[i + 2]);
                }
            }
            const segGroup = hasSeg ? tileSemanticGroups[ty * TX + tx] : -1;

            let bestNoBiasD = Infinity, bestNoBiasPi = 0;
            for (let pi = 0; pi < numNoBias; pi++) {
                let d = tileDist(palettes[pi], flat);
                if (sBias > 0 && pi === segGroup) d *= (1 - sBias);
                if (d < bestNoBiasD) { bestNoBiasD = d; bestNoBiasPi = pi; }
            }
            let minD = bestNoBiasD, minPi = bestNoBiasPi;
            for (let pi = numNoBias; pi < palettes.length; pi++) {
                let d = tileDist(palettes[pi], flat);
                if (sBias > 0 && pi === segGroup) d *= (1 - sBias);
                if (d < minD * fixedBias) { minD = d; minPi = pi; }
            }
            tileMap[ty * TX + tx] = minPi;
        }
    }
    return { tileMap, TX, TY };
}

// ── Row-majority palette enforcement ─────────────────────────────────────── //
// Skips forcing any tile whose semantic group differs from the row winner's group —
// boundary tiles (e.g. a character tile in a sky row) keep their correct palette.
function enforceRowMajority(tileMap, imageData, palettes, TX, TY, W, tileSemanticGroups, numNoBias) {
    const m      = new Uint8Array(tileMap);
    const hasSeg = tileSemanticGroups && tileSemanticGroups.length === TY * TX;

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
        if (sorted.length === 1) continue;

        let mixedDist = 0;
        for (let tx = 0; tx < TX; tx++) {
            const pi   = m[ty * TX + tx];
            const flat = [];
            for (let dy = 0; dy < TILE; dy++)
                for (let dx = 0; dx < TILE; dx++) {
                    const i = ((ty * TILE + dy) * W + (tx * TILE + dx)) * 4;
                    flat.push(imageData[i], imageData[i+1], imageData[i+2]);
                }
            mixedDist += tileDist(palettes[pi], flat);
        }
        const winnerDist = rowTileDist(ty, winner);

        const majority = votes[winner] / TX;
        if (majority <= 0.6 || winnerDist > mixedDist * 1.10) continue;

        let worstTileRatio = 1.0;
        for (let tx = 0; tx < TX; tx++) {
            const pi = m[ty * TX + tx];
            if (pi === winner) continue;
            const flat = [];
            for (let dy = 0; dy < TILE; dy++)
                for (let dx = 0; dx < TILE; dx++) {
                    const i = ((ty * TILE + dy) * W + (tx * TILE + dx)) * 4;
                    flat.push(imageData[i], imageData[i + 1], imageData[i + 2]);
                }
            const curD = tileDist(palettes[pi], flat);
            const winD = tileDist(palettes[winner], flat);
            if (curD > 0) worstTileRatio = Math.max(worstTileRatio, winD / curD);
        }
        if (worstTileRatio > 2.0) continue;

        // Enforce winner, but skip tiles whose semantic group differs from winner's group.
        // Winner palette i < numNoBias was built for group i; fixed palettes have no group.
        for (let tx = 0; tx < TX; tx++) {
            if (hasSeg && winner < numNoBias && tileSemanticGroups[ty * TX + tx] !== winner) continue;
            m[ty * TX + tx] = winner;
        }
    }
    return m;
}

// ── Smooth tile map: remove palette islands (4-neighbor) ─────────────────── //
// When tileSemanticGroups is provided, a tile will not be forced to switch if its
// semantic group differs from the candidate — this preserves boundary tiles
// (e.g. a character surrounded by sky) on their correct palette.
function smoothTileMap(tileMap, imageData, palettes, TX, TY, W, tileSemanticGroups, numNoBias) {
    const m      = new Uint8Array(tileMap);
    const hasSeg = tileSemanticGroups && tileSemanticGroups.length === TY * TX;

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
                const pi    = m[ty * TX + tx];
                const piSeg = hasSeg ? tileSemanticGroups[ty * TX + tx] : -1;

                const neighbors = [];
                if (ty > 0)      neighbors.push(m[(ty - 1) * TX + tx]);
                if (ty < TY - 1) neighbors.push(m[(ty + 1) * TX + tx]);
                if (tx > 0)      neighbors.push(m[ty * TX + (tx - 1)]);
                if (tx < TX - 1) neighbors.push(m[ty * TX + (tx + 1)]);

                const votes = {};
                for (const n of neighbors) {
                    if (n !== pi) votes[n] = (votes[n] || 0) + 1;
                }
                const best = Object.entries(votes).sort((a, b) => b[1] - a[1])[0];
                if (!best) continue;

                const candidate = parseInt(best[0]);
                const voteCount = best[1];

                // Never pull a tile into a palette from a different semantic region.
                // Palette i (i < numNoBias) was built for semantic group i.
                // Fixed palettes (i >= numNoBias) have no group — allow those through.
                if (hasSeg && candidate < numNoBias && candidate !== piSeg) continue;

                const tolerance = voteCount >= 4 ? 2.0 : voteCount >= 3 ? 1.8 : voteCount >= 2 ? 1.4 : 1.2;
                const flat = getTileFlat(ty, tx);
                if (tileDist(palettes[candidate], flat) <= tileDist(palettes[pi], flat) * tolerance) {
                    m[ty * TX + tx] = candidate;
                    changed = true;
                }
            }
        }
        if (!changed) break;
    }

    // ── Final cleanup: completely isolated tiles within the same semantic region //
    for (let ty = 0; ty < TY; ty++) {
        for (let tx = 0; tx < TX; tx++) {
            const pi        = m[ty * TX + tx];
            const piSeg     = hasSeg ? tileSemanticGroups[ty * TX + tx] : -1;
            const neighbors = [];
            const nCoords   = [];
            if (ty > 0)      { neighbors.push(m[(ty-1)*TX+tx]); nCoords.push([ty-1,tx]); }
            if (ty < TY - 1) { neighbors.push(m[(ty+1)*TX+tx]); nCoords.push([ty+1,tx]); }
            if (tx > 0)      { neighbors.push(m[ty*TX+(tx-1)]); nCoords.push([ty,tx-1]); }
            if (tx < TX - 1) { neighbors.push(m[ty*TX+(tx+1)]); nCoords.push([ty,tx+1]); }
            if (neighbors.length < 4) continue;
            if (!neighbors.every(n => n === neighbors[0]) || neighbors[0] === pi) continue;
            const candidate = neighbors[0];
            // Don't cross semantic boundary
            if (hasSeg && candidate < numNoBias && candidate !== piSeg) continue;
            const flat = getTileFlat(ty, tx);
            if (tileDist(palettes[candidate], flat) <= tileDist(palettes[pi], flat) * 2.0) {
                m[ty * TX + tx] = candidate;
            }
        }
    }

    return m;
}

// ── Error diffusion kernels ───────────────────────────────────────────────── //
// Each entry: [dx, dy, numerator]; errors are scaled by (numerator / div) * strength.
const DIFF_KERNELS = {
    fs:       { div: 16, pts: [[ 1,0,7],[-1,1,3],[ 0,1,5],[ 1,1,1]] },
    atkinson: { div:  8, pts: [[ 1,0,1],[ 2,0,1],[-1,1,1],[ 0,1,1],[ 1,1,1],[ 0,2,1]] },
    stucki:   { div: 42, pts: [[ 1,0,8],[ 2,0,4],[-2,1,2],[-1,1,4],[ 0,1,8],[ 1,1,4],[ 2,1,2],[-2,2,1],[-1,2,2],[ 0,2,4],[ 1,2,2],[ 2,2,1]] },
    jarvis:   { div: 48, pts: [[ 1,0,7],[ 2,0,5],[-2,1,3],[-1,1,5],[ 0,1,7],[ 1,1,5],[ 2,1,3],[-2,2,1],[-1,2,3],[ 0,2,5],[ 1,2,3],[ 2,2,1]] },
    sierra:   { div: 32, pts: [[ 1,0,5],[ 2,0,3],[-2,1,2],[-1,1,4],[ 0,1,5],[ 1,1,4],[ 2,1,2],[-1,2,2],[ 0,2,3],[ 1,2,2]] },
};

// ── Generic error diffusion render ───────────────────────────────────────── //
// Error propagates across the whole image; each pixel uses its tile's palette.
// kernelName selects the diffusion matrix (fs, atkinson, stucki, jarvis, sierra).
function renderErrorDiffusion(imageData, W, H, tileMap, palettes, TX, strength, kernelName) {
    const kernel = DIFF_KERNELS[kernelName] || DIFF_KERNELS.fs;
    const outRgb = new Uint8ClampedArray(W * H * 4);
    const outIdx = new Uint8Array(W * H);
    const errR   = new Float32Array(W * H);
    const errG   = new Float32Array(W * H);
    const errB   = new Float32Array(W * H);
    const s = Math.max(0, Math.min(1, strength));

    for (let y = 0; y < H; y++) {
        for (let x = 0; x < W; x++) {
            const pi      = tileMap[Math.floor(y / TILE) * TX + Math.floor(x / TILE)];
            const palette = palettes[pi];
            const src     = (y * W + x) * 4;
            const idx     = y * W + x;

            const r = Math.max(0, Math.min(255, imageData[src]     + errR[idx]));
            const g = Math.max(0, Math.min(255, imageData[src + 1] + errG[idx]));
            const b = Math.max(0, Math.min(255, imageData[src + 2] + errB[idx]));

            const ci  = nearestColor(palette, r, g, b);
            const col = palette[ci];
            outRgb[src]     = col[0];
            outRgb[src + 1] = col[1];
            outRgb[src + 2] = col[2];
            outRgb[src + 3] = 255;
            outIdx[idx] = pi * 16 + ci;

            // Diffuse error only to neighbours that share the same palette —
            // prevents cross-palette contamination (e.g. GEN0 error shifting
            // a green tree pixel into gray when it belongs to FIX0).
            const er = (r - col[0]) * s, eg = (g - col[1]) * s, eb = (b - col[2]) * s;
            const samePal = (nx, ny) =>
                nx >= 0 && nx < W && ny >= 0 && ny < H &&
                tileMap[Math.floor(ny / TILE) * TX + Math.floor(nx / TILE)] === pi;

            for (const [dx, dy, w] of kernel.pts) {
                const nx = x + dx, ny = y + dy;
                if (!samePal(nx, ny)) continue;
                const ni = ny * W + nx, f = w / kernel.div;
                errR[ni] += er * f;
                errG[ni] += eg * f;
                errB[ni] += eb * f;
            }
        }
    }
    return { outRgb, outIdx };
}

// ── Bayer 4×4 ordered dithering render ───────────────────────────────────── //
// Better for animations (no temporal noise). Use DITHER=bayer in CLI.
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
            outRgb[src]     = col[0];
            outRgb[src + 1] = col[1];
            outRgb[src + 2] = col[2];
            outRgb[src + 3] = 255;
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
function rebuildGenPalettes(genPalettes, genPixels, rng, allFixedColors, thr2, quantMethod, quantDistance) {
    return genPalettes.map((pal, g) =>
        genPixels[g].length >= N_PAL_COLORS
            ? buildPal0(genPixels[g], rng, allFixedColors, thr2, quantMethod, quantDistance)
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
// fixedPaletteColors  : [[r,g,b]×15][]  — array of fixed (input) palettes
// numGenerate         : number          — how many fully-generated palettes (default 1)
// partialPaletteConfigs: [{colors:[[r,g,b]], reserveSlots:number}]
//   — palettes where `colors` are locked and remaining slots are auto-generated.
//     The palette appears in the output after fixed palettes.
// tileSemanticGroups  : Uint8Array(TY*TX) | null — optional per-tile semantic group 0..numGenerate-1
//   — when provided, each generated palette i is seeded only from tiles in group i,
//     giving semantically coherent initial palettes instead of luminosity buckets.
// deflate             : async (Uint8Array) => Uint8Array  — injected by caller
// onProgress          : (pct, text) => void               — injected by caller
//
// Output palette order in the indexed PNG:
//   slots 0..numGenerate-1              → generated palettes
//   slots numGenerate..                 → fixed palettes (same order as input)
//   slots after fixed..                 → partial palettes (same order as input)
//
async function processImage({ inputData, fixedPaletteColors, numGenerate, partialPaletteConfigs, tileSemanticGroups, W, H, ditherStrength, ditherMode, residualThr, maxIter, doSmooth, fixedBias, seed, quantMethod, quantDistance }, deflate, onProgress) {
    if (fixedBias === undefined) fixedBias = 0.8;
    const rng = makePrng(seed !== undefined ? seed : 1);
    if (!numGenerate || numGenerate < 0) numGenerate = 0;
    const fixedPalettes   = (fixedPaletteColors || []).map(cols => [[0, 0, 0], ...cols]);
    const partialConfigs  = (partialPaletteConfigs || []).filter(c => c && c.colors && c.colors.length > 0);
    numGenerate = Math.min(numGenerate, Math.max(0, 4 - fixedPalettes.length - partialConfigs.length));
    if (numGenerate === 0 && fixedPalettes.length === 0 && partialConfigs.length === 0) numGenerate = 1;
    const totalPalettes = numGenerate + fixedPalettes.length + partialConfigs.length;
    const thr2 = residualThr * residualThr;
    const allFixedColors = fixedPalettes.flatMap(pal => pal.slice(1));

    // ── Residual analysis ──────────────────────────────────────────────────── //
    onProgress(5, 'Analyzing coverage...');
    const allPixels = Array.from({ length: W * H }, (_, i) =>
        [inputData[i * 4], inputData[i * 4 + 1], inputData[i * 4 + 2]]);
    const residualPixels = allPixels.filter(p => {
        for (const pal of fixedPalettes)
            for (let j = 1; j < pal.length; j++) if (coverDist2(p, pal[j]) <= thr2) return false;
        return true;
    });
    const TX0 = Math.floor(W / TILE);
    const TY0 = Math.floor(H / TILE);
    const numNoBias = numGenerate + partialConfigs.length;

    // ── Semantic-balanced sampling (when AI groups provided) ──────────────── //
    // Collect pixels per semantic group and resample so every group contributes
    // equally to initSamples. This prevents large uniform regions (e.g. sky)
    // from dominating the palette when only one active palette is being built.
    function buildSemanticInitSamples(base) {
        if (!tileSemanticGroups || tileSemanticGroups.length !== TY0 * TX0) return base;
        const numGroups = Math.max(...tileSemanticGroups) + 1;
        const buckets   = Array.from({ length: numGroups }, () => []);
        for (let ty = 0; ty < TY0; ty++) {
            for (let tx = 0; tx < TX0; tx++) {
                const g = tileSemanticGroups[ty * TX0 + tx];
                for (let dy = 0; dy < TILE; dy++) {
                    for (let dx = 0; dx < TILE; dx++) {
                        const i = ((ty * TILE + dy) * W + (tx * TILE + dx)) * 4;
                        buckets[g].push([inputData[i], inputData[i + 1], inputData[i + 2]]);
                    }
                }
            }
        }
        const perGroup = Math.max(N_PAL_COLORS, Math.ceil(base.length / numGroups));
        const balanced = [];
        for (const bucket of buckets) {
            if (bucket.length === 0) continue;
            const step = bucket.length > perGroup ? Math.floor(bucket.length / perGroup) : 1;
            for (let i = 0; i < bucket.length && balanced.length < base.length; i += step)
                balanced.push(bucket[i]);
        }
        return balanced.length >= N_PAL_COLORS ? balanced : base;
    }

    const rawInitSamples = residualPixels.length >= N_PAL_COLORS ? residualPixels : allPixels;
    const initSamples    = buildSemanticInitSamples(rawInitSamples);

    // ── Initial palette generation ─────────────────────────────────────────── //
    let genPalettes = [];
    if (numGenerate > 0) {
        onProgress(15, `Building ${numGenerate} palette(s) (${residualPixels.length} residual px)...`);

        // When semantic tile groups are available, seed each generated palette from
        // pixels belonging to its assigned semantic region instead of a luminosity bucket.
        if (tileSemanticGroups && tileSemanticGroups.length === TY0 * TX0) {
            const semanticSamples = Array.from({ length: numGenerate }, () => []);
            for (let ty = 0; ty < TY0; ty++) {
                for (let tx = 0; tx < TX0; tx++) {
                    const g = Math.min(tileSemanticGroups[ty * TX0 + tx], numGenerate - 1);
                    for (let dy = 0; dy < TILE; dy++) {
                        for (let dx = 0; dx < TILE; dx++) {
                            const i = ((ty * TILE + dy) * W + (tx * TILE + dx)) * 4;
                            semanticSamples[g].push([inputData[i], inputData[i + 1], inputData[i + 2]]);
                        }
                    }
                }
            }
            genPalettes = Array.from({ length: numGenerate }, (_, g) => {
                const group = semanticSamples[g];
                return buildPal0(group.length >= N_PAL_COLORS ? group : initSamples, rng, allFixedColors, thr2, quantMethod, quantDistance);
            });
        } else {
            const sorted = initSamples.slice().sort((a, b) => rgbToOklab(a[0], a[1], a[2])[0] - rgbToOklab(b[0], b[1], b[2])[0]);
            const groupSize = Math.ceil(sorted.length / numGenerate);
            genPalettes = Array.from({ length: numGenerate }, (_, g) => {
                const group = sorted.slice(g * groupSize, (g + 1) * groupSize);
                return buildPal0(group.length >= N_PAL_COLORS ? group : initSamples, rng, allFixedColors, thr2, quantMethod, quantDistance);
            });
        }
    }

    // ── Initial partial palettes (reserved colors + WuQuant fill on all pixels) //
    // When semantic groups are available, seed each partial palette from its
    // assigned group (groups numGenerate..numNoBias-1).
    let partialPalettes = partialConfigs.map((cfg, g) => {
        let samples = initSamples;
        if (tileSemanticGroups && tileSemanticGroups.length === TY0 * TX0) {
            const groupIdx = numGenerate + g;
            const bucket = [];
            for (let ty = 0; ty < TY0; ty++) {
                for (let tx = 0; tx < TX0; tx++) {
                    if (tileSemanticGroups[ty * TX0 + tx] !== groupIdx) continue;
                    for (let dy = 0; dy < TILE; dy++) {
                        for (let dx = 0; dx < TILE; dx++) {
                            const i = ((ty * TILE + dy) * W + (tx * TILE + dx)) * 4;
                            bucket.push([inputData[i], inputData[i + 1], inputData[i + 2]]);
                        }
                    }
                }
            }
            if (bucket.length >= N_PAL_COLORS) samples = bucket;
        }
        return buildPartialPal(cfg.colors, samples, rng, thr2, quantMethod, quantDistance);
    });

    // ── Iterative refinement: assign tiles → rebuild gen + partial palettes ── //
    // Palette order: [gen..., partial..., fixed...]
    // assignTiles treats slots 0..numNoBias-1 as "no bias" (equal footing with generated),
    // and slots numNoBias.. as fixed (need to beat generated by fixedBias to win a tile).
    let palettes = [...genPalettes, ...partialPalettes, ...fixedPalettes];
    let tileMap, TX, TY;

    // Semantic bias: how strongly a tile prefers its semantic-matched palette.
    // 0.25 = that palette's distance is multiplied by 0.75 (25% cheaper) before comparison.
    const semanticBias = tileSemanticGroups ? 0.25 : 0;

    const needIter = numNoBias > 0;
    for (let iter = 1; iter <= (needIter ? maxIter : 0); iter++) {
        onProgress(15 + (iter / maxIter) * 55, `Iteration ${iter}/${maxIter}...`);
        ({ tileMap, TX, TY } = assignTiles(inputData, palettes, W, H, numNoBias, fixedBias, tileSemanticGroups, semanticBias));

        // Rebuild fully-generated palettes (slots 0..numGenerate-1)
        genPalettes = rebuildGenPalettes(genPalettes, collectGenPixels(tileMap, inputData, TX, TY, W, numGenerate), rng, allFixedColors, thr2, quantMethod, quantDistance);

        // Rebuild partial palettes (slots numGenerate..numNoBias-1)
        const partialPixels = collectGenPixels(tileMap, inputData, TX, TY, W, numNoBias).slice(numGenerate);
        partialPalettes = partialConfigs.map((cfg, g) => {
            const px = partialPixels[g] || [];
            return px.length >= N_PAL_COLORS
                ? buildPartialPal(cfg.colors, px, rng, thr2, quantMethod, quantDistance)
                : partialPalettes[g];
        });

        palettes = [...genPalettes, ...partialPalettes, ...fixedPalettes];
    }

    // When no active palettes, the loop above never runs — still need initial tile assignment.
    if (!tileMap) {
        onProgress(70, 'Assigning tiles...');
        ({ tileMap, TX, TY } = assignTiles(inputData, palettes, W, H, numNoBias, fixedBias, tileSemanticGroups, semanticBias));
    }

    // ── Spatial smoothing ──────────────────────────────────────────────────── //
    if (doSmooth) {
        onProgress(73, 'Enforcing row palette bands...');
        tileMap = enforceRowMajority(tileMap, inputData, palettes, TX, TY, W, tileSemanticGroups, numNoBias);
        onProgress(77, 'Smoothing tile islands...');
        tileMap = smoothTileMap(tileMap, inputData, palettes, TX, TY, W, tileSemanticGroups, numNoBias);

        genPalettes = rebuildGenPalettes(genPalettes, collectGenPixels(tileMap, inputData, TX, TY, W, numGenerate), rng, allFixedColors, thr2, quantMethod, quantDistance);

        const partialPixelsS = collectGenPixels(tileMap, inputData, TX, TY, W, numNoBias).slice(numGenerate);
        partialPalettes = partialConfigs.map((cfg, g) => {
            const px = partialPixelsS[g] || [];
            return px.length >= N_PAL_COLORS
                ? buildPartialPal(cfg.colors, px, rng, thr2, quantMethod, quantDistance)
                : partialPalettes[g];
        });

        palettes = [...genPalettes, ...partialPalettes, ...fixedPalettes];
    }

    // ── Tile corrector: fix worst tiles by trying all palettes ────────────── //
    // For each tile, compare its current perceptual score against all other
    // palettes. If any palette is meaningfully better, reassign the tile.
    // Runs one final rebuild of gen/partial palettes after reassignment.
    onProgress(78, 'Correcting tiles…');
    let corrected = 0;
    for (let ty = 0; ty < TY; ty++) {
        for (let tx = 0; tx < TX; tx++) {
            const flat = [];
            for (let dy = 0; dy < TILE; dy++)
                for (let dx = 0; dx < TILE; dx++) {
                    const i = ((ty * TILE + dy) * W + (tx * TILE + dx)) * 4;
                    flat.push(inputData[i], inputData[i + 1], inputData[i + 2]);
                }
            const cur  = tileMap[ty * TX + tx];
            const curD = tileDist(palettes[cur], flat);
            let bestD = curD, bestPi = cur;
            for (let pi = 0; pi < palettes.length; pi++) {
                if (pi === cur) continue;
                const d = tileDist(palettes[pi], flat);
                if (d < bestD) { bestD = d; bestPi = pi; }
            }
            if (bestPi !== cur) { tileMap[ty * TX + tx] = bestPi; corrected++; }
        }
    }
    if (corrected > 0) {
        // Rebuild palettes from corrected tile assignments
        genPalettes = rebuildGenPalettes(genPalettes, collectGenPixels(tileMap, inputData, TX, TY, W, numGenerate), rng, allFixedColors, thr2, quantMethod, quantDistance);
        const partialPixelsC = collectGenPixels(tileMap, inputData, TX, TY, W, numNoBias).slice(numGenerate);
        partialPalettes = partialConfigs.map((cfg, g) => {
            const px = partialPixelsC[g] || [];
            return px.length >= N_PAL_COLORS
                ? buildPartialPal(cfg.colors, px, rng, thr2, quantMethod, quantDistance)
                : partialPalettes[g];
        });
        palettes = [...genPalettes, ...partialPalettes, ...fixedPalettes];
    }

    // ── Render + encode ────────────────────────────────────────────────────── //
    const mode = ditherMode || 'none';
    let outRgb, outIdx;
    if (mode === 'bayer') {
        onProgress(80, 'Rendering (Bayer)...');
        ({ outRgb, outIdx } = renderBayer(inputData, W, H, tileMap, palettes, TX, ditherStrength || 0));
    } else if (mode === 'none') {
        onProgress(80, 'Rendering...');
        ({ outRgb, outIdx } = renderBayer(inputData, W, H, tileMap, palettes, TX, 0));
    } else {
        const label = { fs: 'Floyd-Steinberg', atkinson: 'Atkinson', stucki: 'Stucki', jarvis: 'Jarvis', sierra: 'Sierra' }[mode] || mode;
        onProgress(80, `Rendering (${label})...`);
        ({ outRgb, outIdx } = renderErrorDiffusion(inputData, W, H, tileMap, palettes, TX, (ditherStrength || 75) / 100, mode));
    }

    onProgress(90, 'Encoding PNG...');
    const indexedPng = await encodeIndexedPNG(W, H, outIdx, buildRgbPalette(palettes), deflate);
    const debugData  = buildDebugImage(tileMap, TX, TY, W, H, totalPalettes);
    const { usage, colorUsed } = buildUsageStats(tileMap, outIdx, totalPalettes);

    // Reorder output to match expected: [gen, partial, fixed] — consistent with
    // internal palette array order used during processing.
    return {
        outRgb, outIdx, indexedPng, debugData,
        generatedPaletteColors: genPalettes.map(p => p.slice(1)),
        usage, colorUsed, W, H,
        numGenerate, numPartial: partialConfigs.length, numFixed: fixedPalettes.length,
        palettes,
    };
}

// ── Export (Node.js) — browser loads via importScripts() ─────────────────── //
if (typeof module !== 'undefined') {
    module.exports = { TILE, N_PAL_COLORS, processImage };
}
