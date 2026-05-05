'use strict';

// Module worker — loaded with { type: 'module' } from index.html.
//
// Pipeline:
//   1. SegFormer semantic segmentation  — per-tile region labels
//   2. k-means cluster labels → numGroups — one group per active palette
//
// Input message:  { imageData: Uint8ClampedArray (RGBA), W, H, numGroups }
// Output messages:
//   { type: 'progress', pct: 0-100, text }
//   { type: 'done',     tileGroup: Uint8Array(TY*TX), TX, TY }
//   { type: 'error',    message }

import { pipeline, RawImage } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3';

const TILE      = 8;
const SEG_MODEL = 'Xenova/segformer-b0-finetuned-ade-512-512';

let segmenter = null;

self.onmessage = async ({ data: { imageData, W, H, numGroups } }) => {
    try {
        // ── Semantic segmentation ────────────────────────────────────────── //
        if (!segmenter) {
            postMessage({ type: 'progress', pct: 3, text: 'Downloading segmentation model (~14 MB, cached)…' });
            segmenter = await pipeline('image-segmentation', SEG_MODEL, {
                progress_callback(p) {
                    if (p.status === 'progress')
                        postMessage({
                            type: 'progress',
                            pct:  5 + ((p.progress * 0.35) | 0),
                            text: `Downloading segmentation model: ${p.progress | 0}%`,
                        });
                },
            });
        }

        postMessage({ type: 'progress', pct: 42, text: 'Running semantic segmentation…' });

        const rgb = new Uint8ClampedArray(W * H * 3);
        for (let i = 0; i < W * H; i++) {
            rgb[i * 3]     = imageData[i * 4];
            rgb[i * 3 + 1] = imageData[i * 4 + 1];
            rgb[i * 3 + 2] = imageData[i * 4 + 2];
        }
        const img   = new RawImage(rgb, W, H, 3);
        const masks = await segmenter(img);

        postMessage({ type: 'progress', pct: 72, text: 'Computing tile groups…' });

        // Use original tile grid (W/H), not upscaled size
        const TX = Math.floor(W / TILE);
        const TY = Math.floor(H / TILE);

        if (!masks || masks.length === 0) {
            postMessage({ type: 'done', tileGroup: new Uint8Array(TY * TX), TX, TY });
            return;
        }

        const mW = masks[0].mask.width;
        const mH = masks[0].mask.height;

        // Per-pixel dominant segment (in mask space)
        const pixSeg = new Uint8Array(mW * mH);
        for (let mi = 0; mi < Math.min(masks.length, 254); mi++) {
            const d = masks[mi].mask.data;
            for (let i = 0; i < mW * mH; i++)
                if (d[i] > 0) pixSeg[i] = mi + 1;
        }

        // Per-tile dominant segment (vote within tile, scale mask → original image coords)
        const tileSegId = new Uint8Array(TY * TX);
        for (let ty = 0; ty < TY; ty++) {
            for (let tx = 0; tx < TX; tx++) {
                const votes = {};
                for (let dy = 0; dy < TILE; dy++) {
                    for (let dx = 0; dx < TILE; dx++) {
                        const px = tx * TILE + dx;
                        const py = ty * TILE + dy;
                        const mx = Math.min(mW - 1, Math.round(px * mW / W));
                        const my = Math.min(mH - 1, Math.round(py * mH / H));
                        const s  = pixSeg[my * mW + mx];
                        votes[s] = (votes[s] || 0) + 1;
                    }
                }
                const best = Object.entries(votes).sort((a, b) => b[1] - a[1])[0];
                tileSegId[ty * TX + tx] = best ? +best[0] : 0;
            }
        }

        if (numGroups <= 1) {
            postMessage({ type: 'done', tileGroup: tileSegId, TX, TY });
            return;
        }

        // ── Phase 3: Cluster segment IDs → numGroups ────────────────────── //
        const segColorSum = {};
        const segPixCount = {};
        for (let ty = 0; ty < TY; ty++) {
            for (let tx = 0; tx < TX; tx++) {
                const s = tileSegId[ty * TX + tx];
                if (!segColorSum[s]) { segColorSum[s] = [0, 0, 0]; segPixCount[s] = 0; }
                const cs = segColorSum[s];
                for (let dy = 0; dy < TILE; dy++) {
                    for (let dx = 0; dx < TILE; dx++) {
                        const i = ((ty * TILE + dy) * W + (tx * TILE + dx)) * 4;
                        cs[0] += imageData[i];
                        cs[1] += imageData[i + 1];
                        cs[2] += imageData[i + 2];
                    }
                }
                segPixCount[s] += TILE * TILE;
            }
        }

        const uniqueSegs = Object.keys(segColorSum).map(Number);
        const segMean    = {};
        for (const s of uniqueSegs) {
            const n = segPixCount[s] || 1;
            segMean[s] = segColorSum[s].map(v => v / n);
        }

        const byCount = uniqueSegs.slice().sort((a, b) => (segPixCount[b] || 0) - (segPixCount[a] || 0));
        const centers = byCount.slice(0, numGroups).map(s => segMean[s].slice());
        while (centers.length < numGroups) centers.push([128, 128, 128]);

        const segToGroup = {};
        for (let iter = 0; iter < 20; iter++) {
            for (const s of uniqueSegs) {
                const [r, g, b] = segMean[s];
                let minD = Infinity, minG = 0;
                for (let gi = 0; gi < numGroups; gi++) {
                    const dr = r - centers[gi][0], dg = g - centers[gi][1], db = b - centers[gi][2];
                    const d  = dr * dr + dg * dg + db * db;
                    if (d < minD) { minD = d; minG = gi; }
                }
                segToGroup[s] = minG;
            }
            const sum = Array.from({ length: numGroups }, () => [0, 0, 0]);
            const wt  = new Float64Array(numGroups);
            for (const s of uniqueSegs) {
                const gi = segToGroup[s], w = segPixCount[s] || 0;
                sum[gi][0] += segMean[s][0] * w;
                sum[gi][1] += segMean[s][1] * w;
                sum[gi][2] += segMean[s][2] * w;
                wt[gi] += w;
            }
            let changed = false;
            for (let gi = 0; gi < numGroups; gi++) {
                if (wt[gi] > 0) {
                    const nc = sum[gi].map(v => v / wt[gi]);
                    if (nc.some((v, j) => Math.abs(v - centers[gi][j]) > 0.5)) changed = true;
                    centers[gi] = nc;
                }
            }
            if (!changed) break;
        }

        const tileGroup = new Uint8Array(TY * TX);
        for (let i = 0; i < TY * TX; i++)
            tileGroup[i] = segToGroup[tileSegId[i]] ?? 0;

        postMessage({ type: 'done', tileGroup, TX, TY });

    } catch (err) {
        postMessage({ type: 'error', message: err.message });
    }
};
