'use strict';

importScripts('image-q-browser.js', 'core.js');

async function deflate(data) {
    const cs = new CompressionStream('deflate');
    const writer = cs.writable.getWriter();
    const reader = cs.readable.getReader();
    writer.write(data);
    writer.close();
    const chunks = [];
    while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        chunks.push(value);
    }
    const total = chunks.reduce((s, c) => s + c.length, 0);
    const out = new Uint8Array(total);
    let off = 0;
    for (const c of chunks) { out.set(c, off); off += c.length; }
    return out;
}

onmessage = async function (e) {
    const { inputData, fixedPaletteColors, numGenerate, options } = e.data;
    const { W, H, residualThr, maxIter, ditherMode, ditherStrength, doSmooth, fixedBias, seed, quantMethod, quantDistance } = options;

    try {
        const result = await processImage(
            { inputData, fixedPaletteColors, numGenerate, W, H, ditherMode, ditherStrength, residualThr, maxIter, doSmooth, fixedBias, seed, quantMethod, quantDistance },
            deflate,
            (pct, text) => postMessage({ type: 'progress', pct, text })
        );
        postMessage({ type: 'done', ...result });
    } catch (err) {
        postMessage({ type: 'error', message: err.message });
    }
};
