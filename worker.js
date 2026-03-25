'use strict';

importScripts('core.js');

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
    const { W, H, residualThr, maxIter, ditherStrength, doSmooth } = options;

    try {
        const result = await processImage(
            { inputData, fixedPaletteColors, numGenerate, W, H, ditherStrength, residualThr, maxIter, doSmooth },
            deflate,
            (pct, text) => postMessage({ type: 'progress', pct, text })
        );
        postMessage({ type: 'done', ...result });
    } catch (err) {
        postMessage({ type: 'error', message: err.message });
    }
};
