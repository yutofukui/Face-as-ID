import { FaceLandmarker, FilesetResolver } from "@mediapipe/tasks-vision";
import { CANONICAL_TRIANGLES, CANONICAL_UVS } from "./face_mesh_data.js";

const DEFAULT_TEXT_ARRAY = ["YUTO", "FACE", "MEDIA ART", "IAMAS", "CREATIVE CODING"];


const FACE_OVAL = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378,
    400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21,
    54, 103, 67, 109
];


const TEXTURE = {
    alpha: 0.95,
    fill: "rgba(0,0,0,1.0)",
    stroke: "rgba(255,255,255,0.8)",
    strokeWidth: 5.0,

    fontFamily: "ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, 'Hiragino Sans', 'Noto Sans JP', sans-serif",
    fontWeight: 900,


    minSize: 20,
    maxSize: 100,
    count: 20,
    scatter: 0.8,
    overlapThreshold: 0.15
};

const WARP = {
    composite: "multiply",
    passes: 1,
    maskFaceOval: true,
    warpExaggeration: 0.5
};

const SMOOTH = {
    smoothPasses: 2,
    resampleStepPx: 3.0
};


const drop = document.getElementById("drop");
const camContainer = document.getElementById("camContainer");
const video = document.getElementById("video");

const startCamBtn = document.getElementById("startCam");
const captureBtn = document.getElementById("captureBtn");
const autoZoomChk = document.getElementById("autoZoom");

const fileInput = document.getElementById("file");


const wordsContainer = document.getElementById("sticker-words-container");
const addWordBtn = document.getElementById("add-word-btn");

const applyBtn = document.getElementById("apply");
const randomizeBtn = document.getElementById("randomize");
const exportBtn = document.getElementById("export");

const srcCanvas = document.getElementById("srcCanvas");
const srcCtx = srcCanvas.getContext("2d");

const outCanvas = document.getElementById("outCanvas");
const outCtx = outCanvas.getContext("2d");

const pngOut = document.getElementById("pngOut");

let faceLandmarker = null;
let srcImage = null;
let seed = 1234;
let stream = null;
let lastLandmarks = null;


async function init() {
    const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.17/wasm"
    );
    faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
        baseOptions: {
            modelAssetPath:
                "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        },
        runningMode: "IMAGE",
        numFaces: 1,
        outputFaceBlendshapes: false,
        outputFacialTransformationMatrixes: false
    });
}
await init();


function renderWordInputs(words) {
    wordsContainer.innerHTML = "";
    words.forEach((w, i) => {
        addWordInput(w);
    });
}

function addWordInput(value = "") {
    const row = document.createElement("div");
    row.className = "sticker-word-row";

    const input = document.createElement("input");
    input.type = "text";
    input.className = "sticker-word-input";
    input.value = value;
    input.placeholder = "WORD";

    const removeBtn = document.createElement("button");
    removeBtn.className = "remove-word-btn";
    removeBtn.innerHTML = "-";
    removeBtn.tabIndex = -1;
    removeBtn.onclick = () => {
        if (wordsContainer.children.length > 1) {
            row.remove();
        }
    };

    row.appendChild(input);
    row.appendChild(removeBtn);
    wordsContainer.appendChild(row);
}

function getStickerWords() {
    const inputs = wordsContainer.querySelectorAll(".sticker-word-input");
    const words = [];
    inputs.forEach(inp => {
        const val = inp.value.trim();
        if (val) words.push(val);
    });

    if (words.length === 0) return ["FACE", "STICKER"];
    return words;
}


renderWordInputs(DEFAULT_TEXT_ARRAY);

addWordBtn.addEventListener("click", () => {
    addWordInput("");
});



function rand01(i) {
    const x = Math.sin(seed + i * 999.123) * 10000;
    return x - Math.floor(x);
}

function randRange(i, min, max) {
    return min + rand01(i) * (max - min);
}

function loadImageFromFile(file) {
    return new Promise((resolve, reject) => {
        const url = URL.createObjectURL(file);
        const img = new Image();
        img.onload = () => {
            URL.revokeObjectURL(url);
            resolve(img);
        };
        img.onerror = reject;
        img.src = url;
    });
}

function drawToCanvas(img, canvas, ctx) {
    const maxSide = 1200;
    let w = img.width || img.naturalWidth || img.videoWidth;
    let h = img.height || img.naturalHeight || img.videoHeight;
    const scale = Math.min(1, maxSide / Math.max(w, h));
    w = Math.round(w * scale);
    h = Math.round(h * scale);

    canvas.width = w;
    canvas.height = h;
    ctx.clearRect(0, 0, w, h);


    if (img.tagName === "VIDEO") {
        ctx.save();
        ctx.translate(w, 0);
        ctx.scale(-1, 1);
        ctx.drawImage(img, 0, 0, w, h);
        ctx.restore();
    } else {
        ctx.drawImage(img, 0, 0, w, h);
    }
}

function extractOvalPoints(landmarks, width, height) {
    return FACE_OVAL.map((idx) => {
        const p = landmarks[idx];
        return { x: p.x * width, y: p.y * height };
    });
}

function smoothClosedPolyline(points, passes = 1) {
    let pts = points.map(p => ({ ...p }));
    for (let k = 0; k < passes; k++) {
        const out = [];
        const n = pts.length;
        for (let i = 0; i < n; i++) {
            const p0 = pts[(i - 1 + n) % n];
            const p1 = pts[i];
            const p2 = pts[(i + 1) % n];
            out.push({
                x: (p0.x + 2 * p1.x + p2.x) / 4,
                y: (p0.y + 2 * p1.y + p2.y) / 4
            });
        }
        pts = out;
    }
    return pts;
}

function resampleClosedPolyline(points, stepPx = 3.0) {
    const n = points.length;
    const dists = [];
    let total = 0;
    for (let i = 0; i < n; i++) {
        const a = points[i];
        const b = points[(i + 1) % n];
        const d = Math.hypot(b.x - a.x, b.y - a.y);
        dists.push(d);
        total += d;
    }
    const count = Math.max(60, Math.floor(total / stepPx));
    const out = [];
    let segIdx = 0;
    let segAcc = 0;

    for (let k = 0; k < count; k++) {
        const target = (k / count) * total;
        while (segAcc + dists[segIdx] < target) {
            segAcc += dists[segIdx];
            segIdx = (segIdx + 1) % n;
        }
        const a = points[segIdx];
        const b = points[(segIdx + 1) % n];
        const t = (target - segAcc) / Math.max(1e-6, dists[segIdx]);
        out.push({ x: a.x + (b.x - a.x) * t, y: a.y + (b.y - a.y) * t });
    }
    return out;
}


function makeTextTextureCanvas(width, height, getWordsFn) {
    const c = document.createElement("canvas");
    c.width = width;
    c.height = height;
    const ctx = c.getContext("2d");

    if (TEXTURE.transparentBg) {
        ctx.clearRect(0, 0, width, height);
    } else {
        ctx.fillStyle = "#ffffff";
        ctx.fillRect(0, 0, width, height);
    }

    const words = getWordsFn();
    if (words.length === 0) return c;

    ctx.textBaseline = "middle";
    ctx.textAlign = "center";
    ctx.lineJoin = "round";

    const cx = width / 2;
    const cy = height / 2;
    const scatterW = width * TEXTURE.scatter * 0.5;
    const scatterH = height * TEXTURE.scatter * 0.5;


    const eyeL = { x: 0.29 * width, y: 0.40 * height };
    const eyeR = { x: 0.71 * width, y: 0.40 * height };


    const exclusionRadius = 0.13 * width;
    const orbitRadius = 0.22 * width;

    const placed = [];
    let placedCount = 0;
    let attempt = 0;
    const maxAttempts = TEXTURE.count * 10;

    const tryPlace = (idx) => {
        const rPos = idx * 13 + attempt * 7;

        const x = cx + (rand01(rPos + 1) - 0.5) * 2 * scatterW;
        const y = cy + (rand01(rPos + 2) - 0.5) * 2 * scatterH;

        const fontSize = randRange(rPos + 4, TEXTURE.minSize, TEXTURE.maxSize);
        const word = words[idx % words.length];
        const approxRadius = (fontSize * 0.6) + (word.length * fontSize * 0.2);


        const distL = Math.hypot(x - eyeL.x, y - eyeL.y);
        const distR = Math.hypot(x - eyeR.x, y - eyeR.y);

        if (distL < exclusionRadius || distR < exclusionRadius) return false;


        let angle;
        if (distL < orbitRadius) {

            angle = Math.atan2(y - eyeL.y, x - eyeL.x) + Math.PI / 2;
        } else if (distR < orbitRadius) {

            angle = Math.atan2(y - eyeR.y, x - eyeR.x) + Math.PI / 2;
        } else {

            angle = (rand01(rPos + 3) - 0.5) * Math.PI * 0.5;
        }


        for (let p of placed) {
            const dist = Math.hypot(p.x - x, p.y - y);
            const rSum = p.radius + approxRadius;
            if (dist < rSum * TEXTURE.overlapThreshold) {
                return false;
            }
        }


        ctx.save();
        ctx.translate(x, y);
        ctx.rotate(angle);

        ctx.font = `${TEXTURE.fontWeight} ${fontSize}px ${TEXTURE.fontFamily}`;

        if (TEXTURE.stroke && TEXTURE.strokeWidth > 0 && TEXTURE.stroke !== "none") {
            ctx.strokeStyle = TEXTURE.stroke;
            ctx.lineWidth = TEXTURE.strokeWidth;
            ctx.strokeText(word, 0, 0);
        }

        ctx.fillStyle = TEXTURE.fill;
        ctx.fillText(word, 0, 0);

        ctx.restore();

        placed.push({ x, y, radius: approxRadius });
        return true;
    };

    while (placedCount < TEXTURE.count && attempt < maxAttempts) {
        if (tryPlace(placedCount)) {
            placedCount++;
        }
        attempt++;
    }

    return c;
}


function drawTriangleWarp(dstCtx, srcCanvas, s0, s1, s2, d0, d1, d2) {
    const sx0 = s0.x, sy0 = s0.y;
    const sx1 = s1.x, sy1 = s1.y;
    const sx2 = s2.x, sy2 = s2.y;

    const dx0 = d0.x, dy0 = d0.y;
    const dx1 = d1.x, dy1 = d1.y;
    const dx2 = d2.x, dy2 = d2.y;

    const denom = (sx0 * (sy1 - sy2) + sx1 * (sy2 - sy0) + sx2 * (sy0 - sy1));
    if (Math.abs(denom) < 1e-6) return;

    const a = (dx0 * (sy1 - sy2) + dx1 * (sy2 - sy0) + dx2 * (sy0 - sy1)) / denom;
    const b = (dy0 * (sy1 - sy2) + dy1 * (sy2 - sy0) + dy2 * (sy0 - sy1)) / denom;
    const c = (dx0 * (sx2 - sx1) + dx1 * (sx0 - sx2) + dx2 * (sx1 - sx0)) / denom;
    const d = (dy0 * (sx2 - sx1) + dy1 * (sx0 - sx2) + dy2 * (sx1 - sx0)) / denom;
    const e = (dx0 * (sx1 * sy2 - sx2 * sy1) + dx1 * (sx2 * sy0 - sx0 * sy2) + dx2 * (sx0 * sy1 - sx1 * sy0)) / denom;
    const f = (dy0 * (sx1 * sy2 - sx2 * sy1) + dy1 * (sx2 * sy0 - sx0 * sy2) + dy2 * (sx0 * sy1 - sx1 * sy0)) / denom;

    dstCtx.save();
    dstCtx.beginPath();
    dstCtx.moveTo(dx0, dy0);
    dstCtx.lineTo(dx1, dy1);
    dstCtx.lineTo(dx2, dy2);
    dstCtx.closePath();
    dstCtx.clip();

    dstCtx.setTransform(a, b, c, d, e, f);
    dstCtx.drawImage(srcCanvas, 0, 0);
    dstCtx.restore();
}


function applyDistortion(uvs) {
    if (WARP.warpExaggeration <= 0) return uvs;
    const cx = 0.500;
    const cy = 0.547;


    const power = 1.0 + WARP.warpExaggeration;

    return uvs.map(uv => {
        let dx = uv.x - cx;
        let dy = uv.y - cy;
        let r = Math.sqrt(dx * dx + dy * dy);

        let new_r = Math.pow(r, power);

        if (r < 1e-6) return { x: cx, y: cy };

        let scale = new_r / r;
        return {
            x: cx + dx * scale,
            y: cy + dy * scale
        };
    });
}


async function applySticker({ random = false } = {}) {
    if (!faceLandmarker || !srcImage) return;

    if (random) seed = Math.floor(Math.random() * 999999);

    const w = srcCanvas.width;
    const h = srcCanvas.height;
    const imageData = srcCtx.getImageData(0, 0, w, h);

    const result = faceLandmarker.detect(imageData);
    if (!result.faceLandmarks || result.faceLandmarks.length === 0) {
        alert("顔が検出できませんでした。");
        return;
    }
    const landmarks = result.faceLandmarks[0];
    lastLandmarks = landmarks;

    const dstPts = landmarks.map(p => ({ x: p.x * w, y: p.y * h }));

    const texW = w;
    const texH = h;


    let uvs = CANONICAL_UVS;
    uvs = applyDistortion(uvs);

    const srcPts = uvs.map(uv => ({
        x: uv.x * texW,
        y: uv.y * texH
    }));

    const tex = makeTextTextureCanvas(texW, texH, getStickerWords);

    outCanvas.width = w;
    outCanvas.height = h;
    outCtx.setTransform(1, 0, 0, 1, 0, 0);
    outCtx.clearRect(0, 0, w, h);
    outCtx.drawImage(srcCanvas, 0, 0);

    let oval = extractOvalPoints(landmarks, w, h);
    oval = smoothClosedPolyline(oval, SMOOTH.smoothPasses);
    oval = resampleClosedPolyline(oval, SMOOTH.resampleStepPx);

    outCtx.save();
    outCtx.globalCompositeOperation = WARP.composite;
    outCtx.globalAlpha = TEXTURE.alpha;

    const tris = CANONICAL_TRIANGLES;
    for (let pass = 0; pass < WARP.passes; pass++) {
        for (let t = 0; t < tris.length; t += 3) {
            const i0 = tris[t], i1 = tris[t + 1], i2 = tris[t + 2];
            if (!dstPts[i0] || !dstPts[i1] || !dstPts[i2]) continue;

            drawTriangleWarp(outCtx, tex, srcPts[i0], srcPts[i1], srcPts[i2], dstPts[i0], dstPts[i1], dstPts[i2]);
        }
    }
    outCtx.restore();

    if (WARP.maskFaceOval) {
        const tmp = document.createElement("canvas");
        tmp.width = w;
        tmp.height = h;
        const tmpCtx = tmp.getContext("2d");
        tmpCtx.drawImage(outCanvas, 0, 0);

        outCtx.clearRect(0, 0, w, h);
        outCtx.drawImage(srcCanvas, 0, 0);

        tmpCtx.globalCompositeOperation = "destination-in";
        tmpCtx.beginPath();
        tmpCtx.moveTo(oval[0].x, oval[0].y);
        for (let i = 1; i < oval.length; i++) tmpCtx.lineTo(oval[i].x, oval[i].y);
        tmpCtx.closePath();
        tmpCtx.fill();

        outCtx.drawImage(tmp, 0, 0);
    }

    exportBtn.disabled = false;
}


function drawWireframe(ctx, landmarks, w, h) {
    ctx.clearRect(0, 0, w, h);


    ctx.fillStyle = "#000000";
    ctx.fillRect(0, 0, w, h);


    ctx.strokeStyle = "rgba(255, 255, 255, 0.4)";
    ctx.lineWidth = 1;
    ctx.lineJoin = "round";

    const tris = CANONICAL_TRIANGLES;

    ctx.beginPath();
    for (let t = 0; t < tris.length; t += 3) {
        const i0 = tris[t], i1 = tris[t + 1], i2 = tris[t + 2];
        const p0 = landmarks[i0], p1 = landmarks[i1], p2 = landmarks[i2];
        if (!p0 || !p1 || !p2) continue;

        const x0 = p0.x * w, y0 = p0.y * h;
        const x1 = p1.x * w, y1 = p1.y * h;
        const x2 = p2.x * w, y2 = p2.y * h;

        ctx.moveTo(x0, y0);
        ctx.lineTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.lineTo(x0, y0);
    }
    ctx.stroke();


    ctx.fillStyle = "rgba(255, 255, 255, 0.8)";
    for (let p of landmarks) {
        const x = p.x * w, y = p.y * h;
        ctx.fillRect(x - 1, y - 1, 2, 2);
    }
}


async function exportPng() {
    if (!lastLandmarks) return;


    const W = 2000;
    const H = 1200;
    const pad = 80;
    const gap = 40;
    const bottomH = 100;

    const c = document.createElement("canvas");
    c.width = W;
    c.height = H;
    const ctx = c.getContext("2d");


    ctx.fillStyle = "#12141a";
    ctx.fillRect(0, 0, W, H);


    const meshCanvas = document.createElement("canvas");
    meshCanvas.width = srcCanvas.width;
    meshCanvas.height = srcCanvas.height;
    const meshCtx = meshCanvas.getContext("2d");
    drawWireframe(meshCtx, lastLandmarks, meshCanvas.width, meshCanvas.height);


    const availW = (W - 2 * pad - gap) / 2;
    const availH = H - 2 * pad - bottomH;


    const imgRatio = srcCanvas.width / srcCanvas.height;

    let drawW = availW;
    let drawH = drawW / imgRatio;

    if (drawH > availH) {
        drawH = availH;
        drawW = drawH * imgRatio;
    }


    const startY = pad + (availH - drawH) / 2;
    const x1 = pad + (availW - drawW) / 2;
    const x2 = pad + availW + gap + (availW - drawW) / 2;


    const drawPanel = (img, x, y, w, h, label) => {
        ctx.save();


        ctx.shadowColor = "rgba(0,0,0,0.5)";
        ctx.shadowBlur = 30;
        ctx.shadowOffsetX = 0;
        ctx.shadowOffsetY = 10;

        const r = 20;
        ctx.beginPath();
        ctx.moveTo(x + r, y);
        ctx.lineTo(x + w - r, y);
        ctx.quadraticCurveTo(x + w, y, x + w, y + r);
        ctx.lineTo(x + w, y + h - r);
        ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
        ctx.lineTo(x + r, y + h);
        ctx.quadraticCurveTo(x, y + h, x, y + h - r);
        ctx.lineTo(x, y + r);
        ctx.quadraticCurveTo(x, y, x + r, y);
        ctx.closePath();


        ctx.fillStyle = "#000";
        ctx.fill();

        ctx.shadowColor = "transparent";
        ctx.clip();
        ctx.drawImage(img, x, y, w, h);


        ctx.strokeStyle = "rgba(255,255,255,0.1)";
        ctx.lineWidth = 2;
        ctx.stroke();

        ctx.restore();


        if (label) {
            ctx.save();
            ctx.fillStyle = "rgba(255,255,255,0.6)";
            ctx.font = "700 16px ui-sans-serif, system-ui, sans-serif";
            ctx.letterSpacing = "0.1em";
            ctx.textAlign = "center";
            ctx.fillText(label, x + w / 2, y + h + 30);
            ctx.restore();
        }
    };

    drawPanel(meshCanvas, x1, startY, drawW, drawH, "FACE MESH GEOMETRY");
    drawPanel(outCanvas, x2, startY, drawW, drawH, "STICKER RESULT");

    ctx.save();
    const footerText = getStickerWords().join(" / ");

    ctx.font = "700 14px ui-sans-serif, system-ui, sans-serif";
    ctx.fillStyle = "rgba(255,255,255,0.4)";
    ctx.textAlign = "center";
    ctx.letterSpacing = "0.05em";
    ctx.fillText(footerText, W / 2, H - 40);
    ctx.restore();


    const idata = ctx.getImageData(0, 0, W, H);
    const data = idata.data;
    for (let i = 0; i < data.length; i += 4) {

        const n = (Math.random() - 0.5) * 10;
        data[i] += n;
        data[i + 1] += n;
        data[i + 2] += n;
    }
    ctx.putImageData(idata, 0, 0);


    const dataUrl = c.toDataURL("image/png");

    const download = () => {
        const link = document.createElement("a");
        link.href = dataUrl;
        link.download = `face_sticker_poster_${Date.now()}.png`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    };

    const isMobile = /iPhone|iPad|iPod|Android/i.test(navigator.userAgent);

    if (isMobile && navigator.canShare && navigator.share) {
        c.toBlob(async (blob) => {
            if (!blob) {
                download();
                return;
            }
            const file = new File([blob], "face_sticker_card.png", { type: "image/png" });
            if (navigator.canShare({ files: [file] })) {
                try {
                    await navigator.share({
                        files: [file],
                        title: "Face Sticker Card",
                    });
                } catch (err) {
                    if (err.name !== "AbortError") {
                        console.error(err);
                        download();
                    }
                }
            } else {
                download();
            }
        });
        return;
    }

    download();
}


async function startCamera() {
    try {
        drop.style.display = "none";
        camContainer.style.display = "block";
        captureBtn.disabled = false;

        stream = await navigator.mediaDevices.getUserMedia({
            video: {
                facingMode: "user",
                width: { ideal: 1920 },
                height: { ideal: 1080 }
            }
        });
        video.srcObject = stream;
    } catch (e) {
        console.error(e);
        alert("Camera access denied or error.");
    }
}

async function captureAndApply() {
    if (!video.srcObject) return;


    const vW = video.videoWidth;
    const vH = video.videoHeight;
    const fullCanvas = document.createElement("canvas");
    fullCanvas.width = vW;
    fullCanvas.height = vH;
    const fullCtx = fullCanvas.getContext("2d");


    fullCtx.translate(vW, 0);
    fullCtx.scale(-1, 1);
    fullCtx.drawImage(video, 0, 0);
    fullCtx.setTransform(1, 0, 0, 1, 0, 0);

    const isAutoZoom = autoZoomChk.checked;

    let finalCanvas = fullCanvas;


    if (isAutoZoom) {
        const result = faceLandmarker.detect(fullCanvas);
        if (result.faceLandmarks && result.faceLandmarks.length > 0) {
            const landmarks = result.faceLandmarks[0];
            let minX = vW, minY = vH, maxX = 0, maxY = 0;
            landmarks.forEach(p => {
                minX = Math.min(minX, p.x * vW);
                minY = Math.min(minY, p.y * vH);
                maxX = Math.max(maxX, p.x * vW);
                maxY = Math.max(maxY, p.y * vH);
            });

            const faceW = maxX - minX;
            const faceH = maxY - minY;
            const cx = (minX + maxX) / 2;
            const cy = (minY + maxY) / 2;

            const cropSize = Math.max(faceW, faceH) * 2.2;

            const sx = Math.max(0, cx - cropSize / 2);
            const sy = Math.max(0, cy - cropSize / 2);
            const sw = Math.min(vW - sx, cropSize);
            const sh = Math.min(vH - sy, cropSize);

            const cropCanvas = document.createElement("canvas");
            const outSize = 1000;
            cropCanvas.width = outSize;
            cropCanvas.height = outSize;
            const cropCtx = cropCanvas.getContext("2d");

            cropCtx.drawImage(fullCanvas, sx, sy, sw, sh, 0, 0, outSize, outSize);
            finalCanvas = cropCanvas;
        }
    }

    srcImage = finalCanvas;

    srcCanvas.width = finalCanvas.width;
    srcCanvas.height = finalCanvas.height;
    srcCtx.drawImage(finalCanvas, 0, 0);


    if (stream) {
        stream.getTracks().forEach(t => t.stop());
        stream = null;
    }
    video.srcObject = null;
    drop.style.display = "block";
    camContainer.style.display = "none";

    applyBtn.disabled = false;
    randomizeBtn.disabled = false;

    await applySticker({ random: true });
}

startCamBtn.addEventListener("click", startCamera);
captureBtn.addEventListener("click", captureAndApply);

drop.addEventListener("click", () => fileInput.click());

drop.addEventListener("dragover", (e) => {
    e.preventDefault();
    drop.classList.add("drag");
});
drop.addEventListener("dragleave", () => drop.classList.remove("drag"));
drop.addEventListener("drop", async (e) => {
    e.preventDefault();
    drop.classList.remove("drag");
    const file = e.dataTransfer.files?.[0];
    if (!file) return;
    await handleFile(file);
});

fileInput.addEventListener("change", async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    await handleFile(file);
    fileInput.value = "";
});

async function handleFile(file) {
    const img = await loadImageFromFile(file);
    srcImage = img;

    drawToCanvas(img, srcCanvas, srcCtx);

    outCanvas.width = srcCanvas.width;
    outCanvas.height = srcCanvas.height;
    outCtx.clearRect(0, 0, outCanvas.width, outCanvas.height);
    outCtx.drawImage(srcCanvas, 0, 0);

    applyBtn.disabled = false;
    randomizeBtn.disabled = false;
    exportBtn.disabled = true;
    pngOut.src = "";
}

applyBtn.addEventListener("click", async () => {
    await applySticker({ random: false });
});

randomizeBtn.addEventListener("click", async () => {
    await applySticker({ random: true });
});

exportBtn.addEventListener("click", async () => {
    await exportPng();
});


const navGen = document.getElementById("nav-gen");
const navAbout = document.getElementById("nav-about");
const viewGen = document.getElementById("view-generate");
const viewAbout = document.getElementById("view-about");

function switchView(viewName) {
    if (viewName === "generate") {
        viewGen.classList.add("active");
        viewAbout.classList.remove("active");
        navGen.classList.add("active");
        navAbout.classList.remove("active");
    } else {
        viewGen.classList.remove("active");
        viewAbout.classList.add("active");
        navGen.classList.remove("active");
        navAbout.classList.add("active");
    }
}

navGen.addEventListener("click", () => switchView("generate"));
navAbout.addEventListener("click", () => switchView("about"));