// NeuroLens, browser edition: the whole pipeline from model.py, running client-side.
// ResNet18 runs with onnxruntime-web (WASM). The heatmap is the class activation map,
// which for this architecture equals Grad-CAM exactly (see tools/export_web.py).
/* global ort */

const ORT_BASE = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";
const MODEL_URL = "model/neurolens.onnx";
const HEADS_URL = "model/heads.json";
const MESH_URL = "brain.bin";
const EXAMPLES = ["glioma_1", "glioma_2", "meningioma_1", "meningioma_2", "notumor_1", "notumor_2", "pituitary_1", "pituitary_2"];

const CLASSES = ["glioma", "meningioma", "notumor", "pituitary"];
const PRETTY = { glioma: "Glioma", meningioma: "Meningioma", notumor: "No tumor", pituitary: "Pituitary" };
const ACCENT = { glioma: "#ff5d8f", meningioma: "#ffb547", notumor: "#3ee6a8", pituitary: "#8b7bff" };
const BLURB = {
  glioma: "Tumor arising from glial cells, usually within the brain tissue itself.",
  meningioma: "Tumor of the meninges, the membranes wrapping the brain; often near the skull edge.",
  notumor: "No tumor pattern detected in this scan.",
  pituitary: "Tumor of the pituitary gland at the base of the brain, behind the eyes.",
};
const VIEW_CHOICES = [
  ["Auto-detect", null], ["Axial (top-down)", "axial"], ["Coronal (front)", "coronal"],
  ["Sagittal · face left", "sagittal:left"], ["Sagittal · face right", "sagittal:right"],
];
const VIEW_LABEL = { axial: "Axial", coronal: "Coronal", sagittal: "Sagittal" };
const MEAN = [0.485, 0.456, 0.406], STD = [0.229, 0.224, 0.225];
const N = 224;

// heat colours: deep violet -> magenta -> orange -> pale yellow
const HEAT_STOPS = [0, 0.25, 0.5, 0.75, 1];
const HEAT_RGB = [[12, 7, 50], [110, 20, 140], [225, 50, 110], [255, 150, 40], [255, 250, 190]];

// template geometry (mm, RAS: x+ right, y+ anterior, z+ up); same numbers as model.py
const CENTER = [0, -18, 15];
const PITUITARY = [0, 2, -34];
const HEAD_WIDTH = 150, HEAD_LENGTH = 205, SCALP_TOP = 88, SCALP_BACK = -112, SCALP_FRONT = 92;
const DEPTH_AXIS = { axial: 2, coronal: 1, sagittal: 0 };

const $ = (s) => document.querySelector(s);
const state = { session: null, heads: null, mesh: null, ready: null, image: null, cache: null,
  explain: "top", view: null, opacity: 0.65 };

// ------------------------------------------------------------------ loading
async function fetchBytes(url, onProgress) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  const total = +res.headers.get("content-length") || 0;
  if (!res.body || !total) return new Uint8Array(await res.arrayBuffer());
  const reader = res.body.getReader();
  const out = new Uint8Array(total);
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    if (got + value.length > out.length) return new Uint8Array(await new Blob([out.subarray(0, got), value]).arrayBuffer());
    out.set(value, got);
    got += value.length;
    onProgress?.(got / total);
  }
  return out.subarray(0, got);
}

async function loadAll() {
  renderLoading("Loading the model", null);
  ort.env.wasm.wasmPaths = ORT_BASE;
  ort.env.wasm.numThreads = 1;
  const [bytes, heads, mesh] = await Promise.all([
    fetchBytes(MODEL_URL, (f) => renderLoading("Loading the model", f)),
    fetch(HEADS_URL).then((r) => r.json()),
    fetch(MESH_URL).then((r) => r.arrayBuffer()),
  ]);
  renderLoading("Starting the model", null);
  state.session = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"] });
  state.heads = heads;
  const dv = new DataView(mesh);
  const nv = dv.getUint32(0, true);
  state.mesh = new Float32Array(mesh, 8, nv * 3);
  if (!state.image) $("#card").innerHTML = emptyCard();
}

// ------------------------------------------------------------------ image -> tensors
function drawSquare(img, mirror = false) {
  const c = document.createElement("canvas");
  c.width = c.height = N;
  const g = c.getContext("2d", { willReadFrequently: true });
  g.imageSmoothingEnabled = true;
  g.imageSmoothingQuality = "high";
  if (mirror) { g.translate(N, 0); g.scale(-1, 1); }
  g.drawImage(img, 0, 0, N, N);
  return g.getImageData(0, 0, N, N).data;
}

function toTensor(px) {
  const t = new Float32Array(3 * N * N);
  for (let i = 0; i < N * N; i++) {
    for (let c = 0; c < 3; c++) t[c * N * N + i] = (px[i * 4 + c] / 255 - MEAN[c]) / STD[c];
  }
  return new ort.Tensor("float32", t, [1, 3, N, N]);
}

function toGray(px) {  // PIL "L" conversion
  const g = new Float32Array(N * N);
  for (let i = 0; i < N * N; i++) {
    g[i] = ((px[i * 4] * 19595 + px[i * 4 + 1] * 38470 + px[i * 4 + 2] * 7471 + 0x8000) >> 16) / 255;
  }
  return g;
}

async function runModel(px) {
  const out = await state.session.run({ x: toTensor(px) });
  return { logits: out.logits.data, acts: out.acts.data, pooled: out.pooled.data };
}

const softmax = (v) => {
  const m = Math.max(...v);
  const e = Array.from(v, (x) => Math.exp(x - m));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((x) => x / s);
};
const argmax = (v) => v.reduce((bi, x, i, a) => (x > a[bi] ? i : bi), 0);

// class activation map, upsampled like F.interpolate(bilinear, align_corners=False), min-max scaled
function classMap(acts, target) {
  const w = state.heads.fc_weight.slice(target * 512, target * 512 + 512);
  const small = new Float32Array(49);
  for (let i = 0; i < 49; i++) {
    let s = 0;
    for (let k = 0; k < 512; k++) s += w[k] * acts[k * 49 + i];
    small[i] = Math.max(0, s);
  }
  const cam = new Float32Array(N * N);
  const idx = (o) => {
    const s = Math.max(0, (o + 0.5) * (7 / N) - 0.5);
    const i0 = Math.floor(s);
    return [i0, i0 < 6 ? i0 + 1 : i0, s - i0];
  };
  const cols = Array.from({ length: N }, (_, x) => idx(x));
  let lo = Infinity, hi = -Infinity;
  for (let y = 0; y < N; y++) {
    const [y0, y1, ly] = idx(y);
    for (let x = 0; x < N; x++) {
      const [x0, x1, lx] = cols[x];
      const v = (1 - ly) * ((1 - lx) * small[y0 * 7 + x0] + lx * small[y0 * 7 + x1])
              + ly * ((1 - lx) * small[y1 * 7 + x0] + lx * small[y1 * 7 + x1]);
      cam[y * N + x] = v;
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
  }
  for (let i = 0; i < cam.length; i++) cam[i] = (cam[i] - lo) / (hi - lo + 1e-8);
  return cam;
}

// ------------------------------------------------------------------ scan geometry
function percentile(arr, q) {
  const s = Float32Array.from(arr).sort();
  const pos = (q / 100) * (s.length - 1);
  const lo = Math.floor(pos), hi = Math.ceil(pos);
  return s[lo] + (s[hi] - s[lo]) * (pos - lo);
}

function headMask(gray) {
  const thr = Math.max(0.08, percentile(gray, 60) * 0.5);
  const xs = [], ys = [];
  const mask = new Uint8Array(N * N);
  for (let i = 0; i < N * N; i++) if (gray[i] > thr) { mask[i] = 1; xs.push(i % N); ys.push((i / N) | 0); }
  return { mask, xs, ys };
}

function geometryFeatures(gray, hm) {
  const { mask, xs, ys } = hm;
  if (xs.length < 50) return [0.5, 0, 1];
  const cx = Math.round(xs.reduce((a, b) => a + b, 0) / xs.length);
  const half = Math.max(Math.min(cx, N - cx) - 1, 4);
  const a = [], b = [];
  for (let y = 0; y < N; y++) {
    for (let j = 0; j < half; j++) {
      a.push(gray[y * N + (cx - half + j)]);
      b.push(gray[y * N + (cx + half - j)]);
    }
  }
  const ma = a.reduce((s, v) => s + v, 0) / a.length, mb = b.reduce((s, v) => s + v, 0) / b.length;
  let sab = 0, saa = 0, sbb = 0;
  for (let i = 0; i < a.length; i++) {
    const da = a[i] - ma, db = b[i] - mb;
    sab += da * db; saa += da * da; sbb += db * db;
  }
  const sym = saa && sbb ? sab / Math.sqrt(saa * sbb) : 0;
  let bottom = 0;
  for (let i = (N - 20) * N; i < N * N; i++) bottom += mask[i];
  bottom /= 20 * N;
  const h = Math.max(...ys) - Math.min(...ys), w = Math.max(...xs) - Math.min(...xs);
  return [sym, bottom, h / Math.max(w, 1)];
}

function detectView(pooled, geo) {
  const v = state.heads.view;
  const input = [...pooled];
  for (let r = 0; r < 8; r++) input.push(...geo);
  const logits = [0, 1, 2].map((j) => {
    let s = v.b[j];
    for (let i = 0; i < input.length; i++) s += ((input[i] - v.mu[i]) / v.sd[i]) * v.W[i * 3 + j];
    return s;
  });
  const p = softmax(logits);
  const k = argmax(p);
  return { view: v.views[k], conf: p[k] };
}

async function detectFacing(pooled) {
  if (!state.cache.mirrorPooled) {
    state.cache.mirrorPooled = (await runModel(drawSquare(state.image, true))).pooled;
  }
  const f = state.heads.facing;
  let s = 0;
  for (let i = 0; i < 512; i++) s += ((pooled[i] - state.cache.mirrorPooled[i]) / f.sd[i]) * f.w[i];
  return s > 0 ? "right" : "left";
}

// ------------------------------------------------------------------ 3D placement (mirrors model.py)
function sliceFrame(hm, view, facing) {
  let { xs, ys } = hm;
  if (xs.length < 50) { xs = [20, 204]; ys = [20, 204]; }
  const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys);
  const mm = (view === "sagittal" ? HEAD_LENGTH : HEAD_WIDTH) / Math.max(x1 - x0, 1);
  const at = (px, py, depth) => {
    if (view === "sagittal") {
      const along = facing === "right" ? (px - x0) * mm : (x1 - px) * mm;
      return [depth, SCALP_BACK + along, SCALP_TOP - (py - y0) * mm];
    }
    const x = ((x0 + x1) / 2) * mm - px * mm;  // radiological: patient's left on image right
    if (view === "axial") return [x, SCALP_FRONT - 8 - (py - y0) * mm, depth];
    return [x, depth, SCALP_TOP - (py - y0) * mm];
  };
  const r1 = (v) => v.map((n) => Math.round(n * 10) / 10);
  return {
    mm, at, axis: DEPTH_AXIS[view],
    plane: (d) => [[0, 0], [N, 0], [0, N], [N, N]].map(([px, py]) => r1(at(px, py, d))),
  };
}

function hotspot(cam) {
  let sw = 0, sx = 0, sy = 0, count = 0, best = 0;
  for (let i = 0; i < cam.length; i++) {
    if (cam[i] > cam[best]) best = i;
    if (cam[i] >= 0.6) { sw += cam[i]; sx += cam[i] * (i % N); sy += cam[i] * ((i / N) | 0); count++; }
  }
  if (!count) return { cx: best % N, cy: (best / N) | 0, count: 0 };
  return { cx: sx / sw, cy: sy / sw, count };
}

function columnCenter(axis, known) {
  const m = state.mesh;
  let lo = Infinity, hi = -Infinity, n = 0;
  for (let i = 0; i < m.length; i += 3) {
    let ok = true;
    for (const [a, val] of known) if (Math.abs(m[i + a] - val) >= 7) { ok = false; break; }
    if (!ok) continue;
    const v = m[i + axis];
    if (v < lo) lo = v;
    if (v > hi) hi = v;
    n++;
  }
  return n < 3 ? CENTER[axis] : (lo + hi) / 2;
}

function surfaceRadius(dir) {
  const m = state.mesh, cosMin = Math.cos((9 * Math.PI) / 180);
  let best = 0, all = 0;
  for (let i = 0; i < m.length; i += 3) {
    const rx = m[i] - CENTER[0], ry = m[i + 1] - CENTER[1], rz = m[i + 2] - CENTER[2];
    const r = Math.hypot(rx, ry, rz);
    if (r > all) all = r;
    if ((rx * dir[0] + ry * dir[1] + rz * dir[2]) / (r + 1e-6) > cosMin && r > best) best = r;
  }
  return best || all;
}

function regionName(p, view) {
  const [x, y, z] = p;
  const side = view === "sagittal" ? "" : x > 0 ? "Right " : "Left ";
  const cap = (s) => s.charAt(0).toUpperCase() + s.slice(1);
  if (Math.hypot(x - PITUITARY[0], y - PITUITARY[1], z - PITUITARY[2]) < 14) return "Sellar region (pituitary gland)";
  if (Math.abs(x) < 22 && y > -45 && y < 10 && z > -15 && z < 30) return cap(`${side}deep brain (thalamus / basal ganglia)`.trim().toLowerCase());
  let lobe;
  if (y < -68) lobe = "occipital lobe";
  else if (z < -8 && Math.abs(x) > 28 && y > -50) lobe = "temporal lobe";
  else if (y > -12) lobe = "frontal lobe";
  else lobe = "parietal lobe";
  return cap(`${side}${lobe}`);
}

function locate(cam, hm, view, facing, cls) {
  const frame = sliceFrame(hm, view, facing);
  const a = frame.axis;
  if (cls === "pituitary") {
    return { pos: PITUITARY, radius: 9, region: regionName(PITUITARY, view), plane: frame.plane(PITUITARY[a]) };
  }
  const { cx, cy, count } = hotspot(cam);
  const p = frame.at(cx, cy, NaN);
  const known = [0, 1, 2].filter((i) => i !== a).map((i) => [i, p[i]]);
  let depth = columnCenter(a, known);
  if (view === "sagittal") depth += cls === "meningioma" ? 12 : 28;
  p[a] = depth;

  const d = p.map((v, i) => v - CENTER[i]);
  const dist = Math.hypot(...d) + 1e-6;
  const dir = d.map((v) => v / dist);
  const surf = surfaceRadius(dir);
  let q = p;
  if (cls === "meningioma") q = CENTER.map((c, i) => c + dir[i] * surf);
  else if (dist > surf * 0.78) q = CENTER.map((c, i) => c + dir[i] * surf * 0.72);
  const radius = Math.min(16, Math.max(7, Math.sqrt(count / Math.PI) * frame.mm * 0.28));
  return {
    pos: q.map((v) => Math.round(v * 10) / 10), radius: Math.round(radius * 10) / 10,
    region: regionName(q, view), plane: frame.plane(q[a]),
  };
}

// ------------------------------------------------------------------ rendering
function heatRGB(v) {
  let k = 0;
  while (k < 3 && v > HEAT_STOPS[k + 1]) k++;
  const t = (v - HEAT_STOPS[k]) / (HEAT_STOPS[k + 1] - HEAT_STOPS[k]);
  return [0, 1, 2].map((c) => HEAT_RGB[k][c] + (HEAT_RGB[k + 1][c] - HEAT_RGB[k][c]) * Math.min(1, Math.max(0, t)));
}

function overlayCanvas(gray, cam, alpha, size) {
  const small = document.createElement("canvas");
  small.width = small.height = N;
  const g = small.getContext("2d");
  const img = g.createImageData(N, N);
  for (let i = 0; i < N * N; i++) {
    const base = gray[i] * 255;
    const a = alpha * Math.min(1, cam[i] * 1.4);
    const h = heatRGB(cam[i]);
    for (let c = 0; c < 3; c++) img.data[i * 4 + c] = base * (1 - a) + h[c] * a;
    img.data[i * 4 + 3] = 255;
  }
  g.putImageData(img, 0, 0);
  const out = document.createElement("canvas");
  out.width = out.height = size;
  const o = out.getContext("2d");
  o.imageSmoothingQuality = "high";
  o.drawImage(small, 0, 0, size, size);
  return out;
}

function viewText(view, conf, facing, manual) {
  let name = `${VIEW_LABEL[view]} slice${facing ? ` (face ${facing})` : ""}`;
  if (manual) return `${name} · set manually`;
  if (conf < 0.7) return `${name} · low-confidence guess (${Math.round(conf * 100)}%), set it in Scan view`;
  return `${name} · auto-detected ${Math.round(conf * 100)}%`;
}

function resultCard(probs, target, vtext) {
  const top = argmax(probs), name = CLASSES[top];
  const rows = [...probs.keys()].sort((i, j) => probs[j] - probs[i]).map((i) => `
    <div class="nl-row">
      <span class="nl-row-name">${PRETTY[CLASSES[i]]}</span>
      <div class="nl-bar"><div class="nl-fill" style="--w:${(probs[i] * 100).toFixed(1)}%;--c:${ACCENT[CLASSES[i]]}"></div></div>
      <span class="nl-row-pct">${(probs[i] * 100).toFixed(1)}%</span>
    </div>`).join("");
  const explain = target === top ? "" :
    `<div class="nl-note">Showing evidence for <b>${PRETTY[CLASSES[target]]}</b>, not the top prediction.</div>`;
  return `
    <div class="nl-card" style="--accent:${ACCENT[name]}">
      <div class="nl-kicker">Model prediction</div>
      <div class="nl-head">
        <div class="nl-label">${PRETTY[name]}</div>
        <div class="nl-ringwrap"><div class="nl-ring" style="--p:${(probs[top] * 100).toFixed(1)}"></div><b>${Math.round(probs[top] * 100)}%</b></div>
      </div>
      <div class="nl-blurb">${BLURB[name]}</div>
      <div class="nl-rows">${rows}</div>
      ${explain}
      <div class="nl-view">${vtext}</div>
    </div>`;
}

const emptyCard = () => `
  <div class="nl-card nl-empty">
    <div class="nl-kicker">Model prediction</div>
    <div class="nl-label" style="opacity:.35">Waiting for a scan</div>
    <div class="nl-blurb">Upload an MRI slice or pick an example to see what the network sees.</div>
  </div>`;

function renderLoading(text, frac) {
  $("#card").innerHTML = `
    <div class="nl-card nl-empty nl-loading">
      <div class="nl-kicker">Model prediction</div>
      <div class="nl-label">${text}…</div>
      <div class="nl-blurb">${frac == null ? "One moment." : `${Math.round(frac * 100)}% of 45 MB, downloaded once and cached by your browser.`}</div>
      <div class="progress ${frac == null ? "indeterminate" : ""}"><div style="--p:${((frac || 0) * 100).toFixed(0)}%"></div></div>
    </div>`;
}

function renderError(err) {
  console.error(err);
  $("#card").innerHTML = `
    <div class="nl-card nl-error">
      <div class="nl-kicker">Something went wrong</div>
      <div class="nl-label">Couldn't run the model</div>
      <div class="nl-blurb">${String(err?.message || err).replace(/</g, "&lt;")}</div>
    </div>`;
}

// ------------------------------------------------------------------ main flow
async function setImage(src, label) {
  const img = new Image();
  img.crossOrigin = "anonymous";
  img.src = src;
  await img.decode();
  state.image = img;
  state.cache = null;
  $("#preview").src = src;
  $("#preview").hidden = false;
  $("#drop-empty").hidden = true;
  $("#clear").hidden = false;
  document.querySelectorAll(".ex-grid button").forEach((b) => b.classList.toggle("active", b.dataset.name === label));
  await analyze();
}

function clearImage() {
  state.image = null;
  state.cache = null;
  $("#preview").hidden = true;
  $("#preview").removeAttribute("src");
  $("#drop-empty").hidden = false;
  $("#clear").hidden = true;
  $("#file").value = "";
  document.querySelectorAll(".ex-grid button").forEach((b) => b.classList.remove("active"));
  $("#card").innerHTML = state.session ? emptyCard() : $("#card").innerHTML;
  $("#flat").getContext("2d").clearRect(0, 0, 448, 448);
  $("#flat-empty").hidden = false;
  window.NeuroBrain?.update({ state: "empty", nonce: Date.now() });
}

let busy = Promise.resolve();
function analyze() {
  busy = busy.then(analyzeNow).catch(renderError);
  return busy;
}

async function analyzeNow() {
  if (!state.image) return;
  await state.ready;
  if (!state.cache) {
    const px = drawSquare(state.image);
    const gray = toGray(px);
    const hm = headMask(gray);
    state.cache = { gray, hm, geo: geometryFeatures(gray, hm), ...(await runModel(px)) };
  }
  const c = state.cache;
  const probs = softmax(c.logits);
  const top = argmax(probs);
  const target = state.explain === "top" ? top : CLASSES.indexOf(state.explain);
  const cam = classMap(c.acts, target);

  let view, conf, facing = null;
  const manual = state.view != null;
  if (manual) {
    [view, facing = null] = state.view.split(":");
    conf = 1;
  } else {
    ({ view, conf } = detectView(c.pooled, c.geo));
    if (view === "sagittal") facing = await detectFacing(c.pooled);
  }
  const vtext = viewText(view, conf, facing, manual);
  const cls = CLASSES[target];
  let loc;
  if (cls === "notumor") {
    const frame = sliceFrame(c.hm, view, facing);
    loc = { plane: frame.plane(CENTER[frame.axis] + (view === "sagittal" ? 25 : 0)) };
  } else {
    loc = locate(cam, c.hm, view, facing, cls);
  }

  // card, 2D overlay, 3D viewer
  $("#card").innerHTML = resultCard(probs, target, vtext);
  const flat = overlayCanvas(c.gray, cam, state.opacity, 448);
  $("#flat").getContext("2d").drawImage(flat, 0, 0);
  $("#flat-empty").hidden = true;

  const base = {
    nonce: Date.now(), view, view_conf: conf, view_text: vtext, plane: loc.plane,
    slice: overlayCanvas(c.gray, cam, 0.55, 256).toDataURL("image/png"), top_label: PRETTY[CLASSES[top]],
  };
  const payload = loc.pos ? {
    ...base, state: "tumor", label: PRETTY[cls], color: ACCENT[cls], conf: probs[target], is_top: target === top,
    pos: loc.pos, radius: loc.radius, region: loc.region, blurb: BLURB[cls],
  } : {
    ...base, state: "clear",
    clear_text: CLASSES[top] === "notumor" ? `No tumor detected · ${(probs[2] * 100).toFixed(1)}%`
      : "Showing 'No tumor' evidence: nothing to mark",
  };
  window.NeuroBrain?.update(payload);
  state.last = { probs, view, conf, facing, pos: loc.pos ?? null, radius: loc.radius ?? null, region: loc.region ?? null };
  return state.last;
}

// ------------------------------------------------------------------ UI wiring
function buildControls() {
  const chip = (group, value, text, checked) =>
    `<label><input type="radio" name="${group}" value="${value}" ${checked ? "checked" : ""}>${text}</label>`;
  $("#explain").innerHTML = chip("explain", "top", "Top prediction", true)
    + CLASSES.map((c) => chip("explain", c, PRETTY[c], false)).join("");
  $("#viewpick").innerHTML = VIEW_CHOICES.map(([t, v], i) => chip("view", v ?? "", t, i === 0)).join("");
  $("#explain").addEventListener("change", (e) => { state.explain = e.target.value; analyze(); });
  $("#viewpick").addEventListener("change", (e) => { state.view = e.target.value || null; analyze(); });
  $("#opacity").addEventListener("input", (e) => {
    state.opacity = +e.target.value;
    $("#opacity-val").textContent = state.opacity.toFixed(2);
  });
  $("#opacity").addEventListener("change", () => analyze());

  $("#examples").innerHTML = EXAMPLES.map((n) =>
    `<button data-name="${n}" title="${n.replace("_", " ")}"><img src="examples/${n}.jpg" alt="${n.replace("_", " ")} example" loading="lazy"></button>`).join("");
  $("#examples").addEventListener("click", (e) => {
    const b = e.target.closest("button");
    if (b) setImage(`examples/${b.dataset.name}.jpg`, b.dataset.name).catch(renderError);
  });

  const drop = $("#drop"), file = $("#file");
  const takeFile = (f) => f && f.type.startsWith("image/") && setImage(URL.createObjectURL(f), null).catch(renderError);
  drop.addEventListener("click", (e) => { if (e.target.id !== "clear") file.click(); });
  drop.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); file.click(); } });
  file.addEventListener("change", () => takeFile(file.files[0]));
  drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("over"); });
  drop.addEventListener("dragleave", () => drop.classList.remove("over"));
  drop.addEventListener("drop", (e) => { e.preventDefault(); drop.classList.remove("over"); takeFile(e.dataTransfer.files[0]); });
  document.addEventListener("paste", (e) => {
    const item = [...(e.clipboardData?.items || [])].find((i) => i.type.startsWith("image/"));
    if (item) takeFile(item.getAsFile());
  });
  $("#clear").addEventListener("click", (e) => { e.stopPropagation(); clearImage(); });

  document.querySelectorAll(".tabs button").forEach((b) => b.addEventListener("click", () => {
    document.querySelectorAll(".tabs button").forEach((x) => x.setAttribute("aria-selected", String(x === b)));
    document.querySelectorAll(".tabpanel").forEach((p) => { p.hidden = p.dataset.panel !== b.dataset.tab; });
  }));
}

buildControls();
state.ready = loadAll().catch((e) => { renderError(e); throw e; });

// used by the automated check against the Python pipeline
window.__neurolens = {
  async analyzeUrl(url) {
    await setImage(url, null);
    return state.last;
  },
};
