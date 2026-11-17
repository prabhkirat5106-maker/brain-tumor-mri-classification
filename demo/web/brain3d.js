// NeuroLens 3D brain viewer.
// Template cortex (fsaverage5 pial) + stylised cerebellum and brainstem, the tumor placed
// from the model's Grad-CAM hotspot, and the uploaded slice floating in its own plane.
// Python sends a payload through window.NeuroBrain.update(payload).
import * as THREE from "https://cdn.jsdelivr.net/npm/three@0.170.0/build/three.module.js";

// RAS mm (x+ right, y+ anterior, z+ up) -> three (x right, y up, z toward the back)
const C = { x: 0, y: -18, z: 15 };
const toThree = ([x, y, z]) => new THREE.Vector3(x - C.x, z - C.z, -(y - C.y));
const toRAS = (v) => [v.x + C.x, -v.z + C.y, v.y + C.z];

const GYRUS = new THREE.Color("#7db9ff");
const SULCUS = new THREE.Color("#0e2373");
const TUMOR_RED = new THREE.Color("#ff2244");
const DEFAULT_ROT = { x: 0.22, y: 1.05 };

const U = {
  uTime: { value: 0 },
  uGlowColor: { value: new THREE.Color("#ff1a3a") },
  uRim: { value: 0.55 },
  uRimColor: { value: new THREE.Color("#6fc3ff") },
  uPulse: { value: 1 },
};

function brainMaterial() {
  const m = new THREE.MeshPhysicalMaterial({
    vertexColors: true, roughness: 0.42, metalness: 0.0,
    clearcoat: 0.7, clearcoatRoughness: 0.28, transparent: true, opacity: 1,
  });
  m.onBeforeCompile = (s) => {
    Object.assign(s.uniforms, U);
    s.vertexShader = s.vertexShader
      .replace("#include <common>", "#include <common>\nattribute float glow;\nvarying float vGlow;")
      .replace("#include <begin_vertex>", "#include <begin_vertex>\nvGlow = glow;");
    s.fragmentShader = s.fragmentShader
      .replace("#include <common>", `#include <common>
        varying float vGlow; uniform float uTime; uniform vec3 uGlowColor;
        uniform float uRim; uniform vec3 uRimColor; uniform float uPulse;`)
      .replace("#include <emissivemap_fragment>", `#include <emissivemap_fragment>
        float pulse = 0.6 + 0.4 * sin(uTime * 3.0);
        diffuseColor.rgb = mix(diffuseColor.rgb, uGlowColor, clamp(vGlow * 1.3, 0.0, 1.0));
        totalEmissiveRadiance += uGlowColor * vGlow * (0.3 + 0.5 * pulse * uPulse);
        float fres = pow(1.0 - clamp(dot(normal, normalize(vViewPosition)), 0.0, 1.0), 2.4);
        totalEmissiveRadiance += uRimColor * fres * uRim;`);
  };
  return m;
}

function withAttributes(geo, shade) {
  const n = geo.attributes.position.count;
  const col = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {
    const c = GYRUS.clone().lerp(SULCUS, shade(i));
    col.set([c.r, c.g, c.b], i * 3);
  }
  geo.setAttribute("color", new THREE.BufferAttribute(col, 3));
  geo.setAttribute("glow", new THREE.BufferAttribute(new Float32Array(n), 1));
  return geo;
}

function cerebellum() {
  const g = new THREE.SphereGeometry(1, 128, 96);
  const p = g.attributes.position;
  const shade = new Float32Array(p.count);
  for (let i = 0; i < p.count; i++) {
    const v = new THREE.Vector3().fromBufferAttribute(p, i);
    const lat = Math.atan2(v.y, Math.hypot(v.x, v.z));
    const folia = Math.sin(lat * 44 + Math.sin(v.x * 3) * 0.6);
    const notch = 1 - 0.13 * Math.exp(-((v.x / 0.16) ** 2)) * (v.z > -0.2 ? 1 : 0.3);
    v.multiplyScalar((1 + 0.03 * folia) * notch);
    v.set(v.x * 50, v.y * 19, v.z * 30);
    p.setXYZ(i, v.x, v.y, v.z);
    shade[i] = 0.35 + 0.45 * (0.5 - 0.5 * folia);
  }
  g.computeVertexNormals();
  withAttributes(g, (i) => shade[i]);
  const m = new THREE.Mesh(g, brainMaterial());
  m.position.copy(toThree([0, -66, -36]));
  m.rotation.x = -0.25;
  m.userData.part = "Cerebellum";
  return m;
}

function brainstem() {
  const grp = [];
  const stem = new THREE.CylinderGeometry(12, 8.5, 64, 48, 12);
  withAttributes(stem, () => 0.35);
  const s = new THREE.Mesh(stem, brainMaterial());
  s.position.copy(toThree([0, -27, -42]));
  s.rotation.x = -0.17;
  s.userData.part = "Brainstem";
  grp.push(s);
  const pons = new THREE.SphereGeometry(1, 48, 32);
  pons.scale(15, 13, 13);
  withAttributes(pons, () => 0.3);
  const pm = new THREE.Mesh(pons, brainMaterial());
  pm.position.copy(toThree([0, -22, -30]));
  pm.userData.part = "Brainstem (pons)";
  grp.push(pm);
  return grp;
}

function tumorGeometry(seed) {
  const g = new THREE.IcosahedronGeometry(1, 5);
  const p = g.attributes.position;
  for (let i = 0; i < p.count; i++) {
    const v = new THREE.Vector3().fromBufferAttribute(p, i);
    const n = Math.sin(v.x * 4.1 + seed) * Math.sin(v.y * 3.7 + seed * 1.7) * Math.sin(v.z * 4.6 + seed * 0.6);
    v.multiplyScalar(1 + 0.16 * n + 0.05 * Math.sin(v.x * 11 + v.y * 9));
    p.setXYZ(i, v.x, v.y, v.z);
  }
  g.computeVertexNormals();
  return g;
}

function glowSprite() {
  const c = document.createElement("canvas");
  c.width = c.height = 128;
  const x = c.getContext("2d");
  const gr = x.createRadialGradient(64, 64, 0, 64, 64, 64);
  gr.addColorStop(0, "rgba(255,60,90,0.9)");
  gr.addColorStop(0.35, "rgba(255,30,70,0.35)");
  gr.addColorStop(1, "rgba(255,0,40,0)");
  x.fillStyle = gr;
  x.fillRect(0, 0, 128, 128);
  const s = new THREE.Sprite(new THREE.SpriteMaterial({
    map: new THREE.CanvasTexture(c), blending: THREE.AdditiveBlending, depthTest: false, transparent: true, toneMapped: false,
  }));
  s.renderOrder = 10;
  return s;
}

function regionOf([x, y, z]) {
  const side = x > 0 ? "Right" : "Left";
  if (Math.hypot(x, y - 2, z + 34) < 14) return "Pituitary region";
  if (Math.abs(x) < 22 && y > -45 && y < 10 && z > -15 && z < 30) return `${side} deep brain`;
  if (y < -68) return `${side} occipital lobe`;
  if (z < -8 && Math.abs(x) > 28 && y > -50) return `${side} temporal lobe`;
  if (y > -12) return `${side} frontal lobe`;
  return `${side} parietal lobe`;
}

class Viewer {
  constructor(root) {
    this.root = root;
    this.payload = null;
    this.rot = { ...DEFAULT_ROT };
    this.target = { ...DEFAULT_ROT };
    this.dist = 480;
    this.targetDist = 480;
    this.spin = true;
    this.xray = false;
    this.showSlice = true;
    this.hoverTumor = false;
    this.clock = new THREE.Clock();
    this.buildDom();
    this.buildScene();
    this.loadBrain(root.dataset.mesh);
    this.bindEvents();
    this.renderer.setAnimationLoop(() => this.frame());
  }

  buildDom() {
    this.root.innerHTML = `
      <div class="nb-canvas"></div>
      <div class="nb-hud nb-tl"><span class="nb-chip" data-role="status">Loading template brain…</span></div>
      <div class="nb-hud nb-tr">
        <button data-act="xray" title="See through the brain">X-ray</button>
        <button data-act="slice" class="on" title="Show your scan inside the brain">Slice</button>
        <button data-act="spin" class="on" title="Auto-rotate">Spin</button>
        <button data-act="reset" title="Reset the view">Reset</button>
      </div>
      <div class="nb-tag" data-role="tag"><span class="nb-tag-dot"></span><span data-role="tagtext"></span></div>
      <div class="nb-tip" data-role="tip"></div>
      <div class="nb-hint">${matchMedia("(pointer: coarse)").matches
        ? "Swipe to rotate · pinch to zoom · tap the red area"
        : "Drag to rotate · scroll to zoom · hover the red area"}</div>
      <div class="nb-empty" data-role="empty">Upload a scan to map it onto the brain</div>`;
    this.el = (r) => this.root.querySelector(`[data-role="${r}"]`);
    this.canvasHost = this.root.querySelector(".nb-canvas");
  }

  buildScene() {
    const r = (this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true }));
    r.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    r.toneMapping = THREE.ACESFilmicToneMapping;
    r.toneMappingExposure = 1.0;
    this.canvasHost.appendChild(r.domElement);

    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(30, 1, 1, 4000);
    this.group = new THREE.Group();
    this.scene.add(this.group);

    this.scene.add(new THREE.HemisphereLight("#cfe8ff", "#0b1030", 0.75));
    const key = new THREE.DirectionalLight("#ffffff", 2.2);
    key.position.set(-200, 260, 320);
    const rim = new THREE.DirectionalLight("#5ab8ff", 2.0);
    rim.position.set(240, 80, -320);
    const fill = new THREE.DirectionalLight("#9b7bff", 0.7);
    fill.position.set(300, -120, 120);
    this.scene.add(key, rim, fill);

    this.parts = [];
    const cb = cerebellum();
    this.group.add(cb);
    this.parts.push(cb);
    for (const m of brainstem()) { this.group.add(m); this.parts.push(m); }

    this.tumor = new THREE.Mesh(tumorGeometry(1.3), new THREE.MeshStandardMaterial({
      color: TUMOR_RED, emissive: "#d0001f", emissiveIntensity: 0.6, roughness: 0.35, metalness: 0.0, toneMapped: false,
    }));
    this.ghost = new THREE.Mesh(this.tumor.geometry, new THREE.MeshBasicMaterial({
      color: "#ff1f45", transparent: true, opacity: 0.3, depthTest: false, depthWrite: false, toneMapped: false,
    }));
    this.ghost.renderOrder = 9;
    this.halo = glowSprite();
    this.tumorGroup = new THREE.Group();
    this.tumorGroup.add(this.tumor, this.ghost, this.halo);
    this.tumorGroup.visible = false;
    this.group.add(this.tumorGroup);

    this.slice = new THREE.Mesh(new THREE.BufferGeometry(), new THREE.MeshBasicMaterial({
      transparent: true, opacity: 0.6, side: THREE.DoubleSide, depthWrite: false,
      blending: THREE.AdditiveBlending, toneMapped: false,
    }));
    this.slice.visible = false;
    this.slice.renderOrder = 5;
    this.group.add(this.slice);

    this.raycaster = new THREE.Raycaster();
    this.pointer = new THREE.Vector2(-9, -9);
    new ResizeObserver(() => this.resize()).observe(this.root);
    this.resize();
  }

  async loadBrain(url) {
    try {
      const buf = await (await fetch(url)).arrayBuffer();
      const dv = new DataView(buf);
      const nv = dv.getUint32(0, true), nf = dv.getUint32(4, true);
      let o = 8;
      const ras = new Float32Array(buf, o, nv * 3); o += nv * 12;
      const sulc = new Float32Array(buf, o, nv); o += nv * 4;
      o += 4; // hemisphere split, unused here
      const faces = new Uint32Array(buf, o, nf * 3);
      const pos = new Float32Array(nv * 3);
      for (let i = 0; i < nv; i++) {
        const v = toThree([ras[i * 3], ras[i * 3 + 1], ras[i * 3 + 2]]);
        pos.set([v.x, v.y, v.z], i * 3);
      }
      const g = new THREE.BufferGeometry();
      g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
      g.setIndex(new THREE.BufferAttribute(faces, 1));
      g.computeVertexNormals();
      withAttributes(g, (i) => THREE.MathUtils.clamp((sulc[i] + 0.9) / 1.9, 0, 1));
      this.cortex = new THREE.Mesh(g, brainMaterial());
      this.cortex.userData.part = "Cortex";
      this.group.add(this.cortex);
      this.parts.unshift(this.cortex);
      if (this.payload) this.apply(this.payload);
      else this.setStatus("Template brain · waiting for a scan");
    } catch (e) {
      this.setStatus("Couldn't load the 3D brain");
      console.error(e);
    }
  }

  setStatus(text, tone = "") {
    const s = this.el("status");
    s.textContent = text;
    s.className = "nb-chip " + tone;
  }

  bindEvents() {
    const c = this.renderer.domElement;
    const active = new Map();  // pointerId -> last position; two touches = pinch zoom
    let pinch = 0;
    const aim = (e) => {
      const b = c.getBoundingClientRect();
      this.pointer.set(((e.clientX - b.left) / b.width) * 2 - 1, -((e.clientY - b.top) / b.height) * 2 + 1);
      this.mouse = { x: e.clientX - b.left, y: e.clientY - b.top };
    };
    const spread = () => {
      const [a, b] = [...active.values()];
      return Math.hypot(a.x - b.x, a.y - b.y);
    };
    c.addEventListener("pointerdown", (e) => {
      aim(e);  // on a phone, a tap works like hovering there
      active.set(e.pointerId, { x: e.clientX, y: e.clientY });
      if (active.size === 2) pinch = spread();
      try { c.setPointerCapture(e.pointerId); } catch { /* pointer already gone */ }
    });
    const end = (e) => { active.delete(e.pointerId); pinch = 0; };
    c.addEventListener("pointerup", end);
    c.addEventListener("pointercancel", end);
    c.addEventListener("pointerleave", (e) => {
      if (e.pointerType === "touch") return;  // keep the tooltip from a tap on screen
      this.pointer.set(-9, -9);
      this.hideTip();
    });
    c.addEventListener("pointermove", (e) => {
      const prev = active.get(e.pointerId);
      if (active.size === 2 && prev) {
        active.set(e.pointerId, { x: e.clientX, y: e.clientY });
        const d = spread();
        if (pinch && d) this.targetDist = THREE.MathUtils.clamp(this.targetDist * (pinch / d), 220, 850);
        pinch = d;
        return;
      }
      aim(e);
      if (prev) {
        this.target.y += (e.clientX - prev.x) * 0.008;
        this.target.x = THREE.MathUtils.clamp(this.target.x + (e.clientY - prev.y) * 0.006, -1.3, 1.3);
        active.set(e.pointerId, { x: e.clientX, y: e.clientY });
      }
    });
    c.addEventListener("wheel", (e) => {
      e.preventDefault();
      this.targetDist = THREE.MathUtils.clamp(this.targetDist * (1 + e.deltaY * 0.001), 220, 850);
    }, { passive: false });
    this.root.querySelectorAll("button[data-act]").forEach((b) => b.addEventListener("click", () => {
      const act = b.dataset.act;
      if (act === "reset") { this.target = { ...DEFAULT_ROT }; this.targetDist = 480; return; }
      this[{ xray: "xray", slice: "showSlice", spin: "spin" }[act]] ^= true;
      b.classList.toggle("on");
      if (act === "xray") this.applyXray();
      if (act === "slice") this.slice.visible = this.showSlice && !!this.payload?.slice;
    }));
  }

  applyXray() {
    for (const m of this.parts) {
      m.material.opacity = this.xray ? 0.22 : 1;
      m.material.depthWrite = !this.xray;
      m.material.needsUpdate = true;
    }
    U.uRim.value = this.xray ? 1.1 : 0.55;
  }

  resize() {
    const w = this.root.clientWidth, h = this.root.clientHeight;
    if (!w || !h) return;
    this.renderer.setSize(w, h, false);
    this.renderer.domElement.style.width = "100%";
    this.renderer.domElement.style.height = "100%";
    this.camera.aspect = w / h;
    this.fit = w / h < 1.15 ? 1.15 / (w / h) : 1;  // back off on narrow screens
    this.camera.updateProjectionMatrix();
  }

  update(p) {
    this.payload = p;
    if (this.cortex) this.apply(p);
  }

  apply(p) {
    const empty = !p || p.state === "empty";
    this.el("empty").style.display = empty ? "" : "none";
    this.tumorGroup.visible = false;
    this.el("tag").style.display = "none";
    U.uRimColor.value.set("#6fc3ff");

    // glow on the cortex around the tumor
    const center = p?.state === "tumor" ? toThree(p.pos) : null;
    const reach = center ? p.radius * 2.6 : 0;
    for (const m of this.parts) {
      const pos = m.geometry.attributes.position, glow = m.geometry.attributes.glow;
      const wp = new THREE.Vector3();
      m.updateMatrix();
      for (let i = 0; i < pos.count; i++) {
        let gv = 0;
        if (center) {
          wp.fromBufferAttribute(pos, i).applyMatrix4(m.matrix);
          const d = wp.distanceTo(center);
          gv = d < reach ? Math.pow(1 - d / reach, 1.6) : 0;
        }
        glow.array[i] = gv;
      }
      glow.needsUpdate = true;
    }

    if (center) {
      this.tumorGroup.position.copy(center);
      this.tumor.scale.setScalar(p.radius);
      this.ghost.scale.setScalar(p.radius);
      this.halo.scale.setScalar(p.radius * 4);
      this.tumorGroup.visible = true;
      this.el("tagtext").textContent = `${p.label} · ${(p.conf * 100).toFixed(0)}%`;
      this.el("tag").style.display = "block";
      this.setStatus(`${p.view_text}`, p.view_conf < 0.7 ? "warn" : "");
    } else if (!empty) {
      U.uRimColor.value.set("#3ee6a8");
      this.setStatus(p.clear_text || "No tumor detected", "ok");
    } else {
      this.setStatus("Template brain · waiting for a scan");
    }

    // the uploaded slice, floating in its own plane
    if (p?.slice && p.plane) {
      const [tl, tr, bl, br] = p.plane.map(toThree);
      const g = new THREE.BufferGeometry();
      g.setAttribute("position", new THREE.BufferAttribute(new Float32Array([
        ...tl.toArray(), ...tr.toArray(), ...bl.toArray(), ...br.toArray()]), 3));
      g.setAttribute("uv", new THREE.BufferAttribute(new Float32Array([0, 1, 1, 1, 0, 0, 1, 0]), 2));
      g.setIndex([0, 2, 1, 2, 3, 1]);
      this.slice.geometry.dispose();
      this.slice.geometry = g;
      new THREE.TextureLoader().load(p.slice, (t) => {
        t.colorSpace = THREE.SRGBColorSpace;
        this.slice.material.map?.dispose();
        this.slice.material.map = t;
        this.slice.material.needsUpdate = true;
        this.slice.visible = this.showSlice;
      });
    } else {
      this.slice.visible = false;
    }
  }

  hover() {
    if (!this.cortex || this.pointer.x < -1.5) return this.hideTip();
    this.raycaster.setFromCamera(this.pointer, this.camera);
    const p = this.payload;
    const tumorHit = this.tumorGroup.visible && this.raycaster.intersectObject(this.tumor, false)[0];
    const hit = this.raycaster.intersectObjects(this.parts, false)[0];
    let onTumor = !!tumorHit;
    if (!onTumor && hit && this.tumorGroup.visible) {
      const gl = hit.object.geometry.attributes.glow.array;
      onTumor = (gl[hit.face.a] + gl[hit.face.b] + gl[hit.face.c]) / 3 > 0.22;
    }
    this.hoverTumor = onTumor;
    this.renderer.domElement.style.cursor = onTumor ? "pointer" : "grab";
    if (onTumor) {
      const other = p.is_top ? "" : `<div class="nb-tip-warn">Showing evidence for ${p.label}, not the top prediction (${p.top_label}).</div>`;
      this.showTip(`
        <div class="nb-tip-title" style="--c:${p.color}">${p.label}</div>
        <div class="nb-tip-row"><span>Model confidence</span><b>${(p.conf * 100).toFixed(1)}%</b></div>
        <div class="nb-tip-row"><span>Approx. location</span><b>${p.region}</b></div>
        <div class="nb-tip-row"><span>Rough size</span><b>~${(p.radius * 0.2).toFixed(1)} cm</b></div>
        <div class="nb-tip-blurb">${p.blurb}</div>${other}
        <div class="nb-tip-foot">Position estimated from one 2D slice on a template brain.</div>`, true);
    } else if (hit) {
      const name = hit.object.userData.part === "Cortex"
        ? regionOf(toRAS(this.group.worldToLocal(hit.point.clone())))
        : hit.object.userData.part;
      this.showTip(`<div class="nb-tip-small">${name}</div>`, false);
    } else {
      this.hideTip();
    }
  }

  showTip(html, big) {
    const t = this.el("tip");
    if (t.dataset.html !== html) { t.innerHTML = html; t.dataset.html = html; }
    t.classList.toggle("big", big);
    t.style.display = "block";
    const w = this.root.clientWidth;
    const x = Math.min(this.mouse.x + 16, w - t.offsetWidth - 8);
    t.style.transform = `translate(${x}px, ${this.mouse.y + 14}px)`;
  }

  hideTip() {
    this.el("tip").style.display = "none";
    this.hoverTumor = false;
  }

  frame() {
    if (!this.root.isConnected || !this.root.offsetParent) return;
    const t = this.clock.getElapsedTime();
    U.uTime.value = t;
    if (this.spin && !this.hoverTumor) this.target.y += 0.0022;
    this.rot.x += (this.target.x - this.rot.x) * 0.1;
    this.rot.y += (this.target.y - this.rot.y) * 0.1;
    this.dist += (this.targetDist - this.dist) * 0.1;
    this.group.rotation.set(this.rot.x, this.rot.y, 0);
    this.camera.position.set(0, 0, this.dist * (this.fit || 1));
    this.camera.lookAt(0, 0, 0);

    if (this.tumorGroup.visible) {
      const pulse = 0.5 + 0.5 * Math.sin(t * 3.0);
      this.tumor.material.emissiveIntensity = 0.35 + 0.45 * pulse;
      this.ghost.material.opacity = (this.xray ? 0.6 : 0.18) + 0.14 * pulse;
      this.halo.material.opacity = 0.3 + 0.25 * pulse;
      // pin the label to the tumor
      const v = this.tumorGroup.getWorldPosition(new THREE.Vector3()).project(this.camera);
      const tag = this.el("tag");
      const tx = (v.x * 0.5 + 0.5) * this.root.clientWidth;
      tag.classList.toggle("flip", tx > this.root.clientWidth - 190);
      tag.style.transform = `translate(${tx}px, ${(-v.y * 0.5 + 0.5) * this.root.clientHeight}px)`;
    }
    this.hover();
    this.renderer.render(this.scene, this.camera);
  }
}

let viewer = null;
let pending = null;
function ensure() {
  const root = document.getElementById("nb-root");
  if (!root) return null;
  if (!viewer || viewer.root !== root) {
    try { viewer = new Viewer(root); } catch (e) {
      root.innerHTML = '<div class="nb-empty">3D view needs WebGL.</div>';
      console.error(e);
      return null;
    }
  }
  return viewer;
}

window.NeuroBrain = {
  update(p) {
    pending = p;
    const v = ensure();
    if (v) v.update(p);
  },
};

const boot = setInterval(() => {
  const v = ensure();
  if (v) { clearInterval(boot); if (pending) v.update(pending); }
}, 250);
