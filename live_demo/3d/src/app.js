/* The runs in 3D.  One offline page; everything drawn comes from scene_data.json, which scbf_mppi.export3d
   writes from the same runs as the figures.  3-DOF planar model: no roll, pitch or heave is drawn.
   Rendering: Preetham sky -> image-based lighting (PMREM), reflective water (three.js Water with a normal map), ACES tone
   mapping, physical materials.  The sky, water texture, wakes and horizon are decoration; the boats, paths, rollouts,
   thrust and every number are data.
   Coordinates: simulation (x east, y north, psi ccw from x)  ->  three.js (x, up=y, z = -y_sim), rotation.y = psi. */
(function () {
"use strict";
const DATA = JSON.parse(document.getElementById("scene-data").textContent);
const META = DATA.meta;
const $ = (s) => document.querySelector(s);
const INK = 0x20303C, GROUND = 0xE3EAEF, FOGC = 0xC7D5DE;
const LIFT = { vessel: 0.30, corridor: 0.08 };            // metres of height per second of prediction (a time axis, not altitude)
const THRUST_M_PER_N = 6.0 / META.f_at_max;               // one force scale for every arrow: 6 m = 700 N

// ------------------------------------------------------------------ renderer, scene, camera, sky, lights
const canvas = $("#gl");
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, preserveDrawingBuffer: true, alpha: false, powerPreference: "high-performance" });
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
renderer.shadowMap.enabled = true; renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.outputEncoding = THREE.sRGBEncoding; renderer.toneMapping = THREE.ACESFilmicToneMapping; renderer.toneMappingExposure = 0.7;
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(48, 16 / 9, 0.05, 30000);
const sky = new THREE.Sky(); sky.scale.setScalar(20000); scene.add(sky);
const su = sky.material.uniforms; su.turbidity.value = 2.2; su.rayleigh.value = 3.2; su.mieCoefficient.value = 0.0035; su.mieDirectionalG.value = 0.8;
const sunDir = new THREE.Vector3().setFromSphericalCoords(1, THREE.MathUtils.degToRad(90 - 38), THREE.MathUtils.degToRad(215)); su.sunPosition.value.copy(sunDir);
const pmrem = new THREE.PMREMGenerator(renderer); scene.environment = pmrem.fromScene(sky).texture;
const hemi = new THREE.HemisphereLight(0xCFE0EE, 0x2F4756, 0.28); scene.add(hemi);
const sun = new THREE.DirectionalLight(0xFFF1D6, 3.0); sun.castShadow = true;
sun.shadow.mapSize.set(2048, 2048); sun.shadow.camera.near = 20; sun.shadow.camera.far = 800;
sun.shadow.camera.left = sun.shadow.camera.bottom = -140; sun.shadow.camera.right = sun.shadow.camera.top = 140; sun.shadow.bias = -0.0006; sun.shadow.normalBias = 0.02;
scene.add(sun); scene.add(sun.target);
const sprites = [];
function fitSprites() {                                    // constant on-screen size for every label (sizeAttenuation off)
  const k = 2 * Math.tan(camera.fov * Math.PI / 360) / window.innerHeight;
  for (const sp of sprites) { const px = sp.userData.px; sp.scale.set(px * k * sp.userData.ar, px * k, 1); }
}
function resize() { const w = window.innerWidth, h = window.innerHeight; renderer.setSize(w, h, false); camera.aspect = w / h; camera.updateProjectionMatrix(); fitSprites(); }
window.addEventListener("resize", resize); resize();

// ------------------------------------------------------------------ helpers and procedural textures
const toW = (x, y, up = 0) => new THREE.Vector3(x, up, -y);           // sim -> world
const lerpAng = (a, b, t) => { let d = ((b - a + Math.PI) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI) - Math.PI; return a + d * t; };
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const sgn = (v, nd) => (v >= 0 ? "+" : "") + v.toFixed(nd);
function tex(w, h, draw, repeat = [1, 1]) { const c = document.createElement("canvas"); c.width = w; c.height = h; draw(c.getContext("2d"), w, h); const t = new THREE.CanvasTexture(c); t.wrapS = t.wrapT = THREE.RepeatWrapping; t.repeat.set(repeat[0], repeat[1]); t.encoding = THREE.sRGBEncoding; t.anisotropy = 8; return t; }
const TEX = {
  teak: () => tex(512, 512, (g, w, h) => { g.fillStyle = "#b8926a"; g.fillRect(0, 0, w, h); for (let i = 0; i < 16; i++) { g.fillStyle = `hsl(${28 + Math.random() * 6}, ${38 + Math.random() * 10}%, ${52 + Math.random() * 10}%)`; g.fillRect(0, i * 32, w, 30); g.fillStyle = "rgba(60,40,20,0.35)"; g.fillRect(0, i * 32 + 30, w, 2); for (let k = 0; k < 40; k++) { g.fillStyle = `rgba(80,55,30,${0.05 + Math.random() * 0.08})`; g.fillRect(Math.random() * w, i * 32 + Math.random() * 30, 40 + Math.random() * 200, 1); } } }, [4, 2]),
  solar: () => tex(512, 256, (g, w, h) => { g.fillStyle = "#0f1b2d"; g.fillRect(0, 0, w, h); g.strokeStyle = "#cbd6e2"; g.lineWidth = 3; for (let x = 0; x <= w; x += 64) { g.beginPath(); g.moveTo(x, 0); g.lineTo(x, h); g.stroke(); } for (let y = 0; y <= h; y += 64) { g.beginPath(); g.moveTo(0, y); g.lineTo(w, y); g.stroke(); } g.strokeStyle = "rgba(180,200,220,0.35)"; g.lineWidth = 1; for (let x = 8; x < w; x += 16) { g.beginPath(); g.moveTo(x, 0); g.lineTo(x, h); g.stroke(); } }, [1, 1]),
  steel: () => tex(512, 512, (g, w, h) => { g.fillStyle = "#6c7a86"; g.fillRect(0, 0, w, h); for (let k = 0; k < 2500; k++) { g.fillStyle = `rgba(${90 + Math.random() * 60},${50 + Math.random() * 40},${30 + Math.random() * 30},${Math.random() * 0.35})`; const r = 1 + Math.random() * 6; g.fillRect(Math.random() * w, Math.random() * h, r, r * (0.5 + Math.random())); } g.strokeStyle = "rgba(30,40,50,0.5)"; g.lineWidth = 2; for (let x = 0; x < w; x += 128) { g.beginPath(); g.moveTo(x, 0); g.lineTo(x, h); g.stroke(); } }, [3, 1]),
  streaks: () => tex(512, 512, (g, w, h) => { g.clearRect(0, 0, w, h); for (let k = 0; k < 90; k++) { g.fillStyle = `rgba(255,255,255,${0.35 + Math.random() * 0.4})`; const L = 8 + Math.random() * 40; g.fillRect(Math.random() * w, Math.random() * h, 2 + Math.random() * 2, L); } }, [40, 40]),
  foam: () => tex(128, 128, (g, w, h) => { const r = g.createRadialGradient(64, 64, 4, 64, 64, 62); r.addColorStop(0, "rgba(255,255,255,0.9)"); r.addColorStop(0.55, "rgba(255,255,255,0.35)"); r.addColorStop(1, "rgba(255,255,255,0)"); g.fillStyle = r; g.fillRect(0, 0, w, h); }),
};
function textSprite(text, color = "#20303C", px = 24, bg = "rgba(255,255,255,0.82)") {
  const c = document.createElement("canvas"); const ctx = c.getContext("2d"); const size = 28;
  ctx.font = `600 ${size}px "IBM Plex Sans", "Segoe UI", sans-serif`;
  const w = Math.ceil(ctx.measureText(text).width) + 22; c.width = w; c.height = size + 16;
  ctx.font = `600 ${size}px "IBM Plex Sans", "Segoe UI", sans-serif`; ctx.fillStyle = bg; ctx.beginPath();
  if (ctx.roundRect) ctx.roundRect(0, 0, w, c.height, 8); else ctx.rect(0, 0, w, c.height); ctx.fill();
  ctx.fillStyle = color; ctx.textBaseline = "middle"; ctx.fillText(text, 11, c.height / 2 + 1);
  const t = new THREE.CanvasTexture(c); t.minFilter = THREE.LinearFilter;
  const sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: t, depthTest: false, transparent: true, sizeAttenuation: false, toneMapped: false }));
  sp.userData = { px, ar: w / c.height }; sprites.push(sp); return sp;
}
function ribbon(pts, width, color, y) {                    // a flat ribbon along a polyline (1-px WebGL lines vanish on a screen share)
  const n = pts.length, pos = new Float32Array(n * 6), idx = [];
  for (let i = 0; i < n; i++) {
    const a = pts[Math.max(0, i - 1)], b = pts[Math.min(n - 1, i + 1)]; let tx = b[0] - a[0], ty = b[1] - a[1]; const L = Math.hypot(tx, ty) || 1; tx /= L; ty /= L;
    const nx = -ty * width / 2, ny = tx * width / 2;
    pos.set([pts[i][0] + nx, y, -(pts[i][1] + ny), pts[i][0] - nx, y, -(pts[i][1] - ny)], i * 6);
    if (i < n - 1) idx.push(2 * i, 2 * i + 1, 2 * i + 2, 2 * i + 1, 2 * i + 3, 2 * i + 2);
  }
  const g = new THREE.BufferGeometry(); g.setAttribute("position", new THREE.BufferAttribute(pos, 3)); g.setIndex(idx); g.setDrawRange(0, 0);
  const m = new THREE.Mesh(g, new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.9, depthWrite: false, side: THREE.DoubleSide, toneMapped: false })); m.renderOrder = 2; return m;
}
function ring(inner, outer, color, opacity, y) { const m = new THREE.Mesh(new THREE.RingGeometry(inner, outer, 96).rotateX(-Math.PI / 2), new THREE.MeshBasicMaterial({ color, transparent: true, opacity, side: THREE.DoubleSide, depthWrite: false, toneMapped: false })); m.position.y = y; m.renderOrder = 2; return m; }

// ------------------------------------------------------------------ water: the three.js ocean (reflection + normal map) and a streak layer that drifts with the current
let normalsTex = null;
function makeWater(cx, cz, size, current) {
  const grp = new THREE.Group();
  if (!normalsTex) { normalsTex = new THREE.TextureLoader().load(window.WATER_NORMALS_URI, (t) => { t.wrapS = t.wrapT = THREE.RepeatWrapping; }); normalsTex.wrapS = normalsTex.wrapT = THREE.RepeatWrapping; }
  const water = new THREE.Water(new THREE.PlaneGeometry(size, size), { textureWidth: 1024, textureHeight: 1024, waterNormals: normalsTex, sunDirection: sunDir.clone(),
    sunColor: 0xFFFFFF, waterColor: 0x0C3446, distortionScale: 0.8, fog: true, alpha: 1.0 });
  water.rotation.x = -Math.PI / 2; water.position.set(cx, 0, cz); water.material.uniforms.size.value = 3.5; water.receiveShadow = true; grp.add(water);
  const reflect = water.onBeforeRender;                     // labels are screen-space annotations: keep them out of the reflection
  water.onBeforeRender = function (r, s, c) { const saved = sprites.map((sp) => sp.visible); sprites.forEach((sp) => { sp.visible = false; }); reflect.call(this, r, s, c); sprites.forEach((sp, i) => { sp.visible = saved[i]; }); };
  const streakMat = new THREE.MeshBasicMaterial({ map: TEX.streaks(), transparent: true, opacity: 0.0, depthWrite: false, toneMapped: false });
  const streaks = new THREE.Mesh(new THREE.PlaneGeometry(size, size).rotateX(-Math.PI / 2), streakMat); streaks.position.set(cx, 0.03, cz); streaks.renderOrder = 1; grp.add(streaks);
  const spd = Math.hypot(current[0], current[1]); streakMat.opacity = spd > 0.01 ? 0.32 : 0.0;
  grp.userData = { water, streaks, current: [current[0], current[1]], size };
  return grp;
}
function tickWater(wg, tSim, dtWall) {
  const u = wg.userData; u.water.material.uniforms.time.value += dtWall * 0.6;
  // the streak texture is advected with the current: sim (cx, cy) -> world (cx, -cy); texture v runs along -z
  const r = u.streaks.material.map.repeat; u.streaks.material.map.offset.set((-u.current[0] * tSim / u.size) * r.x, (-u.current[1] * tSim / u.size) * r.y);
}

// ------------------------------------------------------------------ a stylised hull (LOA x BEAM), lofted from stations, with deck, cabin, windows, solar roof and mast
function makeHull(loa, beam, color, opts = {}) {
  const depth = opts.depth ?? 0.65, free = opts.free ?? 0.85, nS = 26, nC = 11;
  const pos = [], idx = [];
  for (let i = 0; i <= nS; i++) {
    const s = i / nS, x = -loa * 0.5 + loa * s;                       // s=0 transom, s=1 bow
    const hw = (beam / 2) * (s < 0.12 ? 0.82 + 1.5 * s : Math.pow(Math.sin(Math.PI * (0.5 + 0.5 * (1 - s))), 0.55) * (s > 0.86 ? (1 - s) / 0.14 : 1));
    const keel = -depth * (0.35 + 0.65 * Math.sin(Math.PI * Math.min(1, s * 1.15)));
    const sheer = free + 0.12 * Math.pow(s, 2.5);                        // a little sheer rising to the bow
    for (let j = 0; j <= nC; j++) {
      const a = -Math.PI / 2 + Math.PI * (j / nC);
      const yy = j === 0 || j === nC ? sheer : keel + (sheer - keel) * Math.pow(Math.abs(Math.sin(a)), 1.35);
      const zz = Math.sin(a) * hw * (j === 0 || j === nC ? 1 : Math.pow(Math.cos(a), 0.12));
      pos.push(x, yy, zz);
    }
  }
  for (let i = 0; i < nS; i++) for (let j = 0; j < nC; j++) { const a = i * (nC + 1) + j, b = a + nC + 1; idx.push(a, b, a + 1, b, b + 1, a + 1); }
  for (let j = 1; j < nC - 1; j++) idx.push(0, j + 1, j);
  const g = new THREE.BufferGeometry(); g.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3)); g.setIndex(idx); g.computeVertexNormals();
  const hull = new THREE.Mesh(g, new THREE.MeshPhysicalMaterial({ color, roughness: 0.32, metalness: 0.0, clearcoat: 0.75, clearcoatRoughness: 0.12, envMapIntensity: 1.0, side: THREE.DoubleSide }));
  hull.castShadow = true; hull.receiveShadow = true;
  const grp = new THREE.Group(); grp.add(hull);
  const deck = new THREE.Mesh(new THREE.BoxGeometry(loa * 0.94, 0.06, beam * 0.9), new THREE.MeshStandardMaterial({ map: TEX.teak(), roughness: 0.75, metalness: 0.0 }));
  deck.position.set(-loa * 0.02, free + 0.02, 0); deck.castShadow = deck.receiveShadow = true; grp.add(deck);
  const rub = new THREE.Mesh(new THREE.TorusGeometry(1, 0.045, 6, 48).scale(loa * 0.47, 1, beam * 0.46), new THREE.MeshStandardMaterial({ color: 0x1F2A33, roughness: 0.8 }));
  rub.rotation.x = Math.PI / 2; rub.position.set(-loa * 0.02, free + 0.02, 0); grp.add(rub);
  if (opts.cabin !== false) {
    const cl = loa * (opts.cabinL ?? 0.42), cw = beam * 0.78, ch = opts.cabinH ?? 1.45, cx = -loa * 0.08;
    const cabin = new THREE.Mesh(new THREE.BoxGeometry(cl, ch, cw), new THREE.MeshPhysicalMaterial({ color: 0xF6F8FA, roughness: 0.5, clearcoat: 0.3 }));
    cabin.position.set(cx, free + ch / 2, 0); cabin.castShadow = cabin.receiveShadow = true; grp.add(cabin);
    const glass = new THREE.MeshPhysicalMaterial({ color: 0x0E1A26, roughness: 0.08, metalness: 0.4, clearcoat: 1.0, envMapIntensity: 1.4 });
    const wh = ch * 0.42, wy = free + ch * 0.62;
    for (const side of [-1, 1]) { const w = new THREE.Mesh(new THREE.BoxGeometry(cl * 0.86, wh, 0.03), glass); w.position.set(cx, wy, side * (cw / 2 + 0.005)); grp.add(w); }
    const wf = new THREE.Mesh(new THREE.BoxGeometry(0.03, wh, cw * 0.86), glass); wf.position.set(cx + cl / 2 + 0.005, wy, 0); grp.add(wf);
    const roof = new THREE.Mesh(new THREE.BoxGeometry(cl * 1.4, 0.07, cw * 1.25), new THREE.MeshPhysicalMaterial({ map: TEX.solar(), roughness: 0.18, metalness: 0.35, clearcoat: 0.9, clearcoatRoughness: 0.08 }));
    roof.position.set(cx + cl * 0.05, free + ch + 0.05, 0); roof.castShadow = true; grp.add(roof);
    const mast = new THREE.Mesh(new THREE.CylinderGeometry(0.03, 0.04, 1.6, 8), new THREE.MeshStandardMaterial({ color: 0xDDE3E8, roughness: 0.4, metalness: 0.7 })); mast.position.set(cx - cl * 0.35, free + ch + 0.85, 0); grp.add(mast);
    const lamp = new THREE.Mesh(new THREE.SphereGeometry(0.09, 10, 8), new THREE.MeshStandardMaterial({ color: 0xFFFFFF, emissive: 0xFFF2C0, emissiveIntensity: 2.0 })); lamp.position.set(cx - cl * 0.35, free + ch + 1.7, 0); grp.add(lamp);
  }
  if (opts.stripe) { const st = new THREE.Mesh(new THREE.BoxGeometry(loa * 0.9, 0.18, beam * 1.01), new THREE.MeshStandardMaterial({ color: opts.stripe, roughness: 0.5 })); st.position.set(-loa * 0.02, free - 0.35, 0); grp.add(st); }
  return grp;
}

// ------------------------------------------------------------------ foam wakes (decoration): sprites shed at the stern, drifting with the current
function makeWake(n = 48) {
  const mat = new THREE.SpriteMaterial({ map: TEX.foam(), transparent: true, opacity: 0.5, depthWrite: false, toneMapped: false });
  const grp = new THREE.Group(); const items = [];
  for (let i = 0; i < n; i++) { const s = new THREE.Sprite(mat.clone()); s.visible = false; s.material.rotation = Math.random() * 6.28; grp.add(s); items.push({ s, born: -1, x: 0, y: 0 }); }
  grp.userData = { items, next: 0, lastIdx: -1 }; return grp;
}
function tickWake(w, tSim, x, y, psi, speed, current, active) {
  const u = w.userData, life = 7.0, every = 0.35;
  const idx = Math.floor(tSim / every);
  if (idx < u.lastIdx) for (const it of u.items) it.born = -1;                    // scrubbed backwards: clear
  if (active && idx !== u.lastIdx && Math.abs(speed) > 0.5) {
    const it = u.items[u.next]; u.next = (u.next + 1) % u.items.length;
    const back = speed >= 0 ? -1 : 1; it.born = tSim; it.x = x + back * 4.2 * Math.cos(psi) * -1 * back * back; it.y = y; it.x = x - 4.2 * Math.cos(psi) * Math.sign(speed || 1); it.y = y - 4.2 * Math.sin(psi) * Math.sign(speed || 1);
  }
  u.lastIdx = idx;
  for (const it of u.items) {
    if (it.born < 0 || tSim < it.born) { it.s.visible = false; continue; }
    const age = tSim - it.born; if (age > life) { it.s.visible = false; continue; }
    const f = age / life; it.s.visible = true; it.s.position.copy(toW(it.x + current[0] * age, it.y + current[1] * age, 0.06));
    const sz = 1.8 + 5.5 * f; it.s.scale.set(sz, sz, 1); it.s.material.opacity = 0.55 * (1 - f) * (1 - f);
  }
}

// ------------------------------------------------------------------ horizon (decoration): a hazy silhouette ring far away
function makeHorizon(cx, cz) {
  const t = tex(2048, 256, (g, w, h) => { g.clearRect(0, 0, w, h);
    for (const [col, base, amp, seed] of [["rgba(150,168,182,0.55)", 150, 70, 3], ["rgba(112,132,148,0.75)", 195, 45, 11]]) {
      g.fillStyle = col; g.beginPath(); g.moveTo(0, h);
      for (let x = 0; x <= w; x += 8) { let y = base; for (let k = 1; k <= 5; k++) y -= amp / k * Math.abs(Math.sin(x * 0.0021 * k * seed + k * 1.7)); g.lineTo(x, y); }
      g.lineTo(w, h); g.closePath(); g.fill(); } });
  const m = new THREE.Mesh(new THREE.CylinderGeometry(3000, 3000, 360, 160, 1, true), new THREE.MeshBasicMaterial({ map: t, transparent: true, side: THREE.BackSide, depthWrite: false, fog: true }));
  m.position.set(cx, 178, cz); return m;
}

// ------------------------------------------------------------------ state
let rig = null, playing = false, tSim = 0, speed = 2, camMode = "orbit", followIdx = 0;
const toggles = { fan: true, lift: true, fronts: true, discs: true, paths: true, labels: true, thrust: true };
const orbit = { theta: -2.35, phi: 0.95, r: 120, pan: new THREE.Vector3() };
const chaseState = { pos: new THREE.Vector3(), look: new THREE.Vector3(), init: false };
const shortName = (b) => b.short || (b.key === "SCBF-MPPI" ? "SCBF" : b.key === "SCBF-MPPI + IS" ? "SCBF + IS" : b.key);

function clearScene() { if (!rig) return; scene.remove(rig.root); sprites.length = 0; rig.root.traverse((o) => { if (o.geometry) o.geometry.dispose(); if (o.material && o.material.map && o.material.map !== normalsTex) o.material.map.dispose(); if (o.material && o.material.dispose) o.material.dispose(); }); rig = null; }

// ---------- vessel scenes (current / ferry)
function buildVessel(sc) {
  const root = new THREE.Group(); scene.add(root);
  const cur = sc.current || [0, 0];
  scene.fog = new THREE.Fog(0xB4CBDB, 1300, 5200);
  const water = makeWater(60, 0, 5000, [cur[0], -cur[1]]); root.add(water);
  root.add(makeHorizon(60, 0));
  const goal = toW(sc.goal[0], sc.goal[1]);
  const gRing = ring(sc.goal_r - 0.5, sc.goal_r, INK, 0.65, 0.05); gRing.position.copy(goal).setY(0.05); root.add(gRing);
  const pole = new THREE.Mesh(new THREE.CylinderGeometry(0.12, 0.14, 6, 12), new THREE.MeshStandardMaterial({ color: 0xE9573F, roughness: 0.5 })); pole.position.copy(goal).setY(3); pole.castShadow = true; root.add(pole);
  const gLab = textSprite("goal", "#20303C", 22); gLab.position.copy(goal).setY(7.2); root.add(gLab);
  const steel = new THREE.MeshStandardMaterial({ map: TEX.steel(), roughness: 0.85, metalness: 0.25 });
  const obs = sc.obstacles.map((o) => {
    const g = new THREE.Group(); const r = Math.max(o.R - sc.r_ego, 1.0);
    g.position.copy(toW(o.c0[0], o.c0[1]));
    let body;
    if (o.kind === "barge") {
      body = new THREE.Group();
      const b = new THREE.Mesh(new THREE.BoxGeometry(r * 1.7, 1.7, r * 0.9), steel); b.position.y = 0.55; b.castShadow = b.receiveShadow = true; body.add(b);
      const wh = new THREE.Mesh(new THREE.BoxGeometry(r * 0.32, 2.3, r * 0.5), new THREE.MeshPhysicalMaterial({ color: 0xD9DEE3, roughness: 0.55 })); wh.position.set(-r * 0.62, 2.5, 0); wh.castShadow = true; body.add(wh);
      const whg = new THREE.Mesh(new THREE.BoxGeometry(r * 0.33, 0.8, r * 0.51), new THREE.MeshPhysicalMaterial({ color: 0x0E1A26, roughness: 0.1, metalness: 0.4, clearcoat: 1 })); whg.position.set(-r * 0.62, 3.0, 0); body.add(whg);
      const cols = [0x8C3B2A, 0x2F5D8A, 0x5B7F3A]; for (let i = 0; i < 3; i++) { const c = new THREE.Mesh(new THREE.BoxGeometry(r * 0.42, 1.5, r * 0.55), new THREE.MeshStandardMaterial({ color: cols[i], roughness: 0.7 })); c.position.set(-r * 0.15 + i * r * 0.45, 2.15, (i % 2 ? 0.08 : -0.06) * r); c.castShadow = true; body.add(c); }
      body.rotation.y = 0.35;
    } else if (o.kind === "moored") {
      body = makeHull(Math.min(r * 1.6, 12), Math.min(r * 0.5, 3.6), 0xAEB9C3, { cabinL: 0.35, cabinH: 1.2, stripe: 0x2A4C6A }); body.rotation.y = -0.6;
      const buoy = new THREE.Mesh(new THREE.SphereGeometry(0.45, 14, 12), new THREE.MeshPhysicalMaterial({ color: 0xE8602C, roughness: 0.35, clearcoat: 0.6 })); buoy.position.set(r * 0.9, 0.3, -r * 0.4); body.add(buoy);
    } else if (o.kind === "buoys") {
      body = new THREE.Group();
      for (let i = 0; i < 7; i++) { const a = (i / 7) * Math.PI * 2; const b = new THREE.Mesh(new THREE.CylinderGeometry(0.42, 0.55, 1.3, 14), new THREE.MeshPhysicalMaterial({ color: 0xF0A030, roughness: 0.35, clearcoat: 0.5 })); b.position.set(Math.cos(a) * r * 0.75, 0.55, Math.sin(a) * r * 0.75); b.castShadow = true; body.add(b);
        const l = new THREE.Mesh(new THREE.SphereGeometry(0.12, 8, 6), new THREE.MeshStandardMaterial({ color: 0xFFFFFF, emissive: 0xFFE070, emissiveIntensity: 1.5 })); l.position.set(Math.cos(a) * r * 0.75, 1.32, Math.sin(a) * r * 0.75); body.add(l); }
    } else if (o.kind === "ferry") {
      body = makeHull(Math.min(r * 1.8, 40), Math.min(r * 0.45, 9), 0xF4F6F8, { depth: 1.4, free: 1.6, cabinL: 0.62, cabinH: 3.4, stripe: 0x1F4E79 }); body.rotation.y = Math.atan2(o.vel[1], o.vel[0]);
      const funnel = new THREE.Mesh(new THREE.CylinderGeometry(0.7, 0.9, 3.2, 14), new THREE.MeshStandardMaterial({ color: 0xC94A3B, roughness: 0.5 })); funnel.position.set(-6, 1.6 + 3.4 + 1.4, 0); funnel.castShadow = true; body.add(funnel);
    } else {
      body = new THREE.Mesh(new THREE.CylinderGeometry(r, r, 1, 24), steel); body.position.y = 0.5;
    }
    g.add(body);
    const disc = ring(o.R - 0.45, o.R, INK, 0.55, 0.05); g.add(disc);
    const shade = new THREE.Mesh(new THREE.CircleGeometry(o.R, 96).rotateX(-Math.PI / 2), new THREE.MeshBasicMaterial({ color: INK, transparent: true, opacity: 0.06, side: THREE.DoubleSide, depthWrite: false, toneMapped: false })); shade.position.y = 0.035; g.add(shade);
    const front = ring(0.97, 1.0, 0x5B4FB0, 0.6, 0.06); front.visible = false; g.add(front);
    const lab = textSprite(o.name, "#20303C", 22); lab.position.set(0, 4.5 + (o.kind === "ferry" ? 4.5 : 0), 0); g.add(lab);
    root.add(g);
    return { o, g, disc, shade, front, lab, moving: Math.hypot(o.vel[0], o.vel[1]) > 0 };
  });
  const T = sc.boats[0].fan.length ? sc.boats[0].fan[0].S[0].length : 15;
  const nFan = sc.boats[0].fan.length ? sc.boats[0].fan[0].S.length : 40;
  const boats = sc.boats.map((b) => {
    const col = new THREE.Color(b.color); const grp = new THREE.Group();
    grp.add(makeHull(META.loa, META.beam, col, { stripe: col.clone().multiplyScalar(0.55).getHex() }));
    const path = ribbon(b.traj.map((s) => [s[0], s[1]]), 1.0, col, 0.08); root.add(path);
    const nSeg = nFan * Math.max(T - 1, 1);
    const fg = new THREE.BufferGeometry(); fg.setAttribute("position", new THREE.BufferAttribute(new Float32Array(nSeg * 6), 3));
    fg.setAttribute("color", new THREE.BufferAttribute(new Float32Array(nSeg * 6), 3)); fg.setDrawRange(0, 0);
    const fan = new THREE.LineSegments(fg, new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.85, toneMapped: false })); root.add(fan);
    const horizonLab = textSprite(`+${T} s`, b.color, 18); horizonLab.visible = false; root.add(horizonLab);
    const thrust = new THREE.ArrowHelper(new THREE.Vector3(1, 0, 0), new THREE.Vector3(), 1, col.getHex(), 1.2, 0.7); thrust.line.material.toneMapped = false; thrust.cone.material.toneMapped = false; root.add(thrust);
    const disk = ring(5.88, 6.0, col, 0.6, 0.1); root.add(disk);
    const diskLab = textSprite("700 N", b.color, 16); root.add(diskLab);
    const gust = new THREE.ArrowHelper(new THREE.Vector3(1, 0, 0), new THREE.Vector3(), 1, 0x5D7180, 1.0, 0.6); root.add(gust);
    const lab = textSprite(shortName(b), b.color, 22); grp.add(lab); lab.position.set(0, 4.6, 0);
    const wake = makeWake(); root.add(wake);
    root.add(grp);
    const vel = b.traj.map((s) => { const [x, y, psi, u, v] = s; return [u * Math.cos(psi) - v * Math.sin(psi), u * Math.sin(psi) + v * Math.cos(psi)]; });
    return { b, col, grp, path, fan, fg, horizonLab, thrust, disk, diskLab, gust, lab, wake, vel, nSteps: b.traj.length - 1, done: false, course: new THREE.Vector3(1, 0, 0) };
  });
  fitSprites();
  return { kind: "vessel", sc, root, water, obs, boats, goal, T, nFan, tMax: Math.max(...sc.boats.map((b) => b.traj.length - 1)) };
}

// ---------- the corridor
function buildCorridor(sc) {
  const root = new THREE.Group(); scene.add(root);
  scene.fog = new THREE.Fog(GROUND, 300, 1400);
  const S = 10;                                                        // 1 corridor unit drawn as 10 m so the cameras behave like the vessel scenes
  const ground = new THREE.Mesh(new THREE.PlaneGeometry(2400, 2400).rotateX(-Math.PI / 2), new THREE.MeshStandardMaterial({ color: GROUND, roughness: 0.95 })); ground.position.set(20, -0.01, -5); ground.receiveShadow = true; root.add(ground);
  const wallMat = new THREE.MeshPhysicalMaterial({ color: 0x2A3A48, roughness: 0.6, clearcoat: 0.2, side: THREE.DoubleSide });
  const xs = sc.walls.xs;
  for (const key of ["lo", "hi"]) {
    const ys = sc.walls[key]; const pos = [], idx = [];
    for (let i = 0; i < xs.length; i++) { const p = toW(xs[i] * S, ys[i] * S); pos.push(p.x, 0, p.z, p.x, 2.2, p.z); }
    for (let i = 0; i < xs.length - 1; i++) { const a = 2 * i; idx.push(a, a + 2, a + 1, a + 2, a + 3, a + 1); }
    const g = new THREE.BufferGeometry(); g.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3)); g.setIndex(idx); g.computeVertexNormals();
    const w = new THREE.Mesh(g, wallMat); w.castShadow = w.receiveShadow = true; root.add(w);
  }
  const goal = toW(sc.goal[0] * S, sc.goal[1] * S);
  const gRing = ring(sc.goal_r * S - 0.25, sc.goal_r * S, INK, 0.65, 0.03); gRing.position.copy(goal).setY(0.03); root.add(gRing);
  const gLab = textSprite("goal (4, 0.5)", "#20303C", 20); gLab.position.copy(goal).setY(5.2); root.add(gLab);
  const T = sc.robots[0].fan.length ? sc.robots[0].fan[0].S[0].length : 20, nFan = sc.robots[0].fan.length ? sc.robots[0].fan[0].S.length : 30;
  const boats = sc.robots.map((b) => {
    const col = new THREE.Color(b.color); const grp = new THREE.Group();
    const body = new THREE.Mesh(new THREE.CylinderGeometry(0.75, 0.85, 0.5, 28), new THREE.MeshPhysicalMaterial({ color: col, roughness: 0.35, clearcoat: 0.7 })); body.position.y = 0.25; body.castShadow = true; grp.add(body);
    const nose = new THREE.Mesh(new THREE.ConeGeometry(0.32, 1.0, 16).rotateZ(-Math.PI / 2), new THREE.MeshStandardMaterial({ color: INK })); nose.position.set(0.95, 0.35, 0); grp.add(nose);
    const path = ribbon(b.traj.map((s) => [s[0] * S, s[1] * S]), 0.35, col, 0.05); root.add(path);
    const nSeg = nFan * Math.max(T - 1, 1);
    const fg = new THREE.BufferGeometry(); fg.setAttribute("position", new THREE.BufferAttribute(new Float32Array(nSeg * 6), 3)); fg.setAttribute("color", new THREE.BufferAttribute(new Float32Array(nSeg * 6), 3)); fg.setDrawRange(0, 0);
    const fan = new THREE.LineSegments(fg, new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.8, toneMapped: false })); root.add(fan);
    const horizonLab = textSprite(`+${(T * sc.dt).toFixed(1)} s`, b.color, 16); horizonLab.visible = false; root.add(horizonLab);
    const lab = textSprite(shortName(b), b.color, 20); lab.position.set(0, 2.2, 0); grp.add(lab);
    root.add(grp);
    return { b, col, grp, path, fan, fg, horizonLab, lab, nSteps: b.traj.length - 1, done: false, course: new THREE.Vector3(1, 0, 0) };
  });
  fitSprites();
  return { kind: "corridor", sc, root, boats, goal, T, nFan, S, tMax: Math.max(...sc.robots.map((b) => b.traj.length - 1)) * sc.dt };
}

// ------------------------------------------------------------------ per-frame update
function stateAt(bt, k, f) {                 // vessel: Hermite between control states k and k+1 with the model's velocities (dt = 1 s)
  const tr = bt.b.traj, v = bt.vel; const n = tr.length - 1;
  if (k >= n) { const s = tr[n]; return { x: s[0], y: s[1], psi: s[2], u: s[3], v: s[4], r: s[5] }; }
  const a = tr[k], b = tr[k + 1], va = v[k], vb = v[k + 1];
  const h00 = 2 * f ** 3 - 3 * f ** 2 + 1, h10 = f ** 3 - 2 * f ** 2 + f, h01 = -2 * f ** 3 + 3 * f ** 2, h11 = f ** 3 - f ** 2;
  return { x: h00 * a[0] + h10 * va[0] + h01 * b[0] + h11 * vb[0], y: h00 * a[1] + h10 * va[1] + h01 * b[1] + h11 * vb[1],
           psi: lerpAng(a[2], b[2], f), u: a[3] + (b[3] - a[3]) * f, v: a[4] + (b[4] - a[4]) * f, r: a[5] + (b[5] - a[5]) * f };
}
const fanShown = (i) => toggles.fan && ((camMode === "orbit" || camMode === "top") || i === followIdx);
const liftOn = () => toggles.lift && (camMode === "orbit" || camMode === "top");
const tmpV = new THREE.Vector3(), tmpV2 = new THREE.Vector3(), tmpC = new THREE.Color(), waterC = new THREE.Color(0x6F93A3), groundC = new THREE.Color(GROUND);

function drawFan(bt, S, kind, scale) {
  const P = bt.fg.attributes.position, C = bt.fg.attributes.color; let q = 0; const lift = liftOn() ? LIFT[kind] : 0.0; const T = S[0].length;
  const bg = kind === "vessel" ? waterC : groundC; let lastPt = null;
  for (let s = 0; s < S.length; s++) {
    const roll = S[s], w = bt.fanW ? bt.fanW[s] : 1; const base = tmpC.copy(bg).lerp(bt.col, 0.3 + 0.7 * Math.pow(w, 0.6)).clone();
    for (let j = 0; j < roll.length - 1; j++) {
      const c0 = base.clone().lerp(bg, 0.55 * j / (T - 1)), c1 = base.clone().lerp(bg, 0.55 * (j + 1) / (T - 1));
      P.setXYZ(q, roll[j][0] * scale, 0.14 + lift * j, -roll[j][1] * scale); C.setXYZ(q, c0.r, c0.g, c0.b); q++;
      P.setXYZ(q, roll[j + 1][0] * scale, 0.14 + lift * (j + 1), -roll[j + 1][1] * scale); C.setXYZ(q, c1.r, c1.g, c1.b); q++;
    }
    if (s === 0) lastPt = [roll[T - 1][0] * scale, 0.14 + lift * (T - 1), -roll[T - 1][1] * scale];
  }
  bt.fg.setDrawRange(0, q); P.needsUpdate = true; C.needsUpdate = true; bt.fan.visible = true;
  bt.horizonLab.visible = lift > 0 && toggles.labels; if (lastPt) bt.horizonLab.position.set(lastPt[0], lastPt[1] + 1.2, lastPt[2]);
}

function updateVessel(dtWall) {
  const R = rig, sc = R.sc, cur = sc.current || [0, 0];
  tickWater(R.water, tSim, dtWall);
  const fb = R.boats[followIdx] || R.boats[0];
  for (const ob of R.obs) {
    if (ob.moving) { const t = Math.min(tSim, R.tMax); ob.g.position.copy(toW(ob.o.c0[0] + ob.o.vel[0] * t, ob.o.c0[1] + ob.o.vel[1] * t)); }
    ob.disc.visible = ob.shade.visible = toggles.discs; ob.lab.visible = toggles.labels; ob.front.visible = false;
    ob.shade.material.opacity = 0.06; ob.shade.material.color.setHex(INK);
  }
  for (let i = 0; i < R.boats.length; i++) {
    const bt = R.boats[i], b = bt.b, n = bt.nSteps;
    const k = Math.min(Math.floor(tSim), n), f = clamp(tSim - Math.floor(tSim), 0, 1);
    const st = stateAt(bt, Math.min(k, n - 1), k >= n ? 1 : f);
    bt.done = tSim >= n;
    bt.grp.position.copy(toW(st.x, st.y)); bt.grp.rotation.y = st.psi;
    const cs = Math.cos(st.psi), sn = Math.sin(st.psi);
    const vx = st.u * cs - st.v * sn + cur[0], vy = st.u * sn + st.v * cs + cur[1];
    if (Math.hypot(vx, vy) > 0.3) bt.course.set(vx, 0, -vy).normalize();
    bt.lab.visible = toggles.labels && !(i === followIdx && (camMode === "chase" || camMode === "helm"));
    tickWake(bt.wake, tSim, st.x, st.y, st.psi, st.u, cur, !bt.done);
    bt.path.geometry.setDrawRange(0, 6 * Math.min(k, n)); bt.path.visible = toggles.paths;
    const fanRec = (k < n && fanShown(i)) ? b.fan[Math.min(k, b.fan.length - 1)] : null;
    if (fanRec && fanRec.k === k) { bt.fanW = fanRec.w; drawFan(bt, fanRec.S, "vessel", 1); } else { bt.fan.visible = false; bt.horizonLab.visible = false; }
    const kc = Math.min(k, b.ctrl.length - 1); const u = b.ctrl[kc]; const fmag = Math.hypot(u[0], u[1]);
    const stern = toW(st.x - 2.9 * cs, st.y - 2.9 * sn, 0.9);
    const fx = u[0] * cs - u[1] * sn, fy = u[0] * sn + u[1] * cs;
    const showT = toggles.thrust && !bt.done;
    if (showT && fmag > 5) { bt.thrust.visible = true; bt.thrust.position.copy(stern); bt.thrust.setDirection(tmpV.set(fx, 0, -fy).normalize()); const L = THRUST_M_PER_N * fmag; bt.thrust.setLength(L, Math.min(1.2, 0.4 * L), 0.7); } else bt.thrust.visible = false;
    bt.disk.visible = showT; bt.disk.position.copy(stern).setY(0.1);
    bt.diskLab.visible = showT && toggles.labels; bt.diskLab.position.copy(stern).add(tmpV2.set(-6.2 * cs, 0.5, 6.2 * sn));
    const d = b.dist[Math.max(0, Math.min(kc - 1, b.dist.length - 1))] || [0, 0, 0];
    const gx = d[0] * cs - d[1] * sn, gy = d[0] * sn + d[1] * cs, gm = Math.hypot(gx, gy);
    if (showT && gm > 15) { bt.gust.visible = true; bt.gust.position.copy(toW(st.x, st.y, 1.6)); bt.gust.setDirection(tmpV2.set(gx, 0, -gy).normalize()); const L = THRUST_M_PER_N * gm; bt.gust.setLength(L, Math.min(0.9, 0.4 * L), 0.5); } else bt.gust.visible = false;
    const hk = b.h[Math.min(k, b.h.length - 1)], hPrev = k > 0 ? b.h[k - 1] : hk; const hNow = hPrev + (hk - hPrev) * (k >= n ? 1 : f);
    let hMin = Infinity, nIn = 0; for (let j = 0; j < Math.min(k, b.h.length); j++) { hMin = Math.min(hMin, b.h[j]); if (b.h[j] < 0) nIn++; }
    if (!isFinite(hMin)) hMin = hNow;
    const ess = b.ess[Math.min(k, b.ess.length - 1)];
    if (hNow < 0 && toggles.discs) { let best = null, bd = Infinity; for (const ob of R.obs) { const dd = ob.g.position.distanceTo(bt.grp.position) - ob.o.R; if (dd < bd) { bd = dd; best = ob; } }
      if (best) { best.shade.material.opacity = 0.3; best.shade.material.color.copy(bt.col); } }
    const astern = st.u < -0.3 ? " astern" : "";
    const end = bt.done ? (b.reached ? "goal reached" : "time limit") + (nIn ? ` · inside a circle for ${nIn} s of the run` : "") : "";
    bt.readout = { l1: `t = ${Math.min(k, n)} s · surge ${st.u.toFixed(1)} m/s${astern} · azimuth ${fmag.toFixed(0)} / ${META.f_at_max.toFixed(0)} N${fmag >= 0.98 * META.f_at_max ? " (at limit)" : ""} · bow ${u[2].toFixed(0)} N`,
                   l2: `h ${sgn(hNow, 1)} m · closest ${sgn(hMin, 1)} m · inside circle ${nIn} s\nESS ${ess.toFixed(0)} / 500 · gust ${gm.toFixed(0)} N`, end };
    if (i === followIdx && toggles.fronts && b.activation != null) for (const ob of R.obs) {
      const c = ob.g.position; const dx = c.x - bt.grp.position.x, dz = c.z - bt.grp.position.z; const dist = Math.hypot(dx, dz) || 1;
      const closing = ((st.u * cs - st.v * sn - ob.o.vel[0]) * dx + (st.u * sn + st.v * cs - ob.o.vel[1]) * (-dz)) / dist;
      if (closing > 0.05 && dist < ob.o.R + 45) { ob.front.visible = true; const rad = ob.o.R + closing / sc.alpha1; ob.front.scale.set(rad, 1, rad); ob.front.material.color.copy(bt.col); }
    }
  }
}

function updateCorridor() {
  const R = rig, sc = R.sc, S = R.S, dt = sc.dt;
  const stepF = tSim / dt;
  for (let i = 0; i < R.boats.length; i++) {
    const bt = R.boats[i], b = bt.b, n = bt.nSteps;
    const k = Math.min(Math.floor(stepF), n), f = clamp(stepF - Math.floor(stepF), 0, 1);
    const a = b.traj[Math.min(k, n)], c = b.traj[Math.min(k + 1, n)];
    const x = a[0] + (c[0] - a[0]) * (k >= n ? 0 : f), y = a[1] + (c[1] - a[1]) * (k >= n ? 0 : f), th = lerpAng(a[2], c[2], k >= n ? 0 : f);
    bt.done = stepF >= n;
    bt.grp.position.copy(toW(x * S, y * S)); bt.grp.rotation.y = th;
    if (k < n) { const vx = c[0] - a[0], vy = c[1] - a[1]; if (Math.hypot(vx, vy) > 1e-4) bt.course.set(vx, 0, -vy).normalize(); }
    bt.lab.visible = toggles.labels && !(i === followIdx && (camMode === "chase" || camMode === "helm"));
    bt.path.geometry.setDrawRange(0, 6 * Math.min(k, n)); bt.path.visible = toggles.paths;
    const fanRec = (k < n && fanShown(i)) ? b.fan[Math.min(k, b.fan.length - 1)] : null;
    if (fanRec) { bt.fanW = null; drawFan(bt, fanRec.S, "corridor", S); } else { bt.fan.visible = false; bt.horizonLab.visible = false; }
    const hNow = b.h[Math.min(k + 1, b.h.length - 1)]; let hMin = Infinity; for (let j = 0; j <= Math.min(k + 1, b.h.length - 1); j++) hMin = Math.min(hMin, b.h[j]);
    const ess = b.ess[Math.min(k, b.ess.length - 1)];
    const end = bt.done ? (b.reached ? "goal reached" : "TIME LIMIT — goal not reached") : "";
    const extra = [b.activation != null ? `constraint active ${(100 * b.activation).toFixed(0)} %` : null, b.var_kept != null ? `Var[δv] kept ${(100 * b.var_kept).toFixed(0)} %` : null].filter(Boolean).join(" · ");
    bt.readout = { l1: `step ${Math.min(k, n)} · min h ${sgn(hNow, 2)} · closest ${sgn(hMin, 2)} · ESS ${ess.toFixed(0)} / 500${hNow < 0 ? " · COLLISION" : ""}`, l2: extra, end };
  }
}

// ------------------------------------------------------------------ cameras
function updateCamera(dtWall) {
  const R = rig; const fb = R.boats[followIdx] || R.boats[0]; const target = fb ? fb.grp.position : new THREE.Vector3();
  const scale = R.kind === "corridor" ? 0.9 : 1.0;
  if (camMode === "orbit") {
    const t = tmpV.copy(target).add(orbit.pan);
    camera.position.set(t.x + orbit.r * Math.sin(orbit.phi) * Math.cos(orbit.theta), t.y + orbit.r * Math.cos(orbit.phi), t.z + orbit.r * Math.sin(orbit.phi) * Math.sin(orbit.theta));
    camera.lookAt(t); chaseState.init = false;
  } else if (camMode === "top") {
    const c = R.kind === "corridor" ? toW(2.0 * R.S, 0.5 * R.S) : new THREE.Vector3(60, 0, 0);
    camera.position.set(c.x, R.kind === "corridor" ? 62 : 170, c.z + 0.01); camera.lookAt(c); chaseState.init = false;
  } else if (fb) {
    const psi = fb.grp.rotation.y; const head = new THREE.Vector3(Math.cos(psi), 0, -Math.sin(psi));
    const fwd = camMode === "chase" ? fb.course : head;                 // chase: behind the course over ground; helm: the helmsman's view along the heading
    let want, look;
    if (camMode === "chase") { want = target.clone().addScaledVector(fwd, -28 * scale).setY(9 * scale); look = target.clone().addScaledVector(fwd, 18 * scale).setY(1.5); }
    else { want = target.clone().addScaledVector(head, 2.4 * scale).setY(R.kind === "corridor" ? 1.4 : 2.7); look = target.clone().addScaledVector(head, 40 * scale).setY(1.6); }
    if (!chaseState.init) { chaseState.pos.copy(want); chaseState.look.copy(look); chaseState.init = true; }
    const a = 1 - Math.exp(-dtWall * (camMode === "helm" ? 8 : 2.6));
    chaseState.pos.lerp(want, a); chaseState.look.lerp(look, a);
    camera.position.copy(chaseState.pos); camera.lookAt(chaseState.look);
  }
  sun.target.position.copy(target); sun.position.copy(target).addScaledVector(sunDir, 320);
}

// ------------------------------------------------------------------ mouse orbit (hand-rolled, no addons); never steals a chase/helm camera on a click
let drag = null;
canvas.addEventListener("contextmenu", (e) => e.preventDefault());
canvas.addEventListener("pointerdown", (e) => { drag = { x: e.clientX, y: e.clientY, btn: e.button, moved: 0 }; });
window.addEventListener("pointerup", () => { drag = null; });
window.addEventListener("pointermove", (e) => {
  if (!drag) return; const dx = e.clientX - drag.x, dy = e.clientY - drag.y; drag.x = e.clientX; drag.y = e.clientY; drag.moved += Math.abs(dx) + Math.abs(dy);
  if (camMode !== "orbit") { if (drag.moved > 24) setCam("orbit"); return; }
  if (drag.btn === 2 || e.shiftKey) { const s = orbit.r * 0.0016; const right = new THREE.Vector3().setFromMatrixColumn(camera.matrix, 0); orbit.pan.addScaledVector(right, -dx * s).addScaledVector(new THREE.Vector3(0, 1, 0), dy * s); }
  else { orbit.theta -= dx * 0.006; orbit.phi = clamp(orbit.phi - dy * 0.005, 0.12, 1.45); }
});
canvas.addEventListener("wheel", (e) => { e.preventDefault(); if (camMode === "orbit") orbit.r = clamp(orbit.r * Math.exp(e.deltaY * 0.0012), 6, 900); }, { passive: false });

// ------------------------------------------------------------------ UI
function setCam(m) { camMode = m; chaseState.init = false; document.querySelectorAll(".cam").forEach((b) => b.classList.toggle("on", b.dataset.cam === m)); }
document.querySelectorAll(".cam").forEach((b) => b.addEventListener("click", () => { setCam(b.dataset.cam); b.blur(); }));
document.querySelectorAll(".tog").forEach((b) => b.addEventListener("click", () => { toggles[b.dataset.tog] = !toggles[b.dataset.tog]; b.classList.toggle("on", toggles[b.dataset.tog]); b.blur(); }));
const playBtn = $("#play"), scrub = $("#scrub"), clock = $("#clock");
function setPlaying(p) { playing = p; playBtn.textContent = p ? "❚❚ pause" : "▶ play"; playBtn.classList.toggle("on", p); }
playBtn.addEventListener("click", () => { setPlaying(!playing); playBtn.blur(); });
$("#reset").addEventListener("click", (e) => { tSim = 0; setPlaying(false); chaseState.init = false; e.target.blur(); });
$("#speed").addEventListener("change", (e) => { speed = parseFloat(e.target.value); e.target.blur(); });
scrub.addEventListener("input", (e) => { tSim = (parseInt(e.target.value, 10) / 1000) * rig.tMax; setPlaying(false); });
scrub.addEventListener("change", () => scrub.blur());
$("#follow").addEventListener("change", (e) => { followIdx = parseInt(e.target.value, 10); chaseState.init = false; renderReadouts(); e.target.blur(); });
$("#scene").addEventListener("change", (e) => { loadScene(e.target.value); e.target.blur(); });
window.addEventListener("keydown", (e) => {
  if (e.target.tagName === "SELECT" || e.target.tagName === "INPUT") return;
  if (e.code === "Space") { e.preventDefault(); setPlaying(!playing); }
  else if (e.key === "1") setCam("orbit"); else if (e.key === "2") setCam("chase"); else if (e.key === "3") setCam("helm"); else if (e.key === "4") setCam("top");
  else if (e.key === "[") { speed = Math.max(0.25, speed / 2); $("#speed").value = String(speed); } else if (e.key === "]") { speed = Math.min(16, speed * 2); $("#speed").value = String(speed); }
  else if (e.key === "r" || e.key === "R") { tSim = 0; chaseState.init = false; }
});

function renderReadouts() {
  const box = $("#readouts"); box.innerHTML = "";
  rig.boats.forEach((bt, i) => {
    const d = document.createElement("div"); d.className = "boat" + (i === followIdx ? " sel" : ""); d.style.borderLeftColor = bt.b.color;
    const st = rig.sc.stats; let count = "";
    if (st && rig.kind === "vessel") count = `${st.touched[i]} of ${st.n} seeds touch a circle · ${st.reached[i]} reach the goal`;
    if (st && rig.kind === "corridor") count = `${st.reached[i]} of ${st.n} seeds reach the goal`;
    d.innerHTML = `<div class="name"><span style="color:${bt.b.color}">${bt.b.label}</span></div><div class="count">${count}</div><div class="ro"></div>`;
    d.addEventListener("click", () => { followIdx = i; $("#follow").value = String(i); chaseState.init = false; renderReadouts(); });
    box.appendChild(d); bt.dom = d.querySelector(".ro");
  });
}
function paintReadouts() {
  for (const bt of rig.boats) { if (!bt.dom || !bt.readout) continue; const r = bt.readout;
    bt.dom.textContent = r.l1 + (r.l2 ? "\n" + r.l2 : "") + (r.end ? "\n· " + r.end : ""); }
  clock.textContent = rig.kind === "corridor" ? `step ${Math.floor(tSim / rig.sc.dt)}` : `t = ${Math.floor(tSim)} s`;
  if (!scrub.matches(":active")) scrub.value = String(Math.round(1000 * clamp(tSim / rig.tMax, 0, 1)));
}

function loadScene(name) {
  clearScene(); const sc = DATA.scenes[name]; $("#scene").value = name;
  rig = sc.kind === "corridor" ? buildCorridor(sc) : buildVessel(sc);
  sky.visible = rig.kind === "vessel"; scene.background = rig.kind === "corridor" ? new THREE.Color(GROUND) : null;
  tSim = 0; followIdx = 0; setPlaying(false); chaseState.init = false;
  if (rig.kind === "corridor") { orbit.r = 55; orbit.theta = -1.9; orbit.phi = 0.85; orbit.pan.set(0, 0, 0); } else { orbit.r = 120; orbit.theta = -2.35; orbit.phi = 0.95; orbit.pan.set(0, 0, 0); }
  const curM = rig.kind === "vessel" ? Math.hypot(sc.current[0], sc.current[1]) : 0;
  $("#title").textContent = sc.title;
  $("#subtitle").textContent = rig.kind === "corridor"
    ? `unicycle, Δt = ${sc.dt} s, horizon ${sc.T} steps, σ = ${sc.sigma}, K = 500 · seed ${sc.seed} of the 30 in E1, same seed and noise for all three · 1 corridor unit drawn as 10 m · faint lines: ${rig.nFan} of the 500 rollouts, evenly indexed, rising ${LIFT.corridor} m per 0.05 s of prediction (height = prediction time)`
    : `Solgenia model unchanged (Ocean Eng. 2025) · forces as inputs, 1 s cycle, 15 s horizon, K = 500 · ${sc.disturbance === "ou" ? "coloured gust (OU, 5 s)" : "white force noise"}${curM > 0 ? ` + ${curM} m/s current flowing south (cross-track) that the model does not know` : ""} · faint lines: the ${rig.nFan} highest-weight of the 500 rollouts, brighter = heavier, rising ${LIFT.vessel} m per second of prediction (height = prediction time, not altitude)`;
  const st = sc.stats;
  $("#stats").innerHTML = rig.kind === "corridor"
    ? (st ? `Over the <b>${st.n} seeds</b> of E1 the goal is reached by <b>${st.reached[0]} / ${st.reached[1]} / ${st.reached[2]}</b> runs (MPPI / as printed / as claimed). This is seed ${sc.seed}, the stored E1 run.` : "")
    : (st ? `Over the <b>${st.n} seeds</b> (same disturbance for all three within a seed): touched a circle <b>${st.touched[0]} / ${st.touched[1]} / ${st.touched[2]}</b> · reached the goal within 150 s <b>${st.reached[0]} / ${st.reached[1]} / ${st.reached[2]}</b> (MPPI / barrier / + IS) · MPPI &lt; barrier &lt; + IS on closest approach in <b>${st.order} of ${st.n}</b> paired seeds. This is seed ${sc.seed}, not a selected one.` : "");
  $("#note").textContent = rig.kind === "corridor"
    ? "One run each. The wall height and the robot are drawn for legibility only; the model is the paper's planar unicycle. A fan that collapses to a single line is the variance collapse (Var[δv] kept 4 %)."
    : "3-DOF planar model — no roll, pitch or heave is simulated or drawn · hull stylised to 8.5 × 2.2 m · positions interpolated between the 1-s control states · the boats run astern when that is cheaper: the input is an isotropic 700 N force disk on the group's model, whose damping is symmetric in surge, and the controllers carry no heading cost — a property of the force-input abstraction, not of the barrier · sky, water texture, wakes and horizon are decoration; the current is the streaks on the surface · rings on the water: the barrier's zero (obstacle + 5 m ego radius) · stern arrow: executed azimuth force, ring = the 700 N disk (6 m ≙ 700 N) · grey arrow: the disturbance force of this second, same scale · coloured ring near an obstacle: where ψ₁ = ḣ + α₁h = 0 at the followed barrier boat's bearing and closing speed (R + v/α₁, α₁ = 0.1 s⁻¹) · h: reference point to the inflated circle · ESS: effective sample size of 500";
  const fol = $("#follow"); fol.innerHTML = rig.boats.map((bt, i) => `<option value="${i}">${bt.b.label}</option>`).join("");
  renderReadouts();
  document.querySelectorAll(".tog").forEach((b) => { if (["fronts", "discs", "thrust"].includes(b.dataset.tog)) b.style.display = rig.kind === "corridor" ? "none" : ""; });
}

// ------------------------------------------------------------------ recording: the whole tab (HUD included) when the browser allows it, else the canvas
let recorder = null, chunks = [];
$("#record").addEventListener("click", async () => {
  const btn = $("#record"); btn.blur();
  if (recorder) { recorder.stop(); return; }
  if (!window.MediaRecorder) { alert("Recording needs Chrome or Edge."); return; }
  let stream = null, how = "tab";
  try { stream = await navigator.mediaDevices.getDisplayMedia({ video: { displaySurface: "browser", frameRate: 30 }, audio: false, preferCurrentTab: true, selfBrowserSurface: "include", surfaceSwitching: "exclude" }); }
  catch (e) { how = "canvas"; stream = canvas.captureStream ? canvas.captureStream(30) : null; }
  if (!stream) return;
  const mime = MediaRecorder.isTypeSupported("video/webm;codecs=vp9") ? "video/webm;codecs=vp9" : "video/webm";
  recorder = new MediaRecorder(stream, { mimeType: mime, videoBitsPerSecond: 14e6 }); chunks = [];
  recorder.ondataavailable = (e) => { if (e.data.size) chunks.push(e.data); };
  recorder.onstop = () => { stream.getTracks().forEach((t) => t.stop()); const blob = new Blob(chunks, { type: "video/webm" }); const url = URL.createObjectURL(blob);
    $("#dl").innerHTML = `<a class="dl" href="${url}" download="scbf_mppi_3d_${$("#scene").value}_${camMode}.webm">save WebM (${(blob.size / 1e6).toFixed(1)} MB, ${how})</a>`;
    recorder = null; btn.classList.remove("on"); $("#recbar").style.display = "none"; $("#hint").style.display = ""; };
  stream.getVideoTracks()[0].addEventListener("ended", () => { if (recorder) recorder.stop(); });
  recorder.start(250); btn.classList.add("on"); $("#recbar").style.display = "block"; $("#hint").style.display = "none";
});

// ------------------------------------------------------------------ main loop
let last = performance.now();
function frame(now) {
  const dtWall = Math.min(0.1, (now - last) / 1000); last = now;
  if (playing) { tSim += dtWall * speed; if (tSim >= rig.tMax) { tSim = rig.tMax; setPlaying(false); } }
  if (rig.kind === "vessel") updateVessel(dtWall); else updateCorridor();
  updateCamera(dtWall); paintReadouts();
  renderer.render(scene, camera);
  requestAnimationFrame(frame);
}
// deep links / scripted states:  viewer_3d.html?scene=ferry&t=22&cam=chase&follow=1&speed=2&play=1
const qs = new URLSearchParams(location.search);
loadScene(DATA.scenes[qs.get("scene")] ? qs.get("scene") : "current");
if (qs.has("cam")) setCam(qs.get("cam"));
if (qs.has("follow")) { followIdx = clamp(parseInt(qs.get("follow"), 10) || 0, 0, rig.boats.length - 1); $("#follow").value = String(followIdx); renderReadouts(); }
if (qs.has("speed")) { speed = parseFloat(qs.get("speed")) || speed; $("#speed").value = String(speed); }
if (qs.has("t")) tSim = clamp(parseFloat(qs.get("t")) || 0, 0, rig.tMax);
if (qs.get("play") === "1") setPlaying(true);
requestAnimationFrame(frame);
window.__scbf_3d = { get rig() { return rig; }, loadScene, setCam, setTime: (t) => { tSim = t; }, play: setPlaying };
})();
