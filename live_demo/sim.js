/* sim.js — simulation core of the live demo (runs in the browser and in Node for cross-validation).
   Ports of the Python package scbf_mppi (corridor: env.py, dynamics.py, scbf.py, mppi.py, simulate.py) and
   scbf_mppi.vessel (solgenia.py, harbour.py, solver_nd.py [single row exact; several rows: sequential sweeps],
   controllers.py, simulate.py).  The reported numbers come from the Python package; this core is for visualisation. */
(function (root, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory();
  else root.SIM = factory();
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  // ---------------------------------------------------------------- RNG (seeded)
  function Rng(seed) { this.s = (seed >>> 0) || 1; this.spare = null; }
  Rng.prototype.u = function () {              // mulberry32
    let t = (this.s += 0x6D2B79F5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  Rng.prototype.n = function () {              // standard normal (Box–Muller)
    if (this.spare !== null) { const v = this.spare; this.spare = null; return v; }
    let u1 = 0; while (u1 < 1e-12) u1 = this.u();
    const u2 = this.u(), r = Math.sqrt(-2 * Math.log(u1)), th = 2 * Math.PI * u2;
    this.spare = r * Math.sin(th); return r * Math.cos(th);
  };
  const Z997 = 2.7477813; // Phi^{-1}(0.997)
  function normPpf(p) { // Acklam's rational approximation, |err| < 1.2e-9
    const a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00];
    const b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01, -1.328068155288572e+01];
    const c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00];
    const d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00];
    const pl = 0.02425, ph = 1 - pl; let q, r;
    if (p < pl) { q = Math.sqrt(-2 * Math.log(p)); return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1); }
    if (p <= ph) { q = p - 0.5; r = q * q; return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1); }
    q = Math.sqrt(-2 * Math.log(1 - p)); return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  }

  // ================================================================ CORRIDOR (the paper's example)
  const CDT = 0.05;
  function Corridor(printed) { this.printed = !!printed; this.alpha = 1.0; }
  Corridor.prototype.w = function (x) { return this.printed ? Math.sin(x) : Math.sin(Math.PI * x / 2); };
  Corridor.prototype.dw = function (x) { return this.printed ? Math.cos(x) : (Math.PI / 2) * Math.cos(Math.PI * x / 2); };
  Corridor.prototype.d2w = function (x) { return this.printed ? -Math.sin(x) : -((Math.PI / 2) ** 2) * Math.sin(Math.PI * x / 2); };
  Corridor.prototype.minh = function (x, y) { const w = this.w(x); return Math.min(y - w, w + this.alpha - y); };

  function corridorMakeController(env, kind, opt) {
    // kind: mppi | scbf_var | scbf_std | scbf_is | det ; opt: {K,T,lam,sv,som,sigma,delta,seed,umax}
    const K = opt.K, T = opt.T, lam = opt.lam, sv = opt.sv, som = opt.som, sigEnv = opt.sigma;
    const rng = new Rng(opt.seed * 7919 + 17);
    const z = normPpf(1 - opt.delta), alpha = z;
    const goal = [4, 0.5], penalty = 1000;
    const isDet = kind === "det", isScbf = kind.startsWith("scbf");
    const form = kind === "scbf_var" ? "variance" : "std", isCorr = kind === "scbf_is";
    const iters = 4, shrink = 0.5, M = 125;
    const Kact = isDet ? M : K;
    const U = new Float64Array(T * 2);                 // nominal plan
    const Rv = lam / (sv * sv), Rom = lam / (som * som);
    const umax = opt.umax || null;
    const S = new Float64Array(Kact), w = new Float64Array(Kact);
    const traj = new Float32Array(Kact * (T + 1) * 2);  // xy of every sample (for drawing)
    const eps = new Float64Array(Kact * T * 2);
    const st = { ess: 0, act: 0, var: 1, sat: NaN, infeas: 0, wmax: 0 };
    const NS = 65;
    const actFlags = new Uint8Array(Kact * T);          // 1 where the barrier constraint was active for that (sample, step)
    function clipU(v, om) { if (!umax) return [v, om]; return [Math.max(-umax[0], Math.min(umax[0], v)), Math.max(-umax[1], Math.min(umax[1], om))]; }
    function rolloutAndCost(x0, sig_v, sig_om, lamUse, detCorr, record) {
      // draws eps, rolls out, returns S (with control terms); SCBF modifies the draws per timestep.
      // record: keep the t = 0 row-space picture in st.row0 (all samples share x0 and the nominal mean at t = 0)
      // The barrier rows are evaluated for EVERY controller: for MPPI / deterministic MPPI they measure how often the
      // unmodified proposal would have violated the chance constraint, and how often its draw satisfies the row.
      const X = new Float64Array(Kact * 3);
      for (let k = 0; k < Kact; k++) { X[3 * k] = x0[0]; X[3 * k + 1] = x0[1]; X[3 * k + 2] = x0[2]; traj[k * (T + 1) * 2] = x0[0]; traj[k * (T + 1) * 2 + 1] = x0[1]; S[k] = 0; }
      let nAct = 0, nInf = 0, varSum = 0, satN = 0, satOk = 0;
      const logq = new Float64Array(Kact);
      for (let t = 0; t < T; t++) {
        const ubv = U[2 * t], ubo = U[2 * t + 1];
        for (let k = 0; k < Kact; k++) {
          const x = X[3 * k], y = X[3 * k + 1], th = X[3 * k + 2];
          let ev, eo, m = 0, s = sig_v;
          const xi_v = rng.n(), xi_o = rng.n();
          // rows: c1 v >= b1, c2 v >= b2  (omega column is zero)
          const wx = env.w(x), h1 = y - wx, h2 = wx + env.alpha - y;
          const c1 = Math.sin(th) - env.dw(x) * Math.cos(th), c2 = -c1;
          const ito1 = 0.5 * sigEnv * sigEnv * (-env.d2w(x));
          const b1 = -h1 - ito1, b2 = -h2 + ito1;
          // active?  (m = 0, s = s0) feasible for both rows?
          const penNom = form === "std" ? z * Math.abs(c1) * sig_v : alpha * c1 * c1 * sig_v * sig_v;
          const r1n = b1 + penNom - c1 * ubv, r2n = b2 + penNom - c2 * ubv;
          let lo = -Infinity, hi = Infinity, zeroOk = true;
          function bounds(c, r) { if (c > 1e-12) lo = Math.max(lo, r / c); else if (c < -1e-12) hi = Math.min(hi, r / c); else if (r > 0) zeroOk = false; }
          bounds(c1, r1n); bounds(c2, r2n);
          const active = !(zeroOk && lo <= 0 && hi >= 0 && lo <= hi);
          actFlags[k * T + t] = active ? 1 : 0;
          if (active) nAct++;
          if (record && t === 0 && k === 0) { // the more violated of the two wall rows, by nominal z-score along the row
            const s1 = Math.abs(c1) * sig_v, zs1 = s1 > 1e-12 ? (c1 * ubv - b1) / s1 : (c1 * ubv - b1), zs2 = s1 > 1e-12 ? (c2 * ubv - b2) / s1 : (c2 * ubv - b2);
            const one = zs1 <= zs2;
            st.row0 = { dim: 1, a: one ? c1 : c2, b: one ? b1 : b2, ubar: ubv, sig: sig_v, mu: 0, s: sig_v, active: active, form: form, z: z, alpha: alpha, name: one ? "lower wall" : "upper wall", filtered: isScbf,
              rows: [{ a: c1, b: b1, name: "lower wall", h: h1 }, { a: c2, b: b2, name: "upper wall", h: h2 }], jSel: one ? 0 : 1 };
          }
          if (isScbf) {
            if (active) {
              let best = Infinity, bm = 0, bs = sig_v, feasAny = false;
              for (let j = 0; j < NS; j++) {
                const sj = sig_v * j / (NS - 1);
                const pen = form === "std" ? z * Math.abs(c1) * sj : alpha * c1 * c1 * sj * sj;
                let lo2 = -Infinity, hi2 = Infinity, zok = true;
                const rr1 = b1 + pen - c1 * ubv, rr2 = b2 + pen - c2 * ubv;
                if (c1 > 1e-12) lo2 = Math.max(lo2, rr1 / c1); else if (c1 < -1e-12) hi2 = Math.min(hi2, rr1 / c1); else if (rr1 > 0) zok = false;
                if (c2 > 1e-12) lo2 = Math.max(lo2, rr2 / c2); else if (c2 < -1e-12) hi2 = Math.min(hi2, rr2 / c2); else if (rr2 > 0) zok = false;
                const feas = zok && lo2 <= hi2;
                let mj;
                if (feas) mj = Math.min(Math.max(0, lo2), hi2);
                else mj = isFinite(lo2) && isFinite(hi2) ? 0.5 * (lo2 + hi2) : (isFinite(lo2) ? lo2 : (isFinite(hi2) ? hi2 : 0));
                const cost = Math.abs(mj) + (sig_v - sj) + (feas ? 0 : 1e6);
                if (cost < best) { best = cost; bm = mj; bs = sj; feasAny = feasAny || feas; }
              }
              if (!feasAny) nInf++;
              m = bm; s = bs;
            }
            varSum += (s * s) / (sig_v * sig_v);
            if (record && t === 0 && k === 0) { st.row0.mu = m; st.row0.s = s; }
            ev = m + s * xi_v; eo = sig_om * xi_o;
            if (isCorr) {
              const sfl = Math.max(s, 1e-9);
              logq[k] += (-0.5 * (ev / sig_v) ** 2 - Math.log(sig_v)) - (-0.5 * ((ev - m) / sfl) ** 2 - Math.log(sfl));
            }
          } else { ev = sig_v * xi_v; eo = sig_om * xi_o; varSum += 1; }
          // delivered satisfaction on the drawn control (tolerance for boundary ties) — every controller
          const uv = ubv + ev;
          const ok = (c1 * uv >= b1 - 1e-7 * (1 + Math.abs(b1))) && (c2 * uv >= b2 - 1e-7 * (1 + Math.abs(b2)));
          if (active) { satN++; if (ok) satOk++; }
          eps[(k * T + t) * 2] = ev; eps[(k * T + t) * 2 + 1] = eo;
          let [v, om] = clipU(ubv + ev, ubo + eo);
          const xn = x + v * Math.cos(th) * CDT, yn = y + v * Math.sin(th) * CDT, thn = th + om * CDT;
          X[3 * k] = xn; X[3 * k + 1] = yn; X[3 * k + 2] = thn;
          traj[(k * (T + 1) + t + 1) * 2] = xn; traj[(k * (T + 1) + t + 1) * 2 + 1] = yn;
          // running cost
          const dx = xn - goal[0], dy = yn - goal[1];
          const inside = env.minh(xn, yn) > 0;
          S[k] += dx * dx + dy * dy + (inside ? 0 : penalty);
          if (!isDet) S[k] += ubv * Rv * ev + 0.5 * ubv * ubv * Rv + ubo * Rom * eo + 0.5 * ubo * ubo * Rom;
        }
      }
      // terminal cost + det correction
      for (let k = 0; k < Kact; k++) {
        const xn = X[3 * k], yn = X[3 * k + 1]; const dx = xn - goal[0], dy = yn - goal[1];
        S[k] += dx * dx + dy * dy;
        if (isDet) { let c = 0; for (let t = 0; t < T; t++) c += eps[(k * T + t) * 2] / (sig_v * sig_v) * U[2 * t] + eps[(k * T + t) * 2 + 1] / (sig_om * sig_om) * U[2 * t + 1]; S[k] += lamUse * c; }
      }
      st.act = nAct / (Kact * T); st.infeas = nInf / (Kact * T); st.var = varSum / (Kact * T); st.sat = satN ? satOk / satN : NaN;
      if (record) { const prof = new Float32Array(T); for (let t = 0; t < T; t++) { let c = 0; for (let k = 0; k < Kact; k++) c += actFlags[k * T + t]; prof[t] = c / Kact; } st.actProfile = prof; }   // share of samples on which the row binds, per horizon step
      return logq;
    }
    function weightsUpdate(lamUse, logq) {
      let smin = Infinity; for (let k = 0; k < Kact; k++) if (S[k] < smin) smin = S[k];
      let lmax = -Infinity; for (let k = 0; k < Kact; k++) { w[k] = -(S[k] - smin) / lamUse + (logq ? logq[k] : 0); if (w[k] > lmax) lmax = w[k]; }
      let sum = 0; for (let k = 0; k < Kact; k++) { w[k] = Math.exp(w[k] - lmax); sum += w[k]; }
      let s2 = 0, wmax = 0; for (let k = 0; k < Kact; k++) { w[k] /= sum; s2 += w[k] * w[k]; if (w[k] > wmax) wmax = w[k]; }
      st.ess = 1 / s2; st.wmax = wmax;
      for (let t = 0; t < T; t++) { let dv = 0, dom = 0; for (let k = 0; k < Kact; k++) { dv += w[k] * eps[(k * T + t) * 2]; dom += w[k] * eps[(k * T + t) * 2 + 1]; } U[2 * t] += dv; U[2 * t + 1] += dom; if (umax) { const c = clipU(U[2 * t], U[2 * t + 1]); U[2 * t] = c[0]; U[2 * t + 1] = c[1]; } }
    }
    return {
      plan: function (x0) {
        if (isDet) {
          // diagnostics are averaged over the inner iterations (equal numbers of (sample, step) pairs); ESS is the mean over
          // iterations, as in the Python package; the fan / flags shown are those of the last iteration
          let aAct = 0, aVar = 0, aSat = 0, nSat = 0, aEss = 0;
          for (let j = 0; j < iters; j++) { const beta = Math.pow(shrink, j); rolloutAndCost(x0, beta * sv, beta * som, beta * beta * lam, true, j === 0); weightsUpdate(beta * beta * lam, null);
            aAct += st.act; aVar += st.var; if (!isNaN(st.sat)) { aSat += st.sat; nSat++; } aEss += st.ess; }
          st.act = aAct / iters; st.var = aVar / iters; st.sat = nSat ? aSat / nSat : NaN; st.ess = aEss / iters;
        } else { const lq = rolloutAndCost(x0, sv, som, lam, false, true); weightsUpdate(lam, isCorr ? lq : null); }
        const u0 = [U[0], U[1]]; st.u0 = u0.slice();
        for (let t = 0; t < T - 1; t++) { U[2 * t] = U[2 * t + 2]; U[2 * t + 1] = U[2 * t + 3]; } U[2 * T - 2] = 0; U[2 * T - 1] = 0;
        return u0;
      },
      traj: traj, w: w, act: actFlags, K: Kact, T: T, stats: st, kind: kind
    };
  }

  function corridorEpisode(kind, opt, onStep) {
    // opt: {K,T,lam,sv,som,sigma,delta,seed,maxSteps,umax}; returns metrics; onStep(k, x, ctrl) optional (returns false to stop)
    const env = new Corridor(false);
    const c = corridorMakeController(env, kind, opt);
    const rng = new Rng(10000 + opt.seed);
    let x = [0, 0.5, 0]; const maxSteps = opt.maxSteps || 250; const sig = opt.sigma;
    let outside = 0, minh = Infinity, ttf = maxSteps, reached = false, essSum = 0, actSum = 0, varSum = 0, satSum = 0, satN = 0, n = 0;
    const path = [[x[0], x[1]]];
    for (let k = 0; k < maxSteps; k++) {
      const u = c.plan(x);
      essSum += c.stats.ess; actSum += c.stats.act; varSum += c.stats.var; if (!isNaN(c.stats.sat)) { satSum += c.stats.sat; satN++; }
      let [v, om] = u; if (opt.umax) { v = Math.max(-opt.umax[0], Math.min(opt.umax[0], v)); om = Math.max(-opt.umax[1], Math.min(opt.umax[1], om)); }
      x = [x[0] + v * Math.cos(x[2]) * CDT + sig * Math.sqrt(CDT) * rng.n(), x[1] + v * Math.sin(x[2]) * CDT + sig * Math.sqrt(CDT) * rng.n(), x[2] + om * CDT + sig * Math.sqrt(CDT) * rng.n()];
      n++; const h = env.minh(x[0], x[1]); if (h < minh) minh = h; if (h < 0) outside++;
      path.push([x[0], x[1]]);
      if (onStep && onStep(k, x, c) === false) break;
      if (Math.hypot(x[0] - 4, x[1] - 0.5) < 0.15) { ttf = k + 1; reached = true; break; }
    }
    return { collision_rate: outside / n, collided: outside > 0, min_h: minh, ttf: ttf, reached: reached, ess: essSum / n, activation: actSum / n, var_ratio: varSum / n, sat_active: satN ? satSum / satN : NaN, steps: n, path: path };
  }

  // ================================================================ VESSEL (Solgenia, ISD)
  const SG = {
    m: 3100, J: 21179, X_du: -155.42, Y_dv: -1070, N_dv: -3328, Y_dr: -1008,
    X_u: -84.01, Y_v: -795.58, N_v: -958.4, N_r: -5319.88, Y_r: -896.11,
    X_uu: -46.73, Y_vv: 0, N_vv: -234.94, Y_rr: -149.38, N_rr: 0, X_rr: -312.21, X_vr: 434.41, Y_uv: 395.0, Y_ur: -368.2, N_ur: 392.76, N_uv: -138.5,
    Y_vr: 70.39, Y_rv: 24.09, N_vr: -85.67, N_rv: 14.09,
    L_AT: 2.9, L_BT: 3.7, F_AT_MAX: 0.63 * (2000 / 60) ** 2, F_BT_MAX0: 0.055 * (4000 / 60) ** 2, D_BT: 0.62, DT: 0.25, DTC: 1.0, LOA: 8.5, BEAM: 2.2
  };
  // M = M_RB + M_A (xg = 0): [[m - X_du, 0, 0],[0, m - Y_dv, -Y_dr],[0, -N_dv, J]]
  const Mm = [[SG.m - SG.X_du, 0, 0], [0, SG.m - SG.Y_dv, -SG.Y_dr], [0, -SG.N_dv, SG.J]];
  function inv3(A) { const [a, b, c] = A[0], [d, e, f] = A[1], [g, h, i] = A[2]; const det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
    return [[(e * i - f * h) / det, (c * h - b * i) / det, (b * f - c * e) / det], [(f * g - d * i) / det, (a * i - c * g) / det, (c * d - a * f) / det], [(d * h - e * g) / det, (b * g - a * h) / det, (a * e - b * d) / det]]; }
  const Minv = inv3(Mm);
  const Balloc = [[1, 0, 0], [0, 1, 1], [0, -SG.L_AT, SG.L_BT]];   // tau = B u, u = (X_AT, Y_AT, F_BT)
  function bowCap(u) { return SG.F_BT_MAX0 * Math.exp(-SG.D_BT * u * u); }
  function clipInputs(u, x) { const mag = Math.hypot(u[0], u[1]); const s = mag > SG.F_AT_MAX ? SG.F_AT_MAX / mag : 1; const cap = bowCap(x[3]); return [u[0] * s, u[1] * s, Math.max(-cap, Math.min(cap, u[2]))]; }
  function cd(nu) { // C_RB nu + N(nu)
    const u = nu[0], v = nu[1], r = nu[2], p = SG;
    const crb0 = -p.m * r * v, crb1 = p.m * r * u, crb2 = 0;
    const n1 = -p.X_u * u - p.X_uu * Math.abs(u) * u + p.X_vr * v * r + p.X_rr * r * r;
    const n2 = -p.Y_v * v - p.Y_r * r - p.Y_vv * Math.abs(v) * v + p.Y_ur * u * r + p.Y_uv * u * v - p.Y_rr * Math.abs(r) * r - p.Y_vr * Math.abs(v) * r - p.Y_rv * Math.abs(r) * v;
    const n3 = -p.N_r * r - p.N_v * v - p.N_rr * Math.abs(r) * r + p.N_uv * u * v + p.N_ur * u * r - p.N_vv * Math.abs(v) * v - p.N_vr * Math.abs(v) * r - p.N_rv * Math.abs(r) * v;
    return [crb0 + n1, crb1 + n2, crb2 + n3];
  }
  function fnu(nu) { const c = cd(nu); return [-(Minv[0][0] * c[0] + Minv[0][1] * c[1] + Minv[0][2] * c[2]), -(Minv[1][0] * c[0] + Minv[1][1] * c[1] + Minv[1][2] * c[2]), -(Minv[2][0] * c[0] + Minv[2][1] * c[1] + Minv[2][2] * c[2])]; }
  function drift(x, uc, taud, cur) { // uc already clipped; taud disturbance force or null; cur current [cx, cy]
    const psi = x[2], u = x[3], v = x[4], r = x[5], c = Math.cos(psi), s = Math.sin(psi);
    const tau = [uc[0] + (taud ? taud[0] : 0), uc[1] + uc[2] + (taud ? taud[1] : 0), -SG.L_AT * uc[1] + SG.L_BT * uc[2] + (taud ? taud[2] : 0)];
    const q = cd([u, v, r]); const t0 = tau[0] - q[0], t1 = tau[1] - q[1], t2 = tau[2] - q[2];
    return [c * u - s * v + (cur ? cur[0] : 0), s * u + c * v + (cur ? cur[1] : 0), r,
      Minv[0][0] * t0 + Minv[0][1] * t1 + Minv[0][2] * t2, Minv[1][0] * t0 + Minv[1][1] * t1 + Minv[1][2] * t2, Minv[2][0] * t0 + Minv[2][1] * t1 + Minv[2][2] * t2];
  }
  function stepVesselSubs(x, u, dt, taud, cur, sigmaF, rng, nsub) {
    // like stepVessel, but returns [x_next, [sub-step states...]] (positions after every RK4 sub-step)
    nsub = nsub || Math.max(1, Math.round(dt / SG.DT)); const h = dt / nsub; const uc = clipInputs(u, x); let xn = x.slice(); const subs = [];
    for (let i = 0; i < nsub; i++) {
      const k1 = drift(xn, uc, taud, cur); const x2 = xn.map((v, j) => v + 0.5 * h * k1[j]);
      const k2 = drift(x2, uc, taud, cur); const x3 = xn.map((v, j) => v + 0.5 * h * k2[j]);
      const k3 = drift(x3, uc, taud, cur); const x4 = xn.map((v, j) => v + h * k3[j]);
      const k4 = drift(x4, uc, taud, cur);
      xn = xn.map((v, j) => v + h / 6 * (k1[j] + 2 * k2[j] + 2 * k3[j] + k4[j]));
      if (sigmaF && rng) { const f = [sigmaF[0] * rng.n() * Math.sqrt(h), sigmaF[1] * rng.n() * Math.sqrt(h), sigmaF[2] * rng.n() * Math.sqrt(h)];
        xn[3] += Minv[0][0] * f[0] + Minv[0][1] * f[1] + Minv[0][2] * f[2]; xn[4] += Minv[1][0] * f[0] + Minv[1][1] * f[1] + Minv[1][2] * f[2]; xn[5] += Minv[2][0] * f[0] + Minv[2][1] * f[1] + Minv[2][2] * f[2]; }
      subs.push(xn.slice());
    }
    xn[2] = ((xn[2] + Math.PI) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI) - Math.PI;
    return [xn, subs];
  }
  function stepVesselOU(x, u, dt, taud, cur, sF, tauc, rng) {
    // one control interval under an Ornstein-Uhlenbeck force disturbance (correlation time tauc, stationary std sF), exact
    // sub-step update; returns [x_next, sub-states, taud_next] — the same arithmetic and RNG order as vesselEpisode's "ou" branch
    const nsub = Math.max(1, Math.round(dt / SG.DT)), h = dt / nsub; const subs = []; let td = taud.slice(); let xn = x;
    for (let i = 0; i < nsub; i++) { xn = stepVessel(xn, u, h, td, cur, null, null, 1); subs.push(xn); const ef = Math.exp(-h / tauc), sq = Math.sqrt(1 - ef * ef); td = [td[0] * ef + sF[0] * sq * rng.n(), td[1] * ef + sF[1] * sq * rng.n(), td[2] * ef + sF[2] * sq * rng.n()]; }
    return [xn, subs, td];
  }
  function stepVessel(x, u, dt, taud, cur, sigmaF, rng, nsub) {
    nsub = nsub || Math.max(1, Math.round(dt / SG.DT)); const h = dt / nsub; const uc = clipInputs(u, x); let xn = x.slice();
    for (let i = 0; i < nsub; i++) {
      const k1 = drift(xn, uc, taud, cur); const x2 = xn.map((v, j) => v + 0.5 * h * k1[j]);
      const k2 = drift(x2, uc, taud, cur); const x3 = xn.map((v, j) => v + 0.5 * h * k2[j]);
      const k3 = drift(x3, uc, taud, cur); const x4 = xn.map((v, j) => v + h * k3[j]);
      const k4 = drift(x4, uc, taud, cur);
      xn = xn.map((v, j) => v + h / 6 * (k1[j] + 2 * k2[j] + 2 * k3[j] + k4[j]));
      if (sigmaF && rng) { const f = [sigmaF[0] * rng.n() * Math.sqrt(h), sigmaF[1] * rng.n() * Math.sqrt(h), sigmaF[2] * rng.n() * Math.sqrt(h)];
        xn[3] += Minv[0][0] * f[0] + Minv[0][1] * f[1] + Minv[0][2] * f[2]; xn[4] += Minv[1][0] * f[0] + Minv[1][1] * f[1] + Minv[1][2] * f[2]; xn[5] += Minv[2][0] * f[0] + Minv[2][1] * f[1] + Minv[2][2] * f[2]; }
    }
    xn[2] = ((xn[2] + Math.PI) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI) - Math.PI;
    return xn;
  }
  function harbour(scenario, a1, a2) {
    const obs = [{ c: [40, 8], R: 12, w: [0, 0], name: "moored boat" }, { c: [72, -12], R: 14, w: [0, 0], name: "barge" }, { c: [92, 14], R: 11, w: [0, 0], name: "buoy field" }];
    if (scenario === "crossing") obs.push({ c: [66, -3 * 22], R: 25, w: [0, 3], name: "ferry (3 m/s)" });
    return { obs: obs, goal: [120, 0], goalR: 5, rEgo: 5, a1: a1 == null ? 0.1 : a1, a2: a2 == null ? 0.4 : a2 };
  }
  function obsCenter(o, t) { return [o.c[0] + o.w[0] * t, o.c[1] + o.w[1] * t]; }
  function minH(H, x, t) { let m = Infinity; for (const o of H.obs) { const c = obsCenter(o, t); const h = Math.hypot(x[0] - c[0], x[1] - c[1]) - o.R; if (h < m) m = h; } return m; }
  function rowsHOCBF(H, x, t) { // returns [{a:[3] in u-coordinates, b, h, psi1}] per obstacle
    const psi = x[2], nu = [x[3], x[4], x[5]], c = Math.cos(psi), s = Math.sin(psi);
    const pdot = [c * nu[0] - s * nu[1], s * nu[0] + c * nu[1]]; const f = fnu(nu); const out = [];
    for (const o of H.obs) {
      const cc = obsCenter(o, t); const rel = [x[0] - cc[0], x[1] - cc[1]]; const d = Math.max(Math.hypot(rel[0], rel[1]), 1e-6); const n = [rel[0] / d, rel[1] / d];
      const h = d - o.R; const vrel = [pdot[0] - o.w[0], pdot[1] - o.w[1]]; const hdot = n[0] * vrel[0] + n[1] * vrel[1]; const psi1 = hdot + H.a1 * h;
      const termN = (vrel[0] * vrel[0] + vrel[1] * vrel[1] - hdot * hdot) / d;
      const nR2 = [n[0] * c + n[1] * s, -n[0] * s + n[1] * c];        // n^T R2
      const Snu = [-nu[1], nu[0]]; const termRot = (nR2[0] * Snu[0] + nR2[1] * Snu[1]) * nu[2];
      const termF = nR2[0] * f[0] + nR2[1] * f[1];
      const atau = [nR2[0] * Minv[0][0] + nR2[1] * Minv[1][0], nR2[0] * Minv[0][1] + nR2[1] * Minv[1][1], nR2[0] * Minv[0][2] + nR2[1] * Minv[1][2]]; // n^T R2 Minv_xy (1x3 in tau)
      const a = [atau[0] * Balloc[0][0] + atau[1] * Balloc[1][0] + atau[2] * Balloc[2][0], atau[0] * Balloc[0][1] + atau[1] * Balloc[1][1] + atau[2] * Balloc[2][1], atau[0] * Balloc[0][2] + atau[1] * Balloc[1][2] + atau[2] * Balloc[2][2]];
      const b = -(termN + termRot + termF + H.a1 * hdot + H.a2 * psi1);
      out.push({ a: a, b: b, h: h, psi1: psi1 });
    }
    return out;
  }
  // ---- per-sample chance-constraint problem (paper eq. (7)/(8)) — a port of the package's solver_nd.solve_rows_nd:
  //   one active row: exact closed form (std form: shrink to s* = clip(slack/z, 0, s_nom), then the l1-cheapest mean
  //   shift; variance form: the same 65-point grid as the package);  two or more rows active, or a row violated by the
  //   single-row solution: the joint candidate solve of the package (Frobenius-minimal factor update for candidate
  //   target stds, exact l1-minimal mean shift over ALL rows by LP vertex enumeration, local refinement).
  function makeL1(A, tol) { // min ||mu||_1  s.t.  A mu >= r  for a fixed A (J x 3) and many r: vertex enumeration as in solver_nd.l1_min_shift
    tol = tol || 1e-9; const J = A.length;
    const pairs = []; for (const [i, k] of [[0, 1], [0, 2], [1, 2]]) for (let j = 0; j < J; j++) for (let l = j + 1; l < J; l++) { // two coordinates, two tight rows: 2x2 systems, fixed per sample
      const m00 = A[j][i], m01 = A[j][k], m10 = A[l][i], m11 = A[l][k]; const det = m00 * m11 - m01 * m10; if (Math.abs(det) > 1e-14) pairs.push({ i, k, j, l, m00, m01, m10, m11, det }); }
    const triples = []; if (J >= 3) for (let j = 0; j < J; j++) for (let l = j + 1; l < J; l++) for (let q = l + 1; q < J; q++) { // three coordinates, three tight rows
      const M = [A[j], A[l], A[q]]; const det = M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1]) - M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0]) + M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0]);
      if (Math.abs(det) > 1e-18) triples.push({ j, l, q, Mi: inv3(M) }); }
    return function (r) {
      let best = Infinity, b0 = 0, b1 = 0, b2 = 0, found = false;
      function consider(m0, m1, m2) { // cost first (strictly better than the incumbent), then feasibility on every row
        const c = Math.abs(m0) + Math.abs(m1) + Math.abs(m2); if (!(c < best)) return;
        for (let j = 0; j < J; j++) { const v = A[j][0] * m0 + A[j][1] * m1 + A[j][2] * m2; if (v - r[j] < -tol * (1 + Math.abs(r[j]))) return; }
        best = c; b0 = m0; b1 = m1; b2 = m2; found = true;
      }
      consider(0, 0, 0);
      for (let i = 0; i < 3; i++) { // one coordinate, one tight row
        let lo = -Infinity, hi = Infinity;
        for (let j = 0; j < J; j++) { const ai = A[j][i]; if (ai > tol) lo = Math.max(lo, r[j] / ai); else if (ai < -tol) hi = Math.min(hi, r[j] / ai); }
        if (lo <= hi) { const val = Math.min(Math.max(0, lo), hi); if (isFinite(val)) consider(i === 0 ? val : 0, i === 1 ? val : 0, i === 2 ? val : 0); }
      }
      for (const p of pairs) { const x = (r[p.j] * p.m11 - p.m01 * r[p.l]) / p.det, y = (p.m00 * r[p.l] - p.m10 * r[p.j]) / p.det; if (isFinite(x) && isFinite(y)) { if (p.i === 0) consider(x, p.k === 1 ? y : 0, p.k === 2 ? y : 0); else consider(0, x, y); } }
      for (const t of triples) { const Mi = t.Mi, r0 = r[t.j], r1 = r[t.l], r2 = r[t.q]; const m0 = Mi[0][0] * r0 + Mi[0][1] * r1 + Mi[0][2] * r2, m1 = Mi[1][0] * r0 + Mi[1][1] * r1 + Mi[1][2] * r2, m2 = Mi[2][0] * r0 + Mi[2][1] * r1 + Mi[2][2] * r2; if (isFinite(m0) && isFinite(m1) && isFinite(m2)) consider(m0, m1, m2); }
      return { mu: [b0, b1, b2], ok: found };
    };
  }
  function pinvSym(G) { // Moore-Penrose inverse of a small symmetric PSD matrix (Jacobi eigen-decomposition; cut-off 1e-15 of the largest eigenvalue, as numpy.linalg.pinv)
    const m = G.length; const A = G.map(r => r.slice()); const V = []; for (let i = 0; i < m; i++) { V.push(new Array(m).fill(0)); V[i][i] = 1; }
    for (let sweep = 0; sweep < 60; sweep++) {
      let off = 0; for (let i = 0; i < m; i++) for (let j = i + 1; j < m; j++) off += A[i][j] * A[i][j];
      if (off < 1e-300) break;
      for (let p = 0; p < m; p++) for (let q = p + 1; q < m; q++) {
        if (Math.abs(A[p][q]) < 1e-300) continue;
        const theta = (A[q][q] - A[p][p]) / (2 * A[p][q]); const t = (theta >= 0 ? 1 : -1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1)); const c = 1 / Math.sqrt(t * t + 1), s = t * c;
        for (let k = 0; k < m; k++) { const akp = A[k][p], akq = A[k][q]; A[k][p] = c * akp - s * akq; A[k][q] = s * akp + c * akq; }
        for (let k = 0; k < m; k++) { const apk = A[p][k], aqk = A[q][k]; A[p][k] = c * apk - s * aqk; A[q][k] = s * apk + c * aqk; }
        for (let k = 0; k < m; k++) { const vkp = V[k][p], vkq = V[k][q]; V[k][p] = c * vkp - s * vkq; V[k][q] = s * vkp + c * vkq; }
      }
    }
    let lmax = 0; for (let i = 0; i < m; i++) lmax = Math.max(lmax, Math.abs(A[i][i]));
    const out = []; for (let i = 0; i < m; i++) out.push(new Array(m).fill(0));
    for (let i = 0; i < m; i++) { const lam = A[i][i]; if (Math.abs(lam) <= 1e-15 * lmax || lam <= 0) continue; for (let a = 0; a < m; a++) for (let b = 0; b < m; b++) out[a][b] += V[a][i] * V[b][i] / lam; }
    return out;
  }
  const PRODUCT_CACHE = {};
  function product(nv, m) { // itertools.product(range(nv), repeat=m): the first index changes slowest (cached)
    const key = nv + "," + m; if (PRODUCT_CACHE[key]) return PRODUCT_CACHE[key];
    const out = []; const total = Math.pow(nv, m); for (let g = 0; g < total; g++) { const c = new Array(m); let v = g; for (let q = m - 1; q >= 0; q--) { c[q] = v % nv; v = Math.floor(v / nv); } out.push(c); } return PRODUCT_CACHE[key] = out;
  }
  function uniq(vals) { const out = []; for (const v of vals) if (out.indexOf(v) < 0) out.push(v); return out; }   // first occurrence of every value, in order
  function solveJoint(ubar, rows, act, s0, z, alpha, form) { // solver_nd._solve_joint for ONE sample; act: boolean per row (the rows to solve jointly)
    const J = rows.length; const idxAct = []; for (let j = 0; j < J; j++) if (act[j]) idxAct.push(j); const m = idxAct.length;
    const Aa = idxAct.map(j => rows[j].a); const W = Aa.map(a => [a[0] * s0[0], a[1] * s0[1], a[2] * s0[2]]); const Wn = W.map(w => Math.hypot(w[0], w[1], w[2]));
    const G = Aa.map(a => Aa.map(b => a[0] * b[0] + a[1] * b[1] + a[2] * b[2])); const Ginv = pinvSym(G);
    const slack = idxAct.map((j, q) => Aa[q][0] * ubar[0] + Aa[q][1] * ubar[1] + Aa[q][2] * ubar[2] - rows[j].b);
    const fStar = slack.map((sl, q) => form === "std" ? Math.min(Math.max(sl / (z * Math.max(Wn[q], 1e-12)), 0), 1) : Math.min(Math.max(Math.sqrt(Math.max(sl, 0) / (alpha * Math.max(Wn[q], 1e-12) ** 2)), 0), 1));
    const Aall = rows.map(r => r.a), ball = rows.map(r => r.b); const au = Aall.map(a => a[0] * ubar[0] + a[1] * ubar[1] + a[2] * ubar[2]); const l1 = makeL1(Aall);
    // Delta = Aa^T Ginv V with V_l = (F_l - 1) W_l:  Delta = sum_l (F_l - 1) C_l ⊗ W_l  where C_l = sum_q Aa_q Ginv[q][l]  (fixed per sample)
    const C = []; for (let l = 0; l < m; l++) { const c = [0, 0, 0]; for (let q = 0; q < m; q++) { const g = Ginv[q][l]; c[0] += Aa[q][0] * g; c[1] += Aa[q][1] * g; c[2] += Aa[q][2] * g; } C.push(c); }
    const D = [[0, 0, 0], [0, 0, 0], [0, 0, 0]], P = [[0, 0, 0], [0, 0, 0], [0, 0, 0]], rreq = new Array(J);
    function evalF(F) { // one candidate: target-std fractions per active row -> cost, mu (P and Delta are left in P / D)
      for (let i = 0; i < 3; i++) { D[i][0] = 0; D[i][1] = 0; D[i][2] = 0; }
      for (let l = 0; l < m; l++) { const f = F[l] - 1; if (f === 0) continue; const w0 = f * W[l][0], w1 = f * W[l][1], w2 = f * W[l][2]; for (let i = 0; i < 3; i++) { const c = C[l][i]; D[i][0] += c * w0; D[i][1] += c * w1; D[i][2] += c * w2; } }
      for (let i = 0; i < 3; i++) { P[i][0] = D[i][0]; P[i][1] = D[i][1]; P[i][2] = D[i][2]; P[i][i] += s0[i]; }
      for (let j = 0; j < J; j++) { const a = Aall[j]; let v = 0; for (let i = 0; i < 3; i++) { const t = P[0][i] * a[0] + P[1][i] * a[1] + P[2][i] * a[2]; v += t * t; } const sd = Math.sqrt(Math.max(v, 0)); rreq[j] = ball[j] + (form === "std" ? z * sd : alpha * sd * sd) - au[j]; }
      const sh = l1(rreq); if (!sh.ok) return { cost: Infinity, mu: sh.mu };
      let fro = 0; for (let i = 0; i < 3; i++) fro += D[i][0] * D[i][0] + D[i][1] * D[i][1] + D[i][2] * D[i][2];
      return { cost: Math.abs(sh.mu[0]) + Math.abs(sh.mu[1]) + Math.abs(sh.mu[2]) + Math.sqrt(fro), mu: sh.mu };
    }
    let bestCost = Infinity, bestMu = null, bestP = null, fBest = null;
    // candidates per active row {0, f*, 1, f*/2}; identical candidates give identical costs, so only the first occurrence of each
    // distinct value per row is kept — the same sequence of distinct combinations, in the same order, as the package's argmin sees
    const lists = fStar.map(f => uniq([0, f, 1, 0.5 * f])); const F = new Array(m);
    const scan = (lsts) => { const nv = Math.max(...lsts.map(l => l.length)); let first = true, bc = Infinity, bm = null, bp = null, bf = null;
      for (const combo of product(nv, m)) { let ok = true; for (let q = 0; q < m; q++) { if (combo[q] >= lsts[q].length) { ok = false; break; } F[q] = lsts[q][combo[q]]; } if (!ok) continue;
        const e = evalF(F); if (first || e.cost < bc) { first = false; bc = e.cost; bm = e.mu; bp = [P[0].slice(), P[1].slice(), P[2].slice()]; bf = F.slice(); } }
      return { cost: bc, mu: bm, P: bp, F: bf }; };
    const r1 = scan(lists); bestCost = r1.cost; bestMu = r1.mu; bestP = r1.P; fBest = r1.F;
    const refine = { 1: 9, 2: 5, 3: 3, 4: 3 }[m] || 3;
    if (refine > 1) { const h = 1 / 6; const grid = []; for (let i = 0; i < refine; i++) grid.push(-h + 2 * h * i / (refine - 1));
      const lists2 = fBest.map(fb => uniq(grid.map(g => Math.min(Math.max(fb + g, 0), 1))));
      const r2 = scan(lists2); if (r2.cost < bestCost) { bestCost = r2.cost; bestMu = r2.mu; bestP = r2.P; } }
    return { mu: bestMu, P: bestP, ok: isFinite(bestCost), cost: bestCost };
  }
  function solveRows(ubar, rows, s0, z, alpha, form) {
    const n = 3, J = rows.length; let mu = [0, 0, 0]; let P = [[s0[0], 0, 0], [0, s0[1], 0], [0, 0, s0[2]]];
    // rows active at the nominal proposal, and the most violated one (first minimum, as numpy's argmin)
    const margin0 = new Array(J); let jstar = 0, nAct = 0;
    for (let j = 0; j < J; j++) { const a = rows[j].a; const wn = Math.hypot(a[0] * s0[0], a[1] * s0[1], a[2] * s0[2]); const pen = form === "std" ? z * wn : alpha * wn * wn;
      margin0[j] = a[0] * ubar[0] + a[1] * ubar[1] + a[2] * ubar[2] - rows[j].b - pen; if (margin0[j] < 0) nAct++; if (margin0[j] < margin0[jstar]) jstar = j; }
    const a = rows[jstar].a, b = rows[jstar].b; const an = Math.hypot(a[0], a[1], a[2]);
    const wn = Math.hypot(a[0] * s0[0], a[1] * s0[1], a[2] * s0[2]);           // std of a·du under P0 (w = P0^T a = s0 ∘ a)
    const active = margin0[jstar] < 0, zeroRow = an < 1e-12;
    let sOpt = wn, rOpt = 0, sNom = wn, detRatio = 1;
    if (active && !zeroRow) {
      const slack = a[0] * ubar[0] + a[1] * ubar[1] + a[2] * ubar[2] - b; const amax = Math.max(Math.abs(a[0]), Math.abs(a[1]), Math.abs(a[2]));
      if (form === "std") { sOpt = Math.min(Math.max(slack / z, 0), wn); rOpt = Math.max(z * sOpt - slack, 0); }
      else { let best = Infinity; const NS = 65; for (let g = 0; g < NS; g++) { const sg_ = wn * g / (NS - 1); const r = alpha * sg_ * sg_ - slack; const cost = (wn - sg_) / an + Math.max(r, 0) / amax; if (cost < best) { best = cost; sOpt = sg_; rOpt = Math.max(r, 0); } } }
      let istar = 0; for (let i = 1; i < n; i++) if (Math.abs(a[i]) > Math.abs(a[istar])) istar = i;
      if (Math.abs(a[istar]) > 1e-12) mu[istar] += rOpt / a[istar];
      const shrink = wn > 1e-12 ? 1 - sOpt / Math.max(wn, 1e-12) : 0; const w = [a[0] * s0[0], a[1] * s0[1], a[2] * s0[2]];
      for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) P[i][j] -= shrink * a[i] * w[j] / (an * an);   // rank-one shrink along the row
      detRatio = wn > 1e-12 ? sOpt / wn : 1;                                                                   // |det P| / det P0 (matrix determinant lemma)
    }
    let infeasible = zeroRow && b > 0, residual = false; const multi = nAct >= 2;
    if (J > 1 && !zeroRow) {
      // rows violated by the single-row solution (a mean shift can cross another row) join the active set
      let anyViol = false; const jointRows = new Array(J);
      for (let j = 0; j < J; j++) { const aj = rows[j].a; let v = 0; for (let i = 0; i < n; i++) { let t = 0; for (let l = 0; l < n; l++) t += P[l][i] * aj[l]; v += t * t; } const sd = Math.sqrt(Math.max(v, 0)); const pen = form === "std" ? z * sd : alpha * sd * sd;
        const mg = aj[0] * (ubar[0] + mu[0]) + aj[1] * (ubar[1] + mu[1]) + aj[2] * (ubar[2] + mu[2]) - rows[j].b - pen; const viol = mg < -1e-9 * (1 + Math.abs(rows[j].b)); if (viol) anyViol = true; jointRows[j] = (margin0[j] < 0) || viol; }
      if (multi || anyViol) {
        const jr = solveJoint(ubar, rows, jointRows, s0, z, alpha, form);
        if (jr.ok) { mu = jr.mu; P = jr.P; let v = 0; for (let i = 0; i < n; i++) { let t = 0; for (let l = 0; l < n; l++) t += P[l][i] * a[l]; v += t * t; } sOpt = Math.sqrt(v); sNom = wn;
          const d = P[0][0] * (P[1][1] * P[2][2] - P[1][2] * P[2][1]) - P[0][1] * (P[1][0] * P[2][2] - P[1][2] * P[2][0]) + P[0][2] * (P[1][0] * P[2][1] - P[1][1] * P[2][0]); detRatio = Math.abs(d) / (s0[0] * s0[1] * s0[2]); }
        else { residual = true; infeasible = true; }
      }
    }
    return { mu: mu, P: P, active: active, sFirst: sOpt, sNomFirst: sNom, infeasible: infeasible, multi: multi ? 1 : 0, detRatio: detRatio, jFirst: active ? jstar : -1, residual: residual };
  }
  function vesselMakeController(H, kind, opt) {
    // kind: mppi | scbf | scbf_is | scbf_var | printed | det ; opt: {K,T,lam,s0,delta,seed}
    const K = opt.K, T = opt.T, lam = opt.lam, s0 = opt.s0, dt = SG.DTC;
    const rng = new Rng(opt.seed * 7919 + 101);
    const z = normPpf(1 - opt.delta), alpha = z;
    const isDet = kind === "det", isScbf = kind === "scbf" || kind === "scbf_is" || kind === "scbf_var" || kind === "printed";
    const form = kind === "scbf_var" ? "variance" : "std", isCorr = kind === "scbf_is", printed = kind === "printed";
    const iters = 4, shrink = 0.5, M = 125; const Kact = isDet ? M : K;
    const U = []; for (let t = 0; t < T; t++) U.push([0, 0, 0]);
    const R = [lam / (s0[0] * s0[0]), lam / (s0[1] * s0[1]), lam / (s0[2] * s0[2])];
    const S = new Float64Array(Kact), w = new Float64Array(Kact);
    const traj = new Float32Array(Kact * (T + 1) * 2); const eps = new Float64Array(Kact * T * 3);
    const st = { ess: 0, act: 0, var: 1, sat: NaN, infeas: 0, multi: 0, clip: 0 };
    const penalty = 20000, vmax = 2.5, wv = 2000;
    const actFlags = new Uint8Array(Kact * T);          // 1 where a barrier row was active for that (sample, step)
    let tnow = 0;
    function nominalActive(rows, ub, sig) {
      // would the unmodified proposal N(0, diag(sig^2)) violate the chance constraint on any row?  (margin < 0)
      for (const r of rows) { const a = r.a; const wn = Math.hypot(a[0] * sig[0], a[1] * sig[1], a[2] * sig[2]); const pen = form === "std" ? z * wn : alpha * wn * wn;
        if (a[0] * ub[0] + a[1] * ub[1] + a[2] * ub[2] - r.b - pen < 0) return true; }
      return false;
    }
    function rowsFor(x, tabs) {
      if (!printed) return rowsHOCBF(H, x, tabs);
      const rows = []; const psi = x[2], c = Math.cos(psi), s_ = Math.sin(psi); const pd = [c * x[3] - s_ * x[4], s_ * x[3] + c * x[4]];
      for (const o of H.obs) { const cc = obsCenter(o, tabs); const rel = [x[0] - cc[0], x[1] - cc[1]]; const d = Math.max(Math.hypot(rel[0], rel[1]), 1e-6); const nn = [rel[0] / d, rel[1] / d]; const h = d - o.R; const hdot = nn[0] * (pd[0] - o.w[0]) + nn[1] * (pd[1] - o.w[1]); rows.push({ a: [0, 0, 0], b: -(hdot + alpha * h) }); }
      return rows;
    }
    function rolloutAndCost(x0, sig, lamUse, record) {
      // The barrier rows are evaluated for EVERY controller: for MPPI / deterministic MPPI they measure how often the
      // unmodified proposal would have violated the chance constraint, and how often its draw satisfies the rows.
      // record: keep the t = 0 row-space picture (most violated row by nominal z-score) in st.row0
      const X = []; for (let k = 0; k < Kact; k++) { X.push(x0.slice()); S[k] = 0; traj[k * (T + 1) * 2] = x0[0]; traj[k * (T + 1) * 2 + 1] = x0[1]; }
      let nAct = 0, nInf = 0, nMulti = 0, nClip = 0, varSum = 0, satN = 0, satOk = 0; const logq = new Float64Array(Kact);
      for (let t = 0; t < T; t++) {
        const tabs = tnow + dt * t; const ub = U[t];
        for (let k = 0; k < Kact; k++) {
          const x = X[k]; let e, active;
          const xi = [rng.n(), rng.n(), rng.n()];
          const rows = rowsFor(x, tabs);
          let res = null;
          if (isScbf) {
            res = solveRows(ub, rows, sig, z, alpha, form);
            active = res.active;
            if (res.infeasible) nInf++; nMulti += res.multi;
            varSum += res.sNomFirst > 1e-12 ? (res.sFirst / res.sNomFirst) ** 2 : 1;
            e = [res.mu[0] + res.P[0][0] * xi[0] + res.P[0][1] * xi[1] + res.P[0][2] * xi[2], res.mu[1] + res.P[1][0] * xi[0] + res.P[1][1] * xi[1] + res.P[1][2] * xi[2], res.mu[2] + res.P[2][0] * xi[0] + res.P[2][1] * xi[1] + res.P[2][2] * xi[2]];
            if (isCorr) { const ratio = Math.max(res.detRatio, 1e-6);
              const lq = -0.5 * (xi[0] * xi[0] + xi[1] * xi[1] + xi[2] * xi[2]) - Math.log(ratio); const lp = -0.5 * ((e[0] / sig[0]) ** 2 + (e[1] / sig[1]) ** 2 + (e[2] / sig[2]) ** 2); logq[k] += lp - lq; }
          } else { e = [sig[0] * xi[0], sig[1] * xi[1], sig[2] * xi[2]]; active = nominalActive(rows, ub, sig); varSum += 1; }
          if (record && t === 0 && k === 0) {
            let best = null, bz = Infinity;
            // the row shown: the one the filter swept first when it acted; otherwise the most violated by nominal z-score
            if (res && res.jFirst >= 0) best = { r: rows[res.jFirst], j: res.jFirst };
            else rows.forEach((r, j) => { const sa = Math.hypot(r.a[0] * sig[0], r.a[1] * sig[1], r.a[2] * sig[2]); const m = r.a[0] * ub[0] + r.a[1] * ub[1] + r.a[2] * ub[2] - r.b; const zsc = sa > 1e-12 ? m / sa : m; if (zsc < bz) { bz = zsc; best = { r: r, j: j }; } });
            const r = best.r;
            st.row0 = { dim: 3, a: r.a.slice(), b: r.b, ubar: ub.slice(), sig: sig.slice(), mu: res ? res.mu.slice() : [0, 0, 0], P: res ? res.P.map(row => row.slice()) : [[sig[0], 0, 0], [0, sig[1], 0], [0, 0, sig[2]]], active: active, multi: res ? res.multi : 0, form: form, z: z, alpha: alpha, h: r.h, psi1: r.psi1, name: H.obs[best.j].name, filtered: isScbf, printed: printed,
              rows: rows.map((rr, jj) => ({ a: rr.a.slice(), b: rr.b, name: H.obs[jj].name, h: rr.h })), jSel: best.j, solverRow: (res && res.jFirst >= 0) ? res.jFirst : -1 };
          }
          actFlags[k * T + t] = active ? 1 : 0;
          if (active) nAct++;
          const uu = [ub[0] + e[0], ub[1] + e[1], ub[2] + e[2]]; const uc = clipInputs(uu, x);
          if (Math.abs(uc[0] - uu[0]) + Math.abs(uc[1] - uu[1]) + Math.abs(uc[2] - uu[2]) > 1e-9) nClip++;
          if (active) { // delivered satisfaction of every row by the clipped draw
            let ok = true; for (const r of rows) { if (r.a[0] * uc[0] + r.a[1] * uc[1] + r.a[2] * uc[2] < r.b - 1e-7 * (1 + Math.abs(r.b))) { ok = false; break; } }
            satN++; if (ok) satOk++;
          }
          eps[(k * T + t) * 3] = e[0]; eps[(k * T + t) * 3 + 1] = e[1]; eps[(k * T + t) * 3 + 2] = e[2];
          const xn = stepVessel(x, uu, dt, null, null, null, null);
          X[k] = xn; traj[(k * (T + 1) + t + 1) * 2] = xn[0]; traj[(k * (T + 1) + t + 1) * 2 + 1] = xn[1];
          const dx = xn[0] - H.goal[0], dy = xn[1] - H.goal[1]; const over = Math.max(Math.abs(xn[3]) - vmax, 0);
          S[k] += dx * dx + dy * dy + wv * over * over + (minH(H, xn, tabs + dt) < 0 ? penalty : 0);
          if (!isDet) S[k] += ub[0] * R[0] * e[0] + 0.5 * ub[0] * ub[0] * R[0] + ub[1] * R[1] * e[1] + 0.5 * ub[1] * ub[1] * R[1] + ub[2] * R[2] * e[2] + 0.5 * ub[2] * ub[2] * R[2];
        }
      }
      for (let k = 0; k < Kact; k++) { const xn = X[k]; const dx = xn[0] - H.goal[0], dy = xn[1] - H.goal[1]; S[k] += dx * dx + dy * dy;
        if (isDet) { let c = 0; for (let t = 0; t < T; t++) for (let i = 0; i < 3; i++) c += eps[(k * T + t) * 3 + i] / (sig[i] * sig[i]) * U[t][i]; S[k] += lamUse * c; } }
      st.act = nAct / (Kact * T); st.infeas = nInf / (Kact * T); st.multi = nMulti / (Kact * T); st.clip = nClip / (Kact * T); st.var = varSum / (Kact * T); st.sat = satN ? satOk / satN : NaN;
      if (record) { const prof = new Float32Array(T); for (let t = 0; t < T; t++) { let c = 0; for (let k = 0; k < Kact; k++) c += actFlags[k * T + t]; prof[t] = c / Kact; } st.actProfile = prof; }   // share of samples on which a row binds, per horizon step
      return logq;
    }
    function weightsUpdate(lamUse, logq, x0) {
      let smin = Infinity; for (let k = 0; k < Kact; k++) if (S[k] < smin) smin = S[k];
      let lmax = -Infinity; for (let k = 0; k < Kact; k++) { w[k] = -(S[k] - smin) / lamUse + (logq ? logq[k] : 0); if (w[k] > lmax) lmax = w[k]; }
      let sum = 0; for (let k = 0; k < Kact; k++) { w[k] = Math.exp(w[k] - lmax); sum += w[k]; }
      let s2 = 0; for (let k = 0; k < Kact; k++) { w[k] /= sum; s2 += w[k] * w[k]; } st.ess = 1 / s2;
      for (let t = 0; t < T; t++) { const d = [0, 0, 0]; for (let k = 0; k < Kact; k++) for (let i = 0; i < 3; i++) d[i] += w[k] * eps[(k * T + t) * 3 + i]; const raw = [U[t][0] + d[0], U[t][1] + d[1], U[t][2] + d[2]]; if (t === 0) st.u0raw = raw.slice(); U[t] = clipInputs(raw, x0); }
      st.u0 = U[0].slice(); st.bowCap = bowCap(x0[3]);
    }
    return {
      plan: function (x0) {
        if (isDet) { let aAct = 0, aVar = 0, aSat = 0, nSat = 0, aEss = 0;
          for (let j = 0; j < iters; j++) { const beta = Math.pow(shrink, j); rolloutAndCost(x0, [beta * s0[0], beta * s0[1], beta * s0[2]], beta * beta * lam, j === 0); weightsUpdate(beta * beta * lam, null, x0);
            aAct += st.act; aVar += st.var; if (!isNaN(st.sat)) { aSat += st.sat; nSat++; } aEss += st.ess; }
          st.act = aAct / iters; st.var = aVar / iters; st.sat = nSat ? aSat / nSat : NaN; st.ess = aEss / iters; }   // averaged over the 4 iterations (ESS as in the Python package)
        else { const lq = rolloutAndCost(x0, s0, lam, true); weightsUpdate(lam, isCorr ? lq : null, x0); }
        const u0 = U[0].slice(); for (let t = 0; t < T - 1; t++) U[t] = U[t + 1]; U[T - 1] = U[T - 1].slice(); tnow += dt; return u0;
      },
      reset: function () { tnow = 0; for (let t = 0; t < T; t++) U[t] = [0, 0, 0]; },
      traj: traj, w: w, act: actFlags, K: Kact, T: T, stats: st, kind: kind
    };
  }
  function vesselEpisode(kind, opt, onStep) {
    // opt: {scenario, K,T,lam,s0,delta,seed, disturbance: white|ou|none, sF:[3], tauc, current:[2], maxTime}
    const H = harbour(opt.scenario, opt.a1, opt.a2); const c = vesselMakeController(H, kind, opt); c.reset();
    const rng = new Rng(10000 + opt.seed); const dt = SG.DTC; const maxSteps = Math.round((opt.maxTime || 150) / dt);
    let x = [0, 0, 0, 1.5, 0, 0]; let taud = [0, 0, 0]; const sF = opt.sF || [120, 300, 400], tauc = opt.tauc || 5, cur = opt.current || null;
    let outside = 0, minh = Infinity, ttf = maxSteps * dt, reached = false, n = 0, essSum = 0, actSum = 0, varSum = 0, multiSum = 0, satSum = 0, satN = 0; const path = [[0, 0]];
    for (let k = 0; k < maxSteps; k++) {
      const u = c.plan(x); essSum += c.stats.ess; actSum += c.stats.act; varSum += c.stats.var; multiSum += c.stats.multi; if (!isNaN(c.stats.sat)) { satSum += c.stats.sat; satN++; }
      let subs;
      if (opt.disturbance === "white") [x, subs] = stepVesselSubs(x, u, dt, null, cur, [sF[0] * Math.sqrt(2 * tauc), sF[1] * Math.sqrt(2 * tauc), sF[2] * Math.sqrt(2 * tauc)], rng);
      else if (opt.disturbance === "ou") [x, subs, taud] = stepVesselOU(x, u, dt, taud, cur, sF, tauc, rng);
      else [x, subs] = stepVesselSubs(x, u, dt, null, cur, null, null);
      // collisions and min h are evaluated at every RK4 sub-step (0.25 s), as in the Python package
      n++; let h = Infinity; for (let i = 0; i < subs.length; i++) { const hi = minH(H, subs[i], k * dt + (i + 1) * dt / subs.length); if (hi < h) h = hi; }
      if (h < minh) minh = h; if (h < 0) outside++; path.push([x[0], x[1]]);
      if (onStep && onStep(k, x, c, H, taud) === false) break;
      if (Math.hypot(x[0] - H.goal[0], x[1] - H.goal[1]) < H.goalR) { ttf = (k + 1) * dt; reached = true; break; }
    }
    return { collision_rate: outside / n, collided: outside > 0, min_h: minh, ttf: ttf, reached: reached, ess: essSum / n, activation: actSum / n, var_ratio: varSum / n, multi: multiSum / n, sat_active: satN ? satSum / satN : NaN, steps: n, path: path };
  }

  return { Rng, normPpf, Corridor, corridorMakeController, corridorEpisode, CDT, SG, harbour, obsCenter, minH, rowsHOCBF, solveRows, stepVessel, stepVesselSubs, stepVesselOU, clipInputs, bowCap, vesselMakeController, vesselEpisode };
});
