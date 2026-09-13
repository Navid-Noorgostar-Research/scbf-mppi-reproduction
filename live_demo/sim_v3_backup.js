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
    function rolloutAndCost(x0, sig_v, sig_om, lamUse, detCorr) {
      // draws eps, rolls out, returns S (with control terms); SCBF modifies the draws per timestep.
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
          for (let j = 0; j < iters; j++) { const beta = Math.pow(shrink, j); rolloutAndCost(x0, beta * sv, beta * som, beta * beta * lam, true); weightsUpdate(beta * beta * lam, null);
            aAct += st.act; aVar += st.var; if (!isNaN(st.sat)) { aSat += st.sat; nSat++; } aEss += st.ess; }
          st.act = aAct / iters; st.var = aVar / iters; st.sat = nSat ? aSat / nSat : NaN; st.ess = aEss / iters;
        } else { const lq = rolloutAndCost(x0, sv, som, lam, false); weightsUpdate(lam, isCorr ? lq : null); }
        const u0 = [U[0], U[1]];
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
  // per-sample solve: single active row exact (closed form), further active rows by sequential sweeps (feasible, near-optimal)
  function solveRows(ubar, rows, s0, z, alpha, form) {
    const n = 3; let mu = [0, 0, 0]; let P = [[s0[0], 0, 0], [0, s0[1], 0], [0, 0, s0[2]]];
    let active = false, sNomFirst = 0, sFirst = 0, infeasible = false, multi = 0, detRatio = 1;
    function stdAlong(a) { let v = 0; for (let i = 0; i < n; i++) { let t = 0; for (let j = 0; j < n; j++) t += P[j][i] * a[j]; v += t * t; } return Math.sqrt(v); }
    function margin(r) { const sd = stdAlong(r.a); const pen = form === "std" ? z * sd : alpha * sd * sd; return r.a[0] * (ubar[0] + mu[0]) + r.a[1] * (ubar[1] + mu[1]) + r.a[2] * (ubar[2] + mu[2]) - pen - r.b; }
    for (let sweep = 0; sweep < 4; sweep++) {
      let jworst = -1, mworst = -1e-9;
      for (let j = 0; j < rows.length; j++) { const mg = margin(rows[j]); if (mg < mworst) { mworst = mg; jworst = j; } }
      if (jworst < 0) break;
      if (sweep === 0) active = true; else multi = 1;
      const a = rows[jworst].a, an = Math.hypot(a[0], a[1], a[2]);
      if (an < 1e-12) { if (rows[jworst].b > 0) infeasible = true; break; }
      // w = P^T a^T, current std along a
      const w = [0, 0, 0]; for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) w[i] += P[j][i] * a[j];
      const wn = Math.hypot(w[0], w[1], w[2]);
      const slack = a[0] * (ubar[0] + mu[0]) + a[1] * (ubar[1] + mu[1]) + a[2] * (ubar[2] + mu[2]) - rows[jworst].b;
      const amax = Math.max(Math.abs(a[0]), Math.abs(a[1]), Math.abs(a[2]));
      let best = Infinity, bs = wn, br = 0; const NS = 65;
      for (let g = 0; g < NS; g++) { const sg_ = wn * g / (NS - 1); const pen = form === "std" ? z * sg_ : alpha * sg_ * sg_; const r = pen - slack; const cost = (wn - sg_) / an + Math.max(r, 0) / amax; if (cost < best) { best = cost; bs = sg_; br = Math.max(r, 0); } }
      if (sweep === 0) { sNomFirst = wn; sFirst = bs; }
      detRatio *= wn > 1e-12 ? bs / wn : 1;
      // rank-one shrink: P -= (1 - bs/wn) a^T w^T / |a|^2
      const shrink = wn > 1e-12 ? 1 - bs / wn : 0;
      for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) P[i][j] -= shrink * a[i] * w[j] / (an * an);
      // mean shift on the largest-|a| coordinate
      let istar = 0; for (let i = 1; i < n; i++) if (Math.abs(a[i]) > Math.abs(a[istar])) istar = i;
      if (br > 0) mu[istar] += br / a[istar];
    }
    return { mu: mu, P: P, active: active, sFirst: sFirst, sNomFirst: sNomFirst, infeasible: infeasible, multi: multi, detRatio: detRatio };
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
    function rolloutAndCost(x0, sig, lamUse) {
      // The barrier rows are evaluated for EVERY controller: for MPPI / deterministic MPPI they measure how often the
      // unmodified proposal would have violated the chance constraint, and how often its draw satisfies the rows.
      const X = []; for (let k = 0; k < Kact; k++) { X.push(x0.slice()); S[k] = 0; traj[k * (T + 1) * 2] = x0[0]; traj[k * (T + 1) * 2 + 1] = x0[1]; }
      let nAct = 0, nInf = 0, nMulti = 0, nClip = 0, varSum = 0, satN = 0, satOk = 0; const logq = new Float64Array(Kact);
      for (let t = 0; t < T; t++) {
        const tabs = tnow + dt * t; const ub = U[t];
        for (let k = 0; k < Kact; k++) {
          const x = X[k]; let e, active;
          const xi = [rng.n(), rng.n(), rng.n()];
          const rows = rowsFor(x, tabs);
          if (isScbf) {
            const res = solveRows(ub, rows, sig, z, alpha, form);
            active = res.active;
            if (res.infeasible) nInf++; nMulti += res.multi;
            varSum += res.sNomFirst > 1e-12 ? (res.sFirst / res.sNomFirst) ** 2 : 1;
            e = [res.mu[0] + res.P[0][0] * xi[0] + res.P[0][1] * xi[1] + res.P[0][2] * xi[2], res.mu[1] + res.P[1][0] * xi[0] + res.P[1][1] * xi[1] + res.P[1][2] * xi[2], res.mu[2] + res.P[2][0] * xi[0] + res.P[2][1] * xi[1] + res.P[2][2] * xi[2]];
            if (isCorr) { const ratio = Math.max(res.detRatio, 1e-6);
              const lq = -0.5 * (xi[0] * xi[0] + xi[1] * xi[1] + xi[2] * xi[2]) - Math.log(ratio); const lp = -0.5 * ((e[0] / sig[0]) ** 2 + (e[1] / sig[1]) ** 2 + (e[2] / sig[2]) ** 2); logq[k] += lp - lq; }
          } else { e = [sig[0] * xi[0], sig[1] * xi[1], sig[2] * xi[2]]; active = nominalActive(rows, ub, sig); varSum += 1; }
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
      return logq;
    }
    function weightsUpdate(lamUse, logq, x0) {
      let smin = Infinity; for (let k = 0; k < Kact; k++) if (S[k] < smin) smin = S[k];
      let lmax = -Infinity; for (let k = 0; k < Kact; k++) { w[k] = -(S[k] - smin) / lamUse + (logq ? logq[k] : 0); if (w[k] > lmax) lmax = w[k]; }
      let sum = 0; for (let k = 0; k < Kact; k++) { w[k] = Math.exp(w[k] - lmax); sum += w[k]; }
      let s2 = 0; for (let k = 0; k < Kact; k++) { w[k] /= sum; s2 += w[k] * w[k]; } st.ess = 1 / s2;
      for (let t = 0; t < T; t++) { const d = [0, 0, 0]; for (let k = 0; k < Kact; k++) for (let i = 0; i < 3; i++) d[i] += w[k] * eps[(k * T + t) * 3 + i]; U[t] = clipInputs([U[t][0] + d[0], U[t][1] + d[1], U[t][2] + d[2]], x0); }
    }
    return {
      plan: function (x0) {
        if (isDet) { let aAct = 0, aVar = 0, aSat = 0, nSat = 0, aEss = 0;
          for (let j = 0; j < iters; j++) { const beta = Math.pow(shrink, j); rolloutAndCost(x0, [beta * s0[0], beta * s0[1], beta * s0[2]], beta * beta * lam); weightsUpdate(beta * beta * lam, null, x0);
            aAct += st.act; aVar += st.var; if (!isNaN(st.sat)) { aSat += st.sat; nSat++; } aEss += st.ess; }
          st.act = aAct / iters; st.var = aVar / iters; st.sat = nSat ? aSat / nSat : NaN; st.ess = aEss / iters; }   // averaged over the 4 iterations (ESS as in the Python package)
        else { const lq = rolloutAndCost(x0, s0, lam); weightsUpdate(lam, isCorr ? lq : null, x0); }
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
      else if (opt.disturbance === "ou") { const nsub = Math.round(dt / SG.DT), h = dt / nsub; subs = []; for (let i = 0; i < nsub; i++) { x = stepVessel(x, u, h, taud, cur, null, null, 1); subs.push(x); const ef = Math.exp(-h / tauc), sq = Math.sqrt(1 - ef * ef); taud = [taud[0] * ef + sF[0] * sq * rng.n(), taud[1] * ef + sF[1] * sq * rng.n(), taud[2] * ef + sF[2] * sq * rng.n()]; } }
      else [x, subs] = stepVesselSubs(x, u, dt, null, cur, null, null);
      // collisions and min h are evaluated at every RK4 sub-step (0.25 s), as in the Python package
      n++; let h = Infinity; for (let i = 0; i < subs.length; i++) { const hi = minH(H, subs[i], k * dt + (i + 1) * dt / subs.length); if (hi < h) h = hi; }
      if (h < minh) minh = h; if (h < 0) outside++; path.push([x[0], x[1]]);
      if (onStep && onStep(k, x, c, H, taud) === false) break;
      if (Math.hypot(x[0] - H.goal[0], x[1] - H.goal[1]) < H.goalR) { ttf = (k + 1) * dt; reached = true; break; }
    }
    return { collision_rate: outside / n, collided: outside > 0, min_h: minh, ttf: ttf, reached: reached, ess: essSum / n, activation: actSum / n, var_ratio: varSum / n, multi: multiSum / n, sat_active: satN ? satSum / satN : NaN, steps: n, path: path };
  }

  return { Rng, normPpf, Corridor, corridorMakeController, corridorEpisode, CDT, SG, harbour, obsCenter, minH, rowsHOCBF, solveRows, stepVessel, stepVesselSubs, clipInputs, vesselMakeController, vesselEpisode };
});
