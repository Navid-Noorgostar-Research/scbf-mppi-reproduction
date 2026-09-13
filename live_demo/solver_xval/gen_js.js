// random instances at the vessel's scale (as solver_nd.validate_against_cvxpy), solved by the JS core; written as JSON for the Python side
const S = require('../sim.js');
const rng = new S.Rng(2024);
const s0 = [350, 350, 120]; const z = S.normPpf(1 - 0.003), alpha = z;
const out = [];
for (const form of ["std", "variance"]) for (const J of [1, 2, 3, 4]) for (let i = 0; i < 400; i++) {
  const rows = []; const ubar = [rng.n() * 200, rng.n() * 200, rng.n() * 50];
  for (let j = 0; j < J; j++) { const a = [rng.n() * 1e-3, rng.n() * 1e-3, rng.n() * 2e-4]; const slack = a[0] * ubar[0] + a[1] * ubar[1] + a[2] * ubar[2]; const wn = Math.hypot(a[0] * s0[0], a[1] * s0[1], a[2] * s0[2]);
    // b so that roughly half the rows are active (nominal margin uniformly in [-1.5, 0.5] z·std)
    const u = (rng.n() + rng.n() + rng.n() + rng.n()) / 4; const b = slack - (u * 1.2 - 0.5) * z * wn; rows.push({ a: a, b: b }); }
  const r = S.solveRows(ubar, rows, s0, z, alpha, form);
  out.push({ form, J, ubar, A: rows.map(q => q.a), b: rows.map(q => q.b), mu: r.mu, P: r.P, active: r.active, multi: r.multi, infeasible: r.infeasible, residual: r.residual, s: r.sFirst, s_nom: r.sNomFirst, detRatio: r.detRatio, row: r.jFirst });
}
require('fs').writeFileSync(__dirname + '/js_instances.json', JSON.stringify({ s0, z, alpha, cases: out }));
// feasibility of every solved instance (all rows' margins >= -tol) and a stat of how many needed the joint solve
let nMulti = 0, nRes = 0, nAct = 0, worst = Infinity;
for (const c of out) { if (c.active) nAct++; if (c.multi) nMulti++; if (c.residual) nRes++;
  if (c.residual) continue;
  for (let j = 0; j < c.J; j++) { const a = c.A[j]; let v = 0; for (let i = 0; i < 3; i++) { let t = 0; for (let l = 0; l < 3; l++) t += c.P[l][i] * a[l]; v += t * t; } const sd = Math.sqrt(v); const pen = c.form === "std" ? c.form && z * sd : alpha * sd * sd;
    const mg = a[0] * (c.ubar[0] + c.mu[0]) + a[1] * (c.ubar[1] + c.mu[1]) + a[2] * (c.ubar[2] + c.mu[2]) - c.b[j] - pen; worst = Math.min(worst, mg / (1 + Math.abs(c.b[j]))); } }
console.log('instances', out.length, 'active', nAct, 'multi', nMulti, 'residual/infeasible', nRes, 'worst relative margin over all rows of all feasible instances', worst.toExponential(2));
