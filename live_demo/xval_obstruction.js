// Check the "Can any Gaussian do this?" panel against FINDINGS.md section 2b: the panel reads row0.rows,
// which the simulation core builds from the same barrier rows as the Python package, and must reproduce
// the published figures at the corridor's default start state.
const SIM = require("./sim.js");
const env = new SIM.Corridor(false);
const opt = {K:200, T:20, lam:1.0, sv:1.0, som:1.5, sigma:0.1, delta:0.003, seed:0, maxSteps:250};
const c = SIM.corridorMakeController(env, "scbf_std", opt);
c.plan([0.0, 0.5, 0.0]);                       // the default start state
const r0 = c.stats.row0;

// exactly the arithmetic drawObstruction performs
let lo = -Infinity, hi = Infinity;
for (const row of r0.rows){ if (row.a > 1e-12) lo = Math.max(lo, row.b/row.a); else if (row.a < -1e-12) hi = Math.min(hi, row.b/row.a); }
const width = hi - lo, z = r0.z, s0 = r0.sig;
const cap = width/(2*z), floorN = s0/Math.SQRT2;
const T = 20, dt = 0.05, lam = opt.lam;
const floorF = Math.sqrt(1/(2/(s0*s0) + 4*(T+1)*dt*dt/lam));
const ok = width > Math.SQRT2*z*s0;

const f = v => v.toFixed(6);
const exp = {lo:-1/Math.PI, hi:1/Math.PI, cap:0.115843, floorN:0.707107, floorF2:0.452489, width:0.636620, need:3.885950, capsq:0.013420};
const rows = [
  ["admissible speed lower l", lo, exp.lo],
  ["admissible speed upper r", hi, exp.hi],
  ["interval width r - l", width, exp.width],
  ["cap (r-l)/2z", cap, exp.cap],
  ["floor sigma0/sqrt(2)", floorN, exp.floorN],
  ["full-target floor squared", floorF*floorF, exp.floorF2],
  ["sqrt(2) z sigma0", Math.SQRT2*z*s0, exp.need],
  ["cap squared", cap*cap, exp.capsq],
];
console.log("Corridor start state x = 0, y = 0.5, theta = 0;  sigma0 = " + s0 + ",  z = " + z.toFixed(6) + "\n");
console.log("  " + "quantity".padEnd(28) + "panel".padStart(12) + "FINDINGS 2b".padStart(14) + "  agree");
console.log("  " + "-".repeat(62));
let worst = 0;
for (const [name, got, want] of rows){
  const d = Math.abs(got - want); worst = Math.max(worst, d);
  console.log("  " + name.padEnd(28) + f(got).padStart(12) + f(want).padStart(14) + "  " + (d < 1e-6 ? "yes" : "NO  (" + d.toExponential(2) + ")"));
}
console.log("\n  verdict chip: " + (ok ? "a Gaussian exists here" : "no Gaussian at all")
  + "   (expected: no Gaussian at all)");
console.log("  fails by a factor of " + (Math.SQRT2*z*s0/width).toFixed(1) + "   (expected: 6.1)");
console.log("\n  worst absolute disagreement with FINDINGS.md section 2b: " + worst.toExponential(2));
