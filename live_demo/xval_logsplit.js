// Cross-validate the live demo's log-density-ratio split against tests/logratio_decomposition.py.
// The split is  log p/q = sum_t[-(1/2)((m+s xi)/s0)^2 - log s0] + sum_t[(1/2)xi^2] + sum_t[log s].
// The arithmetic check that it is the RIGHT split needs no reference at all: for plain MPPI, m = 0 and
// s = s0 at every sample, so the three parts must cancel and the total spread must be exactly zero.
const SIM = require("./sim.js");
const env = new SIM.Corridor(false);
const base = {K:400, T:20, lam:1.0, sv:1.0, som:1.5, sigma:0.1, delta:0.003, seed:3, maxSteps:250, essTarget:0.10};
const WARM = 24;
function at(kind){
  const c = SIM.corridorMakeController(env, kind, base);
  const rng = new SIM.Rng(10000 + base.seed); let x = [0,0.5,0], out = null;
  for (let k = 0; k < WARM + 1; k++){
    const u = c.plan(x); if (k === WARM) out = c.stats.split;
    const dt = SIM.CDT, s = base.sigma;
    x = [x[0]+u[0]*Math.cos(x[2])*dt+s*Math.sqrt(dt)*rng.n(), x[1]+u[0]*Math.sin(x[2])*dt+s*Math.sqrt(dt)*rng.n(), x[2]+u[1]*dt+s*Math.sqrt(dt)*rng.n()];
  }
  return out;
}
console.log(`corridor, K=${base.K}, seed ${base.seed}, cycle ${WARM}  (live-demo JavaScript core)\n`);
console.log("  " + "controller".padEnd(22) + "sd(shift)".padStart(11) + "sd(noise)".padStart(11) + "sd(log s)".padStart(11) + "sd(total)".padStart(11) + "corr".padStart(8) + "  weights");
console.log("  " + "-".repeat(84));
const got = {};
for (const [lab, kind] of [["MPPI","mppi"], ["SCBF as printed (var)","scbf_var"], ["SCBF corrected (std)","scbf_std"],
                           ["SCBF + IS","scbf_is"], ["SCBF + ESS budget","scbf_bud"]]){
  const o = at(kind); got[kind] = o;
  console.log("  " + lab.padEnd(22) + o.shift.toFixed(3).padStart(11) + o.noise.toFixed(3).padStart(11) +
    o.shrink.toFixed(3).padStart(11) + o.total.toFixed(3).padStart(11) +
    (isFinite(o.corr) ? o.corr.toFixed(3) : "n/a").padStart(8) + "  " + (o.applied ? "applied" : "discarded"));
}
const m = got.mppi;
console.log(`\n  ARITHMETIC CHECK - plain MPPI has m = 0 and s = sigma_0 at every sample, so the three`);
console.log(`  parts must cancel exactly:  sd(total) = ${m.total.toExponential(2)}  ->  ${m.total === 0 ? "EXACT, the split is right" : "NOT ZERO - the split is wrong"}`);
const is_ = got.scbf_is;
console.log(`\n  With the weights restored the shrink term carries ${(100*is_.shrink/is_.total).toFixed(1)} % of the total spread,`);
console.log(`  correlated with it at ${is_.corr.toFixed(3)}.  Python, over six cycles of a real episode:`);
console.log(`  sd(shift) 2.49, sd(noise) 3.10, sd(log s) 44.65, corr +0.994 (tests/logratio_decomposition.py).`);
console.log(`\n  The budget flattens the term that does the selecting: sd(log s) ${got.scbf_bud.shrink.toFixed(3)} against`);
console.log(`  ${is_.shrink.toFixed(3)} with the weights merely restored - which is what holding E_q[w^2] at kappa per step buys.`);
