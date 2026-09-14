// Cross-validate the live demo's budgeted controller (scbf_bud) against the GPU study.  The browser core
// searches a 97-point beta grid where scbf_mppi/ess_budget.py uses 193, so the two are not bit-identical;
// what must hold is that the effective sample size rises monotonically with the target and that the
// 10 % setting lands on the 512-seed GPU figure.
const SIM = require("./sim.js");
const N = Number(process.argv[2] || 16);
const base = {K:500, T:20, lam:1.0, sv:1.0, som:1.5, sigma:0.1, delta:0.003, maxSteps:250};
const med = a => { const b = a.slice().sort((x,y)=>x-y); const n=b.length; return n%2 ? b[(n-1)/2] : 0.5*(b[n/2-1]+b[n/2]); };
function run(kind, extra){
  const R = [];
  for (let s = 0; s < N; s++) R.push(SIM.corridorEpisode(kind, Object.assign({}, base, {seed:s}, extra||{})));
  return { reached: R.filter(r=>r.reached).length/N, coll: med(R.map(r=>r.collision_rate)),
           minh: med(R.map(r=>r.min_h)), everBad: R.filter(r=>r.min_h<0).length/N,
           ess: R.reduce((a,r)=>a+r.ess,0)/N, ttf: med(R.map(r=>r.ttf)) };
}
const rows = [["MPPI","mppi",null], ["SCBF corrected (std)","scbf_std",null], ["SCBF + IS","scbf_is",null]];
const TGT = [0.05, 0.10, 0.25, 0.50];
for (const t of TGT) rows.push([`  budget ESS>=${(100*t).toFixed(0)}%`, "scbf_bud", {essTarget:t}]);
console.log(`corridor, ${N} seeds, K=${base.K}, delta=${base.delta}  (live-demo JavaScript core, 97-point beta grid)\n`);
console.log("  " + "controller".padEnd(24) + "reached".padStart(9) + "med coll".padStart(10) + "med min h".padStart(11) + "ever<0".padStart(9) + "ESS".padStart(9) + "med ttf".padStart(9));
console.log("  " + "-".repeat(80));
const ess = {};
for (const [label, kind, ex] of rows){
  const r = run(kind, ex); if (ex) ess[ex.essTarget] = r.ess;
  console.log("  " + label.padEnd(24) + (100*r.reached).toFixed(1).padStart(8) + "%" + r.coll.toFixed(4).padStart(10) +
    ((r.minh>=0?"+":"")+r.minh.toFixed(3)).padStart(11) + (100*r.everBad).toFixed(0).padStart(8) + "%" +
    r.ess.toFixed(1).padStart(9) + r.ttf.toFixed(0).padStart(9));
}
const seq = TGT.map(t=>ess[t]);
console.log(`\n  ESS across targets 5, 10, 25, 50 %: ${seq.map(v=>v.toFixed(1)).join("  ")}`);
console.log(`  monotonically rising: ${seq.every((v,i)=>i===0||v>=seq[i-1]-1e-9) ? "YES" : "no (report as measured)"}`);
console.log(`\n  GPU reference, 512 seeds, target 10 %: reached 97.5 %, med coll 0.0440, med min h -0.049, ESS 33.7`);
console.log(`  browser at the same target:            ESS ${ess[0.10].toFixed(1)}`);
