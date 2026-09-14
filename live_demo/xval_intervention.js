// Cross-validate the live demo's new intervention-penalty controller against what the GPU study says:
//   mu = 0 must reproduce the corrected controller EXACTLY, and ESS must fall monotonically with mu
//   while the barrier margin improves.  Same shape as the repository's existing xval_*.js logs.
const SIM = require("./sim.js");
const N = Number(process.argv[2] || 24);
const base = {K:500, T:20, lam:1.0, sv:1.0, som:1.5, sigma:0.1, delta:0.003, maxSteps:250};
const med = a => { const b = a.slice().sort((x,y)=>x-y); const n=b.length; return n%2 ? b[(n-1)/2] : 0.5*(b[n/2-1]+b[n/2]); };
function run(kind, mu){
  const R = [];
  for (let s=0; s<N; s++) R.push(SIM.corridorEpisode(kind, Object.assign({}, base, {seed:s, mu})));
  return {
    reached: R.filter(r=>r.reached).length / N,
    coll:    med(R.map(r=>r.collision_rate)),
    minh:    med(R.map(r=>r.min_h)),
    everBad: R.filter(r=>r.min_h < 0).length / N,
    ess:     R.reduce((a,r)=>a+r.ess,0)/N,
    ttf:     med(R.map(r=>r.ttf)),
  };
}
const rows = [["MPPI", "mppi", null], ["SCBF corrected (std)", "scbf_std", null], ["SCBF + IS", "scbf_is", null]];
for (const mu of [0, 0.1, 0.2, 0.3, 0.5, 1.0, 3.0]) rows.push([`  + penalty mu=${mu}`, "scbf_iv", mu]);
console.log(`corridor, ${N} seeds, K=${base.K}, delta=${base.delta}  (live-demo JavaScript core)\n`);
console.log("  " + "controller".padEnd(24) + "reached".padStart(9) + "med coll".padStart(10) + "med min h".padStart(11) + "ever<0".padStart(9) + "ESS".padStart(9) + "med ttf".padStart(9));
console.log("  " + "-".padEnd(80, "-"));
const out = {};
for (const [label, kind, mu] of rows){
  const r = run(kind, mu); out[label.trim()] = r;
  console.log("  " + label.padEnd(24) + (100*r.reached).toFixed(1).padStart(8) + "%" + r.coll.toFixed(4).padStart(10) + ((r.minh>=0?"+":"")+r.minh.toFixed(3)).padStart(11) + (100*r.everBad).toFixed(0).padStart(8) + "%" + r.ess.toFixed(1).padStart(9) + r.ttf.toFixed(0).padStart(9));
}
const a = out["SCBF corrected (std)"], b = out["+ penalty mu=0"];
const same = ["reached","coll","minh","everBad","ess","ttf"].every(k => Math.abs(a[k]-b[k]) <= 1e-12 * Math.max(1, Math.abs(a[k])));
console.log(`\n  mu = 0 reproduces the corrected controller: ${same ? "YES, to machine precision" : "NO -- the penalty leaks into the mu = 0 case"}`);
const es = [0,0.1,0.2,0.3,0.5,1.0,3.0].map(m => out[`+ penalty mu=${m}`].ess);
console.log(`  ESS across mu = 0 .. 3: ${es.map(v=>v.toFixed(1)).join("  ")}`);
console.log(`  monotonically falling: ${es.every((v,i)=>i===0||v<=es[i-1]+1e-9) ? "YES" : "no (report it as measured, do not claim monotone)"}`);
