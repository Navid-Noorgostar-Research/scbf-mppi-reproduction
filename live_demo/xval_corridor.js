const SIM = require('./sim.js');
function stats(arr){ const n=arr.length, m=arr.reduce((a,b)=>a+b,0)/n; const sd=Math.sqrt(arr.reduce((a,b)=>a+(b-m)**2,0)/Math.max(n-1,1)); return {mean:m, se: sd/Math.sqrt(n), median: arr.slice().sort((a,b)=>a-b)[Math.floor(n/2)]}; }
const N = parseInt(process.argv[2]||"30");
const base = {K:500, T:20, lam:1.0, sv:1.0, som:1.5, sigma:0.1, delta:0.003, maxSteps:250};
for (const kind of ["mppi","scbf_var","scbf_std","scbf_is","det"]) {
  const t0=Date.now(); const rs=[];
  for (let seed=0; seed<N; seed++) rs.push(SIM.corridorEpisode(kind, Object.assign({seed}, base)));
  const cr=stats(rs.map(r=>r.collision_rate)), tt=stats(rs.map(r=>r.ttf)), act=stats(rs.map(r=>r.activation)), vr=stats(rs.map(r=>r.var_ratio)), sat=stats(rs.map(r=>r.sat_active).filter(v=>!isNaN(v)));
  const reached=rs.filter(r=>r.reached).length/N, touched=rs.filter(r=>r.collided).length/N;
  console.log(kind.padEnd(9), "coll", cr.mean.toFixed(3), "±"+(2*cr.se).toFixed(3), "median", cr.median.toFixed(3), "| ttf", tt.mean.toFixed(0), "±"+(2*tt.se).toFixed(0), "| reached", reached.toFixed(2), "touched", touched.toFixed(2), "| act", act.mean.toFixed(2), "var", vr.mean.toFixed(2), "sat", isNaN(sat.mean)?"-":sat.mean.toFixed(3), "|", ((Date.now()-t0)/1000).toFixed(0)+"s");
}
