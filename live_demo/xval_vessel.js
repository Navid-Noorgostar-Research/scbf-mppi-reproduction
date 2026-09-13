const SIM = require('./sim.js');
function stats(arr){ const n=arr.length, m=arr.reduce((a,b)=>a+b,0)/n; const sd=Math.sqrt(arr.reduce((a,b)=>a+(b-m)**2,0)/Math.max(n-1,1)); return {mean:m, se: sd/Math.sqrt(n)}; }
const N = parseInt(process.argv[2]||"10");
const base = {scenario:"static", K:500, T:15, lam:300, s0:[350,350,120], delta:0.003, disturbance:"white", sF:[120,300,400], tauc:5, maxTime:150};
const out = {};
for (const kind of ["mppi","scbf","scbf_is"]) {
  const t0=Date.now(); const rs=[];
  for (let seed=0; seed<N; seed++) rs.push(SIM.vesselEpisode(kind, Object.assign({seed}, base)));
  const cr=stats(rs.map(r=>r.collision_rate)), tt=stats(rs.map(r=>r.ttf)), act=stats(rs.map(r=>r.activation)), vr=stats(rs.map(r=>r.var_ratio)), sat=stats(rs.map(r=>r.sat_active).filter(v=>!isNaN(v))), ess=stats(rs.map(r=>r.ess)), mh=stats(rs.map(r=>r.min_h));
  const reached=rs.filter(r=>r.reached).length/N, touched=rs.filter(r=>r.collided).length/N;
  out[kind]={coll:cr.mean, coll_se:cr.se, ttf:tt.mean, reached, touched, act:act.mean, var:vr.mean, sat:sat.mean, ess:ess.mean, min_h:mh.mean, n:N};
  console.log(kind.padEnd(8), "coll", cr.mean.toFixed(3), "±"+(2*cr.se).toFixed(3), "| ttf", tt.mean.toFixed(0), "±"+(2*tt.se).toFixed(0), "| reached", reached.toFixed(2), "touched", touched.toFixed(2), "| min_h", mh.mean.toFixed(1), "| ess", ess.mean.toFixed(0), "| act", act.mean.toFixed(2), "var", vr.mean.toFixed(2), "sat", isNaN(sat.mean)?"-":sat.mean.toFixed(3), "|", ((Date.now()-t0)/1000).toFixed(0)+"s");
}
require('fs').writeFileSync(__dirname + '/xval_vessel.json', JSON.stringify(out, null, 1));
