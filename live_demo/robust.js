// Robustness of a story seed to ulp-level arithmetic differences (browsers use different libm): perturb x0 by 1e-9 and repeat
const S = require('./sim.js');
const seeds = process.argv.slice(2).map(Number); const REPS = 6;
const base = {scenario:"crossing", K:500, T:15, lam:300, s0:[350,350,120], delta:0.003, disturbance:"white", sF:[120,300,400], tauc:5, maxTime:150};
function episode(kind, seed, x0){
  const H = S.harbour('crossing'); const c = S.vesselMakeController(H, kind, Object.assign({seed}, base)); c.reset();
  const rng = new S.Rng(10000 + seed); const dt = 1; let x = x0.slice(); let out = 0, first = null, minh = Infinity, ttf = null;
  for (let k = 0; k < 150; k++) { const u = c.plan(x); let subs; [x, subs] = S.stepVesselSubs(x, u, dt, null, null, [120*Math.sqrt(10), 300*Math.sqrt(10), 400*Math.sqrt(10)], rng);
    let h = Infinity; for (let i = 0; i < subs.length; i++) { const hi = S.minH(H, subs[i], k*dt + (i+1)*dt/subs.length); if (hi < h) h = hi; }
    if (h < minh) minh = h; if (h < 0) { out++; if (first === null) first = k+1; }
    if (Math.hypot(x[0]-H.goal[0], x[1]-H.goal[1]) < H.goalR) { ttf = k+1; break; } }
  return {out, first, minh: +minh.toFixed(2), ttf};
}
for (const seed of seeds) {
  const prng = new S.Rng(777 + seed); const res = {};
  for (let rep = 0; rep < REPS; rep++) { const x0 = rep === 0 ? [0,0,0,1.5,0,0] : [1e-9*prng.n(), 1e-9*prng.n(), 1e-10*prng.n(), 1.5 + 1e-9*prng.n(), 0, 0];
    for (const kind of ["mppi","scbf","scbf_is","det"]) { (res[kind] = res[kind] || []).push(episode(kind, seed, x0)); } }
  const line = {seed}; for (const kind of Object.keys(res)) line[kind] = res[kind].map(r => `${r.out}${r.first!==null ? '@'+r.first : ''}/${r.ttf||'-'}`).join(' ');
  console.log(JSON.stringify(line));
}
