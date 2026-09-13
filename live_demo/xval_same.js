const A = require('./sim_v3_backup.js'), B = require('./sim.js');   // v1 = the core before the diagnostics were added for every controller
const base = {scenario:"crossing", K:500, T:15, lam:300, s0:[350,350,120], delta:0.003, disturbance:"white", sF:[120,300,400], tauc:5, maxTime:150};
let same = true;
for (const kind of ["mppi","scbf","scbf_is","det"]) for (const seed of [0,1,2]) {
  const a = A.vesselEpisode(kind, Object.assign({seed}, base)), b = B.vesselEpisode(kind, Object.assign({seed}, base));
  // trajectories must be identical; the collision count / min h may differ because the new core evaluates them at every 0.25 s RK4 sub-step (as the Python package does), the old one at the 1 s control step only
  const eq = a.ttf===b.ttf && a.path.length===b.path.length && a.path.every((p,i)=>p[0]===b.path[i][0]&&p[1]===b.path[i][1]);
  same = same && eq; console.log(kind, seed, eq ? "identical path" : "DIFF", "ttf", a.ttf, b.ttf, "| min h step-only", a.min_h.toFixed(3), "sub-steps", b.min_h.toFixed(3), "| coll", a.collision_rate.toFixed(3), b.collision_rate.toFixed(3), "| diagnostics act", b.activation.toFixed(3), "var", b.var_ratio.toFixed(3), "sat", b.sat_active.toFixed(3), "ess", b.ess.toFixed(1));
}
const cb = {K:500, T:20, lam:1.0, sv:1.0, som:1.5, sigma:0.1, delta:0.003, maxSteps:250};
for (const kind of ["mppi","scbf_var","scbf_std","scbf_is","det"]) for (const seed of [0,1]) {
  const a = A.corridorEpisode(kind, Object.assign({seed}, cb)), b = B.corridorEpisode(kind, Object.assign({seed}, cb));
  const eq = a.collision_rate===b.collision_rate && a.ttf===b.ttf && a.min_h===b.min_h && a.path.every((p,i)=>p[0]===b.path[i][0]&&p[1]===b.path[i][1]);
  same = same && eq; console.log("corridor", kind, seed, eq ? "identical" : "DIFF");
}
console.log(same ? "ALL IDENTICAL (paths bit for bit)" : "MISMATCH");
