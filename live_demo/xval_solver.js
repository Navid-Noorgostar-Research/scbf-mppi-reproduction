// The solver port (13 Sep): MPPI, det. MPPI, printed and the whole corridor must be bit-identical to the previous core
// (sim_v5_backup.js = the core with the sequential-sweep solver); the SCBF variants change only through the solver.
const A = require('./sim_v5_backup.js'), B = require('./sim.js');
const base = {scenario:"crossing", K:500, T:15, lam:300, s0:[350,350,120], delta:0.003, disturbance:"white", sF:[120,300,400], tauc:5, maxTime:150};
let same = true; const T0 = Date.now();
for (const kind of ["mppi","det","printed"]) for (const seed of [0,1,2]) {
  const a = A.vesselEpisode(kind, Object.assign({seed}, base)), b = B.vesselEpisode(kind, Object.assign({seed}, base));
  const eq = a.ttf===b.ttf && a.collision_rate===b.collision_rate && a.min_h===b.min_h && a.path.length===b.path.length && a.path.every((p,i)=>p[0]===b.path[i][0]&&p[1]===b.path[i][1]);
  same = same && eq; console.log(kind, seed, eq ? "identical" : "DIFF", "ttf", a.ttf, b.ttf, "coll", a.collision_rate.toFixed(3), b.collision_rate.toFixed(3));
}
for (const kind of ["scbf","scbf_is","scbf_var"]) for (const seed of [0,1,2]) {
  const t0 = Date.now(); const a = A.vesselEpisode(kind, Object.assign({seed}, base)); const t1 = Date.now(); const b = B.vesselEpisode(kind, Object.assign({seed}, base)); const t2 = Date.now();
  console.log(kind, seed, "old: ttf", a.ttf, "coll", a.collision_rate.toFixed(3), "minh", a.min_h.toFixed(2), "infeas", a.infeasible === undefined ? "-" : a.infeasible, "| new: ttf", b.ttf, "coll", b.collision_rate.toFixed(3), "minh", b.min_h.toFixed(2), "sat", b.sat_active.toFixed(3), "multi", b.multi.toFixed(3), "| ms/step old", ((t1-t0)/a.steps).toFixed(0), "new", ((t2-t1)/b.steps).toFixed(0));
}
const cb = {K:500, T:20, lam:1.0, sv:1.0, som:1.5, sigma:0.1, delta:0.003, maxSteps:250};
for (const kind of ["mppi","scbf_var","scbf_std","scbf_is","det"]) for (const seed of [0,1]) {
  const a = A.corridorEpisode(kind, Object.assign({seed}, cb)), b = B.corridorEpisode(kind, Object.assign({seed}, cb));
  const eq = a.collision_rate===b.collision_rate && a.ttf===b.ttf && a.min_h===b.min_h && a.path.every((p,i)=>p[0]===b.path[i][0]&&p[1]===b.path[i][1]);
  same = same && eq; console.log("corridor", kind, seed, eq ? "identical" : "DIFF");
}
console.log(same ? "UNCHANGED PATHS IDENTICAL (bit for bit)" : "MISMATCH ON A PATH THAT MUST NOT CHANGE", "| total", ((Date.now()-T0)/1000).toFixed(0), "s");
