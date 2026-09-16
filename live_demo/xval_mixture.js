/* Cross-check the mixture panel against tests/mixture_recovery.py.

   node live_demo/xval_mixture.js > live_demo/xval_mixture.log

   Both sides implement the same constructed scalar problem -- x_{t+1} = x_t + 0.05 v_t, safe box
   [-0.5, 0.5], goal 0.4, T = 20, nominal N(0,1), cost sum_t (x_t - 0.4)^2 + (x_T - 0.4)^2, lambda = 1,
   rows h(x_+) >= (1 - 0.05) h(x) -- and estimate the same known first-control mean through a defensive
   mixture, weighted by the exact path density ratio.

   This is a statistical check: different generators, so the quantities to compare are the ones with
   small batch-to-batch spread.  The per-row violation RATE is nearly deterministic and should agree to
   several digits; COVERAGE and RMSE should agree closely; the MEAN ESTIMATE is heavy-tailed at small
   mixing weights and will not, which is itself the finding -- an estimator whose batch mean wanders is
   an estimator that has not recovered its target. */
const SIM = require("./sim.js");

const PY = {                       // tests/mixture_recovery.py, 2000 batches of K = 500
  "0.0000/0.0030": { est: 0.061613, rmse: 0.5450, cover: 0.0015, rate: 0.00300 },
  "0.0000/0.0015": { est: 0.043064, rmse: 0.5716, cover: 0.0000, rate: 0.00150 },
  "0.0015/0.0015": { est: 0.060360, rmse: 0.8510, cover: 0.5430, rate: 0.00197 },
  "0.0049/0.0015": { est: 0.204548, rmse: 0.8718, cover: 0.9105, rate: 0.00302 },
  "0.0200/0.0015": { est: 0.422242, rmse: 0.5427, cover: 1.0000, rate: 0.00768 },
  "0.1000/0.0015": { est: 0.480738, rmse: 0.2456, cover: 1.0000, rate: 0.03224 },
};
const NB = Number(process.argv[2] || 600);

console.log("mixture panel, browser core, " + NB + " batches of K=500 per row");
console.log("exact target mean: browser " + SIM.MIX_TRUTH.toFixed(7) + "   python 0.4943900");
console.log("");
console.log("     w   dTail" + "rate".padStart(11) + "  (py)".padStart(10) +
            "cover".padStart(9) + "  (py)".padStart(9) +
            "RMSE".padStart(9) + "  (py)".padStart(9) + "est".padStart(10) + "  (py)".padStart(9));

let worstRate = 0, worstCover = 0, worstRmse = 0;
for (const key of Object.keys(PY)) {
  const parts = key.split("/");
  const w = Number(parts[0]), dT = Number(parts[1]);
  const r = SIM.mixtureRecovery({ w: w, dTail: dT, nb: NB, K: 500, seed: 3 });
  const py = PY[key];
  worstRate = Math.max(worstRate, Math.abs(r.rate - py.rate));
  worstCover = Math.max(worstCover, Math.abs(r.cover - py.cover));
  worstRmse = Math.max(worstRmse, Math.abs(r.rmse - py.rmse));
  console.log(
    w.toFixed(4).padStart(6) + dT.toFixed(4).padStart(8) +
    r.rate.toFixed(5).padStart(11) + py.rate.toFixed(5).padStart(10) +
    (100 * r.cover).toFixed(1).padStart(8) + "%" + (100 * py.cover).toFixed(1).padStart(8) + "%" +
    r.rmse.toFixed(4).padStart(9) + py.rmse.toFixed(4).padStart(9) +
    r.est.toFixed(4).padStart(10) + py.est.toFixed(4).padStart(9));
}
console.log("");
console.log("worst disagreement   per-row rate " + worstRate.toExponential(2) +
            "   coverage " + worstCover.toFixed(4) + "   RMSE " + worstRmse.toFixed(4));
console.log("");
console.log("The panel's claim is the shape of the coverage and RMSE columns, not any single estimate:");
console.log("below the cap the missing class returns while the RMSE gets worse than with no mixture,");
console.log("and the mean only comes back at a weight roughly ten times the cap.");
