/* Cross-check the certificate panel against tests/certificate_conservatism.py.

   node live_demo/xval_certificate.js > live_demo/xval_certificate.log

   The panel accumulates, over every rollout of every cycle, a two-by-two table: did this rollout's drawn
   control violate a barrier row at any horizon step, and did the rollout's own trajectory ever leave the
   safe set.  The two conditionals are the certificate's soundness and its precision.

   This is a STATISTICAL check, not a bit-for-bit one: the browser core and the Python package draw from
   different generators, so the agreement to look for is between two Monte Carlo estimates of the same
   two conditional probabilities, at sample sizes that differ by two orders of magnitude.  The Python
   figures come from 64 seeds and 4,033,000 rollouts on the GPU; this runs a few hundred thousand.

   The soundness cell is deliberately reported as a raw count.  Rollouts that violate nothing are about
   0.6 % of the total and the conditional is 5 in 10,000, so a browser-sized run expects well under one
   event: "0 of 2,400" is the honest statement, and any attempt to quote 0.000545 from here would not be. */
const SIM = require("./sim.js");

const PY = { p0: 0.000545, p1: 0.2779, EM: 9.8534, exit: 0.276177 };
const OPT = { K: 500, T: 20, lam: 1.0, sv: 1.0, som: 1.5, sigma: 0.1, delta: 0.003,
              maxSteps: 250, umax: null };
const SEEDS = Number(process.argv[2] || 12);

function scoreKind(kind) {
  const acc = { n0: 0, e0: 0, n1: 0, e1: 0 };
  for (let seed = 0; seed < SEEDS; seed++) {
    const o = Object.assign({}, OPT, { seed });
    SIM.corridorEpisode(kind, o, function (k, x, ctrl) {
      const q = ctrl.stats.cert;
      if (q) { acc.n0 += q.n0; acc.e0 += q.e0; acc.n1 += q.n1; acc.e1 += q.e1; }
      return true;
    });
  }
  return acc;
}

function line(name, q) {
  const tot = q.n0 + q.n1;
  const p0 = q.n0 ? q.e0 / q.n0 : NaN;
  const p1 = q.n1 ? q.e1 / q.n1 : NaN;
  console.log(
    name.padEnd(20) +
    String(tot.toLocaleString()).padStart(12) +
    (100 * (q.n1 / tot)).toFixed(2).padStart(11) + "%" +
    (q.n0 ? (q.e0 + " of " + q.n0) : "-").padStart(16) +
    (isNaN(p1) ? "-" : p1.toFixed(4)).padStart(11));
  return { tot, p0, p1 };
}

console.log("certificate panel, browser core, " + SEEDS + " seeds, K=" + OPT.K + ", T=" + OPT.T);
console.log("");
console.log("controller".padEnd(20) + "rollouts".padStart(12) + "  M>=1".padStart(10) +
            "exits | M=0".padStart(17) + "exits | M>=1".padStart(12));
const mppi = line("plain MPPI", scoreKind("mppi"));
const std = line("SCBF corrected", scoreKind("scbf_std"));
console.log("");
console.log("against tests/certificate_conservatism.py (64 seeds, 4,033,000 rollouts, GPU):");
console.log("  Pr(exit | M >= 1)   browser " + mppi.p1.toFixed(4) +
            "   python " + PY.p1.toFixed(4) +
            "   difference " + Math.abs(mppi.p1 - PY.p1).toFixed(4));
console.log("  Pr(exit | M = 0)    browser " + (mppi.n0 === 0 ? "-" : mppi.p0) +
            "   python " + PY.p0 + "   (browser sample far too small to resolve this; count reported above)");
console.log("");
console.log("");
console.log("The two controllers read the certificate completely differently, which is the point of the");
console.log("panel.  Under plain MPPI the row is a sound certificate and an imprecise one.  Under the");
console.log("corrected filter the row holds almost everywhere -- the solver puts it there -- and among the");
console.log("rollouts satisfying every row MORE leave the safe set than among those violating one, because");
console.log("a continuous-time barrier condition imposed at discrete steps on a curved wall does not give");
console.log("discrete forward invariance.  Python, 64 seeds: 0.000545 / 0.2779 for plain MPPI and");
console.log("0.0605 / 0.0273 for the filter (tests/certificate_conservatism.py --kind mppi|paper).");
