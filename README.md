# Reproduction: *Path Integral Methods with Stochastic Control Barrier Functions*

Tao, Yoon, Kim, Hovakimyan, Voulgaris — arXiv:2206.11985 (June 2022); IEEE CDC 2022, pp. 1654–1659.

This package re-implements the paper's Algorithm 1 (SCBF-MPPI) and its baseline (MPPI) on the paper's own
example — a unicycle with additive Brownian noise in a sinusoidal corridor — and turns an independent,
referee-style review of the paper's claims into experiments. Everything is plain Python: NumPy for the vectorised rollouts,
SciPy for quantiles, cvxpy + Clarabel for the exact per-sample SDP/SOCP (validation and wall-clock only),
matplotlib for figures and the animation. One command per figure; every number reported below comes from
`results/*.json`.

```
pip install -r requirements.txt
python -m scbf_mppi.selftest                          # ~5 s: solver vs cvxpy, bridge crossing law, Theorem 2, one episode each
python -m scbf_mppi.experiments --exp all --runs 10   # ~25 min on two cores; writes results/*.json
python -m scbf_mppi.figures                           # writes figures/*.png and results/E*_summary.txt
python -m scbf_mppi.animate                           # figures/anim_mppi_vs_scbf.mp4 and .gif (needs ffmpeg for the mp4)
python -c "from scbf_mppi import scbf; print(scbf.validate_against_cvxpy(200, form='std', z=2.748))"
```

The shipped `results/*.json` were produced with these exact calls — about an hour on two cores:

```
python -m scbf_mppi.experiments --exp E1,E8,E11,E14 --runs 30   # 30 seeds per setting
python -m scbf_mppi.experiments --exp E2 --runs 30              # E2 uses runs/2 = 15 seeds per σ
python -m scbf_mppi.experiments --exp E12,E13 --runs 10         # E13: 10 seeds; E12 uses max(5, runs/2) = 5
python -m scbf_mppi.experiments --exp E3,E4,E6,E9               # deterministic / timing — no seeds
```

## What is implemented, and from where in the paper

| Element | Where | Implementation |
|---|---|---|
| Dynamics | Sec. V-A: unicycle, `σ dW` with σ = I₃, Δt = 0.05 | `dynamics.py`, Euler–Maruyama; nominal (noise-free) model for the rollouts, as in eq. (9) |
| Safe set, barriers | Sec. V-B: C = {sin(πx/2) < y < sin(πx/2)+1}; printed h uses sin x | `env.py` — `Corridor(printed=False)` uses the set as defined; `printed=True` the barriers as printed |
| Cost | Sec. V-B: q = ‖X−X_g‖² + 1000·1[X∉C]; λ = 1; T = 20 | `mppi.py`; terminal φ = ‖X_T−X_g‖² (**not stated — assumption**) |
| MPPI | Sec. III-B, eq. (4), q̃ with ν = 1, R = λΣ₀⁻¹ | `MPPI` |
| SCBF chance constraint | eq. (5); Theorem 2 / eq. (6),(8) | `scbf.solve_rows` — `form="variance"` is the constraint **as printed**, `form="std"` the chance constraint it claims |
| Per-sample problem (7)/(8) | objective ‖μ−μ₀‖₁ + ‖P−P₀‖_F | closed-form one-dimensional solve (exact up to grid resolution because the ω column of L_g h is zero; validated against cvxpy in `scbf.validate_against_cvxpy`) |
| Algorithm 1 | Sec. III-C | `SCBFMPPI` (per sample, per timestep) with modes `both`, `mean_only`, `var_only`, optional importance-sampling correction |
| Deterministic MPPI | Homburger, Messerer, Diehl, Reuter, L-CSS 2025, Alg. 1 | `DeterministicMPPI` (β = ν^j, λ ← β²λ₀, Σ ← β²Σ₀, correction λWᵀΣ⁻¹U) |
| Metrics | Sec. V-C: collision rate = fraction of states outside C; TTF; goal radius 0.15; ≤ 250 steps | `simulate.py` — plus activation fraction, realised Var[δv]/Var₀, delivered Pr(Au ≥ b), ESS, min h, between-sample excursions (Brownian bridge) |

## What the paper does not state, and what was assumed

* **Σ₀, the sampling covariance of the perturbations.** Not reported. `sigma_v = 1.0, sigma_om = 1.5` were chosen because at that
  setting plain MPPI reproduces Table I (collision rate ≈ 0.05, TTF ≈ 130 steps at K = 500). See E1.
* **The process-noise level.** The paper says σ is the identity. At σ = I₃ and Δt = 0.05 every controller — and a robot with no
  control at all — leaves the corridor (E2). Table I is only reproducible at σ ≈ 0.1. All main experiments use σ = 0.1 and say so.
* **α in eq. (6)** is "the confidence interval corresponding to 1−δ": taken as the one-sided quantile z = Φ⁻¹(0.997) = 2.748
  (the paper's 1−γ = 0.997); `alpha=3` is also run (E11).
* **Terminal cost** φ: not given; ‖X_T − X_g‖².
* **The class-K function** in the SCBF condition is the identity (eq. (3): "≥ −h").
* **Input bounds**: none in the paper; none by default (`u_max` available; E11 runs one bounded case).
* **The SDP norm p** in (7)/(8): Frobenius.

## The experiments

| | Question | Result file / figure |
|---|---|---|
| E1 | Does Table I / Fig. 1 reproduce? With the constraint as printed and as corrected? | `E1_table1.json`, `fig_E1_*.png` |
| E2 | Is the stated noise σ = I compatible with Table I? | `E2_sigma_sweep.json`, `fig_E2_sigma.png` |
| E3 | What does Algorithm 1 cost in seconds? (10,000 SDPs per cycle) | `E3_wallclock.json`, `fig_E3_wallclock.png` |
| E4 | What probability does the printed constraint actually deliver? (Theorem 2, variance vs std) | `E4_theorem2.json`, `fig_E4_theorem2.png` |
| E5 | Braking or steering? |v| and |ω| against distance to the wall | `fig_E5_braking.png` |
| E6 | Can Table II (N₁, N₂) be evaluated from measured quantities? | `E6_samplecount.json` |
| E7 | Do zero collisions at sample times mean the path stayed inside? | `fig_E7_bridge.png` |
| E8 | Which mechanism produces the safety — the mean shift, the variance reduction, or both? Against deterministic MPPI at equal rollouts | `E8_mechanisms.json`, `fig_E8_mechanisms.png` |
| E9 | Theorem 1 (Clark 2021) — the So–Clark–Fan counterexample, simulated | `E9_clark.json`, `fig_E9_clark.png` |
| E11 | The corrected constraint at other confidence levels — the exploration/safety trade-off | `E11_delta_sweep.json`, `fig_E11_delta.png` |
| E12 | The printed barriers put the goal outside the safe set | `E12_printed_barriers.json`, `fig_E12_printed.png` |
| E13 | Deterministic MPPI: is the E8 comparison sensitive to the annealing schedule (ν, iterations × samples)? | `E13_det_sensitivity.json` |
| E14 | With input bounds (|v| ≤ 2, |ω| ≤ 4) — does the ranking of MPPI / SCBF-MPPI / + IS correction change? | `E14_bounded.json` |

Seeds are fixed (`seed = 0 … runs−1`); rerunning with the calls above reproduces the JSON files exactly on the same platform. Confidence intervals in the figures are 2000-resample bootstrap intervals over seeds.

## On the water — the method transferred to the ISD research vessel Solgenia (`scbf_mppi/vessel/`)

The second study applies the same three controllers to the group's own vessel, using its published model unchanged:
Homburger, Wirtensohn, Hoher, Baur, Griesser, Diehl, Reuter, *Solgenia — A Test Vessel Toward Energy-Efficient Autonomous
Water Taxi Applications*, Ocean Engineering 2025 (arXiv:2502.01207), Table A.5, and the MATLAB files in
`github.com/hhomb/Solgenia` (`ShipModel.m`, `PropModel.m`, `ParametersSolgenia.m`). `vessel/solgenia.py` is a line-by-line
vectorised port; `python -m scbf_mppi.vessel.selftest` checks its drift against the MATLAB `f(x, u)` at 200 random states (agreement
to 10⁻¹⁵) and reports the one abstraction the port makes — the thrust is held over an integration step instead of being
re-evaluated at every RK4 stage — as a 2 mm difference over the 10 s of `MinimumExample.m`.

```
python -m scbf_mppi.vessel.selftest                       # ~10 s: model vs MATLAB, barrier rows vs finite differences,
                                                          #        L_g h = 0, Itô term, solver vs cvxpy, delivered probability
python -m scbf_mppi.vessel.experiments --exp all --runs 30 # ~2.5 h on two cores; writes results/vessel_V*.json
python -m scbf_mppi.vessel.figures                        # figures/fig_V*.png, results/vessel_summary.txt
python -m scbf_mppi.vessel.animate                        # figures/anim_vessel_mppi_vs_scbf.mp4 / .gif
```

| Element | Choice | Where |
|---|---|---|
| Vessel | 3-DOF Fossen-type model, all 27 dynamic + 12 thruster parameters from Table A.5 / `ParametersSolgenia.m` (repository rounding) | `vessel/solgenia.py` |
| Input | u = (X_AT, Y_AT, F_BT) [N]: azimuth thrust resolved in the body frame + bow thrust, so τ = Bu is linear (the method needs input-affine dynamics; n\|n\| laws and an azimuth angle are not). Azimuth thrust in the 700 N disk (n ≤ 2000/60 Hz, c_AT = 0.63, d_AT = 0); bow thrust ≤ 244 N·e^(−0.62u²) | `solgenia.clip_inputs` |
| Integration | RK4 at 0.25 s (the group's `RK4_step.m`), four sub-steps per 1 s control interval; additive force noise by Euler–Maruyama per sub-step | `solgenia.step` |
| Neglected | actuator time constants (0.1–0.3 s) and rate limits; roll/pitch/heave (as the group does) | — |
| Scenario | start (0, 0) heading east at 1.5 m/s, goal (120, 0), goal radius 5 m; three moored obstacles (inflated radii 12, 14, 11 m = obstacle + 5 m ego radius); the crossing scenario adds the group's Experiment-III ferry: R = 25 m, 3 m/s northbound, on the track at x = 66 m at t = 22 s | `vessel/harbour.py` |
| Barrier | h_j = ‖p − c_j(t)‖ − R_j (their circle scheme). As printed the row is 0 ≥ b (L_g h = 0). Second-order: ψ₁ = ḣ + α₁h, ψ̇₁ + α₂ψ₁ ≥ 0 ⇔ a·τ ≥ b; α₁ = 0.1 s⁻¹, α₂ = 0.4 s⁻¹ (sweep in V6). Itô term computed in general (`ito_term`); zero for velocity noise | `harbour.rows`, `harbour.rows_printed` |
| Per-sample problem (8) in 3 inputs | one active row: closed form, exact (rank-one shrink of P along the row — Frobenius-minimal — plus the ℓ₁-cheapest mean shift, on the largest coefficient; the std form has the closed-form optimum s* = clip(slack/z, 0, ‖w‖)); m ≥ 2 active rows: a near-optimal heuristic — candidate target stds per row, the factor update Δ = Aᵀ(AAᵀ)⁻¹Vᵀ (Frobenius-minimal among row-wise shrinks, not the global minimiser), and the exact ℓ₁ mean shift over all rows by LP vertex enumeration. Always feasible; V0 measures the objective gap to the exact SOCP on real instances, split by the number of active rows (one row: ~10⁻⁷; two rows: median ≈ 8 %, p90 ≈ 40 % more conservative) | `vessel/solver_nd.py` |
| Controllers | `VesselMPPI`, `VesselSCBFMPPI` (modes `hocbf`, `printed_rd1`; forms `std`, `variance`; optional IS correction), `VesselDetMPPI` (L-CSS 2025 Alg. 1) — same weight and update equations as the corridor code | `vessel/controllers.py` |
| Cost | q = ‖p − p_g‖² + 2000·max(0, \|u\| − 2.5)² + 20000·1[min_j h_j < 0], φ = ‖p_T − p_g‖²; K = 500, T = 15 (15 s, the group's NMPC horizon), λ = 300 m², Σ₀ = diag(350², 350², 120²) N² (MPPI reaches the goal in every seed at about 2 m/s, collision rate ≈ 0.02 under the default disturbance) | `vessel/controllers.py` |
| Disturbance | environmental force with stationary std s_F = (120 N, 300 N, 400 N m) and correlation time τ_c = 5 s, three ways with equal spectral density at zero: `white` (intensity s_F√(2τ_c) — the paper's model), `ou` (Ornstein–Uhlenbeck gust), `ou` + 0.5 m/s cross-current unknown to the controller | `vessel/simulate.py` |
| Metrics | collision rate = fraction of 0.25 s integration steps inside a circle (`collision_rate_ctrl`: at the 1 s control instants only), runs touching, time to goal (150 s = not reached), min h over the integration steps, ESS (mean, median, and mean inside the obstacle field x < 100 m), activation (any row), Var kept along the most violated row, delivered Pr(all rows a·u ≥ b) on active pairs after clipping, fraction of draws clipped by the thrust limit, fraction with ≥ 2 active rows | `vessel/simulate.py` |

| | Question | File |
|---|---|---|
| V0 | Solver fidelity on 300 real per-sample instances vs the exact SOCP; SDP time per problem | `vessel_V0_solver.json` |
| V1 | As printed (relative degree 1): identical to MPPI? | `vessel_V1_printed.json`, `fig_V1_printed.png` |
| V2 | Harbour transit, white noise: MPPI K = 500 / 2000, SCBF (2nd-order), + IS, variance form, deterministic MPPI | `vessel_V2_harbour.json`, `fig_V2_*.png` |
| V3 | Disturbance model: white vs 5 s gust vs gust + unknown current | `vessel_V3_disturbance.json`, `fig_V3_disturbance.png` |
| V4 | Crossing ferry | `vessel_V4_crossing.json`, `fig_V4_crossing*.png` |
| V5 | Wall-clock per cycle | `vessel_V5_wallclock.json`, `fig_V5_wallclock.png` |
| V6 | Barrier gains α₁, α₂ | `vessel_V6_gains.json`, `fig_V6_gains.png` |
| V7 | Confidence level δ | `vessel_V7_delta.json`, `fig_V7_delta.png` |
| V8 | Deterministic MPPI (the group's method): λ₀, ν and schedule sweep | `vessel_V8_det_sweep.json`, `fig_V8_det.png` |

Shipped results: 30 seeds (V2, V3, V4), 20 seeds (V6, V7, V8), 10 (V1); `results/vessel_summary.txt` lists every number.

Two things to read with the numbers. In the obstacle field the effective sample size of every sampling controller is small (MPPI ≈ 5 of 500; the 20,000 m² collision penalty with λ = 300 m² makes colliding samples negligible and the remaining weights are still peaked) — this is the weight degeneracy in clutter the paper starts from, and it means the controllers here are close to best-of-K selection; raising λ flattens the weights but lets colliding samples into the average. And the importance-sampling correction's density uses log|det P| of the actual per-sample factor (slogdet), which is exact for the rank-one and for the multi-row update alike; a collapsed direction (singular P) is floored at det P₀·10⁻⁶, which gives such a sample a negligible weight — a convention inherited from the corridor code.

An independent line-by-line review of this package (model port, barrier algebra, solver, controllers, disturbance models, metrics, experiment configurations) was carried out before the final runs; every finding was fixed and the experiments re-run with the corrected code.
`python -m scbf_mppi.vessel.experiments --exp all --runs 30` reproduces them exactly on the same platform.

### Live demo

`live_demo/safety_bench.html` is a single-file browser page (no installation) with both scenes and all controllers. Modes: a
single run, or a *race* of two to four controllers on the same seed — every controller's plant is driven by its own copy of the same
random stream, so the disturbance realisation is identical (common random numbers) and a live leaderboard ranks them by collision
steps, then time to goal. Sliders: K, noise, 1−δ, seed, playback speed. Drawn every cycle: 60 of the K sampled rollouts, opacity by
weight, amber on the segments where the barrier chance constraint binds for the proposal; on the vessel the front ψ₁ = dh/dt + α₁h = 0 of the
second-order barrier at the boat's current velocity (d(θ) = R − |v_rel| cos(θ − θ_v)/α₁, clipped at R — inside it the barrier row
forces recovery); the wake; the last four finished runs as ghosts; a replay scrubber over the recorded frames (fan snapshots with
their binding flags, states and counters, so the stats and the leaderboard follow the scrubber); a sample inspector (hover a rollout:
its weight and the steps on which the constraint bound); pause on collision; a 20-seed batch.

**Added 13 September.** (1) *Unknown current and gust*: the disturbance model (white / OU gust / none) and a constant current
(speed, eight directions) that enters the plant kinematics only — the controller's rollouts and barrier rows never see it (experiment
V3); the water drifts on screen, the gust force is drawn on the hull, and the dotted front shows the true ψ₁ = 0 locus next to the
dashed believed one. (2) *Inside this cycle*: for the row the filter works on at t = 0 of each plan — all K samples share x₀ and the
nominal ū there — the distribution of a·u along the row's normal, drawn in units of the nominal spread σ₀ = √(aᵀΣ₀a) on one
fixed, gently eased axis (so the picture glides between cycles instead of re-scaling): b at zero, the nominal proposal a unit
Gaussian at (a·ū − b)/σ₀, the filtered one N((a·(ū+μ) − b)/σ₀, |Pᵀa|²/σ₀²) solid (a point mass when the solver collapsed the row —
it then sits exactly on b), the mass left of b, the two margins as ticks (z·σ corrected at the constant z; α·σ² as printed at α·σ₀ —
α = z, so their ratio is σ₀ in the row's units, ≈ 0.1 on the vessel and 1.2–1.6 on the corridor), the delivered probabilities before
input clipping, a status chip, and a strip with the share of samples on which a row binds at each horizon step. The shown row is
kept with hysteresis (it switches only when another row is tighter by more than 0.75 σ₀), and the parameters ease over the first
third of each control interval; the thrust needles and the new rollout fan ease the same way. The probabilities were checked against Monte Carlo through the recorded
(μ, P) (40 000 draws, agreement to 0.002). The shown row is the one the solver acted on (the most violated by nominal margin), else the most violated by nominal
z-score; for deterministic MPPI it is recorded from the first of its four iterations. (3) *Thrust gauges*: the applied azimuth vector
inside the 700 N disk, the raw (unclipped) request when it exceeded the disk, the bow thruster against its cap F_BT ≤ 244·e^(−0.62u²),
and the share of draws clipped this cycle; the dial is drawn bow-up with the model's +y sway to the left (the map is an east–north
plane, a mirror of the NED convention the symmetric hull permits, so the page does not say port or starboard). (4) *Camera follow*
at 2× with the hull at true scale. (5) *Story mode*: a scripted run (crossing scene, seed 26, MPPI / SCBF-MPPI / + IS / det. MPPI, white
noise) with captions — at t ≈ 39 s the deterministic MPPI touches the barge circle, the run pauses, rewinds five seconds with its heaviest rollout
highlighted and continues, six seconds later MPPI touches it as well, the two barrier variants stay clean; the end card states the 30-seed context of that scene (MPPI touches in 18, SCBF-MPPI in 8, + IS in 2, det. MPPI in 23;
`story_scan.json`, a scan of this core in Node — a browser's own sin/cos/pow differ in the last bit, so a single seed can end differently there; seed 26 was chosen because its outcome survives 10⁻⁹ perturbations of the initial state in six replicas). Two more animations from the Python package: `python -m scbf_mppi.vessel.animate_current`
(V3 seed 0, gust + 0.5 m/s current, three boats; the run reproduces the stored V3 metrics exactly) and
`python -m scbf_mppi.vessel.animate_rowspace` (map beside the row picture, SCBF-MPPI on the crossing scene, seed 2). None of the
additions changed a trajectory: `node xval_same.js` (against `sim_v3_backup.js`, the core before them) reproduces every path bit for bit.

**Solver port and a background thread (13 September, later).** The vessel solver of the browser core (`solveRows` in `sim.js`) is now a
port of `solver_nd.solve_rows_nd`: one active row exact (std form: shrink to s* = clip(slack/z, 0, s_nom), then the l1-cheapest mean
shift; variance form: the same 65-point grid), and — for two or more rows active at the nominal proposal, or a row violated by the
single-row solution — the package's joint candidate solve (Frobenius-minimal factor update Δ = Aᵀ(AAᵀ)⁺Vᵀ over the candidate target
stds {0, f*, 1, f*/2} per row with the local refinement, exact l1-minimal mean shift over ALL rows by LP vertex enumeration). On
3 200 random instances at the vessel's scale (both forms, 1–4 rows, 2 329 with several active rows; `live_demo/solver_xval/`) it
agrees with the Python solver to 2·10⁻⁷ relative in μ and 10⁻⁹ in P with identical active / multi / infeasible flags, and every
row is satisfied after the solve (worst relative margin −4·10⁻¹³) — the residual violations the earlier sequential sweeps left on
about a third of the multi-row samples are gone, which is what the row view of the crossing scene used to show. MPPI, deterministic
MPPI, the printed rows and the whole corridor are bit-identical to the previous core (`xval_solver.log`, against `sim_v5_backup.js`);
the SCBF trajectories changed, and the 30-seed comparison below and `story_scan.json` were re-run. The page now runs the simulation
in a Web Worker (falling back to the main thread when the page may not create one), one step ahead of the drawing, so the animation
never stalls while 500 rollouts are planned (frame times while racing four controllers: 23 ms mean / 50 ms worst instead of 75 / 217);
the boats are interpolated through the 0.25 s RK4 sub-steps, and the fan, the fronts, the row view and the gauges ease between cycles.
The Step button queues one step per click; the batch runs in the same thread and no longer freezes the tab.

The three diagnostics are defined for every controller. The barrier rows are evaluated on every (sample, timestep) pair of every
controller; *constraint active* is the share of pairs on which the nominal proposal (μ = 0, Σ = Σ₀) violates the chance constraint —
for the SCBF variants that is where problem (8) modified the draw, for MPPI and deterministic MPPI it is where it *would* have
(the corrected, standard-deviation form at the chosen 1−δ is used for this test); *Var kept* is the variance left along the active
row, averaged over all pairs (100 % where the constraint is inactive or nothing filters); *delivered* is the fraction of active pairs
on which the actually drawn control satisfies every row — filtered draws for SCBF, unfiltered draws for the others. Deterministic
MPPI reports these and its ESS averaged over its four inner iterations (125 samples each), as the Python package does for the ESS.
For the printed relative-degree-1 rows (a = 0) the class-K gain is Φ⁻¹(1−δ), the V1 setting, so their "binds" share moves with the
chance level while the trajectory stays MPPI's; their *Var kept* reads 100 % in the browser — nothing can shrink along a zero row —
where the package's ratio s/s_nom degenerates to 0/0 and is reported as 0 (a deliberate difference in a diagnostic, not in a trajectory). Adding the diagnostics changed no trajectory: every path the previous core produced
is reproduced bit for bit (`node xval_same.js`, 12 vessel and 10 corridor episodes, `xval_same.log`, `xval_same_v5.log`). The browser batch now counts
collisions and min h at every 0.25 s RK4 sub-step, as the Python package and the live view do.

Its JavaScript core (`live_demo/sim.js`) is a port of this package: on the corridor, 30 seeds in Node give MPPI 0.041 ± 0.011 / 122 steps
(would bind 0.95, delivered 0.480 — Python 0.956 ± 0.015 / 0.485 ± 0.060), SCBF as printed 0.032 ± 0.011 / 183
(activation 0.80, Var kept 0.35, delivered 0.733), corrected 0.065 / 230 (Var kept 0.07, reached 47 %), + IS 0.004 / 117
(reached 100 %), deterministic MPPI 0.054 / 152 — the Python values are 0.046 / 130, 0.045 / 187 (0.81, 0.34, 0.734),
collapse (0.08, 40 %), 0.016 / 114 (100 %), 0.058 / 134 (`xval_corridor_v3.log`). On the moored-obstacle harbour, 30 seeds in Node against
30 in Python (`xval_vessel_v3_30.log`, `results/vessel_V2_harbour.json`): MPPI 0.024 ± 0.018 / 79 s, 33 % of runs touching, ESS 134
vs 0.033 [0.020, 0.047] / 83 s, 57 %, ESS 133; SCBF-MPPI 0.012 ± 0.009 / 97 s, reached 97 %, activation 0.61, Var kept 0.47, delivered 0.906,
ESS 114 vs 0.009 [0.002, 0.018] / 106 s, 0.56, 0.54, 0.909, ESS 129; + IS 0.001 ± 0.002 / 95 s, reached 100 %, delivered 0.952 vs
0.000 / 106 s, 0.956 (`xval_vessel_v6_30.log`, with the ported solver; before it the delivered probabilities were 0.889 / 0.954). Collision rates, activation, variance kept and delivered probabilities agree within the intervals; the
JavaScript SCBF variants reach the goal about 9 % sooner (97 ± 7 s against 106 s), the two cores drawing different random streams. With a
deterministic plant both cores reach the goal in 55–56 s on every seed. Every number reported in this README
comes from the Python package, not from the browser core.
## The simulation videos

Thirteen clips are in `figures/`. Every one is drawn from a logged run of this package; none is an
illustration. The 2-D clips are written by the animation scripts directly. The 3-D clips draw the same
stored state logs through the offline viewer, which adds no physics: the model is planar, so there is no
roll, pitch or heave in it and none is shown.

| file | what it shows | written by |
|---|---|---|
| `anim_mppi_vs_scbf.mp4` | the paper's corridor, MPPI against SCBF-MPPI | `python -m scbf_mppi.animate` |
| `anim_corridor_three.mp4` | the corridor with the printed and the corrected form side by side | `python -m scbf_mppi.animate --three` |
| `anim_vessel_mppi_vs_scbf.mp4` | the vessel meeting the crossing ferry | `python -m scbf_mppi.vessel.animate` |
| `anim_vessel_current.mp4` | three controllers under a gust and an unmodelled 0.5 m/s current | `python -m scbf_mppi.vessel.animate_current` |
| `anim_vessel_current_ghosts.mp4` | the same run with the rejected candidate paths drawn | `python -m scbf_mppi.vessel.animate_current --ghosts` |
| `anim_vessel_rowspace.mp4` | one control cycle opened up: the barrier rows and the chosen input | `python -m scbf_mppi.vessel.animate_rowspace` |
| `anim3d_corridor_orbit.mp4` | the corridor run in three dimensions | `live_demo/3d/capture_mp4.py` |
| `anim3d_ferry_chase.mp4` | the ferry crossing, camera behind the course over ground | `live_demo/3d/capture_mp4.py` |
| `anim3d_current_chase.mp4` | the current scene, same camera | `live_demo/3d/capture_mp4.py` |
| `anim3d_current_orbit.mp4` | the current scene, orbiting camera | `live_demo/3d/capture_mp4.py` |
| `anim3d_pathtraced_ferry_chase.mp4` | the ferry crossing, path traced at 1920×1080 | `live_demo/3d/blender_render.py` |
| `anim3d_pathtraced_current_chase.mp4` | the current scene, path traced | `live_demo/3d/blender_render.py` |
| `anim3d_pathtraced_current_orbit.mp4` | the current scene from orbit, path traced | `live_demo/3d/blender_render.py` |

`live_demo/3d/README_3D.md` gives the exact command behind each 3-D clip, including the frame ranges.

**One thing to notice before anyone points it out.** In several clips the boats travel astern. That is a
property of the abstraction, not a bug and not the barrier: the input is an isotropic 700 N force, the
damping is symmetric in surge, and the running cost penalises the size of the speed rather than its sign, so
nothing in the problem prefers forwards. `scbf_mppi/vessel/ext/costs.py` adds a heading term and experiment
V9 re-runs everything with it; `results/vessel_V9_heading.json` holds the outcome, and it changes one
conclusion in the crossing-ferry scene. The planar model is *not* exactly invariant under reversing the
boat: the worst mismatch over random states is 0.449 m/s², and it is exact only when the boat is not
turning. `test_why_astern` in `scbf_mppi/vessel/ext/selftest_ext.py` measures both statements.

Two render passes that were superseded are not in the repository: an earlier 1600×900 path-traced pair, and
a larger encode of `anim3d_pathtraced_current_orbit.mp4` at the same resolution. The GIF beside each 2-D clip
is not tracked either, being a lower-resolution copy of the mp4. All of them regenerate from the commands above.

## Reading the results honestly

Everything here is a reimplementation from the paper's text: where the paper is silent, the assumption is listed above and can be
changed on the command line. Two things are therefore *findings about the paper as written*, not about the authors' code:
that σ = I is incompatible with Table I, and that the constraint as printed delivers a lower probability than it claims.
Everything else — the wall-clock cost, the activation fraction, the collapse of the corrected constraint, the mechanism
separation — is a property of the method as specified, in the regime where its own baseline numbers reproduce.

## Beyond the reproduction: four questions the critique opens (`scbf_mppi/vessel/ext/`)

The reproduction ends with a set of objections. These four experiments turn the sharpest of them into
measurements. They are additions, not corrections: no shipped result file is touched, and
`tests/test_regression.py` re-runs twelve shipped configurations and requires bit-identical trajectories
after every change here.

```
python -m scbf_mppi.vessel.ext.selftest_ext                              # ~2 min, 21 checks
python -m scbf_mppi.vessel.ext.experiments_ext --exp all --runs 30       # writes results/vessel_V9..V12*.json
python -m scbf_mppi.vessel.ext.figures_ext                               # writes figures/fig_V9..V12*.png
```

**V9 — the heading term the cost does not have.** The vessel running cost is distance to the goal plus a
speed penalty on `|u|`, so nothing prefers bow-first motion, and every controller spends much of a run going
astern on an isotropic 700 N force disk. That is a property of the force-input abstraction rather than of the
barrier, and it affects all three controllers, but it invites the question whether the comparison survives a
cost that does prefer going forwards. V9 adds an optional heading term and an optional astern penalty and
re-runs the comparison in both scenarios, over a weight sweep.

**V10 — the barrier in the sampler or on the output.** The paper pushes the barrier into the sampling
distribution, at K × T = 7500 constrained solves per control cycle. The standard alternative is one convex
solve per cycle that projects the controller's output onto the same barrier condition. V10 runs both, plus a
chance-constrained filter whose tightening is derived exactly for a force disturbance, with the same MPPI, the
same barrier gains, the same seeds and the real input set (the 700 N azimuth cone and the speed-dependent bow
cap). It also shows what the paper's white-noise assumption costs: the tightening it demands is √(2τ_c/h) —
here a factor 3.2 — larger than the one a correlated gust of the same intensity demands.

**V11 — the disturbance the chance constraint never sees.** In the unknown-current scenario the controllers
are blind to a 0.5 m/s drift that enters the position kinematics, so their barrier derivative is wrong by
n · c, up to 0.5 m/s — the size of the whole α₁h term at h = 5 m. A nominal δ = 0.003 says nothing about
that: δ bounds the modelled noise, not an unmodelled drift. V11 adds a kinematic-residual observer (which
needs no vessel model, only the measured state), injects the estimate into the rollouts and into the barrier
rows separately, and audits at the plant how often the controller's own certificate was false.

**V12 — the effective sample size the correction destroys.** With the importance-sampling weights the paper
omits, the median effective sample size is 13 of 500. V12 retunes the temperature on line to hold the ESS at
a target, exploiting `log w(λ) = −J/λ − C + log q` so one rollout batch gives the whole family of weights.
It reports ESS together with the norm of the control update and the fraction of runs that reach the goal,
because ESS bought by flattening the weights is not a rescue.

## Reproducing it

```
python reproduce.py            # environment, self-tests, bit-exactness guard   (~3 min)
python reproduce.py --full     # the above, then every experiment and figure    (~90 min)
python tests/drift_report.py   # how far the numbers moved when the versions moved
```

`REPRODUCIBILITY.md` states precisely what reproduces and what does not, and which reported numbers
are fragile across library versions. `requirements.txt` is pinned; `results_shipped_2026_09_12/` keeps the
result files the reported numbers were taken from, so the claim that the text matches the code stays checkable.
