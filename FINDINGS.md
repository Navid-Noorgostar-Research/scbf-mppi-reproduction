# What this reproduction found, and how far each finding goes

Everything below is measured by code in this repository. Where a claim is standard mathematics rather
than a new result, it says so. Where a claim was made and later withdrawn, it says that too, because the
withdrawals are the part most likely to matter to anyone building on this.

---

## 1. Theorem 2 uses a variance where its own proof needs a standard deviation

The paper's condition, equation (6), is

```
a'mu - z (a' Sigma a)  >=  b
```

but the proof reaches it by taking "the upper bound of the confidence interval for the Gaussian
variable", and a Gaussian confidence interval is built on the standard deviation:

```
a'mu - z sqrt(a' Sigma a)  >=  b ,      z = Phi^-1(1 - delta)
```

The consequence is not notational. Take `delta = 0.003`, `a = 1`, `b = 0`, a Gaussian input with
standard deviation `0.5` and mean `z sigma^2 = 0.686945`:

| | |
|---|---|
| paper's condition | satisfied, with equality |
| correct condition | violated by 0.687 |
| claimed violation probability | 0.3 % |
| **actual violation probability** | **8.47 %** |

Reproduce with `python -m scbf_mppi.selftest` (check 3, corridor: printed form delivers 0.916 where
0.997 is claimed) and `python -m scbf_mppi.vessel.selftest` (check 6, vessel row: 0.952 against 0.997).

## 2. Correcting it makes the paper's own problem return a singular covariance

For one active row, with the objective of (8) and the corrected constraint, the cost as a function of the
retained row standard deviation `s` has slope `z/||a||_inf - 1/||a||_2` on the binding branch. That is
positive exactly when

```
z  >  ||a||_inf / ||a||_2
```

and since `||a||_inf / ||a||_2 <= 1` always, every `delta <= 0.1587` satisfies it. The optimum is then
`s* = max(0, slack/z)`, which is **zero whenever the nominal mean already violates the row**. The optimal
proposal is a degenerate Gaussian, mutually singular with the base proposal, so:

* the free-energy identity MPPI is derived from does not hold there, since it needs absolute continuity;
* the importance weight `dp/dq` does not exist, so the standard repair is unavailable rather than costly;
* any implementation must invent a regulariser, and that constant then decides the closed-loop result.

Verified against the exact conic solve: the switch sits at the predicted threshold. Measured frequency,
corrected form: 13.0 % of samples degenerate at `t = 0` in the corridor and 0.0 % with the printed form;
on the vessel 6-9 % of the 7500 per-cycle solves are numerically rank deficient. The discarded samples are
the ones approaching the obstacles.

**Assumptions that must be stated with it:** single active row, unconstrained mean, and a Frobenius or
spectral norm on the covariance factor. Non-singular feasible proposals still exist if the mean moves
further into the safe region; the obstruction is the objective's preferred solution, not feasibility.

A four-way prior-art search later rated this a **composition of known parts** rather than a new result:
the hypothesis it shows Algorithm 1 violating is written into MPPI's own covariance-shaping theorem
(Williams, Aldrich, Theodorou, arXiv:1509.01149, Theorem 1, "under the condition that each `A_{t_i}` is
invertible"), and the loss of the optimality guarantee is conceded by Tao et al. themselves after (11).
What is new is only *where the optimum lies*, and that `s* = 0` is therefore the generic case in the one
regime where the filter acts. Section 7 states that position with its citations.


**The cheapest safe Gaussian moves the other way.** Ask instead for the proposal that satisfies the same
corrected row while minimising the exact weight second moment
`E_q[w^2] = beta^2/sqrt(2 beta^2 - 1) exp(m^2/(sigma_0^2 (2 beta^2 - 1)))` with `s = beta sigma_0`. Solved
numerically at `delta = 0.003`, that optimum **inflates** the covariance:

| nominal slack | `s*` from (8) | `beta*` of the divergence-optimal safe proposal |
|---|---|---|
| +0.50 | 0.182 | 1.63 |
| 0.00 | 0 | 2.28 |
| −0.50 | 0 | 3.17 |
| −1.00 | 0 | 4.28 |

The reason is in the exponent: the mean-shift cost is `m^2/(sigma_0^2(2 beta^2 - 1))`, so a **wider**
covariance makes a given shift cheaper, and it is worth buying width to pay less for the shift. Exactly
where (8) drives the spread to zero, the sample-efficient choice widens it two- to four-fold. The two
objectives are not merely different, they point in opposite directions — which is a compact way to say why
(8)'s objective is the wrong one if the weights are ever going to be used.

Neither half of this is new mathematics: the closed form is the Renyi-2 divergence between Gaussians
(Liese and Vajda 1987; van Erven and Harremoes 2014), and the comparison is a numerical minimisation, not
a theorem. It is an observation about the paper's own program.

**What it chooses in flight, which is the question the table does not answer.** Those four rows are
evaluated at four chosen slack values. Solving *both* programs at every state Algorithm 1 actually
reaches — same rows, same nominal plan, same `delta`, 64 seeds and 160 million sample-timesteps — the
barrier is active 95.7 % of the time, and the divergence-optimal proposal then wants a mean `beta` of
**1.15**, against the **0.12** mean `s/sigma_0` that (8) returns. Only 8.4 % of active samples want
`beta > 1.5`, and 1.8 % want `beta > 2`. So the two- to four-fold widening is real at tight slack and
largely absent in flight: what separates the two programs on a real trajectory is the collapse, not the
inflation, and the contrast above is a statement about particular states rather than a design rule.
`python tests/renyi_safe_gaussian.py --closed-loop` reproduces both halves — and the table itself, which
until now had no solver behind it.

## 2b. Stronger than the collapse: at the corridor's own start state, no Gaussian works at all

Section 2 says the *optimiser* collapses the covariance. A sharper statement holds at the default start
state, and it does not depend on the optimiser, the objective, the mean, or correlations with the other
control channel.

The corridor's two walls constrain the same forward-speed variable from opposite sides. Meeting both
chance constraints requires `l + z sigma_q <= mu_q <= r - z sigma_q`, so a Gaussian exists only if
`sigma_q <= (r - l) / (2z)`. Finite nominal importance-weight variance requires `sigma_q > sigma_0/sqrt(2)`.
A suitable Gaussian therefore exists **exactly when**

```
r - l  >  sqrt(2) z sigma_0
```

At `x = 0, y = 0.5, theta = 0` with `sigma_0 = 1` and `delta = 0.003`, this repository's own `scbf_rows`
gives the admissible speed interval `[-1/pi, 1/pi]`, and:

| | |
|---|---|
| largest `sigma_q` allowed by both walls | 0.115843 |
| smallest `sigma_q` giving finite weight variance | 0.707107 |
| existence condition | 0.636620 > 3.885950 — **fails, by a factor of six** |

So at that state **every** non-degenerate Gaussian first-speed proposal meeting both corrected wall
constraints has infinite exact importance-weight variance. Extending to the full cost-weighted target,
divergence follows whenever `sigma_q^2 < (2/sigma_0^2 + 4(T+1)dt^2/lambda)^-1`, which at `T = 20`,
`dt = 0.05`, `lambda = 1` is 0.452489 — and every chance-feasible Gaussian, capped at 0.013420, is
inside it. The MPPI cost does not rescue the example.

Both halves of this are published. The two-sided Gaussian cap is Lubin, Bienstock and Vielma,
arXiv:1507.01995, Lemma 16 (63)-(65) — the per-row pair used here is exactly their axis-aligned
approximation — and the finite-variance floor for a Gaussian-to-Gaussian weight is textbook (Owen ch. 9).
What is not in either is that in this method they bind the **same object**, because the covariance the
barrier rows shrink *is* the importance-sampling proposal covariance. Section 7 has the full position.

Every figure in this section was checked against `Corridor.scbf_rows` in this repository and reproduces
exactly. The construction and the full-target extension are due to an external review of commit
`33497d1`, not to this repository's own work; they are recorded here because they are verified and
because they subsume section 2's statement. Infinite variance does not imply that any particular finite
run fails, and singular proposals escape it only by losing the support that exact correction needs.


**The obstruction belongs to the Gaussian family, and the escape is not free.** A defensive mixture
`q = w p + (1 - w) q1` satisfies `q >= w p` pointwise, so `dp/dq <= 1/w` and `E_q[(dp/dq)^2] <= 1/w` for
any `w > 0` (Hesterberg 1995) — the infinite variance is a property of insisting on a *single* Gaussian,
not of safe sampling as such. But the chance constraint caps the defensive weight: the base component alone
contributes `w * delta0` of violation probability, so `w <= delta/delta0`. At this state the nominal
violation probability is `delta0 = 0.750250` — which is exactly the two tail masses `0.375125` above — so
`w <= 0.003999` and the second moment is bounded above by `250.08`. Section 3's floor bounds it below by
`187.69`. The mixture therefore turns an infinity into a number between **188 and 250**: it escapes the
obstruction and pays section 3's price instead, which is the same thing said twice. Asymptotically that is
`ESS/K <= 1/187.69`, or 2.66 of 500.

**The escape buys support, not accuracy.** That bound prices the mixture; it does not say whether the
mixture then estimates anything correctly. Answering that needs a target whose value is known, which the
corridor's is not — the probes at this state pooled an effective size of `1.17` out of two million draws
— so it is measured on a constructed scalar problem whose target is exactly Gaussian. A mixture of
*path* measures is not a per-step mixture of kernels: along a prefix that has already violated, the
posterior over sources reverts to the nominal component, so the mixture meets the *marginal* per-step cap
while breaking the *conditional* one. At an admissible weight the missing class duly reappears — in
**54 %** of batches, against **0.15 %** without it — and the estimate does not move: `0.060` against a
true `0.494`, with the RMSE **worse**, `0.85` against `0.55`, because the rare nominal draws arrive
carrying enormous weight. The mean comes back only near `w = 0.10`, whose per-row violation rate is about
ten times the cap. `python tests/mixture_recovery.py`.

## 3. A sample-efficiency bound, with its limits

For any law `q` satisfying the per-sample constraint, with `delta0` the violation probability under the
reference law:

```
chi2(p||q)  >=  (delta0 - delta)_+^2 / (delta (1 - delta))
ESS/K       ->  1 / (1 + chi2)
```

The positive part is required: without it, `delta < delta0` fails, since `q = p` then satisfies the
constraint with zero divergence. The bound is sharp, attained by keeping the reference conditionals and
moving the mass.

**Three limits, all of which must be stated.**

*It is asymptotic.* `ESS/K -> 1/(1+chi2)` is a population efficiency, not a cap on a finite run. For the
equality-attaining proposal at `K = 500` and `delta = 0.003`, no sample lands in the violation set in
`(1 - delta)^K = 0.997^500 = 22.3 %` of runs, and the measured ESS is then exactly 500. (An earlier
version of this note reported 21.5 % from a finite simulation where the closed form was available.)

Two further conditions on reading it, both from an external review and both correct. `delta0 = 0.5` requires
the nominal projected mean to sit exactly on the boundary, `a.mu_0 = b`; a physical state on the barrier does
not imply that. And the 70.7 % figure is a geometric fraction of a specified activation band under a scalar
or single-direction contraction, not a universal fraction of Gaussian proposals or of observed timesteps.

*The reference is the target, not the nominal law.* MPPI estimates under `pi` proportional to
`exp(-S/lambda) p`, so the governing divergence is `chi2(pi||q)` and `delta0` must be replaced by
`pi(A)`. If the running cost already suppresses the unsafe set below `delta`, the constraint forces no
divergence at all. So this bounds recovery of the nominal law, and constrains the MPPI estimator only
when `pi(A) > delta`. It is not an impossibility result for every safe MPPI variant.

*The ingredients are classical, and so is the inequality.* The bound is Hammersley-Chapman-Robbins with
the test function taken to be the indicator of the violation set: Polyanskiy and Wu, *Information Theory:
From Coding to Learning*, give the variational form of chi-square at (7.73) and the Bernoulli value at
Prop. 7.2, and set the event rearrangement as a reader **exercise**. `ESS/K -> 1/(1+chi2)` is Kong 1992,
with rigorous versions in Agapiou, Papaspiliopoulos, Sanz-Alonso, Stuart, *Statistical Science* 2017 and
Chatterjee and Diaconis, *Ann. Appl. Prob.* 2018. The Gaussian tail condition is textbook, Owen ch. 9.
Nothing mathematical here is new; what is assembled is the reading of it as a price every
constraint-satisfying proposal in sampling-based MPC pays.

*That same 22.3 % is a guarantee, not a ceiling.* `(1 - delta)^K` lower-bounds the probability that
**every** sample is safe, and since a convex combination of inputs each satisfying the same linear row also
satisfies it, it lower-bounds the probability that the executed average is safe. It does not bound the
executed input's actual safety probability from above: the average can satisfy the row while individual
samples violate it, which is what this repository measures — the averaged input never violated the row in
666 active instances. Tightening per sample by the union bound, `delta/K`, is sufficient for all-samples-safe
at level `1 - delta` and needs `z' = 4.378` at these settings; it is not necessary for safe execution.

The Gaussian corollary, `E[w^2] < infinity` iff `s > s0/sqrt(2)`, is the **scalar** case. In general the
condition is the matrix one, `2 Sigma_q - Sigma_p` positive definite, and an anisotropic shrink can fail
it while its largest ratio still exceeds `1/sqrt(2)`.


**Which effective sample size is reported, everywhere in this repository.** Every ESS figure here — the
tables, the demo panel, `ess_mean` in the result files — is the **full cost-weighted** effective size,
`1/sum_k w_k^2` over the realised weights, which combine the cost softmax `exp(-S/lambda)` with the density
ratio where a controller applies one. It is NOT the density-ratio factor alone. The distinction matters
because the chi-square bound of this section governs only the density-ratio part: the two agree only when
the cost softmax is flat, and on the vessel they are far apart (cost softmax alone 3.3, weights alone 157.4,
both 3.1). So a measured ESS may sit below this section's bound without contradicting it, and the budget of
section 7 targets the density-ratio factor rather than the number the tables print.

## 4. Where the paper's advantage actually comes from

Restoring the omitted weights gives, in the corridor over 512 seeds, the **best** controller tested:
100 % goal-reaching, the fastest time, a median collision rate of zero and a median minimum barrier of
+0.167, at an asymptotic effective sample size of **1.9** out of 500.

That is not path-integral averaging. At an effective size near one the update selects a single sample,
and the dominant term in the log density ratio is `log s`, so the sample it selects is the one the
barrier had to correct least over the whole horizon.

That last step is measured rather than asserted. Only variation *across samples* can select, so split the
per-sample ratio into its three pieces — with `sigma_0 = 1` and `ev = m + s xi`,

```
log p/q  =  sum_t [-0.5 ((m + s xi)/s0)^2 - log s0]  +  sum_t [0.5 xi^2]  +  sum_t [log s]
                 mean shift                                noise              shrink
```

— and compare their standard deviations over the K samples. Replaying the proposal of six cycles of a real
episode (the rebuilt total matches the controller's own `logq` to 1.1e-13, so the split is exact):

| | sd across samples |
|---|---|
| mean-shift term | 2.49 |
| noise term | 3.10 |
| **shrink term, `sum_t log s`** | **44.65** |
| total | 44.88 |

The shrink term is fourteen to eighteen times the spread of either other term, and the total is correlated
with it at **+0.994**. The selection is the shrink.

Charging for that explicitly turns the accident into a dial. Let `I = |m| + (s0 - s)` be the per-sample
problem's own objective value, which the paper computes and discards, and weight by

```
w_k  proportional to  exp( -( S_k + mu lambda sum_t I_k,t ) / lambda )
```

`mu = 0` reproduces the corrected controller to the digit and `mu` large reproduces the accidental rule,
with every value between a valid estimator, because `I` depends on the solve and not on the noise.
Over 512 seeds, `mu = 0.5` gives 99.4 % reached, median collision zero, median minimum barrier +0.133, at
an effective sample size of 69.4 rather than 1.9.

The whole dial, over 512 seeds:

| μ | reached | median collision | median min h | ever unsafe | runaway | ESS |
|---|---|---|---|---|---|---|
| 0 | 35.2 % | 0.0400 | −0.086 | 84.8 % | 16 / 512 | 280.1 |
| 0.1 | 76.6 % | 0.0160 | −0.036 | 62.5 % | 13 / 512 | 254.7 |
| 0.2 | 93.2 % | 0.0000 | +0.048 | 33.8 % | 7 / 512 | 199.9 |
| 0.3 | 97.3 % | 0.0000 | +0.088 | 22.3 % | 9 / 512 | 146.0 |
| 0.5 | 99.4 % | 0.0000 | +0.133 | 16.8 % | 3 / 512 | 69.4 |
| 1 | 98.4 % | 0.0000 | +0.163 | 17.0 % | 8 / 512 | 7.6 |
| 3 | 99.6 % | 0.0000 | +0.152 | 14.8 % | 1 / 512 | 2.0 |

The browser core reproduces it independently at 24 seeds — 280.2, 258.8, 197.8, 143.0, 68.1, 6.8, 2.4
against the 512-seed 280.1, 254.7, 199.9, 146.0, 69.4, 7.6, 2.0 — which is the cross-check that the live
demo shows the same object this table does (`live_demo/xval_intervention.log`).

**This mechanism is not new, and the section should not be read as proposing it.** Charging a safety
filter's own effort as an extra lambda-scaled running cost inside the MPPI exponent is published: Gandhi,
Almubarak, Aoyama, Theodorou, arXiv:2204.05963 (2022), Algorithm 1 accumulates
`S_hat += q(x) + (lambda(1-beta)/2) k_fb' Sigma^-1 k_fb`, with the same tunable multiplier, and Robust
MPPI (RA-L 2021) is the same construction a year earlier. The reason it is a valid estimator is Section
III-B of MPPI's founding paper, "Likelihood Ratio as Additional Running Cost". What is specific here is
only that the scalar charged is the SCBF program's own optimal objective value, and that this filter
re-solves a per-sample covariance as well as a mean, so the exact ratio carries a determinant term and a
cheap deterministic surrogate is attractive for a reason that does not arise in that prior work. Section 7
states the position in full.

**What this does not establish.** It is not demonstrated to be a better controller. Its runaway-episode
rate is 3 of 512, which Fisher's exact test cannot distinguish from zero (p = 0.249), but that rate is
not monotone in `mu` — across `mu` = 0, 0.1, 0.2, 0.3, 0.5, 1, 3 the counts are 16, 13, 7, 9, 3, 8, 1 of 512 — so the
tail is unresolved at this seed count.
The typical-case statistics move smoothly with `mu`; the tail does not, and a safety argument rests on
the tail.

## 4b. How much of a barrier-row violation is an actual collision

Two different events are counted throughout this repository and they are not the same. `sat_active` and
`delivered Pr(a u >= b)` count violations of the barrier **row**, the per-step sufficient condition;
`collision_rate` and `min_h` count the trajectory actually leaving the safe set. Satisfying the row is
enough for safety, and violating it is on its own evidence of nothing — so a sentence like "the sampler
violated the row 8 % of the time" has carried no safety interpretation. The gap had never been measured.

Scoring every rollout of a plain-MPPI episode twice — `M`, the number of horizon timesteps at which the
drawn control violates a row, and whether that rollout's own trajectory ever leaves the safe set — over
64 seeds and 4,033,000 rollouts:

| event | probability |
|---|---|
| the rollout leaves the safe set | 0.2762 |
| ... given `M = 0` | **0.000545** |
| ... given `M >= 1` | **0.2779** |
| ... given `M >= 2` | 0.2821 |
| ... given `M >= 4` | 0.2992 |
| ... given `M >= 8` | 0.3781 |

The certificate is **sound**: of the rollouts that never violate a row, five in ten thousand leave the
safe set. It is not **precise**: a violation is a genuine safety event 27.8 % of the time, so roughly
three quarters of what the barrier refuses would have been fine. And it is graded — more violations do
mean more risk, monotonically — so it is informative rather than arbitrary. That 72 % is the price of a
sufficient condition, and it is what Algorithm 1's reshaped proposal is paying for. Plain MPPI is scored
because under Algorithm 1 the row holds by construction, and with the covariance collapsed it holds
deterministically, leaving almost nothing to score; the question is about the certificate, not about a
controller. `python tests/certificate_conservatism.py`.

## 5. Claims made during this work and withdrawn

* **"The relative-degree collapse is a gap in the paper."** False. Section III-D opens by stating that the
  constraint may fail when `dh/dx g(x) = 0`, and Theorem 3 is its proposed remedy. Experiment V1 verifies
  the paper's own caveat.
* **"The importance-sampling correction is what collapses the effective sample size on the vessel."**
  False there. Decomposed per cycle: cost softmax alone 3.3, importance weights alone 157.4, both 3.1.
  On the vessel the temperature does it. The corridor behaves the other way and the two must not be mixed.
* **"MPPI's cost weighting is adversarially correlated with constraint violation, and that is the transfer
  failure."** Real but small: correlation +0.11 to +0.21, costing 0.24 of 2.75 standard deviations of
  margin, with the averaged input never violating the row in 666 active instances.
* **"Repairing the estimator will improve the controller."** False, and the opposite of what happens.

## 5b. A bug in this repository, found by an external review

`vessel/solver_nd.py` reported the retained standard deviation along a barrier row as `||P a||` where the
sampling covariance `P P'` requires `||P' a||`. `einsum("kij,ki->kj", M, a)` contracts M's FIRST index and
so returns `M' a`; passing `transpose(Pfac)` therefore returned `Pfac a`. The two agree only when the
factor is symmetric.

Severity, measured rather than assumed. The factor is non-symmetric on 92.9 % of multi-row samples, and
the reported value then differs from the true standard deviation by a median factor of 12. But the
affected expression is reached only in the joint multi-row branch, because `_solve_one_row` is called once
per sample and always receives the diagonal initial factor. So the error is confined to the
retained-variance **diagnostic**. Confirmed by the regression guard: after the fix, 7 of 12 shipped cases
changed and every one of them changed only in `var_ratio_mean`, with all trajectory, control and barrier
digests identical. `tests/regression_ref.json` was recaptured to record the corrected diagnostic; the
stored `results/*.json` still carry the old value for that one field.

An intermediate claim of mine, that the error also sat on the sampling path, was wrong. It came from
exercising `_solve_one_row` with a non-diagonal factor, which the real call path never does.

## 5c. A second defect in this repository's own harness, found because three numbers disagreed

The same quantity — plain MPPI's episode-mean effective sample size in the corridor at K = 500 — was
reported three times and three ways:

| source | seeds | value |
|---|---|---|
| this repository's shipped `E1_table1.json` (CPU) | 30 | 159.91 |
| an external review's independent prototype | 30 | 170.62 |
| this work's GPU closed-loop study | 512 | **269.62** |

The GPU was the odd one out, and it was wrong. `tests/gpu_closed_loop.py` freezes a finished seed's
**state** but keeps calling the planner for it on every remaining outer iteration, and divided by a global
cycle counter rather than a per-seed one. `simulate.run_episode` breaks at the goal. Measured on the CPU,
by flying eight episodes and then continuing to plan at the parked state:

```
mean ESS over the cycles actually flown      179.93
mean ESS over the cycles parked at the goal  381.06       <- the cost landscape there is flat
the GPU's statistic, over 250 cycles         263.21       <- 1.46x inflation
```

Those eight seeds are a demonstration of the mechanism, not the reported figure; on the 512 seeds the
column actually shipped with, plain MPPI's 269.62 becomes 160.1, an inflation of **1.68x**.

Worse than a constant factor: how much a controller idles, and whether its filter still acts once parked,
both differ by controller, so the column was inflated by **different** factors per row and could not be
read across rows. Plain MPPI was inflated 1.68x; the two controllers whose weights had already collapsed
were barely touched, because at the corridor's goal the wall is at its steepest --
`w'(4) = pi/2`, so `|a| = 1.5708` at zero heading -- and BOTH chance constraints are violated there even
at zero input, by 4.316 against a bound of -0.5, so the filter is fully active although the state sits in
the middle of the corridor with half a metre of clearance on each side.

Fixed by gating the accumulator with the same `alive` mask `nstep` already used. Verified three ways:

* CPU and GPU now agree **per seed** to 2.8e-08 relative, on ESS, time-to-goal and minimum barrier;
* at 512 seeds the GPU reports **160.1** against the repository's own shipped CPU value of 159.91;
* the external review's 170.62 at 30 seeds is ordinary seed noise, not a disagreement: the shipped 30
  runs have a per-seed range of 99.1 to 226.9 and a standard deviation of 34.4, so a 95 % interval on
  their mean is [147.6, 172.2] — which contains 170.62 and excludes 269.62 by nine standard errors.

**Scope.** Only the ESS column of the V16 table. `reached`, `ttf`, `collision_rate` and `min_h` were
already gated by `alive` and are unchanged — confirmed by diffing the per-seed arrays against the run made
before the fix, which is kept as `results/corridor_V16_before_ess_fix.json` so the diff can be redone: every
one of `reached`, `ttf`, `collision_rate` and `min_h` agrees to 0.0e+00 on all twelve controllers, and the
ESS inflation ranges from 0.95x to 1.93x across them, which is what "different factors per row" means. The V15 sweep in `tests/gpu_ess_scaling.py` is unaffected: it
evaluates weightings at snapshot states and never averages over an episode.

**What this cost.** The inflated column was the evidence for "restoring the weights collapses the sample
size, and the intervention penalty buys it back". The conclusion survives, because the collapse to ~2 and
the recovery with `mu` are both far larger than a 1.5x accounting error, but the *magnitude* of the
recovery was overstated, and the corrected numbers are the ones in section 4 and in the README.

The defect was caught only because an outside number disagreed with mine. `tests/gpu_ess_scaling.py`'s own
docstring had recorded the right value — "an episode mean of about 160 of 500 for plain MPPI" — as a
sanity anchor for a different harness, and this work did not notice that its own closed-loop harness
contradicted it. That is the second time in this project a confident number was wrong and a disagreement
with something already shipped was what found it.

## 5d. A setting that must be quoted with the V15 table

`tests/gpu_ess_scaling.py` builds its reference episode and its controllers from the **class** defaults,
so its sweep runs at `sigma_v = 0.5`, while `experiments.DEFAULT`, the shipped corridor tables and the
closed-loop study all use `sigma_v = 1.0`. The sweep is internally consistent — all three weightings are
evaluated at the same states with the same draws — but its absolute values are not comparable with the
rest of this repository. The shipped sweep uses eight snapshot states and three repetitions; generating
six the same way and evaluating plain MPPI at them gives a median effective sample size at K = 500 of
176.7 at `sigma_v = 0.5` against 3.9 at `sigma_v = 1.0`, because the larger input spread carries the
reference episode further down the corridor — to x = 2.79 rather than 1.58 by the same step — into states
where the cost softmax concentrates. The sweep's **conclusion** is a statement about scaling in K and is unaffected; its
**absolute numbers** belong to `sigma_v = 0.5` and are quoted that way from here on.

## 6. Reproducing

```
python -m scbf_mppi.selftest                 #   5 s
python -m scbf_mppi.vessel.selftest          #  10 s
python -m scbf_mppi.vessel.ext.selftest_ext  #   2 min, 26 checks
python tests/test_regression.py check        #   3 min, twelve configurations, bit for bit
python tests/gpu_ess_scaling.py              #  needs CUDA; CPU/GPU gate then the K sweep
python tests/gpu_closed_loop.py --seeds 512  #  needs CUDA; the closed-loop table above
python tests/logratio_decomposition.py       #  30 s, the section 4 split of the log density ratio
node   live_demo/xval_obstruction.js         #   1 s, section 2b against the browser core
node   live_demo/xval_intervention.js 24     #   2 min, the mu dial against section 4
```

The two GPU scripts refuse to report anything unless they first reproduce the CPU implementation on the
same seed and noise. That gate exists because three separate harness faults in this work produced
confident wrong answers, and every one was caught by disagreeing with a number this repository already
shipped.

## 7. Novelty, checked against the literature rather than assumed

Each of the four claims above was put to a four-way prior-art search — an academic-index sweep, a
citation-graph sweep, an open-web sweep and an implementation sweep — with the searchers instructed to
**refute** novelty rather than confirm it. Every citation below was then read directly and the quoted
condition verified in the source, because a wrong citation is worse than none. The result is that nothing
here is a new theorem, one claim is prior art outright, and the honest description of the rest is
*composition*.

| claim | verdict |
|---|---|
| §2 the shaping optimum is singular | new composition of known parts |
| §2b no Gaussian exists at that state | new composition of known parts |
| §3 the chi-square efficiency floor | new composition of known parts — nothing mathematical is new |
| §4 the intervention penalty | **known prior art** |

**§4 is not new, and this matters most.** Charging a safety filter's own effort as an extra λ-scaled
running cost inside the MPPI exponent is published. Gandhi, Almubarak, Aoyama and Theodorou, *Safety in
Augmented Importance Sampling* (arXiv:2204.05963, 2022), Algorithm 1, accumulates

```
S_hat_n  +=  q(x) + (lambda (1 - beta) / 2) k_fb' Sigma^-1 k_fb
```

— the safe controller's own per-sample feedback `k_fb`, charged into the sampled cost with a tunable
multiplier `(1 - beta)`, inside a free-energy derivation. Its parent, Robust MPPI (Gandhi, Vlahov, Gibson,
Williams, Theodorou, RA-L 2021), is the same construction a year earlier. More fundamentally, the *reason*
it works is Section III-B of MPPI's own founding paper, Williams, Aldrich and Theodorou (arXiv:1509.01149),
titled "Likelihood Ratio as Additional Running Cost": a change of sampling law appears in the estimator
exactly as an added running cost. So the mechanism is not merely known, it is the mechanism MPPI is built
on.

What is left is narrow and should be stated narrowly: the scalar charged here is the SCBF program's **own
optimal objective value** `|m| + (sigma_0 - s)`, which the paper computes at every sample and discards,
and the setting differs in one way that matters — RMPPI and SAIS shift the mean under a fixed covariance,
so their exact Radon-Nikodym derivative is available in closed form, whereas Algorithm 1 re-solves a
per-sample mean **and** covariance factor, so the exact ratio carries a determinant term and the collapse
documented in §2 and §4. A cheap deterministic surrogate is attractive here for a reason that does not
arise there. The `mu -> large` equivalence is measured and directional, not proved.

**§2 and §2b are compositions of published inequalities.** Verified in the sources:

* Williams, Aldrich and Theodorou (arXiv:1509.01149), Theorem 1, states the covariance-shaping likelihood
  ratio only "under the condition that each `A_{t_i}` is invertible and each `Gamma_i` is invertible" — so
  the hypothesis that §2 shows Algorithm 1 violates is written into MPPI's own covariance-shaping theorem.
* Lubin, Bienstock and Vielma, *Two-sided linear chance constraints and extensions* (arXiv:1507.01995),
  Lemma 16, equations (63)-(65), gives the two-sided Gaussian cap; the per-row pair used here is exactly
  their axis-aligned approximation `A_eps`, which they note is the form Bienstock et al. use in practice.
* The finite-variance floor for a Gaussian-to-Gaussian importance weight is textbook (Owen ch. 9) and
  published in matrix form.
* Tao et al. themselves write, immediately after (11), that the update "cannot guarantee the optimality
  anymore", and their Remark 1 already derives the variance cap.

The residue in §2 is *where the optimum lies* — that the program separates into the directional standard
deviation `s = ||P'a||`, that the objective is piecewise linear in `s` with its only kink at `slack/z`, and
therefore that `s* = 0` is the **generic** outcome in precisely the regime where the filter acts, rather
than an edge case. The residue in §2b is the observation that the cap and the floor bind the *same object*,
because in this method the covariance the barrier shrinks **is** the importance-sampling proposal
covariance. Both are pointers, not theorems.

A caveat that sharpens §2 rather than weakening it, and which the sections already reflect: the collapse
belongs to the **standard-deviation** form, the corrected constraint. For the **variance** form the paper
actually prints, the objective is strictly convex on the active branch and the optimum is bounded away from
zero — which is why §2 measures 13.0 % of samples degenerate with the corrected form and **0.0 %** with the
printed one. The degeneracy is a property of the constraint the paper claims to enforce, not of the
inequality it typesets.

**§3 is a two-line corollary of a textbook identity.** The chi-square floor is Hammersley-Chapman-Robbins
with the test function taken to be the indicator of the violation set; Polyanskiy and Wu (*Information
Theory: From Coding to Learning*, CUP, Ch. 7) give the variational form at (7.73), the Bernoulli value at
Prop. 7.2, and set the event rearrangement as a reader exercise. The `ESS = K/(1 + chi2)` half is Kong
(1992), with rigorous versions in Agapiou et al. (2017) and Chatterjee and Diaconis (2018). Nothing
mathematical here is new. What is not standard is the *reading*: that this is a price every constraint-
satisfying proposal in sampling-based MPC pays, and a floor no shaping scheme can engineer around.

### The three equations no search found stated elsewhere

Written out, because the rest of this section is about what is *not* new and the reader deserves the other
half in one place. Each is absent from Tao et al. **and** was not found in any other source by the four-way
search. None is a theorem: they are, in order, three lines of convex analysis, a slope comparison, and one
line of arithmetic joining two published inequalities.

```
s*  =  clip( (a·ubar − b) / z ,  0 ,  ||P0' a|| )      the optimum of the paper's own (8), which it never locates
z   >  ||a||_inf / ||a||_2                             exactly when that optimum is zero
r − l  >  sqrt(2) · z · sigma_0                        exactly when any usable Gaussian exists at that state
```

The honest status of all three is **"not found stated anywhere, and I looked"** — four independent searches
returning nothing is not proof that nothing exists, and none of them should be called new without that
qualifier. What each *buys* is in sections 2 and 2b: the first makes the covariance collapse the generic
outcome in the only regime where the filter acts rather than an edge case; the second says when; the third
says that at some states no Gaussian works at all, whatever the optimiser does.

Two things that look like they belong on this list and do not. The corrected constraint
`a'mu − z sqrt(a' Sigma a) >= b` is not new, it is the textbook chance constraint and merely the one the
paper should have printed. And the intervention-penalty weight `w propto exp(−(S + mu lambda sum_t I)/lambda)`
is not in Tao et al. either, which is what made it look new — but it is in Gandhi et al. (2022). *Not in the
paper under review* and *new* are different things, and that gap is what caught this work out once already.

### The budget equation, checked separately

`scbf_mppi/ess_budget.py` inverts the Gaussian weight second moment into an admissibility budget on the
mean shift and chains it over the horizon. It was not one of the four claims above and was checked on its
own. **Verdict: new composition of known parts — and the only equation here whose closed form the search
found stated nowhere.** Three concessions come with that, all verified in the sources:

* the expression being inverted is published repeatedly — it is the exponentiated Renyi-2 divergence
  between two Gaussians, POIS (Metelli, Papini, Faccio, Restelli, NeurIPS 2018) Appendix C eq. (14) at
  `alpha = 2`, equivalently Sanz-Alonso and Wang Prop. 2.4 — and the inversion is one line of algebra;
* the identical inversion is published in a **different divergence**: Otto et al. (ICLR 2021) invert a
  closed-form Gaussian divergence into an explicit mean-shift ball and impose it as a hard projection,
  for KL, Wasserstein-2 and Frobenius;
* at `beta = 1` the budget collapses to `|m| <= sigma_0 sqrt(log kappa)`, the one-line inversion of the
  classical exponential-tilting identity. Verified exact here at `kappa` = 1.2, 2 and 5.

Also conceded: `ESS = N/d_2` is POIS eq. (6) and Kong 1992; the `beta > 1/sqrt(2)` admissibility is the
Geweke-Pitt finite-variance condition; the per-step-to-horizon chaining is POIS Proposition E.1; and an ESS
floor as a hard constraint on proposal movement is Doubly Adaptive Importance Sampling.

What survives: the inversion itself with its feasibility test (no mean shift admissible once
`kappa <= beta^2/sqrt(2 beta^2 - 1)`); that the `2 beta^2 - 1` factor **couples** mean shift and covariance
shrink into a single feasible region rather than two separate budgets; and its use as a hard per-sample
constraint inside a chance-constrained sampling-based MPC safety filter, where the MPPI safety-filter
family bounds neither the weight second moment nor the effective sample size. Checked here: the `beta = 1`
collapse is exact to 0.0e+00, the feasibility threshold is admissible above and NaN below, and substituting
the bound back into `E_q[w^2]` returns `kappa` to 1.8e-15.

### The three-region sampler is the reviewer's, not this repository's

The research brief's constructive contribution — a sampler keeping the nominal Gaussian's conditional shape
inside and outside the admissible interval while reallocating the three probabilities — is **not
implemented here**, and its reported 30-seed comparison (effective size 1.90 to 5.79) is **not reproduced**;
it needs the author's prototype. It also sits in a known tradition, as the brief itself says: Pitt, Tran,
Scharth and Kohn (arXiv:1307.7975) give the finite-moment condition and develop a two-component mixture
proposal in section 3.1 precisely to impose it, and Patrick and Bakolas (arXiv:2403.18066) eq. (24) put a
truncated Gaussian -- positive piecewise reweighting -- inside MPPI, proved valid because it stays strictly
positive wherever the base density is non-zero. Both were read and confirmed to say that.

**How this should be said out loud.** "None of the measure theory or the inequalities are mine, and the
intervention penalty is Theodorou's group's construction from 2021-22 — what I did was locate the optimum
of this paper's own per-sample program, notice that the object its barrier shrinks is the proposal
covariance itself, and measure what that costs."
