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

This is the strongest claim here and the only one an independent novelty review rated as possibly new to
the literature rather than only to this paper.

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
21.5 % of runs, and the measured ESS is then exactly 500.

*The reference is the target, not the nominal law.* MPPI estimates under `pi` proportional to
`exp(-S/lambda) p`, so the governing divergence is `chi2(pi||q)` and `delta0` must be replaced by
`pi(A)`. If the running cost already suppresses the unsafe set below `delta`, the constraint forces no
divergence at all. So this bounds recovery of the nominal law, and constrains the MPPI estimator only
when `pi(A) > delta`. It is not an impossibility result for every safe MPPI variant.

*The ingredients are classical.* `ESS/K -> 1/(1+chi2)` and the chi-square characterisation of importance
sampling efficiency: Kong 1992; Agapiou, Papaspiliopoulos, Sanz-Alonso, Stuart, *Statistical Science*
2017; Chatterjee and Diaconis, *Ann. Appl. Prob.* 2018. The Gaussian tail condition is textbook, Owen
ch. 9. What is assembled here is the composition, not the inequalities.

The Gaussian corollary, `E[w^2] < infinity` iff `s > s0/sqrt(2)`, is the **scalar** case. In general the
condition is the matrix one, `2 Sigma_q - Sigma_p` positive definite, and an anisotropic shrink can fail
it while its largest ratio still exceeds `1/sqrt(2)`.

## 4. Where the paper's advantage actually comes from

Restoring the omitted weights gives, in the corridor over 512 seeds, the **best** controller tested:
100 % goal-reaching, the fastest time, a median collision rate of zero and a median minimum barrier of
+0.167, at an asymptotic effective sample size of **2.0** out of 500.

That is not path-integral averaging. At an effective size near one the update selects a single sample,
and the dominant term in the log density ratio is `log s`, so the sample it selects is the one the
barrier had to correct least over the whole horizon.

Charging for that explicitly turns the accident into a dial. Let `I = |m| + (s0 - s)` be the per-sample
problem's own objective value, which the paper computes and discards, and weight by

```
w_k  proportional to  exp( -( S_k + mu lambda sum_t I_k,t ) / lambda )
```

`mu = 0` reproduces the corrected controller to the digit and `mu` large reproduces the accidental rule,
with every value between a valid estimator, because `I` depends on the solve and not on the noise.
Over 512 seeds, `mu = 0.5` gives 99.4 % reached, median collision zero, median minimum barrier +0.133, at
an effective sample size of 88.7 rather than 2.0.

**What this does not establish.** It is not demonstrated to be a better controller. Its runaway-episode
rate is 3 of 512, which Fisher's exact test cannot distinguish from zero (p = 0.249), but that rate is
not monotone in `mu` (7, 9, 3, 8, 1 across the sweep), so the tail is unresolved at this seed count.
The typical-case statistics move smoothly with `mu`; the tail does not, and a safety argument rests on
the tail.

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

## 6. Reproducing

```
python -m scbf_mppi.selftest                 #   5 s
python -m scbf_mppi.vessel.selftest          #  10 s
python -m scbf_mppi.vessel.ext.selftest_ext  #   2 min, 26 checks
python tests/test_regression.py check        #   3 min, twelve configurations, bit for bit
python tests/gpu_ess_scaling.py              #  needs CUDA; CPU/GPU gate then the K sweep
python tests/gpu_closed_loop.py --seeds 512  #  needs CUDA; the closed-loop table above
```

The two GPU scripts refuse to report anything unless they first reproduce the CPU implementation on the
same seed and noise. That gate exists because three separate harness faults in this work produced
confident wrong answers, and every one was caught by disagreeing with a number this repository already
shipped.
