# The Python code — where it is and how to run it in VS Code

**Where it is.** This repository. It is the program behind every number, figure and animation reported here: the paper's corridor experiment re-implemented from the text, and the same method on the Solgenia model.

```
scbf-mppi-reproduction/
  README.md                      what is implemented, what was assumed, every experiment and its result
  requirements.txt               numpy, scipy, matplotlib, cvxpy, clarabel
  scbf_mppi/                     the corridor (the paper's own example)
     mppi.py                        MPPI, SCBF-MPPI (Algorithm 1) and deterministic MPPI (Homburger et al.)
     scbf.py                        the per-sample problem (8): closed form, the corrected SOCP, the literal SDP, checked against cvxpy
     env.py · dynamics.py           the corridor and the unicycle with Brownian noise
     simulate.py · experiments.py   closed-loop episodes, metrics, experiments E1–E14
     figures.py · animate.py        the corridor figures; the MPPI against SCBF-MPPI animation
     selftest.py                    a 5-second check that everything is consistent
  scbf_mppi/vessel/              the same method on the boat
     solgenia.py                    the Solgenia 3-DOF model (Ocean Engineering 2025, Table A.5), thrusters, RK4
     harbour.py                     obstacles, the ferry, the barriers (as printed, and second-order)
     solver_nd.py                   the per-sample problem in three inputs: one row exact, several rows jointly
     controllers.py · simulate.py   MPPI, SCBF-MPPI, + IS correction, deterministic MPPI; gust and current
     experiments.py · figures.py    experiments V0–V8 (30 seeds) and their figures
     animate.py · animate_current.py · animate_rowspace.py    the crossing-ferry, gust-and-current and row-space animations
     selftest.py                    a 10-second check: model vs MATLAB, barrier rows, Itô term, solver, delivered Pr
  results/                       every reported number, as JSON (E*.json corridor, vessel_V*.json boat) + summaries
  figures/                       every figure and animation (png, mp4)
  live_demo/                     the browser demo and its JavaScript core (sim.js), with the cross-checks against Python
```

---

## Set it up in VS Code (once, about five minutes)

1. **Clone or download this repository.** Open VS Code → **File → Open Folder…** → choose `scbf-mppi-reproduction` → "Yes, I trust the authors".
2. **Install the Python extension** (Microsoft): Extensions icon (`Ctrl+Shift+X` / `⇧⌘X`) → search **Python** → Install. You need Python 3.10 or newer on the laptop (python.org; on Windows tick "Add python.exe to PATH").
3. **Create the environment and install the packages:** press `Ctrl+Shift+P` (`⇧⌘P` on a Mac) → type **Python: Create Environment** → **Venv** → choose your Python → **tick `requirements.txt`** → OK. VS Code creates `.venv` and installs numpy, scipy, matplotlib, cvxpy and clarabel (one to three minutes).
   *Terminal alternative:* `python3 -m venv .venv` → activate it (`source .venv/bin/activate` on Mac/Linux, `.venv\Scripts\activate` on Windows) → `pip install -r requirements.txt`.
4. **Open a terminal** (menu Terminal → New Terminal). It should show `(.venv)` at the start of the line — if not, close it and open a new one after step 3 finished.

## Check that it works (15 seconds)

```
python -m scbf_mppi.selftest            # the corridor: solver vs cvxpy, Theorem 2, one episode — ends with "all passed"
python -m scbf_mppi.vessel.selftest     # the boat: model vs MATLAB, barrier rows, Itô term, solver — ends with "all passed"
```

## Regenerate an animation

```
python -m scbf_mppi.animate                    # corridor, MPPI vs SCBF-MPPI as printed, seed 1        (about 3 min)
python -m scbf_mppi.vessel.animate             # the boat with the crossing ferry, seed 2              (about 5 min)
python -m scbf_mppi.vessel.animate_current     # three boats under the gust and the current           (about 10 min)
python -m scbf_mppi.vessel.animate_rowspace    # inside one cycle, the row the filter works on        (about 10 min)
```

Each writes `figures/anim_*.gif` and, if **ffmpeg** is installed on the laptop, the `.mp4` as well (without ffmpeg it prints "mp4 failed" and still writes the GIF — the existing MP4s stay in `figures/`). Times are for a normal laptop on two cores.

## Reproduce a reported number

```
python -m scbf_mppi.experiments --exp E1 --runs 10          # Table I reproduction, 10 seeds       (about 10 min)
python -m scbf_mppi.figures                                 # redraws the corridor figures from results/*.json
python -m scbf_mppi.vessel.experiments --exp V2 --runs 5    # the harbour transit, 5 seeds         (about 20 min)
python -m scbf_mppi.vessel.figures                          # redraws the vessel figures
```

The full sets (`--exp all --runs 30`) take about an hour for the corridor and a few hours for the boat; the results they produce are the ones already in `results/`, so you never need to rerun them.

## If they ask "show me the code" — the six files to open

1. `scbf_mppi/mppi.py` — class `SCBFMPPI`: Algorithm 1, the per-sample loop, where the SDP sits (and the importance-sampling correction the paper omits).
2. `scbf_mppi/scbf.py` — `solve_rows`: problem (8) in closed form; `solve_exact_cvxpy` and `solve_sdp_literal`: the exact conic solves used for the timing and for validation.
3. `scbf_mppi/vessel/solgenia.py` — their model, unchanged: `drift`, `step` (RK4), `clip_inputs` (the 700 N disk, the bow cap).
4. `scbf_mppi/vessel/harbour.py` — `Harbour.rows`: the second-order barrier ψ₁ = ḣ + α₁h as an affine row in the force; `rows_printed`: the relative-degree-1 row that is zero.
5. `scbf_mppi/vessel/solver_nd.py` — `solve_rows_nd`: one active row exact, several rows by the joint candidate solve; `validate_against_cvxpy`.
6. `results/vessel_V2_harbour.json` (or any `results/*.json`) — every reported number, seed by seed.

Open a file with `Ctrl+P` and its name. Use **View → Appearance → Zen Mode** (`Ctrl+K Z`) and `Ctrl+=` to enlarge the text before sharing the window.
