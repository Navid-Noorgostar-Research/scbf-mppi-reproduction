"""How far did the numbers move when the dependency versions moved?

    python tests/drift_report.py --old results_shipped_2026_09_12 --new results

The package was first run in 2026 with floors in requirements.txt ("numpy>=1.26", "clarabel>=0.6").  A fresh
install two days after those results were recorded therefore installed a different stack, and the numbers moved --
not because the code changed, but because the per-sample chance-constraint problem is solved by a convex
solver whose iterates differ between releases.  The pure-MPPI numbers, which touch no solver, did not move.

This script quantifies that, per configuration, for every vessel and corridor experiment that exists in
both trees.  A configuration is called CONSISTENT when the new mean lies inside the old 95 % bootstrap
confidence interval and the old mean lies inside the new one; the interesting cases are the ones that are
not, and they are listed individually so any reported claim can be checked against them.

This is the honest version of "it reproduces": it reproduces exactly on a pinned stack (tests/test_regression.py)
and to within the stated interval on a moving one (this script).
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

KEYS = ("collision_rate", "collision_rate_ctrl", "ttf", "min_h", "ess_mean", "path_length", "mean_speed")
PAIRED = ("ttf", "collision_rate", "min_h")


def load(tree, name):
    p = os.path.join(ROOT, tree, name)
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def paired(old_tree, new_tree, verbose=True, tol=1e-9, material=1e-6):
    """Seed-by-seed comparison -- the only comparison that is free of sampling noise.

    Two runs of the same configuration with the same seed are the same deterministic computation.  If they
    disagree, the stack changed the answer; no amount of averaging can explain it away.  This is stronger
    than comparing confidence intervals, and it is immune to the two trees having been run with a different
    number of seeds (which E12 and E13 were: 10 then 30).

    Three regimes are separated, because they mean entirely different things:

      identical    every shared seed agrees to the last bit
      rounding     the seeds differ by less than `material` in relative terms -- floating-point noise from a
                   different BLAS, visible only because time-to-goal is a threshold crossing and can land one
                   control step earlier
      material     the seeds differ by more than that: the computation genuinely took a different path
    """
    old_dir = os.path.join(ROOT, old_tree)
    new_dir = os.path.join(ROOT, new_tree)
    files = [f for f in sorted(set(os.listdir(old_dir)) & set(os.listdir(new_dir))) if f.endswith(".json")]
    rows = []
    for f in files:
        try:
            o, n = load(old_tree, f), load(new_tree, f)
        except Exception:
            continue
        if not isinstance(o, dict):
            continue
        for cfg in o:
            if cfg.startswith("_") or cfg not in n:
                continue
            try:
                do = {r["seed"]: r for r in o[cfg]["runs"]}
                dn = {r["seed"]: r for r in n[cfg]["runs"]}
            except Exception:
                continue
            common = sorted(set(do) & set(dn))
            if not common:
                continue
            n_diff = 0
            n_mat = 0
            worst_rel = 0.0
            d_ttf = 0.0        # worst absolute change in time to goal   [s]
            d_coll = 0.0       # worst absolute change in collision rate [dimensionless, in 0..1]
            d_minh = 0.0       # worst absolute change in the closest approach [m]
            for s in common:
                differs = False
                mat = False
                for k in PAIRED:
                    a, b = do[s].get(k), dn[s].get(k)
                    if a is None or b is None:
                        continue
                    rel = abs(b - a) / max(abs(a), 1e-9)
                    if abs(a - b) > tol * max(1.0, abs(a)):
                        differs = True
                        worst_rel = max(worst_rel, rel)
                        if rel > material:
                            mat = True
                        if k == "ttf":
                            d_ttf = max(d_ttf, abs(b - a))
                        elif k == "collision_rate":
                            d_coll = max(d_coll, abs(b - a))
                        else:
                            d_minh = max(d_minh, abs(b - a))
                if differs:
                    n_diff += 1
                if mat:
                    n_mat += 1
            rows.append((f, cfg, len(common), n_diff, d_ttf, d_coll, d_minh, n_mat, worst_rel))
    if verbose:
        ident = [r for r in rows if r[3] == 0]
        rnd = [r for r in rows if r[3] > 0 and r[7] == 0]
        moved = [r for r in rows if r[7] > 0]
        print("SEED-BY-SEED (the same seed is the same deterministic computation)")
        print(f"  configurations compared:                            {len(rows)}")
        print(f"  identical to the last bit on every shared seed:     {len(ident)}")
        print(f"  differ only at rounding level (rel < {material:g}):        {len(rnd)}")
        print(f"  materially different on at least one shared seed:   {len(moved)}\n")
        for title, group in (("identical to the last bit", ident), ("rounding level only", rnd)):
            if group:
                print(f"  {title}:")
                for r in group:
                    print(f"      {r[0][:26]:<27s} {r[1][:46]:<47s} worst rel {r[8]:.1e}")
                print()
        if moved:
            print("  worst single-seed change, in the units of the quantity:")
            print(f"{'file':<28s}{'configuration':<44s}{'seeds':>6s}{'mat':>5s}"
                  f"{'d ttf [s]':>11s}{'d coll':>9s}{'d min h [m]':>13s}")
            for r in sorted(moved, key=lambda r: -r[4]):
                print(f"{r[0][:27]:<28s}{r[1][:43]:<44s}{r[2]:6d}{r[7]:5d}{r[4]:11.0f}{r[5]:9.3f}{r[6]:13.2f}")
        print()
    return rows


def compare(old_tree, new_tree, verbose=True):
    old_dir = os.path.join(ROOT, old_tree)
    new_dir = os.path.join(ROOT, new_tree)
    files = sorted(set(os.listdir(old_dir)) & set(os.listdir(new_dir)))
    files = [f for f in files if f.endswith(".json")]
    n_cfg = 0
    n_cmp = 0
    n_bad = 0
    bad = []
    for f in files:
        try:
            o = load(old_tree, f)
            n = load(new_tree, f)
        except Exception:
            continue
        if not isinstance(o, dict) or not isinstance(n, dict):
            continue
        for cfg in o:
            if cfg.startswith("_") or cfg not in n:
                continue
            so = o[cfg].get("summary") if isinstance(o[cfg], dict) else None
            sn = n[cfg].get("summary") if isinstance(n[cfg], dict) else None
            if not so or not sn:
                continue
            try:                                    # only compare like with like
                if len(o[cfg]["runs"]) != len(n[cfg]["runs"]):
                    continue
            except Exception:
                pass
            n_cfg += 1
            for k in KEYS:
                if k not in so or k not in sn:
                    continue
                n_cmp += 1
                a, b = so[k], sn[k]
                inside = (a["lo"] <= b["mean"] <= a["hi"]) and (b["lo"] <= a["mean"] <= b["hi"])
                if not inside:
                    n_bad += 1
                    rel = (b["mean"] - a["mean"]) / (abs(a["mean"]) + 1e-12)
                    bad.append((f, cfg, k, a["mean"], a["lo"], a["hi"], b["mean"], b["lo"], b["hi"], rel))
    if verbose:
        print(f"compared {n_cmp} summary quantities over {n_cfg} configurations in {len(files)} files")
        print(f"  consistent (each mean inside the other's 95 % interval): {n_cmp - n_bad}")
        print(f"  moved outside the interval:                             {n_bad}\n")
        if bad:
            print(f"{'file':<28s}{'configuration':<44s}{'quantity':<20s}{'was':>10s}{'now':>10s}{'rel':>9s}")
            for f, cfg, k, am, alo, ahi, bm, blo, bhi, rel in sorted(bad, key=lambda r: -abs(r[-1])):
                print(f"{f[:27]:<28s}{cfg[:43]:<44s}{k:<20s}{am:10.4g}{bm:10.4g}{rel:+8.1%}")
                print(f"{'':<28s}{'':<44s}{'  95% was':<20s}[{alo:.4g}, {ahi:.4g}]   now [{blo:.4g}, {bhi:.4g}]")
    return {"compared": n_cmp, "configs": n_cfg, "moved": n_bad, "detail": bad}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", default="results_shipped_2026_09_12")
    ap.add_argument("--new", default="results")
    a = ap.parse_args()
    if not os.path.isdir(os.path.join(ROOT, a.old)):
        print(f"no such tree: {a.old}\n"
              f"This report needs the result files as they were originally shipped.  They are kept "
              f"in results_shipped_2026_09_12/ in this repository.")
        sys.exit(2)
    paired(a.old, a.new)
    print("SUMMARY MEANS (configurations run with the same number of seeds only)")
    compare(a.old, a.new)
