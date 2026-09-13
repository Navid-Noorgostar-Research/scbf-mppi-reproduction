import json, sys, os, numpy as np
_here = os.path.dirname(os.path.abspath(__file__)); [sys.path.insert(0, c) for c in (os.path.join(_here, '..', '..'), os.path.join(_here, '..', '..', 'scbf-mppi-reproduction'))]   # package layout / development tree
from scbf_mppi.vessel import solver_nd
d = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'js_instances.json'))); s0 = np.asarray(d['s0']); z = d['z']; alpha = d['alpha']
worst_mu = worst_P = worst_cost = 0.0; n_ok = 0; n_diff = 0; diffs = []
by = {}
for c in d['cases']:
    A = np.asarray(c['A'])[None]; b = np.asarray(c['b'])[None]; ubar = np.asarray(c['ubar'])[None]
    r = solver_nd.solve_rows_nd(ubar, A, b, s0, z=z, alpha=alpha, form=c['form'])
    mu_py, P_py = r['mu'][0], r['Pfac'][0]
    mu_js, P_js = np.asarray(c['mu']), np.asarray(c['P'])
    dmu = np.abs(mu_py - mu_js).max() / (1 + np.abs(mu_py).max()); dP = np.abs(P_py - P_js).max() / s0.max()
    cost_py = np.abs(mu_py).sum() + np.linalg.norm(P_py - np.diag(s0)); cost_js = np.abs(mu_js).sum() + np.linalg.norm(P_js - np.diag(s0))
    key = (c['form'], c['J'], bool(r['multi_violation'][0]))
    by.setdefault(key, []).append((dmu, dP, cost_py, cost_js))
    flags_same = (bool(r['active'][0]) == bool(c['active'])) and (bool(r['multi_violation'][0]) == bool(c['multi'])) and (bool(r['infeasible'][0]) == bool(c['infeasible'])) and (int(r['row'][0]) == (c['row'] if c['row'] >= 0 else int(r['row'][0])))
    if not flags_same: n_diff += 1
    if max(dmu, dP) > 1e-6: diffs.append((c['form'], c['J'], bool(r['multi_violation'][0]), dmu, dP, cost_py, cost_js))
    worst_mu = max(worst_mu, dmu); worst_P = max(worst_P, dP)
print('cases', len(d['cases']), 'flag mismatches', n_diff, 'worst rel |dmu|', f'{worst_mu:.2e}', 'worst |dP|/s0max', f'{worst_P:.2e}')
for k in sorted(by):
    v = np.asarray(by[k]); print(k, 'n', len(v), 'max dmu %.1e max dP %.1e' % (v[:,0].max(), v[:,1].max()), ' mean cost py %.4g js %.4g' % (v[:,2].mean(), v[:,3].mean()))
print('instances differing by > 1e-6:', len(diffs))
for x in diffs[:12]: print('  ', x)
