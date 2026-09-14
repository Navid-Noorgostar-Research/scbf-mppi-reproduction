# The runs in 3D — `viewer_3d.html`

One offline HTML file (Three.js r128, vendored) that draws the same runs as the figures, in three
dimensions. Nothing here changes any experiment; every drawn
quantity is the 1-s state log of the exported runs, and the 30-seed statistics in the titles and footer are read
from the stored result files.

| scene | run | what it is |
|---|---|---|
| `current` | V3, seed 0 | MPPI · SCBF-MPPI (2nd-order barrier) · + IS under the coloured gust and a 0.5 m/s current the model does not know |
| `ferry` | V4, seed 2 | the same three through the crossing ferry (3 m/s, R = 25 m), white force noise |
| `corridor` | E1, seed 6 | the paper's corridor: MPPI · as printed · as claimed (corrected) — the stored E1 run |

Cameras: orbit (drag / wheel / right-drag), **chase** (behind the course over ground), **helm** (along the heading),
top. Keys: `space` play, `1–4` cameras, `[` `]` speed, `R` reset. Deep links:
`viewer_3d.html?scene=ferry&t=22&cam=chase&follow=1&speed=2&play=1`. Scripting: `window.__scbf_3d`.

What is disclosed on the page and must stay disclosed: 3-DOF planar model (no roll, pitch or heave); stylised hull
8.5 × 2.2 m; positions interpolated between the 1-s control states; the boats run **astern** when that is cheaper
(isotropic 700 N force input, damping symmetric in surge, no heading cost — a property of the force-input
abstraction, not of the barrier); the fan is the 40 highest-weight rollouts (brighter = heavier) and its height is
prediction time; the ring at the stern is the 700 N azimuth disk (6 m ≙ 700 N, the same scale as the gust arrow);
the coloured ring near an obstacle is ψ₁ = ḣ + α₁h = 0 at the followed barrier boat's bearing (R + v/α₁).

## Regenerate

```
cd code/scbf-mppi-reproduction
python -m scbf_mppi.export3d                 # re-simulates the three scenes with the rollout fan recorded -> live_demo/3d/scene_data.json  (~2.5 min)
python live_demo/3d/build3d.py               # inlines vendor/three.min.js + src/* + the data -> live_demo/3d/viewer_3d.html
```

Backup clips without anyone at the browser (needs Chrome and the demo server: `python -m http.server 8080 --bind 127.0.0.1` in `demo/`):

```
python live_demo/3d/capture_mp4.py --scene current --cam chase --follow 0 --t0 20 --t1 100 --speed 2 --out figures/anim3d_current_chase.mp4
python live_demo/3d/capture_mp4.py --scene ferry   --cam chase --follow 1 --t0 0  --t1 60  --speed 2 --out figures/anim3d_ferry_chase.mp4
python live_demo/3d/capture_mp4.py --scene current --cam orbit --t0 0 --t1 136 --speed 3        --out figures/anim3d_current_orbit.mp4
python live_demo/3d/capture_mp4.py --scene corridor --cam orbit --follow 2 --t0 0 --t1 12.5 --speed 1 --out figures/anim3d_corridor_orbit.mp4
```

The in-page **● record** button captures the tab (HUD included) to a WebM — convert with `ffmpeg -i x.webm -c:v libx264 -pix_fmt yuv420p x.mp4`.

Path-traced clips (Blender 4.2 LTS, Cycles + OptiX on the GPU, a Nishita physical sky lighting the scene; the same data
and the same readout as the page — `figures/anim3d_blender_current_chase.mp4` t = 36–80 s, `anim3d_blender_ferry_chase.mp4`
t = 6–44 s, both 1600×900 at 24 fps; the `*_flatworld.mp4` variants are the first pass with a plain grey world):

```
blender -b --python live_demo/3d/blender_render.py -- --scene current --cam chase --follow 0 --engine cycles --samples 48 --time-limit 3 --res 1600x900 --fps 24 --t0 36 --t1 80 --out C:\render\current_chase
ffmpeg -framerate 24 -start_number 865 -i C:\render\current_chase\frame_%04d.png -c:v libx264 -pix_fmt yuv420p -crf 18 figures/anim3d_blender_current_chase.mp4
```
(`--start_number` is the first frame index written, 1 + t0·fps; `--test-frame 52` renders one frame to check the look first.)

Rendering stack (all decoration, all offline): a Preetham sky (`vendor/Sky.plain.js`, the three.js example converted to a
plain script) feeds image-based lighting through `PMREMGenerator`; the water is the three.js `Water` example
(`vendor/Water.plain.js`, base reflectance patched from 0.3 to a physical 0.05 so the lake colour shows through except
at grazing angles) with a procedural tileable normal map (`vendor/waternormals.jpg`, generated — 40 random tileable
sinusoids); ACES filmic tone mapping; clearcoat hull materials with canvas-generated teak, solar-cell and steel textures;
foam wakes as sprites; a hazy silhouette ring at 3 km. Labels are hidden during the water's reflection pass. The current
is drawn as streaks advected on the surface; there is no wave dynamics in the model and none is implied.

Dependencies beyond `requirements.txt`: `imageio-ffmpeg` (an ffmpeg with libx264; the animation scripts pick it up
automatically), `websocket-client` (for `capture_mp4.py`). Three.js r128 is vendored in `vendor/` because the
r140+ builds have no single-file UMD bundle for an offline page.
