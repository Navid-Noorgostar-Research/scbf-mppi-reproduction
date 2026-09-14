"""Inline vendor/three.min.js (+ the Sky and Water example shaders and the water normal map), src/app.js, src/style.css
and scene_data.json into ONE offline file:
    python build3d.py      -> live_demo/3d/viewer_3d.html
Run scbf_mppi.export3d first (it writes scene_data.json)."""
import os, json, base64
here = os.path.dirname(os.path.abspath(__file__))
rd = lambda *p: open(os.path.join(here, *p), encoding="utf-8").read()
src = rd("src", "index.html"); three = rd("vendor", "three.min.js"); sky = rd("vendor", "Sky.plain.js"); water = rd("vendor", "Water.plain.js")
app = rd("src", "app.js"); css = rd("src", "style.css")
normals = base64.b64encode(open(os.path.join(here, "vendor", "waternormals.jpg"), "rb").read()).decode("ascii")
data = rd("scene_data.json")
json.loads(data)                                            # fail loudly on a broken export
assert "</script" not in data.lower(), "scene data must not contain a script end tag"
out = (src.replace("/*__CSS__*/", css).replace("/*__THREE__*/", three).replace("/*__SKY__*/", sky).replace("/*__WATER__*/", water)
          .replace("/*__WATERNORMALS__*/", normals).replace("/*__DATA__*/", data).replace("/*__APP__*/", app))
dst = os.path.join(here, "viewer_3d.html")
open(dst, "w", encoding="utf-8").write(out)
print(f"built {dst} ({len(out)/1e6:.1f} MB)")
