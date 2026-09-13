"""Scripted MP4 clips of the 3-D page with nobody at the browser: headless Chrome driven over the DevTools protocol,
the page's own clock set frame by frame (window.__scbf_3d.setTime), one screenshot per frame, ffmpeg to h264.
    python capture_mp4.py --scene current --cam chase --follow 0 --t0 20 --t1 80 --speed 2 --fps 24 --out ../../figures/anim3d_current_chase.mp4
--speed = simulated seconds per second of video (2 = the demo's default playback).  Needs the demo server on 8080
(python -m http.server 8080 --bind 127.0.0.1 in demo/), Chrome, and imageio-ffmpeg in the venv.  The in-page ● record
button is the alternative on the real GPU."""
import argparse, base64, json, os, shutil, subprocess, sys, tempfile, time, urllib.request
import websocket

ap = argparse.ArgumentParser()
ap.add_argument("--scene", default="current"); ap.add_argument("--cam", default="chase"); ap.add_argument("--follow", type=int, default=0)
ap.add_argument("--t0", type=float, default=0.0); ap.add_argument("--t1", type=float, default=None); ap.add_argument("--speed", type=float, default=2.0)
ap.add_argument("--fps", type=int, default=24); ap.add_argument("--size", default="1600x900"); ap.add_argument("--out", required=True)
ap.add_argument("--url", default="http://127.0.0.1:8080/viewer_3d.html"); ap.add_argument("--port", type=int, default=9333)
ap.add_argument("--chrome", default=r"C:\Program Files\Google\Chrome\Application\chrome.exe")
a = ap.parse_args()
W, H = (int(v) for v in a.size.split("x"))

prof = tempfile.mkdtemp(prefix="scbf3d_chrome_"); frames = tempfile.mkdtemp(prefix="scbf3d_frames_")
chrome = subprocess.Popen([a.chrome, "--headless=new", "--use-angle=d3d11", "--enable-gpu-rasterization", "--ignore-gpu-blocklist", "--no-sandbox",   # WebGL on the GPU, not SwiftShader
                           f"--remote-debugging-port={a.port}", f"--user-data-dir={prof}", f"--window-size={W},{H}", "--hide-scrollbars", "about:blank"],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
try:
    for _ in range(60):
        try:
            targets = json.load(urllib.request.urlopen(f"http://127.0.0.1:{a.port}/json", timeout=2)); break
        except Exception:
            time.sleep(0.5)
    else:
        raise SystemExit("Chrome did not open its DevTools port")
    page = next(t for t in targets if t.get("type") == "page")
    ws = websocket.create_connection(page["webSocketDebuggerUrl"], suppress_origin=True); mid = [0]
    def call(method, **params):
        mid[0] += 1; ws.send(json.dumps({"id": mid[0], "method": method, "params": params}))
        while True:
            msg = json.loads(ws.recv())
            if msg.get("id") == mid[0]:
                if "error" in msg: raise RuntimeError(f"{method}: {msg['error']}")
                return msg.get("result", {})
    def js(expr, await_promise=False):
        r = call("Runtime.evaluate", expression=expr, awaitPromise=await_promise, returnByValue=True)
        return r.get("result", {}).get("value")
    call("Emulation.setDeviceMetricsOverride", width=W, height=H, deviceScaleFactor=1, mobile=False)
    url = f"{a.url}?scene={a.scene}&cam={a.cam}&follow={a.follow}&t={a.t0}"
    call("Page.navigate", url=url)
    for _ in range(100):
        if js("!!(window.__scbf_3d && window.__scbf_3d.rig)"): break
        time.sleep(0.2)
    else:
        raise SystemExit("page did not initialise")
    t_max = js("window.__scbf_3d.rig.tMax"); t1 = min(a.t1 if a.t1 is not None else t_max, t_max)
    n = int(round((t1 - a.t0) / a.speed * a.fps)); print(f"{a.scene}/{a.cam}: t {a.t0}..{t1:.1f} s at {a.speed}x -> {n} frames", file=sys.stderr)
    js("window.__scbf_3d.play(false); window.__scbf_3d.setTime(%f)" % a.t0)
    js("new Promise(r => setTimeout(r, 600))", True)                       # let the fonts and the first frames settle
    t = time.time()
    for i in range(n):
        ts = a.t0 + i * a.speed / a.fps
        js("window.__scbf_3d.setTime(%f); new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)))" % ts, True)
        png = call("Page.captureScreenshot", format="png")["data"]
        open(os.path.join(frames, f"f_{i:05d}.png"), "wb").write(base64.b64decode(png))
        if i % 100 == 0: print(f"  frame {i}/{n}  ({time.time()-t:.0f} s)", file=sys.stderr)
    ws.close()
finally:
    chrome.terminate()
    try: chrome.wait(timeout=10)
    except Exception: chrome.kill()
    shutil.rmtree(prof, ignore_errors=True)

import imageio_ffmpeg
ff = imageio_ffmpeg.get_ffmpeg_exe(); os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
subprocess.run([ff, "-y", "-hide_banner", "-loglevel", "error", "-framerate", str(a.fps), "-i", os.path.join(frames, "f_%05d.png"),
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "19", "-preset", "medium", "-movflags", "+faststart", a.out], check=True)
shutil.rmtree(frames, ignore_errors=True)
print("wrote", a.out, f"({os.path.getsize(a.out)/1e6:.1f} MB, {n} frames, {n/a.fps:.1f} s)", file=sys.stderr)
