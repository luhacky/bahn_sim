#!/usr/bin/env python3
"""Interactive preview page: pick obstacle, distance, weather, light and camera artifacts, render a quick
low-resolution preview, and collect the settings you like into a condition file for a full run.

    python preview_server.py            # then open http://localhost:8765 in a browser
    python preview_server.py --port 9000 --blend out/scene/sbahn_loop.blend

Only the Python standard library is needed. Blender stays loaded in the background, so after the first
preview (scene load ~1-2 min) each preview takes about 10-20 s at 640x360.
Collected settings go to conditions/custom.json; render them with
    blender -b -P gpu_enable.py -P render_dataset.py -- --mode milestones --conditions conditions/custom.json ...
(the page shows the exact command).
"""
import argparse
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from sbahn_sim.launch import find_blender  # noqa: E402

QUEUE = os.path.join(HERE, "out", "preview")
CUSTOM = os.path.join(HERE, "conditions", "custom.json")
# ponytail: copy of sbahn_sim.conditions.PRESETS keys (that module needs Blender); update both together
PRESETS = ["clear_noon", "clear_morning", "partly_cloudy", "overcast", "low_sun_glare", "sunset", "twilight",
           "night_clear", "night_overcast", "fog_light", "fog_dense", "night_fog", "rain_light", "rain_heavy",
           "night_rain"]
ARTIFACTS = ["lens_flare", "windscreen_dirt", "rain_smear", "lens_smudge"]
FULL_FOG = 0.026
LOCK = threading.Lock()

PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>S-Bahn preview</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
body{font-family:system-ui,sans-serif;margin:0;background:#16191d;color:#e6e6e6}
main{display:grid;grid-template-columns:330px 1fr;gap:16px;padding:16px}
@media(max-width:800px){main{grid-template-columns:1fr}}
fieldset{border:1px solid #333;border-radius:6px;margin:0 0 10px;padding:8px 10px}
legend{color:#9ab}label{display:flex;justify-content:space-between;align-items:center;margin:5px 0;gap:8px;font-size:14px}
input,select,button{font:inherit}input[type=number],select{width:150px;background:#222;color:#eee;border:1px solid #444;border-radius:4px;padding:2px 4px}
button{background:#2f6f9f;color:#fff;border:0;border-radius:5px;padding:8px 14px;cursor:pointer;margin:4px 4px 0 0}
button.alt{background:#555}#img{max-width:100%;border-radius:6px;background:#000;min-height:200px;display:block}
#status{margin:8px 0;color:#9ab;font-size:14px;white-space:pre-wrap}pre{background:#111;padding:8px;border-radius:5px;overflow:auto;font-size:12px}
</style></head><body><main><form id="f">
<fieldset><legend>Obstacle</legend>
<label>case <select name="case">__CASES__</select></label>
<label>distance (m) <input type="number" name="distance_m" value="100" min="10" max="800"></label></fieldset>
<fieldset><legend>Weather and light</legend>
<label>preset <select name="preset">__PRESETS__</select></label>
<label>time of day <input name="time_of_day" placeholder="from preset" size="8"></label>
<label>cloud cover <input type="range" name="cloud_cover" min="-0.05" max="1" step="0.05" value="-0.05"><output></output></label>
<label>fog % <input type="range" name="fog_percent" min="-5" max="100" step="5" value="-5"><output></output></label>
<label>rain mm/h <input type="range" name="rain_rate_mm_h" min="-1" max="50" step="1" value="-1"><output></output></label></fieldset>
<fieldset><legend>Camera artifacts (0 = off)</legend>__ARTS__
<label>camera model <input type="checkbox" name="camera_fx" checked></label></fieldset>
<fieldset><legend>Preview quality</legend>
<label>size <select name="res"><option value="480,270">480x270</option><option value="640,360" selected>640x360</option><option value="1280,720">1280x720</option></select></label>
<label>samples <input type="number" name="samples" value="16" min="4" max="128"></label></fieldset>
<button type="submit">Render preview</button>
<fieldset><legend>Collect for a full run</legend>
<label>name <input name="name" placeholder="my_setting" size="12"></label>
<label>distances (m) <input name="milestones" value="50 100 200" size="12"></label>
<button type="button" id="add" class="alt">Add to custom.json</button></fieldset>
</form><section><img id="img" alt="preview appears here"><div id="status">Blender is starting...</div>
<h3>conditions/custom.json</h3><pre id="custom"></pre><h3>Render it</h3><pre id="cmd"></pre></section></main>
<script>
const f=document.getElementById('f'),st=document.getElementById('status');
f.querySelectorAll('input[type=range]').forEach(r=>{const o=r.nextElementSibling;const u=()=>o.textContent=(+r.value<0?'preset':r.value);r.oninput=u;u();});
function settings(){const d=Object.fromEntries(new FormData(f));const c={preset:d.preset,case:d.case,distance_m:+d.distance_m,
 res:d.res.split(',').map(Number),samples:+d.samples,camera_fx:f.camera_fx.checked};
 if(d.time_of_day)c.time_of_day=d.time_of_day;if(+d.cloud_cover>=0)c.cloud_cover=+d.cloud_cover;
 if(+d.fog_percent>=0)c.fog_percent=+d.fog_percent;if(+d.rain_rate_mm_h>=0)c.rain_rate_mm_h=+d.rain_rate_mm_h;
 const a={};__ARTKEYS__.forEach(k=>{if(+d[k]>0)a[k]=+d[k]});if(Object.keys(a).length)c.camera_artifacts=a;
 c.name=d.name;c.milestones=d.milestones;return c;}
async function post(u,b){const r=await fetch(u,{method:'POST',body:JSON.stringify(b)});return r.json();}
f.onsubmit=async e=>{e.preventDefault();st.textContent='rendering...';const t=Date.now();
 const r=await post('/preview',settings());
 if(r.ok){document.getElementById('img').src='/preview.png?'+t;st.textContent=`done in ${r.seconds} s (render + camera model)`;}
 else st.textContent='failed:\\n'+r.error;};
document.getElementById('add').onclick=async()=>{const r=await post('/add',settings());show(r);};
function show(r){document.getElementById('custom').textContent=JSON.stringify(r.custom,null,1);document.getElementById('cmd').textContent=r.command;}
(async()=>{show(await (await fetch('/custom')).json());for(;;){const r=await (await fetch('/state')).json();
 if(r.ready){if(!f.case.options.length)location.reload();st.textContent='ready - press Render preview';break;}if(r.error){st.textContent=r.error;break;}await new Promise(s=>setTimeout(s,2000));}})();
</script></body></html>"""


def condition_from(req):
    """Page settings -> condition dict as used by render_dataset / render_batch."""
    c = {"preset": req["preset"]}
    for k in ("time_of_day", "cloud_cover", "rain_rate_mm_h", "camera_artifacts"):
        if k in req:
            c[k] = req[k]
    if "fog_percent" in req:
        c.update(fog_percent=req["fog_percent"], fog_density=round(FULL_FOG * req["fog_percent"] / 100, 6),
                 fog_top_m=80.0, fog_patchiness=0.0)
    return c


def load_custom():
    return json.load(open(CUSTOM)) if os.path.exists(CUSTOM) else []


def command():
    return ("blender -b -P gpu_enable.py -P render_dataset.py -- --blend out/scene/sbahn_loop.blend --mode milestones "
            "--conditions conditions/custom.json --res 1280 720 --samples 32 --overlay --out out/custom")


class Handler(BaseHTTPRequestHandler):
    worker = None

    def log_message(self, *a):
        pass

    def send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/":
            cases = json.load(open(os.path.join(QUEUE, "ready.json")))["cases"] if os.path.exists(os.path.join(QUEUE, "ready.json")) else []
            html = (PAGE.replace("__CASES__", "".join(f"<option>{c}</option>" for c in cases))
                    .replace("__PRESETS__", "".join(f"<option{' selected' if p == 'partly_cloudy' else ''}>{p}</option>" for p in PRESETS))
                    .replace("__ARTS__", "".join(f'<label>{a} <input type="number" name="{a}" value="0" min="0" max="1" step="0.1"></label>' for a in ARTIFACTS))
                    .replace("__ARTKEYS__", json.dumps(ARTIFACTS)))
            self.send(200, html.encode(), "text/html; charset=utf-8")
        elif path == "/state":
            dead = self.worker.poll() is not None
            self.send(200, dict(ready=os.path.exists(os.path.join(QUEUE, "ready.json")) and not dead,
                                error="Blender stopped, see out/preview/worker.log" if dead else None))
        elif path == "/preview.png" and os.path.exists(os.path.join(QUEUE, "preview.png")):
            self.send(200, open(os.path.join(QUEUE, "preview.png"), "rb").read(), "image/png")
        elif path == "/custom":
            self.send(200, dict(custom=load_custom(), command=command()))
        else:
            self.send(404, {"error": "not found"})

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        if self.path == "/preview":
            with LOCK:                      # one render at a time
                job = dict(condition_from(req), case=req["case"], distance_m=req["distance_m"], res=req["res"],
                           samples=req["samples"], camera_fx=req["camera_fx"])
                res_path = os.path.join(QUEUE, "result.json")
                if os.path.exists(res_path):
                    os.remove(res_path)
                with open(os.path.join(QUEUE, "request.json.tmp"), "w") as fh:
                    json.dump(job, fh)
                os.replace(os.path.join(QUEUE, "request.json.tmp"), os.path.join(QUEUE, "request.json"))
                while not os.path.exists(res_path):
                    if self.worker.poll() is not None:
                        return self.send(200, dict(ok=False, error="Blender stopped, see out/preview/worker.log"))
                    time.sleep(0.3)
                self.send(200, json.load(open(res_path)))
        elif self.path == "/add":
            custom = load_custom()
            name = (req.get("name") or "").strip() or f"custom_{len(custom) + 1:02d}"
            cond = dict(condition_from(req), name=name, cases=[req["case"]],
                        milestones=[float(x) for x in str(req.get("milestones", "")).split()] or [req["distance_m"]])
            custom = [c for c in custom if c["name"] != name] + [cond]
            os.makedirs(os.path.dirname(CUSTOM), exist_ok=True)
            json.dump(custom, open(CUSTOM, "w"), indent=1)
            self.send(200, dict(custom=custom, command=command()))
        else:
            self.send(404, {"error": "not found"})


def main():
    import subprocess
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to reach the page from other machines")
    ap.add_argument("--blend", default=os.path.join(HERE, "out", "scene", "sbahn_loop.blend"))
    ap.add_argument("--blender")
    args = ap.parse_args()
    if not os.path.exists(args.blend):
        sys.exit(f"scene not found: {args.blend} (run setup.py first)")
    os.makedirs(QUEUE, exist_ok=True)
    for f in ("ready.json", "request.json", "result.json"):
        if os.path.exists(os.path.join(QUEUE, f)):
            os.remove(os.path.join(QUEUE, f))
    blender, _ = find_blender(args.blender)
    log = open(os.path.join(QUEUE, "worker.log"), "w")
    Handler.worker = subprocess.Popen([blender, "-b", args.blend, "-P", os.path.join(HERE, "gpu_enable.py"),
                                      "-P", os.path.join(HERE, "preview_worker.py"), "--", QUEUE],
                                     stdout=log, stderr=subprocess.STDOUT, cwd=HERE)
    print(f"open http://localhost:{args.port}  (Blender is loading the scene in the background; Ctrl+C to stop)")
    try:
        ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()
    finally:
        Handler.worker.terminate()


if __name__ == "__main__":
    main()
