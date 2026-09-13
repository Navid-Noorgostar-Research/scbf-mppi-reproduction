"""Blender (4.2 LTS) render of a scene from scene_data.json — the stretch goal beside the WebGL page.
    blender -b --python blender_render.py -- --scene current --cam chase --follow 0 --out DIR [--engine eevee|cycles]
                                            [--fps 24] [--test-frame 52] [--res 1920x1080] [--samples 48]
Same data as the WebGL page: the boats are keyframed on the 1-s control states (Blender interpolates
between them), the paths grow with time, the ferry moves at its constant velocity, the executed azimuth thrust is an
arrow at the stern scaled to the 700 N disk, and a HUD text shows the readout of the followed boat.  3-DOF planar
model: no roll, pitch or heave.  Output: PNG frames (frame_####.png) — assemble with ffmpeg afterwards.
Coordinates: simulation x, y are Blender x, y (z up); heading psi -> rotation_euler.z."""
import bpy, json, math, sys, os
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
def arg(name, default):
    return argv[argv.index(name) + 1] if name in argv else default
HERE = os.path.dirname(os.path.abspath(__file__))
SCENE = arg("--scene", "current"); CAM = arg("--cam", "chase"); FOLLOW = int(arg("--follow", 0)); FPS = int(arg("--fps", 24))
OUT = arg("--out", os.path.join(HERE, "render", SCENE)); ENGINE = arg("--engine", "eevee"); TEST = arg("--test-frame", None)
RES = arg("--res", "1920x1080"); SAMPLES = int(arg("--samples", 48))
DATA = json.load(open(arg("--data", os.path.join(HERE, "scene_data.json")), encoding="utf-8"))
sc = DATA["scenes"][SCENE]; META = DATA["meta"]
if sc["kind"] != "vessel":
    raise SystemExit("blender_render.py renders the vessel scenes (current, ferry); the corridor is in the WebGL page")
os.makedirs(OUT, exist_ok=True)

# ----------------------------------------------------------------------------------------- helpers
def hexrgb(h):
    h = h.lstrip("#"); return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
def srgb_to_linear(c):
    return tuple((v / 12.92) if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4 for v in c)
def material(name, rgb, rough=0.55, metal=0.0, emit=0.0):
    m = bpy.data.materials.new(name); m.use_nodes = True; b = m.node_tree.nodes["Principled BSDF"]
    lin = srgb_to_linear(rgb); b.inputs["Base Color"].default_value = (*lin, 1.0); b.inputs["Roughness"].default_value = rough; b.inputs["Metallic"].default_value = metal
    if emit: b.inputs["Emission Color"].default_value = (*lin, 1.0); b.inputs["Emission Strength"].default_value = emit
    return m
def add_obj(name, mesh, mat=None, parent=None):
    o = bpy.data.objects.new(name, mesh); bpy.context.collection.objects.link(o)
    if mat is not None: o.data.materials.append(mat)
    if parent is not None: o.parent = parent
    return o
def box(name, sx, sy, sz, mat, loc=(0, 0, 0), parent=None, rot=0.0):
    bpy.ops.mesh.primitive_cube_add(size=1.0); o = bpy.context.active_object; o.name = name; o.scale = (sx, sy, sz); o.location = loc; o.rotation_euler = (0, 0, rot)
    o.data.materials.append(mat)
    if parent is not None: o.parent = parent
    return o
def cylinder(name, r, h, mat, loc=(0, 0, 0), parent=None, verts=32):
    bpy.ops.mesh.primitive_cylinder_add(radius=r, depth=h, vertices=verts); o = bpy.context.active_object; o.name = name; o.location = loc; o.data.materials.append(mat)
    if parent is not None: o.parent = parent
    return o
def ring(name, r, width, mat, z=0.04, parent=None):
    cu = bpy.data.curves.new(name, "CURVE"); cu.dimensions = "3D"; cu.bevel_depth = width / 2; cu.bevel_resolution = 2; cu.fill_mode = "FULL"
    sp = cu.splines.new("NURBS"); n = 64; sp.points.add(n - 1)
    for i in range(n):
        a = 2 * math.pi * i / n; sp.points[i].co = (r * math.cos(a), r * math.sin(a), z, 1.0)
    sp.use_cyclic_u = True; sp.order_u = 4; sp.resolution_u = 6
    o = add_obj(name, cu, mat, parent); return o
def text(name, body, size, mat, loc=(0, 0, 0), parent=None, align="CENTER"):
    cu = bpy.data.curves.new(name, "FONT"); cu.body = body; cu.size = size; cu.align_x = align
    o = add_obj(name, cu, mat, parent); o.location = loc; o.visible_shadow = False; return o        # labels cast no shadows
def empty(name, loc=(0, 0, 0), parent=None):
    o = bpy.data.objects.new(name, None); bpy.context.collection.objects.link(o); o.location = loc
    if parent is not None: o.parent = parent
    return o

def hull_mesh(name, loa, beam, depth=0.65, free=0.85, nS=22, nC=9):
    """the same loft as the WebGL page: stations along x, U-sections from the port sheer round the keel to starboard"""
    verts, faces = [], []
    for i in range(nS + 1):
        s = i / nS; x = -loa * 0.5 + loa * s
        hw = (beam / 2) * ((0.82 + 1.5 * s) if s < 0.12 else math.sin(math.pi * (0.5 + 0.5 * (1 - s))) ** 0.55 * (((1 - s) / 0.14) if s > 0.86 else 1.0))
        keel = -depth * (0.35 + 0.65 * math.sin(math.pi * min(1.0, s * 1.15)))
        for j in range(nC + 1):
            a = -math.pi / 2 + math.pi * (j / nC)
            zz = free if j in (0, nC) else keel + (free - keel) * abs(math.sin(a)) ** 1.4
            yy = math.sin(a) * hw * (1.0 if j in (0, nC) else math.cos(a) ** 0.15)
            verts.append((x, -yy, zz))
    for i in range(nS):
        for j in range(nC):
            a = i * (nC + 1) + j; b = a + nC + 1; faces.append((a, a + 1, b + 1, b))
    faces.append(tuple(range(nC, -1, -1)))                                   # transom
    me = bpy.data.meshes.new(name); me.from_pydata(verts, [], faces); me.update()
    for p in me.polygons: p.use_smooth = True
    return me

def boat_object(name, loa, beam, rgb, cabin=True, cabin_l=0.42, cabin_h=1.45, depth=0.65, free=0.85):
    root = empty(name)
    hull = add_obj(name + "_hull", hull_mesh(name + "_hullmesh", loa, beam, depth, free), material(name + "_m", rgb, 0.5), root)
    box(name + "_deck", loa * 0.94, beam * 0.9, 0.06, material(name + "_deck_m", (0.91, 0.93, 0.95), 0.8), (-loa * 0.02, 0, free), root)
    if cabin:
        cl, cw = loa * cabin_l, beam * 0.78
        box(name + "_cabin", cl, cw, cabin_h, material(name + "_cab_m", (0.96, 0.97, 0.98), 0.6), (-loa * 0.08, 0, free + cabin_h / 2), root)
        box(name + "_roof", cl * 1.35, cw * 1.2, 0.08, material(name + "_roof_m", (0.15, 0.21, 0.29), 0.25, 0.5), (-loa * 0.06, 0, free + cabin_h + 0.05), root)
    return root

# ----------------------------------------------------------------------------------------- fresh scene
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene; scene.render.fps = FPS
INK = (0.125, 0.188, 0.235)
world = bpy.data.worlds.new("sky"); scene.world = world; world.use_nodes = True
wn = world.node_tree; skytex = wn.nodes.new("ShaderNodeTexSky"); skytex.sky_type = "NISHITA"                 # a physical sky: the sun and the atmosphere light the scene
skytex.sun_elevation = math.radians(38.0); skytex.sun_rotation = math.radians(215.0); skytex.sun_intensity = 1.0; skytex.sun_size = math.radians(0.6)
skytex.air_density = 1.0; skytex.dust_density = 0.35; skytex.ozone_density = 1.3; skytex.altitude = 0.0
wn.links.new(skytex.outputs["Color"], wn.nodes["Background"].inputs["Color"]); wn.nodes["Background"].inputs["Strength"].default_value = 0.055
sun = bpy.data.lights.new("sun", "SUN"); sun.energy = 1.6; sun.angle = math.radians(1.2)                      # a lamp on top of the sky's sun for crisp shadows
sun_o = bpy.data.objects.new("sun", sun); bpy.context.collection.objects.link(sun_o)
sun_o.rotation_euler = (0.0, math.radians(90.0 - 38.0), math.radians(215.0))                                   # points along the sky's sun direction

# water: a big plane with a bump from noise; the surface texture drifts with the current (this is the unknown current)
cur = sc.get("current", [0.0, 0.0]); t_end = max(len(b["traj"]) - 1 for b in sc["boats"]); frame_end = int(t_end * FPS)
bpy.ops.mesh.primitive_plane_add(size=1600, location=(60, 0, 0)); water = bpy.context.active_object; water.name = "water"
wm = bpy.data.materials.new("water_m"); wm.use_nodes = True; nt = wm.node_tree; bsdf = nt.nodes["Principled BSDF"]
deep = srgb_to_linear((0.10, 0.27, 0.34))
bsdf.inputs["Base Color"].default_value = (*deep, 1.0); bsdf.inputs["Roughness"].default_value = 0.045; bsdf.inputs["Specular IOR Level"].default_value = 0.5; bsdf.inputs["Coat Weight"].default_value = 0.35
tc = nt.nodes.new("ShaderNodeTexCoord"); mp = nt.nodes.new("ShaderNodeMapping"); nz = nt.nodes.new("ShaderNodeTexNoise"); nz2 = nt.nodes.new("ShaderNodeTexNoise"); bump = nt.nodes.new("ShaderNodeBump"); bump2 = nt.nodes.new("ShaderNodeBump")
fl = nt.nodes.new("ShaderNodeTexNoise"); ramp = nt.nodes.new("ShaderNodeValToRGB"); mix = nt.nodes.new("ShaderNodeMix"); mix.data_type = "RGBA"
nt.links.new(tc.outputs["Object"], mp.inputs["Vector"])
nz.inputs["Scale"].default_value = 0.22; nz.inputs["Detail"].default_value = 9.0; nz.inputs["Roughness"].default_value = 0.62     # swell
nz2.inputs["Scale"].default_value = 2.4; nz2.inputs["Detail"].default_value = 6.0; nz2.inputs["Roughness"].default_value = 0.5     # ripples
nt.links.new(mp.outputs["Vector"], nz.inputs["Vector"]); nt.links.new(mp.outputs["Vector"], nz2.inputs["Vector"])
bump.inputs["Strength"].default_value = 0.22; nt.links.new(nz.outputs["Fac"], bump.inputs["Height"])
bump2.inputs["Strength"].default_value = 0.08; bump2.inputs["Distance"].default_value = 0.4; nt.links.new(nz2.outputs["Fac"], bump2.inputs["Height"]); nt.links.new(bump.outputs["Normal"], bump2.inputs["Normal"])
nt.links.new(bump2.outputs["Normal"], bsdf.inputs["Normal"])
fl.inputs["Scale"].default_value = 1.1; fl.inputs["Detail"].default_value = 2.0; nt.links.new(mp.outputs["Vector"], fl.inputs["Vector"])
ramp.color_ramp.elements[0].position = 0.70; ramp.color_ramp.elements[1].position = 0.76; nt.links.new(fl.outputs["Fac"], ramp.inputs["Fac"])
mix.inputs["A"].default_value = (*deep, 1.0); mix.inputs["B"].default_value = (0.85, 0.92, 0.95, 1.0); nt.links.new(ramp.outputs["Color"], mix.inputs["Factor"]); nt.links.new(mix.outputs["Result"], bsdf.inputs["Base Color"])
water.data.materials.append(wm)
mp.inputs["Location"].default_value = (0.0, 0.0, 0.0); mp.inputs["Location"].keyframe_insert("default_value", frame=1)
mp.inputs["Location"].default_value = (-cur[0] * t_end, -cur[1] * t_end, 0.0); mp.inputs["Location"].keyframe_insert("default_value", frame=frame_end)
for fc in nt.animation_data.action.fcurves:
    for kp in fc.keyframe_points: kp.interpolation = "LINEAR"

# goal
gx, gy = sc["goal"]; ring("goal_ring", sc["goal_r"], 0.35, material("goal_m", INK, 0.6), 0.04).location = (gx, gy, 0)
cylinder("goal_pole", 0.12, 6.0, material("pole_m", INK), (gx, gy, 3.0))
label_m = material("label_m", INK, 0.9, emit=0.4)
labels = [text("goal_lab", "goal", 1.6, label_m, (gx, gy, 7.0))]

# obstacles
obst_m = material("obst_m", (0.44, 0.50, 0.55), 0.9); dark_m = material("dark_m", (0.31, 0.37, 0.42), 1.0); buoy_m = material("buoy_m", (0.89, 0.60, 0.16), 0.6)
circle_m = material("circle_m", INK, 0.7, emit=0.15); circle_m.blend_method = "BLEND"; circle_m.node_tree.nodes["Principled BSDF"].inputs["Alpha"].default_value = 0.55
ferry_obj = None
for k, o in enumerate(sc["obstacles"]):
    r = max(o["R"] - sc["r_ego"], 1.0); root = empty(f"obs{k}", (o["c0"][0], o["c0"][1], 0))
    if o["kind"] == "barge":
        box("barge", r * 1.7, r * 0.9, 1.6, obst_m, (0, 0, 0.6), root, 0.35); box("barge_wh", r * 0.35, r * 0.5, 2.2, material("wh_m", (0.56, 0.63, 0.68)), (-r * 0.6, 0, 2.4), root, 0.35); box("barge_cargo", r * 1.0, r * 0.7, 1.4, dark_m, (r * 0.2, 0, 2.0), root, 0.35)
    elif o["kind"] == "moored":
        b = boat_object("moored", min(r * 1.6, 12), min(r * 0.5, 3.6), (0.54, 0.61, 0.66), cabin_l=0.35, cabin_h=1.2); b.parent = root; b.rotation_euler = (0, 0, -0.6)
        bpy.ops.mesh.primitive_uv_sphere_add(radius=0.45, location=(r * 0.9, -r * 0.4, 0.3)); s = bpy.context.active_object; s.data.materials.append(material("mbuoy_m", (0.88, 0.38, 0.16))); s.parent = root
    elif o["kind"] == "buoys":
        for i in range(7):
            a = 2 * math.pi * i / 7; cylinder(f"buoy{i}", 0.5, 1.3, buoy_m, (math.cos(a) * r * 0.75, math.sin(a) * r * 0.75, 0.55), root, 16)
    elif o["kind"] == "ferry":
        vx, vy = o["vel"]; b = boat_object("ferry", min(r * 1.8, 40), min(r * 0.45, 9), (0.95, 0.96, 0.97), cabin_l=0.6, cabin_h=3.2, depth=1.4, free=1.6)
        b.parent = root; b.rotation_euler = (0, 0, math.atan2(vy, vx))
        root.keyframe_insert("location", frame=1); root.location = (o["c0"][0] + vx * t_end, o["c0"][1] + vy * t_end, 0); root.keyframe_insert("location", frame=frame_end)
        for fc in root.animation_data.action.fcurves:
            for kp in fc.keyframe_points: kp.interpolation = "LINEAR"
        ferry_obj = root
    ring(f"excl{k}", o["R"], 0.3, circle_m, 0.05, root)
    labels.append(text(f"lab{k}", o["name"], 1.5, label_m, (0, 0, 4.5 + (4 if o["kind"] == "ferry" else 0)), root))

# boats: keyframed on the 1-s control states; paths grow with time; thrust arrow at the stern
def unwrap(angles):
    out = [angles[0]]
    for a in angles[1:]:
        d = (a - out[-1] + math.pi) % (2 * math.pi) - math.pi; out.append(out[-1] + d)
    return out
boats = []; COURSES = []
for i, b in enumerate(sc["boats"]):
    rgb = hexrgb(b["color"]); root = boat_object(f"boat{i}", META["loa"], META["beam"], rgb)
    psis = unwrap([s[2] for s in b["traj"]])
    for k, s in enumerate(b["traj"]):
        f = 1 + k * FPS; root.location = (s[0], s[1], 0.0); root.rotation_euler = (0, 0, psis[k])
        root.keyframe_insert("location", frame=f); root.keyframe_insert("rotation_euler", frame=f)
    # the path as a tube that draws itself
    cu = bpy.data.curves.new(f"path{i}", "CURVE"); cu.dimensions = "3D"; cu.bevel_depth = 0.22; cu.bevel_resolution = 3; cu.fill_mode = "FULL"
    cu.bevel_factor_mapping_start = cu.bevel_factor_mapping_end = "SPLINE"
    sp = cu.splines.new("POLY"); sp.points.add(len(b["traj"]) - 1)
    for k, s in enumerate(b["traj"]): sp.points[k].co = (s[0], s[1], 0.12, 1.0)
    cu.bevel_factor_end = 0.0; cu.keyframe_insert("bevel_factor_end", frame=1); cu.bevel_factor_end = 1.0; cu.keyframe_insert("bevel_factor_end", frame=1 + (len(b["traj"]) - 1) * FPS)
    for fc in cu.animation_data.action.fcurves:
        for kp in fc.keyframe_points: kp.interpolation = "LINEAR"
    add_obj(f"path{i}_o", cu, material(f"path{i}_m", rgb, 0.5, emit=0.6))
    # thrust: cone at the stern along the executed azimuth force, length 6 m at 700 N; the 700 N disk as a ring
    stern = empty(f"stern{i}", (-2.9, 0, 0.9), root); arrow_m = material(f"arrow{i}_m", rgb, 0.4, emit=0.8)
    bpy.ops.mesh.primitive_cone_add(radius1=0.45, radius2=0.0, depth=1.0, vertices=16); cone = bpy.context.active_object; cone.name = f"thrust{i}"; cone.parent = stern; cone.data.materials.append(arrow_m)
    cone.rotation_euler = (0, math.pi / 2, 0)                                           # cone axis (z) -> +x of the stern empty
    ring(f"disk{i}", 6.0, 0.16, arrow_m, 0.1, stern)
    for k, u in enumerate(b["ctrl"]):
        f = 1 + k * FPS; mag = math.hypot(u[0], u[1]); L = 6.0 * mag / META["f_at_max"]
        stern.rotation_euler = (0, 0, math.atan2(u[1], u[0])); stern.keyframe_insert("rotation_euler", frame=f)
        cone.scale = (1.0, 1.0, max(L, 0.05)); cone.location = (max(L, 0.05) / 2, 0, 0); cone.keyframe_insert("scale", frame=f); cone.keyframe_insert("location", frame=f)
    labels.append(text(f"boatlab{i}", b.get("short", b["label"]), 1.3, material(f"bl{i}_m", rgb, 0.9, emit=0.5), (0, 0, 4.6), root))
    labels.append(text(f"disklab{i}", "700 N", 0.9, material(f"dl{i}_m", rgb, 0.9, emit=0.5), (-6.4, 0, 0.5), stern))
    # the course over ground (with the current), keyframed per second: the chase camera follows it, so a boat running
    # astern is seen travelling stern-first instead of the camera looking away from where it goes
    course = empty(f"course{i}"); cur = sc.get("current", [0.0, 0.0]); ang = None; angs = []
    for k, s in enumerate(b["traj"]):
        vx = s[3] * math.cos(s[2]) - s[4] * math.sin(s[2]) + cur[0]; vy = s[3] * math.sin(s[2]) + s[4] * math.cos(s[2]) + cur[1]
        if math.hypot(vx, vy) > 0.3 or ang is None: ang = math.atan2(vy, vx)
        angs.append(ang)
    angs = unwrap(angs)
    for k, s in enumerate(b["traj"]):
        f = 1 + k * FPS; course.location = (s[0], s[1], 0.0); course.rotation_euler = (0, 0, angs[k])
        course.keyframe_insert("location", frame=f); course.keyframe_insert("rotation_euler", frame=f)
    boats.append(root); COURSES.append(course)

# camera
cam_data = bpy.data.cameras.new("cam"); cam_data.lens = 32; cam = bpy.data.objects.new("cam", cam_data); bpy.context.collection.objects.link(cam); scene.camera = cam
fb = boats[min(FOLLOW, len(boats) - 1)]; fcourse = COURSES[min(FOLLOW, len(boats) - 1)]
if CAM == "chase":                                                                 # behind the course over ground
    rigE = empty("cam_rig", (-28, 0, 10), fcourse); look = empty("cam_look", (18, 0, 1.5), fcourse)
    cam.parent = rigE; c = cam.constraints.new("TRACK_TO"); c.target = look; c.track_axis = "TRACK_NEGATIVE_Z"; c.up_axis = "UP_Y"
elif CAM == "helm":
    rigE = empty("cam_rig", (2.4, 0, 2.6), fb); look = empty("cam_look", (40, 0, 1.6), fb)
    cam.parent = rigE; c = cam.constraints.new("TRACK_TO"); c.target = look; c.track_axis = "TRACK_NEGATIVE_Z"; c.up_axis = "UP_Y"
elif CAM == "top":
    cam.location = (60, 0, 180); cam.rotation_euler = (0, 0, 0)
else:                                                                              # orbit: a fixed three-quarter view of the harbour
    cam.location = (-40, -110, 70); look = empty("cam_look", (60, 0, 0)); c = cam.constraints.new("TRACK_TO"); c.target = look; c.track_axis = "TRACK_NEGATIVE_Z"; c.up_axis = "UP_Y"
for lab in labels:                                                                 # labels face the camera
    c = lab.constraints.new("TRACK_TO"); c.target = cam; c.track_axis = "TRACK_Z"; c.up_axis = "UP_Y"

# HUD: the followed boat's readout, updated every frame from the data
hud = text("hud", "", 0.062, material("hud_m", INK, 1.0, emit=0.6), (-1.02, 0.54, -2.3), cam, align="LEFT")
bf = sc["boats"][min(FOLLOW, len(boats) - 1)]
def hud_update(scn):
    t = (scn.frame_current - 1) / FPS; k = min(int(t), len(bf["ctrl"]) - 1); n = len(bf["traj"]) - 1
    u = bf["ctrl"][k]; mag = math.hypot(u[0], u[1]); h = bf["h"][min(k, len(bf["h"]) - 1)]; hmin = min(bf["h"][:k + 1]); nin = sum(1 for v in bf["h"][:k + 1] if v < 0)
    ess = bf["ess"][min(k, len(bf["ess"]) - 1)]; surge = bf["traj"][min(k, n)][3]
    end = ("  ·  goal reached" if bf["reached"] else "  ·  time limit") if t >= n else ""
    hud.data.body = (f"{bf['label']}   t = {min(int(t), n):3d} s   surge {surge:+.1f} m/s{'  astern' if surge < -0.3 else ''}   azimuth {mag:3.0f} / {META['f_at_max']:.0f} N{'  (at limit)' if mag >= 0.98 * META['f_at_max'] else ''}   bow {u[2]:+.0f} N{end}\n"
                     f"h {h:+.1f} m   closest so far {hmin:+.1f} m   inside a circle {nin} s   ESS {ess:.0f} / 500")
bpy.app.handlers.frame_change_pre.append(hud_update)
st = sc.get("stats")
foot = text("foot", ("3-DOF planar model — no roll, pitch or heave · hull stylised to 8.5 × 2.2 m · positions interpolated between the 1-s control states · the boats run astern when that is cheaper: isotropic 700 N force input, symmetric damping, no heading cost\n"
                     + (f"over {st['n']} seeds: touched a circle {st['touched'][0]} / {st['touched'][1]} / {st['touched'][2]} · reached the goal {st['reached'][0]} / {st['reached'][1]} / {st['reached'][2]} (MPPI / barrier / + IS) · this is seed {sc['seed']}, not a selected one" if st else "")),
            0.03, material("foot_m", (0.36, 0.44, 0.50), 1.0, emit=0.5), (-1.02, -0.55, -2.3), cam, align="LEFT")

# render settings
w, h = (int(v) for v in RES.split("x")); scene.render.resolution_x = w; scene.render.resolution_y = h; scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "PNG"; scene.render.image_settings.color_mode = "RGB"
scene.frame_start = 1 + int(float(arg("--t0", 0)) * FPS); scene.frame_end = min(frame_end, 1 + int(float(arg("--t1", t_end)) * FPS))
if ENGINE == "cycles":                                                            # path tracing on the GPU: OptiX + OptiX denoiser
    scene.render.engine = "CYCLES"; scene.cycles.device = "GPU"; scene.cycles.samples = SAMPLES
    scene.cycles.use_adaptive_sampling = True; scene.cycles.adaptive_threshold = 0.03; scene.cycles.time_limit = float(arg("--time-limit", 3.0))
    scene.cycles.use_denoising = True; scene.cycles.denoiser = "OPTIX"; scene.cycles.denoising_use_gpu = True
    scene.cycles.max_bounces = 6; scene.cycles.caustics_reflective = False; scene.cycles.caustics_refractive = False
    scene.render.use_persistent_data = True                                        # keep the BVH between frames of the animation
    prefs = bpy.context.preferences.addons["cycles"].preferences; prefs.compute_device_type = "OPTIX"; prefs.get_devices()
    for d in prefs.devices: d.use = (d.type != "CPU")
else:
    scene.render.engine = "BLENDER_EEVEE_NEXT"; scene.eevee.taa_render_samples = SAMPLES; scene.eevee.use_shadows = True
scene.view_settings.view_transform = "AgX"; scene.view_settings.exposure = 0.3
try: scene.view_settings.look = "AgX - Medium High Contrast"
except Exception: pass
if TEST is not None:
    f = 1 + int(float(TEST) * FPS); scene.frame_set(f); scene.render.filepath = os.path.join(OUT, f"test_t{int(float(TEST))}.png")
    bpy.ops.render.render(write_still=True); print("TEST FRAME", scene.render.filepath)
else:
    scene.render.filepath = os.path.join(OUT, "frame_"); bpy.ops.render.render(animation=True); print("FRAMES", OUT, frame_end)
