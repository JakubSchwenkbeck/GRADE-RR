from isaacsim import SimulationApp
import argparse
import confuse
import numpy as np
import os
import sys

from replicator import ReplicatorPipeline

# Ensure 'simulator' directory is in sys.path for robust imports
simulator_dir = os.path.dirname(os.path.abspath(__file__))
if simulator_dir not in sys.path:
	sys.path.insert(0, simulator_dir)
import time
import traceback
from scipy.spatial.transform import Rotation
from time import sleep

# Example commands (from repo root):
# 1) Full data collection (headless):
#    conda activate env_isaaclab && python simulator/zebra_datagen.py \
#      --config_file simulator/configs/config_zebra_datagen.yaml --mode datagen --headless True --record True
# 2) Full data collection (windowed):
#    conda activate env_isaaclab && python simulator/zebra_datagen.py \
#      --config_file simulator/configs/config_zebra_datagen.yaml --mode datagen --headless False --record True
# 3) Interactive roaming preview (no recording):
#    conda activate env_isaaclab && python simulator/zebra_datagen.py \
#      --config_file simulator/configs/config_zebra_datagen.yaml --mode roam --headless False --record False
# 4) Fixed environment selection:
#    ... --fix_env Savana


def _patch_property_window_for_headless():
	"""Guard known property-window callback issue in headless Isaac Sim 5.1."""
	try:
		import omni.kit.window.property.window as _prop_window_mod
		_orig = _prop_window_mod.PropertyWindow.save_scroll_pos

		def _safe_save_scroll_pos(self, *args, **kwargs):
			frame = getattr(self, "properties_frame", None)
			if frame is None:
				return
			if getattr(frame, "scroll_y_max", None) is None or getattr(frame, "scroll_y", None) is None:
				return
			try:
				return _orig(self, *args, **kwargs)
			except AttributeError as e:
				if "scroll_y_max" in str(e):
					return
				raise

		_prop_window_mod.PropertyWindow.save_scroll_pos = _safe_save_scroll_pos
	except Exception:
		pass

def boolean_string(s):
	if s.lower() not in {'false', 'true'}:
		raise ValueError('Not a valid boolean string')
	return s.lower() == 'true'


def compute_points(skel_root_path, prim, ef, stage):
	usdSkelRoot = UsdSkel.Root.Get(stage, skel_root_path)
	UsdSkel.BakeSkinning(usdSkelRoot, Gf.Interval(0, ef))

	prim = UsdGeom.PointBased(prim)
	xformCache = UsdGeom.XformCache()
	final_points = np.zeros((ef, len(prim.GetPointsAttr().Get()), 3))

	for prim in Usd.PrimRange(usdSkelRoot.GetPrim()):
		if prim.GetTypeName() != "Mesh":
			continue
		localToWorld = xformCache.GetLocalToWorldTransform(prim)
		for time in range(ef):
			points = UsdGeom.Mesh(prim).GetPointsAttr().Get(time)
			for index in range(len(points)):
				points[index] = localToWorld.Transform(points[index])
			points = np.array(points)
			final_points[time] = points
	return final_points


def load_asset(asset_base_prim_path, n, asset_path):
	stage = omni.usd.get_context().get_stage()
	res, _ = omni.kit.commands.execute(
		"CreateReferenceCommand",
		usd_context=omni.usd.get_context(),
		path_to=f"{asset_base_prim_path}{n}",
		asset_path=asset_path,
		instanceable=False,
	)
	clear_properties(f"{asset_base_prim_path}{n}")
	return f"{asset_base_prim_path}{n}"


def find_mesh_offset_from_root(root_prim_path, stage):
	"""
	Find the first mesh in the hierarchy and compute its world-space offset from the root prim.
	Returns the offset as an array [x, y, z] in USD units.
	"""
	root_prim = stage.GetPrimAtPath(root_prim_path)
	if not root_prim.IsValid():
		return np.array([0.0, 0.0, 0.0])
	
	# Get root prim's transform
	xform_cache = UsdGeom.XformCache()
	root_to_world = xform_cache.GetLocalToWorldTransform(root_prim)
	world_to_root = root_to_world.GetInverse()
	
	# Find first mesh
	for prim in Usd.PrimRange(root_prim):
		if prim.GetTypeName() == "Mesh":
			# Get mesh world position
			mesh_to_world = xform_cache.GetLocalToWorldTransform(prim)
			mesh_pos_world = mesh_to_world.ExtractTranslation()
			
			# Transform to root-local coordinates
			mesh_pos_local = world_to_root.Transform(mesh_pos_world)
			
			print(f"[DEBUG] Found mesh at: {prim.GetPath()}")
			print(f"[DEBUG] Mesh world pos: {mesh_pos_world}, root-local: {mesh_pos_local}")
			return np.array([float(mesh_pos_local[0]), float(mesh_pos_local[1]), float(mesh_pos_local[2])])
	
	return np.array([0.0, 0.0, 0.0])


def cancel_mesh_offset(root_prim_path, stage, mesh_offset):
	"""
	Zero out all intermediate positions/rotations while preserving scales.
	This pulls the mesh to the root position without changing its size.
	"""
	root_prim = stage.GetPrimAtPath(root_prim_path)
	if not root_prim.IsValid():
		return
	
	# Walk the entire hierarchy and zero positions/rotations but keep scales
	for prim in Usd.PrimRange(root_prim):
		if prim.GetPath() == root_prim_path:
			# Skip the root itself
			continue
		
		prim_type = prim.GetTypeName()
		
		# Only modify non-mesh prims (Xform, Group, etc)
		if prim_type != "Mesh":
			# Zero out translate and rotate, but keep scale
			set_translate(prim, [0.0, 0.0, 0.0])
			set_rotate(prim, [0.0, 0.0, 0.0])
			# Don't modify scale - let it propagate naturally
			print(f"[DEBUG] Zeroed position/rotation on {prim.GetPath()}")
	
	print(f"[DEBUG] Hierarchy flattening complete for {root_prim_path}")


def _build_flight_mission_waypoints(ground_start, flat_floor_points, rng):
	"""Build a closed mission: ground -> takeoff -> cruise -> glide -> ground."""
	ground_start = np.array(ground_start, dtype=float)
	ground_start[2] += rng.uniform(0.2, 0.8)

	head = rng.uniform(0.0, 2.0 * np.pi)
	fwd = np.array([np.cos(head), np.sin(head), 0.0], dtype=float)

	takeoff = ground_start + fwd * rng.uniform(10.0, 18.0)
	takeoff[2] = ground_start[2] + rng.uniform(6.0, 10.0)

	cruise_1 = np.array(flat_floor_points[rng.integers(0, len(flat_floor_points))], dtype=float)
	cruise_2 = np.array(flat_floor_points[rng.integers(0, len(flat_floor_points))], dtype=float)
	cruise_1[2] = ground_start[2] + rng.uniform(14.0, 24.0)
	cruise_2[2] = ground_start[2] + rng.uniform(14.0, 24.0)

	glide = ground_start + fwd * rng.uniform(-8.0, 8.0)
	glide[2] = ground_start[2] + rng.uniform(4.0, 8.0)

	ground_end = ground_start + fwd * rng.uniform(1.0, 4.0)
	ground_end[2] = ground_start[2] + rng.uniform(0.1, 0.4)

	waypoints = [ground_start, takeoff, cruise_1, cruise_2, glide, ground_end]
	waypoints.append(waypoints[0].copy())
	return waypoints


def _sample_closed_polyline(waypoints, speed_mps, elapsed_s):
	"""Sample position and tangent on a closed polyline using arc-length parameterization."""
	if len(waypoints) < 2:
		return np.array(waypoints[0], dtype=float), np.array([1.0, 0.0, 0.0]), 0.0

	segments = []
	total_length = 0.0
	for i in range(len(waypoints) - 1):
		a = np.array(waypoints[i], dtype=float)
		b = np.array(waypoints[i + 1], dtype=float)
		d = b - a
		l = float(np.linalg.norm(d))
		if l > 1e-6:
			segments.append((a, b, d, l))
			total_length += l

	if total_length < 1e-6:
		return np.array(waypoints[0], dtype=float), np.array([1.0, 0.0, 0.0]), 0.0

	distance = (elapsed_s * max(0.1, speed_mps)) % total_length
	progress = distance / total_length
	for a, b, d, l in segments:
		if distance <= l:
			u = distance / l
			position = a + d * u
			tangent = d / l
			return position, tangent, progress
		distance -= l

	# fallback
	a, b, d, l = segments[-1]
	return np.array(b, dtype=float), d / l, 1.0


def _has_skeleton_animation(root_prim_path, stage):
	"""Best-effort check for skel animation prims inside the bird asset."""
	root_prim = stage.GetPrimAtPath(root_prim_path)
	if not root_prim.IsValid():
		return False
	for prim in Usd.PrimRange(root_prim):
		if prim.GetTypeName() in ("SkelAnimation", "Animation"):
			return True
	return False






def randomize_floor_position(floor_data, floor_translation, scale, meters_per_unit, env_name, rng):
	floor_points = np.zeros((len(floor_data), 3))
	if env_name == "Windmills":
		yaw = np.deg2rad(-155)
		rot = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
		floor_translation = np.matmul(floor_translation, rot)

	if env_name == "L_Terrain":
		meters_per_unit = 1

	for i in range(len(floor_data)):
		floor_points[i, 0] = floor_data[i][0] * scale[0] * meters_per_unit + floor_translation[0] * meters_per_unit
		floor_points[i, 1] = floor_data[i][1] * scale[1] * meters_per_unit + floor_translation[1] * meters_per_unit
		floor_points[i, 2] = floor_data[i][2] * scale[2] * meters_per_unit + floor_translation[2] * meters_per_unit

	if env_name == "L_Terrain":
		meters_per_unit = 0.01

	max_floor_x = max(floor_points[:, 0])
	min_floor_x = min(floor_points[:, 0])
	max_floor_y = max(floor_points[:, 1])
	min_floor_y = min(floor_points[:, 1])

	if env_name == "Windmills":
		min_floor_x = -112
		max_floor_x = 161
		min_floor_y = -209
		max_floor_y = 63
		rows = np.where((floor_points[:, 0] > min_floor_x) & (floor_points[:, 0] < max_floor_x) & (floor_points[:, 1] > min_floor_y) & (floor_points[:, 1] < max_floor_y))[0]
		floor_points = floor_points[rows]

	rows = []

	while (len(rows) == 0):
		size_x = rng.integers(40, 120)
		size_y = rng.integers(40, 120)

		# get all floor_points within a size x size square randomly centered
		min_x = rng.uniform(min(floor_points[:, 0]), max(floor_points[:, 0]))
		max_x = min_x + min(size_x, max(floor_points[:, 0]) - min(floor_points[:, 0]))
		while max_x > max(floor_points[:, 0]):
			min_x = rng.uniform(min(floor_points[:, 0]), max(floor_points[:, 0]))
			max_x = min_x + min(size_x, max(floor_points[:, 0]) - min(floor_points[:, 0]))

		min_y = rng.uniform(min(floor_points[:, 1]), max(floor_points[:, 1]))
		max_y = min_y + min(size_y, max(floor_points[:, 1]) - min(floor_points[:, 1]))
		while max_y > max(floor_points[:, 1]):
			min_y = rng.uniform(min(floor_points[:, 1]), max(floor_points[:, 1]))
			max_y = min_y + min(size_y, max(floor_points[:, 1]) - min(floor_points[:, 1]))
		# FIXME this is just an approximation which MAY NOT WORK ALWAYS!
		rows = np.where((min_x <= floor_points[:,0]) & (floor_points[:,0] <= max_x) & (floor_points[:,1]<=max_y) & (floor_points[:,1]>= min_y))[0]
	floor_points = floor_points[rows]

	shape = (len(np.unique(floor_points[:, 0])), -1, 3)
	floor_points = floor_points.reshape(shape)
	if (floor_points[0, 1, 0] - floor_points[0, 0, 0]) > 1:
		zoom_factor = int(floor_points[0, 1, 0] - floor_points[0, 0, 0])
		import scipy.ndimage.interpolation as interpolation

		floor_points = interpolation.zoom(floor_points, (zoom_factor, zoom_factor, 1))

	return floor_points, max_floor_x, min_floor_x, max_floor_y, min_floor_y



try:
	parser = argparse.ArgumentParser(description="Dynamic Worlds Simulator")
	parser.add_argument("--config_file", type=str, default="config.yaml")
	parser.add_argument("--headless", type=boolean_string, default=True, help="Wheter to run it in headless mode or not")
	parser.add_argument("--rtx_mode", type=boolean_string, default=True,
						help="Use rtx when True, use path tracing when False")
	parser.add_argument("--record", type=boolean_string, default=False, help="Writing data to the disk")
	parser.add_argument("--debug_vis", type=boolean_string, default=False,
						help="When true continuosly loop the rendering")
	parser.add_argument("--neverending", type=boolean_string, default=False, help="Never stop the main loop")
	parser.add_argument("--mode", type=str, default="roam", choices=["roam", "datagen"], help="Execution mode")
	parser.add_argument("--fix_env", type=str, default="",
						help="leave it empty to have a random env, fix it to use a fixed one. Useful for loop processing")

	args, unknown = parser.parse_known_args()
	cli_argv = set(sys.argv[1:])

	config = confuse.Configuration("DynamicWorlds", __name__)
	config.set_file(args.config_file)
	# Avoid overriding file values with parser defaults when a flag was not explicitly passed.
	# In particular, --fix_env default "" would mask config-file fix_env.
	if "--fix_env" not in cli_argv:
		try:
			delattr(args, "fix_env")
		except Exception:
			pass
	config.set_args(args)
	can_start = True
	interactive_preview_mode = args.mode == "roam"
	preview_settings_rate = 60 if interactive_preview_mode else config["physics_hz"].get()
	preview_stage_renders = 5 if interactive_preview_mode else 50
	preview_robot_renders = 1 if interactive_preview_mode else 5
	preview_substep = 1 if interactive_preview_mode else 3
	preview_inner_renders = 1 if interactive_preview_mode else 3
	preview_sleep_s = 0.0 if interactive_preview_mode else 0.5

	CONFIG = {"display_options": 3286, "width": 1280, "height": 720, "headless": config["headless"].get()}
	simulation_app = SimulationApp(launch_config=CONFIG)
	kit = simulation_app
	if interactive_preview_mode:
		print("roam mode enabled: reduced render load and no data capture")

	import carb
	import omni
	import omni.client
	if config["headless"].get():
		_patch_property_window_for_headless()
	cloud_path = "http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1"
	omni.client.set_alias("omniverse://localhost", cloud_path)
	omni.client.set_alias("omniverse://localhost/NVIDIA", f"{cloud_path}/NVIDIA")
	if hasattr(omni.client, "mount"):
		try:
			omni.client.mount(
				"omniverse://localhost",
				cloud_path,
			)
			omni.client.mount(
				"omniverse://localhost/NVIDIA",
				f"{cloud_path}/NVIDIA",
			)
		except Exception as e:
			carb.log_warn(f"omni.client.mount unavailable/failed, relying on set_alias only: {e}")

	settings = carb.settings.get_settings()
	mdl_paths = settings.get("/rtx/materialDb/mdlSearchPaths") or []
	if cloud_path not in mdl_paths:
		mdl_paths.append(cloud_path)
	if f"{cloud_path}/NVIDIA/Materials" not in mdl_paths:
		mdl_paths.append(f"{cloud_path}/NVIDIA/Materials")
	if f"{cloud_path}/NVIDIA/Assets/Skies" not in mdl_paths:
		mdl_paths.append(f"{cloud_path}/NVIDIA/Assets/Skies")
	settings.set("/rtx/materialDb/mdlSearchPaths", mdl_paths)

	# Cannot move before SimApp is launched
	import grade_utils.misc_utils
	from grade_utils.misc_utils import *
	from grade_utils.robot_utils import *
	from grade_utils.simulation_utils import *
	from grade_utils.environment_utils import *
	from pxr import UsdGeom, UsdLux, Gf, Vt, UsdPhysics, PhysxSchema, Usd, UsdShade, Sdf, UsdSkel

	def _focus_editor_camera_on_target(target_pos_m):
		"""Best-effort: move default editor camera to look at a target in meters."""
		camera_candidates = ["/OmniverseKit_Persp", "/World/Camera", "/OmniverseKit_Top"]
		cam_prim = None
		for p in camera_candidates:
			prim = stage.GetPrimAtPath(p)
			if prim and prim.IsValid():
				cam_prim = prim
				break
		if cam_prim is None:
			return

		tx = float(target_pos_m[0]) / meters_per_unit
		ty = float(target_pos_m[1]) / meters_per_unit
		tz = float(target_pos_m[2]) / meters_per_unit
		cam_pos = np.array([tx - (8.0 / meters_per_unit), ty - (8.0 / meters_per_unit), tz + (4.0 / meters_per_unit)])

		delta_x = tx - cam_pos[0]
		delta_y = ty - cam_pos[1]
		delta_z = tz - cam_pos[2]
		yaw = np.arctan2(delta_y, delta_x)
		pitch = -np.arctan2(delta_z, max(1e-6, np.sqrt(delta_x * delta_x + delta_y * delta_y)))

		set_translate(cam_prim, [float(cam_pos[0]), float(cam_pos[1]), float(cam_pos[2])])
		set_rotate(cam_prim, [0.0, float(pitch), float(yaw)])

	# Defer extension setup until after stage load to avoid loader-time instability.
	# (Some optional extensions can interfere with stage reopen/load in Isaac Sim 5.1.)
	deferred_extension_setup = (False, True, False)

	all_env_names = ["Bliss", "Forest", "Grasslands", "Iceland", "L_Terrain", "Meadow",
	                 "Moorlands", "Nature_1", 'Nature_2', "Savana", "Windmills", "Woodland"]
	ground_area_name = ["Landscape_1", "Landscape_1", "Landscape_1", "Landscape_0", "Terrain_5", "Landscape_0",
	                    "Landscape_2", "Ground", "Ground", "Landscape_1", "Landscape_0", "Landscape_1"]

	need_sky = [True] * len(all_env_names)
	fix_env_value = config["fix_env"].get()
	if fix_env_value and fix_env_value in all_env_names:
		env_id = all_env_names.index(fix_env_value)
	else:
		# fallback: pick a random environment if not specified or invalid
		import random
		env_id = random.randint(0, len(all_env_names) - 1)
		print(f"[INFO] fix_env not set or invalid, using random environment: {all_env_names[env_id]}")

	rng = np.random.default_rng()
	rng_state = np.random.get_state()

	local_file_prefix = ""

	# setup environment variables
	environment = environment(config, rng, local_file_prefix)
	base_env_name = os.path.splitext(os.path.basename(config["base_env_path"].get()))[0]
	if base_env_name in all_env_names:
		env_id = all_env_names.index(base_env_name)
		print(f"[INFO] Using base_env_path environment '{base_env_name}' (env_id={env_id})")
	elif environment.env_name in all_env_names:
		env_id = all_env_names.index(environment.env_name)
		print(f"[INFO] Using loaded environment '{environment.env_name}' (env_id={env_id})")
	else:
		print(
			f"[WARN] Loaded environment '{environment.env_name}' not in known list; "
			f"keeping env_id={env_id} ({all_env_names[env_id]})"
		)
	requested_env = config["fix_env"].get()
	if requested_env and environment.env_name != requested_env:
		env_root = os.path.abspath(config["env_path"].get())
		available_envs = []
		if os.path.isdir(env_root):
			available_envs = sorted([os.path.splitext(f)[0] for f in os.listdir(env_root) if f.endswith(".usd")])
		raise FileNotFoundError(
			f"Requested --fix_env={requested_env} was not found in env_path={env_root}. "
			f"Loaded '{environment.env_name}' instead. Available .usd environments: {available_envs}"
		)

	out_dir = os.path.join(config['out_folder'].get(), environment.env_name)
	out_dir_npy = os.path.join(config['out_folder_npy'].get(), environment.env_name)
	if not os.path.exists(out_dir):
		os.makedirs(out_dir)

	omni.usd.get_context().open_stage(local_file_prefix + config["base_env_path"].get(), None)

	for _ in range(10):
		kit.update()

	print("Loading stage...")
	load_timeout_secs = 120
	start_time = time.time()
	while is_stage_loading() and (time.time() - start_time) < load_timeout_secs:
		time.sleep(0.1)
		try:
			kit.update()
		except Exception:
			break
	print(f"Loading Complete (elapsed: {time.time() - start_time:.1f}s)")

	context = omni.usd.get_context()
	stage = context.get_stage()
	set_stage_up_axis("Z")

	# Enable optional extensions only after stage is loaded.
	need_ros_ext, need_seq_ext, need_shapenet_ext = deferred_extension_setup
	simulation_environment_setup(
		need_ros=need_ros_ext,
		need_sequencer=need_seq_ext,
		need_shapenet=need_shapenet_ext,
	)

	if stage.GetPrimAtPath("/World/GroundPlane").IsValid():
		omni.kit.commands.execute("DeletePrimsCommand", paths=["/World/GroundPlane"])

	# do this AFTER loading the world
	simulation_context = SimulationContext(physics_dt=1.0 / config["physics_hz"].get(),
											rendering_dt=1.0 / config["render_hz"].get(),
											stage_units_in_meters=0.01)
	simulation_context.initialize_physics()

	simulation_context.play()
	try:
		simulation_context.stop()
	except NameError:
		print("[ERROR] simulation_context is not defined at shutdown.")

	kit.update()
	meters_per_unit = 0.01

	# Configure rendering defaults based on operating mode
	settings = carb.settings.get_settings()
	if interactive_preview_mode:
		settings.set("/rtx/pathtracing/spp", 1)
		settings.set("/rtx/directLighting/samplesPerPixel", 1)
		settings.set("/rtx/raytracing/enabled", False)

	env_prim_path = environment.load_and_center(config["env_prim_path"].get())
	process_semantics(config["env_prim_path"].get(), "World")

	if all_env_names[env_id] == "L_Terrain":
		set_scale(stage.GetPrimAtPath(f"/World/home"), 100)

	while is_stage_loading():
		kit.update()

	floor_data = stage.GetPrimAtPath(f"/World/home/{ground_area_name[env_id]}/{ground_area_name[env_id]}").GetProperty(
		'points').Get()
	floor_translation = np.array(stage.GetPrimAtPath(f"/World/home/{ground_area_name[env_id]}").GetProperty(
		'xformOp:translate').Get())
	scale = np.array(stage.GetPrimAtPath(f"/World/home/{ground_area_name[env_id]}").GetProperty("xformOp:scale").Get())

	for _ in range(preview_stage_renders):
		simulation_context.render()

	floor_points, max_floor_x, min_floor_x, max_floor_y, min_floor_y = randomize_floor_position(floor_data,
																								floor_translation, scale,
																								meters_per_unit, all_env_names[env_id], rng)

	add_semantics(stage.GetPrimAtPath("/World/home"), "world")

	bird_assets_root = os.path.abspath(os.path.join(simulator_dir, os.pardir, "usds", "packed_blends"))
	bird_assets = [
		{"name": "honey_buzzard", "path": os.path.join(bird_assets_root, "honey_buzzard", "honey_buzzard.usdz")},
		{"name": "red_kite", "path": os.path.join(bird_assets_root, "red_kite", "red_kite.usdz")},
	]
	for bird_asset in bird_assets:
		if not os.path.exists(bird_asset["path"]):
			raise FileNotFoundError(f"Missing bird asset: {bird_asset['path']}")

	print(f"Loading birds for {args.mode} mode..")
	bird_prim_paths = []
	for index, bird_asset in enumerate(bird_assets):
		bird_path = load_asset("/bird_", index, bird_asset["path"])
		bird_prim_paths.append(bird_path)
		add_semantics(stage.GetPrimAtPath(bird_path), "bird")
		set_scale(stage.GetPrimAtPath(bird_path), 0.05)
		kit.update()

	if interactive_preview_mode:
		settings.set("/rtx/raytracing/enabled", False)
		settings.set("/rtx/pathtracing/enabled", False)

	for _ in range(5):
		simulation_context.step(render=False)
		sleeping(simulation_context, [], False)

	flat_floor_points = floor_points.reshape(-1, 3)
	chosen_positions = []
	for bird_index in range(len(bird_prim_paths)):
		for _ in range(100):
			candidate = flat_floor_points[rng.integers(0, len(flat_floor_points))]
			candidate = np.array(candidate, dtype=float)
			candidate[2] += rng.uniform(0.2, 0.8)
			if len(chosen_positions) == 0:
				chosen_positions.append(candidate)
				break
			if np.linalg.norm(candidate[:2] - chosen_positions[0][:2]) > 5.0:
				chosen_positions.append(candidate)
				break
		else:
			chosen_positions.append(np.array(flat_floor_points[rng.integers(0, len(flat_floor_points))], dtype=float))

	bird_motion_state = []
	for bird_path, bird_position in zip(bird_prim_paths, chosen_positions):
		yaw = rng.uniform(-np.pi, np.pi)
		mesh_offset = find_mesh_offset_from_root(bird_path, stage)
		print(f"[DEBUG] Bird {bird_path}: computed mesh offset = {mesh_offset}")
		
		cancel_mesh_offset(bird_path, stage, mesh_offset)
		
		compensated_position = bird_position / meters_per_unit
		set_translate(stage.GetPrimAtPath(bird_path), list(compensated_position))
		set_rotate(stage.GetPrimAtPath(bird_path), [0.0, 0.0, float(yaw)])
		print(f"[DEBUG] Bird at: base={bird_position}, usd={bird_position / meters_per_unit}, compensated={compensated_position}")
		flight_waypoints = _build_flight_mission_waypoints(bird_position, flat_floor_points, rng)
		has_anim = _has_skeleton_animation(bird_path, stage)
		bird_motion_state.append(
			{
				"path": bird_path,
				"base_position": bird_position.copy(),
				"waypoints": flight_waypoints,
				"speed": rng.uniform(5.0, 9.0),
				"phase": rng.uniform(0.0, 2.0 * np.pi),
				"offset": np.array([0.0, 0.0, 0.0]),
				"has_animation": has_anim,
				"yaw_offset": np.pi / 2.0,
			}
		)

	if not config["headless"].get() and len(chosen_positions) > 0:
		try:
			_focus_editor_camera_on_target(chosen_positions[0])
		except Exception:
			pass
	# Initialize the modular Replicator pipeline
	rep_pipeline = ReplicatorPipeline(config, out_dir)
	if not interactive_preview_mode:
		rep_pipeline.setup_data_writer()

	print(f"Birds loaded; entering main execution loop in [{args.mode.upper()}] mode.")
	simulation_context.play()
	try:
		import omni.timeline
		timeline = omni.timeline.get_timeline_interface()
		timeline.play()
	except Exception:
		timeline = None

	simulation_step = 0
	roam_start = time.time()
	exp_len = config.get("experiment_length", None) or 100

	while kit.is_running():
		# Handle time mapping dynamically based on runtime mode
		if interactive_preview_mode:
			elapsed = time.time() - roam_start
		else:
			elapsed = simulation_step * (1.0 / config["render_hz"].get())

		if timeline is not None:
			timeline.set_current_time(elapsed)

		target_bird_pos_m = None
		for index, motion in enumerate(bird_motion_state):
			t = elapsed + motion["phase"]
			new_position, tangent, progress = _sample_closed_polyline(motion["waypoints"], motion["speed"], t)

			if progress < 0.22:
				flap_amp, flap_freq = 0.55, 8.0   
			elif progress < 0.62:
				flap_amp, flap_freq = 0.30, 5.5   
			elif progress < 0.84:
				flap_amp, flap_freq = 0.10, 2.0   
			else:
				flap_amp, flap_freq = 0.45, 7.0   

			if motion.get("has_animation", False):
				flap_amp *= 0.55

			new_position[2] += np.sin(t * flap_freq) * flap_amp

			yaw = np.arctan2(tangent[1], tangent[0]) + motion.get("yaw_offset", 0.0)
			horizontal_speed = np.sqrt(max(1e-6, tangent[0] ** 2 + tangent[1] ** 2))
			pitch = -np.arctan2(tangent[2], horizontal_speed)
			roll = np.sin(t * max(2.0, flap_freq * 0.6)) * (0.06 + 0.10 * flap_amp)
			
			compensated = new_position / meters_per_unit
			set_translate(stage.GetPrimAtPath(motion["path"]), list(compensated))
			set_rotate(stage.GetPrimAtPath(motion["path"]), [float(roll), float(pitch), float(yaw)])

			if index == 0:
				target_bird_pos_m = new_position

		# =================================================================
		# BRANCH A: INTERACTIVE PREVIEW MODE
		# =================================================================
		if interactive_preview_mode:
			simulation_context.step(render=False)
			simulation_context.render()
			continue

		# =================================================================
		# BRANCH B: DATASET GENERATION MODE (DATAGEN)
		# =================================================================
		rep_pipeline.update_tracking_cameras(target_bird_pos_m)
		rep_pipeline.assign_metadata(simulation_step, target_bird_pos_m)

		# Triggers internal subframe loops, clearing out raytracing darkness 
		simulation_context.step(render=True)
		simulation_step += 1

		if simulation_step % 10 == 0:
			print(f"[INFO] Sequenced and generated frame {simulation_step}/{exp_len}")

		if exp_len is not None and simulation_step >= exp_len:
			print(f"[INFO] Requested frame constraint reached ({exp_len}). Terminating.")
			break
except:
	traceback.print_exc()
	raise
finally:
	if 'simulation_context' in globals() or 'simulation_context' in locals():
		try:
			simulation_context.stop()
		except Exception:
			pass
	try:
		kit.close()
	except:
		pass