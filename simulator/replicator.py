import os
import numpy as np

class ReplicatorPipeline:
    def __init__(self, config, out_dir):
        # ==========================================
        # DEFERRED IMPORTS
        # These load safely because the class is instantiated 
        # AFTER SimulationApp starts in the main file.
        # ==========================================
        import carb
        import omni.replicator.core as rep
        import omni.kit.viewport.window as wp
        from isaacsim.core.utils.viewports import set_camera_view
        
        # Save references to these modules so you can use them across all methods
        self.carb = carb
        self.rep = rep
        self.wp = wp
        self.set_camera_view = set_camera_view

        self.config = config
        self.out_dir = out_dir
        self.writer = None
        self.meters_per_unit = 0.01
        
        # 1. Enforce high-quality render rules globally at initialization
        self._configure_render_backend()

    def _configure_render_backend(self):
        """Optimizes the path tracer to clear noise and build rays before saving."""
        settings = self.carb.settings.get_settings()
        if not self.config["headless"].get():
            # Ensure path-tracing settings don't run flat on standard interactive viewport
            settings.set("/rtx/pathtracing/enabled", not self.config["rtx_mode"].get())
        
        # FIX: Force internal accumulation loops.
        settings.set("/omni/replicator/RTSubframes", 16)
        settings.set("/rtx/pathtracing/denoiser/enabled", True)
        print("[REPLICATOR] Render engine optimization configured (16 Subframes).")

    def setup_data_writer(self):
        """Registers the Replicator BasicWriter and links active viewports using Render Products."""
        if not self.config["record"].get():
            print("[REPLICATOR] Record flag is False. Skipping writer instantiation.")
            return

        # Initialize the baseline Omniverse writer
        self.writer = self.rep.WriterRegistry.get("BasicWriter")
        self.writer.initialize(
            output_dir=self.out_dir, 
            rgb=True, 
            semantic_segmentation=True
        )
        
        # FIX: Convert the generator into a list to prevent "TypeError: object of type 'generator' has no len()"
        viewport_windows = list(self.wp.get_viewport_window_instances())
        render_products = []
        print(f"[INFO] Detected {len(viewport_windows)} active viewports.")
        
        for index, cam in enumerate(viewport_windows):
            try:
                # Extract the absolute USD path of the viewport's camera
                cam_path = str(cam.get_active_camera()) if hasattr(cam, "get_active_camera") else str(cam)
                print(f"[INFO] Initializing discrete render product for camera: {cam_path}")
                
                # Force Replicator to create a fully tracked render product container
                rp_handle = self.rep.create.render_product(cam_path, (1280, 720), name=f"zebra_cam_{index}")
                render_products.append(rp_handle)
            except Exception as e:
                print(f"[WARN] Skipping viewport item {cam} due to error: {e}")

        # GUARD: Fallback if no active cameras were resolved (very common in headless setup environments)
        if not render_products:
            print("[WARN] render_products list is empty! Falling back to default editor camera.")
            fallback_rp = self.rep.create.render_product("/OmniverseKit_Persp", (1280, 720), name="zebra_fallback_cam")
            render_products.append(fallback_rp)
            
        # CRITICAL STEP: Spin the application lifecycle once. 
        import omni.kit.app
        omni.kit.app.get_app().update()
        
        # Attach the live, hydrated handles directly to the writer
        print(f"[INFO] Attaching {len(render_products)} hydrated render products to BasicWriter.")
        self.writer.attach(render_products)
        print(f"[REPLICATOR] Writer attached successfully.")
    def update_tracking_cameras(self, target_pos_m, viewport_window_list):
            """Calculates dynamic staggered flight views centered on the moving asset.
            
            Matches the layout logic used in the working monolithic setup block.
            """
            if target_pos_m is None:
                return
                
            # Convert the viewport generator/iterable explicitly to a stable list
            viewport_windows = list(viewport_window_list)
            num_cameras = len(viewport_windows)
            if num_cameras == 0:
                return

            from isaacsim.core.utils.viewports import set_camera_view

            # Establish focus point at the center of the active animal's torso (+0.75m elevation)
            zebra_torso_m = target_pos_m + np.array([0.0, 0.0, 0.75])

            for index, cam in enumerate(viewport_windows):
                # Safe extraction of absolute USD path
                cam_path = str(cam.get_active_camera()) if hasattr(cam, "get_active_camera") else str(cam)
                
                # Side profile layout calculation arrays matching the baseline script exactly
                stagger_x_m = (index - (num_cameras - 1) / 2) * 0.6  
                radius_y_m = 6.0                                    
                elevation_z_m = 0.6                                 
                
                cam_pos_m = zebra_torso_m + np.array([stagger_x_m, radius_y_m, elevation_z_m])
                
                eye_units = cam_pos_m / self.meters_per_unit
                target_units = zebra_torso_m / self.meters_per_unit
                
                try:
                    set_camera_view(
                        eye=eye_units,
                        target=target_units,
                        camera_prim_path=cam_path
                    )
                except Exception as e:
                    print(f"[WARNING] Failed setting view for pipeline camera {cam_path}: {e}")
    def assign_metadata(self, simulation_step, target_pos_m=None):
        """Passes tracking vectors to custom dictionary metadata arrays."""
        if self.writer is None:
            return
            
        self.writer.current_frame_info = {
            "step": simulation_step,
            "target_position": target_pos_m.tolist() if target_pos_m is not None else []
        }