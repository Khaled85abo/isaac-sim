import math
import carb
import gymnasium
import numpy as np
from gymnasium import spaces


    

class JetBotEnv(gymnasium.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        skip_frame=1,
        physics_dt=1.0 / 60.0,
        rendering_dt=1.0 / 60.0,
        # HUR MÅNGA STEG MAX EN EPISODE FÅR VARA (256 default)
        max_episode_length=512, 
        seed=0,
        headless=True,
    ) -> None:
        from omni.isaac.kit import SimulationApp

        # Livestream configuration for SimulationApp
        CONFIG = {
            "width": 1280,
            "height": 720,
            "window_width": 1920,
            "window_height": 1080,
            "headless": headless,  # Use the headless parameter passed to the constructor
            "hide_ui": False,  # Show the GUI, useful for livestreaming
            "renderer": "RayTracedLighting",
            "display_options": 3286,  # Set display options to show the default grid
        }

        # Initialize the SimulationApp with the livestream settings
        self._simulation_app = SimulationApp(launch_config=CONFIG)
        self.cubes = []

        from pxr import Usd, UsdGeom 
        
        # Now import the enable_extension function after initializing SimulationApp
        from omni.isaac.core.utils.extensions import enable_extension
    

        # Enable Livestream extensions
        self._simulation_app.set_setting("/app/window/drawMouse", True)
        self._simulation_app.set_setting("/app/livestream/proto", "ws")
        self._simulation_app.set_setting("/ngx/enabled", False)

        enable_extension("omni.kit.streamsdk.plugins-3.2.1")
        enable_extension("omni.kit.livestream.core-3.2.0")
        enable_extension("omni.kit.livestream.native")

        # Rest of the environment setup
        self._skip_frame = skip_frame
        self._dt = physics_dt * self._skip_frame
        self._max_episode_length = max_episode_length
        self._steps_after_reset = int(rendering_dt / physics_dt)

        from omni.isaac.core import World
        from omni.isaac.core.objects import VisualCuboid
        from omni.isaac.core.utils.nucleus import get_assets_root_path
        from omni.isaac.wheeled_robots.controllers.differential_controller import DifferentialController
        from omni.isaac.wheeled_robots.robots import WheeledRobot

        from omni.isaac.core.utils.stage import add_reference_to_stage
        from omni.isaac.core.prims import XFormPrim

        self._my_world = World(physics_dt=physics_dt, rendering_dt=rendering_dt, stage_units_in_meters=1.0)
        self._my_world.scene.add_default_ground_plane()
        
        
         
        assets_root_path = get_assets_root_path()
        if assets_root_path is None:
            carb.log_error("Could not find Isaac Sim assets folder")
            return
        jetbot_asset_path = assets_root_path + "/Isaac/Robots/Jetbot/jetbot.usd"
        self.jetbot = self._my_world.scene.add(
            WheeledRobot(
                prim_path="/jetbot",
                name="my_jetbot",
                wheel_dof_names=["left_wheel_joint", "right_wheel_joint"],
                create_robot=True,
                usd_path=jetbot_asset_path,
                position=np.array([0.6, 0.26, 0.03]),
                orientation=np.array([1.0, 0.0, 0.0, 0.0]),
            )
        )
        self.jetbot_controller = DifferentialController(name="simple_control", wheel_radius=0.0325, wheel_base=0.1125)
        self.goal = self._my_world.scene.add(
            VisualCuboid(
                prim_path="/new_cube_1",
                name="visual_cube",
                position=np.array([3.6, 0.26, 0.05]),
                size=0.1,
                color=np.array([0, 1.0, 0]),
            )
        )
        
        # new - add a wall
        self.obstacle = self. _my_world.scene.add(
            VisualCuboid(
                prim_path="/obstacle_cube",
                name="obstacle_cube",
                position=np.array([-0.29, 0.23, 0.05]),
                size=0.5,
                color=np.array([1.0, 0, 0]),
                # size=np.array([2.0, 0.1, 0.5]),  # Length, Width, Height
            )
        )
        
        # center_x, center_y = 0.0, 0.0  # Center of the circle
        # radius = 3.0  # Radius of the circle
        # num_cubes = int(2 * np.pi * radius)  # Approximate with one cube per unit of circumference

        # for i in range(num_cubes):
        #     theta = (2 * np.pi / num_cubes) * i
        #     x = center_x + radius * np.cos(theta)
        #     y = center_y + radius * np.sin(theta)
            
        #     # Calculate rotation to align cube edge along the circle radius
        #     rotation = -np.degrees(theta)  # Negative to align with the radial line
            
        #     # Add each cuboid to the scene with a fixed size of 1.0
        #     self._my_world.scene.add(
        #         VisualCuboid(
        #             prim_path=f"/obstacle_wall_{i}",
        #             name=f"obstacle_wall_{i}",
        #             position=np.array([x, y, 0.05]),  # Adjusted for correct placement
        #             size=0.1,  # Fixed size
        #             color=np.array([1.0, 0, 0]),
        #             orientation=np.array([0, 0, np.sin(rotation/2), np.cos(rotation/2)])  # Quaternion for rotation about Z-axis
        #         )
        #     )

        # Racetrack configuration
        num_cubes_per_row = 25
        cube_size = 0.1
        distance_between_cubes = 0.15
        row_distance = 0.5

        # Create the first row of cubes
        for i in range(num_cubes_per_row):
            x_position = i * distance_between_cubes
            cube = VisualCuboid(
                prim_path=f"/Cube_Row1_{i}",
                name=f"cube_row1_{i}",
                position=np.array([x_position, 0, cube_size / 2]),
                size=cube_size,
                color=np.array([1.0, 0, 0]),  # Red color
            )
            self._my_world.scene.add(cube)
            self.cubes.append(cube)

        # Create the second row of cubes
        for i in range(num_cubes_per_row):
            x_position = i * distance_between_cubes
            cube = VisualCuboid(
                prim_path=f"/Cube_Row2_{i}",
                name=f"cube_row2_{i}",
                position=np.array([x_position, row_distance, cube_size / 2]),
                size=cube_size,
                color=np.array([1.0, 0, 0]),  # Red color
            )
            self._my_world.scene.add(cube)
            self.cubes.append(cube)
                    
        # # Import the racetrack using the correct method
        # racetrack_usd_path = "/isaac-sim/racetrack_assets/racetrack.usd"
        # add_reference_to_stage(racetrack_usd_path, prim_path="/racetrack")
   
        # self.race_track_road = self._my_world.scene.add(
        #     XFormPrim(
        #         prim_path="/racetrack",
        #         name="race_track",
        #         position=np.array([-0.302, -3.0, 0.01]),
        #         # size=1.0,
        #         # color=np.array([0, 1.0, 0]),
        #     )
        # )

        # stage = Usd.Stage.Open(racetrack_usd_path)
        # self.track_meshes = []

                        
        # stage = Usd.Stage.Open(racetrack_usd_path)
        # self.track_meshes = [
        #     UsdGeom.Mesh(stage.GetPrimAtPath("/root/ID3")),
        #     UsdGeom.Mesh(stage.GetPrimAtPath("/root/ID11")),
        #     UsdGeom.Mesh(stage.GetPrimAtPath("/root/ID19"))
        # ]


        # from pxr import UsdGeom, Gf

        # stage = self._my_world.get_stage()
        # racetrack_prim = stage.GetPrimAtPath("/racetrack")
        # racetrack_xform = UsdGeom.Xformable(racetrack_prim)
        # transform = UsdGeom.XformCommonAPI(racetrack_xform)
        # transform.SetTranslate((0.0, 0.0, 0.2))  # Adjust position as needed
        # transform.SetScale((1.0, 1.0, 1.0))  # Adjust scale as needed


        
        # self.racetrack = XFormPrim(prim_path="/racetrack")
        # self._my_world.scene.add(self.racetrack)
        
        # XFormPrim(
        #     prim_path="/racetrack",
        #     name="racetrack",
        #     position=np.array([0.0, 0.0, 0.05]),
        #     size=1.0,
        # )
        
        
        
        
        
        # # Import the racetrack from the provided USD path -- NEW
        # racetrack_usd_path = "omniverse://localhost/NVIDIA/Assets/Isaac/4.1/Isaac/Environments/Jetracer/jetracer_track_solid.usd"
        # add_reference_to_stage(racetrack_usd_path, prim_path="/Environment/Racetrack")
        # self.racetrack = self._my_world.scene.add_from_usd(
        #     prim_path="/racetrack",
        #     usd_path=racetrack_usd_path,
        #     position=np.array([0, 0, 0])
        # )
    
        
        
        self.seed(seed)
        self.reward_range = (-float("inf"), float("inf"))
        gymnasium.Env.__init__(self)
        self.action_space = spaces.Box(low=-1, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=float("inf"), high=float("inf"), shape=(19,), dtype=np.float32)

        self.max_velocity = 1
        self.max_angular_velocity = math.pi
        self.reset_counter = 0
        return



    def get_dt(self):
        return self._dt

    def step(self, action):
        previous_jetbot_position, _ = self.jetbot.get_world_pose()
        # action forward velocity , angular velocity on [-1, 1]
        raw_forward = action[0]
        raw_angular = action[1]

        # we want to force the jetbot to always drive forward
        # original value: forward = (raw_forward + 1) / 2.0 
        forward = (raw_forward + 1.0) / 2.0 
        # so we transform to [0,1].  we also scale by our max velocity
        forward_velocity = forward * self.max_velocity

        # we scale the angular, but leave it on [-1,1] so the
        # jetbot can remain an ambiturner.
        angular_velocity = raw_angular * self.max_angular_velocity

        # we apply our actions to the jetbot
        for i in range(self._skip_frame):
            self.jetbot.apply_wheel_actions(
                self.jetbot_controller.forward(command=[forward_velocity, angular_velocity])
            )
            self._my_world.step(render=False)

        observations = self.get_observations()
        info = {}
        done = False
        truncated = False
        # HÄR BESTÄMMER DEN HUR LÅNG EN "EPISODE" SKA VARA - DVS ETT "FÖRSÖK" ATT FÅNGA KUBEN - HAR DÅ EN "_max_episode_length" 
        if self._my_world.current_time_step_index - self._steps_after_reset >= self._max_episode_length:
            done = True
            truncated = True
        goal_world_position, _ = self.goal.get_world_pose()
        
        # new - obs position
        obstacle_world_position, _ = self.obstacle.get_world_pose()
        
        current_jetbot_position, _ = self.jetbot.get_world_pose()
        previous_dist_to_goal = np.linalg.norm(goal_world_position - previous_jetbot_position)
        current_dist_to_goal = np.linalg.norm(goal_world_position - current_jetbot_position)
        
        # reward for going to the goal -OLD
        reward = previous_dist_to_goal - current_dist_to_goal
        
        # punish for going into wall - NEW
        dist_to_obstacle = np.linalg.norm(obstacle_world_position - current_jetbot_position)
        
        
        # # new - obs position
        # racetrack_walls_world_position, _ = self.obstacle.get_world_pose()



        
        # test to make racetrack punish
        # dist_racetrack_walls = np.linalg.norm(obstacle_world_position - current_jetbot_position)
        
        
        # race_track_world_position, _ = self.race_track.get_world_pose()
        
        # dist_to_racetrack_wall = np.linalg.norm(obstacle_world_position - current_jetbot_position)
        
        
    # # new  -- for race track 
    #     dist_to_closest_mesh = self.calculate_dist_to_closest_racetrack_wall(current_jetbot_position)
    #     if dist_to_closest_mesh < 0.1:
    #         reward += 1.0
    #     else:
    #         reward -= 5.0
        
        
        for cube in self.cubes:
            cube_position, _= cube.get_world_pose()
            distance_to_cube = np.linalg.norm(current_jetbot_position[:2] - cube_position[:2])
            if distance_to_cube < 0.1:  # Assume collision threshold as 0.1
                reward -= 10  # Large penalty for collision
                break
                # done = True # End the episode due to collision
                # info['collision'] = 'red_cube'
                # return observations, reward, done, truncated, info 
        
        if dist_to_obstacle < 0.5:
            reward -= 10
        
        
        if current_dist_to_goal < 0.1:
            done = True
        return observations, reward, done, truncated, info
    

    
    # def calculate_dist_to_closest_racetrack_wall(self, jetbot_position):
    #     paths = ["/Root/ID3", "/Root/ID11", "/Root/ID19"]
    #     min_dist = float('inf')
    #     for prim_path in paths:
    #         prim = self._my_world.stage.GetPrimAtPath(prim_path)
    #         if not prim.IsValid():
    #             print(f"Warning: Prim {prim_path} is invalid or does not exist.")
    #             continue

    #         mesh = UsdGeom.Mesh(prim)
    #         points = mesh.GetPointsAttr().Get()
    #         for point in points:
    #             dist = np.linalg.norm(jetbot_position[:2] - point[:2])  # Assuming a 2D plane distance
    #             if dist < min_dist:
    #                 min_dist = dist
    #     return min_dist
    
    
    
    def reset(self, seed=None):
        self._my_world.reset()
        self.reset_counter = 0
        # randomize goal location in circle around robot
        # alpha = 2 * math.pi * np.random.rand()
        # r = 1.00 * math.sqrt(np.random.rand()) + 0.20
        # self.goal.set_world_pose([2.9, -1.9,  0.05])

        # new - make to wall be between goal and start postiion for robot
        goal_world_position, _ = self.goal.get_world_pose()
        jetbot_world_position, _ = self.jetbot.get_world_pose()
        # midpoint = (goal_world_position + jetbot_world_position) / 2
        # self.obstacle.set_world_pose(midpoint)

        observations = self.get_observations()
        return observations, {}

    def get_observations(self):
        self._my_world.render()
        jetbot_world_position, jetbot_world_orientation = self.jetbot.get_world_pose()
        jetbot_linear_velocity = self.jetbot.get_linear_velocity()
        jetbot_angular_velocity = self.jetbot.get_angular_velocity()
        goal_world_position, _ = self.goal.get_world_pose()
        
        # new 
        obstacle_world_position, _ = self.obstacle.get_world_pose()
        
        
        obs = np.concatenate(
            [
                jetbot_world_position,
                jetbot_world_orientation,
                jetbot_linear_velocity,
                jetbot_angular_velocity,
                goal_world_position,
                obstacle_world_position,
            ]
        )
        return obs

    def render(self, mode="human"):
        return

    def close(self):
        self._simulation_app.close()
        return

    def seed(self, seed=None):
        self.np_random, seed = gymnasium.utils.seeding.np_random(seed)
        np.random.seed(seed)
        return [seed]
