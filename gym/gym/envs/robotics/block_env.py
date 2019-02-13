import numpy as np

from gym.envs.robotics import rotations, robot_env_m, utils
from gym.utils.dm_utils.rewards import tolerance
from copy import copy
from collections import deque


def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)


class BlockEnv(robot_env_m.RobotEnvM):
    """Superclass for all Fetch environments.
    """

    def __init__(
        self, model_path, n_substeps, gripper_extra_height, block_gripper,
        has_object, target_in_the_air, target_offset, obj_range, target_range,
        distance_threshold, initial_qpos, reward_type,
        add_high_res_output, no_movement, stack_frames, camera_3, high_motion_penalty,
    ):
        """Initializes a new Fetch environment.

        Args:
            model_path (string): path to the environments XML file
            n_substeps (int): number of substeps the simulation runs on every call to step
            gripper_extra_height (float): additional height above the table when positioning
            the gripper
            block_gripper (boolean): whether or not the gripper is blocked (i.e. not movable)
            or not
            has_object (boolean): whether or not the environment has an object
            target_in_the_air (boolean): whether or not the target should be in the air above the
            table or on the table surface
            target_offset (float or array with 3 elements): offset of the target
            obj_range (float): range of a uniform distribution for sampling initial object positions
            target_range (float): range of a uniform distribution for sampling a target
            distance_threshold (float): the threshold after which a goal is considered achieved
            initial_qpos (dict): a dictionary of joint names and values that define the initial
            configuration reward_type ('sparse' or 'dense'): the reward type, i.e. sparse or dense
        """

        self.gripper_extra_height = gripper_extra_height
        self.block_gripper = block_gripper
        self.has_object = has_object
        self.target_in_the_air = target_in_the_air
        self.target_offset = target_offset
        self.obj_range = obj_range
        self.target_range = target_range
        self.distance_threshold = distance_threshold
        self.reward_type = reward_type

        self.add_high_res_output = add_high_res_output
        self.no_movement = no_movement
        self.stack_frames = stack_frames
        self.camera_3 = camera_3
        self.high_motion_penalty = high_motion_penalty

        self.initial_h = 1.2
        self.sqrt_2_g = 4.429446918

        self.target_z_l = 4

        self.last_img = None
        self.last_img2 = None
        self.last_vec = None

        self.end_location = None

        self.tmp_data = []

        self.n_actions = 4

        self.energy_corrected = True
        self.give_reflection_reward = False

        self.action = None

        self.update_fraction = 0.

        super(BlockEnv, self).__init__(
            model_path=model_path, n_substeps=n_substeps, n_actions=self.n_actions,
            initial_qpos=initial_qpos)

    # GoalEnv methods
    # ----------------------------

    def compute_reward(self, achieved_goal, goal, info):
        if not self.high_motion_penalty:
            reward_ctrl = - 0.05 * np.square(self.action).sum()
        else:
            reward_ctrl = - 0.075 * np.square(self.action).sum()


        dist = np.linalg.norm(self.sim.data.get_site_xpos('robot0:grip')[:2] -
                              self.sim.data.get_site_xpos('tar')[:2])
        reward_dist = tolerance(dist, margin=0.5, bounds=(0., 0.02),
                                sigmoid='linear',
                                value_at_margin=0.)

        reward = 0.2 * reward_dist + reward_ctrl

        done = False
        if self.sim.data.get_site_xpos('tar')[2] < 0.4:
            done = True
            reward = -1.

        sparse_reward = 0.
        if self.give_reflection_reward:
            sparse_reward = 1.
            self.give_reflection_reward = False

        reward += 0.2 * sparse_reward

        info = dict(scoring_reward=sparse_reward)

        return reward, done, info

    # RobotEnv methods
    # ----------------------------

    def _step_callback(self):
        if self.block_gripper:
            self.sim.data.set_joint_qpos('robot0:l_gripper_finger_joint', 0.)
            self.sim.data.set_joint_qpos('robot0:r_gripper_finger_joint', 0.)
            self.sim.forward()

    def _set_action(self, action):
        assert action.shape == (self.n_actions,)

        self.action = action

        # energy correction
        h = self.sim.data.get_site_xpos('tar')[2]
        if h < 0.8:
            self.energy_corrected = False
        # each time the target crosses the z == 0.8 line its energy gets corrected
        if not self.energy_corrected:
            if h > 0.8:
                self.sim.data.set_joint_qvel('tar:z', self.sqrt_2_g * np.sqrt(self.initial_h - h))
                self.give_reflection_reward = True
                self.energy_corrected = True

        if not self.no_movement:
            a = np.zeros([8], dtype=np.float32)
            a[0] = 0.015 * action[0]
            a[1] = 0.015 * action[1]
            euler = rotations.mat2euler(self.sim.data.get_site_xmat('robot0:grip'))
            a_4 = euler[0]
            a_5 = euler[1]

            # adjusting z position to a plane (otherwise the arm has a downwards drift)
            if self.sim.data.get_site_xpos('robot0:grip')[2] < 0.705:
                a[2] = 0.665 - self.sim.data.get_site_xpos('robot0:grip')[2]
            else:
                a[2] = 0.665 - self.sim.data.get_site_xpos('robot0:grip')[2]
            a[4] = 0.015 * action[2]
            a[5] = 0.015 * action[3]

            # Restricting rotation action:

            if a_4 > 0.3 and action[2] > 0.:
                a[4] = 0.
            if a_4 < -0.3 and action[2] < 0.:
                a[4] = 0.

            if a_5 > 0.3 and action[3] > 0.:
                a[5] = 0.
            if a_5 < -0.3 and action[3] < 0.:
                a[5] = 0.

            # Apply action to simulation.

            utils.ctrl_set_action(self.sim, a)
            utils.mocap_set_action(self.sim, a)

        else:
            pass

    def _get_obs(self):

        # positions
        grip_pos = self.sim.data.get_site_xpos('robot0:grip')
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp('robot0:grip') * dt
        robot_qpos, robot_qvel = utils.robot_get_obs(self.sim)
        if self.has_object:
            object_pos = self.sim.data.get_site_xpos('object0')
            # rotations
            object_rot = rotations.mat2euler(self.sim.data.get_site_xmat('object0'))
            # velocities
            object_velp = self.sim.data.get_site_xvelp('object0') * dt
            object_velr = self.sim.data.get_site_xvelr('object0') * dt
            # gripper state
            object_rel_pos = object_pos - grip_pos
            object_velp -= grip_velp
        else:
            object_pos = object_rot = object_velp = object_velr = object_rel_pos = np.zeros(0)
        gripper_state = robot_qpos[-2:]
        gripper_vel = robot_qvel[-2:] * dt  # change to a scalar if the gripper is made symmetric

        if not self.has_object:
            achieved_goal = grip_pos.copy()
        else:
            achieved_goal = np.squeeze(object_pos.copy())

        self.renderer.render(84, 84, camera_id=3)

        img = self.renderer.read_pixels(84, 84, depth=False)
        img = img[::-1, :, :]

        self.renderer.render(84, 84, camera_id=4)
        img_top = self.renderer.read_pixels(84, 84, depth=False)
        img_top = img_top[::-1, :, :]
        img = np.concatenate([img_top, img], axis=2)


        vec = np.concatenate([
            grip_pos, grip_velp, gripper_vel,
        ])

        out_dict = {
            "image": img,
            "vector": vec,
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.goal.copy(),
        }

        if self.last_img is not None:
            out_dict["last_image"] = copy(self.last_img)
            out_dict["last_vector"] = copy(self.last_vec)
        else:
            out_dict["last_image"] = copy(img)
            out_dict["last_vector"] = copy(vec)
        self.last_img = img
        self.last_vec = vec

        if self.add_high_res_output:
            self.renderer.render(512, 512, camera_id=3)
            img_high = self.renderer.read_pixels(512, 512, depth=False)
            self.renderer.render(512, 512, camera_id=4)
            img_top_high = self.renderer.read_pixels(512, 512, depth=False)
            img_high = img_high[::-1, :, :]
            img_top_high = img_top_high[::-1, :, :]

            out_dict["image_high_res"] = img_high
            out_dict["image_top_high_res"] = img_top_high

        if self.camera_3:
            self.renderer.render(512, 512, camera_id=5)
            camera_3 = self.renderer.read_pixels(512, 512, depth=False)
            camera_3 = camera_3[::-1, :, :]
            out_dict['image_camera_3'] = camera_3

        if self.stack_frames:
            out_dict['image'] = np.concatenate([out_dict['image'],
                                                out_dict['last_image']], axis=-1)

        return out_dict

    def _viewer_setup(self):
        body_id = self.sim.model.body_name2id('robot0:gripper_link')
        lookat = self.sim.data.body_xpos[body_id]
        for idx, value in enumerate(lookat):
            self.viewer.cam.lookat[idx] = value
        self.viewer.cam.distance = 2.5
        self.viewer.cam.azimuth = 132.
        self.viewer.cam.elevation = -14.

    def _render_callback(self):
        # Visualize target.
        sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
        site_id = self.sim.model.site_name2id('target0')
        self.sim.model.site_pos[site_id] = self.goal - sites_offset[0]
        self.sim.forward()

    def _reset_sim(self):
        self.sim.set_state(self.initial_state)

        theta = self.np_random.uniform(0, 6.283)

        r = self.np_random.uniform(0.06, 0.09)
        dx = r*np.cos(theta) + self.np_random.uniform(-0.11, 0.11)
        dy = r*np.sin(theta) + self.np_random.uniform(-0.11, 0.11)

        v_x = r / 0.11 * np.cos(theta)
        v_y = r / 0.11 * np.sin(theta)

        self.sim.data.set_joint_qpos('tar:x', -0.72 + dx)
        self.sim.data.set_joint_qpos('tar:y', dy)
        self.sim.data.set_joint_qpos('tar:z', 0.5)
        # self.initial_h = self.sim.data.get_site_xpos('tar')[2]
        self.energy_corrected = True
        self.give_reflection_reward = False

        #
        self.sim.data.set_joint_qvel('tar:x', -v_x)
        self.sim.data.set_joint_qvel('tar:y', -v_y)

        self.last_img = None
        self.last_img2 = None
        self.last_vec = None

        if self.has_object:
            object_xpos = self.initial_gripper_xpos[:2]
            while np.linalg.norm(object_xpos - self.initial_gripper_xpos[:2]) < 0.1:
                object_xpos = self.initial_gripper_xpos[:2] + \
                    self.np_random.uniform(-self.obj_range, self.obj_range, size=2)
            object_qpos = self.sim.data.get_joint_qpos('object0:joint')
            assert object_qpos.shape == (7,)
            object_qpos[:2] = object_xpos
            self.sim.data.set_joint_qpos('object0:joint', object_qpos)

        self.sim.forward()
        return True

    def _sample_goal(self):
        if self.has_object:
            goal = self.initial_gripper_xpos[:3] + \
                self.np_random.uniform(-self.target_range, self.target_range, size=3)
            goal += self.target_offset
            goal[2] = self.height_offset
            if self.target_in_the_air and self.np_random.uniform() < 0.5:
                goal[2] += self.np_random.uniform(0, 0.45)
        else:
            goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(-0.2, 0.2, size=3)
        return goal.copy()

    def _is_success(self, achieved_goal, desired_goal):
        d = goal_distance(achieved_goal, desired_goal)
        return (d < self.distance_threshold).astype(np.float32)

    def _env_setup(self, initial_qpos):
        for name, value in initial_qpos.items():
            self.sim.data.set_joint_qpos(name, value)
        utils.reset_mocap_welds(self.sim)
        self.sim.forward()

        # Move end effector into position.
        # gripper_target = np.array([-0.498, 0.005, -0.431 + self.gripper_extra_height]) + \
        #     self.sim.data.get_site_xpos('robot0:grip')
        gripper_target = np.array([-0.398, 0.005, -0.431 + self.gripper_extra_height]) + \
            self.sim.data.get_site_xpos('robot0:grip')
        gripper_rotation = np.array([1., 0., 0., 0.])
        self.sim.data.set_mocap_pos('robot0:mocap', gripper_target)
        self.sim.data.set_mocap_quat('robot0:mocap', gripper_rotation)
        for _ in range(10):
            self.sim.step()

        # Extract information for sampling goals.
        self.initial_gripper_xpos = self.sim.data.get_site_xpos('robot0:grip').copy()
        if self.has_object:
            self.height_offset = self.sim.data.get_site_xpos('object0')[2]
