import numpy as np

from gym.envs.robotics import rotations, robot_env_m, utils
from gym.utils.dm_utils.rewards import tolerance
from copy import copy
from collections import deque


def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)


class CatchEnv(robot_env_m.RobotEnvM):
    """Superclass for all Fetch environments.
    """

    def __init__(
        self, model_path, n_substeps, gripper_extra_height, block_gripper,
        has_object, target_in_the_air, target_offset, obj_range, target_range,
        distance_threshold, initial_qpos, reward_type,
        add_high_res_output, no_movement,
        stack_frames, camera_3
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

        self.target_z_l = 4
        # self.target_z_position = deque(maxlen=self.target_z_l)

        self.last_img = None
        self.last_img2 = None
        self.last_vec = None

        self.end_location = None

        self.tmp_data = []

        self.n_actions = 3

        self.action = None

        super(CatchEnv, self).__init__(
            model_path=model_path, n_substeps=n_substeps, n_actions=self.n_actions,
            initial_qpos=initial_qpos)

    # GoalEnv methods
    # ----------------------------

    def compute_reward(self, achieved_goal, goal, info):
        reward_ctrl = - 0.05 * np.square(self.action).sum()

        dist_to_end_location = np.linalg.norm(self.sim.data.get_site_xpos('robot0:grip') -
                                              self.end_location)
        reward_dist = tolerance(dist_to_end_location, margin=0.8, bounds=(0., 0.02),
                                sigmoid='linear',
                                value_at_margin=0.)

        reward = 0.25 * reward_dist

        if self.sim.data.get_site_xpos('tar')[2] < 0.1:
            self._restart_target()

        sparse_reward = 0.
        dist = np.linalg.norm(self.sim.data.get_site_xpos('robot0:grip') -
                              self.sim.data.get_site_xpos('tar'))
        if dist < 0.05:
            reward += 2.
            sparse_reward += 1.
            self._restart_target()

        reward += reward_ctrl

        info = dict(scoring_reward=sparse_reward)

        return reward, False, info

    def _restart_target(self):
        target_x = self.np_random.uniform(low=0.52, high=0.84)
        target_y = self.np_random.uniform(low=-0.23, high=0.23)

        self.end_location = [2.2 - target_x, 0.75 + target_y, 0.66]

        y = self.np_random.uniform(low=-0.5, high=0.5)
        z = self.np_random.uniform(low=-0.15, high=0.)
        z_offset = 0.34 + z
        v_z = self.np_random.uniform(low=1.9, high=2.3)

        del_y = y - target_y

        target_dist = np.sqrt(del_y * del_y + target_x * target_x)

        sin_theta = target_x / target_dist

        v = target_dist * 9.81 / (v_z + np.sqrt(v_z * v_z + 19.62 * z_offset))

        v_x = v * sin_theta
        v_y = np.sign(del_y) * v * np.sqrt(1. - sin_theta * sin_theta)

        self.sim.data.set_joint_qpos('tar:x', 0.0)
        self.sim.data.set_joint_qpos('tar:y', y)
        self.sim.data.set_joint_qpos('tar:z', z)
        self.sim.data.set_joint_qvel('tar:x', - v_x)
        self.sim.data.set_joint_qvel('tar:y', - v_y)
        self.sim.data.set_joint_qvel('tar:z', v_z)

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

        if not self.no_movement:
            while action.shape < (4,):
                action = np.append(action, [0.])
            if self.sim.data.get_site_xpos('robot0:grip')[0] < 1.3 and action[0] < 0.:
                action[0] = 0.
            if self.sim.data.get_site_xpos('robot0:grip')[0] > 1.8 and action[0] > 0.:
                action[0] = 0.

            if self.sim.data.get_site_xpos('robot0:grip')[1] < 0.4 and action[1] < 0.:
                action[1] = 0.
            if self.sim.data.get_site_xpos('robot0:grip')[1] > 1.1 and action[1] > 0.:
                action[1] = 0.

            if self.sim.data.get_site_xpos('robot0:grip')[2] < 0.47 and action[2] < 0.:
                action[2] = 0.
            if self.sim.data.get_site_xpos('robot0:grip')[2] > 0.87 and action[2] > 0.:
                action[2] = 0.

            action = action.copy()  # ensure that we don't change the action outside of this scope
            pos_ctrl, gripper_ctrl = action[:3], action[3]

            pos_ctrl *= 0.05  # limit maximum change in position
            rot_ctrl = [1., 0., 0., 0.]
            gripper_ctrl = np.array([gripper_ctrl, gripper_ctrl])
            assert gripper_ctrl.shape == (2,)
            if self.block_gripper:
                gripper_ctrl = np.zeros_like(gripper_ctrl)
            action = np.concatenate([pos_ctrl, rot_ctrl, gripper_ctrl])

            # Apply action to simulation.
            utils.ctrl_set_action(self.sim, action)
            utils.mocap_set_action(self.sim, action)
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
            'image': img,
            'vector': vec,
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.goal.copy(),
        }

        if self.last_img is not None:
            out_dict['last_image'] = copy(self.last_img)
            out_dict['last_vector'] = copy(self.last_vec)
        else:
            out_dict['last_image'] = copy(img)
            out_dict['last_vector'] = copy(vec)
        self.last_img = img
        self.last_vec = vec

        if self.add_high_res_output:
            self.renderer.render(512, 512, camera_id=3)
            img_high = self.renderer.read_pixels(512, 512, depth=False)
            self.renderer.render(512, 512, camera_id=4)
            img_top_high = self.renderer.read_pixels(512, 512, depth=False)
            img_high = img_high[::-1, :, :]
            img_top_high = img_top_high[::-1, :, :]

            out_dict['image_high_res'] = img_high
            out_dict['image_top_high_res'] = img_top_high

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
        self.last_img = None
        self.last_img2 = None
        self.last_vec = None
        # Randomize start position of object.
        self._restart_target()

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
            goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(-0.15, 0.15, size=3)
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
        gripper_target = np.array([-0.498, 0.005, -0.431 + self.gripper_extra_height]) + \
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
