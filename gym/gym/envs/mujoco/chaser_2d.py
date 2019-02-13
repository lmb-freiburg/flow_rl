import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from gym.utils.dm_utils.rewards import tolerance
from copy import copy
from collections import deque
from gym.utils.mujoco_random_4_texture_modder import Random4TextureModder


class Chaser2dEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, target_velocity, target_velocity_delta, add_high_res_output, no_movement,
                 stack_frames, random_4_background, random_frequency, pre_state_dt):
        self.target_velocity = target_velocity
        self.target_velocity_delta = target_velocity_delta
        self.add_high_res_output = add_high_res_output
        self.no_movement = no_movement
        self.stack_frames = stack_frames
        self.random_4_background = random_4_background
        self.random_frequency = random_frequency  # -1 <=> change texture after each episode

        self.last_img = deque(maxlen=pre_state_dt)
        self.last_vec = deque(maxlen=pre_state_dt)

        self.previous_target_position = [0., 0.]

        self.n_total_steps = 0

        utils.EzPickle.__init__(self)
        if not self.random_4_background:
            mujoco_env.MujocoEnv.__init__(self, 'reacher_m.xml', 2)
        else:
            mujoco_env.MujocoEnv.__init__(self, 'reacher_m_var.xml', 2)

    def step(self, a):
        if self.no_movement:
            a = np.zeros([2], np.float32)

        if self.random_4_background and \
                self.random_frequency != -1 and \
                not self.n_total_steps % self.random_frequency:
            self._randomize_env()

        self.n_total_steps += 1

        dist = np.linalg.norm(self.get_body_com("fingertip") - self.get_body_com("target"))
        reward_dist = tolerance(dist, margin=0.3, bounds=(0., 0.009),
                                sigmoid='cosine',
                                value_at_margin=0.)
        reward_ctrl = - 0.1 * np.square(a).sum()
        reward = reward_dist + reward_ctrl
        sparse_reward = 0.
        if dist < 0.02:
            sparse_reward = 1.
            reward += sparse_reward

        self.do_simulation(a, self.frame_skip)

        # target wall reflection:
        if np.abs(self.sim.data.qpos[2]) > 0.205:
            self.sim.data.qvel[2] = - self.sim.data.qvel[2]
        if np.abs(self.sim.data.qpos[3]) > 0.205:
            self.sim.data.qvel[3] = - self.sim.data.qvel[3]

        ob = self._get_obs()
        done = False
        return ob, reward, done, dict(scoring_reward=sparse_reward)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0

    def reset_model(self):
        qpos = self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq) + self.init_qpos
        if self.no_movement:
            qpos = self.np_random.uniform(low=-3.14, high=3.14, size=self.model.nq) + self.init_qpos
        initial_target_position = self.np_random.uniform(low=-.2, high=.2, size=2)
        while np.linalg.norm(initial_target_position) > .2:
            initial_target_position = self.np_random.uniform(low=-.2, high=.2, size=2)
        qpos[-2:] = initial_target_position
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        if self.no_movement:
            qvel[:2] = np.zeros([2], np.float32)
        target_angle = self.np_random.uniform(low=0., high=2. * np.pi)
        target_velocity = self.target_velocity + \
            self.np_random.uniform(low=-self.target_velocity_delta,
                                   high=self.target_velocity_delta)
        qvel[2] = target_velocity * np.cos(target_angle)
        qvel[3] = target_velocity * np.sin(target_angle)
        self.set_state(qpos, qvel)
        self.previous_target_position = copy(qpos[-2:])
        self.last_img.clear()
        self.last_vec.clear()

        if self.random_4_background and self.random_frequency == -1:
            self._randomize_env()

        return self._get_obs()

    def _randomize_env(self):
        modder = Random4TextureModder(self.sim)
        if self.random_4_background:
            modder.rand_4("skybox")

    def _get_obs(self):
        self.renderer.render(self.y_dim, self.x_dim, camera_id=0)
        img = self.renderer.read_pixels(self.y_dim, self.x_dim, depth=False)[::-1, :, :]

        # robot arm state:
        theta = self.sim.data.qpos[:2]
        v_vector = self.sim.data.qvel.flat[:2]

        vec = np.concatenate([
            np.cos(theta),
            np.sin(theta),
            v_vector,
            self.get_body_com("fingertip")[:2]
        ])
        out_dict = {
            "image": img,
            "vector": vec
        }

        self.last_img.append(img)
        self.last_vec.append(vec)

        out_dict["last_image"] = copy(self.last_img[0])
        out_dict["last_vector"] = copy(self.last_vec[0])

        if self.stack_frames:
            out_dict["image"] = np.concatenate([img, out_dict["last_image"]], axis=-1)

        if self.add_high_res_output:
            self.renderer.render(512, 512, camera_id=0)
            img_high = self.renderer.read_pixels(512, 512, depth=False)[::-1, :, :]
            out_dict["image_high_res"] = img_high

        return out_dict
