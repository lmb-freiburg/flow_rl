import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from copy import copy
from collections import deque

class SwimmerImgEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, screen_output, vector_output, add_high_res_output, stack_frames):
        self.screen_output = screen_output
        self.vector_output = vector_output
        self.add_high_res_output = add_high_res_output
        self.stack_frames = stack_frames

        self.last_img = deque(maxlen=4)
        self.last_vec = deque(maxlen=4)

        mujoco_env.MujocoEnv.__init__(self, 'swimmer_img.xml', 4)
        utils.EzPickle.__init__(self)

    def step(self, a):
        ctrl_cost_coeff = 0.0001
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        xposafter = self.sim.data.qpos[0]
        reward_fwd = (xposafter - xposbefore) / self.dt
        reward_ctrl = - ctrl_cost_coeff * np.square(a).sum()
        reward = reward_fwd + reward_ctrl
        ob = self._get_obs()
        return ob, reward, False, dict(scoring_reward=reward)

    def _get_obs(self):
        if self.screen_output:
            if self.vector_output:
                qpos = self.sim.data.qpos
                qvel = self.sim.data.qvel
                vec = np.concatenate([qpos.flat[2:], qvel.flat])
            else:
                vec = np.array([0.], dtype=np.float32)

            self.renderer.render(self.y_dim, self.x_dim, camera_id=0)
            img = self.renderer.read_pixels(84, 84, depth=False)[::-1, :, :]

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
        else:
            qpos = self.sim.data.qpos
            qvel = self.sim.data.qvel
            return np.concatenate([qpos.flat[2:], qvel.flat])

    def reset_model(self):
        self.last_img.clear()
        self.last_vec.clear()
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-.1, high=.1, size=self.model.nv)
        )
        return self._get_obs()
