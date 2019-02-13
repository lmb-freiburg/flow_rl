from mujoco_py.modder import TextureModder
import numpy as np


class Random4TextureModder(TextureModder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def rand_4(self, name):
        choices_1 = [
            np.array([0.48841119, 0.61174386, 0.76590786]) * 255,
            np.array([0.51841799, 0.2968005 , 0.18772123]) * 255,
            np.array([0.08074127, 0.7384403 , 0.44130922]) * 255,
            np.array([0.15830987, 0.87993703, 0.27408646]) * 255
        ]
        choices_2 = [
            np.array([0.28468588, 0.25358821, 0.32756395]) * 255,
            np.array([0.1441643 , 0.16561286, 0.96393053]) * 255,
            np.array([0.96022672, 0.18841466, 0.02430656]) * 255,
            np.array([0.20455555, 0.69984361, 0.77951459]) * 255
        ]

        def _rand_rgb():
            choice = self.random_state.randint(len(choices_1))
            return np.array(choices_1[choice], dtype=np.uint8), \
                   np.array(choices_2[choice], dtype=np.uint8)

        rgb1, rgb2 = _rand_rgb()
        return self.set_checker(name, rgb1, rgb2)
