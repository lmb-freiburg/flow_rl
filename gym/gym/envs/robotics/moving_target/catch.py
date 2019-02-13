from gym import utils
from gym.envs.robotics import catch_env


class Catcher3dEnv(catch_env.CatchEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse',
                 add_high_res_output=False, no_movement=False, stack_frames=False, camera_3=False):
        initial_qpos = {
            'robot0:slide0': 0.4049,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
        }
        catch_env.CatchEnv.__init__(
            self, 'moving_target/catcher.xml', has_object=False, block_gripper=True, n_substeps=20,
            gripper_extra_height=0.3, target_in_the_air=True, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type,
            add_high_res_output=add_high_res_output, no_movement=no_movement,
            stack_frames=stack_frames, camera_3=camera_3
        )
        utils.EzPickle.__init__(self)
