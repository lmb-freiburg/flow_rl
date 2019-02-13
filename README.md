# Code for the paper: Motion Perception in Reinforcement Learning with Dynamic Objects

**Artemij Amiranashvili, Alexey Dosovitskiy, Vladlen Koltun and Thomas Brox, CoRL 2018 ([paper link](https://lmb.informatik.uni-freiburg.de/projects/flowrl/)).**

## Dependencies:

- Python3
- TensorFlow
- [mujoco-py](https://github.com/openai/mujoco-py#obtaining-the-binaries-and-license-key)


## Installation:

Run `pip3 install -e .` in the `gym` and `baselines` directories.

## Tasks:

- Chaser2d-v2
- Catcher2d-v2
- Catcher3d-v1
- KeepUp3d-v1
- ChaserWithRandom4Backgrounds2d-v2
- KeepUpHighMotionPenalty3d-v1

## Running Examples:

Training `Catcher3d-v1` with additional TinyFlowNet flow prediction input (replace LOG_DIR with path for logging): 

    python3 baselines/baselines/ppo2/run_mujoco_imvec.py --main_path LOG_DIR --env_id Catcher3d-v1 --add_flownet True --flownet_path networks/Catcher3d/

Training `Chaser2d-v2` with additional image difference input:

    python3 baselines/baselines/ppo2/run_mujoco_imvec.py --main_path LOG_DIR --diff_frames
    
Training `Chaser2d-v2` with image stack input:

    python3 baselines/baselines/ppo2/run_mujoco_imvec.py --main_path LOG_DIR --stack_frames
