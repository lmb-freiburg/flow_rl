import os


def get_gpu_id():
    try:
        file_path = os.environ["PBS_GPUFILE"]
    except KeyError:
        print("Unknown GPU id, using 0 (default defined at gym/gym/utils/gpu_id.py)")
        return 0

    with open(file_path, "r") as f:
        return int(f.read()[-2])
