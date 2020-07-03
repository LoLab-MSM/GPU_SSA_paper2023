import os

gpu_id = "0"
os.environ['CUDA_DEVICE'] = gpu_id

import socket
import numpy as np
import pandas as pd
import time
from pysb.simulator import BngSimulator, OpenCLSSASimulator, \
    StochKitSimulator, CudaSSASimulator
from utils import get_info, write
import logging
from pysb.logging import setup_logger

setup_logger(logging.INFO)
computer_name = socket.gethostname().lower()

print("computer = {}".format(computer_name))

cur_dir = os.path.dirname(__file__)


# set_path('stochkit_ssa', '/home/pinojc/git/StochKit')


def run(n_sim, model, tspan, simulator='cuda'):
    v = False
    if 'cl' in simulator:
        sim = OpenCLSSASimulator(model, tspan=tspan, verbose=v,
                                 precision=np.float64)
    elif simulator == 'cuda':
        sim = CudaSSASimulator(model, tspan=tspan, verbose=v,
                               precision=np.float64)
    elif simulator == 'bng':
        sim = BngSimulator(model, tspan=tspan, verbose=v)
    elif simulator == 'stochkit':
        sim = StochKitSimulator(model, tspan=tspan, verbose=v)
    else:
        return
    st = time.time()
    if simulator in ('bng', 'stochkit'):
        sim.run(tspan, n_runs=n_sim)
    else:
        # sim.run(tspan, number_sim=n_sim)
        traj = sim.run(tspan, number_sim=n_sim)
        # print(traj.dataframe.min())
    # quit()
    total_time = time.time() - st

    return total_time, sim._time


def run_model(model, t_end, n_timesteps, max_sim=17):
    tspan = np.linspace(0, t_end, n_timesteps)
    # max_sim = 15
    min_sim = 7
    # n_sims = [2 ** i for i in range(min_sim, max_sim)]
    n_sims = [2 ** 10, ] * 4
    info = dict()
    info['model_name'] = model.name
    info['device_name'] = computer_name
    info['n_ts'] = n_timesteps
    info['end_time'] = t_end

    def _run(sim_name):
        local_only = [None] * len(n_sims)

        for n, i in enumerate(n_sims):
            print(i)
            d = info.copy()
            d['n_sim'] = i
            d['simulator'] = sim_name

            # get name for CPU/GPU
            # for backwards comparability, im adding cpu name into gpu col
            if sim_name == 'cl_amd_gpu':
                gpu_name = 'Radeon Vii'
            elif sim_name in ('cl_nvidia', 'cuda'):
                try:
                    from pycuda.driver import Device
                    gpu_name = ''.join([i for i in
                                        Device(int(gpu_id)).name().split(' ')
                                        if i != 'GeForce'])
                except:
                    gpu_name = 'RTX2080'
            else:
                gpu_name = get_info()['cpu_name']

            d['gpu_name'] = gpu_name
            d['total_time'], d['sim_time'] = run(i, model, tspan, sim_name)
            write(d)
            local_only[n] = d

        tmp_pd = pd.DataFrame(local_only)
        print(tmp_pd[['n_sim', 'sim_time']])

        out_name = os.path.join(
            cur_dir,
            'Timings',
            '{}_{}_{}_timing.csv'.format(computer_name, sim_name, model.name)
        )
        # tmp_pd.to_csv(out_name)

    # if computer_name == 'buu':
    #     _run('cl_amd_gpu')
    # elif computer_name == 'beerus':
    #     if os.environ['PYOPENCL_CTX'] == '1':
    #         _run('cl_amd_cpu')
    #     else:
    #         _run('cl_nvidia')
    _run('cuda')
    # _run("stochkit")
    # _run('bng')


if __name__ == "__main__":
    from pysb.examples.schloegl import model as scholgl_model

    # run_tpb_test2()
    # quit()
    # 4.3341310024261475
    os.environ['PYOPENCL_CTX'] = '0'
    # os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
    # run_model(michment, 100, 101)
    run_model(scholgl_model, 100, 101)
    # run_model(kinase_model, 50, 101)
    # run_model(earm_1, 20000, 101)

    # xilox
    # 65536  0.118101
    # 65536  4.238649

    # MT
    # 65536  0.679058
    # 65536  23.882557
    # 8192  200.544625

    # tyche
    # 65536  0.113598
    # 65536  4.086017
    #  8192  54.534443
