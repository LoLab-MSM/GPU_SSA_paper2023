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


def run_model(model, t_end, n_timesteps, max_sim=17):
    tspan = np.linspace(0, t_end, n_timesteps)
    # max_sim = 15
    min_sim = 7
    n_sims = [2 ** i for i in range(min_sim, max_sim)]
    # n_sims = [2 ** 10, ] * 4
    # n_sims = [2 ** 5, ] * 4

    precision = np.float64
    verbose = False

    info = dict()
    info['model_name'] = model.name
    info['device_name'] = computer_name
    info['n_ts'] = n_timesteps
    info['end_time'] = t_end
    info['opencl_args'] = os.environ.get('PYOPENCL_CTX')

    def _run(sim_name):

        if 'cl' in sim_name:
            sim = OpenCLSSASimulator(model, tspan=tspan, verbose=verbose,
                                     precision=precision)
            sim._setup()
            gpu_name = sim._device_name.replace(' ', '')

        elif sim_name == 'cuda':
            sim = CudaSSASimulator(model, tspan=tspan, verbose=verbose,
                                   precision=precision)
            from pycuda.driver import Device
            gpu_name = ''.join([i for i in
                                Device(int(gpu_id)).name().split(' ')
                                if i != 'GeForce'])
        elif sim_name == 'bng':
            sim = BngSimulator(model, tspan=tspan, verbose=verbose)
            gpu_name = get_info()['cpu_name']
        elif sim_name == 'stochkit':
            sim = StochKitSimulator(model, tspan=tspan, verbose=verbose)
            gpu_name = get_info()['cpu_name']
        else:
            return
        local_only = []
        for n_sim in n_sims:
            print(n_sim)
            d = info.copy()
            d['n_sim'] = n_sim
            d['simulator'] = sim_name
            d['gpu_name'] = gpu_name
            st = time.time()
            if sim_name in ('bng', 'stochkit'):
                sim.run(tspan, n_runs=n_sim)
            else:
                sim.run(tspan, number_sim=n_sim)
            et = time.time() - st
            d['total_time'] = et

            if hasattr(sim, '_time'):
                d['sim_time'] = sim._time

            write(d)
            local_only.append(d)

        tmp_pd = pd.DataFrame(local_only)
        print(tmp_pd[['n_sim', 'sim_time']])

        out_name = os.path.join(
            cur_dir,
            'Timings',
            '{}_{}_{}.csv'.format(
                computer_name,
                sim_name,
                model.name.split('.')[-1],

            )
        )
        # tmp_pd.to_csv(out_name)
        return

    #
    # if computer_name == 'buu':
    #     _run('cl_amd_gpu')
    # elif computer_name == 'beerus':
    #     if os.environ['PYOPENCL_CTX'] == '1':
    #         _run('cl_amd_cpu')
    #     else:
    #         _run('cl_nvidia')
    # #

    # _run('cuda')
    _run('cl_nvidia')
    # _run("stochkit")
    # _run('bng')


if __name__ == "__main__":
    from pysb.examples.schloegl import model as scholgl_model
    from pysb.examples.kinase_cascade import model as kinase_model
    from pysb.examples.earm_1_0 import model as earm_1
    from pysb.examples.michment import model as michment

    os.environ['PYOPENCL_CTX'] = '0'
    # os.environ['PYOPENCL_CTX'] = '0:3'
    run_model(michment, 100, 101)
    quit()
    run_model(scholgl_model, 100, 101)
    run_model(kinase_model, 100, 101)
    run_model(earm_1, 20000, 101)
