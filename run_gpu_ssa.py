from pysb.simulator.cutauleaping import CuTauLeapingSimulator
from pysb.simulator.stochkit import StochKitSimulator
from pysb.simulator.gpu_ssa import GPUSimulator
import numpy as np
from pysb.examples.schlogl import model as scholgl_model
from pysb.examples.michment import model as michment
from pysb.examples.kinase_cascade import model as kinase_model
import time
import matplotlib.pyplot as plt
import pandas as pd
from pysb.logging import setup_logger
import logging
from pysb.bng import generate_equations
import os
import socket
from pycuda.tools import  make_default_context
import pycuda.driver as cuda
computer_name = socket.gethostname()

import os
dev = os.environ.get("CUDA_DEVICE")
if dev is None:
    dev = cuda.Device(0)
gpu_name = dev.name()

# generate_equations(kinase_model)
# print(len(kinase_model.reactions))
# print(len(kinase_model.species))
# quit()

setup_logger(logging.INFO)
cur_dir = os.path.dirname(__file__)


needed_info = dict(model_name=None,
                   n_sim=None,
                   n_ts=None,
                   end_time=None,
                   sim_time=None
                   )


def run(n_sim, model, tspan, simulator='gpu_ssa'):
    if simulator == 'gpu_ssa':
        sim = GPUSimulator(model, tspan=tspan)
        st = time.time()
        sim.run_one_step(tspan, number_sim=n_sim)
        # sim.run(tspan, number_sim=n_sim)
        end_time = time.time()
        return end_time - st

    elif simulator == 'cutauleaping':
        sim = CuTauLeapingSimulator(model, tspan=tspan)
        st = time.time()
        sim.run(number_sim=n_sim, tspan=tspan)

        end_time = time.time()
        return end_time - st
    elif simulator == 'stochkit_eight_cpu_ssa':
        sim = StochKitSimulator(model, tspan=tspan)
        st = time.time()
        sim.run(num_processors=8, n_runs=n_sim, tspan=tspan,
                algorithm='ssa')
        end_time = time.time()
        return end_time - st
    elif simulator == 'stochkit_eight_cpu_tau':
        sim = StochKitSimulator(model, tspan=tspan)
        st = time.time()
        sim.run(num_processors=8, n_runs=n_sim, tspan=tspan,
                algorithm='tau_leaping')

        end_time = time.time()
        return end_time - st
    else:
        print("need simulator!")
        quit()


def run_model(model, t_end, n_timesteps):

    tspan = np.linspace(0, t_end, n_timesteps)
    n_sims = [2 ** i for i in range(7, 15)]
    all_timing = []
    info = needed_info.copy()
    info['model_name'] = model.name
    info['device_name'] = computer_name
    info['n_ts'] = n_timesteps
    info['end_time'] = t_end
    info['gpu_name'] = gpu_name

    def _run(sim_name):
        local_only = []
        for i in n_sims:
            local_info = info.copy()
            local_info['n_sim'] = i
            local_info['simulator'] = sim_name
            t_taken = run(i, model, tspan, sim_name)
            local_info['sim_time'] = t_taken
            all_timing.append(local_info)
            local_only.append(local_info)
        tmp_pd = pd.DataFrame(local_only)
        tmp_pd.to_csv(
            os.path.join(cur_dir, 'Timings',
                         '{}_{}_timing.csv'.format(sim_name, model.name)))

    _run('gpu_ssa')
    _run('cutauleaping')
    _run('stochkit_eight_cpu_ssa')
    _run('stochkit_eight_cpu_tau')

    return all_timing


if __name__ == "__main__":
    # """

    t1 = run_model(scholgl_model, 100, 101, )
    t = run_model(michment, 100, 101)
    t2 = run_model(kinase_model, 100, 101)
    df_1 = pd.DataFrame(t)
    df_2 = pd.DataFrame(t1)
    df_3 = pd.DataFrame(t2)
    df = pd.concat([df_1, df_2, df_3])

    df.to_csv('output.csv')
