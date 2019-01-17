import csv
import logging
import os
import socket

import numpy as np
import pandas as pd
# import pycuda.driver as cuda
import time

os.environ['CUDA_DEVICE'] = "0"
# os.environ['CUDAPATH'] = r"/opt/cuda/bin"
os.environ[
    'CUDAPATH'] = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\bin"

from pysb.simulator import BngSimulator, StochKitSimulator, GPUSimulator
from pysb.examples.schlogl import model as scholgl_model
# from pysb.simulator.cutauleaping import CuTauLeapingSimulator
from pysb.examples.kinase_cascade import model as kinase_model
# from pysb.examples.ras_camp_pka import model as ras_model
from pysb.logging import setup_logger

computer_name = socket.gethostname().lower()
dev = os.environ.get("CUDA_DEVICE")
# if dev is None:
#     dev = cuda.Device(dev)
#     gpu_name = dev.name()
# else:
dev = 'RTX2080'
gpu_name = 'RTX2080'

setup_logger(logging.INFO)
cur_dir = os.path.dirname(__file__)

needed_info = dict(model_name=None,
                   n_sim=None,
                   n_ts=None,
                   end_time=None,
                   sim_time=None
                   )


def write(row_dict):
    f_name = r'Timings/times.csv'
    fields = []
    values = []
    for key in sorted(row_dict):
        fields.append(key)
        values.append(row_dict[key])
    if os.path.exists(f_name):
        # with open(f_name, 'a', newline='') as f:
        with open(f_name, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(values)
    else:
        # with open(f_name, 'w', newline='') as f:
        with open(f_name, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(fields)
            writer.writerow(values)


def run(n_sim, model, tspan, simulator='gpu_ssa'):
    if simulator == 'gpu_ssa':
        sim = GPUSimulator(model, tspan=tspan, verbose=False)
        st = time.time()
        sim.run(tspan, number_sim=n_sim, threads=32)
        end_time = time.time()
        return end_time - st, sim._time

    elif simulator == 'bng':
        sim = BngSimulator(model, tspan=tspan, verbose=False)
        st = time.time()
        sim.run(n_runs=n_sim)
        eet = time.time()

        return eet - st, eet - st

    elif simulator == 'cutauleaping':
        sim = CuTauLeapingSimulator(model, tspan=tspan)
        st = time.time()
        sim.run(number_sim=n_sim, tspan=tspan)
        end_time = time.time()

        return end_time - st

    elif simulator == 'stochkit_eight_cpu_ssa':
        sim = StochKitSimulator(model, tspan=tspan)
        st = time.time()
        sim.run(num_processors=8, n_runs=n_sim, algorithm='ssa')
        end_time = time.time()
        return end_time - st

    elif simulator == 'stochkit_eight_cpu_tau':
        sim = StochKitSimulator(model, tspan=tspan)
        st = time.time()
        sim.run(num_processors=8, n_runs=n_sim, algorithm='tau_leaping')

        end_time = time.time()

        return end_time - st, sim._time
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
            total_time, t_taken = run(i, model, tspan, sim_name)
            local_info['sim_time'] = t_taken
            local_info['total_time'] = total_time
            all_timing.append(local_info)
            local_only.append(local_info)
            write(local_info)
        # return
        tmp_pd = pd.DataFrame(local_only)
        print(tmp_pd)
        tmp_pd.to_csv(
            os.path.join(cur_dir, 'Timings',
                         '{}_{}_{}_timing.csv'.format(computer_name,
                                                      sim_name,
                                                      model.name, )))

    # _run('cutauleaping')
    # _run('gpu_ssa')
    _run('bng')
    # _run('stochkit_eight_cpu_ssa')
    # _run('stochkit_eight_cpu_tau')

    return all_timing


def run2(n_sim, model, t_end, n_timesteps, threads):
    tspan = np.linspace(0, t_end, n_timesteps)
    sim = GPUSimulator(model, tspan=tspan, verbose=False)
    sim.run(tspan, number_sim=n_sim, threads=threads)
    print(n_sim, threads, sim._blocks, sim._time)


def run_tpb_test():
    n_sims = [2 ** i for i in range(7, 18)]
    for i in n_sims:
        for j in [2, 4, 8, 16, 32, 64, 128, 256]:
            try:
                run2(i, kinase_model, 100, 101, j)
            except:
                print("ops")


if __name__ == "__main__":
    run_model(scholgl_model, 100, 3, )
    run_model(scholgl_model, 100, 101, )
    quit()
    run_model(michment, 100, 101)

    # run_model(kinase_model, 100, 101)
    # run_model(earm_1, 20000, 101)
    run_model(ras_model, 20000, 101)
