import csv
import os

gpu_id = "0"
os.environ['CUDA_DEVICE'] = gpu_id

import logging
import socket
import numpy as np
import pandas as pd
import time

from pysb.simulator import BngSimulator, CUDASimulator, OpenCLSimulator
from pysb.logging import setup_logger
from pycuda.driver import Device
computer_name = socket.gethostname().lower()

gpu_name = ''.join([i for i in Device(int(gpu_id)).name().split(' ')
                    if i != 'GeForce'])
print("computer = {}".format(computer_name))
print("gpu = {}".format(gpu_name))

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
        open_type = 'a'
    else:
        open_type = 'w'
    with open(f_name, open_type) as f:
            writer = csv.writer(f)
            if open_type == 'a':
                writer.writerow(fields)
            writer.writerow(values)


def run(n_sim, model, tspan, simulator='gpu_ssa'):
    v = False
    if simulator == 'gpu_ssa':
        sim = CUDASimulator(model, tspan=tspan, verbose=v)
    elif simulator == 'cl':
        sim = OpenCLSimulator(model, tspan=tspan, verbose=v, device='gpu',
                              multi_gpu=False)
    elif simulator == 'bng':
        sim = BngSimulator(model, tspan=tspan, verbose=v)
    else:
        return
    st = time.time()
    sim.run(tspan, number_sim=n_sim)
    end_time = time.time()
    return end_time - st, sim._time


def run_model(model, t_end, n_timesteps):
    tspan = np.linspace(0, t_end, n_timesteps)
    n_sims = [2 ** i for i in range(7, 17)]
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


    _run('cl')
    # _run('gpu_ssa')
    # _run('bng')
    return all_timing


def run2(n_sim, model, t_end, n_timesteps, threads):
    tspan = np.linspace(0, t_end, n_timesteps)
    sim = CUDASimulator(model, tspan=tspan, verbose=False)
    sim.run(tspan, number_sim=n_sim, threads_per_block=threads)
    print(n_sim, threads, sim._time)


def run_tpb_test():
    n_sims = [2 ** i for i in range(7, 18)]
    for i in n_sims:
        for j in [2, 4, 8, 16, 32, 64, 128, 256]:
            try:
                run2(i, kinase_model, 100, 101, j)
            except:
                print("ooops")


if __name__ == "__main__":
    from pysb.examples.schlogl import model as scholgl_model
    from pysb.examples.michment import model as michment
    from pysb.examples.kinase_cascade import model as kinase_model
    from pysb.examples.ras_camp_pka import model as ras_model
    from pysb.examples.earm_1_0 import model as earm_1
    # 4.3341310024261475
    # run_model(scholgl_model, 100, 101, )
    # run_model(michment, 100, 101)
    run_model(kinase_model, 100, 101)
    run_model(earm_1, 20000, 101)
    # run_model(ras_model, 20000, 101)
