from pysb.simulator.cutauleaping import CuTauLeapingSimulator
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


setup_logger(logging.INFO)


needed_info = dict(model_name=None,
                   n_sim=None,
                   n_ts=None,
                   end_time=None,
                   sim_time=None
                   )


def run(n_sim, model, tspan, simulator='gpu_ssa'):
    if simulator == 'gpu_ssa':
        sim = GPUSimulator(model, tspan=tspan)
    elif simulator == 'cutauleaping':
        sim = CuTauLeapingSimulator(model, tspan=tspan)
    else:
        print("need simulator!")
        quit()
    st = time.time()
    sim.run(number_sim=n_sim, tspan=tspan)

    end_time = time.time()
    return end_time - st



def run_model(model, t_end, n_timesteps):

    tspan = np.linspace(0, t_end, n_timesteps)
    n_sims = [2 ** i for i in range(7, 16)]
    all_timing = []
    info = needed_info.copy()
    info['model_name'] = model.name

    info['n_ts'] = n_timesteps
    info['end_time'] = t_end
    # """
    sim = 'cutauleaping'
    for i in n_sims:
        local_info = info.copy()
        local_info['n_sim'] = i
        local_info['simulator'] = sim
        t_taken = run(i, model, tspan, sim)
        local_info['sim_time'] = t_taken
        all_timing.append(local_info)
    # """
    sim = 'gpu_ssa'
    for i in n_sims:
        local_info = info.copy()
        local_info['n_sim'] = i
        local_info['simulator'] = sim
        t_taken = run(i, model, tspan, sim)
        local_info['sim_time'] = t_taken
        all_timing.append(local_info)

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
    # """
    df = pd.read_csv('output.csv')
    plt.figure(figsize=(8,4))
    for n, (model, data) in enumerate(df.groupby('model_name')):
        df1 = data[data['simulator'] == 'gpu_ssa']
        df2 = data[data['simulator'] == 'cutauleaping']
        plt.subplot(1, 3, n + 1)
        plt.title(model)
        plt.plot(np.log2(df1['n_sim']), df1['sim_time'], '*', label='gpu_ssa')
        plt.plot(np.log2(df2['n_sim']), df2['sim_time'], 'o', label='cutauleaping', alpha=0.5)
        # plt.xscale('log', base=2)
        # plt.yscale('log', base=10)
        plt.xlabel('2**N')
        plt.ylabel("Total time (IO + simulation)")
        plt.legend(loc=0)
    plt.tight_layout()
    plt.savefig('gpu_ssa_vs_cutauleaping.png')
    # plt.show()
    plt.close()
