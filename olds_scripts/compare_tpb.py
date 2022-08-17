import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pysb.examples.schlogl import model as scholgl_model

from pysb.examples.kinase_cascade import model as kinase_model
from pysb.simulator import CUDASimulator, OpenCLSimulator

"""

Code to look how changing threads per block affects run times.

Results: Use 32 threads per block!


"""


def run_tpb_test():
    def run_tpb(n_sim, model, threads):
        sim = CUDASimulator(
            model, tspan=tspan, verbose=False
        ).run(number_sim=n_sim, threads_per_block=threads)
        print(n_sim, threads, sim._time)

    tspan = np.linspace(0, 100, 101)
    n_sims = [2 ** i for i in range(7, 18)]
    for i in n_sims:
        for j in [2, 4, 8, 16, 32, 64, 128, 256]:
            try:
                run_tpb(i, kinase_model, j)
            except:
                print("ooops")


def run_tpb_test2():
    def run_work_per_group(n_sim, model, threads):

        sim = OpenCLSimulator(
            model, tspan=tspan, verbose=False, device='cpu', platform='AMD'
        ).run(tspan, number_sim=n_sim, d_size=threads)
        print(n_sim, threads, sim._time)

    tspan = np.linspace(0, 100, 101)
    n_sims = [2 ** i for i in range(10, 11)]
    for i in n_sims:
        for j in [1, 32, 64, 256, i]:
            try:
                run_work_per_group(i, scholgl_model, j)
            except Exception as e:
                print(e)
                print("ooops")


if __name__ == '__main__':
    df = pd.read_csv('Timings/gtx1070_tpb_times.csv',
                     names=['n_sim', 'tpb', 'block', 'time'], delimiter=' ')

    print(df)
    df['time'] = np.log2(df['time'])
    ax2 = sns.catplot(x='n_sim', y='time', hue='tpb', kind="point", data=df)
    ax2.set(yscale="log")
    plt.show()
