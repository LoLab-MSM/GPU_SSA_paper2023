from pysb.simulator.cutauleaping import CuTauLeapingSimulator
from pysb.simulator.stochkit import StochKitSimulator
from pysb.simulator.gpu_ssa import GPUSimulator
import numpy as np
from pysb.examples.schlogl import model as scholgl_model
from pysb.examples.michment import model as michment
from pysb.simulator.scipyode import ScipyOdeSimulator
from pysb.examples.kinase_cascade import model as kinase_model
from pysb.examples.earm_1_0 import model as earm_model
from pysb.examples.ras_camp_pka import model as ras_model
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


def run_model(model, tspan, n_sim=1000, simulator='gpu_ssa', precision=np.float32):

    if simulator == 'gpu_ssa':
        sim = GPUSimulator(model, tspan=tspan, verbose=True, precision=precision)
        traj = sim.run_one_step(tspan=tspan, number_sim=n_sim, threads=32)
        # traj = sim.run_one_step(tspan=tspan, number_sim=n_sim, threads=32)
        # traj = sim.run_one_step(tspan=tspan, number_sim=n_sim, threads=64)
        # traj = sim.run_one_step(tspan=tspan, number_sim=n_sim, threads=128)
        # traj = sim.run_one_step(tspan=tspan, number_sim=n_sim, threads=256)
        # traj = sim.run_one_step(tspan=tspan, number_sim=n_sim, threads=512)
        # traj = sim.run(tspan=tspan, number_sim=n_sim, verbose=True, threads=32)
        return traj

    elif simulator == 'cutauleaping':
        sim = CuTauLeapingSimulator(model, tspan=tspan, cleanup=False)
        traj = sim.run(number_sim=n_sim, tspan=tspan)
        return traj
    elif simulator == 'stochkit_ssa':
        sim = StochKitSimulator(model, tspan=tspan)
        traj = sim.run(num_processors=8, n_runs=n_sim, tspan=tspan)
        return traj
    elif simulator == 'stochkit_tauleaping':
        sim = StochKitSimulator(model, tspan=tspan)
        traj = sim.run(num_processors=8, n_runs=n_sim, tspan=tspan,
                       algorithm='tau_leaping')
        return traj


if __name__ == "__main__":
    # """
    name = 'Product'


    opt = 'earm'

    #


    if opt == 'earm':
        sol = ScipyOdeSimulator(earm_model)
        tspan = np.linspace(0, 20000, 101)
        name = 'cSmac_total'
        # name = 'tBid_total'
        traj = sol.run(tspan=tspan)
        plt.figure(0)
        plt.plot(tspan, traj.observables[name], '--o', color='black', label='ode-lsoda')

        traj = run_model(earm_model, tspan, 500, simulator='gpu_ssa')
        # traj1 = run_model(earm_model, tspan, 50, simulator='gpu_ssa', precision=np.float64)
        # traj2 = run_model(earm_model, tspan, 500, simulator='cutauleaping')

    if opt =='ras':
        tspan = np.linspace(0, 1000, 101)
        traj = run_model(ras_model, tspan, 1000, simulator='gpu_ssa')
        traj1 = run_model(ras_model, tspan, 1000, simulator='gpu_ssa', precision=np.float64)
        traj2 = run_model(ras_model, tspan, 1000, simulator='cutauleaping')
        name = 'obs_cAMP'

    if opt == 'kinase':
        tspan = np.linspace(0, 100, 101)
        traj = run_model(kinase_model, tspan, 1000, simulator='gpu_ssa')
        traj1 = run_model(kinase_model, tspan, 100, simulator='stochkit_ssa')
        traj2 = run_model(kinase_model, tspan, 100, simulator='cutauleaping')
        name = 'ppMEK'

    if opt == 'scholgl':
        tspan = np.linspace(0, 100, 101)
        traj = run_model(scholgl_model, tspan, 2**12, simulator='gpu_ssa')
        traj = run_model(scholgl_model, tspan, 100, simulator='cutauleaping')
        traj = run_model(scholgl_model, tspan, 100, simulator='gpu_ssa', precision=np.float64)
        name = 'X_total'

    def plot(traj, num, title, color='blue'):
        result = np.array(traj.observables)[name]

        x = np.array([tr[:] for tr in result]).T
        t = np.array(traj.tout).T
        plt.figure(num)
        plt.title(name)

        # plt.plot(t, x, '0.5', lw=2, alpha=0.25)  # individual trajectories
        plt.plot(tspan, x.mean(axis=1), 'b', color=color, lw=3, label=title)
        # plt.plot(t, x.max(axis=1), 'b--', lw=2, label="min/max")
        # plt.plot(t, x.min(axis=1), 'b--', lw=2)

    plot(traj, 0, 'float', color='blue')
    # plot(traj1, 0, 'double', color='red')
    # plot(traj2, 0, 'double', color='green')
    plt.legend(loc=0)
    # plot(traj2, 1, 'gpu_ssa')
    plt.savefig("Comparing_precision.png")
    # plt.show()

    # t1 = run_model(scholgl_model, 100, 101, )
    # t = run_model(michment, 100, 101)
