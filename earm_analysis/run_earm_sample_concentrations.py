import os

os.environ['CUDA_DEVICE'] = "1"
import logging

import matplotlib.pyplot as plt
import numpy as np

from pysb.examples.earm_1_0 import model as earm_model
from pysb.logging import setup_logger
from pysb.simulator.gpu_ssa import GPUSimulator
from pysb.simulator.scipyode import ScipyOdeSimulator
from pysb.simulator.stochkit import StochKitSimulator


setup_logger(logging.INFO)

# options for observables
obs = ['cSmac_total', 'tBid_total', 'CPARP_total',
       'Bid_unbound', 'PARP_unbound', 'mSmac_unbound']

name = 'cSmac_total'
# name = 'tBid_total'
tspan = np.linspace(0, 20000, 201)


def run_model(ic1=None, factor1=1.0, ic2=None, factor2=1.0,
              n_sim=5000, simulator='gpu_ssa',
              precision=np.float64):

    if ic1 is not None:
        ic1_default_value = earm_model.parameters[ic1].value
        earm_model.parameters[ic1].value = ic1_default_value * factor1

    if ic2 is not None:
        ic2_default_value = earm_model.parameters[ic2].value
        earm_model.parameters[ic2].value = ic2_default_value * factor2

    if simulator == 'gpu_ssa':
        sim = GPUSimulator(earm_model, tspan=tspan, verbose=True,
                           precision=precision)
        # can run with one_step for personal pc. Prevents timeout
        # traj = sim.run_one_step(tspan=tspan, number_sim=n_sim, threads=32)
        traj = sim.run(tspan=tspan, number_sim=n_sim, threads=32)

    elif simulator == 'stochkit_ssa':
        sim = StochKitSimulator(earm_model, tspan=tspan)
        traj = sim.run(num_processors=8, n_runs=n_sim, tspan=tspan)

    elif simulator == 'stochkit_tauleaping':
        sim = StochKitSimulator(earm_model, tspan=tspan)
        traj = sim.run(num_processors=8, n_runs=n_sim, tspan=tspan,
                       algorithm='tau_leaping')

    df = traj.dataframe.reset_index()
    df = df[['simulation', 'time'] + obs]
    save_name = 'conc_sample'

    if ic1 is not None:
        save_name += '_{}_{}'.format(ic1, factor1)
        earm_model.parameters[ic1].value = ic1_default_value
    if ic2 is not None:
        save_name += '_{}_{}'.format(ic2, factor2)
        earm_model.parameters[ic2].value = ic2_default_value

    df.to_csv('{}.csv.gz'.format(save_name), index=False)
    plot(tspan, traj, name, save_name + '.png')


def plot(tspan, traj, obs, save_name, title=None):
    if title is not None:
        plt.title(title)

    result = np.array(traj.observables)[obs]
    x = np.array([tr[:] for tr in result]).T

    time_in_hours = tspan / 60. / 60.
    plt.plot(time_in_hours, x, '0.5', lw=2,
             alpha=0.25)  # individual trajectories
    plt.plot(time_in_hours, x.mean(axis=1), 'b', lw=3, label='mean')
    plt.plot(time_in_hours, x.max(axis=1), 'b--', lw=2, label="min/max")
    plt.plot(time_in_hours, x.min(axis=1), 'b--', lw=2)

    sol = ScipyOdeSimulator(earm_model, tspan)
    traj = sol.run()

    plt.plot(time_in_hours, np.array(traj.observables)[obs], c='red',
             label='ode')

    plt.legend(loc=0)

    plt.savefig(save_name)
    print("Done plotting {}".format(save_name))
    # plt.show()
    plt.close()


if __name__ == "__main__":

    # number of simulations

    n_sim = 2 ** 8
    run_model(n_sim=n_sim, simulator='gpu_ssa')
    quit()
    # does single parameter scan for all initial conditions
    for each in earm_model.initial_conditions:
        print(each[1].name)
        run_model(each[1].name, factor1=0.8, n_sim=n_sim, simulator='gpu_ssa')
        run_model(each[1].name, factor1=1.2, n_sim=n_sim, simulator='gpu_ssa')

    for ic1 in earm_model.initial_conditions:
        name1 = ic1[1].name
        for ic2 in earm_model.initial_conditions:
            name2 = ic2[1].name
            if name1 == name2:
                continue
            else:
                run_model(ic1=name1, factor1=0.8,
                          ic2=name2, factor2=1.8,
                          n_sim=n_sim,
                          simulator='gpu_ssa')
