from pysb.simulator.stochkit import StochKitSimulator
from pysb.simulator.scipyode import ScipyOdeSimulator
from pysb.simulator.gpu_ssa import GPUSimulator
import numpy as np
from pysb.examples.earm_1_0 import model as earm_model
import matplotlib.pyplot as plt
import pandas as pd
from pysb.logging import setup_logger
import logging

setup_logger(logging.INFO)


def run_model(param='Bax_0', scale=1.0,  n_sim=1000, simulator='gpu_ssa', precision=np.float32):
    name = 'cSmac_total'
    # name = 'tBid_total'
    tspan = np.linspace(0, 20000, 201)
    default_value = earm_model.parameters[param].value
    earm_model.parameters[param].value = default_value * scale

    if simulator == 'gpu_ssa':
        sim = GPUSimulator(earm_model, tspan=tspan, verbose=True, precision=precision)
        traj = sim.run_one_step(tspan=tspan, number_sim=n_sim, threads=32)

    elif simulator == 'stochkit_ssa':
        sim = StochKitSimulator(earm_model, tspan=tspan)
        traj = sim.run(num_processors=8, n_runs=n_sim, tspan=tspan)

    elif simulator == 'stochkit_tauleaping':
        sim = StochKitSimulator(earm_model, tspan=tspan)
        traj = sim.run(num_processors=8, n_runs=n_sim, tspan=tspan,
                       algorithm='tau_leaping')
    df = traj.dataframe
    df.to_csv("earm_{}_{}.csv.gz".format(param, scale), compression='gzip')
    result = np.array(traj.observables)[name]

    x = np.array([tr[:] for tr in result]).T
    save_name = "earm_{}_{}.png".format(param, scale)

    plt.title("Param {} scaled by {}".format(param, scale))

    print("Done plotting {}".format(save_name))
    plt.plot(tspan, x, '0.5', lw=2, alpha=0.25)  # individual trajectories
    plt.plot(tspan, x.mean(axis=1), 'b', lw=3, label='mean')
    plt.plot(tspan, x.max(axis=1), 'b--', lw=2, label="min/max")
    plt.plot(tspan, x.min(axis=1), 'b--', lw=2)

    sol = ScipyOdeSimulator(earm_model, tspan)
    traj = sol.run()

    plt.plot(tspan, np.array(traj.observables)[name], label='ode')

    plt.legend(loc=0)

    plt.savefig(save_name)
    print("Done plotting {}".format(save_name))
    # plt.show()
    plt.close()
    earm_model.parameters[param].value = default_value

if __name__ == "__main__":
    n_sim = 1000
    for each in earm_model.initial_conditions:
        print(each[1].name)
        run_model(each[1].name, scale=0.8, n_sim=n_sim, simulator='gpu_ssa')
        run_model(each[1].name, scale=1.2, n_sim=n_sim, simulator='gpu_ssa')
    quit()

    run_model('Bax_0', scale=0.8,  n_sim=n_sim, simulator='gpu_ssa')
    run_model('Bax_0', scale=1.2,  n_sim=n_sim, simulator='gpu_ssa')

    run_model('pC8_0', scale=0.8,  n_sim=n_sim, simulator='gpu_ssa')
    run_model('pC8_0', scale=1.2,  n_sim=n_sim, simulator='gpu_ssa')

    run_model('Bcl2c_0', scale=0.8,  n_sim=n_sim, simulator='gpu_ssa')
    run_model('Bcl2c_0', scale=1.2,  n_sim=n_sim, simulator='gpu_ssa')

    run_model('pR_0', scale=0.8,  n_sim=n_sim, simulator='gpu_ssa')
    run_model('pR_0', scale=1.2,  n_sim=n_sim, simulator='gpu_ssa')



