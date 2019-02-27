from pysb.simulator.scipyode import ScipyOdeSimulator
from pysb.simulator.gpu_ssa import GPUSimulator
import numpy as np
from pysb.examples.earm_1_0 import model
import matplotlib.pyplot as plt
from pysb.logging import setup_logger
import logging

setup_logger(logging.INFO)

obs = ['cSmac_total', 'tBid_total', 'CPARP_total',
       'Bid_unbound', 'PARP_unbound', 'mSmac_unbound']


def run(n_sim=1000):
    tspan = np.linspace(0, 20000, 201)
    name = 'tBid_total'
    # model.parameters['L_0'].value = 100
    # 2.48300004005s 2.42300009727s old way
    # 2.33200001717s 2.11500000954s

    sim = GPUSimulator(model, tspan=tspan, verbose=True,
                       precision=np.float64)
    traj = sim.run(tspan=tspan, number_sim=n_sim, threads=32)
    # quit()

    for name in obs:
        result = traj.dataframe[name]
        tout = result.index.levels[1].values
        result = result.unstack(0)
        result = result.as_matrix()

        x = np.array([tr[:] for tr in result]).T

        plt.plot(tout, x, '0.5', lw=2, alpha=0.25)  # individual trajectories
        plt.plot(tout, x.mean(axis=1), 'b', lw=3, label='mean')
        plt.plot(tout, x.max(axis=1), 'b--', lw=2, label="min/max")
        plt.plot(tout, x.min(axis=1), 'b--', lw=2)

        sol = ScipyOdeSimulator(model, tspan)
        traj = sol.run()

        plt.plot(tspan, np.array(traj.observables)[name], 'k-', label='ode')

        plt.legend(loc=0)
        plt.tight_layout()
        plt.savefig('example_ssa_earm_{}.png'.format(name), dpi=200,
                    bbox_inches='tight')
        plt.close()


if __name__ == "__main__":
    n_sim = 2 ** 12
    run(n_sim)
