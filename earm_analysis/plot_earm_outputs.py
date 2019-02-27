import pandas as pd
from pysb.examples.earm_1_0 import model as earm_model
import numpy as np
import scipy.interpolate
import matplotlib.pyplot as plt
import os
from pysb.simulator.scipyode import ScipyOdeSimulator
import pickle

directory = os.path.dirname(__file__)
data_path = os.path.join(directory, 'data',
                         'EC-RP_IMS-RP_IC-RP_data_for_models.csv')

exp_data = pd.read_csv(data_path, delimiter=',')

# print(exp_data)

momp_obs_total = earm_model.parameters['mSmac_0'].value
momp_data = np.array([9810.0, 180.0, momp_obs_total])
momp_var = np.array([7245000.0, 3600.0, 1e4])

n_sim = 1000
scale_1 = 0.8
scale_2 = 1.2

tspan = np.linspace(0, 20000, 201)

sim = ScipyOdeSimulator(earm_model, tspan)
traj = sim.run()
smac = np.array(traj.observables['cSmac_total'])
smac = smac / np.nanmax(smac)
st, sc, sk = scipy.interpolate.splrep(tspan, smac)
t10 = scipy.interpolate.sproot((st, sc - 0.10, sk))[0]
t90 = scipy.interpolate.sproot((st, sc - 0.90, sk))[0]
td_standard = (t10 + t90) / 2


def likelihood(traj):
    # traj = pd.pivot_table(traj, columns=['simulation'], index='time',
    #                       values='cSmac_total')
    time_of_death = []
    for t, trajectory in traj.groupby('simulation'):
        # if t > 100:
        #     continue
        smac = np.array(trajectory['cSmac_total'])

        smac = smac / np.nanmax(smac)
        st, sc, sk = scipy.interpolate.splrep(tspan, smac)
        try:
            t10 = scipy.interpolate.sproot((st, sc - 0.10, sk))[0]
            t90 = scipy.interpolate.sproot((st, sc - 0.90, sk))[0]
        except IndexError:
            t10 = 0
            t90 = 0
        td = (t10 + t90) / 2
        if td == 0:
            continue
        time_of_death.append(td)
    return np.array(time_of_death) / 3600.


default_data = pd.read_csv('earm_default.csv')
tod = likelihood(default_data)
avg_tod = np.average(tod)
std_tod = np.std(tod)
print("Average tod with no changes = {}".format(avg_tod))

"""
import matplotlib.mlab as mlab

# the histogram of the data
n, bins, patches = plt.hist(tod, 50, normed=1, facecolor='green',
                            alpha=0.75, histtype='bar')
mu = np.mean(tod)
sigma = np.std(tod)
# add a 'best fit' line
y = mlab.normpdf(bins, mu, sigma)
l = plt.plot(bins, y, 'r--', linewidth=1)
plt.title(r'$\mathrm{{Histogram\ of\ IQ:}}\ \mu={},\ \sigma={}$'.format(mu, sigma))
plt.tight_layout()
plt.savefig('avg_distribution.png')
plt.show()
# quit()
"""
"""
dict_of_values = {}
for each in earm_model.initial_conditions:
    param = each[1].name

    f_name = 'EARM_OUTPUT/earm_{}_{}.csv'.format(param, scale_1)
    f_name2 = 'EARM_OUTPUT/earm_{}_{}.csv'.format(param,scale_2)

    traj = pd.read_csv(f_name)[['time', 'simulation', 'cSmac_total']]
    value_1 = likelihood(traj)

    traj2 = pd.read_csv(f_name2)[['time', 'simulation', 'cSmac_total']]
    value_2 = likelihood(traj2)
    dict_of_values[param] = [value_1, value_2]

    print("{} change with increase by 1.2 = {}, decrease by .8 = {}"
          "".format(param, np.average(value_2), np.average(value_1)))
pickle.dump(dict_of_values, open('dict_of_values.p', 'w'))
"""

dict_of_values = pickle.load(open('dict_of_values.p', 'r'))
x_values = range(1, len(dict_of_values) + 1)

all_up = []
all_down = []

from matplotlib.pyplot import cm

color = cm.rainbow(np.linspace(0, 1, len(earm_model.initial_conditions)))

fig = plt.Figure()
fig2 = plt.Figure()
ax1 = fig.add_subplot(111)
ax2 = fig2.add_subplot(111)

for n, i in enumerate(sorted(dict_of_values)):
    up, down = dict_of_values[i]
    all_up.append(up)
    all_down.append(down)
    if n % 2 == 0:
        marker = 'o'
    else:
        marker = '*'
    ax1.plot(np.average(up) - avg_tod,
             np.average(down) - avg_tod,
             marker, c=color[n], label=i)
    ax2.plot(np.std(up) - std_tod,
             np.std(down) - std_tod,
             marker, c=color[n], label=i)

ax1.axhline(0, c="red", linewidth=2.5, zorder=0)
ax1.axvline(0, c="red", linewidth=2.5, zorder=0)

ax1.set_xlim(-.5, .5)
ax1.set_ylim(-.5, .5)
ax1.set_xlabel("$\Delta$ Initial condition by 1.2")
ax1.set_ylabel("$\Delta$ Initial condition by 0.8")
ax1.legend(loc=0)
fig.tight_layout()
fig.savefig("species_vs_up_down_avg.png")

ax2.axhline(0, c="red", linewidth=2.5, zorder=0)
ax2.axvline(0, c="red", linewidth=2.5, zorder=0)

ax2.set_xlim(-.2, .2)
ax2.set_ylim(-.2, .2)
ax2.set_xlabel("$\Delta$ Initial condition by 1.2")
ax2.set_ylabel("$\Delta$ Initial condition by 0.8")

ax2.legend(loc=0)
fig2.tight_layout()

fig2.savefig("species_vs_up_down_std.png")
# plt.show()


plt.figure(figsize=(8, 8))
# things that are up
ax1 = plt.subplot(211)
# plt.boxplot(all_up, 'og', showfliers=False)
plt.violinplot(all_up, showmeans=True)
ax1.axhline(avg_tod, c="red", linewidth=1.5, zorder=0)
plt.xticks(x_values, dict_of_values, rotation=90)

# things that are down
ax2 = plt.subplot(212)
# plt.boxplot(all_down, 'or', showfliers=False)
plt.violinplot(all_down, showmeans=True)
ax2.axhline(avg_tod, c="red", linewidth=1.5, zorder=0)
plt.xticks(x_values, dict_of_values, rotation=90)

plt.tight_layout()
plt.savefig('comparing_changes_in_pertubation.png')
# plt.show()
quit()
