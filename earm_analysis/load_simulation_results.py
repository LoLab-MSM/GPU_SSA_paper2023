from pysb.examples.earm_1_0 import model as earm_model
import numpy as np
import scipy.interpolate
import matplotlib.pyplot as plt
import os
from pysb.simulator.scipyode import ScipyOdeSimulator
import pickle
from matplotlib.pyplot import cm
import seaborn as sns
import pandas as pd

dir_path = os.path.dirname(__file__)
data_path = os.path.join(dir_path, 'data',
                         'EC-RP_IMS-RP_IC-RP_data_for_models.csv')

exp_data = pd.read_csv(data_path, delimiter=',')

# print(exp_data)
color = cm.rainbow(np.linspace(0, 1, len(earm_model.initial_conditions)))

momp_obs_total = earm_model.parameters['mSmac_0'].value
momp_data = np.array([9810.0, 180.0, momp_obs_total])
momp_var = np.array([7245000.0, 3600.0, 1e4])

tspan = np.linspace(0, 20000, 201)

sim = ScipyOdeSimulator(earm_model, tspan)
traj = sim.run()
smac = np.array(traj.observables['cSmac_total'])
smac = smac / np.nanmax(smac)
st, sc, sk = scipy.interpolate.splrep(tspan, smac)
t10_0 = scipy.interpolate.sproot((st, sc - 0.10, sk))[0]
t90_0 = scipy.interpolate.sproot((st, sc - 0.90, sk))[0]
td_standard = (t10_0 + t90_0) / 2


def likelihood(traj, time=tspan):
    time_of_death = []
    for t, trajectory in traj.groupby('simulation'):

        smac = np.array(trajectory['cSmac_total'])
        smac = smac / np.nanmax(smac)
        st, sc, sk = scipy.interpolate.splrep(time, smac)
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
tod = likelihood(default_data, np.linspace(0, 20000, 201))
avg_tod = np.average(tod)

quart_25 = np.percentile(tod, 25)
quart_50 = np.percentile(tod, 50)
quart_75 = np.percentile(tod, 75)
std_tod = np.std(tod)


def calculate_tod(directory, save_name, scale_1, scale_2):
    dict_of_values = {}
    for each in earm_model.initial_conditions:
        param = each[1].name

        f_name = '{}/earm_{}_{}.csv.gz'.format(directory, param, scale_1)
        f_name2 = '{}/earm_{}_{}.csv.gz'.format(directory, param, scale_2)

        traj = pd.read_csv(f_name)[['time', 'simulation', 'cSmac_total']]
        value_1 = likelihood(traj)

        traj2 = pd.read_csv(f_name2)[['time', 'simulation', 'cSmac_total']]
        value_2 = likelihood(traj2)
        dict_of_values[param] = [value_1, value_2]

        print("{} change with increase by {} = {}, decrease by {} = {}"
              "".format(param, scale_2, np.average(value_2), scale_1,
                        np.average(value_1)))
    pickle.dump(dict_of_values, open('{}.p'.format(save_name), 'w'))
    return dict_of_values


def calculate_tod_pair():
    file_names = [
        'earm_bax_0.8_bcl2_0.8.csv.gz',
        'earm_bax_0.8_bcl2_1.2.csv.gz',
        'earm_bax_1.2_bcl2_0.8.csv.gz',
        'earm_bax_1.2_bcl2_1.2.csv.gz'
    ]
    all_data = []
    for i in file_names:
        traj = pd.read_csv(i)[['time', 'simulation', 'cSmac_total']]
        df = pd.DataFrame()
        df['tod'] = likelihood(traj)
        df['name'] = i.replace('csv.gz', '').replace('earm_', '')
        all_data.append(df)

    new_d = pd.DataFrame()
    new_d['tod'] = tod
    new_d['name'] = 'default'
    all_data.append(new_d)
    newdf = pd.concat(all_data)
    print(newdf['name'].unique())
    print(newdf.head(10))
    sns.violinplot(x='tod', y="name", data=newdf, palette="muted", )
    plt.tight_layout()
    plt.savefig('double_ko_bax_bcl2.png', dpi=200)
    plt.show()
    return all_data


def load_data(dict_name):
    dict_of_values = pickle.load(open('{}.p'.format(dict_name), 'r'))

    all_list = []

    for n, i in enumerate(sorted(dict_of_values)):
        up, down = dict_of_values[i]
        for t in up:
            all_list.append(dict(ic=i, tod=t, direction='up'))
        for t in down:
            all_list.append(dict(ic=i, tod=t, direction='down'))

    all_df = pd.DataFrame(all_list, columns=['ic', 'tod', 'direction'])
    all_df['change'] = dict_name
    all_df.to_csv('{}_data.csv'.format(dict_name))
    return all_df


def plot_violin(data, save_name):
    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(111)
    sns.violinplot(x="ic", y="tod", hue="direction",
                   data=data, palette="muted",
                   split=True,
                   ax=ax
                   )

    plt.tight_layout()
    plt.xticks(rotation=90)
    ax.axhline(quart_25, c="red", linewidth=1.5, zorder=0)
    ax.axhline(quart_50, c="red", linewidth=1.5, zorder=0)
    ax.axhline(quart_75, c="red", linewidth=1.5, zorder=0)

    plt.savefig('violin_plot_{}.png'.format(save_name),
                bbox_inches='tight',
                dpi=150
                )
    plt.close()


def plot_avg_std(data):
    fig = plt.figure(figsize=(8, 8))

    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    n = -1
    for ic, row in data.groupby('ic'):
        n += 1
        up = row[row['direction'] == 'up']['tod']
        down = row[row['direction'] == 'down']['tod']
        if n % 2 == 0:
            marker = 'o'
        else:
            marker = '*'
        ax1.plot(up.mean() - avg_tod,
                 down.mean() - avg_tod,
                 marker, c=color[n], label=ic)

        ax2.plot(up.std() - std_tod,
                 down.std() - std_tod,
                 marker, c=color[n], label=ic)

    ax2.legend(loc='center left', bbox_to_anchor=(1, 0.75))
    plt.savefig('std_vs_avg.png', bbox_inches='tight')
    # plt.show()


def plot_violin_mpl(data, save_name):
    all_up = dict(
        data[data['direction'] == 'up'].groupby('ic')['tod'].apply(
            np.array))

    all_down = dict(
        data[data['direction'] == 'down'].groupby('ic')['tod'].apply(
            np.array))

    up = [value for (key, value) in sorted(all_up.items())]
    down = [value for (key, value) in sorted(all_down.items())]
    plt.figure(figsize=(8, 4))
    # things that are up
    ax1 = plt.subplot(111)
    parts = plt.violinplot(up,
                           showmeans=True,
                           showextrema=False,
                           showmedians=False
                           )

    for pc in parts['bodies']:
        pc.set_facecolor('blue')
        pc.set_edgecolor('black')
        pc.set_alpha(1)

    # things that are down
    parts = plt.violinplot(down,
                           showmeans=True,
                           showextrema=False,
                           showmedians=False
                           )
    for pc in parts['bodies']:
        pc.set_facecolor('red')
        pc.set_edgecolor('black')
        pc.set_alpha(.5)

    ax1.axhline(quart_25, c="red", linewidth=1.5, zorder=0)
    ax1.axhline(quart_50, c="red", linewidth=1.5, zorder=0)
    ax1.axhline(quart_75, c="red", linewidth=1.5, zorder=0)
    x_values = range(1, len(all_up.keys()) + 1)
    plt.xticks(x_values, sorted(all_up.keys()), rotation=90)

    plt.tight_layout()
    plt.savefig('comparing_changes_in_pertubation{}.png'.format(save_name),
                bbox_inches='tight', dpi=150)


def plot_results(dict_name):
    data = load_data(dict_name=dict_name)
    plot_avg_std(data)
    # plot_violin(data, dict_name)
    # plot_violin_mpl(data, dict_name)
    # plot_avg_std()


def load_all_data():
    df_1 = pd.read_csv('del10_data.csv')
    df_2 = pd.read_csv('del20_data.csv')
    df_3 = pd.read_csv('del25_data.csv')
    df_4 = pd.read_csv('del50_data.csv')
    all_df = pd.concat([df_1, df_2, df_3, df_4])

    hits = [
        'Bax_0',
        'Bcl2_0',
        # 'Bid_0',
        # 'L_0',
        # 'pR_0',
        # 'pC8_0',

    ]
    # all_df = all_df[all_df['ic'].isin(hits)]
    fig = plt.figure(figsize=(4, 8))
    ax1 = fig.add_subplot(211)
    sns.violinplot(x="change", y="tod", hue="direction",
                   data=all_df[all_df['ic'] == 'pC6_0'],
                   palette="muted",
                   split=True,
                   ax=ax1
                   )

    plt.tight_layout()
    plt.xticks(rotation=90)
    plt.ylabel('Time of death')
    ax1.axhline(quart_25, c="red", linewidth=1.5, zorder=0)
    ax1.axhline(quart_50, c="red", linewidth=1.5, zorder=0)
    ax1.axhline(quart_75, c="red", linewidth=1.5, zorder=0)

    ax2 = fig.add_subplot(212)
    sns.violinplot(x="change", y="tod", hue="direction",
                   data=all_df[all_df['ic'] == 'pC9_0'],
                   palette="muted",
                   split=True,
                   ax=ax2
                   )

    # sns.factorplot(x="change", y="tod",
    #                hue="direction", col="ic",
    #                data=all_df, kind="violin", split=True,
    #                size=8, aspect=.7)

    plt.tight_layout()
    plt.xticks(rotation=90)
    plt.ylabel('Time of death')
    ax2.axhline(quart_25, c="red", linewidth=1.5, zorder=0)
    ax2.axhline(quart_50, c="red", linewidth=1.5, zorder=0)
    ax2.axhline(quart_75, c="red", linewidth=1.5, zorder=0)

    plt.savefig('violin_plot_all_time_no_change.png',
                bbox_inches='tight',
                dpi=150
                )
    plt.close()


if __name__ == '__main__':
    # calculate_tod('EARM_CHANGING_IC_BY_.1', 'del10', .9, 1.1)
    # calculate_tod('EARM_CHANGING_IC_BY_.25', 'del25', .75, 1.25)
    # calculate_tod('EARM_OUTPUT', 'del20', .8, 1.2)
    calculate_tod_pair()
    quit()

    # plot_results('del10')
    # plot_results('del20')
    # plot_results('del25')
    # plot_results('del50')

    load_all_data()
