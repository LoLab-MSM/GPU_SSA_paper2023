import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from data import gather_data

df = gather_data()

df['model_name'] = df['model_name'].str.split('pysb.examples.').str.get(-1)

df = df[df.n_sim < 2 ** 17].copy()
df = df[df.n_sim > 2 ** 7].copy()

df.loc[df['gpu_name'] == 'gtx1080', 'gpu_name'] = 'GTX1080'
df.loc[df['gpu_name'] == 'GeForce GTX 1080', 'gpu_name'] = 'GTX1080'
df.loc[df['gpu_name'] == 'GeForce GTX 1060', 'gpu_name'] = 'GTX1060'
df.loc[df['gpu_name'] == 'TeslaV100-SXM2-16GB', 'gpu_name'] = 'TeslaV100'
df.loc[df['gpu_name'] == 'VOLTA_V100', 'gpu_name'] = 'TeslaV100'
df.loc[df['gpu_name'] == 'Tesla K20c', 'gpu_name'] = 'K20c'

df.loc[df['device_name'] == 'bad.mc.vanderbilt.edu', 'device_name'] = 'bad'
df.loc[df['device_name'] == 'ip-172-31-22-73', 'device_name'] = 'aws'

df.loc[df['simulator'] == 'gpu_ssa', 'simulator'] = 'cuda'

crit = (df.simulator == 'stochkit') & (df.n_cpu == 64)
df.loc[crit, 'simulator'] = 'stochkit_64'

df.loc[(df['gpu_name'] == 'i5_6500T') &
       (df['simulator'] == 'cl_intel_gpu'),
       'gpu_name'] = 'HD530'

'''
df.loc[(df['gpu_name'] == 'RTX2080') &
       (df['simulator'] == 'cl_nvidia'),
       'gpu_name'] = 'RTX2080_cl'

df.loc[(df['gpu_name'] == 'TeslaV100') &
       (df['simulator'] == 'stochkit'),
       'gpu_name'] = 'POWER9_1_stochkit'

df.loc[(df['gpu_name'] == 'TeslaV100') &
       (df['simulator'] == 'stochkit_64'),
       'gpu_name'] = 'POWER9_64_stochkit'

df.loc[(df['gpu_name'] == 'TeslaV100') &
       (df['simulator'] == 'bng'),
       'gpu_name'] = 'POWER9_1_bng'

df.loc[(df['gpu_name'] == 'RTX2080') &
       (df['simulator'] == 'gpu_ssa'),
       'gpu_name'] = 'RTX2080_cuda'


df.loc[(df['gpu_name'] == 'TeslaV100') &
       (df['simulator'] == 'cl'),
       'gpu_name'] = 'TeslaV100_cl'
'''
df['sim_card'] = df['gpu_name'] + '_' + df['simulator']


def print_opts(d):
    print(d['device_name'].unique())
    print(d['model_name'].unique())
    print(d['gpu_name'].unique())
    print(d['simulator'].unique())


def compare_times(df, save_name="time_compare"):
    g = sns.catplot(
        x="n_sim", y="sim_time", hue="simulator", col="model_name",
        kind="point", data=df, sharey=False, col_wrap=2, height=4,
    )
    g.set_xticklabels(rotation=45)
    g.set_xlabels('')
    g.set_ylabels('')
    g.set(yscale="log")
    plt.savefig('{}.png'.format(save_name), dpi=300, bbox_inches='tight')
    plt.savefig('{}.pdf'.format(save_name), dpi=300, bbox_inches='tight')


def plot_ratio(data, save_name="bng_gpu_ratio"):
    """
    Plots the ratio of BNG run times to GPU_SSA run times
    """

    d = pd.pivot_table(
        data,
        index=['n_sim', 'model_name'],
        columns='simulator',
        values='sim_time'
    )

    d['ratio'] = d['bng'] / d['gpu_ssa']
    print(d[['ratio', 'bng', 'gpu_ssa']])
    d.reset_index(inplace=True)
    d.to_csv('all_times.csv')
    # quit()
    print(d.shape)
    print(d)
    g = sns.catplot(x='n_sim', y='ratio', data=d, hue='model_name',
                    kind="point", height=6, )
    g.set_xticklabels(rotation=45)
    g.set(yscale="log")
    plt.savefig('{}.png'.format(save_name), dpi=1200, bbox_inches='tight')
    plt.savefig('{}.pdf'.format(save_name), dpi=1200, bbox_inches='tight')


"""
For now, i am having to subset data until the rest of the simulations are done.


BUU is the newest GPU. I think the paper stats can focus on these times
Thus, buu_only timing can be used for analysis in paper.
"""

# """

# waiting on kinase_cascase and ras_camp_pka to finish on cpu
models = [
    'michment',
    'schlogl',
    'earm_1_0',
    'kinase_cascade',
]
buu_only = df.copy()
print_opts(buu_only)
buu_only = buu_only[buu_only['device_name'].isin(['bad'])].copy()

buu_only = buu_only.loc[buu_only.model_name.isin(models)].copy()
# buu_only = buu_only.loc[buu_only.gpu_name.isin(['RTX2080'])].copy()

simulator = [
    'gpu_ssa',
    # 'cl',
    'bng',
    'stochkit',
    'stochkit_64',
]
buu_only = buu_only.loc[buu_only.simulator.isin(simulator)].copy()


# earm_only = buu_only.loc[(buu_only.model_name == 'pysb.examples.earm_1_0') &
#                          (buu_only.simulator == 'stochkit_64')].copy()
# plt.plot(earm_only['n_sim'], earm_only['sim_time'], 'o-')
# plt.show()


# buu_only = buu_only.loc[buu_only.n_sim < 2049]
# plot_ratio(buu_only)
# compare_times(buu_only)
# plt.show()
# quit()


def compare_gpus():
    pal = sns.light_palette("purple", as_cmap=True)

    gpus = [
        'gpu_ssa', 'cl', 'cl_intel_gpu', 'cl_amd', 'cl_nvidia'
    ]
    print(df.simulator.unique())
    # df_gpu = df.loc[df.simulator.isin(gpus)].copy()
    df_gpu = df.copy()
    models = [
        'michment',
        'schlogl',
        'kinase_cascade',
        'earm_1_0',
    ]
    df_gpu = df_gpu.loc[df_gpu.model_name.isin(models)].copy()
    df_gpu = df_gpu.loc[~df_gpu.gpu_name.isin(['GTX1060'])]

    fig = plt.figure(figsize=(12, 18))

    for n, m in enumerate(models):
        subset = df_gpu.loc[df_gpu['model_name'] == m].copy()

        d = pd.pivot_table(
            subset, index='sim_card', columns='n_sim', values='sim_time',
            fill_value=np.nan,
        )
        print(d.index.values)
        # quit()
        # d.reindex([])
        keep = ['POWER9_1_bng',
                'POWER9_1_stochkit',
                'POWER9_64_stochkit',
                'i5_6500T',
                'Ryzen_1600x',
                'HD530',
                'HD7970',
                'GTX980Ti',
                'GTX1080',
                'RTX2080_cuda',
                'RTX2080_cl',
                'TeslaV100',
                'TeslaV100_cl'
                ]
        # keep = list(set(d.index.values).intersection(set(keep)))
        # d = d.reindex(keep)
        # d.sort_index(inplace=True)

        ax = fig.add_subplot(5, 1, n + 1)
        ax.set_title(m)
        sns.heatmap(data=d, cmap=pal, linewidths=0.01, vmin=0, annot=True,
                    fmt=".3f", ax=ax)

    plt.tight_layout()
    plt.savefig("compare_gpus.png", bbox_inches='tight', dpi=300)
    plt.show()


compare_gpus()
