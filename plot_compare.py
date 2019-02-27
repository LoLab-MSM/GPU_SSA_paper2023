import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from data import gather_data

df = gather_data()

df.loc[df['gpu_name'] == 'gtx1080', 'gpu_name'] = 'GTX1080'
df.loc[df['gpu_name'] == 'GeForce GTX 1080', 'gpu_name'] = 'GTX1080'
df.loc[df['gpu_name'] == 'GeForce GTX 1060', 'gpu_name'] = 'GTX1060'
df.loc[df['gpu_name'] == 'TeslaV100-SXM2-16GB', 'gpu_name'] = 'TeslaV100'
df.loc[df['gpu_name'] == 'VOLTA_V100', 'gpu_name'] = 'TeslaV100'
df.loc[df['gpu_name'] == 'Tesla K20c', 'gpu_name'] = 'K20c'
df = df[df.n_sim < 2 ** 17].copy()
df = df[df.n_sim > 2 ** 8].copy()


def print_opts(d):
    print(d['device_name'].unique())
    print(d['model_name'].unique())
    print(d['gpu_name'].unique())
    print(d['simulator'].unique())


def compare_times(df, save_name="time_compare"):
    sns.catplot(
        x="n_sim", y="sim_time", hue="simulator", col="model_name",
        kind="point", data=df, sharey=False
    )

    plt.savefig('{}.png'.format(save_name), dpi=300, bbox_inches='tight')


def plot_ratio(data, save_name="bng_gpu_ratio"):
    """
    Plots the ratio of BNG run times to GPU_SSA run times
    """

    d = pd.pivot_table(
        data,
        index=['n_sim', 'model_name'],
        columns='simulator',
        values='total_time'
    )

    d['ratio'] = d['bng'] / d['gpu_ssa']

    d.reset_index(inplace=True)
    g = sns.catplot(x='n_sim', y='ratio', data=d, hue='model_name',
                    kind="point")

    g.set(yscale="log")

    plt.savefig('{}.png'.format(save_name), dpi=300, bbox_inches='tight')


"""
For now, i am having to subset data until the rest of the simulations are done.


BUU is the newest GPU. I think the paper stats can focus on these times
Thus, buu_only timing can be used for analysis in paper.
"""

"""

# waiting on kinase_cascase and ras_camp_pka to finish on cpu
models = [
    'pysb.examples.michment',
    'pysb.examples.schlogl',
    'pysb.examples.earm_1_0',
    # 'pysb.examples.kinase_cascade',
    # 'pysb.examples.ras_camp_pka'
]
buu_only = df.copy()
# buu_only = buu_only[buu_only['device_name'].isin(['buu', 'bad'])].copy()

buu_only = buu_only.loc[buu_only.model_name.isin(models)].copy()
print_opts(buu_only)

plot_ratio(buu_only)
compare_times(buu_only)

plt.show()
#"""



def compare_gpus():
    pal = sns.light_palette("purple", as_cmap=True)
    df_gpu = df.loc[df.simulator == 'gpu_ssa'].copy()

    models = [
        'pysb.examples.michment',
        'pysb.examples.schlogl',
        'pysb.examples.earm_1_0',
        'pysb.examples.kinase_cascade',
        # 'pysb.examples.ras_camp_pka'
    ]
    df_gpu = df_gpu.loc[df_gpu.model_name.isin(models)].copy()
    # df_gpu = df_gpu.loc[df_gpu.gpu_name.isin(['RTX2080', 'VOLTA_V100'])]

    fig = plt.figure(figsize=(12, 18))

    for n, m in enumerate(df_gpu.model_name.unique()):
        subset = df_gpu.loc[df_gpu['model_name'] == m].copy()
        d = pd.pivot_table(
            subset, index='gpu_name', columns='n_sim', values='sim_time',
            fill_value=np.nan,
        )
        d.sort_index(inplace=True)

        ax = fig.add_subplot(5, 1, n + 1)
        ax.set_title(m)
        sns.heatmap(data=d, cmap=pal, linewidths=0.01, vmin=0, annot=True,
                    fmt=".3f", ax=ax)

    plt.tight_layout()
    plt.savefig("compare_gpus.png", bbox_inches='tight', dpi=300)
    plt.show()


compare_gpus()
