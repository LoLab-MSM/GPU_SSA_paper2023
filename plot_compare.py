import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from data import gather_data

df = gather_data()

df.loc[df['gpu_name'] == 'gtx1080', 'gpu_name'] = 'GTX1080'
df.loc[df['gpu_name'] == 'GeForce GTX 1080', 'gpu_name'] = 'GTX1080'


def print_opts(d):
    print(d['device_name'].unique())
    print(d['model_name'].unique())
    print(d['gpu_name'].unique())
    print(d['simulator'].unique())


def compare_times(df, save_name=None):
    sns.catplot(
        x="n_sim", y="sim_time", hue="simulator", col="model_name",
        kind="point", data=df, sharey=False
    )
    if save_name is not None:
        plt.savefig('{}.png'.format(save_name), dpi=300, bbox_inches='tight')


def plot_ratio(data, save_name=None):
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
    if save_name is not None:
        plt.savefig('{}.png'.format(save_name), dpi=300, bbox_inches='tight')


"""
For now, i am having to subset data until the rest of the simulations are done.


BUU is the newest GPU. I think the paper stats can focus on these times
Thus, buu_only timing can be used for analysis in paper.
"""

buu_only = df[df['device_name'] == 'buu'].copy()

# waiting on kinase_cascase and ras_camp_pka to finish on cpu
models = [
    'pysb.examples.michment',
    'pysb.examples.schlogl',
    'pysb.examples.earm_1_0',
    # 'pysb.examples.kinase_cascade',
    # 'pysb.examples.ras_camp_pka'
]


buu_only = buu_only.loc[buu_only.model_name.isin(models)].copy()
print_opts(buu_only)
plot_ratio(buu_only)
compare_times(buu_only)

plt.show()

quit()
# Looking at mule results (OLD, compared other SSA methods
mule_only = df[df['device_name'] == 'mule'].copy()
mule_only = mule_only.loc[mule_only.model_name.isin(models)].copy()
sim_subset = ['cutauleaping', 'gpu_ssa', 'stochkit_eight_cpu_ssa', 'stochkit_eight_cpu_tau']
mule_only = mule_only.loc[mule_only.simulator.isin(sim_subset)].copy()


compare_times(mule_only)





def compare_gpus():
    pal = sns.light_palette("purple", as_cmap=True)
    for m in df.model_name.unique():
        subset = df.loc[df['model_name'] == m].copy()
        d = pd.pivot_table(
            subset, index='gpu_name', columns='n_sim', values='sim_time'
        )
        d.sort_index(inplace=True)
        fig = plt.figure(figsize=(10, 4))
        ax = fig.add_subplot(111)
        sns.heatmap(data=d, cmap=pal, linewidths=0.01, vmin=0, annot=True,
                    fmt=".2f", ax=ax)
        plt.tight_layout()
        plt.show()
