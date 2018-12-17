import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

all_df = []

for i in os.listdir('Timings'):
    all_df.append(pd.read_csv('Timings/' + i, index_col=0))

df = pd.concat(all_df)
print(df['device_name'].unique())
print(df['model_name'].unique())
print(df['gpu_name'].unique())
print(df['simulator'].unique())

df.loc[df['gpu_name'] == 'gtx1080', 'gpu_name'] = 'GTX1080'
df.loc[df['gpu_name'] == 'GeForce GTX 1080', 'gpu_name'] = 'GTX1080'
# df = df[df['n_sim'] > 2 ** 7]


# df = df.loc[df.gpu_name != 'Tesla K20c']
# df = df[df['device_name'] == 'buu']

# df = df.loc[df.simulator == 'gpu_ssa']
# df = df.loc[df.model_name.isin(['pysb.examples.schlogl', 'pysb.examples.michment'])].copy()
# df = df.loc[df.device_name.isin(['mule', 'piccolo', 'puma'])].copy()
# df = df.loc[df.simulator.isin(['gpu_ssa', 'cutauleaping', 'stochkit_eight_cpu_ssa', 'stochkit_eight_cpu_tau'])].copy()
#
# g = sns.catplot(
#     x="n_sim", y="sim_time", hue="simulator", col="model_name",  row='device_name',
#     kind="point", data=df, sharey=False
# )
# plt.show()
# plt.savefig('compare_gpu_to_bng.png', dpi=300, bbox_inches='tight')
# quit()
# plt.close()
df = df[df['device_name'] == 'buu']
df = df.loc[df.model_name.isin(
    ['pysb.examples.schlogl', 'pysb.examples.michment'])].copy()

d = pd.pivot_table(df, index=['n_sim', 'model_name'], columns='simulator',
                   values='total_time')

d['ratio'] = d['bng'] / d['gpu_ssa']
print(d)
d.reset_index(inplace=True)
print(d)

g = sns.catplot(x='n_sim', y='ratio', data=d, hue='model_name', kind="point", )

g.set(yscale="log")
plt.tight_layout()
plt.show()
print(d)


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
