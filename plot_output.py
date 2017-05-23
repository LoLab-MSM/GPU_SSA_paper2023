import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

all_df = []

for i in os.listdir('Timings'):
    all_df.append(pd.read_csv('Timings/' + i))

df = pd.concat(all_df)
print(df['device_name'].unique())
print(df['model_name'].unique())
print(df['gpu_name'].unique())
print(df['simulator'].unique())
df.loc[df['gpu_name'] == 'gtx1080', 'gpu_name'] = 'GeForce GTX 1080'
df.loc[df['gpu_name'] == 'K20c', 'gpu_name'] = 'Tesla K20c'
# df = df[df['device_name'] == 'puma']
df = df.sort_values(['model_name', 'simulator'], ascending=True)



df = df[df['n_sim'] > 2 ** 7]
print(df.dtypes)


# df = df[df['model_name'] == 'pysb.examples.kinase_cascade']
# print(df)
# quit()

def plot_per_model():
    plt.figure(figsize=(4, 12))
    df_local = df[df['device_name'] == 'piccolo']
    for n, (model, data) in enumerate(df_local.groupby('model_name')):
        df1 = data[data['simulator'] == 'gpu_ssa']
        df2 = data[data['simulator'] == 'cutauleaping']
        df3 = data[data['simulator'] == 'stochkit_eight_cpu_ssa']
        df4 = data[data['simulator'] == 'stochkit_eight_cpu_tau']
        ax = plt.subplot(3, 1, n + 1)
        plt.title(model)
        plt.plot(df1['n_sim'], df1['sim_time'], '*g', label='gpu_ssa')
        plt.plot(df2['n_sim'], df2['sim_time'], '^r', label='cutauleaping')
        plt.plot(df3['n_sim'], df3['sim_time'], '<b',
                 label='stochkit_eight_cpu_ssa')
        plt.plot(df4['n_sim'], df4['sim_time'], 'o', color='orange',
                 label='stochkit_eight_cpu_tau')
        # plt.xscale('log', base=2)
        # plt.yscale('log', base=10)
        ax.set_xscale('log', basex=2)
        ax.set_yscale('log', basey=10)
        # plt.xlabel('2**N')
        plt.ylabel("IO + simulation total time (s)")
        if n == 2:
            plt.xlabel("Number of simulations")
        plt.legend(loc=0)
    plt.tight_layout()
    plt.savefig('gpu_ssa_vs_cutauleaping.png')
    # plt.show()
    plt.close()


def plot_per_gpu():
    plt.figure(figsize=(16, 12))
    counter = 1
    df_storage = df.sort_values(['model_name','simulator', 'n_sim'], ascending=False)
    for m, (model, df_local) in enumerate(
            df_storage.groupby('model_name', sort=False)):
        y_max = df_local['sim_time'].max()
        y_min = df_local['sim_time'].min()
        y_min *= .1
        y_max *= 10.

        for n, (device_name, data) in enumerate(df_local.groupby('gpu_name')):
            print(m, n, counter, model, device_name)

            ax = plt.subplot(
                len(df['model_name'].unique()),
                len(df['device_name'].unique()),
                # len(df['model_name'].unique()),
                counter)

            if n == 0:
                plt.ylabel(model.replace('pysb.examples.', ''), fontsize=18)
            if m == 0 and counter in [1, 2, 3, 4]:
                plt.title(device_name, fontsize=18)

            colors = ['*-g', '^-r', '<-b', 'o-k']
            simulators = ['stochkit_eight_cpu_ssa', 'gpu_ssa', 'cutauleaping',
                          'stochkit_eight_cpu_tau' ]
            for label, c in zip(simulators, colors):
                df1 = data[data['simulator'] == label]
                x = df1['n_sim']
                y = df1['sim_time'].copy()
                if label == 'stochkit_eight_cpu_ssa' or label == 'stochkit_eight_cpu_tau':
                    y *= 8
                ax.plot(x, y, c)

            ax.set_xscale('log', basex=2)
            ax.set_yscale('log', basey=10)
            plt.ylim(y_min, y_max)

            if m != 3:
                plt.setp(ax.get_xticklabels(), visible=False)
            else:
                plt.setp(ax.get_xticklabels(), fontsize=18)
            if n != 0:
                plt.setp(ax.get_yticklabels(), visible=False)
            else:
                plt.setp(ax.get_yticklabels(), fontsize=18)
            counter += 1

    labels = ['cpu SSA', 'gpu SSA','gpu tauleaping', 'cpu tauleaping']
    leg = plt.legend(bbox_to_anchor=(1.75, 2.50), labels=labels, ncol=1,
                     fontsize=18)
    plt.tight_layout()
    plt.savefig('compare_gpu.png', bbox_extra_artists=(leg,),
                bbox_inches='tight')
    # plt.show()
    plt.close()

def print_speedups():
    local_data = df[df['gpu_name'] == 'GeForce GTX 1080']
    for n, (model_name, data) in enumerate(local_data.groupby('model_name')):
        cpu_time = data[data['simulator'] == 'stochkit_eight_cpu_ssa']['sim_time']
        gpu_time = data[data['simulator'] == 'gpu_ssa']['sim_time']
        increase = (cpu_time-gpu_time)/cpu_time * 100
        print(model_name)
        print(increase.max())


if __name__ == "__main__":
    # plot_per_gpu()
    print_speedups()
    # plot_per_model()
