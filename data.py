import os

import pandas as pd


def gather_data():
    all_df = []
    cols = ['device_name', 'end_time', 'gpu_name', 'model_name', 'n_sim',
            'n_ts', 'sim_time', 'simulator', 'total_time', 'n_cpu',
            'opencl_args', 'precision']
    for i in os.listdir('Timings'):
        if i in ('times.csv', 'times2.csv', 'gtx1070_tpb_times.csv',
                 'rtx2080_tpb_times.csv',
                 'stochkit', 'RAGE_TIMES'):
            continue
        converters = {
            'device_name': str.strip,
            'simulator': str.strip,
            'gpu_name': str.strip,
        }

        try:
            d = pd.read_csv(
                'Timings/' + i,
                converters=converters
            )
            for col in d.columns.values:
                d.rename(columns={col: col.strip()}, inplace=True)
            # print(d.head(5))
            # for col in cols:
            #     d[col] = d[col].strip()
            # I do this for old simulations ran.
            # Need to rerun to place both times.
            # total_time includes  all pysb overhead
            # (simulator construction and results)
            if 'total_time' not in d.columns:
                d['total_time'] = d['sim_time']
            if 'n_cpu' not in d.columns:
                d['n_cpu'] = 1
            if 'opencl_args' not in d.columns:
                d['opencl_args'] = ''
            if 'precision' not in d.columns:
                d['precision'] = 'fp64'
            all_df.append(d[cols])
        except:
            print("Need to fix {}".format(i))

    return pd.concat(all_df)


def load_rage():
    all_df = []
    cols = ['device_name', 'end_time', 'gpu_name', 'model_name', 'n_sim',
            'n_ts', 'sim_time', 'simulator', 'total_time', 'n_cpu',
            'opencl_args', 'precision']

    for i in os.listdir('Timings/RAGE_TIMES'):
        if i in ('times.csv', 'gtx1070_tpb_times.csv', 'rtx2080_tpb_times.csv',
                 'stochkit'):
            continue

        if 'fp' not in i:
            continue
        # d = pd.read_csv('Timings/RAGE_TIMES/' + i)
        # all_df.append(d[cols])
        try:
            d = pd.read_csv('Timings/RAGE_TIMES/' + i)
            for col in d.columns.values:
                d.rename(columns={col: col.strip()}, inplace=True)
            # print(d.head(5))
            # for col in cols:
            #     d[col] = d[col].strip()
            # I do this for old simulations ran.
            # Need to rerun to place both times.
            # total_time includes  all pysb overhead
            # (simulator construction and results)
            if 'total_time' not in d.columns:
                d['total_time'] = d['sim_time']
            if 'n_cpu' not in d.columns:
                d['n_cpu'] = 1
            if 'opencl_args' not in d.columns:
                d['opencl_args'] = ''
            if 'precision' not in d.columns:
                d['precision'] = 'fp64'
            all_df.append(d[cols])
        except:
            print("Need to fix {}".format(i))

    return pd.concat(all_df)


def load_data(rage=False):
    if not rage:
        df = gather_data()
    else:
        df = load_rage()
    # df = gather_precision()
    df['model_name'] = df['model_name'].str.split('pysb.examples.').str.get(-1)
    df = df.loc[~df.model_name.isin(['ras_camp_pka'])]
    df = df.loc[(df.n_sim < 2 ** 17) & (df.n_sim > 2 ** 7)]

    # rename gpus
    df.loc[df['gpu_name'] == 'gtx1080', 'gpu_name'] = 'GTX1080'
    df.loc[df['gpu_name'] == 'GeForce GTX 1060', 'gpu_name'] = 'GTX1060'
    df.loc[df['gpu_name'] == 'GeForce GTX 1080', 'gpu_name'] = 'GTX1080'
    df.loc[df['gpu_name'] == 'RTX2080 ', 'gpu_name'] = 'RTX2080'
    df.loc[df['gpu_name'] == 'GTX1080 ', 'gpu_name'] = 'GTX1080'
    df.loc[df['gpu_name'] == 'TeslaV100-SXM2-16GB', 'gpu_name'] = 'TeslaV100'
    df.loc[df['gpu_name'] == 'TeslaV100-SXM2-32GB', 'gpu_name'] = 'TeslaV100'
    df.loc[df['gpu_name'] == 'VOLTA_V100', 'gpu_name'] = 'Tesla-V100'
    df.loc[df['gpu_name'] == 'Tesla K20c', 'gpu_name'] = 'Tesla-K20c'

    # rename pcs
    df.loc[df['device_name'] == 'bad.mc.vanderbilt.edu', 'device_name'] = 'bad'
    df.loc[df['device_name'] == 'ip-172-31-22-73', 'device_name'] = 'aws'

    # rename simulators
    df.loc[df['simulator'] == 'gpu_ssa', 'simulator'] = 'cuda'
    df.loc[df['simulator'] == 'cl_amd_gpu', 'simulator'] = 'opencl'
    df.loc[df['simulator'] == 'cl_amd', 'simulator'] = 'opencl'
    df.loc[df['simulator'] == 'cl_nvidia', 'simulator'] = 'opencl'
    df.loc[df['simulator'] == 'cl', 'simulator'] = 'opencl'

    crit = (df.simulator == 'stochkit') & (df.n_cpu == 64)
    df.loc[crit, 'simulator'] = 'stochkit_64'

    df.loc[(df['gpu_name'] == 'i5_6500T') & (
            df['simulator'] == 'cl_intel_gpu'), 'gpu_name'] = 'HD530'

    df['sim_card'] = df['gpu_name'] + '_' + df['simulator']
    df.sim_time = df.sim_time.astype('float')
    df.loc[df.model_name == 'michment', 'model_name'] = 'Michaelis-Menten'
    df.loc[df.model_name == 'earm_1_0', 'model_name'] = 'EARM 1.0'
    df.loc[
        df.model_name == 'kinase_cascade', 'model_name'] = 'Kinase Cascade'
    df.loc[df.model_name == 'schlogl', 'model_name'] = 'Schlögl'
    df.loc[df.model_name == 'schloegl', 'model_name'] = 'Schlögl'
    # model_names = ['Michaelis-Menten', 'Schlögl', 'Kinase Cascade', 'EARM 1.0']
    return df


if __name__ == '__main__':
    # gather_data()
    load_data(rage=True)
