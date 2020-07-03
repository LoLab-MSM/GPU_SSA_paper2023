import os

import pandas as pd


def gather_data():
    all_df = []
    cols = ['device_name', 'end_time', 'gpu_name', 'model_name', 'n_sim',
            'n_ts',
            'sim_time', 'simulator', 'total_time', 'n_cpu']
    for i in os.listdir('Timings'):
        if i in ('times.csv', 'gtx1070_tpb_times.csv', 'rtx2080_tpb_times.csv',
                 'stochkit'):
            continue
        try:
            d = pd.read_csv('Timings/' + i)
            # I do this for old simulations ran.
            # Need to rerun to place both times.
            # total_time includes  all pysb overhead
            # (simulator construction and results)
            if 'total_time' not in d.columns:
                d['total_time'] = d['sim_time']
            if 'n_cpu' not in d.columns:
                d['n_cpu'] = 1
            all_df.append(d[cols])
        except:
            print("Need to fix {}".format(i))
    return pd.concat(all_df)
