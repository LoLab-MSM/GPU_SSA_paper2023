import csv
import os
import platform

import numpy as np
import psutil
import scipy
from scipy.interpolate import splrep

rename = {
    'AMD64 Family 23 Model 1 Stepping 1, AuthenticAMD': 'Ryzen_1600x',
    'AMD64 Family 23 Model 113 Stepping 0, AuthenticAMD': 'Ryzen_3900x',
    'x86_64': 'Ryzen_3900x'
}


def get_info():
    """ attempts to gather cpu name, speed, and os from pc

    Returns
    -------
    dict
    """

    n_cpus = psutil.cpu_count()
    freq = list(psutil.cpu_freq())
    max_freq = freq[2]
    p_name = platform.processor()
    if p_name in rename:
        p_name = rename[p_name]
    os_name = '{}_{}'.format(platform.system(), platform.release())
    results = {
        'n_core': n_cpus,
        'os_name': os_name,
        'cpu_name': p_name,
        'cpu_freq': max_freq
    }
    return results


def write(row_dict):
    """ Writes row of timing to file

    Parameters
    ----------
    row_dict: dict

    """
    f_name = r'Timings/times.csv'
    fields = []
    values = []
    for key in sorted(row_dict):
        fields.append(key)
        values.append(row_dict[key])
    if os.path.exists(f_name):
        open_type = 'a'
    else:
        open_type = 'w'
    with open(f_name, open_type, newline='\n') as f:
        writer = csv.writer(f)
        if open_type == 'w':
            writer.writerow(fields)
        writer.writerow(values)


def print_info():
    info = get_info()
    for i, j in info.items():
        print("{} : {}".format(i, j))


def get_tod(tspan, traj):
    tod = list()
    for momp_traj in traj.T:
        tod.append(calc_tod(tspan, momp_traj))
    return np.array(tod)


def calc_tod(tspan, traj):
    ysim_momp_norm = traj / np.nanmax(traj)
    st, sc, sk = scipy.interpolate.splrep(tspan, ysim_momp_norm)
    try:
        t10 = scipy.interpolate.sproot((st, sc - 0.10, sk))[0]
        t90 = scipy.interpolate.sproot((st, sc - 0.90, sk))[0]
        # time of death  = halfway point between 10 and 90%
        td = (t10 + t90) / 2

        # time of switch is time between 90 and 10 %
        ts = t90 - t10
    except IndexError:
        td = 0
        ts = 0
    # final fraction of aSMAC (last value)
    return [td, ts, traj[-1]]


if __name__ == "__main__":
    header = 'device_name,end_time,gpu_name,model_name,n_sim,n_ts,sim_time,simulator,total_time'
    old = 'device_name,end_time,gpu_name,model_name,n_sim,n_ts,sim_time,simulator,total_time'
    with open(r'Timings/times.csv', 'r') as f:
        g = [header]
        for i in f.read().splitlines():
            if 'device_name,end_time,gpu_name,model_name,n_sim,n_ts,sim_time,simulator,total_time' in i:
                continue
            elif i != '':
                g.append(i)
            elif i.endswith('\n'):
                g.append(i.rstrip('\n'))
            else:
                print(i)
    with open(r'Timings/times2.csv', 'w') as f:
        f.write('\n'.join(g))
    get_info()
