import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
"""

Code to look how changing threads per block affects run times.

Results: Use 32 threads per block!


"""
df = pd.read_csv('Timings/gtx1070_tpb_times.csv',
                 names=['n_sim', 'tpb', 'block', 'time'], delimiter=' ')

print(df)
df['time'] = np.log2(df['time'])
ax2 = sns.catplot(x='n_sim', y='time', hue='tpb', kind="point", data=df)
ax2.set(yscale="log")
plt.show()
