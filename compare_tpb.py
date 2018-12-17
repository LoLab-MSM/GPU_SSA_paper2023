import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

df = pd.read_csv('Timings/rtx2080_tpb_times.csv',
                 names=['n_sim', 'tpb', 'block', 'time'], delimiter=' ')

print(df)
df['time'] = np.log2(df['time'])
ax2 = sns.catplot(x='n_sim', y='time', hue='tpb', kind="point", data=df)
ax2.set(yscale="log")
plt.show()
