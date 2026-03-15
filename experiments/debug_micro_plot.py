"""Plot micro_experiment.csv: glnorm-D for fresh vs warm, 2 iterations."""

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('/Users/samsonkoelle/isometry-pursuit/experiments/micro_experiment.csv')
print(df.to_string())

fig, ax = plt.subplots(figsize=(6, 4))
for method, grp in df.groupby('method'):
    ax.plot(grp['cumulative_iters'], grp['glnorm_minus_D'], 'o-', ms=6, label=method)
ax.set_yscale('log')
ax.set_xlabel('cumulative iterations')
ax.set_ylabel('glnorm − D')
ax.set_title('Micro experiment: fresh vs warm, 2 iters')
ax.legend()
plt.tight_layout()
plt.savefig('/Users/samsonkoelle/isometry-pursuit/experiments/debug_micro_plot.png', dpi=150)
plt.close()
print("saved debug_micro_plot.png")
