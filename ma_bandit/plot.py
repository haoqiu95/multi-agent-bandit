import matplotlib.pyplot as plt
import os
import numpy as np


plt.rcParams.update(plt.rcParamsDefault)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath,amssymb}'

# plt.ticklabel_format(style='sci', scilimits=(0,0))

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
os.chdir(DIR_PATH + "/../")


def load_data(filepath):
    steps, mean, ust, lst = [], [], [], []
    with open(filepath, 'r') as file:
        for line in file:
            if line.startswith("#"): continue
            t, m, s = map(float, line.strip().split())
            steps.append(t)
            mean.append(m)
            ust.append(m + s)
            lst.append(m - s)
    return steps, mean, ust, lst

# Color palette
colors = {
    "GSE": '#1f77b4',       # Blue
    "UCB": '#ff7f0e',       # Orange
}

# GSE
steps, mean, ust, lst = load_data('data/GSE-hete_COMPLETE_N16_K5_T10000_p0.2')
plt.plot(steps, mean, color=colors["GSE"], label='GSE')
plt.fill_between(steps, ust, lst, color=colors["GSE"], alpha=0.2)

# UCB
steps, mean, ust, lst = load_data('data/GSE-hete_COMPLETE_N16_K5_T10000_p0.2')
plt.plot(steps, mean, color=colors["UCB"], label='DrFed-UCB')
plt.fill_between(steps, ust, lst, color=colors["UCB"], alpha=0.2)

# Labels and legend
plt.xlabel(r'$T$', fontsize=20)
plt.ylabel(r'$Regret$', fontsize=20)
plt.legend(loc='upper left', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.4)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()
plt.show()
