import numpy as np
import pandas as pd
import seaborn as sns
import string
import pickle
import os
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib import cm

# =========================
# Config
# =========================
# True if plotting uncertainty (e.g. 90% confidence region),
# False if plotting error (predicted - actual)
is_uncertainty = True
# True if plotting relative (%) values, False if absolute
is_relative = False
# Whether to plot transparent histogram
is_transparent = False
is_balanced = False
lw_med = 1
lw_bound = 0.1

show_params_low = np.array([0, 0.8, 0.1, 0.8, 0.8])
show_params_high = np.array([0.2, 1, 0.2, 1, 1])

xbar_histogram_uncertainty = 0.2
xbar_histogram_error = 0.2
suffix_save = "v2"

# =========================
# Declare datasets here
# Comment out any entry to hide that model everywhere.
# =========================
DATASETS = {
    "Standard": {
        "file": r"standard0_high_A_N10_0.0_0.2_0.8_1.0_0.0_0.2_0.8_1.0_0.8_1.0_unbalanced_090225.npz",
        "label": "Standard (N=10)",
        "color": "black",
    },
    "Optimal_1": {
        "file": r"optimal1_high_A_N10_0.0_0.2_0.8_1.0_0.0_0.2_0.8_1.0_0.8_1.0_unbalanced_090225.npz",
        "label": "Optimal/H/A (N=10)",
        "color": "red",
    },
    "Optimal_2": {
        "file": r"optimal2_high_A_N10_0.0_0.2_0.8_1.0_0.0_0.2_0.8_1.0_0.8_1.0_unbalanced_090225.npz",
        "label": "Updated/H/A (N=10)",
        "color": "lightcoral",
    },
    "Optimal_3": {
        "file": r"optimal1_low_A_N10_0.0_0.2_0.8_1.0_0.0_0.2_0.8_1.0_0.8_1.0_unbalanced_090225.npz",
        "label": "Optimal/L/A (N=10)",
        "color": "blue",
    },
    "Optimal_4": {
        "file": r"optimal2_low_A_N10_0.0_0.2_0.8_1.0_0.0_0.2_0.8_1.0_0.8_1.0_unbalanced_090225.npz",
        "label": "Updated/L/A (N=10)",
        "color": "lightblue",
    },
    "Optimal_5": {
        "file": r"optimal1_high_D_N10_0.0_0.2_0.8_1.0_0.0_0.2_0.8_1.0_0.8_1.0_unbalanced_090225.npz",
        "label": "Optimal/H/D (N=10)",
        "color": "lime",
    },
    "Optimal_6": {
        "file": r"optimal2_high_D_N10_0.0_0.2_0.8_1.0_0.0_0.2_0.8_1.0_0.8_1.0_unbalanced_090225.npz",
        "label": "Updated/H/D (N=10)",
        "color": "lightgreen",
    },
    "Optimal_7": {
        "file": r"optimal1_low_D_N10_0.0_0.2_0.8_1.0_0.0_0.2_0.8_1.0_0.8_1.0_unbalanced_090225.npz",
        "label": "Optimal/L/D (N=10)",
        "color": "orange",
    },
    "Optimal_8": {
        "file": r"optimal2_low_D_N10_0.0_0.2_0.8_1.0_0.0_0.2_0.8_1.0_0.8_1.0_unbalanced_090225.npz",
        "label": "Updated/L/D (N=10)",
        "color": "moccasin",
    },
}

# Preferred plotting order if present:
ORDER = ["Standard", "Optimal_1", "Optimal_2", "Optimal_3", "Optimal_4", "Optimal_5", "Optimal_6", "Optimal_7", "Optimal_8"]

# =========================
# Load available datasets
# =========================
loaded = {}
for name, meta in DATASETS.items():
    path = os.path.join(os.getcwd(), meta["file"])
    try:
        loaded[name] = np.load(path)
    except Exception as e:
        print(f"[INFO] Skipping {name}: {e}")

active = [k for k in ORDER if k in loaded]
if "Standard" not in active:
    raise RuntimeError("The 'Standard' dataset is required for True_params and baseline plots.")

# Labels/colors in active order
label_list = [DATASETS[k]["label"] for k in active]
color_list = [DATASETS[k]["color"] for k in active]

Standard = loaded["Standard"]
num_mcmc_samples = len(Standard["True_params"])

# =========================
# Pairplot figure (uses Standard truth)
# =========================
True_params = np.zeros((num_mcmc_samples, 5))
for k in range(num_mcmc_samples):
    True_params[k, :] = Standard["True_params"][k]

True_params_df = pd.DataFrame(
    True_params,
    columns=["R_f_c", "c_tilde_c", "R_f_a", "c_tilde_a", "c_lyte"],
)
sns.set(font_scale=1.5)
sns.set_style("whitegrid")
sns.pairplot(True_params_df)
plt.savefig("pairplot.png", dpi=300)
plt.close()

idx_show = (
    (True_params_df["R_f_c"] >= show_params_low[0]) & (True_params_df["R_f_c"] <= show_params_high[0]) &
    (True_params_df["c_tilde_c"] >= show_params_low[1]) & (True_params_df["c_tilde_c"] <= show_params_high[1]) &
    (True_params_df["R_f_a"] >= show_params_low[2]) & (True_params_df["R_f_a"] <= show_params_high[2]) &
    (True_params_df["c_tilde_a"] >= show_params_low[3]) & (True_params_df["c_tilde_a"] <= show_params_high[3]) &
    (True_params_df["c_lyte"] >= show_params_low[4]) & (True_params_df["c_lyte"] <= show_params_high[4])
).values

# =========================
# Build metrics per model
# =========================
# Containers keyed by model name -> (num_mcmc_samples, 5)
LB = {}
UB = {}
MED = {}
UNC = {}
UNC_LB = {}
UNC_UB = {}
ERR = {}
REL_UNC = {}
REL_ERR = {}

for name in active:
    arr = loaded[name]
    # core arrays
    lb = np.zeros((num_mcmc_samples, 5))
    ub = np.zeros((num_mcmc_samples, 5))
    med = np.zeros((num_mcmc_samples, 5))
    unc = np.zeros((num_mcmc_samples, 5))
    unc_lb = np.zeros((num_mcmc_samples, 5))
    unc_ub = np.zeros((num_mcmc_samples, 5))
    err = np.zeros((num_mcmc_samples, 5))

    for k in range(num_mcmc_samples):
        true_k = arr["True_params"][k]
        lb_k = arr["lb_90"][k]
        ub_k = arr["ub_90"][k]
        est_k = arr["Estimated_params"][k]

        lb[k, :] = lb_k
        ub[k, :] = ub_k
        med[k, :] = est_k

        unc_lb[k, :] = lb_k - true_k
        unc_ub[k, :] = ub_k - true_k
        unc[k, :] = ub_k - lb_k
        err[k, :] = est_k - true_k

    LB[name], UB[name], MED[name] = lb, ub, med
    UNC[name], UNC_LB[name], UNC_UB[name], ERR[name] = unc, unc_lb, unc_ub, err

# Relative metrics need "True_params"
True_params_rep = np.vstack((True_params, True_params))
for name in active:
    REL_UNC[name] = (UNC[name] / True_params) * 100
    REL_ERR[name] = (ERR[name] / True_params) * 100

# For “_all” stacks (lb/ub combined), build if needed later
REL_UNC_ALL = {}
for name in active:
    unc_all = np.vstack((UNC_LB[name], UNC_UB[name]))
    REL_UNC_ALL[name] = (unc_all / True_params_rep) * 100

# =========================
# Helpers for plotting lists in active order
# =========================
def collect(metric_dict, idx_mask, idx_param, clip_max=None, absolute=False):
    """Collect metric arrays for all active models, in order, optionally clip and abs."""
    series = []
    for name in active:
        arr = metric_dict[name][idx_mask, idx_param]
        if absolute:
            arr = np.abs(arr)
        if clip_max is not None:
            arr = arr.clip(min=0, max=clip_max)
        series.append(arr)
    return series

def line_components(idx_sorted, idx_param):
    """Return lb, ub, med arrays per model (for line/filled bands) in active order."""
    lbs = [LB[name][idx_sorted, idx_param] for name in active]
    ubs = [UB[name][idx_sorted, idx_param] for name in active]
    meds = [MED[name][idx_sorted, idx_param] for name in active]
    return lbs, ubs, meds

# =========================
# Histogram figure (error + uncertainty, absolute/relative)
# =========================
yaxes = [r'$R_{f,c}$', r'$\tilde{c}_c$', r'$R_{f,a}$', r'$\tilde{c}_a$', '$c_+$']
fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(16, 8))
axs = axs.ravel()

for idx, ax in enumerate(axs):
    if idx < 5:
        if is_relative:
            ranges1 = [100, 5, 100, 5, 5]
            ranges2 = [100, 20, 100, 20, 20]
            err_series = [np.abs(REL_ERR[name][idx_show, idx]).clip(min=0, max=ranges1[idx]) for name in active]
            unc_series = [REL_UNC[name][idx_show, idx].clip(min=0, max=ranges2[idx]) for name in active]
        else:
            ranges1 = xbar_histogram_error * np.array([1, 1, 1, 1, 1])
            ranges2 = xbar_histogram_uncertainty * np.array([1, 1, 1, 1, 1])
            err_series = [np.abs(ERR[name][idx_show, idx]).clip(min=0, max=ranges1[idx]) for name in active]
            unc_series = [UNC[name][idx_show, idx].clip(min=0, max=ranges2[idx]) for name in active]

        ax.hist(err_series, bins=10, color=color_list, range=(0, ranges1[idx]))
        ax.set_ylabel(yaxes[idx])
        ax.set_xlim(0, ranges1[idx])
        ax.set_ylim(0, np.sum(idx_show))

        # small inset with uncertainty
        ax2 = fig.add_axes([0.15 + 0.3315 * (idx % 3), 0.78 - 0.51 * (np.floor(idx / 3)), 0.15, 0.15])
        ax2.hist(unc_series, bins=5, color=color_list, range=(0, ranges2[idx]))
        ax2.set_xlim(0, ranges2[idx])
        ax2.set_ylim(0, np.sum(idx_show))

        for tick in ax.xaxis.get_ticklabels():
            tick.set_fontsize(20)
        for tick in ax.yaxis.get_ticklabels():
            tick.set_fontsize(20)
        for tick in ax2.xaxis.get_ticklabels():
            tick.set_fontsize(15)
        for tick in ax2.yaxis.get_ticklabels():
            tick.set_fontsize(15)

        ax.minorticks_on(); ax.grid(which='major', linestyle='-', linewidth='0.5'); ax.grid(which='minor', linestyle=':', linewidth='0.5')
        ax2.minorticks_on(); ax2.grid(which='major', linestyle='-', linewidth='0.5'); ax2.grid(which='minor', linestyle=':', linewidth='0.5')
    else:
        ax.hist([np.array([100])] * len(color_list), bins=20, color=color_list, label=label_list, range=(0, 1))
        ax.legend(loc='upper left', fontsize=11)
        ax.axis('off')

fig3_name = f"mcmc_unbalanced_histogram{'_rel' if is_relative else ''}_{show_params_low[2]}_{show_params_high[2]}"
plt.style.use('seaborn-v0_8-muted')
plt.tight_layout()
plt.subplots_adjust(wspace=0.4, hspace=0.4)
plt.savefig(fig3_name + ".png", dpi=300)
plt.close()

# =========================
# Line plot: predicted vs actual with bands
# =========================
yaxes = [r'Predicted $R_{f,c}$', r'Predicted $\tilde{c}_c$', r'Predicted $R_{f,a}$', r'Predicted $\tilde{c}_a$', 'Predicted $c_+$']
xaxes = [r'Actual $R_{f,c}$', r'Actual $\tilde{c}_c$', r'Actual $R_{f,a}$', r'Actual $\tilde{c}_a$', 'Actual $c_+$']
fig, axs = plt.subplots(nrows=1, ncols=5, figsize=(16, 3))
axs = axs.ravel()

ranges_map = np.column_stack((show_params_low, show_params_high)).ravel()

for idx, ax in enumerate(axs):
    if idx < 5:
        sort_idx = np.argsort(True_params[idx_show, idx])
        sort_idx = np.where(idx_show)[0][sort_idx]

        for j, name in enumerate(active):
            lb = LB[name][sort_idx, idx]
            ub = UB[name][sort_idx, idx]
            med = MED[name][sort_idx, idx]
            tp = True_params[sort_idx, idx]
            ax.plot(tp, lb, color=color_list[j], linewidth=lw_bound)
            ax.plot(tp, ub, color=color_list[j], linewidth=lw_bound)
            ax.plot(tp, med, color=color_list[j], linewidth=lw_med)
            ax.fill_between(tp, lb, ub, alpha=.1, color=color_list[j])

        ax.set_ylabel(yaxes[idx])
        ax.set_xlabel(xaxes[idx])
        ax.set_xlim(ranges_map[2 * idx], ranges_map[2 * idx + 1])
        ax.set_ylim(ranges_map[2 * idx], ranges_map[2 * idx + 1])
        ax.minorticks_on()
        ax.grid(which='major', linestyle='-', linewidth='0.5')
        ax.grid(which='minor', linestyle=':', linewidth='0.5')
    else:
        ax.hist([np.array([100])] * len(color_list), bins=20, color=color_list, label=label_list, range=(0, 1))
        ax.legend(loc='upper left')
        ax.axis('off')

fig3_name = f"mcmc_unbalanced_line_{show_params_low[2]}_{show_params_high[2]}"
plt.style.use('seaborn-v0_8-muted')
plt.tight_layout()
plt.subplots_adjust(wspace=0.4, hspace=0.4)
plt.savefig(fig3_name + ".png", dpi=300)
plt.close()

# =========================
# Grid lines: Uncertainty vs Actual (all pairwise)
# =========================
yaxes = [r'Uncertainty $R_{f,c}$', r'Uncertainty $\tilde{c}_c$', r'Uncertainty $R_{f,a}$', r'Uncertainty $\tilde{c}_a$', 'Uncertainty $c_+$']
xaxes = [r'Actual $R_{f,c}$', r'Actual $\tilde{c}_c$', r'Actual $R_{f,a}$', r'Actual $\tilde{c}_a$', 'Actual $c_+$']
fig, axs = plt.subplots(nrows=5, ncols=5, figsize=(16, 15))
axs = axs.ravel()

for idx, ax in enumerate(axs):
    idx_x = int(idx // 5)
    idx_y = int(idx - idx_x * 5)
    sort_idx = np.argsort(True_params[idx_show, idx_x])
    sort_idx = np.where(idx_show)[0][sort_idx]

    for j, name in enumerate(active):
        ax.plot(True_params[sort_idx, idx_x], UNC[name][sort_idx, idx_y], color=color_list[j], linewidth=lw_bound)

    ax.set_ylabel(yaxes[idx_y])
    ax.set_xlabel(xaxes[idx_x])
    ax.set_xlim(show_params_low[idx_x], show_params_high[idx_x])
    ax.set_ylim(0, 0.2)
    ax.minorticks_on()
    ax.grid(which='major', linestyle='-', linewidth='0.5')
    ax.grid(which='minor', linestyle=':', linewidth='0.5')

fig8_name = f"mcmc_unbalanced_line_uncertainty_{show_params_low[2]}_{show_params_high[2]}"
plt.style.use('seaborn-v0_8-muted')
plt.tight_layout()
plt.subplots_adjust(wspace=0.4, hspace=0.4)
plt.savefig(fig8_name + ".png", dpi=300)
plt.close()

# =========================
# Grid lines: |Error| vs Actual (all pairwise)
# =========================
yaxes = [r'Error $R_{f,c}$', r'Error $\tilde{c}_c$', r'Error $R_{f,a}$', r'Error $\tilde{c}_a$', 'Error $c_+$']
fig, axs = plt.subplots(nrows=5, ncols=5, figsize=(16, 15))
axs = axs.ravel()

for idx, ax in enumerate(axs):
    idx_x = int(idx // 5)
    idx_y = int(idx - idx_x * 5)
    sort_idx = np.argsort(True_params[idx_show, idx_x])
    sort_idx = np.where(idx_show)[0][sort_idx]

    for j, name in enumerate(active):
        ax.plot(True_params[sort_idx, idx_x], np.abs(ERR[name][sort_idx, idx_y]), color=color_list[j], linewidth=lw_bound)

    ax.set_ylabel(yaxes[idx_y])
    ax.set_xlabel(xaxes[idx_x])
    ax.set_xlim(show_params_low[idx_x], show_params_high[idx_x])
    ax.set_ylim(0, 0.2)
    ax.minorticks_on()
    ax.grid(which='major', linestyle='-', linewidth='0.5')
    ax.grid(which='minor', linestyle=':', linewidth='0.5')

fig8_name = f"mcmc_unbalanced_line_error_{show_params_low[2]}_{show_params_high[2]}"
plt.style.use('seaborn-v0_8-muted')
plt.tight_layout()
plt.subplots_adjust(wspace=0.4, hspace=0.4)
plt.savefig(fig8_name + ".png", dpi=300)
plt.close()

# =========================
# Relative CDF-like panels (Rel Error & Rel Uncertainty)
# =========================
yaxes = [r'$R_{f,c}$', r'$\tilde{c}_c$', r'$R_{f,a}$', r'$\tilde{c}_a$', '$c_+$']
fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(8, 8))
axs = axs.ravel()

ranges1 = [100, 5, 100, 1, 1]
ranges2 = [100, 2, 100, 1, 1]
for idx, ax in enumerate(axs):
    if idx < 5:
        x1 = np.linspace(0, ranges1[idx], 21)
        x2 = np.linspace(0, ranges2[idx], len(x1))
        # init counts
        y_err = {name: np.zeros_like(x1) for name in active}
        y_unc = {name: np.zeros_like(x2) for name in active}

        for k in range(len(x1)):
            for name in active:
                y_err[name][k] = np.sum(np.abs(REL_ERR[name][idx_show, idx]) < x1[k])
                y_unc[name][k] = np.sum(np.abs(REL_UNC[name][idx_show, idx]) < x2[k])

        for j, name in enumerate(active):
            ax.plot(x1, y_err[name] / np.sum(idx_show), label=label_list[j], color=color_list[j])

        ax.set_ylabel(yaxes[idx])
        ax.set_xlim(0, ranges1[idx])
        ax.set_ylim(0, 1)

        ax2 = fig.add_axes([0.3 + 0.47 * (idx % 2), 0.77 - 0.33 * (np.floor(idx / 2)), 0.15, 0.15])
        for j, name in enumerate(active):
            ax2.plot(x2, y_unc[name] / np.sum(idx_show), label=label_list[j], color=color_list[j])
        ax2.set_xlim(0, ranges2[idx])
        ax2.set_ylim(0, 1)

        for tick in ax2.xaxis.get_ticklabels():
            tick.set_fontsize(8)
        for tick in ax2.yaxis.get_ticklabels():
            tick.set_fontsize(8)
        ax.minorticks_on(); ax.grid(which='major', linestyle='-', linewidth='0.5'); ax.grid(which='minor', linestyle=':', linewidth='0.5')
        ax2.minorticks_on(); ax2.grid(which='major', linestyle='-', linewidth='0.5'); ax2.grid(which='minor', linestyle=':', linewidth='0.5')
    else:
        ax.hist([np.array([100])] * len(color_list), bins=20, color=color_list, label=label_list, range=(0, 1))
        ax.legend(loc='upper left')
        ax.axis('off')

fig3_name = f"mcmc_unbalanced_plot_rel_{show_params_low[2]}_{show_params_high[2]}"
plt.style.use('seaborn-v0_8-muted')
plt.tight_layout()
plt.subplots_adjust(wspace=0.4, hspace=0.4)
plt.savefig(fig3_name + ".png", dpi=300)
plt.close()

# =========================
# Absolute CDF-like panels (Error & Uncertainty)
# =========================
fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(8, 8))
axs = axs.ravel()

ranges1 = [0.1, 0.01, 0.2, 0.01, 0.01]
ranges2 = show_params_high - show_params_low
for idx, ax in enumerate(axs):
    if idx < 5:
        x1 = np.linspace(0, ranges1[idx], 21)
        x2 = np.linspace(0, ranges2[idx], len(x1))

        y_err = {name: np.zeros_like(x1) for name in active}
        y_unc = {name: np.zeros_like(x2) for name in active}

        for k in range(len(x1)):
            for name in active:
                y_err[name][k] = np.sum(np.abs(ERR[name][idx_show, idx]) < x1[k])
                y_unc[name][k] = np.sum(np.abs(UNC[name][idx_show, idx]) < x2[k])

        for j, name in enumerate(active):
            ax.plot(x1, y_err[name] / np.sum(idx_show), label=label_list[j], color=color_list[j])

        ax.set_ylabel(yaxes[idx])
        ax.set_xlim(0, ranges1[idx])
        ax.set_ylim(0, 1)

        ax2 = fig.add_axes([0.3 + 0.47 * (idx % 2), 0.76 - 0.33 * (np.floor(idx / 2)), 0.15, 0.12])
        for j, name in enumerate(active):
            ax2.plot(x2, y_unc[name] / np.sum(idx_show), label=label_list[j], color=color_list[j])
        ax2.set_xlim(0, ranges2[idx])
        ax2.set_ylim(0, 1)

        for tick in ax2.xaxis.get_ticklabels():
            tick.set_fontsize(8)
        for tick in ax2.yaxis.get_ticklabels():
            tick.set_fontsize(8)
        ax.minorticks_on(); ax.grid(which='major', linestyle='-', linewidth='0.5'); ax.grid(which='minor', linestyle=':', linewidth='0.5')
        ax2.minorticks_on(); ax2.grid(which='major', linestyle='-', linewidth='0.5'); ax2.grid(which='minor', linestyle=':', linewidth='0.5')
    else:
        ax.hist([np.array([100])] * len(color_list), bins=20, color=color_list, label=label_list, range=(0, 1))
        ax.legend(loc='upper left', fontsize=10)
        ax.axis('off')

fig3_name = f"mcmc_unbalanced_plot_{show_params_low[2]}_{show_params_high[2]}"
plt.style.use('seaborn-v0_8-muted')
plt.tight_layout()
plt.subplots_adjust(wspace=0.4, hspace=0.4)
plt.savefig(fig3_name + ".png", dpi=300)
plt.close()
