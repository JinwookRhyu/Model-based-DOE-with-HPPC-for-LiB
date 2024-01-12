import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import string
import pickle

# Input parameters
is_uncertainty = True # True if plotting uncertainty (e.g. 90% confidence region), False if plotting error (i.e. predicted - actual)
is_relative = True # True if plotting relative (%) values, False if plotting absolute values
is_transparent = False # Whether to plot transparent histogram



Standard = []
with (open("C:/Users/Jinwook/PyCharm_projects/DOE/dummy_N5_similar_0.0_10.0_0.5_1.0_0.0_10.0_0.5_1.0_0.7_1.0.pkl", "rb")) as openfile:
    while True:
        try:
            Standard.append(pickle.load(openfile))
        except EOFError:
            break

Optimal = []
with (open("C:/Users/Jinwook/PyCharm_projects/DOE/optimal_N5_similar_0.0_10.0_0.5_1.0_0.0_10.0_0.5_1.0_0.7_1.0.pkl", "rb")) as openfile:
    while True:
        try:
            Optimal.append(pickle.load(openfile))
        except EOFError:
            break

General = []
with (open("C:/Users/Jinwook/PyCharm_projects/DOE/general_N5_similar_0.0_10.0_0.5_1.0_0.0_10.0_0.5_1.0_0.7_1.0.pkl", "rb")) as openfile:
    while True:
        try:
            General.append(pickle.load(openfile))
        except EOFError:
            break

Random = []
with (open("C:/Users/Jinwook/PyCharm_projects/DOE/random_N5_0.0_10.0_0.5_1.0_0.0_10.0_0.5_1.0_0.7_1.0.pkl", "rb")) as openfile:
    while True:
        try:
            Random.append(pickle.load(openfile))
        except EOFError:
            break

####################################################################################################################
# Pairplot figure
True_params = np.zeros((len(Standard[0]), 5))

for k in range(len(Standard[0])):
    True_params[k,:] = Standard[0][k]['True_params']

True_params_df = pd.DataFrame(True_params, columns=['R_f_c', 'c_tilde_c', 'R_f_a', 'c_tilde_a', 'c_lyte'])
sns.set(font_scale=1.5)
sns.set_style("whitegrid")
sns.pairplot(True_params_df)
#plt.savefig("pairplot.png", dpi=300)
plt.show()

####################################################################################################################
# Effectiveness of model-based DOE

Uncertainty_optimal_lb = np.zeros((len(Optimal[0]), 5))
Uncertainty_optimal_ub = np.zeros((len(Optimal[0]), 5))
Uncertainty_optimal = np.zeros((len(Optimal[0]), 5))
Error_optimal = np.zeros((len(Optimal[0]), 5))
Uncertainty_general_lb = np.zeros((len(General[0]), 5))
Uncertainty_general_ub = np.zeros((len(General[0]), 5))
Uncertainty_general = np.zeros((len(General[0]), 5))
Error_general = np.zeros((len(General[0]), 5))
Uncertainty_standard_lb = np.zeros((len(Standard[0]), 5))
Uncertainty_standard_ub = np.zeros((len(Standard[0]), 5))
Uncertainty_standard = np.zeros((len(Standard[0]), 5))
Error_standard = np.zeros((len(Standard[0]), 5))
Uncertainty_random_lb = np.zeros((len(Random[0]), 5))
Uncertainty_random_ub = np.zeros((len(Random[0]), 5))
Uncertainty_random = np.zeros((len(Random[0]), 5))
Error_random = np.zeros((len(Random[0]), 5))

for k in range(len(Standard[0])):
    Uncertainty_optimal_lb[k, :] = Optimal[0][k]['lb_90'] - Optimal[0][k]['True_params']
    Uncertainty_optimal_ub[k, :] = Optimal[0][k]['ub_90'] - Optimal[0][k]['True_params']
    Uncertainty_optimal[k, :] = Optimal[0][k]['ub_90'] - Optimal[0][k]['lb_90']
    Error_optimal[k, :] = Optimal[0][k]['Estimated_params'] - Optimal[0][k]['True_params']
    Uncertainty_general_lb[k, :] = General[0][k]['lb_90'] - General[0][k]['True_params']
    Uncertainty_general_ub[k, :] = General[0][k]['ub_90'] - General[0][k]['True_params']
    Uncertainty_general[k, :] = General[0][k]['ub_90'] - General[0][k]['lb_90']
    Error_general[k, :] = General[0][k]['Estimated_params'] - General[0][k]['True_params']
    Uncertainty_standard_lb[k, :] = Standard[0][k]['lb_90'] - Standard[0][k]['True_params']
    Uncertainty_standard_ub[k, :] = Standard[0][k]['ub_90'] - Standard[0][k]['True_params']
    Uncertainty_standard[k, :] = Standard[0][k]['ub_90'] - Standard[0][k]['lb_90']
    Error_standard[k, :] = Standard[0][k]['Estimated_params'] - Standard[0][k]['True_params']
    Uncertainty_random_lb[k, :] = Random[0][k]['lb_90'] - Random[0][k]['True_params']
    Uncertainty_random_ub[k, :] = Random[0][k]['ub_90'] - Random[0][k]['True_params']
    Uncertainty_random[k, :] = Random[0][k]['ub_90'] - Random[0][k]['lb_90']
    Error_random[k, :] = Random[0][k]['Estimated_params'] - Random[0][k]['True_params']

Uncertainty_optimal_all = np.vstack((Uncertainty_optimal_lb, Uncertainty_optimal_ub))
Uncertainty_general_all = np.vstack((Uncertainty_general_lb, Uncertainty_general_ub))
Uncertainty_standard_all = np.vstack((Uncertainty_standard_lb, Uncertainty_standard_ub))
Uncertainty_random_all = np.vstack((Uncertainty_random_lb, Uncertainty_random_ub))
True_params_rep = np.vstack((True_params, True_params))

Rel_Uncertainty_optimal_all = np.divide(Uncertainty_optimal_all, True_params_rep) * 100
Rel_Uncertainty_general_all = np.divide(Uncertainty_general_all, True_params_rep) * 100
Rel_Uncertainty_standard_all = np.divide(Uncertainty_standard_all, True_params_rep) * 100
Rel_Uncertainty_random_all = np.divide(Uncertainty_random_all, True_params_rep) * 100
Rel_Error_optimal = np.divide(Error_optimal, True_params) * 100
Rel_Error_general = np.divide(Error_general, True_params) * 100
Rel_Error_standard = np.divide(Error_standard, True_params) * 100
Rel_Error_random = np.divide(Error_random, True_params) * 100

yaxes = ['R_f_c', 'c_tilde_c', 'R_f_a', 'c_tilde_a', 'c_lyte']
fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(10,6))
axs = axs.ravel()


if is_uncertainty:
    if is_relative:
        ranges1 = [-50, 50, -20, 20, -2, 2, -0.2, 0.2, -0.4, 0.4]
        for idx, ax in enumerate(axs):
            if idx < 5:
                ax.hist(Rel_Uncertainty_random_all[:, idx].clip(min=ranges1[2 * idx], max=ranges1[2 * idx + 1]),
                        bins=20, color="gray", alpha=0.5, range=(ranges1[2 * idx], ranges1[2 * idx + 1]))
                ax.axvline(x=Rel_Uncertainty_optimal_all[56, idx], color='blue', linewidth=2)
                ax.set_ylabel(yaxes[idx])
                ax.set_xlim(ranges1[2 * idx], ranges1[2 * idx + 1])
                ax.set_ylim(0, 100)
                ax.plot((0), (1), ls="", marker="^", ms=10, color="k",
                        transform=ax.get_xaxis_transform(), clip_on=False)
                ax.axvline(x=0, color='k', linewidth=2)
                ax.ticklabel_format(style="sci", scilimits=(-3, 3))
                ax.text(-0.29, 1.0, "(" + string.ascii_lowercase[idx] + ")", transform=ax.transAxes,
                        size=25, weight='bold')
            else:
                ax.hist([np.array([100]), np.array([100])], bins=20,
                        color=['gray', 'blue'],
                        label=['Random pulses (N=5)', 'Individual optimal (N=5)'],
                        range=(0, 1))
                ax.legend(loc='upper left')
                ax.axis('off')
        fig2_name = "modelbasedDOE_uncertainty_rel_1"

    else:
        ranges1 = [-5, 5, -0.1, 0.1, -0.2, 0.2, -0.001, 0.001, -0.01, 0.01]
        for idx, ax in enumerate(axs):
            if idx < 5:
                ax.hist(Uncertainty_random_all[:, idx].clip(min=ranges1[2 * idx], max=ranges1[2 * idx + 1]), bins=20,
                        color="gray", alpha=0.5, range=(ranges1[2 * idx], ranges1[2 * idx + 1]))
                ax.axvline(x=Uncertainty_optimal_all[56, idx], color='blue', linewidth=2)
                ax.set_ylabel(yaxes[idx])
                ax.set_xlim(ranges1[2 * idx], ranges1[2 * idx + 1])
                ax.set_ylim(0, 100)
                ax.plot((0), (1), ls="", marker="^", ms=10, color="k",
                        transform=ax.get_xaxis_transform(), clip_on=False)
                ax.axvline(x=0, color='k', linewidth=2)
                ax.ticklabel_format(style="sci", scilimits=(-3, 3))
                ax.text(-0.29, 1.0, "(" + string.ascii_lowercase[idx] + ")", transform=ax.transAxes,
                        size=25, weight='bold')
            else:
                ax.hist([np.array([100]), np.array([100])], bins=20,
                        color=['gray', 'blue'],
                        label=['Random pulses (N=5)', 'Individual optimal (N=5)'],
                        range=(0, 1))
                ax.legend(loc='upper left')
                ax.axis('off')
        fig2_name = "modelbasedDOE_uncertainty_1"
else:
    if is_relative:
        ranges1 = [0, 50, -10, 5, -0.2, 0.2, -0.05, 0.05, -0.05, 0.1]
        for idx, ax in enumerate(axs):
            if idx < 5:
                ax.hist(Rel_Error_random[:, idx].clip(min=ranges1[2 * idx], max=ranges1[2 * idx + 1]), bins=20,
                        color="gray", alpha=0.5, range=(ranges1[2 * idx], ranges1[2 * idx + 1]))
                ax.axvline(x=Rel_Error_optimal[56, idx], color='blue', linewidth=2)
                ax.set_ylabel(yaxes[idx])
                ax.set_xlim(ranges1[2 * idx], ranges1[2 * idx + 1])
                ax.set_ylim(0, 100)
                ax.plot((0), (1), ls="", marker="^", ms=10, color="k",
                        transform=ax.get_xaxis_transform(), clip_on=False)
                ax.axvline(x=0, color='k', linewidth=2)
                ax.ticklabel_format(style="sci", scilimits=(-3, 3))
                ax.text(-0.29, 1.0, "(" + string.ascii_lowercase[idx] + ")", transform=ax.transAxes,
                        size=25, weight='bold')
            else:
                ax.hist([np.array([100]), np.array([100])], bins=20,
                        color=['gray', 'blue'],
                        label=['Random pulses (N=5)', 'Individual optimal (N=5)'],
                        range=(0, 1))
                ax.legend(loc='upper left')
                ax.axis('off')
        fig2_name = "modelbasedDOE_error_rel_1"

    else:
        ranges1 = [-1, 1, -0.01, 0.01, -0.01, 0.01, -0.0001, 0.0001, -0.0001, 0.0001]
        for idx, ax in enumerate(axs):
            if idx < 5:
                ax.hist(Error_random[:, idx].clip(min=ranges1[2 * idx], max=ranges1[2 * idx + 1]), bins=20,
                        color="gray", alpha=0.5, range=(ranges1[2 * idx], ranges1[2 * idx + 1]))
                ax.axvline(x=Error_optimal[56, idx], color='blue', linewidth=2)
                ax.set_ylabel(yaxes[idx])
                ax.set_xlim(ranges1[2 * idx], ranges1[2 * idx + 1])
                ax.set_ylim(0, 100)
                ax.plot((0), (1), ls="", marker="^", ms=10, color="k",
                        transform=ax.get_xaxis_transform(), clip_on=False)
                ax.axvline(x=0, color='k', linewidth=2)
                ax.ticklabel_format(style="sci", scilimits=(-3, 3))
                ax.text(-0.29, 1.0, "(" + string.ascii_lowercase[idx] + ")", transform=ax.transAxes,
                        size=25, weight='bold')
            else:
                ax.hist([np.array([100]), np.array([100])], bins=20,
                        color=['gray', 'blue'],
                        label=['Random pulses (N=5)', 'Individual optimal (N=5)'],
                        range=(0, 1))
                ax.legend(loc='upper left')
                ax.axis('off')
        fig2_name = "modelbasedDOE_error_1"

plt.style.use('seaborn-v0_8-muted')
plt.tight_layout()
plt.subplots_adjust(wspace=0.4, hspace=0.4)
plt.savefig(fig2_name + ".png", dpi=300)
plt.show()

####################################################################################################################
# Comparison btw model-based DOE (optimal) / dummy / mean-field (general)
# Here, dummy model is using uniformly distributed HPPC protocol with 10 pulses. Others are using 5 pulses
# Briefly, the performance is optimal > dummy > general

yaxes = ['R_f_c', 'c_tilde_c', 'R_f_a', 'c_tilde_a', 'c_lyte']
fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(10,6))
axs = axs.ravel()

ranges1 = [-2, 2, -2, 2, -2, 2, -0.5, 0.5, -0.5, 0.5]
ranges2 = [-2, 2, -2, 2, -0.2, 0.2, -0.02, 0.02, -0.02, 0.02]

for idx, ax in enumerate(axs):
    if idx < 5:
        ax.hist([Rel_Uncertainty_standard_all[:, idx].clip(min=ranges1[2 * idx], max=ranges1[2 * idx + 1]),
                 Rel_Uncertainty_optimal_all[:, idx].clip(min=ranges1[2 * idx], max=ranges1[2 * idx + 1]),
                 Rel_Uncertainty_general_all[:, idx].clip(min=ranges1[2 * idx], max=ranges1[2 * idx + 1])],
                bins=20, color=['green', 'blue', 'red'], range=(ranges1[2 * idx], ranges1[2 * idx + 1]))
        ax.set_ylabel(yaxes[idx])
        ax.set_xlim(ranges1[2 * idx], ranges1[2 * idx + 1])
        ax.set_ylim(0, 100)
        ax.plot((0), (1), ls="", marker="^", ms=10, color="k",
                transform=ax.get_xaxis_transform(), clip_on=False)
        ax.axvline(x = 0, color = 'k', linewidth=2)
        ax.ticklabel_format(style="sci", scilimits=(-3, 3))
        ax.text(-0.29, 1.0, "(" + string.ascii_lowercase[idx] + ")", transform=ax.transAxes,
                size=25, weight='bold')
        ax2 = fig.add_axes([0.35+0.48*(idx%2), 0.78-0.3*(np.floor(idx/2)), 0.11, 0.12])
        ax2.hist([Rel_Error_standard[:, idx].clip(min=ranges2[2 * idx], max=ranges2[2 * idx + 1]),
                  Rel_Error_optimal[:, idx].clip(min=ranges2[2 * idx], max=ranges2[2 * idx + 1]),
                 Rel_Error_general[:, idx].clip(min=ranges2[2 * idx], max=ranges2[2 * idx + 1])], bins=8,
                color=['green', 'blue', 'red'], range=(ranges2[2 * idx], ranges2[2 * idx + 1]))
        ax2.set_xlim(ranges2[2 * idx], ranges2[2 * idx + 1])
        ax2.set_ylim(0, 100)
        ax2.plot((0), (1), ls="", marker="^", ms=10, color="k",
                transform=ax2.get_xaxis_transform(), clip_on=False)
        ax2.axvline(x=0, color='k', linewidth=2)
        ax2.ticklabel_format(style="sci", scilimits=(-3, 3))
        for tick in ax2.xaxis.get_ticklabels():
            tick.set_fontsize(8)
        for tick in ax2.yaxis.get_ticklabels():
            tick.set_fontsize(8)
        ax.minorticks_on()
        ax.grid(which='major', linestyle='-', linewidth='0.5')
        ax.grid(which='minor', linestyle=':', linewidth='0.5')
        ax2.minorticks_on()
        ax2.grid(which='major', linestyle='-', linewidth='0.5')
        ax2.grid(which='minor', linestyle=':', linewidth='0.5')

    else:
        ax.hist([np.array([100]), np.array([100]), np.array([100])], bins=20,
                color=['green', 'blue', 'red'],
                label=['Standard (N=10)', 'Individual optimal (N=5)', 'Generalized optimal (N=5)'],
                range=(0, 1))
        ax.legend(loc='upper left')
        ax.axis('off')
fig3_name = "fig2_balanced"

plt.style.use('seaborn-v0_8-muted')
plt.tight_layout()
plt.subplots_adjust(wspace=0.4, hspace=0.4)
plt.savefig(fig3_name + ".png", dpi=300)
plt.show()

'''
if is_uncertainty:
    if is_relative:
        ranges1 = [-100, 100, -40, 40, -4, 4, -1, 1, -1, 1]
        ranges2 = [-20, 20, -5, 5, -0.5, 0.5, -0.05, 0.05, -0.05, 0.05]
        if is_transparent:
            for idx, ax in enumerate(axs):
                if idx < 5:
                    ax.hist(Rel_Uncertainty_optimal_all[:, idx].clip(min=ranges1[2 * idx], max=ranges1[2 * idx + 1]),
                            bins=20, color="blue", alpha=0.5, range=(ranges1[2 * idx], ranges1[2 * idx + 1]))
                    ax.hist(Rel_Uncertainty_general_all[:, idx].clip(min=ranges1[2 * idx], max=ranges1[2 * idx + 1]),
                            bins=20, color="red", alpha=0.5, range=(ranges1[2 * idx], ranges1[2 * idx + 1]))
                    ax.hist(Rel_Uncertainty_standard_all[:, idx].clip(min=ranges1[2 * idx], max=ranges1[2 * idx + 1]),
                            bins=20, color="green", alpha=0.5, range=(ranges1[2 * idx], ranges1[2 * idx + 1]))
                    ax.set_ylabel(yaxes[idx])
                    ax.set_xlim(ranges1[2 * idx], ranges1[2 * idx + 1])
                    ax.set_ylim(0, 100)
                    ax.plot((0), (1), ls="", marker="^", ms=10, color="k",
                            transform=ax.get_xaxis_transform(), clip_on=False)
                    ax.axvline(x=0, color='k', linewidth=2)
                    ax.ticklabel_format(style="sci", scilimits=(-3, 3))
                    ax.text(-0.29, 1.0, "(" + string.ascii_lowercase[idx] + ")", transform=ax.transAxes,
                            size=25, weight='bold')
                else:
                    ax.hist([np.array([100]), np.array([100]), np.array([100])], bins=20,
                            color=['blue', 'green', 'red'],
                            label=['Individual optimal (N=5)', 'Standard (N=10)', 'Generalized optimal (N=5)'],
                            range=(0, 1))
                    ax.legend(loc='upper left')
                    ax.axis('off')
            fig3_name = "performance_uncertainty_rel_1"
        else:
            for idx, ax in enumerate(axs):
                if idx < 5:
                    ax.hist([Rel_Uncertainty_optimal_all[:, idx].clip(min=ranges1[2 * idx], max=ranges1[2 * idx + 1]),
                             Rel_Uncertainty_standard_all[:, idx].clip(min=ranges1[2 * idx], max=ranges1[2 * idx + 1]),
                             Rel_Uncertainty_general_all[:, idx].clip(min=ranges1[2 * idx], max=ranges1[2 * idx + 1])],
                            bins=20, color=['blue', 'green', 'red'], range=(ranges1[2 * idx], ranges1[2 * idx + 1]))
                    ax.set_ylabel(yaxes[idx])
                    ax.set_xlim(ranges1[2 * idx], ranges1[2 * idx + 1])
                    ax.set_ylim(0, 100)
                    ax.plot((0), (1), ls="", marker="^", ms=10, color="k",
                            transform=ax.get_xaxis_transform(), clip_on=False)
                    ax.axvline(x = 0, color = 'k', linewidth=2)
                    ax.ticklabel_format(style="sci", scilimits=(-3, 3))
                    ax.text(-0.29, 1.0, "(" + string.ascii_lowercase[idx] + ")", transform=ax.transAxes,
                            size=25, weight='bold')
                    ax2 = fig.add_axes([0.35+0.48*(idx%2), 0.78-0.3*(np.floor(idx/2)), 0.11, 0.12])
                    ax2.hist([Rel_Error_optimal[:, idx].clip(min=ranges2[2 * idx], max=ranges2[2 * idx + 1]),
                             Rel_Error_standard[:, idx].clip(min=ranges2[2 * idx], max=ranges2[2 * idx + 1]),
                             Rel_Error_general[:, idx].clip(min=ranges2[2 * idx], max=ranges2[2 * idx + 1])], bins=8,
                            color=['blue', 'green', 'red'], range=(ranges2[2 * idx], ranges2[2 * idx + 1]))
                    ax2.set_xlim(ranges2[2 * idx], ranges2[2 * idx + 1])
                    ax2.set_ylim(0, 100)
                    ax2.plot((0), (1), ls="", marker="^", ms=10, color="k",
                            transform=ax2.get_xaxis_transform(), clip_on=False)
                    ax2.axvline(x=0, color='k', linewidth=2)
                    ax2.ticklabel_format(style="sci", scilimits=(-3, 3))
                    for tick in ax2.xaxis.get_ticklabels():
                        tick.set_fontsize(8)
                    for tick in ax2.yaxis.get_ticklabels():
                        tick.set_fontsize(8)
                    ax.minorticks_on()
                    ax.grid(which='major', linestyle='-', linewidth='0.5')
                    ax.grid(which='minor', linestyle=':', linewidth='0.5')
                    ax2.minorticks_on()
                    ax2.grid(which='major', linestyle='-', linewidth='0.5')
                    ax2.grid(which='minor', linestyle=':', linewidth='0.5')

                else:
                    ax.hist([np.array([100]), np.array([100]), np.array([100])], bins=20,
                            color=['blue', 'green', 'red'],
                            label=['Individual optimal (N=5)', 'Standard (N=10)', 'Generalized optimal (N=5)'],
                            range=(0, 1))
                    ax.legend(loc='upper left')
                    ax.axis('off')
            fig3_name = "performance_uncertainty_rel_2"
    else:
        ranges1 = [-5, 5, -0.1, 0.1, -0.2, 0.2, -0.001, 0.001, -0.01, 0.01]
        if is_transparent:
            for idx, ax in enumerate(axs):
                if idx < 5:
                    ax.hist(Uncertainty_optimal_all[:, idx].clip(min=ranges1[2 * idx], max=ranges1[2 * idx + 1]), bins=20,
                            color="blue", alpha=0.5, range=(ranges1[2 * idx], ranges1[2 * idx + 1]))
                    ax.hist(Uncertainty_general_all[:, idx].clip(min=ranges1[2 * idx], max=ranges1[2 * idx + 1]), bins=20,
                            color="red", alpha=0.5, range=(ranges1[2 * idx], ranges1[2 * idx + 1]))
                    ax.hist(Uncertainty_standard_all[:, idx].clip(min=ranges1[2 * idx], max=ranges1[2 * idx + 1]), bins=20,
                            color="green", alpha=0.5, range=(ranges1[2 * idx], ranges1[2 * idx + 1]))
                    ax.set_ylabel(yaxes[idx])
                    ax.set_xlim(ranges1[2 * idx], ranges1[2 * idx + 1])
                    ax.set_ylim(0, 100)
                    ax.plot((0), (1), ls="", marker="^", ms=10, color="k",
                            transform=ax.get_xaxis_transform(), clip_on=False)
                    ax.axvline(x=0, color='k', linewidth=2)
                    ax.ticklabel_format(style="sci", scilimits=(-3, 3))
                    ax.text(-0.29, 1.0, "(" + string.ascii_lowercase[idx] + ")", transform=ax.transAxes,
                            size=25, weight='bold')
                else:
                    ax.hist([np.array([100]), np.array([100]), np.array([100])], bins=20,
                            color=['blue', 'green', 'red'],
                            label=['Individual optimal (N=5)', 'Standard (N=10)', 'Generalized optimal (N=5)'],
                            range=(0, 1))
                    ax.legend(loc='upper left')
                    ax.axis('off')
            fig3_name = "performance_uncertainty_1"
        else:
            for idx, ax in enumerate(axs):
                if idx < 5:
                    ax.hist([Uncertainty_optimal_all[:, idx].clip(min=ranges1[2 * idx], max=ranges1[2 * idx + 1]),
                             Uncertainty_standard_all[:, idx].clip(min=ranges1[2 * idx], max=ranges1[2 * idx + 1]),
                             Uncertainty_general_all[:, idx].clip(min=ranges1[2 * idx], max=ranges1[2 * idx + 1])],
                            bins=20, color=['blue', 'green', 'red'], range=(ranges1[2 * idx], ranges1[2 * idx + 1]))
                    ax.set_ylabel(yaxes[idx])
                    ax.set_xlim(ranges1[2 * idx], ranges1[2 * idx + 1])
                    ax.set_ylim(0, 100)
                    ax.plot((0), (1), ls="", marker="^", ms=10, color="k",
                            transform=ax.get_xaxis_transform(), clip_on=False)
                    ax.axvline(x=0, color='k', linewidth=2)
                    ax.ticklabel_format(style="sci", scilimits=(-3, 3))
                    ax.text(-0.29, 1.0, "(" + string.ascii_lowercase[idx] + ")", transform=ax.transAxes,
                            size=25, weight='bold')
                else:
                    ax.hist([np.array([100]), np.array([100]), np.array([100])], bins=20,
                            color=['blue', 'green', 'red'],
                            label=['Individual optimal (N=5)', 'Standard (N=10)', 'Generalized optimal (N=5)'],
                            range=(0, 1))
                    ax.legend(loc='upper left')
                    ax.axis('off')
            fig3_name = "performance_uncertainty_2"
else:
    if is_relative:
        ranges1 = [-20, 20, -4, 4, -0.4, 0.4, -0.04, 0.04, -0.04, 0.04]
        if is_transparent:
            for idx, ax in enumerate(axs):
                if idx < 5:
                    ax.hist(Rel_Error_optimal[:, idx].clip(min=ranges1[2 * idx], max=ranges1[2 * idx + 1]), bins=20,
                            color="blue", alpha=0.5, range=(ranges1[2 * idx], ranges1[2 * idx + 1]))
                    ax.hist(Rel_Error_general[:, idx].clip(min=ranges1[2 * idx], max=ranges1[2 * idx + 1]), bins=20,
                            color="red", alpha=0.5, range=(ranges1[2 * idx], ranges1[2 * idx + 1]))
                    ax.hist(Rel_Error_standard[:, idx].clip(min=ranges1[2 * idx], max=ranges1[2 * idx + 1]), bins=20,
                            color="green", alpha=0.5, range=(ranges1[2 * idx], ranges1[2 * idx + 1]))
                    ax.set_ylabel(yaxes[idx])
                    ax.set_xlim(ranges1[2 * idx], ranges1[2 * idx + 1])
                    ax.set_ylim(0, 100)
                    ax.plot((0), (1), ls="", marker="^", ms=10, color="k",
                            transform=ax.get_xaxis_transform(), clip_on=False)
                    ax.axvline(x=0, color='k', linewidth=2)
                    ax.ticklabel_format(style="sci", scilimits=(-3, 3))
                    ax.text(-0.29, 1.0, "(" + string.ascii_lowercase[idx] + ")", transform=ax.transAxes,
                            size=25, weight='bold')
                else:
                    ax.hist([np.array([100]), np.array([100]), np.array([100])], bins=20,
                            color=['blue', 'green', 'red'],
                            label=['Individual optimal (N=5)', 'Standard (N=10)', 'Generalized optimal (N=5)'],
                            range=(0, 1))
                    ax.legend(loc='upper left')
                    ax.axis('off')
            fig3_name = "performance_error_rel_1"
        else:
            for idx, ax in enumerate(axs):
                if idx < 5:
                    ax.hist([Rel_Error_optimal[:, idx].clip(min=ranges1[2 * idx], max=ranges1[2 * idx + 1]),
                             Rel_Error_standard[:, idx].clip(min=ranges1[2 * idx], max=ranges1[2 * idx + 1]),
                             Rel_Error_general[:, idx].clip(min=ranges1[2 * idx], max=ranges1[2 * idx + 1])], bins=20,
                            color=['blue', 'green', 'red'], range=(ranges1[2 * idx], ranges1[2 * idx + 1]))
                    ax.set_ylabel(yaxes[idx])
                    ax.set_xlim(ranges1[2 * idx], ranges1[2 * idx + 1])
                    ax.set_ylim(0, 100)
                    ax.plot((0), (1), ls="", marker="^", ms=10, color="k",
                            transform=ax.get_xaxis_transform(), clip_on=False)
                    ax.axvline(x=0, color='k', linewidth=2)
                    ax.ticklabel_format(style="sci", scilimits=(-3, 3))
                    ax.text(-0.29, 1.0, "(" + string.ascii_lowercase[idx] + ")", transform=ax.transAxes,
                            size=25, weight='bold')
                else:
                    ax.hist([np.array([100]), np.array([100]), np.array([100])], bins=20,
                            color=['blue', 'green', 'red'], label=['Individual optimal (N=5)', 'Standard (N=10)', 'Generalized optimal (N=5)'],
                            range=(0, 1))
                    ax.legend(loc='upper left')
                    ax.axis('off')
            fig3_name = "performance_error_rel_2"
    else:
        ranges1 = [-1, 1, -0.01, 0.01, -0.01, 0.01, -0.0001, 0.0001, -0.0001, 0.0001]
        if is_transparent:
            for idx, ax in enumerate(axs):
                if idx < 5:
                    ax.hist(Error_optimal[:, idx].clip(min=ranges1[2 * idx], max=ranges1[2 * idx + 1]), bins=20,
                            color="blue", alpha=0.5, range=(ranges1[2 * idx], ranges1[2 * idx + 1]))
                    ax.hist(Error_general[:, idx].clip(min=ranges1[2 * idx], max=ranges1[2 * idx + 1]), bins=20,
                            color="red", alpha=0.5, range=(ranges1[2 * idx], ranges1[2 * idx + 1]))
                    ax.hist(Error_standard[:, idx].clip(min=ranges1[2 * idx], max=ranges1[2 * idx + 1]), bins=20,
                            color="green", alpha=0.5, range=(ranges1[2 * idx], ranges1[2 * idx + 1]))
                    ax.set_ylabel(yaxes[idx])
                    ax.set_xlim(ranges1[2 * idx], ranges1[2 * idx + 1])
                    ax.set_ylim(0, 100)
                    ax.plot((0), (1), ls="", marker="^", ms=10, color="k",
                            transform=ax.get_xaxis_transform(), clip_on=False)
                    ax.axvline(x=0, color='k', linewidth=2)
                    ax.ticklabel_format(style="sci", scilimits=(-3, 3))
                    ax.text(-0.29, 1.0, "(" + string.ascii_lowercase[idx] + ")", transform=ax.transAxes,
                            size=25, weight='bold')
                else:
                    ax.hist([np.array([100]), np.array([100]), np.array([100])], bins=20,
                            color=['blue', 'green', 'red'],
                            label=['Individual optimal (N=5)', 'Standard (N=10)', 'Generalized optimal (N=5)'],
                            range=(0, 1))
                    ax.legend(loc='upper left')
                    ax.axis('off')
            fig3_name = "performance_error_1"
        else:
            for idx, ax in enumerate(axs):
                if idx < 5:
                    ax.hist([Error_optimal[:, idx].clip(min=ranges1[2 * idx], max=ranges1[2 * idx + 1]),
                             Error_standard[:, idx].clip(min=ranges1[2 * idx], max=ranges1[2 * idx + 1]),
                             Error_general[:, idx].clip(min=ranges1[2 * idx], max=ranges1[2 * idx + 1])], bins=20,
                            color=['blue', 'green', 'red'], range=(ranges1[2 * idx], ranges1[2 * idx + 1]))
                    ax.set_ylabel(yaxes[idx])
                    ax.set_xlim(ranges1[2 * idx], ranges1[2 * idx + 1])
                    ax.set_ylim(0, 100)
                    ax.plot((0), (1), ls="", marker="^", ms=10, color="k",
                            transform=ax.get_xaxis_transform(), clip_on=False)
                    ax.axvline(x=0, color='k', linewidth=2)
                    ax.ticklabel_format(style="sci", scilimits=(-3, 3))
                    ax.text(-0.29, 1.0, "(" + string.ascii_lowercase[idx] + ")", transform=ax.transAxes,
                            size=25, weight='bold')
                else:
                    ax.hist([np.array([100]), np.array([100]), np.array([100])], bins=20,
                            color=['blue', 'green', 'red'],
                            label=['Individual optimal (N=5)', 'Standard (N=10)', 'Generalized optimal (N=5)'],
                            range=(0, 1))
                    ax.legend(loc='upper left')
                    ax.axis('off')
            fig3_name = "performance_error_2"

plt.style.use('seaborn-v0_8-muted')
plt.tight_layout()
plt.subplots_adjust(wspace=0.4, hspace=0.4)
plt.savefig(fig3_name + ".png", dpi=300)
plt.show()
'''



d