import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import string
import pickle
import os
from matplotlib import cm

# Input parameters
is_uncertainty = True # True if plotting uncertainty (e.g. 90% confidence region), False if plotting error (i.e. predicted - actual)
is_relative = True # True if plotting relative (%) values, False if plotting absolute values
is_transparent = False # Whether to plot transparent histogram
is_balanced = False
lw_med = 0.3
lw_bound = 2


if is_balanced:
    Standard = []
    Individual = []
    General = []

    with (open(os.getcwd() + "/standard_N10_0.0_10.0_0.8_1.0_0.0_10.0_0.8_1.0_0.8_1.0_balanced.pkl", "rb")) as openfile:
        while True:
            try:
                Standard.append(pickle.load(openfile))
            except EOFError:
                break


    with (open(os.getcwd() + "/individual_optimal_N5_0.0_10.0_0.8_1.0_0.0_10.0_0.8_1.0_0.8_1.0_balanced.pkl", "rb")) as openfile:
        while True:
            try:
                Individual.append(pickle.load(openfile))
            except EOFError:
                break

    '''
    with (open(os.getcwd() + "/generalized_optimal_N5_0.0_10.0_0.8_1.0_0.0_10.0_0.8_1.0_0.8_1.0_balanced.pkl", "rb")) as openfile:
        while True:
            try:
                General.append(pickle.load(openfile))
            except EOFError:
                break
    '''
else:
    Standard = []
    Individual5 = []
    Individual6 = []
    Individual7 = []
    General10_1 = []
    General10_2 = []
    General10_3 = []

    with (open(os.getcwd() + "/standard_N10_0.0_10.0_0.8_1.0_0.0_10.0_0.8_1.0_0.8_1.0_unbalanced.pkl", "rb")) as openfile:
        while True:
            try:
                Standard.append(pickle.load(openfile))
            except EOFError:
                break

    with (open(os.getcwd() + "/individual_optimal_N5_0.0_10.0_0.8_1.0_0.0_10.0_0.8_1.0_0.8_1.0_unbalanced.pkl",
               "rb")) as openfile:
        while True:
            try:
                Individual5.append(pickle.load(openfile))
            except EOFError:
                break

    with (open(os.getcwd() + "/individual_optimal_N6_0.0_10.0_0.8_1.0_0.0_10.0_0.8_1.0_0.8_1.0_unbalanced.pkl",
               "rb")) as openfile:
        while True:
            try:
                Individual6.append(pickle.load(openfile))
            except EOFError:
                break

    with (open(os.getcwd() + "/individual_optimal_N7_0.0_10.0_0.8_1.0_0.0_10.0_0.8_1.0_0.8_1.0_unbalanced.pkl",
               "rb")) as openfile:
        while True:
            try:
                Individual7.append(pickle.load(openfile))
            except EOFError:
                break

    with (open(os.getcwd() + "/generalized_optimal_N10_0.0_10.0_0.8_1.0_0.0_10.0_0.8_1.0_0.8_1.0_unbalanced_1.pkl",
               "rb")) as openfile:
        while True:
            try:
                General10_1.append(pickle.load(openfile))
            except EOFError:
                break

    with (open(os.getcwd() + "/generalized_optimal_N10_0.0_10.0_0.8_1.0_0.0_10.0_0.8_1.0_0.8_1.0_unbalanced_2.pkl",
               "rb")) as openfile:
        while True:
            try:
                General10_2.append(pickle.load(openfile))
            except EOFError:
                break

    with (open(os.getcwd() + "/generalized_optimal_N10_0.0_10.0_0.8_1.0_0.0_10.0_0.8_1.0_0.8_1.0_unbalanced_3.pkl",
               "rb")) as openfile:
        while True:
            try:
                General10_3.append(pickle.load(openfile))
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
Uncertainty_standard_lb = np.zeros((len(Standard[0]), 5))
Uncertainty_standard_ub = np.zeros((len(Standard[0]), 5))
Uncertainty_standard = np.zeros((len(Standard[0]), 5))
Error_standard = np.zeros((len(Standard[0]), 5))

Standard_lb = np.zeros((len(Standard[0]), 5))
Standard_ub = np.zeros((len(Standard[0]), 5))
Standard_med = np.zeros((len(Standard[0]), 5))

if is_balanced:

    Uncertainty_individual_lb = np.zeros((len(Individual[0]), 5))
    Uncertainty_individual_ub = np.zeros((len(Individual[0]), 5))
    Uncertainty_individual = np.zeros((len(Individual[0]), 5))
    Error_individual = np.zeros((len(Individual[0]), 5))

    Individual_lb = np.zeros((len(Individual[0]), 5))
    Individual_ub = np.zeros((len(Individual[0]), 5))
    Individual_med = np.zeros((len(Individual[0]), 5))
    '''
    Uncertainty_general_lb = np.zeros((len(General[0]), 5))
    Uncertainty_general_ub = np.zeros((len(General[0]), 5))
    Uncertainty_general = np.zeros((len(General[0]), 5))
    Error_general = np.zeros((len(General[0]), 5))

    General_lb = np.zeros((len(General[0]), 5))
    General_ub = np.zeros((len(General[0]), 5))
    General_med = np.zeros((len(General[0]), 5))
    '''
    for k in range(len(Standard[0])):
        Uncertainty_standard_lb[k, :] = Standard[0][k]['lb_90'] - Standard[0][k]['True_params']
        Uncertainty_standard_ub[k, :] = Standard[0][k]['ub_90'] - Standard[0][k]['True_params']
        Uncertainty_standard[k, :] = Standard[0][k]['ub_90'] - Standard[0][k]['lb_90']
        Error_standard[k, :] = Standard[0][k]['Estimated_params'] - Standard[0][k]['True_params']
        Standard_lb[k, :] = Standard[0][k]['lb_90']
        Standard_ub[k, :] = Standard[0][k]['ub_90']
        Standard_med[k, :] = Standard[0][k]['Estimated_params']

        Uncertainty_individual_lb[k, :] = Individual[0][k]['lb_90'] - Individual[0][k]['True_params']
        Uncertainty_individual_ub[k, :] = Individual[0][k]['ub_90'] - Individual[0][k]['True_params']
        Uncertainty_individual[k, :] = Individual[0][k]['ub_90'] - Individual[0][k]['lb_90']
        Error_individual[k, :] = Individual[0][k]['Estimated_params'] - Individual[0][k]['True_params']
        Individual_lb[k, :] = Individual[0][k]['lb_90']
        Individual_ub[k, :] = Individual[0][k]['ub_90']
        Individual_med[k, :] = Individual[0][k]['Estimated_params']
        '''
        Uncertainty_general_lb[k, :] = General[0][k]['lb_90'] - General[0][k]['True_params']
        Uncertainty_general_ub[k, :] = General[0][k]['ub_90'] - General[0][k]['True_params']
        Uncertainty_general[k, :] = General[0][k]['ub_90'] - General[0][k]['lb_90']
        Error_general[k, :] = General[0][k]['Estimated_params'] - General[0][k]['True_params']
        General_lb[k, :] = General[0][k]['lb_90']
        General_ub[k, :] = General[0][k]['ub_90']
        General_med[k, :] = General[0][k]['Estimated_params']
        '''

    Uncertainty_standard_all = np.vstack((Uncertainty_standard_lb, Uncertainty_standard_ub))
    Uncertainty_individual_all = np.vstack((Uncertainty_individual_lb, Uncertainty_individual_ub))
    #Uncertainty_general_all = np.vstack((Uncertainty_general_lb, Uncertainty_general_ub))
    True_params_rep = np.vstack((True_params, True_params))

    Rel_Uncertainty_standard_all = np.divide(Uncertainty_standard_all, True_params_rep) * 100
    Rel_Uncertainty_individual_all = np.divide(Uncertainty_individual_all, True_params_rep) * 100
    #Rel_Uncertainty_general_all = np.divide(Uncertainty_general_all, True_params_rep) * 100

    Rel_Error_standard = np.divide(Error_standard, True_params) * 100
    Rel_Error_individual = np.divide(Error_individual, True_params) * 100
    #Rel_Error_general = np.divide(Error_general, True_params) * 100


else:
    Uncertainty_individual5_lb = np.zeros((len(Individual5[0]), 5))
    Uncertainty_individual5_ub = np.zeros((len(Individual5[0]), 5))
    Uncertainty_individual5 = np.zeros((len(Individual5[0]), 5))
    Error_individual5 = np.zeros((len(Individual5[0]), 5))
    Uncertainty_individual6_lb = np.zeros((len(Individual6[0]), 5))
    Uncertainty_individual6_ub = np.zeros((len(Individual6[0]), 5))
    Uncertainty_individual6 = np.zeros((len(Individual6[0]), 5))
    Error_individual6 = np.zeros((len(Individual6[0]), 5))
    Uncertainty_individual7_lb = np.zeros((len(Individual7[0]), 5))
    Uncertainty_individual7_ub = np.zeros((len(Individual7[0]), 5))
    Uncertainty_individual7 = np.zeros((len(Individual7[0]), 5))
    Error_individual7 = np.zeros((len(Individual7[0]), 5))

    Individual5_lb = np.zeros((len(Individual5[0]), 5))
    Individual5_ub = np.zeros((len(Individual5[0]), 5))
    Individual5_med = np.zeros((len(Individual5[0]), 5))
    Individual6_lb = np.zeros((len(Individual6[0]), 5))
    Individual6_ub = np.zeros((len(Individual6[0]), 5))
    Individual6_med = np.zeros((len(Individual6[0]), 5))
    Individual7_lb = np.zeros((len(Individual7[0]), 5))
    Individual7_ub = np.zeros((len(Individual7[0]), 5))
    Individual7_med = np.zeros((len(Individual7[0]), 5))


    Uncertainty_general10_1_lb = np.zeros((len(General10_1[0]), 5))
    Uncertainty_general10_1_ub = np.zeros((len(General10_1[0]), 5))
    Uncertainty_general10_1 = np.zeros((len(General10_1[0]), 5))
    Error_general10_1 = np.zeros((len(General10_1[0]), 5))
    Uncertainty_general10_2_lb = np.zeros((len(General10_2[0]), 5))
    Uncertainty_general10_2_ub = np.zeros((len(General10_2[0]), 5))
    Uncertainty_general10_2 = np.zeros((len(General10_2[0]), 5))
    Error_general10_2 = np.zeros((len(General10_2[0]), 5))
    Uncertainty_general10_3_lb = np.zeros((len(General10_3[0]), 5))
    Uncertainty_general10_3_ub = np.zeros((len(General10_3[0]), 5))
    Uncertainty_general10_3 = np.zeros((len(General10_3[0]), 5))
    Error_general10_3 = np.zeros((len(General10_3[0]), 5))

    General10_1_lb = np.zeros((len(General10_1[0]), 5))
    General10_1_ub = np.zeros((len(General10_1[0]), 5))
    General10_1_med = np.zeros((len(General10_1[0]), 5))
    General10_2_lb = np.zeros((len(General10_2[0]), 5))
    General10_2_ub = np.zeros((len(General10_2[0]), 5))
    General10_2_med = np.zeros((len(General10_2[0]), 5))
    General10_3_lb = np.zeros((len(General10_3[0]), 5))
    General10_3_ub = np.zeros((len(General10_3[0]), 5))
    General10_3_med = np.zeros((len(General10_3[0]), 5))

    for k in range(len(Standard[0])):
        Uncertainty_standard_lb[k, :] = Standard[0][k]['lb_90'] - Standard[0][k]['True_params']
        Uncertainty_standard_ub[k, :] = Standard[0][k]['ub_90'] - Standard[0][k]['True_params']
        Uncertainty_standard[k, :] = Standard[0][k]['ub_90'] - Standard[0][k]['lb_90']
        Error_standard[k, :] = Standard[0][k]['Estimated_params'] - Standard[0][k]['True_params']
        Standard_lb[k, :] = Standard[0][k]['lb_90']
        Standard_ub[k, :] = Standard[0][k]['ub_90']
        Standard_med[k, :] = Standard[0][k]['Estimated_params']

        Uncertainty_individual5_lb[k, :] = Individual5[0][k]['lb_90'] - Individual5[0][k]['True_params']
        Uncertainty_individual5_ub[k, :] = Individual5[0][k]['ub_90'] - Individual5[0][k]['True_params']
        Uncertainty_individual5[k, :] = Individual5[0][k]['ub_90'] - Individual5[0][k]['lb_90']
        Error_individual5[k, :] = Individual5[0][k]['Estimated_params'] - Individual5[0][k]['True_params']
        Individual5_lb[k, :] = Individual5[0][k]['lb_90']
        Individual5_ub[k, :] = Individual5[0][k]['ub_90']
        Individual5_med[k, :] = Individual5[0][k]['Estimated_params']
        Uncertainty_individual6_lb[k, :] = Individual6[0][k]['lb_90'] - Individual6[0][k]['True_params']
        Uncertainty_individual6_ub[k, :] = Individual6[0][k]['ub_90'] - Individual6[0][k]['True_params']
        Uncertainty_individual6[k, :] = Individual6[0][k]['ub_90'] - Individual6[0][k]['lb_90']
        Error_individual6[k, :] = Individual6[0][k]['Estimated_params'] - Individual6[0][k]['True_params']
        Individual6_lb[k, :] = Individual6[0][k]['lb_90']
        Individual6_ub[k, :] = Individual6[0][k]['ub_90']
        Individual6_med[k, :] = Individual6[0][k]['Estimated_params']
        Uncertainty_individual7_lb[k, :] = Individual7[0][k]['lb_90'] - Individual7[0][k]['True_params']
        Uncertainty_individual7_ub[k, :] = Individual7[0][k]['ub_90'] - Individual7[0][k]['True_params']
        Uncertainty_individual7[k, :] = Individual7[0][k]['ub_90'] - Individual7[0][k]['lb_90']
        Error_individual7[k, :] = Individual7[0][k]['Estimated_params'] - Individual7[0][k]['True_params']
        Individual7_lb[k, :] = Individual7[0][k]['lb_90']
        Individual7_ub[k, :] = Individual7[0][k]['ub_90']
        Individual7_med[k, :] = Individual7[0][k]['Estimated_params']

        Uncertainty_general10_1_lb[k, :] = General10_1[0][k]['lb_90'] - General10_1[0][k]['True_params']
        Uncertainty_general10_1_ub[k, :] = General10_1[0][k]['ub_90'] - General10_1[0][k]['True_params']
        Uncertainty_general10_1[k, :] = General10_1[0][k]['ub_90'] - General10_1[0][k]['lb_90']
        Error_general10_1[k, :] = General10_1[0][k]['Estimated_params'] - General10_1[0][k]['True_params']
        General10_1_lb[k, :] = General10_1[0][k]['lb_90']
        General10_1_ub[k, :] = General10_1[0][k]['ub_90']
        General10_1_med[k, :] = General10_1[0][k]['Estimated_params']
        Uncertainty_general10_2_lb[k, :] = General10_2[0][k]['lb_90'] - General10_2[0][k]['True_params']
        Uncertainty_general10_2_ub[k, :] = General10_2[0][k]['ub_90'] - General10_2[0][k]['True_params']
        Uncertainty_general10_2[k, :] = General10_2[0][k]['ub_90'] - General10_2[0][k]['lb_90']
        Error_general10_2[k, :] = General10_2[0][k]['Estimated_params'] - General10_2[0][k]['True_params']
        General10_2_lb[k, :] = General10_2[0][k]['lb_90']
        General10_2_ub[k, :] = General10_2[0][k]['ub_90']
        General10_2_med[k, :] = General10_2[0][k]['Estimated_params']
        Uncertainty_general10_3_lb[k, :] = General10_3[0][k]['lb_90'] - General10_3[0][k]['True_params']
        Uncertainty_general10_3_ub[k, :] = General10_3[0][k]['ub_90'] - General10_3[0][k]['True_params']
        Uncertainty_general10_3[k, :] = General10_3[0][k]['ub_90'] - General10_3[0][k]['lb_90']
        Error_general10_3[k, :] = General10_3[0][k]['Estimated_params'] - General10_3[0][k]['True_params']
        General10_3_lb[k, :] = General10_3[0][k]['lb_90']
        General10_3_ub[k, :] = General10_3[0][k]['ub_90']
        General10_3_med[k, :] = General10_3[0][k]['Estimated_params']

    Uncertainty_standard_all = np.vstack((Uncertainty_standard_lb, Uncertainty_standard_ub))
    Uncertainty_individual5_all = np.vstack((Uncertainty_individual5_lb, Uncertainty_individual5_ub))
    Uncertainty_individual6_all = np.vstack((Uncertainty_individual6_lb, Uncertainty_individual6_ub))
    Uncertainty_individual7_all = np.vstack((Uncertainty_individual7_lb, Uncertainty_individual7_ub))
    Uncertainty_general10_1_all = np.vstack((Uncertainty_general10_1_lb, Uncertainty_general10_1_ub))
    Uncertainty_general10_2_all = np.vstack((Uncertainty_general10_2_lb, Uncertainty_general10_2_ub))
    Uncertainty_general10_3_all = np.vstack((Uncertainty_general10_3_lb, Uncertainty_general10_3_ub))
    True_params_rep = np.vstack((True_params, True_params))

    Rel_Uncertainty_standard_all = np.divide(Uncertainty_standard_all, True_params_rep) * 100
    Rel_Uncertainty_individual5_all = np.divide(Uncertainty_individual5_all, True_params_rep) * 100
    Rel_Uncertainty_individual6_all = np.divide(Uncertainty_individual6_all, True_params_rep) * 100
    Rel_Uncertainty_individual7_all = np.divide(Uncertainty_individual7_all, True_params_rep) * 100
    Rel_Uncertainty_general10_1_all = np.divide(Uncertainty_general10_1_all, True_params_rep) * 100
    Rel_Uncertainty_general10_2_all = np.divide(Uncertainty_general10_2_all, True_params_rep) * 100
    Rel_Uncertainty_general10_3_all = np.divide(Uncertainty_general10_3_all, True_params_rep) * 100

    Rel_Error_standard = np.divide(Error_standard, True_params) * 100
    Rel_Error_individual5 = np.divide(Error_individual5, True_params) * 100
    Rel_Error_individual6 = np.divide(Error_individual6, True_params) * 100
    Rel_Error_individual7 = np.divide(Error_individual7, True_params) * 100
    Rel_Error_general10_1 = np.divide(Error_general10_1, True_params) * 100
    Rel_Error_general10_2 = np.divide(Error_general10_2, True_params) * 100
    Rel_Error_general10_3 = np.divide(Error_general10_3, True_params) * 100

####################################################################################################################
# Comparison btw model-based DOE (optimal) / dummy / mean-field (general)
# Here, dummy model is using uniformly distributed HPPC protocol with 10 pulses. Others are using 5 pulses
# Briefly, the performance is optimal > dummy > general
cmap_blues = cm.Blues(range(256))
cmap_reds = cm.Reds(range(256))
cmap_reds_1 = cmap_reds[40, :]
cmap_reds_2 = cmap_reds[120, :]
cmap_reds_3 = cmap_reds[200, :]
cmap_blues_1 = cmap_blues[40, :]
cmap_blues_2 = cmap_blues[120, :]
cmap_blues_3 = cmap_blues[200, :]

yaxes = ['R_f_c', 'c_tilde_c', 'R_f_a', 'c_tilde_a', 'c_lyte']
fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(10,6))
axs = axs.ravel()


if is_balanced:
    ranges1 = [-2, 2, -2, 2, -2, 2, -2, 2, -2, 2]
    ranges2 = [-2, 2, -2, 2, -2, 2, -2, 2, -2, 2]
    for idx, ax in enumerate(axs):
        if idx < 5:
            ax.hist([Rel_Uncertainty_standard_all[:, idx].clip(min=ranges1[2 * idx], max=ranges1[2 * idx + 1]),
                     Rel_Uncertainty_individual_all[:, idx].clip(min=ranges1[2 * idx], max=ranges1[2 * idx + 1])],
                     #Rel_Uncertainty_general_all[:, idx].clip(min=ranges1[2 * idx], max=ranges1[2 * idx + 1])],
                    bins=20, color=['green', 'red'], range=(ranges1[2 * idx], ranges1[2 * idx + 1]))
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
                     Rel_Error_individual[:, idx].clip(min=ranges2[2 * idx], max=ranges2[2 * idx + 1])],
                     #Rel_Error_general[:, idx].clip(min=ranges2[2 * idx], max=ranges2[2 * idx + 1])], bins=8,
                    color=['green', 'red'], range=(ranges2[2 * idx], ranges2[2 * idx + 1]))
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
                    color=['green', 'red', 'blue'],
                    label=['Standard (N=10)', 'Individual optimal (N=5)', 'Generalized optimal (N=5)'],
                    range=(0, 1))
            ax.legend(loc='upper left')
            ax.axis('off')
    fig3_name = "fig1_balanced"
else:
    ranges1 = [-100, 100, -20, 20, -2, 2, -0.5, 0.5, -0.5, 0.5]
    ranges2 = [-10, 10, -2, 2, -0.2, 0.2, -0.02, 0.02, -0.02, 0.02]
    for idx, ax in enumerate(axs):
        if idx < 5:
            ax.hist([Rel_Uncertainty_standard_all[:, idx].clip(min=ranges1[2 * idx], max=ranges1[2 * idx + 1]),
                     #Rel_Uncertainty_individual5_all[:, idx].clip(min=ranges1[2 * idx], max=ranges1[2 * idx + 1]),
                     #Rel_Uncertainty_individual6_all[:, idx].clip(min=ranges1[2 * idx], max=ranges1[2 * idx + 1]),
                     #Rel_Uncertainty_individual7_all[:, idx].clip(min=ranges1[2 * idx], max=ranges1[2 * idx + 1]),
                     Rel_Uncertainty_general10_1_all[:, idx].clip(min=ranges1[2 * idx], max=ranges1[2 * idx + 1]),
                     Rel_Uncertainty_general10_2_all[:, idx].clip(min=ranges1[2 * idx], max=ranges1[2 * idx + 1]),
                     Rel_Uncertainty_general10_3_all[:, idx].clip(min=ranges1[2 * idx], max=ranges1[2 * idx + 1])],
                    bins=20, color=['green', cmap_blues_1, cmap_blues_2, cmap_blues_3], range=(ranges1[2 * idx], ranges1[2 * idx + 1]))
            ax.set_ylabel(yaxes[idx])
            ax.set_xlim(ranges1[2 * idx], ranges1[2 * idx + 1])
            ax.set_ylim(0, 100)
            ax.plot((0), (1), ls="", marker="^", ms=10, color="k",
                    transform=ax.get_xaxis_transform(), clip_on=False)
            ax.axvline(x=0, color='k', linewidth=2)
            ax.ticklabel_format(style="sci", scilimits=(-3, 3))
            ax.text(-0.29, 1.0, "(" + string.ascii_lowercase[idx] + ")", transform=ax.transAxes,
                    size=25, weight='bold')
            ax2 = fig.add_axes([0.35 + 0.48 * (idx % 2), 0.78 - 0.3 * (np.floor(idx / 2)), 0.11, 0.12])
            ax2.hist([Rel_Error_standard[:, idx].clip(min=ranges2[2 * idx], max=ranges2[2 * idx + 1]),
                      #Rel_Error_individual5[:, idx].clip(min=ranges2[2 * idx], max=ranges2[2 * idx + 1]),
                      #Rel_Error_individual6[:, idx].clip(min=ranges2[2 * idx], max=ranges2[2 * idx + 1]),
                      #Rel_Error_individual7[:, idx].clip(min=ranges2[2 * idx], max=ranges2[2 * idx + 1]),
                      Rel_Error_general10_1[:, idx].clip(min=ranges2[2 * idx], max=ranges2[2 * idx + 1]),
                      Rel_Error_general10_2[:, idx].clip(min=ranges2[2 * idx], max=ranges2[2 * idx + 1]),
                      Rel_Error_general10_3[:, idx].clip(min=ranges2[2 * idx], max=ranges2[2 * idx + 1])], bins=8,
                     color=['green', cmap_blues_1, cmap_blues_2, cmap_blues_3], range=(ranges2[2 * idx], ranges2[2 * idx + 1]))
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
            ax.hist([np.array([100]), np.array([100]), np.array([100]), np.array([100]), np.array([100]), np.array([100]), np.array([100])], bins=20,
                    color=['green', cmap_reds_1, cmap_reds_2, cmap_reds_3, cmap_blues_1, cmap_blues_2, cmap_blues_3],
                    label=['Standard (N=10)', 'Individual (N=5)', 'Individual (N=6)', 'Individual (N=7)', 'General_1 (N=10)', 'General_2 (N=10)', 'General_3 (N=10)'],
                    range=(0, 1))
            ax.legend(loc='upper left')
            ax.axis('off')

    fig3_name = "fig1_unbalanced"

plt.style.use('seaborn-v0_8-muted')
plt.tight_layout()
plt.subplots_adjust(wspace=0.4, hspace=0.4)
plt.savefig(fig3_name + ".png", dpi=300)
plt.show()

###############################################################################

yaxes = ['R_f_c', 'c_tilde_c', 'R_f_a', 'c_tilde_a', 'c_lyte']
fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(14,6))
axs = axs.ravel()

ranges1 = [0, 10, 0.8, 1.0, 0, 10, 0.8, 1.0, 0.8, 1.0]

if is_balanced:
    for idx, ax in enumerate(axs):
        if idx < 5:
            sort_idx = np.argsort(True_params[:, idx])
            ax.plot(True_params[sort_idx, idx], Standard_lb[sort_idx, idx], color='green', linewidth=lw_bound)
            ax.plot(True_params[sort_idx, idx], Standard_ub[sort_idx, idx], color='green', linewidth=lw_bound)
            ax.plot(True_params[sort_idx, idx], Standard_med[sort_idx, idx], color='green', linewidth=lw_med)
            ax.fill_between(True_params[sort_idx, idx], Standard_lb[sort_idx, idx], Standard_ub[sort_idx, idx], alpha=.1, color='green')
            ax.plot(True_params[sort_idx, idx], Individual_lb[sort_idx, idx], color='red', linewidth=lw_bound)
            ax.plot(True_params[sort_idx, idx], Individual_ub[sort_idx, idx], color='red', linewidth=lw_bound)
            ax.plot(True_params[sort_idx, idx], Individual_med[sort_idx, idx], color='red', linewidth=lw_med)
            ax.fill_between(True_params[sort_idx, idx], Individual_lb[sort_idx, idx], Individual_ub[sort_idx, idx], alpha=.1, color='red')
            '''
            ax.plot(True_params[sort_idx, idx], General_lb[sort_idx, idx], color='blue')
            ax.plot(True_params[sort_idx, idx], General_ub[sort_idx, idx], color='blue')
            ax.fill_between(True_params[sort_idx, idx], General_lb[sort_idx, idx], General_ub[sort_idx, idx], alpha=.1, color='red')
            '''
            ax.text(-0.35, 1.0, "(" + string.ascii_lowercase[idx] + ")", transform=ax.transAxes,
                    size=25, weight='bold')
            ax.set_ylabel(yaxes[idx])
            ax.set_xlim(ranges1[2 * idx], ranges1[2 * idx + 1])
            ax.set_ylim(ranges1[2 * idx], ranges1[2 * idx + 1])
            ax.minorticks_on()
            ax.grid(which='major', linestyle='-', linewidth='0.5')
            ax.grid(which='minor', linestyle=':', linewidth='0.5')

        else:
            ax.hist([np.array([100]), np.array([100]), np.array([100])], bins=20,
                    color=['green', 'red', 'blue'],
                    label=['Standard (N=10)', 'Individual (N=5)', 'Generalized (N=5)'],
                    range=(0, 1))
            ax.legend(loc='upper left')
            ax.axis('off')
    fig3_name = "fig2_balanced"
else:
    for idx, ax in enumerate(axs):
        if idx < 5:
            sort_idx = np.argsort(True_params[:, idx])
            ax.plot(True_params[sort_idx, idx], Standard_lb[sort_idx, idx], color='green', linewidth=lw_bound)
            ax.plot(True_params[sort_idx, idx], Standard_ub[sort_idx, idx], color='green', linewidth=lw_bound)
            ax.plot(True_params[sort_idx, idx], Standard_med[sort_idx, idx], color='green', linewidth=lw_med)
            ax.fill_between(True_params[sort_idx, idx], Standard_lb[sort_idx, idx], Standard_ub[sort_idx, idx],
                            alpha=.1, color='green')
            '''
            ax.plot(True_params[sort_idx, idx], Individual5_lb[sort_idx, idx], color=cmap_reds_1, linewidth=lw_bound)
            ax.plot(True_params[sort_idx, idx], Individual5_ub[sort_idx, idx], color=cmap_reds_1, linewidth=lw_bound)
            ax.plot(True_params[sort_idx, idx], Individual5_med[sort_idx, idx], color=cmap_reds_1, linewidth=lw_med)
            ax.fill_between(True_params[sort_idx, idx], Individual5_lb[sort_idx, idx], Individual5_ub[sort_idx, idx],
                            alpha=.1, color=cmap_reds_1)
            ax.plot(True_params[sort_idx, idx], Individual6_lb[sort_idx, idx], color=cmap_reds_2, linewidth=lw_bound)
            ax.plot(True_params[sort_idx, idx], Individual6_ub[sort_idx, idx], color=cmap_reds_2, linewidth=lw_bound)
            ax.plot(True_params[sort_idx, idx], Individual6_med[sort_idx, idx], color=cmap_reds_2, linewidth=lw_med)
            ax.fill_between(True_params[sort_idx, idx], Individual6_lb[sort_idx, idx], Individual6_ub[sort_idx, idx],
                            alpha=.1, color=cmap_reds_2)
            ax.plot(True_params[sort_idx, idx], Individual7_lb[sort_idx, idx], color=cmap_reds_3, linewidth=lw_bound)
            ax.plot(True_params[sort_idx, idx], Individual7_ub[sort_idx, idx], color=cmap_reds_3, linewidth=lw_bound)
            ax.plot(True_params[sort_idx, idx], Individual7_med[sort_idx, idx], color=cmap_reds_3, linewidth=lw_med)
            ax.fill_between(True_params[sort_idx, idx], Individual7_lb[sort_idx, idx], Individual7_ub[sort_idx, idx],
                            alpha=.1, color=cmap_reds_3)
            '''
            ax.plot(True_params[sort_idx, idx], General10_1_lb[sort_idx, idx], color=cmap_reds_3, linewidth=lw_bound)
            ax.plot(True_params[sort_idx, idx], General10_1_ub[sort_idx, idx], color=cmap_reds_3, linewidth=lw_bound)
            ax.plot(True_params[sort_idx, idx], General10_1_med[sort_idx, idx], color=cmap_reds_3, linewidth=lw_med)
            ax.fill_between(True_params[sort_idx, idx], General10_1_lb[sort_idx, idx], General10_1_ub[sort_idx, idx], alpha=.1,
                            color=cmap_reds_3)
            #ax.plot(True_params[sort_idx, idx], General10_2_lb[sort_idx, idx], color=cmap_blues_2, linewidth=lw_bound)
            #ax.plot(True_params[sort_idx, idx], General10_2_ub[sort_idx, idx], color=cmap_blues_2, linewidth=lw_bound)
            #ax.plot(True_params[sort_idx, idx], General10_2_med[sort_idx, idx], color=cmap_blues_2, linewidth=lw_med)
            #ax.fill_between(True_params[sort_idx, idx], General10_2_lb[sort_idx, idx], General10_2_ub[sort_idx, idx],
            #                alpha=.1,
            #                color=cmap_blues_2)
            ax.plot(True_params[sort_idx, idx], General10_3_lb[sort_idx, idx], color=cmap_blues_3, linewidth=lw_bound)
            ax.plot(True_params[sort_idx, idx], General10_3_ub[sort_idx, idx], color=cmap_blues_3, linewidth=lw_bound)
            ax.plot(True_params[sort_idx, idx], General10_3_med[sort_idx, idx], color=cmap_blues_3, linewidth=lw_med)
            ax.fill_between(True_params[sort_idx, idx], General10_3_lb[sort_idx, idx], General10_3_ub[sort_idx, idx],
                            alpha=.1,
                            color=cmap_blues_3)

            ax.text(-0.35, 1.0, "(" + string.ascii_lowercase[idx] + ")", transform=ax.transAxes,
                    size=25, weight='bold')
            ax.set_ylabel(yaxes[idx])
            ax.set_xlim(ranges1[2 * idx], ranges1[2 * idx + 1])
            ax.set_ylim(ranges1[2 * idx], ranges1[2 * idx + 1])
            ax.minorticks_on()
            ax.grid(which='major', linestyle='-', linewidth='0.5')
            ax.grid(which='minor', linestyle=':', linewidth='0.5')

        else:
            ax.hist([np.array([100]), np.array([100]), np.array([100]), np.array([100]), np.array([100]), np.array([100]), np.array([100])], bins=20,
                    color=['green', cmap_reds_1, cmap_reds_2, cmap_reds_3, cmap_blues_1, cmap_blues_2, cmap_blues_3],
                    label=['Standard (N=10)', 'Individual (N=5)', 'Individual (N=6)', 'Individual (N=7)', 'General_1 (N=10)', 'General_2 (N=10)', 'General_3 (N=10)'],
                    range=(0, 1))
            ax.legend(loc='upper left')
            ax.axis('off')

    fig3_name = "fig2_unbalanced"

plt.style.use('seaborn-v0_8-muted')
plt.tight_layout()
plt.subplots_adjust(wspace=0.4, hspace=0.4)
plt.savefig(fig3_name + ".png", dpi=300)
plt.show()


d