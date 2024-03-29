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
lw_med = 1
lw_bound = 0.1

Standard = []
General10_1 = []
General10_2 = []
General13_1 = []
General13_2 = []



with (open(os.getcwd() + "/standard_N10_0.0_10.0_0.8_1.0_0.0_10.0_0.8_1.0_0.8_1.0_unbalanced.pkl", "rb")) as openfile:
    while True:
        try:
            Standard.append(pickle.load(openfile))
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

with (open(os.getcwd() + "/generalized_optimal_N13_0.0_10.0_0.8_1.0_0.0_10.0_0.8_1.0_0.8_1.0_unbalanced_1.pkl",
           "rb")) as openfile:
    while True:
        try:
            General13_1.append(pickle.load(openfile))
        except EOFError:
            break

with (open(os.getcwd() + "/generalized_optimal_N13_0.0_10.0_0.8_1.0_0.0_10.0_0.8_1.0_0.8_1.0_unbalanced_2.pkl",
           "rb")) as openfile:
    while True:
        try:
            General13_2.append(pickle.load(openfile))
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


Uncertainty_general10_1_lb = np.zeros((len(General10_1[0]), 5))
Uncertainty_general10_1_ub = np.zeros((len(General10_1[0]), 5))
Uncertainty_general10_1 = np.zeros((len(General10_1[0]), 5))
Error_general10_1 = np.zeros((len(General10_1[0]), 5))
Uncertainty_general10_2_lb = np.zeros((len(General10_2[0]), 5))
Uncertainty_general10_2_ub = np.zeros((len(General10_2[0]), 5))
Uncertainty_general10_2 = np.zeros((len(General10_2[0]), 5))
Error_general10_2 = np.zeros((len(General10_2[0]), 5))
Uncertainty_general13_1_lb = np.zeros((len(General13_1[0]), 5))
Uncertainty_general13_1_ub = np.zeros((len(General13_1[0]), 5))
Uncertainty_general13_1 = np.zeros((len(General13_1[0]), 5))
Error_general13_1 = np.zeros((len(General13_1[0]), 5))
Uncertainty_general13_2_lb = np.zeros((len(General13_2[0]), 5))
Uncertainty_general13_2_ub = np.zeros((len(General13_2[0]), 5))
Uncertainty_general13_2 = np.zeros((len(General13_2[0]), 5))
Error_general13_2 = np.zeros((len(General13_2[0]), 5))


General10_1_lb = np.zeros((len(General10_1[0]), 5))
General10_1_ub = np.zeros((len(General10_1[0]), 5))
General10_1_med = np.zeros((len(General10_1[0]), 5))
General10_2_lb = np.zeros((len(General10_2[0]), 5))
General10_2_ub = np.zeros((len(General10_2[0]), 5))
General10_2_med = np.zeros((len(General10_2[0]), 5))
General13_1_lb = np.zeros((len(General13_1[0]), 5))
General13_1_ub = np.zeros((len(General13_1[0]), 5))
General13_1_med = np.zeros((len(General13_1[0]), 5))
General13_2_lb = np.zeros((len(General13_2[0]), 5))
General13_2_ub = np.zeros((len(General13_2[0]), 5))
General13_2_med = np.zeros((len(General13_2[0]), 5))


for k in range(len(Standard[0])):
    Uncertainty_standard_lb[k, :] = Standard[0][k]['lb_90'] - Standard[0][k]['True_params']
    Uncertainty_standard_ub[k, :] = Standard[0][k]['ub_90'] - Standard[0][k]['True_params']
    Uncertainty_standard[k, :] = Standard[0][k]['ub_90'] - Standard[0][k]['lb_90']
    Error_standard[k, :] = Standard[0][k]['Estimated_params'] - Standard[0][k]['True_params']
    Standard_lb[k, :] = Standard[0][k]['lb_90']
    Standard_ub[k, :] = Standard[0][k]['ub_90']
    Standard_med[k, :] = Standard[0][k]['Estimated_params']

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
    Uncertainty_general13_1_lb[k, :] = General13_1[0][k]['lb_90'] - General13_1[0][k]['True_params']
    Uncertainty_general13_1_ub[k, :] = General13_1[0][k]['ub_90'] - General13_1[0][k]['True_params']
    Uncertainty_general13_1[k, :] = General13_1[0][k]['ub_90'] - General13_1[0][k]['lb_90']
    Error_general13_1[k, :] = General13_1[0][k]['Estimated_params'] - General13_1[0][k]['True_params']
    General13_1_lb[k, :] = General13_1[0][k]['lb_90']
    General13_1_ub[k, :] = General13_1[0][k]['ub_90']
    General13_1_med[k, :] = General13_1[0][k]['Estimated_params']
    Uncertainty_general13_2_lb[k, :] = General13_2[0][k]['lb_90'] - General13_2[0][k]['True_params']
    Uncertainty_general13_2_ub[k, :] = General13_2[0][k]['ub_90'] - General13_2[0][k]['True_params']
    Uncertainty_general13_2[k, :] = General13_2[0][k]['ub_90'] - General13_2[0][k]['lb_90']
    Error_general13_2[k, :] = General13_2[0][k]['Estimated_params'] - General13_2[0][k]['True_params']
    General13_2_lb[k, :] = General13_2[0][k]['lb_90']
    General13_2_ub[k, :] = General13_2[0][k]['ub_90']
    General13_2_med[k, :] = General13_2[0][k]['Estimated_params']


Uncertainty_standard_all = np.vstack((Uncertainty_standard_lb, Uncertainty_standard_ub))
Uncertainty_general10_1_all = np.vstack((Uncertainty_general10_1_lb, Uncertainty_general10_1_ub))
Uncertainty_general10_2_all = np.vstack((Uncertainty_general10_2_lb, Uncertainty_general10_2_ub))
Uncertainty_general13_1_all = np.vstack((Uncertainty_general13_1_lb, Uncertainty_general13_1_ub))
Uncertainty_general13_2_all = np.vstack((Uncertainty_general13_2_lb, Uncertainty_general13_2_ub))
True_params_rep = np.vstack((True_params, True_params))

Rel_Uncertainty_standard_all = np.divide(Uncertainty_standard_all, True_params_rep) * 100
Rel_Uncertainty_general10_1_all = np.divide(Uncertainty_general10_1_all, True_params_rep) * 100
Rel_Uncertainty_general10_2_all = np.divide(Uncertainty_general10_2_all, True_params_rep) * 100
Rel_Uncertainty_general13_1_all = np.divide(Uncertainty_general13_1_all, True_params_rep) * 100
Rel_Uncertainty_general13_2_all = np.divide(Uncertainty_general13_2_all, True_params_rep) * 100

Rel_Uncertainty_standard = np.divide(Uncertainty_standard, True_params) * 100
Rel_Uncertainty_general10_1 = np.divide(Uncertainty_general10_1, True_params) * 100
Rel_Uncertainty_general10_2 = np.divide(Uncertainty_general10_2, True_params) * 100
Rel_Uncertainty_general13_1 = np.divide(Uncertainty_general13_1, True_params) * 100
Rel_Uncertainty_general13_2 = np.divide(Uncertainty_general13_2, True_params) * 100

Rel_Error_standard = np.divide(Error_standard, True_params) * 100
Rel_Error_general10_1 = np.divide(Error_general10_1, True_params) * 100
Rel_Error_general10_2 = np.divide(Error_general10_2, True_params) * 100
Rel_Error_general13_1 = np.divide(Error_general13_1, True_params) * 100
Rel_Error_general13_2 = np.divide(Error_general13_2, True_params) * 100

#####################################################################################################

yaxes = [r'$R_{f,c}$', r'$\tilde{c}_c$', r'$R_{f,a}$', r'$\tilde{c}_a$', '$c_+$']
fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(8,8))
axs = axs.ravel()

ranges1 = [60, 2, 1, 1, 1]
ranges2 = [100, 20, 5, 5, 5]
for idx, ax in enumerate(axs):
    if idx < 5:
        ax.hist([np.abs(Rel_Error_standard[:, idx]).clip(min=0, max=ranges1[idx ]),
                 np.abs(Rel_Error_general10_1[:, idx]).clip(min=0, max=ranges1[idx ]),
                 np.abs(Rel_Error_general10_2[:, idx]).clip(min=0, max=ranges1[idx ]),
                 np.abs(Rel_Error_general13_1[:, idx]).clip(min=0, max=ranges1[idx ]),
                 np.abs(Rel_Error_general13_2[:, idx]).clip(min=0, max=ranges1[idx ])],
                 bins=10, color=['black', 'red', 'lime', 'blue', 'orange'], range=(0, ranges1[idx]))

        #ax.text(-0.29, 1.0, string.ascii_lowercase[idx] + ")", transform=ax.transAxes,
        #        size=25, weight='bold')
        ax.set_ylabel(yaxes[idx])
        ax.set_xlim(0, ranges1[idx])
        ax.set_ylim(0, 100)

        ax2 = fig.add_axes([0.3 + 0.47 * (idx % 2), 0.77 - 0.33 * (np.floor(idx / 2)), 0.15, 0.15])
        ax2.hist([Rel_Uncertainty_standard[:, idx].clip(min=0, max=ranges2[idx ]),
                  Rel_Uncertainty_general10_1[:, idx].clip(min=0, max=ranges2[idx]),
                  Rel_Uncertainty_general10_2[:, idx].clip(min=0, max=ranges2[idx ]),
                  Rel_Uncertainty_general13_1[:, idx].clip(min=0, max=ranges2[idx ]),
                  Rel_Uncertainty_general13_2[:, idx].clip(min=0, max=ranges2[idx ])],
                  bins=5,
                 color=['black', 'red', 'lime', 'blue', 'orange'], range=(0, ranges2[idx]))

        ax2.set_xlim(0, ranges2[idx])
        ax2.set_ylim(0, 100)

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
        ax.hist([np.array([100]), np.array([100]), np.array([100]), np.array([100]), np.array([100])], bins=20,
                color=['black', 'red', 'lime', 'blue', 'orange'],
                label=['Standard (N=10)', 'Optimal 1 (N=10)', 'Optimal 2 (N=10)', 'Optimal 3 (N=13)', 'Optimal 4 (N=13)'],
                range=(0, 1))
        ax.legend(loc='upper left')
        ax.axis('off')

fig3_name = "mcmc_unbalanced_histogram_3"

plt.style.use('seaborn-v0_8-muted')
plt.tight_layout()
plt.subplots_adjust(wspace=0.4, hspace=0.4)
plt.savefig(fig3_name + ".png", dpi=300)
plt.show()


###############################################################################

yaxes = [r'Predicted $R_{f,c}$', r'Predicted $\tilde{c}_c$', r'Predicted $R_{f,a}$', r'Predicted $\tilde{c}_a$', 'Predicted $c_+$']
xaxes = [r'Actual $R_{f,c}$', r'Actual $\tilde{c}_c$', r'Actual $R_{f,a}$', r'Actual $\tilde{c}_a$', 'Actual $c_+$']
fig, axs = plt.subplots(nrows=1, ncols=5, figsize=(16,3))
axs = axs.ravel()

ranges1 = [0, 10, 0.8, 1.0, 0, 10, 0.8, 1.0, 0.8, 1.0]


for idx, ax in enumerate(axs):
    if idx < 5:
        sort_idx = np.argsort(True_params[:, idx])
        ax.plot(True_params[sort_idx, idx], Standard_lb[sort_idx, idx], color='black', linewidth=lw_bound)
        ax.plot(True_params[sort_idx, idx], Standard_ub[sort_idx, idx], color='black', linewidth=lw_bound)
        ax.plot(True_params[sort_idx, idx], Standard_med[sort_idx, idx], color='black', linewidth=lw_med)
        ax.fill_between(True_params[sort_idx, idx], Standard_lb[sort_idx, idx], Standard_ub[sort_idx, idx],
                        alpha=.1, color='black')

        ax.plot(True_params[sort_idx, idx], General10_1_lb[sort_idx, idx], color='red', linewidth=lw_bound)
        ax.plot(True_params[sort_idx, idx], General10_1_ub[sort_idx, idx], color='red', linewidth=lw_bound)
        ax.plot(True_params[sort_idx, idx], General10_1_med[sort_idx, idx], color='red', linewidth=lw_med)
        ax.fill_between(True_params[sort_idx, idx], General10_1_lb[sort_idx, idx], General10_1_ub[sort_idx, idx], alpha=.1,
                        color='red')
        ax.plot(True_params[sort_idx, idx], General10_2_lb[sort_idx, idx], color='lime', linewidth=lw_bound)
        ax.plot(True_params[sort_idx, idx], General10_2_ub[sort_idx, idx], color='lime', linewidth=lw_bound)
        ax.plot(True_params[sort_idx, idx], General10_2_med[sort_idx, idx], color='lime', linewidth=lw_med)
        ax.fill_between(True_params[sort_idx, idx], General10_2_lb[sort_idx, idx], General10_2_ub[sort_idx, idx],
                        alpha=.1,
                        color='lime')
        ax.plot(True_params[sort_idx, idx], General13_1_lb[sort_idx, idx], color='blue', linewidth=lw_bound)
        ax.plot(True_params[sort_idx, idx], General13_1_ub[sort_idx, idx], color='blue', linewidth=lw_bound)
        ax.plot(True_params[sort_idx, idx], General13_1_med[sort_idx, idx], color='blue', linewidth=lw_med)
        ax.fill_between(True_params[sort_idx, idx], General13_1_lb[sort_idx, idx], General13_1_ub[sort_idx, idx],
                        alpha=.1,
                        color='blue')
        ax.plot(True_params[sort_idx, idx], General13_2_lb[sort_idx, idx], color='orange', linewidth=lw_bound)
        ax.plot(True_params[sort_idx, idx], General13_2_ub[sort_idx, idx], color='orange', linewidth=lw_bound)
        ax.plot(True_params[sort_idx, idx], General13_2_med[sort_idx, idx], color='orange', linewidth=lw_med)
        ax.fill_between(True_params[sort_idx, idx], General13_2_lb[sort_idx, idx], General13_2_ub[sort_idx, idx],
                        alpha=.1,
                        color='orange')


        #ax.text(-0.35, 1.0, string.ascii_lowercase[idx] + ")", transform=ax.transAxes,
        #        size=25, weight='bold')
        ax.set_ylabel(yaxes[idx])
        ax.set_xlabel(xaxes[idx])
        ax.set_xlim(ranges1[2 * idx], ranges1[2 * idx + 1])
        ax.set_ylim(ranges1[2 * idx], ranges1[2 * idx + 1])
        ax.minorticks_on()
        ax.grid(which='major', linestyle='-', linewidth='0.5')
        ax.grid(which='minor', linestyle=':', linewidth='0.5')

    else:
        ax.hist([np.array([100]), np.array([100]), np.array([100]), np.array([100]), np.array([100])], bins=20,
                color=['black', 'red', 'lime', 'blue', 'orange'],
                label=['Standard (N=10)', 'Optimal 1 (N=10)', 'Optimal 2 (N=10)', 'Optimal 3 (N=13)', 'Optimal 4 (N=13)'],
                range=(0, 1))
        ax.legend(loc='upper left')
        ax.axis('off')

fig3_name = "mcmc_unbalanced_line"

plt.style.use('seaborn-v0_8-muted')
plt.tight_layout()
plt.subplots_adjust(wspace=0.4, hspace=0.4)
plt.savefig(fig3_name + ".png", dpi=300)
plt.show()

#####################################################################################################

yaxes = [r'$R_{f,c}$', r'$\tilde{c}_c$', r'$R_{f,a}$', r'$\tilde{c}_a$', '$c_+$']
fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(8,8))
axs = axs.ravel()

ranges1 = [60, 2, 1, 1, 1]
ranges2 = [100, 20, 5, 5, 5]
for idx, ax in enumerate(axs):
    if idx < 5:
        x1 = np.linspace(0, ranges1[idx], 21)
        y01 = np.zeros((len(x1),))
        y11 = np.zeros((len(x1),))
        y21 = np.zeros((len(x1),))
        y31 = np.zeros((len(x1),))
        y41 = np.zeros((len(x1),))
        x2 = np.linspace(0, ranges2[idx], len(x1))
        y02 = np.zeros((len(x1),))
        y12 = np.zeros((len(x1),))
        y22 = np.zeros((len(x1),))
        y32 = np.zeros((len(x1),))
        y42 = np.zeros((len(x1),))
        for k in range(len(x1)):
            y01[k] = np.sum(np.abs(Rel_Error_standard[:, idx]) < x1[k])
            y11[k] = np.sum(np.abs(Rel_Error_general10_1[:, idx]) < x1[k])
            y21[k] = np.sum(np.abs(Rel_Error_general10_2[:, idx]) < x1[k])
            y31[k] = np.sum(np.abs(Rel_Error_general13_1[:, idx]) < x1[k])
            y41[k] = np.sum(np.abs(Rel_Error_general13_2[:, idx]) < x1[k])
            y02[k] = np.sum(np.abs(Rel_Uncertainty_standard[:, idx]) < x2[k])
            y12[k] = np.sum(np.abs(Rel_Uncertainty_general10_1[:, idx]) < x2[k])
            y22[k] = np.sum(np.abs(Rel_Uncertainty_general10_2[:, idx]) < x2[k])
            y32[k] = np.sum(np.abs(Rel_Uncertainty_general13_1[:, idx]) < x2[k])
            y42[k] = np.sum(np.abs(Rel_Uncertainty_general13_2[:, idx]) < x2[k])
        ax.plot(x1, y01, label='Standard (N=10)', color='black')
        ax.plot(x1, y11, label='Optimal 1 (N=10)', color='red')
        ax.plot(x1, y21, label='Optimal 2 (N=10)', color='lime')
        ax.plot(x1, y31, label='Optimal 3 (N=13)', color='blue')
        ax.plot(x1, y41, label='Optimal 4 (N=14)', color='orange')

        ax.set_ylabel(yaxes[idx])
        ax.set_xlim(0, ranges1[idx])
        ax.set_ylim(0, 100)

        ax2 = fig.add_axes([0.3 + 0.47 * (idx % 2), 0.77 - 0.33 * (np.floor(idx / 2)), 0.15, 0.15])
        ax2.plot(x2, y02, label='Standard (N=10)', color='black')
        ax2.plot(x2, y12, label='Optimal 1 (N=10)', color='red')
        ax2.plot(x2, y22, label='Optimal 2 (N=10)', color='lime')
        ax2.plot(x2, y32, label='Optimal 3 (N=13)', color='blue')
        ax2.plot(x2, y42, label='Optimal 4 (N=14)', color='orange')
        ax2.set_xlim(0, ranges2[idx])
        ax2.set_ylim(0, 100)

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

        print("########################################")
        print(y01)
        print(y11)
        print(y21)
        print(y31)
        print(y41)
        print(y02)
        print(y12)
        print(y22)
        print(y32)
        print(y42)


    else:
        ax.hist([np.array([100]), np.array([100]), np.array([100]), np.array([100]), np.array([100])], bins=20,
                color=['black', 'red', 'lime', 'blue', 'orange'],
                label=['Standard (N=10)', 'Optimal 1 (N=10)', 'Optimal 2 (N=10)', 'Optimal 3 (N=13)', 'Optimal 4 (N=13)'],
                range=(0, 1))
        ax.legend(loc='upper left')
        ax.axis('off')

fig3_name = "mcmc_unbalanced_plot"

plt.style.use('seaborn-v0_8-muted')
plt.tight_layout()
plt.subplots_adjust(wspace=0.4, hspace=0.4)
plt.savefig(fig3_name + ".png", dpi=300)
plt.show()