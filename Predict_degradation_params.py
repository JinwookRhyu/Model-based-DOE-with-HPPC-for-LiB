import numpy as np
import matplotlib.pyplot as plt
import emcee
import corner
import multiprocessing
import warnings
from MCMC_autocorrelation_help import lnprob, Tesla_graphite, Tesla_NCA_Si, get_muR_from_OCV, W, autocorr_gw2010, \
    autocorr_new, W_initial
import pickle
import os

suffix_save = "091725"

n_processes = 12     # Number of cores for multiprocessing
saveplot = True      # Whether to save the plots for MCMC simulations
I_err = 0.0005       # Measurement error in current due to battery cycler (0.0005 -> 0.05%)
V_limit_high = 0.200 # Upper limit for the voltage pulse in [V]
V_limit_low = 0.050  # Lower limit for the voltage pulse in [V]
c_c_limit_high = 0.8 # Upper limit for the cathode filling fraction
c_c_limit_low = 0.4  # Lower limit for the cathode filling fraction
rxn_method = "CIET"  # 'CIET' for coupled ion electron transfer model. 'BV' for Butler-Volmer
num_file = 0

# HPPC protocol setting
c_c = np.array([0.8,   0.8,   0.8,   0.67, 0.666, 0.4, 0.4, 0.4, 0.4, 0.4])
dV = np.array([0.05, -0.05,  -0.2, -0.2, -0.2, -0.05, 0.2, 0.2, 0.05, -0.05])
W_list = np.array([[0.852, 0.755, 0.72 , 0.712, 0.712, 0.702, 0.758, 0.759, 0.708, 0.701]])

N = len(c_c)  # Number of pulses

# LiB parameters
# Particle size
r_c = 0.46e-6
r_a = 0.5e-6
# Electrode thickness
L_c = 64e-6
L_a = 83e-6
# Volume loading percents of active material (volume fraction of solid that is active material)
P_L_c = 0.7452
P_L_a = 0.8277
# Porosities (liquid volume fraction in each region)
poros_c = 0.2298
poros_a = 0.1473
# Site density of electrode active materials (sites/m^3)
rho_s_c = 3.276e28
rho_s_a = 1.7438e28
# Initial concentration of electrodes
c_s_0_a = 0.0142
c_s_0_c = 0.8595

# Rescaling factor to convert particle level current to electrode level
f_c = L_c * (1 - poros_c) * P_L_c * 3 / r_c
f_a = L_a * (1 - poros_a) * P_L_a * 3 / r_a

# Rescaling factor to balance electrode concentrations
p_c = L_c * (1 - poros_c) * P_L_c * rho_s_c
p_a = L_a * (1 - poros_a) * P_L_a * rho_s_a

# Set reference chemical potentials
mu_c = Tesla_NCA_Si
mu_a = Tesla_graphite
muR_ref_c = -mu_c(y=np.array([c_s_0_c]), muR_ref=0)[0]
muR_ref_a = -mu_a(y=np.array([c_s_0_a]), muR_ref=0)[0]

params_c = {'rxn_method': rxn_method, 'k0': 18, 'lambda': 5, 'f': f_c, 'p': p_c, 'c0': c_s_0_c,
            'mu': mu_c, 'muR_ref': muR_ref_c}
params_a = {'rxn_method': rxn_method, 'k0': 0.2, 'lambda': 8, 'f': f_a, 'p': p_a, 'c0': c_s_0_a,
            'mu': mu_a, 'muR_ref': muR_ref_a}

# Lower and upper limits for degradation parameters in R_f_c, c_tilde_c, R_f_a, c_tilde_a, c_lyte order
R_f_c_range = np.array([0, 0.2])       # Range for R_f_c (lb, ub)
c_tilde_c_range = np.array([0.8, 1])  # Range for c_tilde_c (lb, ub)
R_f_a_range = np.array([0, 0.2])       # Range for R_f_a (lb, ub)
c_tilde_a_range = np.array([0.8, 1])  # Range for c_tilde_a (lb, ub)
c_lyte_range = np.array([0.8, 1])     # Range for c_lyte (lb, ub)

deg_params_bound = np.vstack([R_f_c_range, c_tilde_c_range, R_f_a_range, c_tilde_a_range, c_lyte_range])

deg_params_lower = deg_params_bound[:,0]
deg_params_upper = deg_params_bound[:,1]
str_deg_params = str(deg_params_bound[0][0]) + "_" + str(deg_params_bound[0][1]) + "_" + str(deg_params_bound[1][0])\
                 + "_" + str(deg_params_bound[1][1]) + "_" + str(deg_params_bound[2][0]) + "_" + str(deg_params_bound[2][1])\
                 + "_" + str(deg_params_bound[3][0]) + "_" + str(deg_params_bound[3][1]) + "_" + str(deg_params_bound[4][0])\
                 + "_" + str(deg_params_bound[4][1])

# Directory for saving MCMC figures and autocorrelation figures (to check that we are using sufficient number of steps in MCMC)
dir_savefig = os.path.join(os.getcwd(), f"Identifiability_analysis_{suffix_save}")

if not os.path.exists(dir_savefig):
    os.mkdir(dir_savefig)

results={} # Variable for saving the MCMC results

for mm in range(len(W_list)):
    warnings.filterwarnings("ignore")

    # MCMC parameters
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_DYNAMIC"] = "FALSE"
    nwalkers = 24  # Number of walkers
    niter = 10000  # Number of iterations. Users can adjust niter based on autocorrelation figures

    plt.rcParams['figure.figsize'] = (20, 10)

    # Spread walkers
    ndim = len(deg_params_lower)
    rng = np.random.RandomState(0)
    p0 = [deg_params_lower + np.multiply(deg_params_upper - deg_params_lower, rng.uniform(0, 1, ndim)) for i in
          range(int(nwalkers * 4 / 3))]

    # Directory for saving the figures
    savename = os.path.join(dir_savefig, f"MCMC_opt_predict_{str_deg_params}_sample_{mm}_{suffix_save}.png")
    savename_autocorr = os.path.join(dir_savefig, f"Autocorr_predict_{str_deg_params}_sample_{mm}_{suffix_save}.png")

    # Calculate anode filling fraction using material balance
    c_a = params_a["c0"] - params_c["p"] / params_a["p"] * (c_c - params_c["c0"])
    # Calculate chemical potential for cathode and anode
    pulse_range = get_muR_from_OCV(OCV=dV, muR_ref=0)  # Difference in chemical potential introduced by voltage pulse
    y = W_list[mm]
    # Parameters other than degradation parameter for MCMC simulation
    data = (c_c, c_a, mu_c, mu_a, params_c, params_a, y, I_err, pulse_range, deg_params_bound)

    if __name__ == '__main__':
        pool = multiprocessing.Pool(processes=n_processes)

        sampler = emcee.EnsembleSampler(int(nwalkers * 4 / 3), ndim, lnprob, args=data, pool=pool)

        # ---- Burn-in ----
        print("Running burn-in...")
        state = sampler.run_mcmc(p0, 100, progress=True)  # state: emcee.State
        coords = state.coords  # (nwalkers, ndim)
        logp = state.log_prob

        # ---- Select survivors (e.g., keep top 75%) ----
        keep_idx = np.argsort(logp)[-nwalkers:]
        coords_keep = coords[keep_idx]

        nkeep = coords_keep.shape[0]
        assert nkeep >= 2 * ndim, "Too few walkers after dropping; lower the cutoff or reseed instead."

        # ---- Start a NEW sampler for production with fewer walkers ----
        # (Reuse the same moves/pool/args you had before.)
        sampler_prod = emcee.EnsembleSampler(nkeep, ndim, lnprob, args=data, pool=pool)

        # Optional: reset the old sampler (not strictly necessary since we won't use it)
        sampler.reset()

        # ---- Production run ----
        print("Running production...")
        state_prod = sampler_prod.run_mcmc(coords_keep, niter, progress=True)

        pool.close()

        print("Mean acceptance fraction: {0:.3f}".format(np.mean(sampler.acceptance_fraction)))
        chain = sampler_prod.get_chain()[:, :, 0].T

        # Compute the estimators for a few different chain lengths
        Nsteps = np.exp(np.linspace(np.log(100), np.log(chain.shape[1]), 10)).astype(int)
        gw2010 = np.empty(len(Nsteps))
        new = np.empty(len(Nsteps))
        for i, n in enumerate(Nsteps):
            gw2010[i] = autocorr_gw2010(chain[:, :n])
            new[i] = autocorr_new(chain[:, :n])

        # Plot the comparisons
        plt.loglog(Nsteps, gw2010, "o-", label="G&W 2010")
        plt.loglog(Nsteps, new, "o-", label="new")
        ylim = plt.gca().get_ylim()
        plt.plot(Nsteps, Nsteps / 50.0, "--k", label=r"$\tau = N/50$")
        plt.ylim(ylim)
        plt.xlabel("number of samples, $N$")
        plt.ylabel(r"$\tau$ estimates")
        plt.legend(fontsize=14)
        if saveplot:
            plt.savefig(savename_autocorr, dpi=300, bbox_inches='tight')

        samples = sampler_prod.flatchain

        results[mm] = {}
        deg_params_predicted = np.percentile(samples, 50, axis=0)
        results[mm]['Estimated_params'] = np.percentile(samples, 50, axis=0)
        results[mm]['lb_95'] = np.percentile(samples, 2.5, axis=0)
        results[mm]['ub_95'] = np.percentile(samples, 97.5, axis=0)
        results[mm]['lb_90'] = np.percentile(samples, 5, axis=0)
        results[mm]['ub_90'] = np.percentile(samples, 95, axis=0)
        results[mm]['c_c'] = c_c
        results[mm]['dV'] = dV
        results[mm]['y'] = y
        results[mm]['raw'] = samples
        labels = ['R_f_c', 'c_tilde_c', 'R_f_a', 'c_tilde_a', 'c_lyte']
        fig = corner.corner(samples, show_titles=True, labels=labels, plot_datapoints=True,
                            quantiles=[0.05, 0.5, 0.95],
                            title_fmt='.3f', range=[(deg_params_bound[0][0], deg_params_bound[0][1]), (deg_params_bound[1][0], deg_params_bound[1][1]), (deg_params_bound[2][0], deg_params_bound[2][1]), (deg_params_bound[3][0], deg_params_bound[3][1]), (deg_params_bound[4][0], deg_params_bound[4][1])],
                            plot_contours=True)
        fig_title = 'Predicted values'
        for k in range(len(deg_params_predicted)):
            fig_title = fig_title + '\n' + "{0:<10} = {1: .3f}".format(labels[k], deg_params_predicted[k])
        fig_title = fig_title + '\n\n\nPulses'
        for k in range(N):
            fig_title = fig_title + '\n' + "#{0:<2} = ({1:.3f}, {2} mV, {3:.3f})".format(int(k + 1), c_c[k], round(
                dV[
                    k] * 1000), W_list[mm][k])
        fig.suptitle(fig_title, x=0.67, fontsize=15, ha='left')

        # Extract the axes
        axes = np.array(fig.axes).reshape((ndim, ndim))

        # Loop over the diagonal
        for i in range(ndim):
            ax = axes[i, i]
            ax.axvline(deg_params_predicted[i], color="r")

        # Loop over the histograms
        for yi in range(ndim):
            for xi in range(yi):
                ax = axes[yi, xi]
                ax.axvline(deg_params_predicted[xi], color="r")
                ax.axhline(deg_params_predicted[yi], color="r")
                ax.plot(deg_params_predicted[xi], deg_params_predicted[yi], "sr")
        if saveplot:
            plt.savefig(savename, dpi=300, bbox_inches='tight')
        # plt.show()
        sampler_prod.reset()

# Save the identifiability analysis results in npz and pkl files
save_path = os.path.join(os.getcwd(), f"predict_{str_deg_params}_{suffix_save}.npz")

np.savez_compressed(
    save_path,
    Estimated_params=np.array([results[mm]['Estimated_params'] for mm in results]),
    lb_95=np.array([results[mm]['lb_95'] for mm in results]),
    ub_95=np.array([results[mm]['ub_95'] for mm in results]),
    lb_90=np.array([results[mm]['lb_90'] for mm in results]),
    ub_90=np.array([results[mm]['ub_90'] for mm in results]),
    c_c=np.array([results[mm]['c_c'] for mm in results]),
    dV=np.array([results[mm]['dV'] for mm in results]),
    y=np.array([results[mm]['y'] for mm in results]),
    raw=np.array([results[mm]['raw'] for mm in results]),
)
