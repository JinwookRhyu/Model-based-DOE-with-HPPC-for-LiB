import numpy as np
import matplotlib.pyplot as plt
import emcee
import corner
import multiprocessing
import warnings
from MCMC_autocorrelation_help import lnprob, Tesla_graphite, Tesla_NCA_Si, get_muR_from_OCV, W, autocorr_gw2010, autocorr_new
import pickle

I_err = 0.0001
tpe = "D"
rxn_method = "CIET"

# Lower and upper limits for degradation parameters in R_f_c, c_tilde_c, R_f_a, c_tilde_a, c_lyte order
R_f_c_range = np.array([0, 10]) # Range for R_f_c (lb, ub)
c_tilde_c_range = np.array([0.8, 1]) # Range for c_tilde_c (lb, ub)
R_f_a_range = np.array([0, 10]) # Range for R_f_a (lb, ub)
c_tilde_a_range = np.array([0.8, 1]) # Range for c_tilde_a (lb, ub)
c_lyte_range = np.array([0.8, 1]) # Range for c_lyte (lb, ub)

deg_params_range = np.reshape(np.concatenate((R_f_c_range, c_tilde_c_range, R_f_a_range, c_tilde_a_range, c_lyte_range), axis=0), (1, 10))

deg_params_lower = np.array([deg_params_range[0][0], deg_params_range[0][2], deg_params_range[0][4], deg_params_range[0][6], deg_params_range[0][8]])
deg_params_upper = np.array([deg_params_range[0][1], deg_params_range[0][3], deg_params_range[0][5], deg_params_range[0][7], deg_params_range[0][9]])
str_deg_params_range = str(deg_params_range[0][0]) + "_" + str(deg_params_range[0][1]) + "_" + str(deg_params_range[0][2]) + "_" + str(deg_params_range[0][3]) + \
                        "_" + str(deg_params_range[0][4]) + "_" + str(deg_params_range[0][5]) + "_" + str(deg_params_range[0][6]) + "_" + str(deg_params_range[0][7]) + \
                         "_" + str(deg_params_range[0][8]) + "_" + str(deg_params_range[0][9])

# Directory for saving MCMC figures and autocorrelation figures
dir_savefig = "C:/Users/Jinwook/PyCharm_projects/DOE/I_err_" + str(I_err) + "_standard_N5_similar_" + str_deg_params_range
dir_txtfile = "C:/Users/Jinwook/PyCharm_projects/DOE/I_err_" + str(I_err) + "_standard_N5_" + str_deg_params_range

results={}
ii = 0
import glob, os
for mm in range(100):
    np.random.seed(mm)
    deg_params = deg_params_lower + np.multiply((deg_params_upper - deg_params_lower), np.random.uniform(0, 1, 5))
    deg_params = np.round(deg_params * 1000) / 1000
    #if "error_bound_CIET_D_50mV" in file:

    numpulse = 10  # Number of pulses.
    saveplot = True

    warnings.filterwarnings("ignore")

    os.environ["OMP_NUM_THREADS"] = "1"
    nwalkers = 24
    niter = 10000

    plt.rcParams['figure.figsize'] = (20, 10)
    V_limit = 0.050

    # particle size
    r_c = 11e-6
    r_a = 17e-6
    # lengths
    L_c = 64e-6
    L_a = 83e-6
    # Volume loading percents of active material (volume fraction of solid
    # that is active material)
    P_L_c = 0.7452
    P_L_a = 0.8277
    # Porosities (liquid volume fraction in each region)
    poros_c = 0.2298
    poros_a = 0.1473
    # width and thickness of electrodes
    width_a = 65e-3
    width_c = 63e-3
    thick_a = 86.5e-2
    thick_c = 85.9e-2

    # now call optimization funciton
    # site density of electrode active materials (sites/m^3)
    rho_s_c = 3.276e28
    rho_s_a = 1.7438e28

    # initial concentration of electrodes
    c_s_0_a = 0.833
    c_s_0_c = 0.167

    # rescaling factor to convert particle level current to electrode level
    f_c = L_c * (1 - poros_c) * P_L_c * 1 / r_c
    f_a = L_a * (1 - poros_a) * P_L_a * 1 / r_a

    # rescaling factor to balance electrode concentrations
    p_c = L_c * (1 - poros_c) * P_L_c * rho_s_c
    p_a = L_a * (1 - poros_a) * P_L_a * rho_s_a

    # set reference chemical ptoentials
    muR_ref_c = -Tesla_NCA_Si(np.array([c_s_0_c]), 0)[0]
    muR_ref_a = -Tesla_graphite(np.array([c_s_0_a]), 0)[0]
    params_c = {'rxn_method': rxn_method, 'k0': 1, 'lambda': 5, 'f': f_c, 'p': p_c, 'c0': c_s_0_c,
                'mu': Tesla_NCA_Si,
                'muR_ref': muR_ref_c}
    params_a = {'rxn_method': rxn_method, 'k0': 1, 'lambda': 5, 'f': f_a, 'p': p_a, 'c0': c_s_0_a,
                'mu': Tesla_graphite,
                'muR_ref': muR_ref_a}

    #items = file.split('_')
    #R_f_c = float(items[5])
    #c_tilde_c = float(items[6])
    #R_f_a = float(items[7])
    #c_tilde_a = float(items[8])
    #c_lyte = float(items[9].replace('.txt', ''))
    R_f_c = deg_params[0]
    c_tilde_c = deg_params[1]
    R_f_a = deg_params[2]
    c_tilde_a = deg_params[3]
    c_lyte = deg_params[4]

    deg_params_true = np.array([R_f_c, c_tilde_c, R_f_a, c_tilde_a, c_lyte])

    # TODO: Check I_err value of the experimental equipment [A/m^2]
    # TODO: Check with Debbie on err_y expression... R_value has [A/m^2]. Should I_err also have [A/m^2]?
    str_deg_params = str(deg_params_true[0]) + "_" + str(deg_params_true[1]) + "_" + str(
        deg_params_true[2]) + "_" + str(deg_params_true[3]) + "_" + str(deg_params_true[4])
    '''with open(dir_txtfile + "/optimized_output_" + str(
            rxn_method) + "_" + str(tpe) + "_" + str(int(1000 * V_limit)) + "mV_" + str(
            str_deg_params) + ".txt") as f:
        contents = f.readlines()'''

    N = 10
    #N_max = int(float(contents[-1].split()[0]))

    c_c = np.array([0.8, 0.8, 0.7, 0.7, 0.6, 0.6, 0.5, 0.5, 0.4, 0.4])
    delta_V = np.array([0.2, -0.2, 0.2, -0.2, 0.2, -0.2, 0.2, -0.2, 0.2, -0.2])

    # Spread walkers
    initial = np.array([deg_params_range[0][0], deg_params_range[0][2], deg_params_range[0][4], deg_params_range[0][6], deg_params_range[0][8]])
    ndim = len(initial)
    np.random.seed(0)
    p0 = [np.array(initial) + np.multiply(np.array([deg_params_range[0][1] - deg_params_range[0][0], deg_params_range[0][3] - deg_params_range[0][2], deg_params_range[0][5] - deg_params_range[0][4], deg_params_range[0][7] - deg_params_range[0][6], deg_params_range[0][9] - deg_params_range[0][8]]), np.random.uniform(0, 1, ndim)) for i in
          range(nwalkers)]

    savename = dir_savefig + "/MCMC_opt_" + str(
        rxn_method) + "_" + str(tpe) + "_" + str(int(1000 * V_limit)) + "mV_" + str(N) + "N_" + str(
        str_deg_params) + ".png"
    savename_autocorr = dir_savefig + "/Autocorr_" + str(
        rxn_method) + "_" + str(tpe) + "_" + str(int(1000 * V_limit)) + "mV_" + str(N) + "N_" + str(
        str_deg_params) + ".png"

    c_a = params_a["c0"] - params_c["p"] / params_a["p"] * (c_c - params_c["c0"])
    pulse_range = get_muR_from_OCV(delta_V, 0)
    voltage_range = -Tesla_NCA_Si(c_c, params_c["muR_ref"]) + Tesla_graphite(c_a, params_a["muR_ref"]) + pulse_range
    # solve for the initial voltage pulses
    y = W(deg_params_true, c_c, c_a, params_c, params_a, voltage_range)

    data = (c_c, c_a, params_c, params_a, y, I_err, pulse_range, deg_params_range)

    if __name__ == '__main__':
        pool = multiprocessing.Pool()

        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=data, pool=pool)

        print("Running burn-in...")
        p0_burn, _, _ = sampler.run_mcmc(p0, 100, progress=True)
        sampler.reset()

        print("Running production...")
        sampler.run_mcmc(p0_burn, niter, progress=True)
        pool.close()

        print("Mean acceptance fraction: {0:.3f}".format(np.mean(sampler.acceptance_fraction)))
        chain = sampler.get_chain()[:, :, 0].T

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

        samples = sampler.flatchain
        results[ii] = {}
        results[ii]['True_params'] = deg_params_true
        results[ii]['Estimated_params'] = np.percentile(samples, 50, axis=0)
        results[ii]['lb_95'] = np.percentile(samples, 2.5, axis=0)
        results[ii]['ub_95'] = np.percentile(samples, 97.5, axis=0)
        results[ii]['lb_90'] = np.percentile(samples, 5, axis=0)
        results[ii]['ub_90'] = np.percentile(samples, 95, axis=0)
        results[ii]['c_c'] = c_c
        results[ii]['delta_V'] = delta_V
        labels = ['R_f_c', 'c_tilde_c', 'R_f_a', 'c_tilde_a', 'c_lyte']
        fig = corner.corner(samples, show_titles=True, labels=labels, plot_datapoints=True,
                            quantiles=[0.05, 0.5, 0.95],
                            title_fmt='.3f', range=[(deg_params_range[0][0], deg_params_range[0][1]), (deg_params_range[0][2], deg_params_range[0][3]), (deg_params_range[0][4], deg_params_range[0][5]), (deg_params_range[0][6], deg_params_range[0][7]), (deg_params_range[0][8], deg_params_range[0][9])],
                            plot_contours=True)
        fig_title = 'True values'
        for k in range(len(deg_params_true)):
            fig_title = fig_title + '\n' + "{0:<10} = {1: .3f}".format(labels[k], deg_params_true[k])
        fig_title = fig_title + '\n\n\nPulses'
        for k in range(N):
            fig_title = fig_title + '\n' + "#{0:<2} = ({1:.3f}, {2} mV)".format(int(k + 1), c_c[k], round(
                delta_V[
                    k] * 1000))  # ' + str(k+1) + ' =\t(' + str(c_c[k]) + ',\t' + str(round(delta_V[k]*1000)) + ' mV)'
        fig.suptitle(fig_title, x=0.7, fontsize=15, ha='left')

        # Extract the axes
        axes = np.array(fig.axes).reshape((ndim, ndim))

        # Loop over the diagonal
        for i in range(ndim):
            ax = axes[i, i]
            ax.axvline(deg_params_true[i], color="r")

        # Loop over the histograms
        for yi in range(ndim):
            for xi in range(yi):
                ax = axes[yi, xi]
                ax.axvline(deg_params_true[xi], color="r")
                ax.axhline(deg_params_true[yi], color="r")
                ax.plot(deg_params_true[xi], deg_params_true[yi], "sr")
        if saveplot:
            plt.savefig(savename, dpi=300, bbox_inches='tight')
        plt.show()
        sampler.reset()
        ii = ii + 1
'''
os.chdir(dir_txtfile)

for file in glob.glob("MCMC_opt_CIET_D_50mV_*.png"):
    if "MCMC_opt_CIET_D_50mV" in file:

        numpulse = 10  # Number of pulses.
        saveplot = True

        warnings.filterwarnings("ignore")

        os.environ["OMP_NUM_THREADS"] = "1"
        nwalkers = 24
        niter = 10000

        plt.rcParams['figure.figsize'] = (20, 10)
        V_limit = 0.050

        # particle size
        r_c = 11e-6
        r_a = 17e-6
        # lengths
        L_c = 64e-6
        L_a = 83e-6
        # Volume loading percents of active material (volume fraction of solid
        # that is active material)
        P_L_c = 0.7452
        P_L_a = 0.8277
        # Porosities (liquid volume fraction in each region)
        poros_c = 0.2298
        poros_a = 0.1473
        # width and thickness of electrodes
        width_a = 65e-3
        width_c = 63e-3
        thick_a = 86.5e-2
        thick_c = 85.9e-2

        # now call optimization funciton
        # site density of electrode active materials (sites/m^3)
        rho_s_c = 3.276e28
        rho_s_a = 1.7438e28

        # initial concentration of electrodes
        c_s_0_a = 0.833
        c_s_0_c = 0.167

        # rescaling factor to convert particle level current to electrode level
        f_c = L_c * (1 - poros_c) * P_L_c * 1 / r_c
        f_a = L_a * (1 - poros_a) * P_L_a * 1 / r_a

        # rescaling factor to balance electrode concentrations
        p_c = L_c * (1 - poros_c) * P_L_c * rho_s_c
        p_a = L_a * (1 - poros_a) * P_L_a * rho_s_a

        # set reference chemical ptoentials
        muR_ref_c = -Tesla_NCA_Si(np.array([c_s_0_c]), 0)[0]
        muR_ref_a = -Tesla_graphite(np.array([c_s_0_a]), 0)[0]
        params_c = {'rxn_method': rxn_method, 'k0': 1, 'lambda': 5, 'f': f_c, 'p': p_c, 'c0': c_s_0_c,
                    'mu': Tesla_NCA_Si,
                    'muR_ref': muR_ref_c}
        params_a = {'rxn_method': rxn_method, 'k0': 1, 'lambda': 5, 'f': f_a, 'p': p_a, 'c0': c_s_0_a,
                    'mu': Tesla_graphite,
                    'muR_ref': muR_ref_a}

        items = file.split('_')
        R_f_c = float(items[6])
        c_tilde_c = float(items[7])
        R_f_a = float(items[8])
        c_tilde_a = float(items[9])
        c_lyte = float(items[10].replace('.png', ''))

        deg_params_true = np.array([R_f_c, c_tilde_c, R_f_a, c_tilde_a, c_lyte])

        # TODO: Check I_err value of the experimental equipment [A/m^2]
        # TODO: Check with Debbie on err_y expression... R_value has [A/m^2]. Should I_err also have [A/m^2]?
        str_deg_params = str(deg_params_true[0]) + "_" + str(deg_params_true[1]) + "_" + str(
            deg_params_true[2]) + "_" + str(deg_params_true[3]) + "_" + str(deg_params_true[4])
        with open(dir_txtfile + "/optimized_output_" + str(
                rxn_method) + "_" + str(tpe) + "_" + str(int(1000 * V_limit)) + "mV_" + str(
                str_deg_params) + ".txt") as f:
            contents = f.readlines()

        N = 10
        #N_max = int(float(contents[-1].split()[0]))

        c_c = np.array([0.8, 0.8, 0.7, 0.7, 0.6, 0.6, 0.5, 0.5, 0.4, 0.4])
        delta_V = np.array([0.2, -0.2, 0.2, -0.2, 0.2, -0.2, 0.2, -0.2, 0.2, -0.2])

        # Spread walkers
        initial = np.array([deg_params_range[0][0], deg_params_range[0][2], deg_params_range[0][4], deg_params_range[0][6], deg_params_range[0][8]])
        ndim = len(initial)
        np.random.seed(0)
        p0 = [np.array(initial) + np.multiply(np.array([deg_params_range[0][1] - deg_params_range[0][0], deg_params_range[0][3] - deg_params_range[0][2], deg_params_range[0][5] - deg_params_range[0][4], deg_params_range[0][7] - deg_params_range[0][6], deg_params_range[0][9] - deg_params_range[0][8]]), np.random.uniform(0, 1, ndim)) for i in
              range(nwalkers)]

        savename = dir_savefig + "/MCMC_opt_" + str(
            rxn_method) + "_" + str(tpe) + "_" + str(int(1000 * V_limit)) + "mV_" + str(N) + "N_" + str(
            str_deg_params) + ".png"
        savename_autocorr = dir_savefig + "/Autocorr_" + str(
            rxn_method) + "_" + str(tpe) + "_" + str(int(1000 * V_limit)) + "mV_" + str(N) + "N_" + str(
            str_deg_params) + ".png"

        c_a = params_a["c0"] - params_c["p"] / params_a["p"] * (c_c - params_c["c0"])
        pulse_range = get_muR_from_OCV(delta_V, 0)
        voltage_range = -Tesla_NCA_Si(c_c, params_c["muR_ref"]) + Tesla_graphite(c_a, params_a["muR_ref"]) + pulse_range
        # solve for the initial voltage pulses
        y = W(deg_params_true, c_c, c_a, params_c, params_a, voltage_range)

        data = (c_c, c_a, params_c, params_a, y, I_err, pulse_range, deg_params_range)

        if __name__ == '__main__':
            pool = multiprocessing.Pool()

            sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=data, pool=pool)

            print("Running burn-in...")
            p0_burn, _, _ = sampler.run_mcmc(p0, 100, progress=True)
            sampler.reset()

            print("Running production...")
            sampler.run_mcmc(p0_burn, niter, progress=True)
            pool.close()

            print("Mean acceptance fraction: {0:.3f}".format(np.mean(sampler.acceptance_fraction)))
            chain = sampler.get_chain()[:, :, 0].T

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

            samples = sampler.flatchain
            results[ii] = {}
            results[ii]['True_params'] = deg_params_true
            results[ii]['Estimated_params'] = np.percentile(samples, 50, axis=0)
            results[ii]['lb_95'] = np.percentile(samples, 2.5, axis=0)
            results[ii]['ub_95'] = np.percentile(samples, 97.5, axis=0)
            results[ii]['lb_90'] = np.percentile(samples, 5, axis=0)
            results[ii]['ub_90'] = np.percentile(samples, 95, axis=0)
            results[ii]['c_c'] = c_c
            results[ii]['delta_V'] = delta_V
            labels = ['R_f_c', 'c_tilde_c', 'R_f_a', 'c_tilde_a', 'c_lyte']
            fig = corner.corner(samples, show_titles=True, labels=labels, plot_datapoints=True,
                                quantiles=[0.05, 0.5, 0.95],
                                title_fmt='.3f', range=[(deg_params_range[0][0], deg_params_range[0][1]), (deg_params_range[0][2], deg_params_range[0][3]), (deg_params_range[0][4], deg_params_range[0][5]), (deg_params_range[0][6], deg_params_range[0][7]), (deg_params_range[0][8], deg_params_range[0][9])],
                                plot_contours=True)
            fig_title = 'True values'
            for k in range(len(deg_params_true)):
                fig_title = fig_title + '\n' + "{0:<10} = {1: .3f}".format(labels[k], deg_params_true[k])
            fig_title = fig_title + '\n\n\nPulses'
            for k in range(N):
                fig_title = fig_title + '\n' + "#{0:<2} = ({1:.3f}, {2} mV)".format(int(k + 1), c_c[k], round(
                    delta_V[
                        k] * 1000))  # ' + str(k+1) + ' =\t(' + str(c_c[k]) + ',\t' + str(round(delta_V[k]*1000)) + ' mV)'
            fig.suptitle(fig_title, x=0.7, fontsize=15, ha='left')

            # Extract the axes
            axes = np.array(fig.axes).reshape((ndim, ndim))

            # Loop over the diagonal
            for i in range(ndim):
                ax = axes[i, i]
                ax.axvline(deg_params_true[i], color="r")

            # Loop over the histograms
            for yi in range(ndim):
                for xi in range(yi):
                    ax = axes[yi, xi] 
                    ax.axvline(deg_params_true[xi], color="r")
                    ax.axhline(deg_params_true[yi], color="r")
                    ax.plot(deg_params_true[xi], deg_params_true[yi], "sr")
            if saveplot:
                plt.savefig(savename, dpi=300, bbox_inches='tight')
            plt.show()
            sampler.reset()
            ii = ii + 1
'''
with open("C:/Users/Jinwook/PyCharm_projects/DOE/standard_N5_similar_" + str_deg_params_range + ".pkl", 'wb') as f:
    pickle.dump(results, f)