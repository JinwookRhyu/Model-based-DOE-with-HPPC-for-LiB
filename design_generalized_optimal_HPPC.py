import numpy as np
import os
import pygmo as pg
import time
from src import (Tesla_NCA_Si, Tesla_graphite, f_time, f_uncertainty_fixtlast, f_uncertainty_fixtlast_multi,
                    uncertainty_function_fixtlast, uncertainty_function_fixtlast_multi)

start_time = time.time()

is_balanced_list = ["unbalanced", "unbalanced", "unbalanced", "unbalanced", "unbalanced", "unbalanced", "unbalanced", "unbalanced", "unbalanced", "unbalanced", "unbalanced", "unbalanced"] # False when using actual k0 values. True for hypothetical case where k0's are set to intentionally match the overpotential balance
is_initial_high_list = [True, False, True, False, True, False, True, False, True, False, True, False] # True when the HPPC protocol starts from high cathode filling fraction (i.e., low voltage). Otherwise False
N_list = [5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10] # Number of pulses
two_objective = True # True when solving two-objective optimization problem. False when solving one-objective (D-optimality) problem with using the total diagnostic time as a constraint
gp_for_surrogate = False # True when using GP surrogate model for estimating the averaged first objective function (i.e., D-optimality) across degradation parameter space

#load NCA/graphite diffusivities
diffNCA = 10**np.loadtxt('amin_diffusion/NCA_diffusion.txt', delimiter = ',')
diffgraphite = np.loadtxt('diffusion_carelli_et_all/graphite_diffusion.txt', delimiter = ',')

# LiB parameters
# Particle size
r_c_primary = 0.46e-6
r_a_primary = 0.5e-6
r_c_secondary = 11e-6
r_a_secondary = 16e-6
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

V_limit_high = 0.200 # Upper limit for the voltage pulse in [V]
V_limit_low = 0.050  # Lower limit for the voltage pulse in [V]
c_c_limit_high = 0.8 # Upper limit for the cathode filling fraction
c_c_limit_low = 0.4  # Lower limit for the cathode filling fractio
tpe = "A"            # Optimality criterion "A" / "D" / "E"
rxn_method = "CIET"  # 'CIET' for coupled ion electron transfer model. 'BV' for Butler-Volmer

# OCV functional forms for cathode and anode
mu_c = Tesla_NCA_Si
mu_a = Tesla_graphite
muR_ref_c = -mu_c(y=np.array([c_s_0_c]), muR_ref=0)[0]
muR_ref_a = -mu_a(y=np.array([c_s_0_a]), muR_ref=0)[0]

# Lower and upper limits for degradation parameters in R_f_c, c_tilde_c, R_f_a, c_tilde_a, c_lyte order
R_f_c_range = np.array([0, 0.2])      # Range for R_f_c (lb, ub)
c_tilde_c_range = np.array([0.8, 1]) # Range for c_tilde_c (lb, ub)
R_f_a_range = np.array([0, 0.2])      # Range for R_f_a (lb, ub)
c_tilde_a_range = np.array([0.8, 1]) # Range for c_tilde_a (lb, ub)
c_lyte_range = np.array([0.8, 1])    # Range for c_lyte (lb, ub)

t_pulse = 5 # pulse time in seconds
alpha_t = 1 # Coeff for CC time + relaxation time
# Time limit for total diagnostics. Needed only when two_objective = False
t_limit_list = 3 * np.array([20, 19, 18, 17, 16, 15, 14.5, 14, 13.5, 13, 12.5, 12, 11.5, 11, 10.5, 10, 9.5, 9, 8.5, 8, 7.5, 7, 6.5, 6, 5.5, 5, 4.5, 4, 3.5, 3, 2.5, 2, 1.5, 1, 0.5])

deg_params_bound = np.vstack([R_f_c_range, c_tilde_c_range, R_f_a_range, c_tilde_a_range, c_lyte_range])

# Rescaling factor to convert particle level current to electrode level
f_c = L_c * (1 - poros_c) * P_L_c * 3 / r_c_primary
f_a = L_a * (1 - poros_a) * P_L_a * 3 / r_a_primary

# Rescaling factor to balance electrode concentrations
p_c = L_c * (1 - poros_c) * P_L_c * rho_s_c
p_a = L_a * (1 - poros_a) * P_L_a * rho_s_a

str_deg_params = str(deg_params_bound[0][0]) + "_" + str(deg_params_bound[0][1]) + "_" + str(deg_params_bound[1][0]) + "_" + str(deg_params_bound[1][1]) + "_" + str(deg_params_bound[2][0]) + "_" + str(deg_params_bound[2][1]) + "_" + str(deg_params_bound[3][0]) + "_" + str(deg_params_bound[3][1]) + "_" + str(deg_params_bound[4][0]) + "_" + str(deg_params_bound[4][1])

for n in range(len(N_list)):

    is_balanced = is_balanced_list[n]
    is_initial_high = is_initial_high_list[n]
    N = int(N_list[n])
    dim = 2 * N - 1 # Number of pulse parameters for characterizing the HPPC protocol with N pulses

    # Set directory for saving the results
    if two_objective:
        if is_balanced == "balanced":
            if is_initial_high:
                savedir = os.path.join(os.getcwd(), f"pareto_{tpe}_N{N}_balanced_high_multi")
            else:
                savedir = os.path.join(os.getcwd(), f"pareto_{tpe}_N{N}_balanced_low_multi")
        elif is_balanced == "midbalanced":
            if is_initial_high:
                savedir = os.path.join(os.getcwd(), f"pareto_{tpe}_N{N}_midbalanced_high_multi")
            else:
                savedir = os.path.join(os.getcwd(), f"pareto_{tpe}_N{N}_midbalanced_low_multi")
        elif is_balanced == "unbalanced":
            if is_initial_high:
                savedir = os.path.join(os.getcwd(), f"pareto_{tpe}_N{N}_unbalanced_high_multi")
            else:
                savedir = os.path.join(os.getcwd(), f"pareto_{tpe}_N{N}_unbalanced_low_multi")
    else:
        if is_balanced == "balanced":
            if is_initial_high:
                savedir = os.path.join(os.getcwd(), f"pareto_{tpe}_N{N}_balanced_high")
            else:
                savedir = os.path.join(os.getcwd(), f"pareto_{tpe}_N{N}_balanced_low")
        elif is_balanced == "midbalanced":
            if is_initial_high:
                savedir = os.path.join(os.getcwd(), f"pareto_{tpe}_N{N}_midbalanced_high")
            else:
                savedir = os.path.join(os.getcwd(), f"pareto_{tpe}_N{N}_midbalanced_low")
        elif is_balanced == "unbalanced":
            if is_initial_high:
                savedir = os.path.join(os.getcwd(), f"pareto_{tpe}_N{N}_unbalanced_high")
            else:
                savedir = os.path.join(os.getcwd(), f"pareto_{tpe}_N{N}_unbalanced_low")

    if not os.path.exists(savedir):
        os.mkdir(savedir)


    # Parameters for electrodes
    if is_balanced == "balanced": # Hypothetical case where k0 is set based on the scaling analysis to achieve overpotential balance
        params_c = {'rxn_method': rxn_method, 'k0': np.sqrt(f_a/f_c), 'lambda': 5, 'f': f_c, 'p': p_c, 'c0': c_s_0_c, 'mu': mu_c,
                'muR_ref': muR_ref_c, 'diff': diffNCA, 'particle_size': r_c_secondary, 't_pulse': t_pulse}
        params_a = {'rxn_method': rxn_method, 'k0': np.sqrt(f_c/f_a), 'lambda': 8, 'f': f_a, 'p': p_a, 'c0': c_s_0_a,
                    'mu': mu_a, 'muR_ref': muR_ref_a, 'diff': diffgraphite, 'particle_size': r_a_secondary}
    elif is_balanced == "unbalanced": # Using actual electrode parameters
        params_c = {'rxn_method': rxn_method, 'k0': 18, 'lambda': 5, 'f': f_c, 'p': p_c, 'c0': c_s_0_c,
                    'mu': mu_c, 'muR_ref': muR_ref_c, 'diff': diffNCA, 'particle_size': r_c_secondary, 't_pulse': t_pulse}
        params_a = {'rxn_method': rxn_method, 'k0': 0.2, 'lambda': 8, 'f': f_a, 'p': p_a, 'c0': c_s_0_a,
                    'mu': mu_a, 'muR_ref': muR_ref_a, 'diff': diffgraphite, 'particle_size': r_a_secondary}

    if two_objective: # Solve optimization problem for two objective functions simultaneously
        prob = pg.problem(
            uncertainty_function_fixtlast_multi(N=N, is_initial_high=is_initial_high, params_a=params_a, params_c=params_c, mu_a=mu_a, mu_c=mu_c, deg_params_bound=deg_params_bound, alpha_t=alpha_t,
                                                V_limit_high=V_limit_high, V_limit_low=V_limit_low, c_c_limit_high=c_c_limit_high, c_c_limit_low=c_c_limit_low, t_pulse=t_pulse, Phi_func_option=tpe, gp_for_surrogate=gp_for_surrogate))

        pop_size = 40 * dim    # number of samples per each generation
        total_gens = 10 * dim  # total generations you want to run

        pop = pg.population(prob=prob, size=pop_size, seed=42)

        algo = pg.algorithm(pg.nsga2(gen=total_gens, seed=42))
        algo.set_verbosity(1)  # will print once per 'gens_per_step'
        pop = algo.evolve(pop)

        # --- collect final nondominated set ---
        F_final = pop.get_f()
        X_final = pop.get_x()

        # --- optional: extract Pareto front only ---
        fronts = pg.fast_non_dominated_sorting(points=F_final)[0]
        pareto_idx = fronts[0]  # first front indices
        F_pareto = F_final[pareto_idx]
        X_pareto = X_final[pareto_idx]
        n_archive = X_pareto.shape[0]

        # saves SOC and voltage values for each pulse
        optimization_save = np.zeros((n_archive, 2 * N)) * np.nan
        # saves rest time in between pulses
        time_save = np.ones((n_archive, N)) * np.nan
        # saves f_uncertainty and f_time
        pareto_save = np.ones((n_archive, 2)) * np.nan

        for i, out in enumerate(X_pareto):
            # Extract cathode filling fraction at each pulse from the optimization result "out"
            if is_initial_high:
                c_range = c_c_limit_high + np.cumsum(np.insert(out[:N - 1], 0, 0)) / (np.sum(out[:N])) * (
                            c_c_limit_low - c_c_limit_high)
            else:
                c_range = c_c_limit_low + np.cumsum(np.insert(out[:N - 1], 0, 0)) / (np.sum(out[:N])) * (
                            c_c_limit_high - c_c_limit_low)
            c_range = np.round(c_range * 1000) / 1000

            # Extract voltage magnitude at each pulse from the optimization result "out"
            dV = np.zeros(N)
            mask1 = out[-N:] >= 0.5
            dV[mask1] = (V_limit_high - V_limit_low) * 2 * (out[-N:][mask1] - 0.5) + V_limit_low
            mask2 = out[-N:] < 0.5
            dV[mask2] = - ((V_limit_high - V_limit_low) * 2 * (0.5 - out[-N:][mask2]) + V_limit_low)
            dV = np.round(dV * 1000) / 1000

            # Calculate D-optimality objective function from the optimization result "out"
            J1, R_value = f_uncertainty_fixtlast_multi(opt_params=out, N=N, deg_params_bound=deg_params_bound, params_c=params_c, params_a=params_a, mu_c=mu_c,
                                                 mu_a=mu_a, is_initial_high=is_initial_high, V_limit_high=V_limit_high, V_limit_low=V_limit_low, c_c_limit_high=c_c_limit_high, c_c_limit_low=c_c_limit_low, Phi_func_option=tpe)
            # Calculate diagnostic time objective function from the optimization result "out"
            c_c = c_range
            optimization_save[i, :N] = c_range
            optimization_save[i, N:2 * N] = dV
            if is_initial_high:
                c_c_t = np.concatenate((np.array([c_c_limit_high]), c_range))
            else:
                c_c_t = np.concatenate((np.array([c_c_limit_low]), c_range))
            c_a = params_a["c0"] - params_c["p"] / params_a["p"] * (c_c - params_c["c0"])
            c_a_t = params_a["c0"] - params_c["p"] / params_a["p"] * (c_c_t - params_c["c0"])
            time_save[i, :N] = f_time(alpha_t, c_c_t[:-1], c_c_t[1:], c_a_t[:-1], c_a_t[1:], params_c, params_a,
                                      R_value, t_pulse)
            print("N: ", N, "   Optimum Parameters: c: ", c_range, "; dV: ", dV, "; value: ", J1, "; relaxation time (hr): ", time_save[i, :N])
            J2 = np.sum(time_save[i, :N])
            print("f_uncertainty = ", J1, "     f_time = ", J2)
            pareto_save[i, :] = np.array([J1, J2])

        # Save the optimization results
        np.savetxt(os.path.join(savedir, f"optimized_output_{params_c['rxn_method']}_{tpe}_{int(1000 * V_limit_high)}mV_{int(1000 * V_limit_low)}mV_{str_deg_params}.txt"), optimization_save)
        np.savetxt(os.path.join(savedir, f"time_{params_c['rxn_method']}_{tpe}_{int(1000 * V_limit_high)}mV_{int(1000 * V_limit_low)}mV_{str_deg_params}.txt"), time_save)
        np.savetxt(os.path.join(savedir, f"pareto_{params_c['rxn_method']}_{tpe}_{int(1000 * V_limit_high)}mV_{int(1000 * V_limit_low)}mV_{str_deg_params}.txt"), pareto_save)

    else: # Solve optimization problem for D-optimality while using total diagnostic time as a constraint
        # First we determine t_last_0 (We fix t_last = 0 for f_uncertainty_fixtlast. Details can be found in J. Rhyu et al. Journal of The Electrochemical Society 171 (7), 070544)
        prob = pg.problem(uncertainty_function_fixtlast(N=N, t_last=0, idx=0, is_initial_high=is_initial_high, params_a=params_a,
                                                        params_c=params_c, mu_a=mu_a, mu_c=mu_c, deg_params_bound=deg_params_bound, alpha_t=alpha_t,
                                                        t_limit_list=t_limit_list, V_limit_high=V_limit_high, V_limit_low=V_limit_low,
                                                        c_c_limit_high=c_c_limit_high, c_c_limit_low=c_c_limit_low, t_pulse=t_pulse, Phi_func_option=tpe, gp_for_surrogate=gp_for_surrogate))
        n_iter = 20 * dim
        pop_size = dim
        algo = pg.algorithm(pg.ihs(n_iter, seed=42))

        algo.set_verbosity(10)
        pop = pg.population(prob=prob, size=pop_size, seed=42)
        pop.problem.c_tol = [1e-8] * 1
        pop = algo.evolve(pop)
        out = pop.champion_x
        print("-----------Champion------------")
        f_uncertainty_fixtlast(opt_params=out, N=N, deg_params_bound=deg_params_bound, params_c=params_c, params_a=params_a, mu_c=mu_c, mu_a=mu_a, t_last=0,
                               is_initial_high=is_initial_high, V_limit_high=V_limit_high, V_limit_low=V_limit_low, c_c_limit_high=c_c_limit_high, c_c_limit_low=c_c_limit_low, Phi_func_option=tpe, verbose=True)
        if is_initial_high:
            c_range = c_c_limit_high + np.cumsum(np.insert(out[:N-1], 0, 0)) / (np.sum(np.insert(out[:N-1], 0, 0)) + 0) * (c_c_limit_low - c_c_limit_high)
        else:
            c_range = c_c_limit_low + np.cumsum(np.insert(out[:N-1], 0, 0)) / (np.sum(np.insert(out[:N-1], 0, 0)) + 0) * (c_c_limit_high - c_c_limit_low)
        c_range = np.round(c_range * 1000) / 1000

        dV = np.zeros(N)
        mask1 = out[-N:] >= 0.5
        dV[mask1] = (V_limit_high - V_limit_low) * 2 * (out[-N:][mask1] - 0.5) + V_limit_low
        mask2 = out[-N:] < 0.5
        dV[mask2] = - ((V_limit_high - V_limit_low) * 2 * (0.5 - out[-N:][mask2]) + V_limit_low)
        dV = np.round(dV * 1000) / 1000

        J1, R_value = f_uncertainty_fixtlast(opt_params=out, N=N, deg_params_bound=deg_params_bound, params_c=params_c, params_a=params_a, mu_c=mu_c, mu_a=mu_a,
                                             t_last=0, is_initial_high=is_initial_high, V_limit_high=V_limit_high, V_limit_low=V_limit_low, c_c_limit_high=c_c_limit_high, c_c_limit_low=c_c_limit_low, Phi_func_option=tpe)
        c_c = c_range
        if is_initial_high:
            c_c_t = np.concatenate((np.array([c_c_limit_high]), c_range))
        else:
            c_c_t = np.concatenate((np.array([c_c_limit_low]), c_range))
        c_a = params_a["c0"] - params_c["p"] / params_a["p"] * (c_c - params_c["c0"])
        c_a_t = params_a["c0"] - params_c["p"] / params_a["p"] * (c_c_t - params_c["c0"])
        t_last_0 = np.sum(f_time(alpha_t, c_c_t[:-1], c_c_t[1:], c_a_t[:-1], c_a_t[1:], params_c, params_a, R_value, t_pulse))
        print("t_last_0 = ", t_last_0)
        c_range_last_0 = c_range
        dV_last_0 = dV
        time_last_0 = f_time(alpha_t=alpha_t, c_min_c=c_c_t[:-1], c_max_c=c_c_t[1:], c_min_a=c_a_t[:-1], c_max_a=c_a_t[1:], params_c=params_c, params_a=params_a, R_value=R_value, t_pulse=t_pulse)
        pareto_last_0 = np.array([J1, t_last_0])

        # saves SOC and voltage values for each pulse
        optimization_save = np.zeros((len(t_limit_list), 2 * N)) * np.nan
        # saves rest time in between pulses
        time_save = np.ones((len(t_limit_list), N)) * np.nan
        # saves f_uncertainty and f_time
        pareto_save = np.ones((len(t_limit_list), 2)) * np.nan

        # Now, use the determined t_last_0 for the actual run (Details can be found in J. Rhyu et al. Journal of The Electrochemical Society 171 (7), 070544)
        for i in range(len(t_limit_list)):
            if t_limit_list[i] > t_last_0: # Skip when time constraint is not activated
                optimization_save[i, :N] = c_range_last_0
                optimization_save[i, N:2 * N] = dV_last_0
                time_save[i, :N] = time_last_0
                pareto_save[i, :] = pareto_last_0
            else: # When optimization problem is updated with tighter time constraint
                t_last = (t_last_0 - t_limit_list[i]) / (t_last_0) * 0.2 * N
                prob = pg.problem(
                    uncertainty_function_fixtlast(N=N, t_last=t_last, idx=i, is_initial_high=is_initial_high, params_a=params_a, params_c=params_c, mu_a=mu_a, mu_c=mu_c, deg_params_bound=deg_params_bound,
                                                  alpha_t=alpha_t, t_limit_list=t_limit_list, V_limit_high=V_limit_high, V_limit_low=V_limit_low, c_c_limit_high=c_c_limit_high,
                                                  c_c_limit_low=c_c_limit_low, t_pulse=t_pulse, Phi_func_option=tpe, gp_for_surrogate=gp_for_surrogate))
                algo = pg.algorithm(pg.ihs(n_iter, seed=42))
                algo.set_verbosity(10)
                pop = pg.population(prob=prob, size=pop_size, seed=42)
                pop.problem.c_tol = [1e-8] * 1
                pop = algo.evolve(pop)
                out = pop.champion_x
                f_uncertainty_fixtlast(opt_params=out, N=N, deg_params_bound=deg_params_bound, params_c=params_c, params_a=params_a, mu_c=mu_c, mu_a=mu_a, t_last=t_last,
                                       is_initial_high=is_initial_high, V_limit_high=V_limit_high, V_limit_low=V_limit_low, c_c_limit_high=c_c_limit_high, c_c_limit_low=c_c_limit_low, Phi_func_option=tpe, verbose=True)

                if is_initial_high:
                    c_range = c_c_limit_high + np.cumsum(np.insert(out[:N-1], 0, 0)) / (np.sum(np.insert(out[:N-1], 0, 0))+t_last) * (c_c_limit_low - c_c_limit_high)
                else:
                    c_range = c_c_limit_low + np.cumsum(np.insert(out[:N-1], 0, 0)) / (np.sum(np.insert(out[:N-1], 0, 0))+t_last) * (c_c_limit_high - c_c_limit_low)

                c_range = np.round(c_range * 1000) / 1000
                optimization_save[i, :N] = c_range
                dV = np.zeros(N)
                mask1 = out[-N:] >= 0.5
                dV[mask1] = (V_limit_high - V_limit_low) * 2 * (out[-N:][mask1] - 0.5) + V_limit_low
                mask2 = out[-N:] < 0.5
                dV[mask2] = - ((V_limit_high - V_limit_low) * 2 * (0.5 - out[-N:][mask2]) + V_limit_low)
                dV = np.round(dV * 1000) / 1000
                optimization_save[i, N:2 * N] = dV
                J1, R_value = f_uncertainty_fixtlast(opt_params=out, N=N, deg_params_bound=deg_params_bound, params_c=params_c, params_a=params_a, mu_c=mu_c, mu_a=mu_a,
                                                     t_last=t_last, is_initial_high=is_initial_high, V_limit_high=V_limit_high, V_limit_low=V_limit_low, c_c_limit_high=c_c_limit_high, c_c_limit_low=c_c_limit_low, Phi_func_option=tpe)
                c_c = c_range
                if is_initial_high:
                    c_c_t = np.concatenate((np.array([c_c_limit_high]), c_range))
                else:
                    c_c_t = np.concatenate((np.array([c_c_limit_low]), c_range))
                c_a = params_a["c0"] - params_c["p"] / params_a["p"] * (c_c - params_c["c0"])
                c_a_t = params_a["c0"] - params_c["p"] / params_a["p"] * (c_c_t - params_c["c0"])
                time_save[i, :N] = f_time(alpha_t=alpha_t, c_min_c=c_c_t[:-1], c_max_c=c_c_t[1:], c_min_a=c_a_t[:-1], c_max_a=c_a_t[1:], params_c=params_c, params_a=params_a, R_value=R_value, t_pulse=t_pulse)
                print("N: ", N, "   Optimum Parameters: c: ", c_range, "; dV: ", dV, "; value: ", J1, "; relaxation time (hr): ", time_save[i, :N])
                J2 = np.sum(time_save[i, :N])
                print("f_uncertainty = ", J1, "     f_time = ", J2)
                pareto_save[i, :] = np.array([J1, J2])

        np.savetxt(os.path.join(savedir, f"optimized_output_{params_c['rxn_method']}_{tpe}_{int(1000 * V_limit_high)}mV_{int(1000 * V_limit_low)}mV_{str_deg_params}.txt"), optimization_save)
        np.savetxt(os.path.join(savedir, f"time_{params_c['rxn_method']}_{tpe}_{int(1000 * V_limit_high)}mV_{int(1000 * V_limit_low)}mV_{str_deg_params}.txt"), time_save)
        np.savetxt(os.path.join(savedir, f"pareto_{params_c['rxn_method']}_{tpe}_{int(1000 * V_limit_high)}mV_{int(1000 * V_limit_low)}mV_{str_deg_params}.txt"), pareto_save)
