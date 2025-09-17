import numpy as np
from MCMC_autocorrelation_help import (Tesla_NCA_Si, Tesla_graphite, f_uncertainty_fixtlast_multi, f_time)

is_balanced = False    # False when using actual k0 values. True for hypothetical case where k0's are set to intentionally match the overpotential balance

option = 6
Phi_func_option = "A"

if option == 0:
    c_c = np.array([0.8, 0.8, 0.7, 0.7, 0.6, 0.6, 0.5, 0.5, 0.4, 0.4])     # List of cathode filling fractions at each voltage pulse
    dV = np.array([0.2, -0.2, 0.2, -0.2, 0.2, -0.2, 0.2, -0.2, 0.2, -0.2])
elif option == 1:
    # Optimal1_high_A_N10
    c_c = np.array([0.8,   0.8,   0.8,   0.749, 0.749, 0.707, 0.572, 0.534, 0.534, 0.534])
    dV = np.array([-0.05, -0.2,  -0.05, -0.2,   0.05, -0.2,   0.05, -0.05,  0.2,  -0.05])
elif option == 2:
    # Optimal1_low_A_N10
    c_c = np.array([0.4,   0.418, 0.5,   0.561, 0.591, 0.597, 0.684, 0.741, 0.799, 0.8  ])
    dV = np.array([ 0.05,  -0.051, -0.051,  0.199,  0.05,   0.199, -0.2,   -0.2,   -0.2,   -0.05 ])
elif option == 3:
    # Optimal2_high_A_N10_0.1_0.2
    c_c = np.array([0.8,   0.8,   0.8,   0.731, 0.723, 0.723, 0.584, 0.536, 0.536, 0.536])
    dV = np.array([-0.05, -0.2,  -0.05, -0.2,  -0.2,   0.05,  0.05,  0.2,  -0.05, -0.05])
elif option == 4:
    # Optimal2_low_A_N10_0.1_0.2
    c_c = np.array([0.4,   0.401, 0.485, 0.554, 0.599, 0.681, 0.711, 0.735, 0.797, 0.797])
    dV = np.array([0.05,  -0.053, -0.05,   0.197,  0.2,   -0.2,    0.05,  -0.199, -0.05,  -0.199])
elif option == 5:
    # Optimal1_high_D_N10
    c_c = np.array([0.8, 0.8, 0.8, 0.67, 0.666, 0.4, 0.4, 0.4, 0.4, 0.4])
    dV = np.array([0.05, -0.05, -0.2, -0.2, -0.2, -0.05, 0.2, 0.2, 0.05, -0.05])
elif option == 6:
    # Optimal1_low_D_N10
    c_c = np.array([0.4, 0.4, 0.405, 0.405, 0.507, 0.68, 0.8, 0.8, 0.8, 0.8])
    dV = np.array([0.05, -0.05, -0.05, 0.2, 0.2, -0.2, 0.05, 0.05, -0.2, -0.05])
elif option == 7:
    # Optimal2_high_D_N10_0.1_0.2
    c_c = np.array([0.8,   0.8,   0.8,   0.678, 0.462, 0.403, 0.402, 0.401, 0.401, 0.401])
    dV = np.array([0.05, -0.05, -0.2,  -0.2,   0.2,   0.2,  -0.05,  0.05,  0.05, -0.05])
elif option == 8:
    # Optimal2_low_D_N10_0.1_0.2
    c_c = np.array([0.4, 0.401, 0.401, 0.42, 0.456, 0.661, 0.799, 0.799, 0.799, 0.8])
    dV = np.array([0.05, -0.05, -0.05, 0.199, 0.199, -0.2, 0.05, -0.2, -0.05, 0.05])


N = len(c_c)  # Number of pulses

V_limit_high = 0.200  # Upper limit for the voltage pulse in [V]
V_limit_low = 0.050   # Lower limit for the voltage pulse in [V]
c_c_limit_high = 0.8  # Upper limit for the cathode filling fraction
c_c_limit_low = 0.4   # Lower limit for the cathode filling fraction
rxn_method = "CIET"   # 'CIET' for coupled ion electron transfer model. 'BV' for Butler-Volmer
is_initial_high = True if np.abs(c_c[0] - c_c_limit_high) < 0.001 else False

# Lower and upper limits for degradation parameters in R_f_c, c_tilde_c, R_f_a, c_tilde_a, c_lyte order
R_f_c_range = np.array([0, 0.2])       # Range for R_f_c (lb, ub)
c_tilde_c_range = np.array([0.8, 1])  # Range for c_tilde_c (lb, ub)
R_f_a_range = np.array([0.1, 0.2])       # Range for R_f_a (lb, ub)
c_tilde_a_range = np.array([0.8, 1])  # Range for c_tilde_a (lb, ub)
c_lyte_range = np.array([0.8, 1])     # Range for c_lyte (lb, ub)

t_pulse = 5  # pulse time
alpha_t = 1  # Coeff for CC time + relaxation time

deg_params_bound = np.vstack([R_f_c_range, c_tilde_c_range, R_f_a_range, c_tilde_a_range, c_lyte_range])

# Load NCA/graphite diffusivity
diffNCA = 10**np.loadtxt('amin_diffusion/NCA_diffusion.txt', delimiter = ',')
diffgraphite = np.loadtxt('diffusion_carelli_et_all/graphite_diffusion.txt', delimiter=',')

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

# Rescaling factor to convert particle level current to electrode level
f_c = L_c * (1 - poros_c) * P_L_c * 3 / r_c_primary
f_a = L_a * (1 - poros_a) * P_L_a * 3 / r_a_primary

# Rescaling factor to balance electrode concentrations
p_c = L_c * (1 - poros_c) * P_L_c * rho_s_c
p_a = L_a * (1 - poros_a) * P_L_a * rho_s_a

# Set reference chemical potentials
mu_c = Tesla_NCA_Si
mu_a = Tesla_graphite
muR_ref_c = -mu_c(y=np.array([c_s_0_c]), muR_ref=0)[0]
muR_ref_a = -mu_a(y=np.array([c_s_0_a]), muR_ref=0)[0]

if is_balanced: # Hypothetical case where k0 is set based on the scaling analysis to achieve overpotential balance
    params_c = {'rxn_method': rxn_method, 'k0': np.sqrt(f_a/f_c), 'lambda': 5, 'f': f_c, 'p': p_c, 'c0': c_s_0_c, 'mu': mu_c,
            'muR_ref': muR_ref_c, 'diff': diffNCA, 'particle_size': r_c_secondary, 't_pulse': t_pulse}
    params_a = {'rxn_method': rxn_method, 'k0': np.sqrt(f_a/f_c), 'lambda': 8, 'f': f_a, 'p': p_a, 'c0': c_s_0_a,
                'mu': mu_a, 'muR_ref': muR_ref_a, 'diff': diffgraphite, 'particle_size': r_a_secondary}
else: # Using actual electrode parameters
    params_c = {'rxn_method': rxn_method, 'k0': 18, 'lambda': 5, 'f': f_c, 'p': p_c, 'c0': c_s_0_c,
                'mu': mu_c, 'muR_ref': muR_ref_c, 'diff': diffNCA, 'particle_size': r_c_secondary, 't_pulse': t_pulse}
    params_a = {'rxn_method': rxn_method, 'k0': 0.2, 'lambda': 8, 'f': f_a, 'p': p_a, 'c0': c_s_0_a,
                'mu': mu_a, 'muR_ref': muR_ref_a, 'diff': diffgraphite, 'particle_size': r_a_secondary}

# Calculate the first objective function: D_optimality of the uncertainties for identifying degradation parameters
J1, R_value = f_uncertainty_fixtlast_multi(opt_params=[], N=N, deg_params_bound=deg_params_bound, params_c=params_c,
                                           params_a=params_a, mu_c=mu_c, mu_a=mu_a, is_initial_high=is_initial_high,
                                           V_limit_high=V_limit_high, V_limit_low=V_limit_low, c_c_limit_high=c_c_limit_high,
                                           c_c_limit_low=c_c_limit_low, Phi_func_option=Phi_func_option, need_to_convert_opt = False, c_range = c_c, dV = dV)


c_c_t = np.insert(c_c, 0, c_c[0])
c_a = params_a["c0"] - params_c["p"] / params_a["p"] * (c_c - params_c["c0"])
c_a_t = params_a["c0"] - params_c["p"] / params_a["p"] * (c_c_t - params_c["c0"])

# Calculate the second objective function: Total diagnostic time of the given HPPC protocol
J2 = np.sum(f_time(alpha_t=alpha_t, c_min_c=c_c_t[:-1], c_max_c=c_c_t[1:], c_min_a=c_a_t[:-1], c_max_a=c_a_t[1:], params_c=params_c, params_a=params_a, R_value=R_value, t_pulse=t_pulse))
print("J1 = " + str(J1) + "\nJ2 = " + str(J2))


