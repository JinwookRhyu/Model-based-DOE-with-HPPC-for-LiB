import numpy as np
from scipy import constants
from scipy.special import erf
from scipy.optimize import fsolve
from pyDOE3 import *
import os

# import pygmo as pg
# from pygmo import *

is_balanced = False
is_initial_high = True

c_range = np.array([8.00E-01,7.41E-01,7.07E-01,6.53E-01,6.45E-01,5.87E-01,5.30E-01,4.72E-01,4.18E-01,4.17E-01])
V_values = np.array([-2.00E-01,-2.00E-01,-2.00E-01,1.99E-01,2.00E-01,2.00E-01,-2.00E-01,-2.00E-01,-2.00E-01,2.00E-01])

N = len(c_range)  # Number of pulses
num_meshpoints = 1000

V_limit = 0.050  # Lower limit for delta_V in [V]
I_err = 0.0003  # Measurement error of current in [A]
tpe = "D"
rxn_method = "CIET"
sig_digits = 4

# Lower and upper limits for degradation parameters in R_f_c, c_tilde_c, R_f_a, c_tilde_a, c_lyte order
R_f_c_range = np.array([0, 10])  # Range for R_f_c (lb, ub)
c_tilde_c_range = np.array([0.8, 1])  # Range for c_tilde_c (lb, ub)
R_f_a_range = np.array([0, 10])  # Range for R_f_a (lb, ub)
c_tilde_a_range = np.array([0.8, 1])  # Range for c_tilde_a (lb, ub)
c_lyte_range = np.array([0.8, 1])  # Range for c_lyte (lb, ub)

CC = 0.5  # C-rate for CC (dis)charge step
t_pulse = 5 # pulse time
alpha_t = 1  # Coeff for CC time + relaxation time

deg_params = np.reshape(
    np.concatenate((R_f_c_range, c_tilde_c_range, R_f_a_range, c_tilde_a_range, c_lyte_range), axis=0), (1, 10))


def W_hat(deg_params, c_c, c_a, V_c, V_a, params_c, params_a, mu_c, mu_a):
    """Defines W_hat value for half cell electrode:
    Inputs: c: state of charge
    V: voltage
    R_f: film resistance
    c_tilde: rescaled capacity
    c_lyte: electrolyte concentration
    params: parameters related to electrode
    mu_c: functional form of OCV
    Returns: W_hat value for this half cell"""

    R_f_c = deg_params[0, :, :]
    c_tilde_c = deg_params[1, :, :]
    R_f_a = deg_params[2, :, :]
    c_tilde_a = deg_params[3, :, :]
    c_lyte = deg_params[4, :, :]

    match params_c["rxn_method"]:
        case "BV":
            W_c_hat = np.multiply(np.multiply(((c_tilde_c - c_c) / (1 - c_c)) ** 0.5,
                                              (1 / (1 - R_f_c * dideta(c_c, V_c, params_c, mu_c)))),
                                  np.multiply(a_plus(c_lyte) ** 0.5, (
                                          1 + dideta(c_c, V_c, params_c, mu_c) / R(c_c, V_c, params_c, mu_c, 1) * (
                                              1 - c_lyte) * dlnadlnc(1))))
            W_a_hat = np.multiply(np.multiply(((c_tilde_a - c_a) / (1 - c_a)) ** 0.5, (
                    1 / (1 - R_f_a * dideta(c_a, V_a, params_a, mu_a)))), np.multiply(a_plus(c_lyte) ** 0.5, (
                    1 + dideta(c_a, V_a, params_a, mu_a) / R(c_a, V_a, params_a, mu_a, 1) * (
                    1 - c_lyte) * dlnadlnc(1))))
        # CIET
        case "CIET":
            W_c_hat = np.multiply(
                np.multiply((c_tilde_c - c_c) / (1 - c_c), (1 / (1 - R_f_c * dideta(c_c, V_c, params_c, mu_c)))), (
                        1 - (1 - c_lyte) * iredoi(c_c, V_c, params_c, mu_c) * dlnadlnc(1)))
            W_a_hat = np.multiply(
                np.multiply((c_tilde_a - c_a) / (1 - c_a), (1 / (1 - R_f_a * dideta(c_a, V_a, params_a, mu_a)))), (
                        1 - (1 - c_lyte) * iredoi(c_a, V_a, params_a, mu_a) * dlnadlnc(1)))

    dideta_c_a = dideta(c_c, V_c, params_c, mu_c) / dideta(c_a, V_a, params_a, mu_a)
    f_c_a = params_c["f"] / params_a["f"]
    W = (W_c_hat + W_a_hat * f_c_a * dideta_c_a) / (1 + f_c_a * dideta_c_a)

    return W


def dW_hat(c, V, R_f_list, c_tilde_list, c_lyte_list, params, mu):
    """Defines W_hat value for half cell electrode:
    Inputs: c: state of charge
    V: voltage
    R_f: film resistance
    c_tilde: rescaled capacity
    c_lyte: electrolyte concentration
    params: parameters related to electrode
    mu_c: functional form of OCV
    Returns: components of sensitivity matrix [dW/dR_f; dWdc_tilde; dWdc_lyte] for this half cell.
    The full cell version needs to be reassembled from the weighed version."""

    # print an array of things that influence
    match params["rxn_method"]:
        case "CIET":
            #    print("dideta", dideta(c, V, params, mu_c))
            dWdRf = np.multiply(np.divide(dideta(c, V, params, mu), (1 - R_f_list * dideta(c, V, params, mu)) ** 2),
                                np.multiply(np.divide((c_tilde_list - c), (
                                        1 - c)), (1 - (1 - c_lyte_list) * iredoi(c, V, params, mu) * dlnadlnc(1))))
            dWdctilde = 1 / (1 - c) * np.multiply((1 / (1 - R_f_list * dideta(c, V, params, mu))), (
                    1 - (1 - c_lyte_list) * iredoi(c, V, params, mu) * dlnadlnc(1)))
            dWdclyte = np.multiply(iredoi(c, V, params, mu) * dlnadlnc(1) * (c_tilde_list - c) / (1 - c), (
                    1 / (1 - R_f_list * dideta(c, V, params, mu))))
        #    print("redcurrentfrac", iredoi(c, V, params, mu_c, c_lyte))
        case "BV":
            #         print("DWDRF", dideta(c, V, params, mu_c)/(1-R_f*dideta(c, V, params, mu_c))**2)
            dWdRf = np.multiply(np.multiply(dideta(c, V, params, mu) / (1 - R_f_list * dideta(c, V, params, mu)) ** 2, (
                    (c_tilde_list - c) / (1 - c)) ** 0.5), np.multiply(a_plus(c_lyte_list) ** 0.5, (
                    1 + dideta(c, V, params, mu) / R(c, V, params, mu, 1) * (1 - c_lyte_list) * dlnadlnc(1))))
            dWdctilde = np.multiply(np.multiply(0.5 / np.sqrt((c_tilde_list - c) * (1 - c)), (
                    1 / (1 - R_f_list * dideta(c, V, params, mu)))), np.multiply(a_plus(c_lyte_list) ** 0.5, (
                    1 + dideta(c, V, params, mu) / R(c, V, params, mu, 1) * (1 - c_lyte_list) * dlnadlnc(1))))
            dWdclyte = np.multiply((np.multiply(np.divide(0.5 * a_plus(c_lyte_list) ** 0.5, c_lyte_list),
                                                np.multiply(dlnadlnc(c_lyte_list),
                                                            (1 + dideta(c, V, params, mu) / R(c, V, params, mu, 1)
                                                             * (1 - c_lyte_list) * dlnadlnc(1)))) - a_plus(
                c_lyte_list) ** 0.5 * dideta(c, V, params, mu) / R(c, V, params, mu, 1)
                                    * dlnadlnc(1)), np.multiply(((c_tilde_list - c) / (1 - c)) ** 0.5,
                                                                (1 / (1 - R_f_list * dideta(c, V, params, mu)))))
    output = np.stack((dWdRf, dWdctilde, dWdclyte), axis=2)
    return output


def dideta(c, mures, params, mu_c):
    """dideta for some overpotential and some reaction rate"""
    muh = np.reshape(mu_c(c, params["muR_ref"]), [1, -1])
    eta = muh - mures
    etaf = eta - np.log(c)
    #   print("eta", eta, etaf)
    match params["rxn_method"]:
        case "BV":
            # it's actually dhdeta for BV, too laz to write anotuerh f(x)
            out = params["k0"] * (1 - c) ** 0.5 * c ** 0.5 * (
                    -0.5 * np.exp(-0.5 * eta) - (1 - 0.5) * np.exp((1 - 0.5) * eta))
        #   %temporary
        case "CIET":
            out = params["k0"] * (1 - c) * (
                    - dhelper_fundetaf(-etaf, params["lambda"]) - c * dhelper_fundetaf(etaf, params["lambda"]))

    return out


def iredoi(c, mures, params, mu_c):
    """ired/i for CIET only. we don't use it in BV"""
    muh = mu_c(c, params["muR_ref"])
    eta = muh - mures
    etaf = eta - np.log(c)
    match params["rxn_method"]:
        case "CIET":
            out = helper_fun(-etaf, params["lambda"]) / (
                    helper_fun(-etaf, params["lambda"]) - c * helper_fun(etaf, params["lambda"]))
    return out


def helper_fun(eta_f, lmbda):
    """Marcus helper function for CIET reaction rate"""
    return (np.sqrt(np.pi * lmbda) / (1 + np.exp(-eta_f)) * \
            (1 - erf((lmbda - np.sqrt(1 + np.sqrt(lmbda) + eta_f ** 2)) / (2 * np.sqrt(lmbda)))))


def dhelper_fundetaf(eta_f, lmbda):
    """dhelper/detaf, useful for CIET senitivity"""
    return (eta_f * np.exp(-(lmbda - (eta_f ** 2 + lmbda ** (1 / 2) + 1) ** (1 / 2)) ** 2 / (4 * lmbda))) \
        / ((np.exp(-eta_f) + 1) * (eta_f ** 2 + lmbda ** (1 / 2) + 1) ** (1 / 2)) - \
        (lmbda ** (1 / 2) * np.pi ** (1 / 2) * np.exp(-eta_f) * (erf((lmbda - \
                                                                      (eta_f ** 2 + lmbda ** (1 / 2) + 1) ** (
                                                                              1 / 2)) / (
                                                                             2 * lmbda ** (1 / 2))) - 1)) / (
                np.exp(-eta_f) + 1) ** 2


def W_obj(phi_offset_c, c_c, c_a, phi, params_c, params_a, mu_c, mu_a, c_lyte):
    # all in units of A/m^2
    """Solves for the phi_offset_c under the constant current constraint"""
    return params_c['f'] * R(c_c, phi_offset_c, params_c, mu_c, c_lyte) + params_a['f'] * R(c_a, phi_offset_c + phi,
                                                                                            params_a, mu_a, c_lyte)


def dlnadlnc(c_lyte):
    """Returns thermodynamic factor"""
    return 601 / 620 - 24 / 31 * 0.5 * c_lyte ** (0.5) + 100164 / 96875 * 1.5 * c_lyte ** (1.5)


def a_plus(c_lyte):
    """Returns activity coefficient"""
    return np.multiply(c_lyte ** (601 / 620),
                       np.exp(-1299 / 5000 - 24 / 31 * c_lyte ** (0.5) + 100164 / 96875 * c_lyte ** (1.5)))


def current_magnitude(phi_offset_c, c_c, c_a, phi, params_c, params_a, mu_c, mu_a, c_lyte):
    # all in units of A/m^2
    """Solves for the cell level current magnitude, useful for calculating the error of the y matrix"""
    return np.abs(params_c['f'] * R(c_c, phi_offset_c, params_c, mu_c, c_lyte))


def R(c, mures, params, mu_c, c_lyte):
    """Reaction current density in A/m^2 for BV/CIET"""
    muh = mu_c(c, params["muR_ref"])
    eta = muh - mures
    etaf = eta - np.log(c / a_plus(c_lyte))
    match params["rxn_method"]:
        case "BV":
            # this is only h
            rxn = params["k0"] * (1 - c) ** 0.5 * c ** 0.5 * a_plus(c_lyte) ** 0.5 * (
                        np.exp(-0.5 * eta) - np.exp(0.5 * eta))
        case "CIET":
            # %temperoary
            i_red = helper_fun(-etaf, params["lambda"])
            i_ox = helper_fun(etaf, params["lambda"])
            rxn = params["k0"] * (1 - c) * (a_plus(c_lyte) * i_red - c * i_ox)
    return rxn


def W_initial(c_c, c_a, mu, params_c, params_a, mu_c, mu_a, c_lyte):
    """finds initial values of phi while keeping the current constraint for all mu values given"""
    mu_value = np.zeros(len(c_c))
    R_value = np.zeros(len(c_c))
    for i in range(len(c_c)):
        opt = fsolve(W_obj, mu_c(c_c[i], params_c["muR_ref"]),
                     (c_c[i], c_a[i], mu[i], params_c, params_a, mu_c, mu_a, c_lyte))
        mu_value[i] = opt[0]

    R_value = current_magnitude(mu_value, c_c, c_a, mu, params_c, params_a, mu_c, mu_a, c_lyte)
    return mu_value, R_value


def Tesla_NCA_Si_OCV(y):
    """Open circuit voltage measurement of NCA half cell"""
    a = np.array([0.145584993881910, 2.526321858618340, 172.0810484337340, 1.007518156438100,
                  1.349501707184530, 0.420519124096827, 2.635800979146210,
                  3.284611867463240]).reshape([1, -1])
    b = np.array([0.7961299985689542, 0.2953029849791878, -1.3438627370872127, 0.6463272973815986,
                  0.7378056244779166, 0.948857021183584, 0.5372357238527894,
                  0.8922020984716097]).reshape([1, -1])
    c = np.array([0.060350976183950786, 0.20193410562543265, 0.7371221766768185,
                  0.10337785458522612, 0.09513470475980132, 0.0422930728072207, 0.1757549310633964,
                  0.1413934223088055]).reshape([1, -1])
    y = y.reshape([-1, 1])
    OCV = np.sum(a * np.exp(-((y - b) / c) ** 2), axis=1)
    return OCV


def Tesla_NCA_Si(y, muR_ref):
    """ Berliner et al., 2022.
    chemical potential for graphite [kBT]
    """
    muR = get_muR_from_OCV(Tesla_NCA_Si_OCV(y), muR_ref)
    return muR


def Tesla_graphite_OCV(y):
    """Open circuit voltage measurement of graphite half cell"""
    a0 = -48.9921992984694
    a = np.array([29.9816001180044, 161.854109570929, -0.283281555638378,
                  - 47.7685802868867, -65.0631963216785]).reshape([1, -1])
    b = np.array([0.005700461982903098, -0.1056830819588037, 0.044467320399373095,
                  - 18.947769999614668, 0.0022683366694012178]).reshape([1, -1])
    c = np.array([-0.050928145838337484, 0.09687316296868148, 0.04235223640014242,
                  7.040771011524739, 0.0011604439514018858]).reshape([1, -1])
    y = y.reshape([-1, 1])

    OCV = a0 + np.squeeze(a[0, 0] * np.exp((y - b[0, 0]) / c[0, 0])) + \
          np.sum(a[0, 1:] * np.tanh((y - b[0, 1:]) / c[0, 1:]), axis=1)
    return OCV


def Tesla_graphite(y, muR_ref):
    """ Berliner et al., 2022.
    chemical potential for graphite [kBT]
    """
    muR = get_muR_from_OCV(Tesla_graphite_OCV(y), muR_ref)
    return muR


def get_muR_from_OCV(OCV, muR_ref):
    """gets chemical potential from OCV"""
    eokT = constants.e / (constants.k * 298)
    return -eokT * OCV + muR_ref


def get_OCV_from_muR(mu, muR_ref):
    """gets OCV from chemical potential"""
    eokT = constants.e / (constants.k * 298)
    return -1 / eokT * (mu - muR_ref)


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


# function that gets the min NCA diffusivity in range
def min_Dc(cmin, cmax, diff_data):
    # minimum diffusivity at a certain point
    Dc = np.nan * np.ones((len(cmin),))
    for k in range(len(cmin)):
        ind = np.argwhere(
            ((diff_data[:, 0] > cmin[k]) & (diff_data[:, 0] < cmax[k])) | (
                        (diff_data[:, 0] > cmax[k]) & (diff_data[:, 0] < cmin[k])))
        if len(ind) > 0:
            Dc[k] = np.min(diff_data[ind, 1] * diff_data[ind, 0])
        else:
            opt_ind = find_nearest(diff_data[:, 0], (cmin[k] + cmax[k]) / 2)
            Dc[k] = diff_data[opt_ind, 1] * diff_data[opt_ind, 0]
    return Dc

def time_obj(alpha_t, c_min_c, c_max_c, c_min_a, c_max_a, params_c, params_a, R_value):
    """returns minimum objective function depending on time in hours"""
    R_value_prev = np.concatenate((np.array([0]), R_value[:-1]))
    t_c = alpha_t * np.divide(
        (np.abs(c_max_c - c_min_c) + np.abs(R_value_prev / (constants.e * params_c["p"]) * t_pulse)) * params_c[
            'particle_size'] ** 2, min_Dc(c_min_c, c_max_c, params_c['diff']))
    t_a = alpha_t * np.divide(
        (np.abs(c_max_a - c_min_a) + np.abs(R_value_prev / (constants.e * params_a["p"]) * t_pulse)) * params_a[
            'particle_size'] ** 2, min_Dc(c_min_a, c_max_a, params_a['diff']))

    return np.maximum(t_c, t_a) / 3600


# set degradation parameters

# load NCA/graphite diffusivities
diffNCA = 10**np.loadtxt('amin_diffusion/NCA_diffusion.txt', delimiter = ',')
diffgraphite = np.loadtxt('diffusion_carelli_et_all/graphite_diffusion.txt', delimiter=',')

# particle_size_c = 0.24e-6
# particle_size_a = 16e-6


# particle size
r_c = 2e-7
r_a = 16e-6
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
c_s_0_a = 0.0142
c_s_0_c = 0.8595

# rescaling factor to convert particle level current to electrode level
f_c = L_c * (1 - poros_c) * P_L_c * 3 / r_c
f_a = L_a * (1 - poros_a) * P_L_a * 3 / r_a

# rescaling factor to balance electrode concentrations
p_c = L_c * (1 - poros_c) * P_L_c * rho_s_c
p_a = L_a * (1 - poros_a) * P_L_a * rho_s_a

# set reference chemical ptoentials
muR_ref_c = -Tesla_NCA_Si(np.array([c_s_0_c]), 0)[0]
muR_ref_a = -Tesla_graphite(np.array([c_s_0_a]), 0)[0]

# alpha in params_c represents the scaling of relaxation time.

# input parameters for electrodes
if is_balanced:
    # input parameters for electrodes
    params_c = {'rxn_method': rxn_method, 'k0': 1, 'lambda': 5, 'f': f_c, 'p': p_c, 'c0': c_s_0_c, 'mu': Tesla_NCA_Si,
            'muR_ref': muR_ref_c, 'diff': diffNCA, 'particle_size': r_c, 't_pulse': t_pulse}
    params_a = {'rxn_method': rxn_method, 'k0': 1, 'lambda': 8, 'f': f_a, 'p': p_a, 'c0': c_s_0_a,
                'mu': Tesla_graphite,
                'muR_ref': muR_ref_a, 'diff': diffgraphite, 'particle_size': r_a}
else:
    # input parameters for electrodes
    params_c = {'rxn_method': rxn_method, 'k0': 74, 'lambda': 5, 'f': f_c, 'p': p_c, 'c0': c_s_0_c,
                'mu': Tesla_NCA_Si,
                'muR_ref': muR_ref_c, 'diff': diffNCA, 'particle_size': r_c, 't_pulse': t_pulse}
    params_a = {'rxn_method': rxn_method, 'k0': 0.6, 'lambda': 8, 'f': f_a, 'p': p_a, 'c0': c_s_0_a,
                'mu': Tesla_graphite,
                'muR_ref': muR_ref_a, 'diff': diffgraphite, 'particle_size': r_a}

c_c = c_range
pulse_range = np.zeros((N))
for n in range(N):
    pulse_range[n] = get_muR_from_OCV(V_values[n], 0)
if is_initial_high:
    c_c_t = np.concatenate((np.array([0.8]), c_range))
else:
    c_c_t = np.concatenate((np.array([0.4]), c_range))
c_a = params_a["c0"] - params_c["p"] / params_a["p"] * (c_c - params_c["c0"])
c_a_t = params_a["c0"] - params_c["p"] / params_a["p"] * (c_c_t - params_c["c0"])

# solve for the initial voltage pulses
voltage_range = -Tesla_NCA_Si(c_c, params_c["muR_ref"]) + Tesla_graphite(c_a, params_a["muR_ref"]) + pulse_range

mu_range_c, R_value = W_initial(c_c, c_a, voltage_range, params_c, params_a, Tesla_NCA_Si, Tesla_graphite, 1) # No degradation
mu_range_a = mu_range_c + voltage_range

J2 = np.sum(time_obj(alpha_t, c_c_t[:-1], c_c_t[1:], c_a_t[:-1], c_a_t[1:], params_c, params_a, R_value))
# correct for voltage shift of mu_range_c and mu_range_a each with the deltaphi correction in Eq. 13 of hppc paper 1
V_c = mu_range_c
V_a = mu_range_a

DOE_list = lhs(5, samples=num_meshpoints, criterion = 'center', random_state=42)
dWdtheta_list = np.zeros((len(DOE_list), 1))

R_f_c_list = (deg_params[:, 0] + (deg_params[:, 1] - deg_params[:, 0]) * DOE_list[:, 0]).reshape(-1, 1)
c_tilde_c_list = (deg_params[:, 2] + (deg_params[:, 3] - deg_params[:, 2]) * DOE_list[:, 1]).reshape(-1, 1)
R_f_a_list = (deg_params[:, 4] + (deg_params[:, 5] - deg_params[:, 4]) * DOE_list[:, 2]).reshape(-1, 1)
c_tilde_a_list = (deg_params[:, 6] + (deg_params[:, 7] - deg_params[:, 6]) * DOE_list[:, 3]).reshape(-1, 1)
c_lyte_list = (deg_params[:, 8] + (deg_params[:, 9] - deg_params[:, 8]) * DOE_list[:, 4]).reshape(-1, 1)

dW_c_hat = dW_hat(c_c, mu_range_c, R_f_c_list, c_tilde_c_list, c_lyte_list, params_c, Tesla_NCA_Si)

# this is only from the cathode, so in the full cell it doesn't affect the anode rows
dW_c_hat = np.insert(dW_c_hat, (2, 2), 0, axis=2)
dW_a_hat = dW_hat(c_a, mu_range_a, R_f_a_list, c_tilde_a_list, c_lyte_list, params_a, Tesla_graphite)
# this is only from the anode, so in the full cell it doesn't affect the cathode rows
dW_a_hat = np.insert(dW_a_hat, (0, 0), 0, axis=2)
# print("dW", dW_c_hat, dW_a_hat)
# we should have different mures
dideta_c_a = dideta(c_c, mu_range_c, params_c, Tesla_NCA_Si) / dideta(c_a, mu_range_a, params_a, Tesla_graphite)
f_c_a = params_c["f"] / params_a["f"]
dWdtheta = np.divide((dW_c_hat + dW_a_hat * f_c_a * dideta_c_a[:, :, None]), (1 + f_c_a * dideta_c_a[:, :, None]))

W_range = W_hat(np.array([R_f_c_list, c_tilde_c_list, R_f_a_list, c_tilde_a_list, c_lyte_list]), c_c, c_a,
                mu_range_c, mu_range_a, params_c, params_a, Tesla_NCA_Si, Tesla_graphite)
err_y = np.abs(np.divide((1 - W_range), R_value))
for ind in range(num_meshpoints):
    sigma_y_inv = np.diag(err_y[ind] ** (-2))
    sigma_inv = np.dot(np.dot(dWdtheta[ind, :, :].T, sigma_y_inv), dWdtheta[ind, :, :])
    if np.linalg.det(sigma_inv) != 0:
        dWdtheta_list[ind] = 1 / np.linalg.det(sigma_inv)
    else:
        dWdtheta_list[ind] = np.exp(100)

J1 = np.log(np.mean(dWdtheta_list[~np.isnan(dWdtheta_list)]))
print("J1 = " + str(J1) + "\nJ2 = " + str(J2))

params_matrix = np.zeros((N, 8))
params_matrix[:,0] = c_c
params_matrix[:,1] = c_a
params_matrix[:,2] = get_OCV_from_muR(V_c, params_c["muR_ref"])
params_matrix[:,3] = get_OCV_from_muR(V_a, params_a["muR_ref"])
params_matrix[:,4] = params_matrix[:,2] - params_matrix[:,3]
params_matrix[:,5] = params_matrix[:,2] - params_matrix[:,3] - V_values
params_matrix[:,6] = V_values
params_matrix[:,7] = time_obj(alpha_t, c_c_t[:-1], c_c_t[1:], c_a_t[:-1], c_a_t[1:], params_c, params_a, R_value)

