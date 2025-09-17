import numpy as np
from scipy import constants
from scipy.special import erf
from scipy.optimize import fsolve
import pygmo as pg
from pyDOE3 import *
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel as C

def W(deg_params, c_c, c_a, V_c, V_a, params_c, params_a, mu_c, mu_a):
    """Defines W value for full cell using W_hat values for half cell electrode:
    Inputs:
    deg_params: array with degradation parameters: R_f_c, c_tilde_c, R_f_a, c_tilde_a, and c_lyte
    c_c, c_a: lithium filling fraction of the cathode and anode, respectively
    V_c, V_a: electrode potential at the cathode and anode in mu unit, respectively
    params_c, params_a: dictionary of parameters for the cathode and anode, respectively
    mu_c, mu_a: functional form of OCV at the cathode and anode, respectively
    Returns:
    W value for this full-cell based on W_hat values for half-cell electrode"""

    R_f_c = deg_params[0, :, :]
    c_tilde_c = deg_params[1, :, :]
    R_f_a = deg_params[2, :, :]
    c_tilde_a = deg_params[3, :, :]
    c_lyte = deg_params[4, :, :]


    match params_c["rxn_method"]:
        case "BV":
            W_c_hat = np.multiply(np.multiply(((c_tilde_c - c_c) / (1 - c_c)) ** 0.5, (1 / (1 - R_f_c * dideta(c=c_c, mures=V_c, params=params_c, mu_c=mu_c)))), np.multiply(a_plus(c_lyte=c_lyte) ** 0.5, (
                        1 + dideta(c=c_c, mures=V_c, params=params_c, mu_c=mu_c) / R(c=c_c, mures=V_c, params=params_c, mu_c=mu_c, c_lyte=1) * (1 - c_lyte) * dlnadlnc(c_lyte=1))))
            W_a_hat = np.multiply(np.multiply(((c_tilde_a - c_a) / (1 - c_a)) ** 0.5, (
                        1 / (1 - R_f_a * dideta(c=c_a, mures=V_a, params=params_a, mu_c=mu_a)))), np.multiply(a_plus(c_lyte=c_lyte) ** 0.5, (
                          1 + dideta(c=c_a, mures=V_a, params=params_a, mu_c=mu_a) / R(c=c_a, mures=V_a, params=params_a, mu_c=mu_a, c_lyte=1) * (
                              1 - c_lyte) * dlnadlnc(c_lyte=1))))
        # CIET
        case "CIET":
            W_c_hat = np.multiply(np.multiply((c_tilde_c - c_c) / (1 - c_c), (1 / (1 - R_f_c * dideta(c=c_c, mures=V_c, params=params_c, mu_c=mu_c)))), (
                        1 - (1 - c_lyte) * iredoi(c=c_c, mures=V_c, params=params_c, mu_c=mu_c) * dlnadlnc(c_lyte=1)))
            W_a_hat = np.multiply(np.multiply((c_tilde_a - c_a) / (1 - c_a), (1 / (1 - R_f_a * dideta(c=c_a, mures=V_a, params=params_a, mu_c=mu_a)))), (
                        1 - (1 - c_lyte) * iredoi(c=c_a, mures=V_a, params=params_a, mu_c=mu_a) * dlnadlnc(c_lyte=1)))

    dideta_c_a = dideta(c=c_c, mures=V_c, params=params_c, mu_c=mu_c) / dideta(c=c_a, mures=V_a, params=params_a, mu_c=mu_a)
    f_c_a = params_c["f"] / params_a["f"]
    W = (W_c_hat + W_a_hat * f_c_a * dideta_c_a) / (1 + f_c_a * dideta_c_a)

    return W



def dW_hat(c, V, R_f_list, c_tilde_list, c_lyte_list, params, mu):
    """Defines sensitivity matrix for W_hat (i.e., partial derivatives of W_hat with respect to degradation parameters) for half cell electrode over various degradation parameter values:
    Inputs:
    c: lithium filling fraction at the half-cell electrode
    V: electrode potential in mu unit
    R_f_list: list of film resistance values
    c_tilde_list: list of rescaled capacity values
    c_lyte: list of electrolyte concentration values
    Returns:
    components of sensitivity matrix [dW/dR_f; dWdc_tilde; dWdc_lyte] for this half cell.
    The full cell version needs to be reassembled from the weighed version."""

    # print an array of things that influence
    match params["rxn_method"]:
        case "CIET":
            dWdRf = np.multiply(np.divide(dideta(c=c, mures=V, params=params, mu_c=mu), (1 - R_f_list * dideta(c=c, mures=V, params=params, mu_c=mu)) ** 2), np.multiply(np.divide((c_tilde_list - c), (
                        1 - c)), (1 - (1 - c_lyte_list) * iredoi(c=c, mures=V, params=params, mu_c=mu) * dlnadlnc(c_lyte=1))))
            dWdctilde = 1 / (1 - c) * np.multiply((1 / (1 - R_f_list * dideta(c=c, mures=V, params=params, mu_c=mu))), (
                        1 - (1 - c_lyte_list) * iredoi(c=c, mures=V, params=params, mu_c=mu) * dlnadlnc(c_lyte=1)))
            dWdclyte = np.multiply(iredoi(c=c, mures=V, params=params, mu_c=mu) * dlnadlnc(c_lyte=1) * (c_tilde_list - c) / (1 - c), (
                        1 / (1 - R_f_list * dideta(c=c, mures=V, params=params, mu_c=mu))))
        case "BV":
            dWdRf = np.multiply(np.multiply(dideta(c=c, mures=V, params=params, mu_c=mu) / (1 - R_f_list * dideta(c=c, mures=V, params=params, mu_c=mu)) ** 2, (
                        (c_tilde_list - c) / (1 - c)) ** 0.5), np.multiply(a_plus(c_lyte=c_lyte_list) ** 0.5, (
                                1 + dideta(c=c, mures=V, params=params, mu_c=mu) / R(c=c, mures=V, params=params, mu_c=mu, c_lyte=1) * (1 - c_lyte_list) * dlnadlnc(c_lyte=1))))
            dWdctilde = np.multiply(np.multiply(0.5 / np.sqrt((c_tilde_list - c) * (1 - c)), (
                        1 / (1 - R_f_list * dideta(c=c, mures=V, params=params, mu_c=mu)))), np.multiply(a_plus(c_lyte=c_lyte_list) ** 0.5, (
                                    1 + dideta(c=c, mures=V, params=params, mu_c=mu) / R(c=c, mures=V, params=params, mu_c=mu, c_lyte=1) * (1 - c_lyte_list) * dlnadlnc(c_lyte=1))))
            dWdclyte = np.multiply((np.multiply(np.divide(0.5 * a_plus(c_lyte=c_lyte_list) ** 0.5, c_lyte_list), np.multiply(dlnadlnc(c_lyte=1), (1 + dideta(c=c, mures=V, params=params, mu_c=mu) / R(c=c, mures=V, params=params, mu_c=mu, c_lyte=1)
                        * (1 - c_lyte_list) * dlnadlnc(c_lyte=1)))) - a_plus(c_lyte=c_lyte_list) ** 0.5 * dideta(c=c, mures=V, params=params, mu_c=mu) / R(c=c, mures=V, params=params, mu_c=mu, c_lyte=1)
                        * dlnadlnc(c_lyte=1)), np.multiply(((c_tilde_list - c) / (1 - c)) ** 0.5, (1 / (1 - R_f_list * dideta(c=c, mures=V, params=params, mu_c=mu)))))
    output = np.stack((dWdRf, dWdctilde, dWdclyte), axis=2)
    return output


def dideta(c, mures, params, mu_c):
    """dideta for the given electrode potential and the OCV functional form"""
    muh = np.reshape(mu_c(y=c, muR_ref=params["muR_ref"]), [1, -1])
    eta = muh - mures
    etaf = eta - np.log(c)
    match params["rxn_method"]:
        case "BV":
            out = params["k0"] * (1 - c) ** 0.5 * c ** 0.5 * (
                        -0.5 * np.exp(-0.5 * eta) - (1 - 0.5) * np.exp((1 - 0.5) * eta))
        case "CIET":
            out = params["k0"] * (1 - c) / np.sqrt(4 * np.pi * params["lambda"]) * (
                        - dhelper_fundetaf(eta_f=-etaf, lmbda=params["lambda"]) - c * dhelper_fundetaf(eta_f=etaf, lmbda=params["lambda"]))

    return out


def iredoi(c, mures, params, mu_c):
    """ired/i for CIET only. we don't use it in BV"""
    muh = mu_c(y=c, muR_ref=params["muR_ref"])
    eta = muh - mures
    etaf = eta - np.log(c)
    match params["rxn_method"]:
        case "CIET":
            out = helper_fun(eta_f=-etaf, lmbda=params["lambda"]) / (
                        helper_fun(eta_f=-etaf, lmbda=params["lambda"]) - c * helper_fun(eta_f=etaf, lmbda=params["lambda"]))
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
    return params_c['f'] * R(c=c_c, mures=phi_offset_c, params=params_c, mu_c=mu_c, c_lyte=c_lyte) + params_a['f'] * R(c=c_a, mures=phi_offset_c + phi,
                                                                                            params=params_a, mu_c=mu_a, c_lyte=c_lyte)




def dlnadlnc(c_lyte):
    """Returns thermodynamic factor"""
    return 601/620 - 24 / 31 * 0.5 * c_lyte ** (0.5) + 100164 / 96875 * 1.5 * c_lyte ** 1.5


def a_plus(c_lyte):
    """Returns activity coefficient"""
    return np.multiply(c_lyte ** (601 / 620), np.exp(-1299 / 5000 - 24 / 31 * c_lyte ** (0.5) + 100164 / 96875 * c_lyte ** (1.5)))


def current_magnitude(phi_offset_c, c_c, c_a, phi, params_c, params_a, mu_c, mu_a, c_lyte):
    # all in units of A/m^2
    """Calculates cell-level current magnitude, useful for calculating the error of the y matrix"""
    return np.abs(params_c['f'] * R(c=c_c, mures=phi_offset_c, params=params_c, mu_c=mu_c, c_lyte=c_lyte))


def R(c, mures, params, mu_c, c_lyte):
    """Reaction current density in A/m^2 for BV/CIET"""
    muh = mu_c(y=c, muR_ref=params["muR_ref"])
    eta = muh - mures
    etaf = eta - np.log(c/a_plus(c_lyte=c_lyte))
    match params["rxn_method"]:
        case "BV":
            rxn = params["k0"] * (1 - c) ** 0.5 * c ** 0.5 * a_plus(c_lyte=c_lyte) ** 0.5 * (np.exp(-0.5 * eta) - np.exp(0.5 * eta))
        case "CIET":
            i_red = helper_fun(eta_f=-etaf, lmbda=params["lambda"])
            i_ox = helper_fun(eta_f=etaf, lmbda=params["lambda"])
            rxn = params["k0"] * (1 - c) / np.sqrt(4 * np.pi * params["lambda"]) * (a_plus(c_lyte=c_lyte) * i_red - c * i_ox)
    return rxn


def W_initial(c_c, c_a, mu, params_c, params_a, mu_c, mu_a, c_lyte):
    """finds initial values of phi while keeping the current constraint for all mu values given"""
    mu_value = np.zeros(len(c_c))
    for i in range(len(c_c)):
        opt = fsolve(W_obj, mu_c(y=c_c[i], muR_ref=params_c["muR_ref"]), (c_c[i], c_a[i], mu[i], params_c, params_a, mu_c, mu_a, c_lyte))
        mu_value[i] = opt[0]

    R_value = current_magnitude(phi_offset_c=mu_value, c_c=c_c, c_a=c_a, phi=mu, params_c=params_c, params_a=params_a, mu_c=mu_c, mu_a=mu_a, c_lyte=c_lyte)
    return mu_value, R_value


def eval_Phi_func_values(Xn, deg_params_bound, c_c, c_a, params_c, params_a, mu_c, mu_a, pulse_range, Phi_func_option):
    """solves phi_func_values for degradation parameter values in Xn"""

    # Unpack the input sample
    R_f_c_norm_list = Xn[:, 0].reshape(-1, 1)
    c_tilde_c_norm_list = Xn[:, 1].reshape(-1, 1)
    R_f_a_norm_list = Xn[:, 2].reshape(-1, 1)
    c_tilde_a_norm_list = Xn[:, 3].reshape(-1, 1)
    c_lyte_norm_list = Xn[:, 4].reshape(-1, 1)

    # Scale normalized inputs to physical values
    def scale(x, min_val, max_val):
        return min_val + (max_val - min_val) * x

    R_f_c_list = scale(x=R_f_c_norm_list, min_val=deg_params_bound[0, 0], max_val=deg_params_bound[0, 1]).reshape(-1, 1)
    c_tilde_c_list = scale(x=c_tilde_c_norm_list, min_val=deg_params_bound[1, 0], max_val=deg_params_bound[1, 1]).reshape(-1, 1)
    R_f_a_list = scale(x=R_f_a_norm_list, min_val=deg_params_bound[2, 0], max_val=deg_params_bound[2, 1]).reshape(-1, 1)
    c_tilde_a_list = scale(x=c_tilde_a_norm_list, min_val=deg_params_bound[3, 0], max_val=deg_params_bound[3, 1]).reshape(-1, 1)
    c_lyte_list = scale(x=c_lyte_norm_list, min_val=deg_params_bound[4, 0], max_val=deg_params_bound[4, 1]).reshape(-1, 1)

    voltage_range = -mu_c(y=c_c, muR_ref=params_c["muR_ref"]) + mu_a(y=c_a, muR_ref=params_a["muR_ref"]) + pulse_range
    # solve for the initial voltage pulses
    mu_range_c, R_value = W_initial(c_c=c_c, c_a=c_a, mu=voltage_range, params_c=params_c, params_a=params_a, mu_c=mu_c, mu_a=mu_a, c_lyte=1)  # No degradation
    mu_range_a = mu_range_c + voltage_range

    dW_c_hat = dW_hat(c=c_c, V=mu_range_c, R_f_list=R_f_c_list, c_tilde_list=c_tilde_c_list, c_lyte_list=c_lyte_list, params=params_c, mu=mu_c)

    # this is only from the cathode, so in the full cell it doesn't affect the anode rows
    dW_c_hat = np.insert(dW_c_hat, (2, 2), 0, axis=2)
    dW_a_hat = dW_hat(c=c_a, V=mu_range_a, R_f_list=R_f_a_list, c_tilde_list=c_tilde_a_list, c_lyte_list=c_lyte_list, params=params_a, mu=mu_a)
    # this is only from the anode, so in the full cell it doesn't affect the cathode rows
    dW_a_hat = np.insert(dW_a_hat, (0, 0), 0, axis=2)
    dideta_c_a = dideta(c=c_c, mures=mu_range_c, params=params_c, mu_c=mu_c) / dideta(c=c_a, mures=mu_range_a, params=params_a, mu_c=mu_a)
    f_c_a = params_c["f"] / params_a["f"]
    dWdtheta = np.divide((dW_c_hat + dW_a_hat * f_c_a * dideta_c_a[:, :, None]), (1 + f_c_a * dideta_c_a[:, :, None]))

    W_range = W(deg_params=np.array([R_f_c_list, c_tilde_c_list, R_f_a_list, c_tilde_a_list, c_lyte_list]), c_c=c_c, c_a=c_a,
                V_c=mu_range_c, V_a=mu_range_a, params_c=params_c, params_a=params_a, mu_c=mu_c, mu_a=mu_a)
    err_y = np.abs(np.multiply((1 - W_range), W_range))

    phi_func_list = np.full((Xn.shape[0],), np.nan)
    for ind in range(Xn.shape[0]):
        # --- guards ---
        ey = np.asarray(err_y[ind], dtype=np.float64)
        if (not np.all(np.isfinite(ey))) or np.any(ey <= 0):
            raise ValueError(f"err_y[{ind}] must be positive and finite")

        A = np.asarray(dWdtheta[ind], dtype=np.float64)  # shape (m, p)

        # Build B = A / ey (row scaling once) so S = B^T B
        B = A / ey[:, None]

        # Singular values of B (most stable way to get logdet of B^T B)
        s = np.linalg.svd(B, compute_uv=False)

        # Scale-aware tolerance for numerical zero
        smax = s.max() if s.size else 0.0
        tol = 1e-12 * max(1.0, smax)

        if np.any(s <= tol):
            # Rank-deficient (det(S)=prod s_i^2 ~ 0) -> Phi_D = 1/det -> +inf in log domain
            phi_func_list[ind] = np.inf  # or set to a large cap like 1e6 if you prefer
            continue

        # log det(S) = 2 * sum(log s_i)
        if Phi_func_option == "D":
            Phi_func = - 2.0 * np.sum(np.log(s))
        elif Phi_func_option == "A":
            Phi_func = np.log(np.sum(1.0 / (s**2)))
        elif Phi_func_option == "E":
            Phi_func = np.log(1.0 / np.min(s)**2)
        phi_func_list[ind] = Phi_func  # log(Phi_D)

    return phi_func_list, R_value


def Tesla_NCA_Si_OCV(y):
    """Open circuit voltage measurement of NCA half cell as a function of lithium filling fraction"""
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
    muR = get_muR_from_OCV(OCV=Tesla_NCA_Si_OCV(y=y), muR_ref=muR_ref)
    return muR


def Tesla_graphite_OCV(y):
    """Open circuit voltage measurement of graphite half cell as a function of lithium filling fraction"""
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
    muR = get_muR_from_OCV(OCV=Tesla_graphite_OCV(y=y), muR_ref=muR_ref)
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
    """returns the minimum D(c) * c within the discretized c chunks"""
    Dc = np.nan * np.ones((len(cmin), ))
    for k in range(len(cmin)):
        ind = np.argwhere(
            ((diff_data[:, 0] > cmin[k]) & (diff_data[:, 0] < cmax[k])) | ((diff_data[:, 0] > cmax[k]) & (diff_data[:, 0] < cmin[k])))
        if len(ind) > 0:
            Dc[k] = np.min(diff_data[ind, 1] * diff_data[ind, 0])
        else:
            opt_ind = find_nearest(array=diff_data[:, 0], value=(cmin[k] + cmax[k]) / 2)
            Dc[k] =  diff_data[opt_ind, 1] * diff_data[opt_ind, 0]
    return Dc


def f_time(alpha_t, c_min_c, c_max_c, c_min_a, c_max_a, params_c, params_a, R_value, t_pulse):
    """returns f_time in hours"""
    R_value_prev = np.concatenate((np.array([0]), R_value[:-1]))
    t_c = alpha_t * np.divide((np.abs(c_max_c - c_min_c) + np.abs(R_value_prev / (constants.e * params_c["p"]) * t_pulse)) * params_c['particle_size'] ** 2, min_Dc(cmin=c_min_c, cmax=c_max_c, diff_data=params_c['diff']))
    t_a = alpha_t * np.divide((np.abs(c_max_a - c_min_a) + np.abs(R_value_prev / (constants.e * params_a["p"]) * t_pulse)) * params_a['particle_size'] ** 2, min_Dc(cmin=c_min_a, cmax=c_max_a, diff_data=params_a['diff']))

    return np.maximum(t_c, t_a) / 3600

def f_uncertainty_fixtlast(opt_params, N, deg_params_bound, params_c, params_a, mu_c, mu_a, t_last, is_initial_high, V_limit_high, V_limit_low, c_c_limit_high, c_c_limit_low, Phi_func_option="D",
                           gp_kernel="matern", gp_nu=2.5, gp_n_restarts=5, gp_normalize_y=True, gp_noise_floor=1e-6,
                           n_train = 250, n_val = 100, n_sample = 10000, verbose=False, gp_for_surrogate = False):
    """Calculate the first objective function, which is the averaged D-optimality across the degradation parameter space, from the given opt_params.
    Here, opt_params is the solution for the optimization problem, which can be further translated to physical pulse parameters (c_c and dVs at each pulse)"""

    if is_initial_high:
        c_range = c_c_limit_high + np.cumsum(np.insert(opt_params[:N-1], 0, 0)) / (np.sum(np.insert(opt_params[:N-1], 0, 0)) + t_last) * (c_c_limit_low - c_c_limit_high)
    else:
        c_range = c_c_limit_low + np.cumsum(np.insert(opt_params[:N-1], 0, 0)) / (np.sum(np.insert(opt_params[:N-1], 0, 0)) + t_last) * (c_c_limit_high - c_c_limit_low)
    c_range = np.round(c_range * 1000)/1000

    dV = np.zeros(N)
    mask1 = opt_params[-N:] >= 0.5
    dV[mask1] = (V_limit_high - V_limit_low) * 2 * (opt_params[-N:][mask1] - 0.5) + V_limit_low
    mask2 = opt_params[-N:] < 0.5
    dV[mask2] = - ((V_limit_high - V_limit_low) * 2 * (0.5 - opt_params[-N:][mask2]) + V_limit_low)
    dV = np.round(dV * 1000)/1000
    pulse_range = get_muR_from_OCV(OCV=dV, muR_ref=0)

    c_c = c_range
    if is_initial_high:
        c_c_t = np.concatenate((np.array([c_c_limit_high]), c_range))
    else:
        c_c_t = np.concatenate((np.array([c_c_limit_low]), c_range))
    c_a = params_a["c0"] - params_c["p"] / params_a["p"] * (c_c - params_c["c0"])
    c_a_t = params_a["c0"] - params_c["p"] / params_a["p"] * (c_c_t - params_c["c0"])
    d = deg_params_bound.shape[0]
    if gp_for_surrogate:
        # Evaluate the model on physical samples
        Xn_train = lhs(d, samples=n_train, criterion='center', random_state=42)
        Phi_func_train, R_value = eval_Phi_func_values(
            Xn=Xn_train, deg_params_bound=deg_params_bound, c_c=c_c, c_a=c_a, params_c=params_c, params_a=params_a, mu_c=mu_c, mu_a=mu_a, pulse_range=pulse_range, Phi_func_option=Phi_func_option
        )
        # ---- fit GP on normalized inputs ----
        gp = fit_gp_model(
            Xn_train, Phi_func_train,
            kernel=gp_kernel, nu=gp_nu, noise_floor=gp_noise_floor,
            n_restarts=gp_n_restarts, normalize_y=gp_normalize_y
        )

        Xn_val = lhs(d, samples=n_val, criterion='center', random_state=0)
        Phi_func_val, R_value = eval_Phi_func_values(
            Xn=Xn_val, deg_params_bound=deg_params_bound, c_c=c_c, c_a=c_a, params_c=params_c, params_a=params_a, mu_c=mu_c, mu_a=mu_a, pulse_range=pulse_range, Phi_func_option=Phi_func_option
        )
        mu_val, std_val = gp.predict(Xn_val, return_std=True)
        cov = ((Phi_func_val >= mu_val - 1.96 * std_val) & (Phi_func_val <= mu_val + 1.96 * std_val)).mean()
        print("Evaluation of the GP surrogate model for Phi-D: 95% coverage:", cov)

        # ---- expectation of phi_ED under the Uniform prior (via GP mean) ----
        Xn_sample = lhs(d, samples=n_sample, criterion='center', random_state=42)
        mu_sample, std_sample = gp.predict(Xn_sample, return_std=True)
        phi_ED = float(np.mean(mu_sample))
    else:
        Xn_sample = lhs(d, samples=n_sample, criterion='center', random_state=42)
        Phi_func, R_value = eval_Phi_func_values(Xn=Xn_sample, deg_params_bound=deg_params_bound, c_c=c_c, c_a=c_a, params_c=params_c, params_a=params_a, mu_c=mu_c, mu_a=mu_a,
                                           pulse_range=pulse_range, Phi_func_option=Phi_func_option)
        phi_ED = np.mean(Phi_func)
    if verbose:
        if gp_for_surrogate:
            print(f"Average mean & std of samples for GP surrogate for Phi-D: {np.mean(mu_sample):.4f}, {np.mean(std_sample):.4f}")
        print("Pulse at c = " + str(c_range) + " with dV = " + str(dV) + " resulted det = " + str(phi_ED))

    return phi_ED, R_value

def f_uncertainty_fixtlast_multi(opt_params, N, deg_params_bound, params_c, params_a, mu_c, mu_a, is_initial_high, V_limit_high, V_limit_low,
                            c_c_limit_high, c_c_limit_low, Phi_func_option = "D",
                           gp_kernel="matern", gp_nu=2.5, gp_n_restarts=5, gp_normalize_y=True, gp_noise_floor=1e-6,
                           n_train = 250, n_val = 100, n_sample = 10000, verbose=False, gp_for_surrogate = False, need_to_convert_opt = True,
                                 c_range = [], dV = []):
    """Same as f_uncertainty_fixtlast, except that it's for solving the two-objective optimzation problem"""
    if need_to_convert_opt:
        # generate matrix of N pulses and M dedgradation mechanisms, N*M
        #   (c_range, V_range) = opt_params # both size (N*1, N*1)
        if is_initial_high:
            c_range = c_c_limit_high + np.cumsum(np.insert(opt_params[:N-1], 0, 0)) / (np.sum(opt_params[:N])) * (c_c_limit_low - c_c_limit_high)
        else:
            c_range = c_c_limit_low + np.cumsum(np.insert(opt_params[:N-1], 0, 0)) / (np.sum(opt_params[:N])) * (c_c_limit_high - c_c_limit_low)
        c_range = np.round(c_range * 1000)/1000

        dV = np.zeros(N)
        mask1 = opt_params[-N:] >= 0.5
        dV[mask1] = (V_limit_high - V_limit_low) * 2 * (opt_params[-N:][mask1] - 0.5) + V_limit_low
        mask2 = opt_params[-N:] < 0.5
        dV[mask2] = - ((V_limit_high - V_limit_low) * 2 * (0.5 - opt_params[-N:][mask2]) + V_limit_low)
        dV = np.round(dV * 1000)/1000
    pulse_range = get_muR_from_OCV(OCV=dV, muR_ref=0)

    c_c = c_range
    if is_initial_high:
        c_c_t = np.concatenate((np.array([c_c_limit_high]), c_range))
    else:
        c_c_t = np.concatenate((np.array([c_c_limit_low]), c_range))
    c_a = params_a["c0"] - params_c["p"] / params_a["p"] * (c_c - params_c["c0"])
    c_a_t = params_a["c0"] - params_c["p"] / params_a["p"] * (c_c_t - params_c["c0"])
    d = deg_params_bound.shape[0]
    if gp_for_surrogate:
        # Evaluate the model on physical samples
        Xn_train = lhs(d, samples=n_train, criterion='center', random_state=42)
        Phi_func_train, R_value = eval_Phi_func_values(
            Xn=Xn_train, deg_params_bound=deg_params_bound, c_c=c_c, c_a=c_a, params_c=params_c, params_a=params_a, mu_c=mu_c, mu_a=mu_a, pulse_range=pulse_range, Phi_func_option=Phi_func_option
        )
        # ---- fit GP on normalized inputs ----
        gp = fit_gp_model(
            Xn_train, Phi_func_train,
            kernel=gp_kernel, nu=gp_nu, noise_floor=gp_noise_floor,
            n_restarts=gp_n_restarts, normalize_y=gp_normalize_y
        )

        Xn_val = lhs(d, samples=n_val, criterion='center', random_state=0)
        Phi_func_val, R_value = eval_Phi_func_values(
            Xn=Xn_val, deg_params_bound=deg_params_bound, c_c=c_c, c_a=c_a, params_c=params_c, params_a=params_a, mu_c=mu_c, mu_a=mu_a, pulse_range=pulse_range, Phi_func_option=Phi_func_option
        )
        mu_val, std_val = gp.predict(Xn_val, return_std=True)
        cov = ((Phi_func_val >= mu_val - 1.96 * std_val) & (Phi_func_val <= mu_val + 1.96 * std_val)).mean()
        print("Evaluation of the GP surrogate model for Phi-D: 95% coverage:", cov)

        # ---- expectation of phi_ED under the Uniform prior (via GP mean) ----
        Xn_sample = lhs(d, samples=n_sample, criterion='center', random_state=42)
        mu_sample, std_sample = gp.predict(Xn_sample, return_std=True)
        phi_ED = float(np.mean(mu_sample))
    else:
        Xn_sample = lhs(d, samples=n_sample, criterion='center', random_state=42)
        Phi_func, R_value = eval_Phi_func_values(Xn=Xn_sample, deg_params_bound=deg_params_bound, c_c=c_c, c_a=c_a, params_c=params_c, params_a=params_a, mu_c=mu_c, mu_a=mu_a, pulse_range=pulse_range, Phi_func_option=Phi_func_option)
        E_phi_func = np.mean(Phi_func)
    if verbose:
        if gp_for_surrogate:
            print(f"Average mean & std of samples for GP surrogate for Phi-D: {np.mean(mu_sample):.4f}, {np.mean(std_sample):.4f}")
        print("Pulse at c = " + str(c_range) + " with dV = " + str(dV) + " resulted det = " + str(phi_ED))

    return E_phi_func, R_value


def fit_gp_model(Xn, y, kernel="matern", nu=2.5, n_restarts=5, normalize_y=True, noise_floor=1e-6):
    """Constructing GP model for mapping from Xn to y"""
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    d = Xn.shape[1]
    if kernel == "rbf":
        base_k = RBF(length_scale=np.ones(d), length_scale_bounds=(1e-3, 1e3))
    else:
        # Matérn(ν) is a good default for physical systems
        base_k = Matern(length_scale=np.ones(d), length_scale_bounds=(1e-3, 1e3), nu=nu)

    k = C(1.0, (1e-3, 1e3)) * base_k + WhiteKernel(noise_level=noise_floor, noise_level_bounds=(1e-9, 1e-1))
    gp = GaussianProcessRegressor(
        kernel=k,
        alpha=0.0,  # rely on WhiteKernel for noise
        normalize_y=normalize_y,
        n_restarts_optimizer=n_restarts,
        random_state=0
    )
    gp.fit(Xn, y)
    return gp

class uncertainty_function_fixtlast:
    """Defining one-objective optimization problem in order to use pygmo package"""
    def __init__(self, N, t_last, idx, is_initial_high, params_a, params_c, mu_a, mu_c, deg_params_bound, alpha_t, t_limit_list, V_limit_high, V_limit_low, c_c_limit_high, c_c_limit_low, t_pulse, Phi_func_option, gp_for_surrogate):
        self.N = N
        self.dim = 2 * N - 1
        self.t_last = t_last
        self.idx = idx
        self.is_initial_high = is_initial_high
        self.params_a = params_a
        self.params_c = params_c
        self.mu_a = mu_a
        self.mu_c = mu_c
        self.deg_params_bound = deg_params_bound
        self.alpha_t = alpha_t
        self.t_limit_list = t_limit_list
        self.V_limit_high = V_limit_high
        self.V_limit_low = V_limit_low
        self.c_c_limit_high = c_c_limit_high
        self.c_c_limit_low = c_c_limit_low
        self.t_pulse = t_pulse
        self.Phi_func_option = Phi_func_option
        self.gp_for_surrogate = gp_for_surrogate

    def fitness(self, x):
        N = self.N
        t_last = self.t_last
        idx = self.idx
        is_initial_high = self.is_initial_high
        params_a = self.params_a
        params_c = self.params_c
        mu_a = self.mu_a
        mu_c = self.mu_c
        deg_params_bound = self.deg_params_bound
        alpha_t = self.alpha_t
        t_limit_list = self.t_limit_list
        V_limit_high = self.V_limit_high
        V_limit_low = self.V_limit_low
        c_c_limit_high = self.c_c_limit_high
        c_c_limit_low = self.c_c_limit_low
        t_pulse = self.t_pulse
        Phi_func_option = self.Phi_func_option
        gp_for_surrogate = self.gp_for_surrogate

        if is_initial_high:
            c_range = c_c_limit_high + np.cumsum(np.insert(x[:N-1], 0, 0)) / (np.sum(np.insert(x[:N-1], 0, 0)) + t_last) * (c_c_limit_low - c_c_limit_high)
        else:
            c_range = c_c_limit_low + np.cumsum(np.insert(x[:N-1], 0, 0)) / (np.sum(np.insert(x[:N-1], 0, 0)) + t_last) * (c_c_limit_high - c_c_limit_low)
        c_range = np.round(c_range * 1000) / 1000

        if is_initial_high:
            c_c_t = np.concatenate((np.array([c_c_limit_high]), c_range))
        else:
            c_c_t = np.concatenate((np.array([c_c_limit_low]), c_range))
        c_a_t = params_a["c0"] - params_c["p"] / params_a["p"] * (c_c_t - params_c["c0"])

        obj, R_value = f_uncertainty_fixtlast(opt_params=x, N=N, deg_params_bound=deg_params_bound, params_c=params_c, params_a=params_a, mu_c=mu_c, mu_a=mu_a,
                                              t_last=t_last, is_initial_high=is_initial_high, V_limit_high=V_limit_high, V_limit_low=V_limit_low, c_c_limit_high=c_c_limit_high, c_c_limit_low=c_c_limit_low, Phi_func_option=Phi_func_option, gp_for_surrogate=gp_for_surrogate)
        ci1 = np.sum(f_time(alpha_t=alpha_t, c_min_c=c_c_t[:-1], c_max_c=c_c_t[1:], c_min_a=c_a_t[:-1], c_max_a=c_a_t[1:], params_c=params_c, params_a=params_a, R_value=R_value, t_pulse=t_pulse)) - t_limit_list[idx]

        return [obj, ci1]

    def get_bounds(self):
        N = self.N
        return ([0.00001] * (2*N-1), [1] * (2*N-1))
    def get_nic(self):
        return 1
    def get_nec(self):
        return 0
    def gradient(self, x):
        return pg.estimate_gradient_h(lambda x: self.fitness(x), x)
    def get_name(self):
        return "Uncertainty Function"

    def get_extra_info(self):
        return "\tDimensions: " + str(self.dim)

class uncertainty_function_fixtlast_multi:
    """Defining two-objective optimization problem in order to use pygmo package"""
    def __init__(self, N, is_initial_high, params_a, params_c, mu_a, mu_c, deg_params_bound, alpha_t, V_limit_high, V_limit_low, c_c_limit_high, c_c_limit_low, t_pulse, Phi_func_option, gp_for_surrogate):
        self.N = N
        self.dim = 2 * N
        self.is_initial_high = is_initial_high
        self.params_a = params_a
        self.params_c = params_c
        self.mu_a = mu_a
        self.mu_c = mu_c
        self.deg_params_bound = deg_params_bound
        self.alpha_t = alpha_t
        self.V_limit_high = V_limit_high
        self.V_limit_low = V_limit_low
        self.c_c_limit_high = c_c_limit_high
        self.c_c_limit_low = c_c_limit_low
        self.t_pulse = t_pulse
        self.Phi_func_option = Phi_func_option
        self.gp_for_surrogate = gp_for_surrogate

    def fitness(self, x):
        N = self.N
        is_initial_high = self.is_initial_high
        params_a = self.params_a
        params_c = self.params_c
        mu_a = self.mu_a
        mu_c = self.mu_c
        deg_params_bound = self.deg_params_bound
        alpha_t = self.alpha_t
        V_limit_high = self.V_limit_high
        V_limit_low = self.V_limit_low
        c_c_limit_high = self.c_c_limit_high
        c_c_limit_low = self.c_c_limit_low
        t_pulse = self.t_pulse
        Phi_func_option = self.Phi_func_option
        gp_for_surrogate = self.gp_for_surrogate

        if is_initial_high:
            c_range = c_c_limit_high + np.cumsum(np.insert(x[:N-1], 0, 0)) / (np.sum(x[:N])) * (c_c_limit_low - c_c_limit_high)
        else:
            c_range = c_c_limit_low + np.cumsum(np.insert(x[:N-1], 0, 0)) / (np.sum(x[:N])) * (c_c_limit_high - c_c_limit_low)
        c_range = np.round(c_range * 1000) / 1000

        if is_initial_high:
            c_c_t = np.concatenate((np.array([c_c_limit_high]), c_range))
        else:
            c_c_t = np.concatenate((np.array([c_c_limit_low]), c_range))
        c_a_t = params_a["c0"] - params_c["p"] / params_a["p"] * (c_c_t - params_c["c0"])

        obj1, R_value = f_uncertainty_fixtlast_multi(opt_params=x, N=N, deg_params_bound=deg_params_bound, params_c=params_c, params_a=params_a, mu_c=mu_c, mu_a=mu_a,
                                              is_initial_high=is_initial_high, V_limit_high=V_limit_high, V_limit_low=V_limit_low, c_c_limit_high=c_c_limit_high, c_c_limit_low=c_c_limit_low, Phi_func_option=Phi_func_option, gp_for_surrogate=gp_for_surrogate)
        obj2 = np.sum(f_time(alpha_t=alpha_t, c_min_c=c_c_t[:-1], c_max_c=c_c_t[1:], c_min_a=c_a_t[:-1], c_max_a=c_a_t[1:], params_c=params_c, params_a=params_a, R_value=R_value, t_pulse=t_pulse))

        return [obj1, obj2]

    def get_bounds(self):
        N = self.N
        return ([0.00001] * (2*N), [1] * (2*N))
    def get_nobj(self):
        return 2
    def get_nic(self):
        return 0
    def get_nec(self):
        return 0
    def gradient(self, x):
        return pg.estimate_gradient_h(lambda x: self.fitness(x), x)
    def get_name(self):
        return "Uncertainty Function"

    def get_extra_info(self):
        return "\tDimensions: " + str(self.dim)

