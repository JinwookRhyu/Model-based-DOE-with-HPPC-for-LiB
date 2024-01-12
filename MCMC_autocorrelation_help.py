from scipy import constants
from scipy.special import erf
from scipy.optimize import fsolve
import numpy as np

def W_hat(c, V, R_f, c_tilde, c_lyte, params, mu_c):
    """Defines W_hat value for half cell electrode:
    Inputs: c: state of charge
    V: voltage
    R_f: film resistance
    c_tilde: rescaled capacity
    c_lyte: electrolyte concentration
    params: parameters related to electrode
    mu_c: functional form of OCV
    Returns: W_hat value for this half cell"""
    # TODO: plug in a_plus term for BV model
    match params["rxn_method"]:
        case "BV":
            return ((c_tilde - c) / (1 - c)) ** 0.5 * (1 / (1 - R_f * dideta(c, V, params, mu_c))) * a_plus(
                c_lyte) ** 0.5 * (
                    1 + dideta(c, V, params, mu_c) / R(c, V, params, mu_c, 1) * (1 - c_lyte) * dlnadlnc(1))
        # CIET
        case "CIET":
            return (c_tilde - c) / (1 - c) * (1 / (1 - R_f * dideta(c, V, params, mu_c))) * (
                    1 - (1 - c_lyte) * iredoi(c, V, params, mu_c) * dlnadlnc(1))


def dW_hat(c, V, R_f, c_tilde, c_lyte, params, mu_c):
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
    # TODO: plug in a_plus term for BV model

    # print an array of things that influence
    match params["rxn_method"]:
        case "CIET":
            #    print("dideta", dideta(c, V, params, mu_c))
            dWdRf = dideta(c, V, params, mu_c) / (1 - R_f * dideta(c, V, params, mu_c)) ** 2 * (c_tilde - c) / (
                    1 - c) * (1 - (1 - c_lyte) * iredoi(c, V, params, mu_c) * dlnadlnc(1))
            dWdctilde = 1 / (1 - c) * (1 / (1 - R_f * dideta(c, V, params, mu_c))) * (
                    1 - (1 - c_lyte) * iredoi(c, V, params, mu_c) * dlnadlnc(1))
            dWdclyte = iredoi(c, V, params, mu_c) * dlnadlnc(1) * (c_tilde - c) / (1 - c) * (
                    1 / (1 - R_f * dideta(c, V, params, mu_c)))
        #    print("redcurrentfrac", iredoi(c, V, params, mu_c, c_lyte))
        case "BV":
            #         print("DWDRF", dideta(c, V, params, mu_c)/(1-R_f*dideta(c, V, params, mu_c))**2)
            dWdRf = dideta(c, V, params, mu_c) / (1 - R_f * dideta(c, V, params, mu_c)) ** 2 * (
                    (c_tilde - c) / (1 - c)) ** 0.5 * a_plus(c_lyte) ** 0.5 * (
                            1 + dideta(c, V, params, mu_c) / R(c, V, params, mu_c, 1) * (1 - c_lyte) * dlnadlnc(1))
            dWdctilde = 0.5 / np.sqrt((c_tilde - c) * (1 - c)) * (
                    1 / (1 - R_f * dideta(c, V, params, mu_c))) * a_plus(c_lyte) ** 0.5 * (
                                1 + dideta(c, V, params, mu_c) / R(c, V, params, mu_c, 1) * (1 - c_lyte) * dlnadlnc(1))
            dWdclyte = (0.5 * a_plus(c_lyte) ** 0.5 / c_lyte * dlnadlnc(c_lyte) * (
                        1 + dideta(c, V, params, mu_c) / R(c, V, params, mu_c, 1) * (1 - c_lyte) * dlnadlnc(
                    1)) - a_plus(c_lyte) ** 0.5 * dideta(c, V, params, mu_c) / R(c, V, params, mu_c, 1) * dlnadlnc(
                1)) * ((c_tilde - c) / (1 - c)) ** 0.5 * (
                               1 / (1 - R_f * dideta(c, V, params, mu_c)))
    output = np.vstack((np.vstack((dWdRf, dWdctilde)), dWdclyte))
    return output


def dideta(c, mures, params, mu_c):
    """dideta for some overpotential and some reaction rate"""
    muh = np.reshape(mu_c(c, params["muR_ref"]), [1, -1])
    eta = muh - mures
    etaf = eta - np.log(c)
    #   print("eta", eta, etaf)
    match params["rxn_method"]:
        case "BV":
            alpha = 0.5
            # it's actually dhdeta for BV, too laz to write anotuerh f(x)
            out = params["k0"] * (1 - c) ** 0.5 * c ** 0.5 * (
                    -alpha * np.exp(-alpha * eta) - (1 - alpha) * np.exp((1 - alpha) * eta))
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


def W_obj(phi_offset_c, c_c, c_a, phi, params_c, params_a, c_lyte):
    # all in units of A/m^2
    """Solves for the phi_offset_c under the constant current constraint"""
    return params_c['f'] * R(c_c, phi_offset_c, params_c, Tesla_NCA_Si, c_lyte) + params_a['f'] * R(c_a,
                                                                                                    phi_offset_c + phi,
                                                                                                    params_a,
                                                                                                    Tesla_graphite,
                                                                                                    c_lyte)


def dlnadlnc(c_lyte):
    """Returns thermodynamic factor"""
    return 601 / 620 - 24 / 31 * 0.5 * c_lyte ** (0.5) + 100164 / 96875 * 1.5 * c_lyte ** (1.5)


def a_plus(c_lyte):
    """Returns activity coefficient"""
    return c_lyte ** (601 / 620) * np.exp(-1299 / 5000 - 24 / 31 * c_lyte ** (0.5) + 100164 / 96875 * c_lyte ** (1.5))


def current_magnitude(phi_offset_c, c_c, c_a, phi, params_c, params_a, c_lyte):
    # all in units of A/m^2
    """Solves for the cell level current magnitude, useful for calculating the error of the y matrix"""
    return np.abs(params_c['f'] * R(c_c, phi_offset_c, params_c, Tesla_NCA_Si, c_lyte))


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


def W_initial(c_c, c_a, mu, params_c, params_a, c_lyte):
    """finds initial values of phi while keeping the current constraint for all mu values given"""
    mu_value = np.zeros(len(c_c))
    R_value = np.zeros(len(c_c))
    for i in range(len(c_c)):
        #  opt = minimize(W_obj, -Tesla_NCA_Si(c_c[i], params_c["muR_ref"]), (c_c[i], c_a[i], mu[i], params_c, params_a, c_lyte))
        opt = fsolve(W_obj, Tesla_NCA_Si(c_c[i], params_c["muR_ref"]), (c_c[i], c_a[i], mu[i], params_c, params_a, c_lyte))
        mu_value[i] = opt[0]
    #  mu_value[i] = opt.x
    #  print("etac", Tesla_NCA_Si(c_c[i], params_c["muR_ref"]), Tesla_NCA_Si(c_c[i], params_c["muR_ref"])-mu_value[i]-np.log(c_c[i]), "Rxn", R(c_c[i], mu_value[i], params_c, Tesla_NCA_Si))
    #  print("etaa", Tesla_graphite(c_a[i], params_a["muR_ref"]), Tesla_graphite(c_a[i], params_a["muR_ref"])-(mu_value[i]+mu[i])-np.log(c_a[i]), "Rxn", R(c_a[i], mu_value[i]+mu[i], params_a, Tesla_graphite))

    R_value = current_magnitude(mu_value, c_c, c_a, mu, params_c, params_a, c_lyte)
    return mu_value, R_value


def W(deg_params, c_c, c_a, params_c, params_a, voltage_range):
    """solves overall W from the linear weight equation"""
    # (R_f_c, c_tilde_c, R_f_a, c_tilde_a, c_lyte) = deg_params
    R_f_c = deg_params[0]
    c_tilde_c = deg_params[1]
    R_f_a = deg_params[2]
    c_tilde_a = deg_params[3]
    c_lyte = deg_params[4]
    V_c, R_value = W_initial(c_c, c_a, voltage_range, params_c, params_a, 1)
    V_a = V_c + voltage_range

    W_c_hat = W_hat(c_c, V_c, R_f_c, c_tilde_c, c_lyte, params_c, Tesla_NCA_Si)
    W_a_hat = W_hat(c_a, V_a, R_f_a, c_tilde_a, c_lyte, params_a, Tesla_graphite)
    # we should have different mures
    dideta_c_a = dideta(c_c, V_c, params_c, Tesla_NCA_Si) / dideta(c_a, V_a, params_a, Tesla_graphite)
    f_c_a = params_c["f"] / params_a["f"]

    W = (W_c_hat + W_a_hat * f_c_a * dideta_c_a) / (1 + f_c_a * dideta_c_a)
    W[np.isnan(W)] = 1e8
    return W


def dWdtheta(deg_params, c_c, c_a, V_c, V_a, params_c, params_a):
    """solves overall dW/dtheta (sensitivity matrix) from weighing equation"""
    # (R_f_c, c_tilde_c, R_f_a, c_tilde_a, c_lyte) = deg_params
    R_f_c = deg_params[0]
    c_tilde_c = deg_params[1]
    R_f_a = deg_params[2]
    c_tilde_a = deg_params[3]
    c_lyte = deg_params[4]
    dW_c_hat = dW_hat(c_c, V_c, R_f_c, c_tilde_c, c_lyte, params_c, Tesla_NCA_Si)
    # this is only from the cathode, so in the full cell it doesn't affect the anode rows
    dW_c_hat = np.insert(dW_c_hat, [2, 2], np.zeros((2, len(c_c))), axis=0)
    dW_a_hat = dW_hat(c_a, V_a, R_f_a, c_tilde_a, c_lyte, params_a, Tesla_graphite)
    # this is only from the anode, so in the full cell it doesn't affect the cathode rows
    dW_a_hat = np.insert(dW_a_hat, [0, 0], np.zeros((2, len(c_c))), axis=0)
    # print("dW", dW_c_hat, dW_a_hat)
    # we should have different mures
    dideta_c_a = dideta(c_c, V_c, params_c, Tesla_NCA_Si) / dideta(c_a, V_a, params_a, Tesla_graphite)
    f_c_a = params_c["f"] / params_a["f"]
    dWdtheta = (dW_c_hat + dW_a_hat * f_c_a * dideta_c_a) / (1 + f_c_a * dideta_c_a)
    return dWdtheta

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

def model(theta, c_c, c_a, params_c, params_a, voltage_range):
    R_f_c = theta[0]
    c_tilde_c = theta[1]
    R_f_a = theta[2]
    c_tilde_a = theta[3]
    c_lyte = theta[4]

    V_c, R_value = W_initial(c_c, c_a, voltage_range, params_c, params_a, 1)
    V_a = V_c + voltage_range
    W_c_hat = W_hat(c_c, V_c, R_f_c, c_tilde_c, c_lyte, params_c, Tesla_NCA_Si)
    W_a_hat = W_hat(c_a, V_a, R_f_a, c_tilde_a, c_lyte, params_a, Tesla_graphite)
    # we should have different mures
    dideta_c_a = dideta(c_c, V_c, params_c, Tesla_NCA_Si) / dideta(c_a, V_a, params_a, Tesla_graphite)
    f_c_a = params_c["f"] / params_a["f"]
    W = (W_c_hat + W_a_hat * f_c_a * dideta_c_a) / (1 + f_c_a * dideta_c_a)
    W[np.isnan(W)] = 1e8
    return W

def lnlike(theta, c_c, c_a, params_c, params_a, y, I_err, pulse_range):
    voltage_range = -Tesla_NCA_Si(c_c, params_c["muR_ref"]) + Tesla_graphite(c_a, params_a["muR_ref"]) + pulse_range
    _, R_value = W_initial(c_c, c_a, voltage_range, params_c, params_a, 1)
    # sigma_y should scale iwth the inverse of the current magnitude
    W_range = W(theta, c_c, c_a, params_c, params_a, voltage_range)[0]
    err_y = I_err * np.abs(np.divide((1 - W_range), R_value))

    LnLike = -0.5 * np.sum(np.divide(y-model(theta, c_c, c_a, params_c, params_a, voltage_range), err_y) ** 2)
    return LnLike

def lnprior(theta, deg_params_range):
    R_f_c = theta[0]
    c_tilde_c = theta[1]
    R_f_a = theta[2]
    c_tilde_a = theta[3]
    c_lyte = theta[4]
    if (R_f_c > deg_params_range[0][0]) & (R_f_a > deg_params_range[0][4]) & (R_f_c < deg_params_range[0][1]) & (R_f_a < deg_params_range[0][5]) & (c_tilde_c > deg_params_range[0][2]) & (c_tilde_c < deg_params_range[0][3]) & (c_tilde_a > deg_params_range[0][6]) & (c_tilde_a < deg_params_range[0][7]) & (c_lyte > deg_params_range[0][8]) & (c_lyte < deg_params_range[0][9]):
        return 0.0
    else:
        return -np.inf

def lnprob(theta, c_c, c_a, params_c, params_a, y, I_err, pulse_range, deg_params_range):
    lp = lnprior(theta, deg_params_range)
    return lp + lnlike(theta, c_c, c_a, params_c, params_a, y, I_err, pulse_range) #recall if lp not -inf, its 0, so this just returns likelihood

# Automated windowing procedure following Sokal (1989)
def auto_window(taus, c):
    m = np.arange(len(taus)) < c * taus
    if np.any(m):
        return np.argmin(m)
    return len(taus) - 1


# Following the suggestion from Goodman & Weare (2010)
def autocorr_gw2010(y, c=5.0):
    f = autocorr_func_1d(np.mean(y, axis=0))
    taus = 2.0 * np.cumsum(f) - 1.0
    window = auto_window(taus, c)
    return taus[window]


def autocorr_new(y, c=5.0):
    f = np.zeros(y.shape[1])
    for yy in y:
        f += autocorr_func_1d(yy)
    f /= len(y)
    taus = 2.0 * np.cumsum(f) - 1.0
    window = auto_window(taus, c)
    return taus[window]

def next_pow_two(n):
    i = 1
    while i < n:
        i = i << 1
    return i
def autocorr_func_1d(x, norm=True):
    x = np.atleast_1d(x)
    if len(x.shape) != 1:
        raise ValueError("invalid dimensions for 1D autocorrelation function")
    n = next_pow_two(len(x))

    # Compute the FFT and then (from that) the auto-correlation function
    f = np.fft.fft(x - np.mean(x), n=2 * n)
    acf = np.fft.ifft(f * np.conjugate(f))[: len(x)].real
    acf /= 4 * n

    # Optionally normalize
    if norm:
        acf /= acf[0]

    return acf

