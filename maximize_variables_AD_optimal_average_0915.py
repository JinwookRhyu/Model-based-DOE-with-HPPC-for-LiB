import numpy as np
import matplotlib.pyplot as plt
from math import ceil, floor, pow
from scipy import constants
from scipy.special import erf
from scipy.optimize import minimize
from scipy import optimize
from itertools import product
from scipy.optimize import fsolve
from datetime import datetime
import time

R_f_c_range = np.array([0, 10]) # Range for R_f_c (lb, ub)
c_tilde_c_range = np.array([0.5, 1]) # Range for c_tilde_c (lb, ub)
R_f_a_range = np.array([0, 10]) # Range for R_f_a (lb, ub)
c_tilde_a_range = np.array([0.5, 1]) # Range for c_tilde_a (lb, ub)
c_lyte_range = np.array([0.7, 1]) # Range for c_lyte (lb, ub)

deg_params = np.reshape(np.concatenate((R_f_c_range, c_tilde_c_range, R_f_a_range, c_tilde_a_range, c_lyte_range), axis=0), (1, 10))
Nrange = np.arange(5, 9)

tpe = "D" # Which optimal? A/D/E
rxn_method = "CIET" # Which rxn model? BV/CIET
sig_digits = 4 # Number of significant digits to use for \Sigma_y
V_limit = 0.050 # Lower limit for delta_V in [V]
I_err = 0.001 # Measurement error of current in [A/m^2]. Surface area ~ 0.063 [m^2]
#TODO: Check I_err value of the experimental equipment [A/m^2]


str_deg_params = str(deg_params[0][0]) + "_" + str(deg_params[0][1]) + "_" + str(deg_params[0][2]) + "_" + str(deg_params[0][3]) + "_" + str(deg_params[0][4]) + "_" + str(deg_params[0][5]) + "_" + str(deg_params[0][6]) + "_" + str(deg_params[0][7]) + "_" + str(deg_params[0][8]) + "_" + str(deg_params[0][9])
print(str_deg_params)

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

    match params["rxn_method"]:
        case "BV":
            return ((c_tilde - c) / (1 - c)) ** 0.5 * (1 / (1 - R_f * dideta(c, V, params, mu_c))) * a_plus(c_lyte) ** 0.5 * (
                        1 + dideta(c, V, params, mu_c) / R(c, V, params, mu_c, 1) * (1 - c_lyte) * dlnadlnc(1))
        case "CIET":
            return (c_tilde - c) / (1 - c) * (1 / (1 - R_f * dideta(c, V, params, mu_c))) * (
                        1 - (1 - c_lyte) * iredoi(c, V, params, mu_c) * dlnadlnc(1))


def W_hat_averaged(c, V, R_f, c_tilde, c_lyte, params, mu_c):
    """Defines W_hat value for half cell electrode:
    Inputs: c: state of charge
    V: voltage
    R_f: film resistance
    c_tilde: rescaled capacity
    c_lyte: electrolyte concentration
    params: parameters related to electrode
    mu_c: functional form of OCV
    Returns: W_hat value for this half cell"""
    # print an array of things that influence
    di = dideta(c, V, params, mu_c)
    # print("LI di", dideta(c, V, params, mu_c), "iredoi", iredoi(c, V, params, mu_c))
    av_W_R_f = np.log((1-di*R_f[:,1])/(1-di*R_f[:,0]))/(di*(R_f[:,0]-R_f[:,1]))
    R_f_array = np.linspace(R_f[:,0], R_f[:,1])
    c_lyte_array = np.linspace(c_lyte[:,0], c_lyte[:,1])
    c_tilde_array = np.linspace(c_tilde[:,0], c_tilde[:,1])
   # print("averaged What R_f", np.average(W_hat(c, V, R_f_array, 1, 1, params, mu_c), 0))
   # print("averaged What c_lyte", np.average(W_hat(c, V, 0, 1, c_lyte_array, params, mu_c), 0))
   # print("averaged What c_tilde", np.average(W_hat(c, V, 0, c_tilde_array, 1, params, mu_c), 0))
    match params["rxn_method"]:
        case "CIET":
            alpha = 1
            iredoi_val = iredoi(c, V, params, mu_c)
            #av_W_c_lyte = ((c_lyte[:,1]+0.5*dlnadlnc(1)*(c_lyte[:,1]-2)*c_lyte[:,1]*iredoi_val)-(c_lyte[:,0]+0.5*dlnadlnc(1)*(c_lyte[:,0]-2)*c_lyte[:,0]*iredoi_val))/(c_lyte[:,1]-c_lyte[:,0])
            av_W_c_lyte = 1 + 0.5 * dlnadlnc(1) * (-2 + c_lyte[:,1] + c_lyte[:,0])* iredoi_val
        case "BV":
            alpha = 0.5
            kinetic_h = dideta(c, V, params, mu_c) / R(c, V, params, mu_c, 1) *dlnadlnc(1)
           # av_W_c_lyte = (((-0.4*c_lyte[:,1]**2.5-2/3*c_lyte[:,1]**1.5)*dideta(c, V, params, mu_c) / R(c, V, params, mu_c, 1) *dlnadlnc(1)+2/3*c_lyte[:,1]**1.5)-((-0.4*c_lyte[:,0]**2.5-2/3*c_lyte[:,0]**1.5)*dideta(c, V, params, mu_c) / R(c, V, params, mu_c, 1) *dlnadlnc(1)+2/3*c_lyte[:,0]**1.5))/(c_lyte[:,1]-c_lyte[:,0])
            av_W_c_lyte = (0.4*c_lyte[:,0]**2.5*kinetic_h - 0.4*c_lyte[:,1]**2.5*kinetic_h - 2/3*c_lyte[:,0]**1.5*(1+kinetic_h) + 2/3*c_lyte[:,1]**1.5*(1+kinetic_h))/(c_lyte[:,1]-c_lyte[:,0])
    #        print("rxn", R(c, V, params, mu_c, 1))
    av_W_c_tilde = (1+alpha)**(-1)*(1-c)**(-alpha)*((c_tilde[:,1]-c)**(1+alpha)-(c_tilde[:,0]-c)**(1+alpha))/(c_tilde[:,1]-c_tilde[:,0])
    return av_W_R_f, av_W_c_tilde, av_W_c_lyte
   


def dW_hat_averaged(c, V, R_f, c_tilde, c_lyte, params, mu_c):
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
    av_W_R_f, av_W_c_tilde, av_W_c_lyte =  W_hat_averaged(c, V, R_f, c_tilde, c_lyte, params, mu_c)
   # print("hi", av_W_R_f, av_W_c_tilde, av_W_c_lyte)
   # print("non av vals",  1/ (R_f[:,1]-R_f[:,0]) * (W_hat(c, V, R_f[:,1], 1, 1, params, mu_c) - W_hat(c, V, R_f[:,0], 1, 1, params, mu_c)), 1/ (c_tilde[:,1]-c_tilde[:,0]) * (W_hat(c, V, 0, c_tilde[:,1], 1, params, mu_c) - W_hat(c, V, 0, c_tilde[:,0], 1, params, mu_c)), 1/ (c_lyte[:,1]-c_lyte[:,0]) * (W_hat(c, V, 0, 1, c_lyte[:,1], params, mu_c) - W_hat(c, V, 0, 1, c_lyte[:,0], params, mu_c)))
    dWdRf = av_W_c_lyte * av_W_c_tilde / (R_f[:,1]-R_f[:,0]) * (W_hat(c, V, R_f[:,1], 1, 1, params, mu_c) - W_hat(c, V, R_f[:,0], 1, 1, params, mu_c))
    dWdctilde = av_W_R_f * av_W_c_lyte / (c_tilde[:,1]-c_tilde[:,0]) * (W_hat(c, V, 0, c_tilde[:,1], 1, params, mu_c) - W_hat(c, V, 0, c_tilde[:,0], 1, params, mu_c))
    dWdclyte = av_W_R_f * av_W_c_tilde / (c_lyte[:,1]-c_lyte[:,0]) * (W_hat(c, V, 0, 1, c_lyte[:,1], params, mu_c) - W_hat(c, V, 0, 1, c_lyte[:,0], params, mu_c))
    output = np.vstack((np.vstack((dWdRf, dWdctilde)), dWdclyte))
   # print("out", output)
    return output

 
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
            dWdclyte = (0.5 * a_plus(c_lyte) ** 0.5 / c_lyte * dlnadlnc(c_lyte) * (1 + dideta(c, V, params, mu_c) / R(c, V, params, mu_c, 1) * (1 - c_lyte) * dlnadlnc(1)) - a_plus(c_lyte) ** 0.5 * dideta(c, V, params, mu_c) / R(c, V, params, mu_c, 1) * dlnadlnc(1))  * ((c_tilde - c) / (1 - c)) ** 0.5 * (
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
    return params_c['f'] * R(c_c, phi_offset_c, params_c, Tesla_NCA_Si, c_lyte) + params_a['f'] * R(c_a, phi_offset_c + phi,
                                                                                            params_a, Tesla_graphite, c_lyte)


    

def dlnadlnc(c_lyte):
    """Returns thermodynamic factor"""    
    return 601/620 - 24 / 31 * 0.5 * c_lyte ** (0.5) + 100164 / 96875 * 1.5 * c_lyte ** (1.5)


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
    etaf = eta - np.log(c/a_plus(c_lyte))
    match params["rxn_method"]:
        case "BV":
            # this is only h
            rxn = params["k0"] * (1 - c) ** 0.5 * c ** 0.5 * a_plus(c_lyte) ** 0.5 * (np.exp(-0.5 * eta) - np.exp(0.5 * eta))
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

def W_averaged(deg_params, c_c, c_a, V_c, V_a, params_c, params_a):
    """solves overall W from the linear weight equation"""
    # (R_f_c, c_tilde_c, R_f_a, c_tilde_a, c_lyte) = deg_params
    R_f_c = deg_params[:, 0:2]
    c_tilde_c = deg_params[:, 2:4]
    R_f_a = deg_params[:, 4:6]
    c_tilde_a = deg_params[:, 6:8]
    c_lyte = deg_params[:, 8:10]
    av_W_R_f, av_W_c_tilde, av_W_c_lyte = W_hat_averaged(c_c, V_c, R_f_c, c_tilde_c, c_lyte, params_c, Tesla_NCA_Si)
    W_c_hat = av_W_R_f*av_W_c_tilde*av_W_c_lyte 
    av_W_R_f, av_W_c_tilde, av_W_c_lyte = W_hat_averaged(c_a, V_a, R_f_a, c_tilde_a, c_lyte, params_a, Tesla_graphite)
    W_a_hat = av_W_R_f*av_W_c_tilde*av_W_c_lyte 
    # we should have different mures
    dideta_c_a = dideta(c_c, V_c, params_c, Tesla_NCA_Si) / dideta(c_a, V_a, params_a, Tesla_graphite)
    f_c_a = params_c["f"] / params_a["f"]
    W = (W_c_hat + W_a_hat * f_c_a * dideta_c_a) / (1 + f_c_a * dideta_c_a)
    W[np.isnan(W)] = 1e8
    return W


def dWdtheta_averaged(deg_params, c_c, c_a, V_c, V_a, params_c, params_a):
    """solves overall dW/dtheta_averaged (sensitivity matrix) from weighing equation
    inputs are the min and max values of each degradation parameter, which are input slighlty different from the normal way"""
    # (R_f_c, c_tilde_c, R_f_a, c_tilde_a, c_lyte) = deg_params
    R_f_c = deg_params[:, 0:2]
    c_tilde_c = deg_params[:, 2:4]
    R_f_a = deg_params[:, 4:6]
    c_tilde_a = deg_params[:, 6:8]
    c_lyte = deg_params[:, 8:10]
#    R_f_c = deg_params[:, [0]]
#    c_tilde_c = deg_params[:, [2]]
#    R_f_a = deg_params[:, [4]]
#    c_tilde_a = deg_params[:, [6]]
#    c_lyte = deg_params[:, [8]]
    dW_c_hat = dW_hat_averaged(c_c, V_c, R_f_c, c_tilde_c, c_lyte, params_c, Tesla_NCA_Si)
  #  print("dWc", dW_c_hat)
  #  print("test", test_derivatives(c_c, V_c, R_f_c, c_tilde_c, c_lyte, params_c, Tesla_NCA_Si))
    # this is only from the cathode, so in the full cell it doesn't affect the anode rows
    dW_c_hat = np.insert(dW_c_hat, [2, 2], np.zeros((2, N)), axis=0)
    dW_a_hat = dW_hat_averaged(c_a, V_a, R_f_a, c_tilde_a, c_lyte, params_a, Tesla_graphite)
    # this is only from the anode, so in the full cell it doesn't affect the cathode rows
    dW_a_hat = np.insert(dW_a_hat, [0, 0], np.zeros((2, N)), axis=0)
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


from math import ceil, floor, pow


# Function to round - off the number
def Round_off(N, n):
    """To fix the number of significant digits of matrix N to n"""
    i = np.zeros(N.shape)
    sgn = np.sign(N)
    N_abs = np.abs(N)
    A = N_abs
    if np.max(N_abs) == 0:
        return i

    if np.max(N_abs) >= 1:
        while (np.max(A) >= 1):
            A = A / 10
            i = i + (A >= 0.1)
    A = N_abs

    if np.min(N_abs[np.nonzero(N_abs)]) <= 1:
        while (np.min(A[np.nonzero(A)]) <= 1):
            A = A * 10
            i = i - (A <= 1)

    d = n - i
    A = np.multiply(N_abs, 10 ** d)

    A = np.multiply(sgn, np.divide(np.round(A), 10 ** d))
    return A


# now i have to write a dynamic programming htingy :/

def optimize_function(opt_params, N, deg_params, params_c, params_a, tpe):
    """N is the maximum number of pulses we want, c_range is the discretized c values, V_range i the discretized V values.
    Matrix W: (degradation parameters, c_values, V_values)
    opt_params: optimizing over (soc_array, voltage_array), both arrays of size N"""
    # generate matrix of N pulses and M dedgradation mechanisms, N*M
    #   (c_range, V_range) = opt_params # both size (N*1, N*1)
    c_range = 0.85 + np.cumsum(opt_params[:N]) / np.sum(opt_params[:N+1]) * (0.4 - 0.85)
    # mu_range = get_muR_from_OCV(opt_params[N:], 0)
    pulse_range = opt_params[-N:]
    if any(np.abs(get_OCV_from_muR(pulse_range, 0)) < V_limit):
        phi = 1e50
    else:
        # this is only the pulse value. we should do OCV + pulse value
        c_c = c_range
        c_a = params_a["c0"] - params_c["p"] / params_a["p"] * (c_c - params_c["c0"])
        # first, solve for the V_offsets
        # all the V's are actually phis
        # get the voltage pulse values
        voltage_range = -Tesla_NCA_Si(c_c, params_c["muR_ref"]) + Tesla_graphite(c_a, params_a["muR_ref"]) + pulse_range
        # solve for the initial voltage pulses
        mu_range_c, _ = W_initial(c_c, c_a, voltage_range, params_c, params_a, 0.5 * (deg_params[:, [8]][0] + deg_params[:, [9]][0])[0])
        mu_range_a = mu_range_c + voltage_range
        # using the current constraint equations, solve for both cathode and anode
        # how do we know how much degradation is happening??? help.
        # stack degradatio parameters so that we have a 5*1*1 matrix, where each set only has one degradation parameter
        #S = dWdtheta(deg_params, c_c, c_a, mu_range_c, mu_range_a, params_c, params_a)
        S = dWdtheta_averaged(deg_params, c_c, c_a, mu_range_c, mu_range_a, params_c, params_a)
        # sigma_y should scale iwth the inverse of the current magnitude
        W_range = W_averaged(deg_params, c_c, c_a, mu_range_c, mu_range_a, params_c, params_a)[0]
        _, R_value = W_initial(c_c, c_a, voltage_range, params_c, params_a, 1)
        err_y = I_err * np.abs(np.divide((1 - W_range), R_value))
        #rescale_current = R_value / np.max(R_value)
        #sigma_y_inv = np.diag(rescale_current ** 2)
        sigma_y_inv = np.diag(err_y ** (-2))
        sigma_inv = np.dot(np.dot(Round_off(S, sig_digits), Round_off(sigma_y_inv, sig_digits)), Round_off(S, sig_digits).T)
        # check if determinant is zero
        if np.linalg.det(sigma_inv) != 0:
            sigma = np.linalg.inv(sigma_inv)
            # sigma = np.linalg.inv(np.dot(S, S.T))

            if tpe == "D":
                phi = (np.linalg.det(sigma))
            elif tpe == "A":
                phi = np.trace(sigma) / len(deg_params)
            elif tpe == "E":
                phi = np.max(np.linalg.eigvals(sigma))
            if np.any(np.diag(sigma) < 0):
                phi = 1e50
        else:
            phi = 1e50
    phi = np.log(phi)
    print("Pulse at c = " + str(c_range) + " with eta = " + str(pulse_range) + " resulted det = " + str(phi))

    return phi


def bound_error(opt_params, N, deg_params, params_c, params_a, tpe):
    """N is the maximum number of pulses we want, c_range is the discretized c values, V_range i the discretized V values.
    Matrix W: (degradation parameters, c_values, V_values)
    opt_params: optimizing over (soc_array, voltage_array), both arrays of size N"""
    # generate matrix of N pulses and M dedgradation mechanisms, N*M
    #   (c_range, V_range) = opt_params # both size (N*1, N*1)
    c_range = 0.85 + np.cumsum(opt_params[:N]) / np.sum(opt_params[:N+1]) * (0.4 - 0.85)
    # mu_range = get_muR_from_OCV(opt_params[N:], 0)
    pulse_range = opt_params[-N:]
    # this is only the pulse value. we should do OCV + pulse value
    c_c = c_range
    c_a = params_a["c0"] - params_c["p"] / params_a["p"] * (c_c - params_c["c0"])
    # first, solve for the V_offsets
    # all the V's are actually phis
    # get the voltage pulse values
    voltage_range = -Tesla_NCA_Si(c_c, params_c["muR_ref"]) + Tesla_graphite(c_a, params_a["muR_ref"]) + pulse_range
    # solve for the initial voltage pulses
    mu_range_c, _ = W_initial(c_c, c_a, voltage_range, params_c, params_a, 0.5 * (deg_params[:, [8]][0] + deg_params[:, [9]][0])[0])
    mu_range_a = mu_range_c + voltage_range
    # using the current constraint equations, solve for both cathode and anode
    # how do we know how much degradation is happening??? help.
    # stack degradatio parameters so that we have a 5*1*1 matrix, where each set only has one degradation parameter
    S = dWdtheta_averaged(deg_params, c_c, c_a, mu_range_c, mu_range_a, params_c, params_a)
    # sigma_y should scale iwth the inverse of the current magnitude
    W_range = W_averaged(deg_params, c_c, c_a, mu_range_c, mu_range_a, params_c, params_a)[0]
    _, R_value = W_initial(c_c, c_a, voltage_range, params_c, params_a, 1)
    err_y = I_err * np.abs(np.divide((1 - W_range), R_value))
    # rescale_current = R_value / np.max(R_value)
    # sigma_y_inv = np.diag(rescale_current ** 2)
    sigma_y_inv = np.diag(err_y ** (-2))
    sigma_inv = np.dot(np.dot(Round_off(S, sig_digits), Round_off(sigma_y_inv, sig_digits)), Round_off(S, sig_digits).T)
    sigma = np.linalg.inv(sigma_inv)
    return np.sqrt(np.diag(sigma))


# set degradation parameters

# ref_params = np.reshape(np.array([0,1,0,1,1]), (1,5))

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

# number of pulses
# N = 8

# input parameters for electrodes
params_c = {'rxn_method': rxn_method, 'k0': 1, 'lambda': 5, 'f': f_c, 'p': p_c, 'c0': c_s_0_c, 'mu': Tesla_NCA_Si,
            'muR_ref': muR_ref_c}
params_a = {'rxn_method': rxn_method, 'k0': 1, 'lambda': 5, 'f': f_a, 'p': p_a, 'c0': c_s_0_a, 'mu': Tesla_graphite,
            'muR_ref': muR_ref_a}
# out = optimization_protocol(N, ref_params, params_c, params_a, tpe)

# print("Optimum Parameters: c: ", out[:N], "; V: ", get_OCV_from_muR(out[N:], 0), "; value: ", optimize_function(out, N, ref_params, params_c, params_a, tpe))


error_save = np.zeros((len(Nrange), 5))
optimization_save = np.ones((len(Nrange), 2*Nrange[-1]))*np.nan

for i in range(len(Nrange)):
    # optimal
    N = Nrange[i]
    # we want N_pulses
    # generate initial condition from "best guess"
    # initial_condition
    mu_guess = get_muR_from_OCV(0.01 * np.ones(N), 0)
    x0 = np.concatenate((0.5 * np.ones([N+1]), mu_guess))  # initial_condition
    # set bounds
    l = (0.0001, 1)  # for SOC
    m = (get_muR_from_OCV(0.2, 0), get_muR_from_OCV(-0.2, 0))  # for pulse sizes
    bnds = (l,) * (N+1) + (m,) * N
    #  opt = minimize(optimize_function, x0, (N, ref_params, deg_params, params_c, params_a), bounds = bnds)
    # use a global optimization algorithm
    opt = optimize.dual_annealing(optimize_function, bnds, args=(N, deg_params, params_c, params_a, tpe))
    out = opt.x
    c_range = 0.85 + np.cumsum(out[:N]) / np.sum(out[:N+1]) * (0.4 - 0.85)
    optimization_save[i, :N] = c_range
    optimization_save[i, Nrange[-1]:Nrange[-1] + N] = get_OCV_from_muR(out[-N:], 0)
    print("N: ", N, "   Optimum Parameters: c: ", c_range, "; V: ", get_OCV_from_muR(out[-N:], 0), "; value: ",
          optimize_function(out, N, deg_params, params_c, params_a, tpe))
    print("Error: ", bound_error(out, N, deg_params, params_c, params_a, tpe))
    error_save[i, :] = bound_error(out, N, deg_params, params_c, params_a, tpe)

np.savetxt("error_bound_optimal_output_averaged_N5_similar/optimized_output_" + params_c["rxn_method"] + "_" + str(tpe) + "_" + str(int(1000*V_limit)) + "mV_" + str(str_deg_params) + ".txt", np.concatenate((Nrange.reshape(-1,1),optimization_save),axis = 1))
np.savetxt("error_bound_optimal_output_averaged_N5_similar/error_bound_" + params_c["rxn_method"] + "_" + str(tpe) + "_" + str(int(1000*V_limit)) + "mV_"+ str(str_deg_params) + ".txt", np.concatenate((Nrange.reshape(-1,1),error_save),axis = 1))



print("True values for degradation parameters = " + str(deg_params))


