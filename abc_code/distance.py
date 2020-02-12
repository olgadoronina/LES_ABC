import abc_code.global_var as g
import numpy as np
from abc_code import utils

keys = ['uu', 'uv', 'uw', 'vu', 'vv', 'vw', 'prod']
########################################################################################################################
## Distance functions
########################################################################################################################
def calc_dist(C, dist_func):
    pdf = g.TEST_Model.sum_stat_from_C(C)
    distance = 0
    for i in range(len(pdf)):
        d = dist_func(pdf_modeled=pdf[i], key=keys[i], axis=0)
        distance += d
    return distance


def distance_sigma_pdf_LSElog(pdf_modeled, key, axis=1):
    """ Calculate statistical distance between two pdf as mean((ln(P1)-ln(P2))^2).
    :param pdf_modeled: array of modeled pdf
    :param key: tensor component(key of dict)
    :return: 1D array of calculated distance
    """
    log_modeled = utils.take_safe_log(pdf_modeled)
    dist = np.mean((log_modeled - g.sum_stat_true[key]) ** 2, axis=axis)
    return dist


def distance_sigma_pdf_L1log(pdf_modeled, key, axis=1):
    """Calculate statistical distance between two pdf as
    :param pdf_modeled: array of modeled pdf
    :return:            scalar of calculated distance
    """

    log_modeled = utils.take_safe_log(pdf_modeled)
    dist = 0.5 * np.sum(np.abs(log_modeled - g.sum_stat_true[key]), axis=axis)
    return dist


def distance_sigma_L2log(pdf_modeled, key, axis=1):
    """ Calculate statistical distance between two pdf.
    :param pdf_modeled:
    :param key:
    :param axis:
    :return:
    """
    log_modeled = utils.take_safe_log(pdf_modeled)
    dist = np.sum((log_modeled - g.sum_stat_true[key]) ** 2, axis=axis)
    return dist


#######################################################################################
#######################################################################################
def distance_production_L2log(pdf_modeled, key, axis=1):
    """ Calculate statistical distance between two pdf.
    :param pdf_modeled:
    :param key:
    :param axis:
    :return:
    """
    log_modeled = utils.take_safe_log(pdf_modeled)
    dist = np.sum((log_modeled - g.sum_stat_true) ** 2, axis=axis)
    return dist


def distance_production_LSElog(pdf_modeled, key, axis=1):
    """ Calculate statistical distance between two pdf as mean((ln(P1)-ln(P2))^2).
    :param pdf_modeled: array of modeled pdf
    :param key: tensor component(key of dict)
    :return: 1D array of calculated distance
    """
    log_modeled = utils.take_safe_log(pdf_modeled)
    dist = np.mean((log_modeled - g.sum_stat_true) ** 2, axis=axis)
    return dist


def distance_production_L1log(pdf_modeled, key, axis=1):
    """Calculate statistical distance between two pdf as
    :param pdf_modeled: array of modeled pdf
    :return:            scalar of calculated distance
    """

    log_modeled = utils.take_safe_log(pdf_modeled)
    dist = 0.5 * np.sum(np.abs(log_modeled - g.sum_stat_true), axis=axis)
    return dist
#######################################################################################
#######################################################################################


def distance_both_L2log(pdf_modeled, key, axis=1):
    """ Calculate statistical distance between two pdf.
    :param pdf_modeled:
    :param key:
    :param axis:
    :return:
    """
    log_modeled = utils.take_safe_log(pdf_modeled)
    dist = np.sum((log_modeled - g.sum_stat_true[key]) ** 2, axis=axis)
    if key == 'prod':
        dist = 3*(dist-3100)
        if dist < 0:
            print(dist)

    return dist


#
# def distance_between_pdf_KL(pdf_modeled, key, axis=1):
#     """Calculate statistical distance between two pdf as
#     the Kullback-Leibler (KL) divergence (no symmetry).
#     Function for N_params_in_task > 0
#     :param pdf_modeled: array of modeled pdf
#     :param key: tensor component(key of dict)
#     :return: 1D array of calculated distance
#     """
#
#     log_modeled = take_safe_log(pdf_modeled)
#     dist = np.sum(np.multiply(g.TEST_sp.tau_pdf_true[key], (g.TEST_sp.log_tau_pdf_true[key] - log_modeled)), axis=axis)
#
#     return dist
#
# def distance_between_pdf_LSE(pdf_modeled, key, axis=1):
#     """ Calculate statistical distance between two pdf as mean((P1-P2)^2).
#     :param pdf_modeled: array of modeled pdf
#     :param key: tensor component(key of dict)
#     :param axis: equal 1 when pdf_modeled is 2D array
#     :return: scalar or 1D array of calculated distance
#     """
#     dist = np.mean((pdf_modeled - g.TEST_sp.tau_pdf_true[key]) ** 2, axis=axis)
#     return dist
#
#
# def distance_between_pdf_L2(pdf_modeled, key, axis=1):
#     """ Calculate statistical distance between two pdf as sqrt(sum((P1-P2)^2)).
#     :param pdf_modeled: array of modeled pdf
#     :param key: tensor component(key of dict)
#     :param axis: equal 1 when pdf_modeled is 2D array
#     :return: scalar or 1D array of calculated distance
#     """
#     dist = np.sqrt(np.sum((pdf_modeled - g.TEST_sp.tau_pdf_true[key]) ** 2, axis=axis))
#     return dist