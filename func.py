import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import weibull_min

def create_number_list(dataframe, column_name):
    min_val = dataframe[column_name].min()
    max_val = dataframe[column_name].max()
    number_list = list(range(int(min_val), int(max_val)+1))
    number_df = pd.DataFrame(number_list, columns=["spent"])

    return number_df



def saturation_robyn(x, coeff, alpha, gamma):
    return coeff * (x ** alpha / (x ** alpha + gamma ** alpha))



def saturation_hill(x, alpha, gamma, x_marginal=None):
        inflexion = (np.min(x) * (1 - gamma)) + (np.max(x) * gamma)
        if x_marginal is None:
            x_scurve = x**alpha / (x**alpha + inflexion**alpha)
        else:
            x_scurve = x_marginal**alpha / (x_marginal**alpha + inflexion**alpha)
        return x_scurve


def adstock( shape, scale, windlen=None, type="pdf"):
        if windlen is None:
            windlen = len(list(range(1,101)))
            x_bin = np.arange(1, windlen + 1)
            scale_trans = round(np.quantile(x_bin, scale), 0)
            if shape == 0:
                theta_vec_cum = theta_vec = np.zeros(windlen)
            else:
                #if type.lower() == "cdf":
                    #theta_vec = np.concatenate([[1], 1 - stats.weibull_min.cdf(x_bin[:-1], shape=shape, scale=scale_trans)])
                    #theta_vec_cum = np.cumprod(theta_vec)
                if type.lower() == "pdf":
                    theta_vec_cum = _normalize(weibull_min.pdf(x_bin, c=shape, scale=scale_trans))
            #x_decayed = [_decay(x_val, x_pos, theta_vec_cum, windlen) for x_val, x_pos in zip(x, x_bin[:len(x)])]
            #x_decayed = np.sum(x_decayed, axis=0)
            x_decayed=1
        else:
            x_decayed = 1
            theta_vec_cum = 1
        return {"x": 1, "x_decayed": x_decayed, "theta_vec_cum": theta_vec_cum, 'day':x_bin}

def _normalize(x):
    min_x, max_x = np.min(x), np.max(x)
    print('max_min',min_x, max_x)
    if max_x - min_x == 0:
        return np.concatenate([[1], np.zeros(len(x) - 1)])
    else:
        return (x - min_x) / (max_x - min_x)

def _decay(x_val, x_pos, theta_vec_cum, windlen):
    x_vec = np.concatenate([np.zeros(x_pos - 1), np.full(windlen - x_pos + 1, x_val)])
    theta_vec_cum_lag = list(pd.Series(theta_vec_cum.copy()).shift(periods=x_pos-1, fill_value=0))
    x_prod = x_vec * theta_vec_cum_lag
    return x_prod