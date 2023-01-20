import numpy as np
import pandas as pd
import os
import streamlit as st


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
