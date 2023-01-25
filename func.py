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


def transform_json(df):
    values=df['ExportedModel']['hyper_values']
    transformed_dict = ""
    for key, value in values.items():
        original_value = value[0]
        lower_value = round(original_value * 0.8, 4)
        higher_value = round(original_value * 1.2, 4)
        transformed_dict += key + " = c(" + str(lower_value) + "," + str(higher_value) + "),\n"
    return transformed_dict



def display_dict(data):
    result = ""
    for key, value in data.items():
        for sub_key, sub_value in value.items():
            original_value = sub_value
            lower_value = round(original_value * 0.8, 4)
            higher_value = round(original_value * 1.2, 4)
            result += f"{key}_{sub_key} = c{lower_value,higher_value},\n"
    return result


def generate_robyn_inputs(date, output, media, organic, start_date, end_date):
    if "revenue" in output:
        output_type='revenue'
    else:
        output_type='conversions'
    script = "InputCollect <- robyn_inputs(\n"
    script += f"  dt_input = data.table::fread('./dataset.csv')\n"
    script += "  ,dt_holidays = dt_prophet_holidays\n"
    script += f"  ,date_var = '{date}'\n"
    script += f"  ,dep_var = '{output}'\n"
    script += f"  ,dep_var_type = '{output_type}'\n"
    script += "  ,prophet_vars = c('season', 'holiday')\n"
    script += "  ,prophet_country = 'US'\n"
    script += f"  ,paid_media_vars = c({', '.join(media)})\n"
    script += f"  ,paid_media_spends = c({', '.join(media)})\n"
    script += f"  ,organic_vars = c({', '.join(organic)})\n"
    script += "  ,context_vars = c('apertura_monomarca')\n"
    script += "  ,context_sign = c('positive')\n"
    script += f"  ,window_start = '{start_date}'\n"
    script += f"  ,window_end = '{end_date}'\n"
    script += "  ,adstock = 'weibull_pdf'\n"
    script += ")"
    return script
