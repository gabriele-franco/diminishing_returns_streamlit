import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import weibull_min
from scipy.optimize import minimize


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

def adstock_transf(x, shape, scale, windlen=None, type="pdf"):
        if windlen is None:
            windlen = len(x)
        if len(x) > 1:
            if type.lower() not in ("cdf", "pdf"):
                raise ValueError("Invalid value for `type`")
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
            x_decayed = [_decay(x_val, x_pos, theta_vec_cum, windlen) for x_val, x_pos in zip(x, x_bin[:len(x)])]
            x_decayed = np.sum(x_decayed, axis=0)
        else:
            x_decayed = x
            theta_vec_cum = 1
        return {"x": x, "x_decayed": x_decayed, "theta_vec_cum": theta_vec_cum}


def transform_json(df):
    values=df['ExportedModel']['hyper_values']
    transformed_dict = ""
    for key, value in values.items():
        original_value = value[0]
        lower_value = round(original_value * 0.8, 4)
        higher_value = round(original_value * 1.2, 4)
        transformed_dict += key + " = c(" + str(lower_value) + "," + str(higher_value) + "),\n"
    return transformed_dict



def display_dict(data,variance):
    result = ""
    for key, value in data.items():
        for sub_key, sub_value in value.items():
            original_value = sub_value
            lower_value = round(original_value * (1-(variance[key][sub_key]/100)), 4)
            higher_value = round(original_value * (1+(variance[key][sub_key]/100)), 4)
            result += f"{key}_{sub_key} = c{lower_value,higher_value},\n"
    return result


def generate_robyn_inputs(date, output, media, organic, start_date, end_date, iterations,data, variance):
    if "revenue" in output:
        output_type='revenue'
    else:
        output_type='conversions'
    hyper=display_dict(data,variance)
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
    script += ")\n"
    script += "   ##############\n"
    script += f"{hyper}\n"
    script += "   ##############\n"
    script += "InputCollect <- robyn_inputs(InputCollect = InputCollect, hyperparameters = hyperparameters)\n"
    script += "saveRDS(InputCollect, 'input.rds')\n"
    script += "   ##############\n"
    script += f"OutputModels <- robyn_run(InputCollect = InputCollect,cores = 32,iterations = {iterations},trials = 10)\n"
    script += "saveRDS(OutputModels, 'output.rds')"
    return script



def get_average_last_15_days(df, features):
    result = {}
    for feature in features:
        last_15_days = df[feature]
        last_15_days_non_zero = last_15_days[last_15_days != 0]
        average = last_15_days_non_zero.mean()
        result[feature] = average
    return result
"""
def ridge_regression_positive_coef(data,output, features_positive,media, test_size=0.2):
    # Split data into training and test sets
    X = data[features_positive]
    y = data[output]
     # Adstock transformation
    for feature in media:
        X[feature] = adstock_transf(X[feature], shape=column_values[feature]['shape'], scale=column_values[feature]['scale'])['x_decayed']
        X[feature] = saturation_hill(X[feature], alpha=column_values[feature]['alpha'], gamma=column_values[feature]['gamma'])
        
    # Create Ridge regression model
    model = Lasso(alpha=1.0, fit_intercept=True)
    
    # Force specific features to have positive coefficients
    for i, feature in enumerate(X.columns):
        if feature in features_positive:
            model.set_params(alpha=0, positive=True)
    model.fit(X, y)
    # Get R^2 and NRMSE of the model on the test set
    y_pred = model.predict(X)
    data['y_pred']=y_pred
    r2 = r2_score(y, y_pred)
    nrmse = np.sqrt(mean_squared_error(y, y_pred)) / (y.max() - y.min())
    
    return model, r2, nrmse, data"""


def measure_delayed_effect(df, output, column_b):
    delayed_effect_days_range = range(0, 30)
    max_corr = 0
    optimal_delayed_effect_days = None
    for delayed_days in delayed_effect_days_range:
        data = df[[output, column_b]]
        data['delayed_input'] = data[column_b].shift(delayed_days)
        data=data.dropna()
        corr = data[[output, 'delayed_input']].corr().iloc[0,1]
        if corr > max_corr:
            max_corr = corr
            optimal_delayed_effect_days = delayed_days

    return optimal_delayed_effect_days, max_corr


def correlation(df, x, output,params):
    data={}
    data[x]=df[x]
    data[output]=df[output]
    shape, scale = params
    x_decayed = adstock_transf(data[x], shape, scale, type="pdf")["x_decayed"]
    data['adstock']=x_decayed
    corr = data[[output, 'adstock']].corr().iloc[0,1]
    return -corr

def measure_delayed_effect(df, output, column_b):
    def adstock_correlation(params, df, output, column_b):
        shape, scale = params
        x_decayed = adstock_transf(df[column_b], shape, scale)["x_decayed"]
        corr = np.corrcoef( df[output],x_decayed)[0,1]
        return -corr

    bnds = [(0.1, 10), (0.01, 0.5)]
    x0 = [1, 0.1]
    res = minimize(adstock_correlation, x0=x0, bounds=bnds, args=(df, output, column_b), method='L-BFGS-B')
    optimal_params = res.x
    corr=adstock_correlation(optimal_params, df, output, column_b)
    return optimal_params,corr

def measure_dim_effect(df, output, column_b):
    def dim_correlation(params, df, output, column_b):
        alpha, gamma = params
        diminished =saturation_hill(df[column_b], alpha, gamma)
        corr = np.corrcoef( df[output],diminished)[0,1]
        return -corr

    bnds = [(0.1, 3.1), (0.1, 1.1)]
    x0 = [1, 0.1]
    res = minimize(dim_correlation, x0=x0, bounds=bnds, args=(df, output, column_b), method='L-BFGS-B')
    optimal_params = res.x
    corr=dim_correlation(optimal_params, df, output, column_b)
    return optimal_params,corr