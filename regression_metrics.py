# -*- coding: utf-8 -*-
"""
@CodeAuthor   : Gaosong Shi
@Email        : shgsong@foxmail.com
@Time         : 2023/6/24    0024 11:25
@ProjectName  : CSDLv2
@File         : regression_metrics.py
@Software     : PyCharm
@CodePurpose  : 
"""


from typing import Tuple
import pandas as pd
from scipy import stats
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import matplotlib.colors as colors


def nrmse(y_true, y_pred):
    """
    计算归一化均方根误差(NRMSE)

    参数:
    - y_true: 真实值的数组或列表
    - y_pred: 预测值的数组或列表

    返回值:
    - nrmse: 归一化均方根误差
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # 计算均方根误差
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

    # 计算目标变量的范围
    y_range = np.max(y_true) - np.min(y_true)

    # 计算归一化均方根误差
    nrmse = rmse / y_range

    return nrmse

def Series_mean_error(y_true, y_pred):
    """
    纯python的计算方式。与论文中的公式对用
    计算平均误差（Mean Error）

    参数：
    y_true：实际值的列表或数组
    y_pred：预测值的列表或数组

    返回值：
    mean_error：平均误差（Mean Error）
    """
    n = len(y_true)
    error = sum(y_pred[i] - y_true[i] for i in range(n))
    mean_error = error / n
    return mean_error

def mean_error(y_true, y_pred):
    """
    使用了NumPy库来计算平均误差。它将输入的实际值和预测值转换为NumPy数组，
    并直接使用NumPy的函数来计算平均误差。

    计算平均误差（Mean Error）

    参数：
    y_true：实际值的列表或数组
    y_pred：预测值的列表或数组

    返回值：
    mean_error：平均误差（Mean Error）
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    error = np.mean(y_pred - y_true)
    return error

def lin_ccc(y_true, y_pred):
    """
    默认情况下，ddof 的值为 0，表示使用总体方差的计算方法。
    当 ddof 设置为 1 时，表示使用样本方差的计算方法。
    在计算样本方差时，通常建议将 ddof 设置为 1，以获得无偏估计的方差
    y_true：实际值的列表或数组    y_pred：预测值的列表或数组
    """
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    var_true = np.var(y_true, ddof=1)
    var_pred = np.var(y_pred, ddof=1)
    cov = np.cov(y_true, y_pred, ddof=1)[0, 1]
    lccc = (2 * cov) / (var_true + var_pred + (mean_true - mean_pred) ** 2) ##  分子为计算的是协方差，与先计算r结果一致。
    return lccc


def picp(y_true: np.ndarray, y_lower: np.array, y_upper: np.array) -> float:
    """
参考：https://github.com/IBM/UQ360/blob/3495d2b53b77ab1d675021741e8252ff395e3ab4/uq360/metrics/regression_metrics.py#L8
上面参考链接中 对不确定性的相关衡量方法都进行了介绍。
    Prediction Interval Coverage Probability (PICP). Computes the fraction of samples for which the grounds truth lies
    within predicted interval. Measures the prediction interval calibration for regression.

    Args:
        y_true: Ground truth
        y_lower: predicted lower bound
        y_upper: predicted upper bound

    Returns:
        float: the fraction of samples for which the grounds truth lies within predicted interval.
    """
    satisfies_upper_bound = y_true <= y_upper
    satisfies_lower_bound = y_true >= y_lower
    return np.mean(satisfies_upper_bound * satisfies_lower_bound)


def picp_auxiliary(low_ci: np.ndarray, y_true: np.ndarray, up_ci: np.ndarray) -> Tuple[float, float]:
    """
    Return the Percentage Interval Coverage Probability (PICP):
    a percentage that quantifies the amount of in situ chla reference values that lay within the BNN estimated confidence intervals (CIs).

    Args:
        low_ci: 1D array of floats - the lower confidence interval values.
        y_true: 1D array of floats - the in situ reference chla data.
        up_ci: 1D array of floats - the upper confidence interval values.

    Returns:
        A tuple of floats representing the percentages of observations in and outside of the CIs.
    """
    # Count the number of elements in y_true that are greater than or equal to the corresponding element in low_ci and less than or equal to the corresponding element in up_ci.
    count_in = np.sum(np.greater_equal(y_true, low_ci) & np.less_equal(y_true, up_ci))

    # Count the number of elements in y_true that are outside of the confidence interval.
    count_out = len(y_true) - count_in

    # Compute the percentage of elements in y_true that are inside of the confidence interval.
    perc_in = count_in / len(y_true) * 100

    # Compute the percentage of elements in y_true that are outside of the confidence interval.
    perc_out = count_out / len(y_true) * 100

    # Return the percentages of elements in and outside of the confidence interval.
    return perc_in, perc_out



def MEC(y_pred, y_true):
    y_true_mean = np.mean(y_true)
    num = np.sum((y_pred - y_true)**2)
    den = np.sum((y_true - y_true_mean)**2)
    return 1 - (num/den)




def nash_sutcliffe_efficiency(observed, simulated):
    """
    Calculate Nash-Sutcliffe Efficiency (NSE) between observed and simulated values.

    Args:
    observed: Array or list of observed values.
    simulated: Array or list of simulated/forecasted values.

    Returns:
    nse: Nash-Sutcliffe Efficiency value.
    """

    # Convert inputs to numpy arrays
    observed = np.array(observed)
    simulated = np.array(simulated)

    # Calculate mean of observed values
    mean_observed = np.mean(observed)

    # Calculate NSE
    numerator = np.sum((observed - simulated) ** 2)
    denominator = np.sum((observed - mean_observed) ** 2)
    nse = 1 - numerator / denominator

    return nse


def calculate_r_squared_residual_total(y_true, y_pred):
    # y_true: 实际观测值
    # y_pred: 模型预测值

    mean_y_true = np.mean(y_true)

    ss_residual = np.sum((y_true - y_pred) ** 2)
    ss_total = np.sum((y_true - mean_y_true) ** 2)

    r_squared = 1 - (ss_residual / ss_total)

    return r_squared


def calculate_r_squared_reg_total(y_true, y_pred):
    # y_true: 实际观测值
    # y_pred: 模型预测值

    mean_y_true = np.mean(y_true)

    ss_reg = np.sum((y_pred - mean_y_true) ** 2)
    ss_total = np.sum((y_true - mean_y_true) ** 2)

    r_squared = ss_reg / ss_total

    return r_squared




def calculate_r2(observed, simulated):
    observed, simulated = np.array(observed), np.array(simulated)
    mean_observed = np.mean(observed)
    fenzi = []
    fenmu = []
    for i in range(len(observed)):
        fenzi_i_value = (observed[i] - simulated[i]) ** 2
        fenmu_i_value = (observed[i] - mean_observed) ** 2
        fenzi.append(fenzi_i_value)
        fenmu.append(fenmu_i_value)
    MEC_value = 1 - np.sum(fenzi) / np.sum(fenmu)
    return MEC_value





def evaluateEval(x, y):
    x, y = np.array(x), np.array(y)
    # mean error
    ME = np.round(np.mean(y - x, axis=0, where=~np.isnan(y - x)), decimals=2)

    # root mean square error
    RMSE = np.round(np.sqrt(np.mean((y - x) ** 2, axis=0, where=~np.isnan(y - x))), decimals=2)

    # mean absolute error
    MAE = np.round(np.mean(np.abs(y - x), axis=0, where=~np.isnan(np.abs(y - x))), decimals=2)

    # Pearson's correlation squared
    r = pearsonr(x, y)
    r2 = np.round(r[0] ** 2, decimals=2)

    # Nash-Sutcliffe efficiency
    SSE = np.sum((y - x) ** 2, where=~np.isnan(y - x))
    SST = np.sum((y - np.mean(y, where=~np.isnan(y))) ** 2, where=~np.isnan(y))
    NSE = np.round(1 - SSE / SST, decimals=2)

    # concordance correlation coefficient
    n = len(x)
    sdx = np.std(x, ddof=1)
    sdy = np.std(y, ddof=1)
    r = pearsonr(x, y)[0]
    v = sdx / sdy
    sx2 = np.var(x, ddof=1) * (n - 1) / n
    sy2 = np.var(y, ddof=1) * (n - 1) / n
    u = (np.mean(x) - np.mean(y)) / ((sx2 * sy2) ** 0.25)
    Cb = ((v + 1 / v + u ** 2) / 2) ** -1
    rCb = r * Cb
    rhoC = np.round(rCb, decimals=2)

    Cb = np.round(Cb, decimals=2)
    r = np.round(r, decimals=2)

    # return the results
    evalRes = {
        'ME': ME,
        'MAE': MAE,
        'RMSE': RMSE,
        'r': r,
        'r2': r2,
        'NSE': NSE,
        'rhoC': rhoC,
        'Cb': Cb
    }

    # return evalRes
    return r2, RMSE, ME, MAE



def accu(a,b):
    RMSE = mean_squared_error(a, b) ** 0.5
    MAE = mean_absolute_error(a, b)
    R2 = r2_score(a, b)
    #####
    x=a
    R = pearsonr(x.reshape(x.shape[0]), b.reshape(b.shape[0]))[0]
    alpha=np.std(a)/np.std(b)
    Beta=np.mean(a)/np.mean(b)
    KGE=1-np.sqrt((R-1)**2+(alpha-1)**2+(Beta-1)**2)
    LCC=(2*np.var(a)*np.var(b))/\
        (np.mean((a-b)**2)+np.var(b)**2+np.var(b)**2)
    std_obv = np.std(a)
    std_pre = np.std(b)
    CCC = (2 * R * std_obv * std_pre) / \
          (std_obv ** 2 + std_pre ** 2 + (np.mean(a) - np.mean(b)) ** 2)
    ME = np.sum(a - b) / len(a)

    return RMSE, MAE, R2, R, alpha, Beta, KGE, LCC, CCC, ME















