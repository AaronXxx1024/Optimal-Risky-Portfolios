#%% 1. Financial Data & ratios of all assets (Expected return, std, correlation matrix
import pandas_datareader as data
import numpy as np
import pandas as pd
import scipy.optimize as opt
import math as ma
import matplotlib.pyplot as plt

# import data from pandas API
asset = data.DataReader(name=['SPY', 'GOOG', 'AAPL','XOM', 'WMT'], data_source='yahoo',
                        start='1/1/2008', end='4/15/2011')
df = asset.iloc[:,[5,6,7,8,9]]
df.columns = ['SPY', 'GOOG', 'AAPL','XOM', 'WMT']

# transfer daily price to daily return
df = df.pct_change()
df = df.iloc[1:,:]

# calculate info from initial data
std = df.std() * ma.sqrt(252)  # std of each asset * trading days = annual std
exp_ret = pd.Series(data=[0.0536, 0.05, 0.045, 0.055, 0.03], index=std.index)  # expected return

# correlation matrix: return or price?
plt.plot(asset.iloc[:,[5,6,7,8,9]])
plt.show()
plt.plot(df)    # Let's choose return here
plt.show()
cor_mat = np.corrcoef(df, rowvar=False)
cor_mat = pd.DataFrame(data=cor_mat, index=df.columns, columns=df.columns)

# optimal sharpe calculation:
# 1) without short-selling
def max_sharpe(x):
    por_ret = x[0]*exp_ret[0] + x[1]*exp_ret[1] + x[2]*exp_ret[2] + x[3]*exp_ret[3] + x[4]*exp_ret[4]
    rf = 0.0022
    square_part = pow(x[0],2) * pow(std[0],2) + pow(x[1],2) * pow(std[1],2) + pow(x[2],2) * pow(std[2],2) + pow(x[3],2) * pow(std[3],2) + pow(x[4],2) * pow(std[4],2)
    mul_part = (2*x[0]*x[1]*std[0]*std[1]*cor_mat.iat[1,0]) + (2*x[0]*x[2]*std[0]*std[2]*cor_mat.iat[2,0]) +\
               (2 * x[0] * x[3] * std[0] * std[3] * cor_mat.iat[3, 0]) + (2*x[0]*x[4]*std[0]*std[4]*cor_mat.iat[4,0]) + \
               (2 * x[1] * x[2] * std[1] * std[2] * cor_mat.iat[2, 1]) + (2 * x[1] * x[3] * std[1] * std[3] * cor_mat.iat[3, 1]) + \
               (2 * x[1] * x[4] * std[1] * std[4] * cor_mat.iat[4, 1]) + (2 * x[2] * x[3] * std[2] * std[3] * cor_mat.iat[3, 2]) + \
               (2 * x[2] * x[4] * std[2] * std[4] * cor_mat.iat[4, 2]) + (2 * x[3] * x[4] * std[3] * std[4] * cor_mat.iat[4, 3])
    por_std = square_part + mul_part
    sharpe = (por_ret - rf) / ma.sqrt(por_std)
    return (-1) * sharpe


x0_bounds = (0, None)
x1_bounds = (0, None)
x2_bounds = (0, None)
x3_bounds = (0, None)
x4_bounds = (0, None)
bounds = (x0_bounds, x1_bounds, x2_bounds, x3_bounds, x4_bounds)

eq_cons = {'type': 'eq', 'fun': lambda x: np.array([x[0] + x[1] + x[2] + x[3] + x[4] - 1])}
# method='SLSQP',
opt_portfolio = opt.minimize(max_sharpe, x0=np.array([0.2, 0.2, 0.2, 0.2, 0.2]),
                             bounds=bounds, constraints=eq_cons,
                             options={'disp': True})
print(opt_portfolio)
opt_sharpe = -opt_portfolio.fun     # Sharpe Ratio: 0.1783
weight_opt = opt_portfolio.x        # Weight: 0.69573239, 0.0422932,  0., 0.16684255, 0.09513187
print(opt_sharpe, weight_opt)

# 2) with short-selling
optss_portfolio = opt.minimize(max_sharpe, x0=np.array([0.2, 0.2, 0.2, 0.2, 0.2]),
                               constraints=eq_cons, options={'disp':True})
print(optss_portfolio)
# Sharpe Ratio: 0.1792
# Weight: 0.78938606,  0.07765181, -0.10355287,  0.14905528,  0.08745972
