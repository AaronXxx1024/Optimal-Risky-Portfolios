#%%
import numpy as np
import pandas as pd
import Aaron as aa  # Personal Module
import statsmodels.api as sm
from math import sqrt

aa.check_wd()
wd = 'D:\PycharmProjects\Learning\Empirical Study of Portfolio Management\Optimal Risky-Portfolio & Treynor-Black'
aa.change_wd(wd)

#%% Descriptive Stats
# relative path and wd are based on personal environment
path = 'Index Model.csv'
raw = pd.read_csv(path)

df = raw.iloc[:, 2:]
# Annualized Standard Deviation
std = np.std(df, ddof=1)
ann_std = std * sqrt(12)    # transfer monthly std to annual std
# Mean
mean = np.average(df, axis=0)
mean = pd.Series(mean, index=std.index)
# Correlation Coefficient
cor = np.corrcoef(df, rowvar=False)
cor = pd.DataFrame(cor, index=std.index, columns=std.index)
cor_with_index = pd.Series(cor.iloc[:, 0], name='Correlation with the Market Index')

#%% Parameters Estimate Tables
# Panel A: Risk Parameters of the Investable Universe (annualized)

# todo: write a function that collects info from regression fitted model
def risk_parameters(ticker: str, index: str, data_source):
    input_parm = ticker + ' ~ ' + index
    lm = sm.OLS.from_formula(input_parm, data_source)
    ols_result = lm.fit()
    # assume input old_result is an 'OLS fitted result' object
    sd_systematic_component = ols_result.params[index] * ann_std[index]
    sd_residual = np.std(ols_result.resid, ddof=2) * sqrt(12)  # residual from regression, Degrees of Freedom = N - ddof
    cor_with_market = cor_with_index[ticker]

    # Form risk parameters series
    panel_index = ['SD of Excess Return', 'BETA', 'SD of Systematic Component',
                   'SD of Residual', 'Correlation with the Market Index']
    panel = [ann_std[ticker], ols_result.params[index], sd_systematic_component, sd_residual, cor_with_market]
    panel = pd.Series(panel, index=panel_index, name=ticker)
    return panel, ols_result


# Market Index
Market_Index, Index_result = risk_parameters('Market_Index', 'Market_Index', df)
# Walmart
WMT, WMT_result = risk_parameters('WMT', 'Market_Index', df)
# Target
TGT, TGT_result = risk_parameters('TGT', 'Market_Index', df)
# Verizon
VZ, VZ_result = risk_parameters('VZ', 'Market_Index', df)
# AT&T
T, T_result = risk_parameters('T', 'Market_Index', df)
# Ford
Ford, Ford_result = risk_parameters('Ford', 'Market_Index', df)
# General Motors
GM, GM_result = risk_parameters('GM', 'Market_Index', df)

Panel_A = pd.DataFrame([Market_Index, WMT, TGT, VZ, T, Ford, GM], index=std.index, columns=Market_Index.index)

#%% Panel B: Correlation of Residuals
Panel_resid = pd.DataFrame([WMT_result.resid, TGT_result.resid, VZ_result.resid, T_result.resid, Ford_result.resid,
                            GM_result.resid], index=std.index[1:])
Panel_B = np.corrcoef(Panel_resid)
Panel_B = pd.DataFrame(Panel_B, index=Panel_resid.index, columns=Panel_resid.index)

#%% Panel C: The Index Model Covariance Matrix
cov_mat = np.identity(7)
# Identity diagonal is the variance for each securities and market index
for i in range(len(ann_std)):
    cov_mat[i,i] = cov_mat[i,i] * pow(Panel_A.iloc[i,0], 2)

for i in range(len(ann_std)):
    # where cov_mat[0, 0] is sigma_m, the variance of market index
    # cov = beta_i * beta_j * (sigma_m ** 2)
    # first column and first row
    if i + 1 > 6:
        break
    cov_mat[i + 1, 0] = cov_mat[0, 0] * Panel_A.iloc[i + 1, 1] * Panel_A.iloc[0, 1]
    cov_mat[0,:] = cov_mat[:,0]

for i in range(len(ann_std)):
    # second
    if i + 2 > 6:
        break
    cov_mat[i + 2, 1] = cov_mat[0, 0] * Panel_A.iloc[i + 2, 1] * Panel_A.iloc[1, 1]
    cov_mat[1, 2:] = cov_mat[2:, 1]

for i in range(len(ann_std)):
    # third
    if i + 3 > 6:
        break
    cov_mat[i + 3, 2] = cov_mat[0, 0] * Panel_A.iloc[i + 3, 1] * Panel_A.iloc[2, 1]
    cov_mat[2, 3:] = cov_mat[3:, 2]

for i in range(len(ann_std)):
    # fourth
    if i + 4 > 6:
        break
    cov_mat[i + 4, 3] = cov_mat[0, 0] * Panel_A.iloc[i + 4, 1] * Panel_A.iloc[3, 1]
    cov_mat[3, 4:] = cov_mat[4:, 3]

for i in range(len(ann_std)):
    # fifth
    if i + 5 > 6:
        break
    cov_mat[i + 5, 4] = cov_mat[0, 0] * Panel_A.iloc[i + 5, 1] * Panel_A.iloc[4, 1]
    cov_mat[4, 5:] = cov_mat[5:, 4]

for i in range(len(ann_std)):
    # sixth
    if i + 6 > 6:
        break
    cov_mat[i + 6, 5] = cov_mat[0, 0] * Panel_A.iloc[i + 6, 1] * Panel_A.iloc[5, 1]
    cov_mat[5, 6:] = cov_mat[6:, 5]

Panel_C = pd.DataFrame(cov_mat, index=std.index, columns=std.index)

#%% Panel D: Macro Forecast and Forecasts of Alpha Values
# market risk premium and estimated alpha are provided by book
# to be honest, the spirit of index model is market risk premium and estimated alpha
# we assume they are estimated by analysts, since this is only a practice of index model instead of security analysis
market_risk_premium = 0.06
alpha = np.array([0.0000, 0.0150, -0.0100, -0.0050, 0.0075, 0.0120, 0.0025])
risk_premium = pd.Series(np.arange(len(Panel_A)), dtype=float, index=std.index, name='Risk Premium')
for i in range(len(Panel_A)):
    # risk_premium = alpha + beta * market_risk_premium
    risk_premium[i] = alpha[i] + (Panel_A.iloc[i,1] * market_risk_premium)

Panel_D = pd.DataFrame([alpha, Panel_A.iloc[:,1], risk_premium], index=['Alpha', 'Beta', 'Risk Premium'],
                       columns=std.index)

#%% Panel E: Computation of the Optimal Risky Portfolio

var_resid = Panel_A['SD of Residual'] ** 2
# 1) initial position of each security in the active portfolio
ini_pos = alpha / var_resid

# 2) scale initial positions and make them sum to 1
scale_pos = pd.Series(np.zeros_like(6), dtype=float)
for i in range(len(ini_pos)):
    scale_pos[i] = ini_pos[i] / np.sum(ini_pos)

# 3) alpha of active portfolio
# simply define sumproduct function
def sumproduct(x, y):
    tmp = []
    for m, n in zip(x, y):
        tmp.append(m * n)
    return np.sum(tmp)


alpha_active = sumproduct(scale_pos, alpha)

# 4) residual variance of the active portfolio
resid_active = sumproduct(scale_pos**2, var_resid)

# 5) initial position of active portfolio (short-sale allowed)
w_active = (alpha_active / resid_active) / (market_risk_premium / Panel_A.iloc[0,0] ** 2)

# 6) beta of the active portfolio
beta_active = sumproduct(scale_pos, Panel_A.iloc[:,1])

# 7) adjust the initial position in the active portfolio
w_active_star = w_active / (1 + (1 - beta_active) * w_active)

# 8) weight of market index in the optimal risky portfolio
w_market_star = 1 - w_active_star

# 9) risk premium of the optimal risky portfolio
Ret_port = (w_market_star + w_active_star * beta_active) * market_risk_premium + w_active_star * alpha_active

# 10) variance of the optimal risky portfolio
var_port = pow((w_market_star + w_active_star * beta_active), 2) * pow(Panel_A.iloc[0,0], 2) + \
           pow((w_active_star * sqrt(resid_active)), 2)
sd_port = sqrt(var_port)

# 11) Sharpe ratio of the optimal risky portfolio: 0.5165
sharpe_ratio = Ret_port / sd_port
