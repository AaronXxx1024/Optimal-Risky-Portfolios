# Optimal-Risky-Portfolios
Practice of Markowitz Portfolio Optimization Model and Index Model

Typically, an investment decision could be separted to three parts: 
1) capital allocation between the risky portfolio and risk-free portfolio;
2) asset allocation within the risky portfolio across multi-classes;
3) security selection of individual assets whithin in each class.

Markowitz Portfolio Optimization Model and Index Model both are models that seek to find the optimal portfolio in the mean-variance plane. They have their own advantages and disadvantages. Markowitz model require massive estimates in the input list. Index model, however, simplifies this step and is helpful in decentralizing macro and security analysis. But at certain level, the index model should be inferior to the Markowitz model. 


## Optimal Risky Portfolios
Annualized return and std are based on daily data. I used return instead of price to construct correlation matrix and covariance matrix. Besides, both short-selling allowed and disallowed scenario are considered.

## Index Model
Data about excess return of each securities, estimated market risk premium and estimated alpha, which are the most important elements in the index model, are actually provided by *Investments_11th_Edition_Zvi_Bodie*. Because this is only a practice of portfolio model instead of security analysis. 
