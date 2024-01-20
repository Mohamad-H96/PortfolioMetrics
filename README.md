# PortfolioMetrics

Overview
The PortfolioMetrics package provides a set of functions for portfolio analysis in Python. It includes tools for computing risk metrics, portfolio optimization, efficient frontier visualization, backtesting, and more.

Features
1. Risk Metrics

    drawdown: Calculate drawdown metrics for asset returns.
    semideviation: Compute the semideviation (negative semideviation) of returns.
    skewness and kurtosis: Measure the skewness and kurtosis of returns.
    is_normal: Check if a series of returns follows a normal distribution.
    VaR_historic and VaR_gaussian: Compute Value at Risk (VaR) using historic and Gaussian methods.
    cVaR_historic: Compute Conditional Value at Risk (CVaR) using historic data.

2. Portfolio Optimization

    portfolio_return and portfolio_vol: Calculate the expected return and volatility of a portfolio.
    plot_ef2: Plot the efficient frontier for a two-asset portfolio.
    minimize_vol: Minimize the volatility to find the optimal weights for a target return.
    optimal_weights: Generate weights to minimize portfolio volatility for various target returns.
    msr: Get the weights of the portfolio with the maximum Sharpe ratio.
    gmv: Get the weights of the Global Minimum Volatility portfolio.
    plot_ef: Plot the efficient frontier for a multi-asset portfolio.

3. Backtesting

    run_cppi: Backtest the Constant Proportion Portfolio Insurance (CPPI) strategy.

4. Summary Statistics

    summary_stats: Generate aggregated summary statistics for a set of returns.

5. Monte Carlo Simulation

    gbm: Simulate the evolution of a stock price using Geometric Brownian Motion.

6. Black-Litterman Model

    bl: Compute the posterior expected returns based on the Black-Litterman model.

7. Diversification

    risk_contribution: Compute the contribution to risk of the constituents of a portfolio.
    target_risk_contribution: Get the weights for a target contribution to portfolio risk.
    equal_risk_contributions: Get weights for a portfolio with equal risk contributions.
    weight_erc: Produce weights for an Equal Risk Contribution (ERC) portfolio.
