import numpy as np
import pandas as pd

import alphalens as al

def get_sharpe_ratio(factor_returns, annualization_factor=np.sqrt(252)):
    return annualization_factor * factor_returns.mean() / factor_returns.std()

def get_factor_returns(factor_data):
    ls_factor_returns = pd.DataFrame()

    for factor, factor_data in factor_data.items():
        ls_factor_returns[factor] = al.performance.factor_returns(factor_data).iloc[:, 0]

    return ls_factor_returns

def get_qr_factor_returns(factor_data):
    qr_factor_returns = pd.DataFrame()

    for factor_name, data in factor_data.items():
        qr_factor_returns[factor_name] = al.performance.mean_return_by_quantile(data)[0].iloc[:, 0]

    return qr_factor_returns

def plot_factor_returns(factor_returns, ymin, ymax):
    (1 + factor_returns).cumprod().plot(ylim=(ymin, ymax), figsize=(12,7))
    
def plot_qr_factor_returns(qr_factor_returns):
    (10000*qr_factor_returns).plot.bar(
    subplots=True,
    sharey=True,
    layout=(5,3),
    figsize=(14, 14),
    legend=False)
    
def plot_factor_rank_autocorrelation(factor_data):
    ls_FRA = pd.DataFrame()

    unixt_factor_data = {
        factor: df.set_index(pd.MultiIndex.from_tuples(
            [(x.timestamp(), y) for x, y in df.index.values],
            names=['date', 'asset']))
        for factor, df in factor_data.items()}

    for factor, df in unixt_factor_data.items():
        ls_FRA[factor] = al.performance.factor_rank_autocorrelation(df)

    ls_FRA.plot(title="Factor Rank Autocorrelation", ylim=(0.8, 1.0), figsize=(12,7))
    

def build_factor_data(factor_data, pricing):
    return {factor_name: al.utils.get_clean_factor_and_forward_returns(factor=data, prices=pricing, max_loss=0.35, periods=[1])
        for factor_name, data in factor_data.items()}

def show_sample_results(data, samples, classifier, factors, pricing, ymin=0.9, ymax=1.5):
    factors_sample = data.loc[samples.index].copy()
    factors_label = factors

    # Add AI_ALPHA factor if classifier is not None
    if classifier:
        # Calculate the Alpha Score
        prob_array=[-1,0,1]
        alpha_score = classifier.predict_proba(samples).dot(np.array(prob_array))
        
        # Add Alpha Score to rest of the factors
        alpha_score_label = 'AI_ALPHA'
        factors_sample[alpha_score_label] = alpha_score
        factors_label = factors + [alpha_score_label]
    
    # Setup data for AlphaLens
    print('Cleaning Data...\n')
    factor_data = build_factor_data(factors_sample[factors_label], pricing)
    print('\n-----------------------\n')
    
    # Calculate Factor Returns and Sharpe Ratio
    factor_returns = get_factor_returns(factor_data)
    qr_factor_returns = get_qr_factor_returns(factor_data)
    sharpe_ratio = get_sharpe_ratio(factor_returns)
    
    # Show Results
    print('             Sharpe Ratios')
    print(sharpe_ratio.round(2))
    plot_factor_returns(factor_returns, ymin, ymax)
    plot_qr_factor_returns(qr_factor_returns)
    plot_factor_rank_autocorrelation(factor_data)
    
    return factor_data

def get_alpha_vector_mean_lastday(factors, labels):
    selected_factors = factors[labels].copy()
    selected_factors['alpha_vector'] = selected_factors.mean(axis=1)
    alphas = selected_factors[['alpha_vector']]
    alpha_vector = alphas.loc[selected_factors.index.get_level_values(0)[-1]]
    return alpha_vector

def get_alpha_vector2(alpha_factors_today, factor_columns, shape_ratio_value):
    scale = 1
    shape_ratio_value = np.nan_to_num(shape_ratio_value, nan=0.0, posinf=0.0, neginf=0.0)
    total = np.sum(shape_ratio_value)
    if total == 0.0:
        raise ValueError("Sum of Sharpe ratios is zero; cannot normalize factor weights.")
    shape_ratio_value = shape_ratio_value / total
    result = alpha_factors_today.copy()
    result['AI_ALPHA'] = np.dot(result[factor_columns], shape_ratio_value)
    return scale * result[['AI_ALPHA']]


def get_factor_exposures(factor_betas, weights):
    return factor_betas.loc[weights.index].T.dot(weights)