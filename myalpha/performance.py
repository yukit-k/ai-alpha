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
        factor: factor_data.set_index(pd.MultiIndex.from_tuples(
            [(x.timestamp(), y) for x, y in factor_data.index.values],
            names=['date', 'asset']))
        for factor, factor_data in factor_data.items()}

    for factor, factor_data in factor_data.items():
        ls_FRA[factor] = al.performance.factor_rank_autocorrelation(factor_data)

    ls_FRA.plot(title="Factor Rank Autocorrelation", ylim=(0.8, 1.0), figsize=(12,7))
    

def build_factor_data(factor_data, pricing):
    return {factor_name: al.utils.get_clean_factor_and_forward_returns(factor=data, prices=pricing, max_loss=0.35, periods=[1])
        for factor_name, data in factor_data.iteritems()}

def show_sample_results(data, samples, classifier, factors, pricing, ymin=0.9, ymax=1.5):
    factors_sample = data.loc[samples.index].copy()
    factors_label = factors

    # Add AI_ALPHA factor if classifier is not None
    if classifier:
        # Calculate the Alpha Score
        prob_array=[-1,1]
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
    selected_factors = factors[labels]
    # print('Selected Factors: {}'.format(', '.join(selected_factors)))

    selected_factors['alpha_vector'] = selected_factors.mean(axis=1)
    alphas = selected_factors[['alpha_vector']]
    alpha_vector = alphas.loc[selected_factors.index.get_level_values(0)[-1]]
    return alpha_vector

def get_alpha_vector2(alpha_factors_today, factor_columns, shape_ratio_value):
    scale = 1
    shape_ratio_value = np.nan_to_num(shape_ratio_value)
    shape_ratio_value = shape_ratio_value / np.sum(shape_ratio_value)
    alpha_factors_today['AI_ALPHA'] = np.dot(alpha_factors_today[factor_columns], shape_ratio_value)
    alpha_vector  = alpha_factors_today[['AI_ALPHA']]
    return scale * alpha_vector


def get_factor_exposures(factor_betas, weights):
    return factor_betas.loc[weights.index].T.dot(weights)