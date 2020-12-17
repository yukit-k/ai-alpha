import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

def fit_pca(returns, num_factor_exposures, svd_solver):
    """
    Fit PCA model with returns.

    Parameters
    ----------
    returns : DataFrame
        Returns for each ticker and date
    num_factor_exposures : int
        Number of factors for PCA
    svd_solver: str
        The solver to use for the PCA model

    Returns
    -------
    pca : PCA
        Model fit to returns
    """
    pca = PCA(n_components=num_factor_exposures, svd_solver=svd_solver)
    pca.fit(returns)    
    return pca

def factor_betas(pca, factor_beta_indices, factor_beta_columns):
    """Get the factor betas from the PCA model."""
    assert len(factor_beta_indices.shape) == 1
    assert len(factor_beta_columns.shape) == 1
    
    factor_betas = pd.DataFrame(pca.components_.T, index=factor_beta_indices, columns=factor_beta_columns)
    
    return factor_betas

def factor_returns(pca, returns, factor_return_indices, factor_return_columns):
    """Get the factor returns from the PCA model."""
    assert len(factor_return_indices.shape) == 1
    assert len(factor_return_columns.shape) == 1
    
    factor_returns = pd.DataFrame(pca.transform(returns), index=factor_return_indices, columns=factor_return_columns)
    
    return factor_returns

def factor_cov_matrix(factor_returns, ann_factor):
    factor_cov_matrix = np.diag(np.var(factor_returns, axis=0, ddof=1)*ann_factor)
    
    return factor_cov_matrix

def idiosyncratic_var_matrix(returns, factor_returns, factor_betas, ann_factor):
    _common_returns = pd.DataFrame(np.dot(factor_returns, factor_betas.T), returns.index, returns.columns)
    _residuals = (returns - _common_returns)
    idiosyncratic_var_matrix = pd.DataFrame(np.diag(np.var(_residuals)*ann_factor), returns.columns, returns.columns)
    
    return idiosyncratic_var_matrix

def idiosyncratic_var_vector(returns, idiosyncratic_var_matrix):
    idiosyncratic_var_vector = pd.DataFrame(np.diag(idiosyncratic_var_matrix), returns.columns)
    
    return idiosyncratic_var_vector

def predict_portfolio_risk(factor_betas, factor_cov_matrix, idiosyncratic_var_matrix, weights):
    """
    Get the predicted portfolio risk
    
    Formula for predicted portfolio risk is sqrt(X.T(BFB.T + S)X) where:
      X is the portfolio weights
      B is the factor betas
      F is the factor covariance matrix
      S is the idiosyncratic variance matrix

    Parameters
    ----------
    factor_betas : DataFrame
        Factor betas
    factor_cov_matrix : 2 dimensional Ndarray
        Factor covariance matrix
    idiosyncratic_var_matrix : DataFrame
        Idiosyncratic variance matrix
    weights : DataFrame
        Portfolio weights

    Returns
    -------
    predicted_portfolio_risk : float
        Predicted portfolio risk
    """
    assert len(factor_cov_matrix.shape) == 2
    
    predicted_portfolio_risk = np.sqrt(weights.T.dot(factor_betas.dot(factor_cov_matrix).dot(factor_betas.T) + idiosyncratic_var_matrix).dot(weights))
    
    return predicted_portfolio_risk[0]

def get_risk_exposures(factor_betas, weights):
    return factor_betas.loc[weights.index].T.dot(weights)

def get_risk_factors(returns_df, num_factors=20, ann_factor=252):
    pca = fit_pca(returns_df, num_factors, 'full')
    risk_model = {}
    risk_model['factor_betas'] = factor_betas(pca, returns_df.columns.values, np.arange(num_factors))
    risk_model['factor_returns'] = factor_returns(pca, returns_df, returns_df.index, np.arange(num_factors))
    risk_model['factor_cov_matrix'] = factor_cov_matrix(risk_model['factor_returns'], ann_factor)
    risk_model['idiosyncratic_var_matrix'] = idiosyncratic_var_matrix(returns_df, risk_model['factor_returns'], risk_model['factor_betas'], ann_factor)
    risk_model['idiosyncratic_var_vector'] = idiosyncratic_var_vector(returns_df, risk_model['idiosyncratic_var_matrix'])
    return risk_model