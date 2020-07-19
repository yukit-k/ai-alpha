import alphalens as al
import graphviz
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cvxpy as cvx

from IPython.display import Image
from sklearn.tree import export_graphviz
from zipline.assets._assets import Equity  # Required for USEquityPricing
from zipline.pipeline.data import USEquityPricing
from zipline.pipeline.classifiers import Classifier
from zipline.pipeline.engine import SimplePipelineEngine
from zipline.pipeline.loaders import USEquityPricingLoader
from zipline.utils.numpy_utils import int64_dtype
from zipline.pipeline.factors import CustomFactor, DailyReturns, Returns, SimpleMovingAverage, AnnualizedVolatility
from zipline.pipeline import Pipeline
from zipline.pipeline.factors import AverageDollarVolume
from zipline.utils.calendars import get_calendar

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

import abc
from abc import ABC, abstractmethod


from sklearn.ensemble import VotingClassifier
from sklearn.base import clone
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import Bunch

EOD_BUNDLE_NAME = 'eod-quotemedia'

def plot(xs, ys, labels, title='', x_label='', y_label=''):
    for x, y, label in zip(xs, ys, labels):
        plt.ylim((0.5, 0.55))
        plt.plot(x, y, label=label)

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plt.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0)
    plt.show()


class PricingLoader(object):
    def __init__(self, bundle_data):
        self.loader = USEquityPricingLoader(
            bundle_data.equity_daily_bar_reader,
            bundle_data.adjustment_reader)

    def get_loader(self, column):
        if column not in USEquityPricing.columns:
            raise Exception('Column not in USEquityPricing')
        return self.loader
        
class Sector(Classifier):
    dtype = int64_dtype
    window_length = 0
    inputs = ()
    missing_value = -1

    def __init__(self):
        self.data = np.load('./data/sector/data.npy')

    def _compute(self, arrays, dates, assets, mask):
        return np.where(
            mask,
            self.data[assets],
            self.missing_value,
        )

def build_pipeline_engine(bundle_data, trading_calendar):
    pricing_loader = PricingLoader(bundle_data)

    engine = SimplePipelineEngine(
        get_loader=pricing_loader.get_loader,
        calendar=trading_calendar.all_sessions,
        asset_finder=bundle_data.asset_finder)

    return engine

def get_universe_tickers(current_date, universe, engine):
    """ current_date in pd.Timestamp """
    universe_tickers = engine\
        .run_pipeline(
            Pipeline(screen=universe),
            current_date,
            current_date)\
        .index.get_level_values(1)\
        .values.tolist()

    return universe_tickers

def get_pricing(data_portal, trading_calendar, assets, start_date, end_date, field='close'):
    end_dt = pd.Timestamp(end_date.strftime('%Y-%m-%d'), tz='UTC', freq='C')
    start_dt = pd.Timestamp(start_date.strftime('%Y-%m-%d'), tz='UTC', freq='C')

    end_loc = trading_calendar.closes.index.get_loc(end_dt)
    start_loc = trading_calendar.closes.index.get_loc(start_dt)

    return data_portal.get_history_window(
        assets=assets,
        end_dt=end_dt,
        bar_count=end_loc - start_loc,
        frequency='1d',
        field=field,
        data_frequency='daily')

######### Utility for Factors #########
def clean_factor_data(all_factors, pricing):
    clean_factor_data = {
    factor: al.utils.get_clean_factor_and_forward_returns(factor=factor_data, prices=pricing, periods=[1])
    for factor, factor_data in all_factors.iteritems()}
    return clean_factor_data

def get_unix_time(factor_data):
    unixt_factor_data = {
    factor: data.set_index(pd.MultiIndex.from_tuples(
        [(x.timestamp(), y) for x, y in data.index.values],
        names=['date', 'asset']))
    for factor, data in factor_data.items()}
    return unixt_factor_data


def build_factor_data(factor_data, pricing):
    return {factor_name: al.utils.get_clean_factor_and_forward_returns(factor=data, prices=pricing, max_loss=0.35, periods=[1])
        for factor_name, data in factor_data.iteritems()}

def wins(x,a,b):
    return np.where(x <= a,a, np.where(x >= b, b, x))

def clean_nas(df): 
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    for numeric_column in numeric_columns: 
        df[numeric_column] = np.nan_to_num(df[numeric_column])
    
    return df

######### Risk Model #########
def fit_pca(returns, num_factor_exposures, svd_solver):
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
    """Get the predicted portfolio risk"""
    assert len(factor_cov_matrix.shape) == 2
    
    predicted_portfolio_risk = np.sqrt(weights.T.dot(factor_betas.dot(factor_cov_matrix).dot(factor_betas.T) + idiosyncratic_var_matrix).dot(weights))
    
    return predicted_portfolio_risk['h.opt'][0]

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

######### Alpha Factors #########
def momentum_1yr(window_length, universe, sector):
    return Returns(window_length=window_length, mask=universe) \
        .demean(groupby=sector) \
        .rank() \
        .zscore()

def mean_reversion_5day_sector_neutral_smoothed(window_length, universe, sector):
    """
    Generate the mean reversion 5 day sector neutral factor
    """
    unsmoothed_factor = -Returns(window_length=5, mask=universe) \
        .demean(groupby=sector) \
        .rank() \
        .zscore()
    return SimpleMovingAverage(inputs=[unsmoothed_factor], window_length=window_length) \
        .rank() \
        .zscore()

class CTO(Returns):
    """
    Computes the overnight return, per hypothesis from
    https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2554010
    """
    inputs = [USEquityPricing.open, USEquityPricing.close]
    
    def compute(self, today, assets, out, opens, closes):
        """
        The opens and closes matrix is 2 rows x N assets, with the most recent at the bottom.
        As such, opens[-1] is the most recent open, and closes[0] is the earlier close
        """
        out[:] = (opens[-1] - closes[0]) / closes[0]
        
class TrailingOvernightReturns(Returns):
    """
    Sum of trailing 1m O/N returns
    """
    window_safe = True
    
    def compute(self, today, asset_ids, out, cto):
        out[:] = np.nansum(cto, axis=0)

def overnight_sentiment_smoothed(cto_window_length, trail_overnight_returns_window_length, universe):
    cto_out = CTO(mask=universe, window_length=cto_window_length)
    unsmoothed_factor = TrailingOvernightReturns(inputs=[cto_out], window_length=trail_overnight_returns_window_length) \
        .rank() \
        .zscore()
    return SimpleMovingAverage(inputs=[unsmoothed_factor], window_length=trail_overnight_returns_window_length) \
        .rank() \
        .zscore()

class MarketDispersion(CustomFactor):
    inputs = [DailyReturns()]
    window_length = 1
    window_safe = True

    def compute(self, today, assets, out, returns):
        # returns are days in rows, assets across columns
        out[:] = np.sqrt(np.nanmean((returns - np.nanmean(returns))**2))

class MarketVolatility(CustomFactor):
    inputs = [DailyReturns()]
    window_length = 1
    window_safe = True
    
    def compute(self, today, assets, out, returns):
        mkt_returns = np.nanmean(returns, axis=1)
        out[:] = np.sqrt(260.* np.nanmean((mkt_returns-np.nanmean(mkt_returns))**2))


def get_alpha_factors(universe, sector, engine, factor_start_date, universe_end_date):
    pipeline = Pipeline(screen=universe)
    pipeline.add(
        momentum_1yr(252, universe, sector),
        'Momentum_1YR')
    pipeline.add(
        mean_reversion_5day_sector_neutral_smoothed(20, universe, sector),
        'Mean_Reversion_Sector_Neutral_Smoothed')
    pipeline.add(
        overnight_sentiment_smoothed(2, 10, universe),
        'Overnight_Sentiment_Smoothed')
    pipeline.add(AnnualizedVolatility(window_length=20, mask=universe).rank().zscore(), 'volatility_20d')
    pipeline.add(AnnualizedVolatility(window_length=120, mask=universe).rank().zscore(), 'volatility_120d')
    pipeline.add(AverageDollarVolume(window_length=20, mask=universe).rank().zscore(), 'adv_20d')
    pipeline.add(AverageDollarVolume(window_length=120, mask=universe).rank().zscore(), 'adv_120d')
    pipeline.add(AverageDollarVolume(window_length=20, mask=universe), 'adv_tc')
    pipeline.add(sector, 'sector_code')
    pipeline.add(SimpleMovingAverage(inputs=[MarketDispersion(mask=universe)], window_length=20), 'dispersion_20d')
    pipeline.add(SimpleMovingAverage(inputs=[MarketDispersion(mask=universe)], window_length=120), 'dispersion_120d')
    pipeline.add(MarketVolatility(window_length=20), 'market_vol_20d')
    pipeline.add(MarketVolatility(window_length=120), 'market_vol_120d')
    pipeline.add(Returns(window_length=5, mask=universe).quantiles(2), 'return_5d')

    all_factors = engine.run_pipeline(pipeline, factor_start_date, universe_end_date)

    all_factors['is_Janaury'] = all_factors.index.get_level_values(0).month == 1
    all_factors['is_December'] = all_factors.index.get_level_values(0).month == 12
    all_factors['weekday'] = all_factors.index.get_level_values(0).weekday
    all_factors['quarter'] = all_factors.index.get_level_values(0).quarter
    all_factors['qtr_yr'] = all_factors.quarter.astype('str') + '_' + all_factors.index.get_level_values(0).year.astype('str')
    all_factors['month_end'] = all_factors.index.get_level_values(0).isin(pd.date_range(start=factor_start_date, end=universe_end_date, freq='BM'))
    all_factors['month_start'] = all_factors.index.get_level_values(0).isin(pd.date_range(start=factor_start_date, end=universe_end_date, freq='BMS'))
    all_factors['qtr_end'] = all_factors.index.get_level_values(0).isin(pd.date_range(start=factor_start_date, end=universe_end_date, freq='BQ'))
    all_factors['qtr_start'] = all_factors.index.get_level_values(0).isin(pd.date_range(start=factor_start_date, end=universe_end_date, freq='BQS'))

    # One Hot Encode Sectors
    sector_lookup = pd.read_csv('./data/sector/labels.csv', index_col='Sector_i')['Sector'].to_dict()
    sector_columns = []
    for sector_i, sector_name in sector_lookup.items():
        secotr_column = 'sector_{}'.format(sector_name)
        sector_columns.append(secotr_column)
        all_factors[secotr_column] = (all_factors['sector_code'] == sector_i)

    all_factors['return_5d_2'] = all_factors.groupby(level=1)['return_5d'].shift(-2)
    all_factors['target'] = all_factors.groupby(level=1)['return_5d_2'].shift(-5)
    return all_factors

def get_alpha_exposures(factor_betas, weights):
    return factor_betas.loc[weights.index].T.dot(weights)

######### ML Training #########
class NoOverlapVoterAbstract(VotingClassifier):
    @abc.abstractmethod
    def _calculate_oob_score(self, classifiers):
        raise NotImplementedError
        
    @abc.abstractmethod
    def _non_overlapping_estimators(self, x, y, classifiers, n_skip_samples):
        raise NotImplementedError
    
    def __init__(self, estimator, voting='soft', n_skip_samples=4):
        # List of estimators for all the subsets of data
        estimators = [('clf'+str(i), estimator) for i in range(n_skip_samples + 1)]
        
        self.n_skip_samples = n_skip_samples
        super().__init__(estimators, voting)
    
    def fit(self, X, y, sample_weight=None):
        estimator_names, clfs = zip(*self.estimators)
        self.le_ = LabelEncoder().fit(y)
        self.classes_ = self.le_.classes_
        
        clone_clfs = [clone(clf) for clf in clfs]
        self.estimators_ = self._non_overlapping_estimators(X, y, clone_clfs, self.n_skip_samples)
        self.named_estimators_ = Bunch(**dict(zip(estimator_names, self.estimators_)))
        self.oob_score_ = self._calculate_oob_score(self.estimators_)
        
        return self

def calculate_oob_score(classifiers):
    '''
    Calculate the mean out-of-bag score from the classifiers.
    '''
    oob_score = 0
    for clf in classifiers:
        oob_score += clf.oob_score_ 
    return oob_score / len(classifiers)

def non_overlapping_estimators(x, y, classifiers, n_skip_samples):
    '''
    Fit the classifiers to non overlapping data.

    Parameters
    ----------
    x : [DataFrame] The input samples
    y : [Pandas Series] The target values
    '''
    fit_classifiers = []
    
    for i in range(n_skip_samples):
        fit_classifiers.append(
            classifiers[i].fit(x[i::n_skip_samples], y[i::n_skip_samples])
        )

    return fit_classifiers

class NoOverlapVoter(NoOverlapVoterAbstract):
    def _calculate_oob_score(self, classifiers):
        return calculate_oob_score(classifiers)
        
    def _non_overlapping_estimators(self, x, y, classifiers, n_skip_samples):
        return non_overlapping_estimators(x, y, classifiers, n_skip_samples)

def train_model(alpha_factors, features, target_label, clf_parameters, train):
    if train:
        temp = alpha_factors.dropna().copy()
        X = temp[features]
        y = temp[target_label]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        clf = RandomForestClassifier(**clf_parameters)
        clf_nov = NoOverlapVoter(clf)
        clf_nov.fit(X_train, y_train)

        train_score = clf_nov.score(X_train, y_train.values)
        test_score = clf_nov.score(X_test, y_test.values)
        oob_score = clf_nov.oob_score_

        # Re-training
        clf_nov.fit(X, y)
        train_score_rt = clf_nov.score(X, y.values)
        oob_score_rt = clf_nov.oob_score_

        return [clf_nov, train_score, test_score, oob_score, train_score_rt, oob_score_rt]
    else:
        return None

def plot_tree_classifier(clf, feature_names=None):
    dot_data = export_graphviz(
        clf,
        out_file=None,
        feature_names=feature_names,
        filled=True,
        rounded=True,
        special_characters=True,
        rotate=True)

    return Image(graphviz.Source(dot_data).pipe(format='png'))

def rank_features_by_importance(importances, feature_names):
    indices = np.argsort(importances)[::-1]
    max_feature_name_length = max([len(feature) for feature in feature_names])

    print('      Feature{space: <{padding}}      Importance'.format(padding=max_feature_name_length - 8, space=' '))

    for x_train_i in range(len(importances)):
        print('{number:>2}. {feature: <{padding}} ({importance})'.format(
            number=x_train_i + 1,
            padding=max_feature_name_length,
            feature=feature_names[indices[x_train_i]],
            importance=importances[indices[x_train_i]]))


######### Model Result #########
def sharpe_ratio(factor_returns, annualization_factor=np.sqrt(252)):
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

    for factor, factor_data in unixt_factor_data.items():
        ls_FRA[factor] = al.performance.factor_rank_autocorrelation(factor_data)

    ls_FRA.plot(title="Factor Rank Autocorrelation", ylim=(0.8, 1.0), figsize=(12,7))
    
def show_sample_results(all_factors, pricing, samples, classifier, ymin=0.9, ymax=1.5):
    factors = [
        'Mean_Reversion_Sector_Neutral_Smoothed',
        'Momentum_1YR',
        'Overnight_Sentiment_Smoothed',
        'adv_120d',
        'volatility_20d']
    # Calculate the Alpha Score
    prob_array=[-1,1]
    alpha_score = classifier.predict_proba(samples).dot(np.array(prob_array))
    
    # Add Alpha Score to rest of the factors
    alpha_score_label = 'AI_ALPHA'
    factors_with_alpha = all_factors.loc[samples.index].copy()
    factors_with_alpha[alpha_score_label] = alpha_score
    
    # Setup data for AlphaLens
    print('Cleaning Data...\n')
    factor_data = build_factor_data(factors_with_alpha[factors + [alpha_score_label]], pricing)
    print('\n-----------------------\n')
    
    # Calculate Factor Returns and Sharpe Ratio
    factor_returns = get_factor_returns(factor_data)
    qr_factor_returns = get_qr_factor_returns(factor_data)
    sharpe_ratio = sharpe_ratio(factor_returns)
    
    # Show Results
    print('             Sharpe Ratios')
    print(sharpe_ratio.round(2))
    plot_factor_returns(factor_returns, ymin, ymax)
    plot_qr_factor_returns(qr_factor_returns)
    plot_factor_rank_autocorrelation(factor_data)
    
    return factor_data

def get_alpha_vector(clf, alpha_factors_today, features, target_label):
    scale = 1
    X = alpha_factors_today[features]
    alpha_score = clf.predict_proba(X).dot(np.array([-1,1]))
    X['AI_ALPHA'] = alpha_score
    alpha_vector = X[['AI_ALPHA']]
    return alpha_vector * scale

def get_alpha_vector2(alpha_factors_today, factor_columns, shape_ratio_value):
    scale = 1
    shape_ratio_value = np.nan_to_num(shape_ratio_value)
    shape_ratio_value = shape_ratio_value / np.sum(shape_ratio_value)
    alpha_factors_today['AI_ALPHA'] = np.dot(alpha_factors_today[factor_columns], shape_ratio_value)
    alpha_vector  = alpha_factors_today[['AI_ALPHA']]
    return scale * alpha_vector

######### Optimization with Constraints by Risk Model #########

def get_lambda(df):
    df.loc[np.isnan(df['adv_tc']), 'adv_tc'] = 1.0e4
    df.loc[df['adv_tc'] == 0, 'adv_tc'] = 1.0e4 

    adv = df['adv_tc']
    
    return 0.1 / adv

class AbstractOptimalHoldings(ABC):    
    @abstractmethod
    def _get_obj(self, weights, alpha_vector, Lambda):
        """
        Get the objective function

        Parameters
        ----------
        weights : CVXPY Variable
            Portfolio weights
        alpha_vector : DataFrame
            Alpha vector

        Returns
        -------
        objective : CVXPY Objective
            Objective function
        """
        
        raise NotImplementedError()
    
    @abstractmethod
    def _get_constraints(self, weights, factor_betas, risk):
        """
        Get the constraints

        Parameters
        ----------
        weights : CVXPY Variable
            Portfolio weights
        factor_betas : 2 dimensional Ndarray
            Factor betas
        risk: CVXPY Atom
            Predicted variance of the portfolio returns

        Returns
        -------
        constraints : List of CVXPY Constraint
            Constraints
        """
        
        raise NotImplementedError()
        
    def _get_risk(self, weights, factor_betas, alpha_vector_index, factor_cov_matrix, idiosyncratic_var_vector):
        f = factor_betas.loc[alpha_vector_index].values.T * weights
        X = factor_cov_matrix
        S = np.diag(idiosyncratic_var_vector.loc[alpha_vector_index].values.flatten())

        return cvx.quad_form(f, X) + cvx.quad_form(weights, S)
    
    def find(self, alpha_vector, factor_betas, factor_cov_matrix, idiosyncratic_var_vector):
        weights = cvx.Variable(len(alpha_vector))
        risk = self._get_risk(weights, factor_betas, alpha_vector.index, factor_cov_matrix, idiosyncratic_var_vector)
        
        obj = self._get_obj(weights, alpha_vector)
        constraints = self._get_constraints(weights, factor_betas.loc[alpha_vector.index].values, risk)
        
        prob = cvx.Problem(obj, constraints)
        prob.solve(max_iters=500, solver='SCS')

        optimal_weights = np.asarray(weights.value).flatten()
        
        return pd.DataFrame(data=optimal_weights, index=alpha_vector.index, columns=['h.opt'])

class OptimalHoldings(AbstractOptimalHoldings):
    def _get_obj(self, weights, alpha_vector):
        assert(len(alpha_vector.columns) == 1)
        objective = cvx.Minimize(-alpha_vector.values.flatten()*weights)
        
        return objective
    
    def _get_constraints(self, weights, factor_betas, risk):

        assert(len(factor_betas.shape) == 2)
        
        #TODO: Implement function
        constraints = [
        risk <= self.risk_cap ** 2,   
        factor_betas.T*weights <= self.factor_max,
        factor_betas.T*weights >= self.factor_min,
        sum(weights) == 0.0,
        sum(cvx.abs(weights)) <= 1.0,
        weights >= self.weights_min,
        weights <= self.weights_max        
        ]
        
        return constraints

    def __init__(self, risk_cap=0.05, factor_max=10.0, factor_min=-10.0, weights_max=0.55, weights_min=-0.55):
        self.risk_cap=risk_cap
        self.factor_max=factor_max
        self.factor_min=factor_min
        self.weights_max=weights_max
        self.weights_min=weights_min

class OptimalHoldingsRegualization(OptimalHoldings):
    def _get_obj(self, weights, alpha_vector):
        """
        Get the objective function

        Parameters
        ----------
        weights : CVXPY Variable
            Portfolio weights
        alpha_vector : DataFrame
            Alpha vector

        Returns
        -------
        objective : CVXPY Objective
            Objective function
        """
        assert(len(alpha_vector.columns) == 1)
        
        objective = cvx.Minimize(-alpha_vector.values.flatten()*weights + self.lambda_reg*cvx.norm(weights,2))
        
        return objective

    def __init__(self, lambda_reg=0.5, risk_cap=0.05, factor_max=10.0, factor_min=-10.0, weights_max=0.55, weights_min=-0.55):
        self.lambda_reg = lambda_reg
        self.risk_cap=risk_cap
        self.factor_max=factor_max
        self.factor_min=factor_min
        self.weights_max=weights_max
        self.weights_min=weights_min
        
class OptimalHoldingsStrictFactor(OptimalHoldings):
    def _get_obj(self, weights, alpha_vector):
        """
        Get the objective function

        Parameters
        ----------
        weights : CVXPY Variable
            Portfolio weights
        alpha_vector : DataFrame
            Alpha vector

        Returns
        -------
        objective : CVXPY Objective
            Objective function
        """
        assert(len(alpha_vector.columns) == 1)
        
        #TODO: Implement function
        objective = cvx.Minimize(cvx.norm(alpha_vector.values.flatten()-weights,2))
        
        return objective

def plot_portfolio_characteristics(weights, alpha_vector, risk_model):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,5))
    weights.plot.bar(legend=None, title='Portfolio % Holdings by Stock', ax=ax1)
    ax1.axes.xaxis.set_visible(False)
    get_factor_exposures(risk_model['factor_betas'], optimal_weights_2).plot.bar(
        title='Portfolio Net Factor Exposures', legend=False, ax=ax2)

    transfer_coef = np.corrcoef(alpha_vector['AI_ALPHA'].values.tolist(), weights[0].values.tolist())[0][1]
    print("Transfer Coefficient: ", transfer_coef)

def get_total_transaction_costs(previous, h_star, Lambda):
    tmp = h_star.merge(previous, left_index=True, right_index=True, how='left')
    tmp = clean_nas(tmp)
    return np.dot(np.asarray((tmp['h.opt'] - tmp['h.opt.previous'])**2), np.asarray(Lambda))