import pandas as pd
from typing import List
import numpy as np
import pandas as pd
import polars as pl
import time
from numba import njit, prange
from typing import List, Tuple
from scipy.sparse import coo_matrix, csc_matrix, csr_array, vstack
from sklearn.linear_model import LassoCV, Lasso
from pandas.api.types import is_datetime64_any_dtype as is_datetime


# %% Functions to perform forecast reconciliation
def reconcile_forecasts(
    yhat: np.ndarray,
    S: np.ndarray,
    y_train: np.ndarray = None,
    yhat_train: np.ndarray = None,
    method: str = "ols",
    positive: bool = False,
) -> np.ndarray:
    """Optimal reconciliation of hierarchical forecasts using various approaches.

    Based on approaches from:

    ['ols', 'wls_var', 'wls_struct', 'mint_cov', 'mint_shrink']
    Wickramasuriya, S. L., Athanasopoulos, G., & Hyndman, R. J. (2019).
    Optimal forecast reconciliation for hierarchical and grouped time series through
    trace minimization. Journal of the American Statistical Association, 114(526), 804-819.

    ['erm', 'erm_reg', 'erm_bu']
    Ben Taieb, Souhaib, and Bonsoo Koo.
    ‘Regularized Regression for Hierarchical Forecasting Without Unbiasedness Conditions’.
    In Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, 1337–47. Anchorage AK USA: ACM, 2019.
    https://doi.org/10.1145/3292500.3330976.


    :param yhat_test: out-of-sample forecasts for each time series for each timestep of size [n_timeseries x n_timesteps]. These forecasts will be reconciled according to the hierarchy specified by S.
    :type yhat_test: numpy.ndarray
    :param S: summing matrix detailing the hierarchical tree of size [n_timeseries x n_bottom_timeseries]
    :type S: numpy.ndarray
    :param y_train: ground truth for each time series for a set of historical timesteps of size [n_timeseries x n_timesteps_train]. Required when using 'wls_var', 'mint_cov', 'mint_shrink', 'erm', 'erm_reg', 'erm_bu'
    :type y_train: numpy.ndarray, optional
    :param yhat_train: forecasts for each time series for a set of historical timesteps of size [n_timeseries x n_timesteps_residuals]. Required when using 'wls_var', 'mint_cov', 'mint_shrink', 'erm', 'erm_reg', 'erm_bu'
    :type yhat_train: numpy.ndarray, optional
    :param method: reconciliation method, defaults to 'ols'. Options are: 'ols', 'wls_var', 'wls_struct', 'mint_cov', 'mint_shrink', 'erm', 'erm_reg', 'erm_bu'
    :type method: str, optional
    :param positive: Boolean to enforce reconciled forecasts are >= zero, defaults to False.
    :type positive: bool, optional

    :return: ytilde, reconciled forecasts for each time series for each timestep of size [n_timeseries x n_timesteps]
    :rtype: numpy.ndarray

    """
    n_timeseries = S.shape[0]
    n_bottom_timeseries = S.shape[1]
    ms = n_timeseries - n_bottom_timeseries
    assert (
        yhat.shape[0] == n_timeseries
    ), "Forecasts and summing matrix S do not contain the same amount of time series"
    if method in ["wls_var", "mint_cov", "mint_shrink", "erm", "erm_reg", "erm_bu"]:
        assert y_train is not None, f"Method {method} requires you to provide y_train"
        assert (
            yhat_train is not None
        ), f"Method {method} requires you to provide yhat_train"
        assert (
            y_train.shape[0] == n_timeseries
        ), "y_train and summing matrix S should contain the same amount of time series"
        assert (
            yhat_train.shape[0] == n_timeseries
        ), "y_train and summing matrix S should contain the same amount of time series"

    # Prepare arrays for reconciliation
    if method in ["ols", "wls_var", "wls_struct", "mint_cov", "mint_shrink"]:
        J = np.concatenate(
            (np.zeros((n_bottom_timeseries, ms), dtype=np.float32), S[ms:]), axis=1
        )
        Ut = np.concatenate((np.eye(ms, dtype=np.float32), -S[:ms]), axis=1)
    # Select correct weight matrix W according to specified reconciliation method
    if method == "ols":
        # Ordinary least squares, default option. W = np.eye(n_timeseries), thus UtW = Ut @ W = Ut * Wdiag = Ut
        UtW = Ut
    elif method == "wls_struct":
        # Weighted least squares using structural scaling. W matrix has non-zero elements on diagonal only.
        Wdiag = np.sum(S, axis=1)
        UtW = Ut * Wdiag
    elif method == "wls_var":
        # Weighted least squares using variance scaling. W matrix has non-zero elements on diagonal only.
        residuals = yhat_train - y_train
        Wdiag = np.sum(residuals**2, axis=1) / residuals.shape[1]
        UtW = Ut * Wdiag
    elif method == "mint_cov":
        # Trace minimization using the empirical error covariance matrix
        residuals = yhat_train - y_train
        W = np.cov(residuals)
        UtW = Ut @ W
    elif method == "mint_shrink":
        # Trace minimization using the shrunk empirical covariance matrix
        residuals = yhat_train - y_train
        residuals_mean = np.mean(residuals, axis=1)
        residuals_std = np.maximum(np.std(residuals, axis=1), 1e-6)
        W = shrunk_covariance_schaferstrimmer(residuals, residuals_mean, residuals_std)
        UtW = Ut @ W
    elif method == "erm":
        # Ref. eq. 18, 19 and 25 of Taieb, 2019.
        Bt = np.linalg.inv(S.T @ S) @ S.T @ y_train
        P = (np.linalg.pinv(yhat_train.T) @ Bt.T).T
    elif method == "erm_reg":
        X = np.kron(S, yhat_train.T)
        X = np.asfortranarray(X, dtype=np.float64)
        z = y_train.reshape(-1)
        lasso = LassoCV(selection="cyclic", n_jobs=-1)
        lasso.fit(X, z)
        P = lasso.coef_.reshape(S.shape).T
    elif method == "erm_bu":
        X = np.kron(S, yhat_train.T)
        X = np.asfortranarray(X, dtype=np.float64)
        Pbu = np.zeros_like(S)
        Pbu[ms:] = S[ms:]
        z = y_train.reshape(-1) - X @ Pbu.reshape(-1)
        lasso = LassoCV(selection="cyclic", n_jobs=-1)
        lasso.fit(X, z)
        Beta = lasso.coef_
        P = Beta + Pbu.reshape(-1)
        P = P.reshape(S.shape).T
    else:
        raise NotImplementedError(
            "Method not implemented. Options are: ['ols', 'wls_var', 'wls_struct', 'mint_cov', 'mint_shrink', 'erm', 'erm_reg', 'erm_bu']"
        )

    # Compute P for non-ERM methods
    if method in ["ols", "wls_var", "wls_struct", "mint_cov", "mint_shrink"]:
        P = (
            J
            - np.linalg.solve(
                UtW[:, ms:] @ Ut.T[ms:] + UtW[:, :ms], UtW[:, ms:] @ J.T[ms:]
            ).T
            @ Ut
        )
    # Compute reconciled forecasts
    ytilde = (yhat.T @ P.T @ S.T).T
    # Clamp to zero if required
    if positive:
        ytilde = np.maximum(ytilde, 0)

    return ytilde


def get_aggregations(sectional_level=False):
    if sectional_level:
        cross_sectional_aggregations = [
            ["cat_id_enc"],
            ["dept_id_enc"],
            ["item_id_enc"],
        ]
    temporal_aggregations = [["year"], ["year", "month"], ["year", "week"]]
    return cross_sectional_aggregations, temporal_aggregations


def hierarchy_cross_sectional(
    df: pd.DataFrame,
    aggregations: List[List[str]],
    sparse: bool = False,
    name_bottom_timeseries: str = "bottom_timeseries",
) -> pd.DataFrame:
    """Given a dataframe of timeseries and columns indicating their groupings, this function calculates a cross-sectional hierarchy according to a set of specified aggregations for the time series.

    :param df: DataFrame containing information about time series and their groupings
    :type df: pd.DataFrame
    :param aggregations: List of Lists containing the aggregations required.
    :type aggregations: List[List[str]]
    :param sparse: Boolean to indicate whether the returned summing matrix should be backed by a SparseArray (True) or a regular Numpy array (False), defaults to False.
    :type sparse: bool
    :param name_bottom_timeseries: name for the bottom level time series in the hierarchy, defaults to 'bottom_timeseries'.
    :type name_bottom_timeseries: str

    :return: df_Sc, output dataframe containing the summing matrix of shape [(n_bottom_timeseries + n_aggregate_timeseries) x n_bottom_timeseries]. The number of aggregate time series is the result of applying all the required aggregations.
    :rtype: pd.DataFrame filled with np.float32

    """
    # Check whether aggregations are in the df
    aggregation_cols_in_aggregations = list(
        dict.fromkeys([col for cols in aggregations for col in cols])
    )
    for col in aggregation_cols_in_aggregations:
        assert col in df.columns, f"Column {col} in aggregations not present in df"
    # Find the unique aggregation columns from the given set of aggregations
    levels = df[aggregation_cols_in_aggregations].drop_duplicates()
    levels[name_bottom_timeseries] = (
        levels[aggregation_cols_in_aggregations].astype(str).agg("-".join, axis=1)
    )
    levels = levels.sort_values(by=name_bottom_timeseries).reset_index(drop=True)
    n_bottom_timeseries = len(levels)
    aggregations_total = aggregations + [[name_bottom_timeseries]]
    # Check if we have not introduced redundant columns. If so, remove that column.
    for col, n_uniques in levels.nunique().items():
        if col != name_bottom_timeseries and n_uniques == n_bottom_timeseries:
            levels = levels.drop(columns=col)
            aggregations_total.remove([col])
    # Create summing matrix for all aggregation levels
    ones = np.ones(n_bottom_timeseries, dtype=np.float32)
    idx_range = np.arange(n_bottom_timeseries)
    df_S_aggs = []
    # Create summing matrix (=row vector) for top level (=total) series
    df_S_top = pd.DataFrame(ones[None, :], index=["Total"])
    df_S_top = pd.concat({"Total": df_S_top}, names=["Aggregation", "Value"])
    df_S_aggs.append(df_S_top)
    for aggregation in aggregations_total:
        aggregation_name = "-".join(aggregation)
        agg = pd.Categorical(levels[aggregation].astype(str).agg("-".join, axis=1))
        S_agg_sp = coo_matrix((ones, (agg.codes, idx_range)))
        if sparse:
            S_agg = pd.DataFrame.sparse.from_spmatrix(S_agg_sp, index=agg.categories)
        else:
            S_agg = pd.DataFrame(S_agg_sp.todense(), index=agg.categories)
        S_agg = pd.concat(
            {f"{aggregation_name}": S_agg}, names=["Aggregation", "Value"]
        )
        df_S_aggs.append(S_agg)

    # Stack all summing matrices: top, aggregations, bottom
    df_S = pd.concat(df_S_aggs)
    df_S.columns = levels[name_bottom_timeseries]

    return df_S


def hierarchy_temporal(
    df: pd.DataFrame,
    time_column: str,
    aggregations: List[List[str]],
    sparse: bool = False,
) -> pd.DataFrame:
    """Given a dataframe of timeseries and a time_column indicating the timestamp of each series, this function calculates a temporal hierarchy according to a set of specified aggregations for the time series.

    :param df: DataFrame containing information about time series and their groupings
    :type df: pd.DataFrame
    :param time_column: String containing the column name that contains the time column of the timeseries
    :type time_column: str
    :param aggregations: List of Lists containing the aggregations required.
    :type aggregations: List[List[str]]
    :param sparse: Boolean to indicate whether the returned summing matrix should be backed by a SparseArray (True) or a regular Numpy array (False), defaults to False.
    :type sparse: bool

    :return: df_St, output dataframe containing a summing matrix of shape [n_timesteps x (n_timesteps + n_aggregate_timesteps)]. The number of aggregate timesteps is the result of applying all the required temporal aggregations.
    :rtype: pd.DataFrame filled with np.float32

    """
    assert time_column in df.columns, "The time_column is not a column in the dataframe"
    assert is_datetime(
        df[time_column]
    ), "The time_column should be a datetime64-dtype. Use `pd.to_datetime` to convert objects to the correct datetime format."
    # Check whether aggregations are in the df
    aggregation_cols_in_aggregations = list(
        dict.fromkeys([col for cols in aggregations for col in cols])
    )
    for col in aggregation_cols_in_aggregations:
        assert col in df.columns, f"Column {col} in aggregations not present in df"
    # Find the unique aggregation columns from the given set of aggregations
    levels = df[aggregation_cols_in_aggregations + [time_column]].drop_duplicates()
    levels = levels.sort_values(by=time_column).reset_index(drop=True)
    n_bottom_timestamps = len(levels)
    aggregations_total = aggregations + [[time_column]]
    # Create summing matrix for all aggregation levels
    ones = np.ones(n_bottom_timestamps, dtype=np.float32)
    idx_range = np.arange(n_bottom_timestamps)
    df_S_aggs = []
    for aggregation in aggregations_total:
        aggregation_name = "-".join(aggregation)
        agg = pd.Categorical(levels[aggregation].astype(str).agg("-".join, axis=1))
        S_agg_sp = coo_matrix((ones, (agg.codes, idx_range)))
        if sparse:
            S_agg = pd.DataFrame.sparse.from_spmatrix(S_agg_sp, index=agg.categories)
        else:
            S_agg = pd.DataFrame(S_agg_sp.todense(), index=agg.categories)
        S_agg = pd.concat(
            {f"{aggregation_name}": S_agg}, names=["Aggregation", "Value"]
        )
        df_S_aggs.append(S_agg)

    # Stack all summing matrices: aggregations, bottom
    df_S = pd.concat(df_S_aggs)
    df_S.columns = levels[time_column]


def apply_reconciliation_methods(
    forecasts: pd.DataFrame,
    df_S: pd.DataFrame,
    y_train: pd.DataFrame,
    yhat_train: pd.DataFrame,
    methods: List[str] = None,
    positive: bool = False,
    return_timing: bool = False,
) -> pd.DataFrame:
    """Apply all hierarchical forecasting reconciliation methods to a set of forecasts.

    :param forecasts: dataframe containing forecasts for all aggregations
    :type forecasts: pd.DataFrame
    :param df_S: Dataframe containing the summing matrix for all aggregations in the hierarchy.
    :type df_S: pd.DataFrame
    :param y_train: dataframe containing the ground truth on the training set for all timeseries.
    :type y_train: pd.DataFrame
    :param yhat_train: dataframe containing the forecasts on the training set for all timeseries.
    :type yhat_train: pd.DataFrame
    :param methods: list containing which reconciliation methods to be applied, defaults to None. Choose from: 'ols', 'wls_var', 'wls_struct', 'mint_cov', 'mint_shrink', 'erm', 'erm_reg', 'erm_bu'. None means all methods will be applied.
    :type methods: List[str]
    :param positive: Boolean to enforce reconciled forecasts are >= zero, defaults to False.
    :type positive: bool, optional
    :param return_timing: Flag to return execution time for reconciliation methods
    :type return_timing: bool, optional

    :return: forecasts_methods, dataframe containing forecasts for all reconciliation methods
    :rtype: pd.DataFrame

    """
    forecasts_method = pd.concat({"base": forecasts}, names=["Method"])
    cols = forecasts_method.columns
    # Convert to float32
    yhat = forecasts_method.values.astype("float32")
    S = df_S.values.astype("float32")
    y_train = y_train.values.astype("float32")
    yhat_train = yhat_train.values.astype("float32")
    # Apply all reconciliation methods
    if methods == None:
        methods = [
            "ols",
            "wls_struct",
            "wls_var",
            "mint_cov",
            "mint_shrink",
            "erm",
            "erm_reg",
            "erm_bu",
        ]
    forecasts_methods = []
    forecasts_methods.append(forecasts_method)
    timings = {}
    for method in methods:
        t0 = time.perf_counter()
        ytilde = reconcile_forecasts(
            yhat, S, y_train, yhat_train, method=method, positive=positive
        )
        t1 = time.perf_counter()
        print(f"Method {method}, reconciliation time: {t1-t0:.4f}s")
        timings[method] = t1 - t0
        forecasts_method = pd.DataFrame(
            data=ytilde, index=forecasts.index, columns=cols
        )
        forecasts_method = pd.concat({f"{method}": forecasts_method}, names=["Method"])
        forecasts_methods.append(forecasts_method)

    forecasts_methods = pd.concat(forecasts_methods)
    if return_timing:
        return forecasts_methods, timings
    else:
        return forecasts_methods
