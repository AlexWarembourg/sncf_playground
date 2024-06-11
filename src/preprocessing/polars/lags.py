import polars as pl
from typing import List, Union, Dict, Optional


def reference_shift_from_day(
    df: pl.DataFrame, target_col: str, dayofyear_col: str, ts_uid: str
) -> pl.DataFrame:
    return df.with_columns(
        pl.col(target_col)
        .shift(1)
        .over([ts_uid, dayofyear_col])
        .alias(f"reference_{target_col}")
    )


def pl_compute_lagged_features(
    df: pl.DataFrame, target_col: str, lag_list: List[int], ts_uid: str, date: str
) -> pl.DataFrame:
    sort_keys = [date] + [ts_uid] if isinstance(ts_uid, str) else ts_uid
    concat_keys = ts_uid if isinstance(ts_uid, str) else "@".join(ts_uid)
    df = df.sort(sort_keys)
    lags_expr = [
        pl.col(target_col)
        .shift(lag)
        .over(ts_uid)
        .alias(f"ar_{target_col}_lag{lag}_{concat_keys}")
        for lag in lag_list
    ]
    df = df.with_columns(lags_expr)
    return df


def pl_compute_moving_features(
    df: pl.DataFrame,
    target_col: str,
    func_list: List[str],
    shift_list: List[int],
    win_list: List[int],
    ts_uid: Union[List[str], str],
    date: str,
    horizon: int,
) -> pl.DataFrame:
    sort_keys = [date] + [ts_uid] if isinstance(ts_uid, str) else ts_uid
    concat_keys = ts_uid if isinstance(ts_uid, str) else "@".join(ts_uid)
    df = df.sort(sort_keys)
    if min(shift_list) < horizon:
        raise ValueError("We must shift at least by the length of horizon")

    windows_expr = []
    for win in win_list:
        for shift in shift_list:
            for func in func_list:
                w_expr_ = (
                    getattr(pl.col(target_col).shift(shift), func)(win)
                    .over(ts_uid)
                    .alias(
                        f"ar_{target_col}_win{win}_shift{shift}_{func}_{concat_keys}"
                    )
                )
                windows_expr.append(w_expr_)
    df = df.with_columns(windows_expr)
    return df


def compute_autoreg_features(
    data: pl.DataFrame,
    target_col: str,
    auto_reg_params: Dict[str, Dict[str, List[Union[str, int]]]],
    date_str: str,
):
    for group_fe in auto_reg_params.keys():
        params_group = auto_reg_params.get(group_fe)
        # compute in loop.
        data = data.pipe(
            pl_compute_lagged_features,
            target_col=target_col,
            lag_list=params_group["lags"],
            ts_uid=params_group["groups"],
            date=date_str,
        ).pipe(
            pl_compute_moving_features,
            target_col=target_col,
            func_list=params_group["funcs"],
            win_list=params_group["wins"],
            shift_list=params_group["shifts"],
            ts_uid=params_group["groups"],
            date=date_str,
            horizon=params_group["horizon"],
        )
    return data
