import polars as pl


def error_between_forecast(data, y, yhat_1, yhat_2, ts_uid):
    error_df = (
        data.group_by(ts_uid)
        .sum()
        .with_columns(
            pl.col(yhat_1).round(1),
            pl.col(yhat_2).round(1),
            ((pl.col(y) - pl.col(yhat_1)) / pl.col(y))
            .alias(f"{yhat_1}_model_error (%)")
            .round(2)
            .abs()
            * 100,
            ((pl.col(y) - pl.col(yhat_2)) / pl.col(y))
            .alias(f"{yhat_2}_model_error (%)")
            .round(2)
            .abs()
            * 100,
        )
        .select(["station", f"{yhat_1}_model_error (%)", f"{yhat_2}_model_error (%)"])
    )
    return error_df
