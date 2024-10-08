import polars as pl
from datetime import timedelta


def freeze_validation_set(
    df: pl.DataFrame,
    date: str,
    val_size: int,
    return_train: bool = True,
) -> pl.DataFrame:
    max_dt = df[date].max()
    cut = max_dt - timedelta(days=val_size)
    valid = df.filter(pl.col(date) > cut)
    if return_train:
        train = df.filter(pl.col(date) <= cut)
        return train, valid
    else:
        return valid
