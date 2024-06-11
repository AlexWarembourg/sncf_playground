import polars as pl
from typing import List


class PanelStandardScaler:

    name = "temporal"

    def __init__(self, ts_uid: str, target: str):
        self.norm: pl.DataFrame
        self.ts_uid = ts_uid
        self.target = target

    def fit(self, X: pl.DataFrame) -> None:
        self.norm = X.group_by(self.ts_uid).agg(
            mu=pl.col(self.target).mean(), sigma=pl.col(self.target).std()
        )

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        transformed = X.join(self.norm, how="left", on=[self.ts_uid]).with_columns(
            ((pl.col(self.target) - pl.col("mu")) / pl.col("sigma")).alias(self.target)
        )
        return transformed

    def inverse_transform(self, X: pl.DataFrame, target: str = None) -> pl.DataFrame:
        if target is None:
            target = self.target
        transformed = (
            X.join(self.norm, how="left", on=[self.ts_uid])
            .with_columns(
                ((pl.col(target) * pl.col("sigma")) + pl.col("mu")).alias(target)
            )
            .drop(["mu", "sigma"])
        )
        return transformed
