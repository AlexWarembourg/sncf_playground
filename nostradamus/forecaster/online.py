from polars import DataFrame, Series


class OnlineLGBM:

    def __init__(self, model):
        self.model = model

    def update(self, X: DataFrame, y: Series, sample_weights: Series) -> None:
        self.model.fit(
            X,
            y,
            sample_weight=sample_weights,
            init_model=self.model.get_booster(),
        )
