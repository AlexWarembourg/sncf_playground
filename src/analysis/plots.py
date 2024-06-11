import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


class TopTSAnalysis:

    @staticmethod
    def top_level_plot(full_data, time, target):
        aggregate_ts = full_data.groupby(time)[target].sum().sort_index().reset_index()
        aggregate_ts["week"] = aggregate_ts[time].dt.isocalendar().week
        aggregate_ts["year"] = aggregate_ts[time].dt.isocalendar().year
        week_agg = aggregate_ts.groupby(["year", "week"])[[target]].mean().unstack(-1)
        fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(14, 7))
        ax[0].plot(
            aggregate_ts["date"],
            aggregate_ts["y"],
            label="historical",
            color="black",
            alpha=0.9,
        )
        ax[0].plot(
            aggregate_ts["date"],
            aggregate_ts["y"].rolling(364).mean(),
            label="yearly_rolling",
            color="green",
            alpha=0.9,
        )
        sns.heatmap(week_agg, axes=ax[1], cmap="viridis")
        fig.tight_layout()
        ax[0].legend()
        plt.show()
