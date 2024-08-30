from typing import Any, Dict, Tuple

import io
from pathlib import Path

import polars as pl
import requests
import streamlit as st
from datetime import timedelta
import numpy as np
import toml
from PIL import Image
from attendance.experiment.project_utils import load_data
import gc


def get_project_root() -> str:
    """Returns project root path.

    Returns
    -------
    str
        Project root path.
    """
    return str(Path(__file__).resolve().parents[2])


@st.cache_data
def load_dataset(project_root) -> pl.DataFrame:
    """Loads dataset from user's file system as a pandas dataframe.

    Parameters
    ----------
    file
        Uploaded dataset file.
    load_options : Dict
        Dictionary containing separator information.

    Returns
    -------
    pd.DataFrame
        Loaded dataset.
    """
    try:
        train_data, _, _ = load_data(project_root)
        # cut older obs
        train_data = train_data.filter(
            pl.col("date").cast(pl.Date)
            >= pl.col("date").cast(pl.Date).min() + timedelta(days=364 * 2)
        )

        cols = ["y_hat", "lower_band", "upper_band"]
        statsmodel_valid = pl.read_csv(
            project_root / "out/cqr_lgb.csv", infer_schema_length=1000
        ).select(["date", "station"] + cols)

        true_y = (
            train_data.filter(
                pl.col("date").cast(pl.String).str.to_datetime()
                > pl.col("date").cast(pl.String).str.to_datetime().max() - timedelta(days=60)
            )
            .select(["station", "date", "y"])
            .rename({"y": "true_y"})
        )

        output_data = (
            pl.concat(
                (
                    train_data.filter(
                        pl.col("date").cast(pl.String).str.to_datetime()
                        < pl.col("date").cast(pl.String).str.to_datetime().max()
                        - timedelta(days=60)
                    )
                    .with_columns([pl.lit(np.nan).alias(col) for col in cols])
                    .with_columns(pl.lit("historical").alias("type"))
                    .select(["date", "station", "y", "type"] + cols),
                    statsmodel_valid.with_columns(
                        pl.lit(np.nan).alias("y"), pl.lit("forecast").alias("type")
                    ).select(["date", "station", "y", "type"] + cols),
                ),
                how="vertical_relaxed",
            )
            .with_columns(pl.col("date").cast(pl.String).str.to_datetime())
            .join(true_y, how="left", on=["station", "date"])
        )

        del train_data, statsmodel_valid
        gc.collect()
        output_data = (
            output_data.rename(
                {
                    "lower_band": "lower_bound",
                    "upper_band": "upper_bound",
                }
            )
            .with_columns(pl.col("lower_bound").clip_min(pl.lit(0)))
            .with_columns(pl.col("upper_bound").clip_min(pl.col("lower_bound")))
            .with_columns(
                pl.when(
                    (pl.col("true_y").is_between(pl.col("lower_bound"), pl.col("upper_bound")))
                    & (pl.col("type") == "forecast")
                )
                .then(pl.lit("normal"))
                .otherwise(pl.lit("anomaly"))
                .alias("state")
            )
        )
        return output_data.to_pandas()
    except:
        st.error(
            "This file can't be converted into a dataframe. Please import a csv file with a valid separator."
        )
        st.stop()


def load_config(
    config_streamlit_filename: str,
    config_instructions_filename: str,
    config_readme_filename: str,
) -> Tuple[Dict[Any, Any], Dict[Any, Any], Dict[Any, Any]]:
    """Loads configuration files.

    Parameters
    ----------
    config_streamlit_filename : str
        Filename of lib configuration file.
    config_instructions_filename : str
        Filename of custom config instruction file.
    config_readme_filename : str
        Filename of readme configuration file.

    Returns
    -------
    dict
        Lib configuration file.
    dict
        Readme configuration file.
    """
    config_streamlit = toml.load(Path(get_project_root()) / f"config/{config_streamlit_filename}")
    config_instructions = toml.load(
        Path(get_project_root()) / f"config/{config_instructions_filename}"
    )
    config_readme = toml.load(Path(get_project_root()) / f"config/{config_readme_filename}")
    return dict(config_streamlit), dict(config_instructions), dict(config_readme)


def load_image(image_name: str) -> Image:
    """Displays an image.

    Parameters
    ----------
    image_name : str
        Local path of the image.

    Returns
    -------
    Image
        Image to be displayed.
    """
    return Image.open(f"app/assets/{image_name}")
