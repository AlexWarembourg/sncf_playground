from typing import Any, Dict, Tuple

import io
from pathlib import Path

import polars as pl
import requests
import streamlit as st
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
    return str(Path(__file__).parent.parent.parent)


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

        cols = ["HoltWinters", "HoltWinters-lo-95", "HoltWinters-hi-95"]
        statsmodel_valid = (
            pl.read_csv(
                project_root / "out/nixtla_validation.csv", separator=",", infer_schema_length=1000
            )
            .rename({"unique_id": "station", "ds": "date"})
            .select(["date", "station"] + cols)
        )
        output_data = pl.concat(
            (
                train_data.with_columns([pl.lit(np.nan).alias(col) for col in cols])
                .with_columns(pl.lit("historical").alias("type"))
                .select(["date", "station", "y"] + cols),
                statsmodel_valid.with_columns(
                    pl.lit(np.nan).alias("y"), pl.lit("forecast").alias("type")
                ).select(["date", "station", "y", "type"] + cols),
            ),
            how="vertical_relaxed",
        )
        del train_data, statsmodel_valid
        gc.collect()
        output_data = output_data.rename(
            {
                "HoltWinters": "y_hat",
                "HoltWinters-lo-95": "lower_bound",
                "HoltWinters-hi-95": "upper_bound",
            }
        ).with_columns(
            pl.when(pl.col("y_hat").is_between(pl.col("lower_bound"), pl.col("upper_bound")))
            .then(pl.lit("normal"))
            .otherwise(pl.lit("anomaly"))
            .alias("anomaly")
        )
        return output_data
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
    return Image.open(Path(get_project_root()) / f"attendance/app/assets/{image_name}")
