from typing import Any, Dict, Tuple

import io
from pathlib import Path

import polars as pl
import requests
import streamlit as st
import toml
from PIL import Image
from attendance.experiment.project_utils import load_data


def get_project_root() -> str:
    """Returns project root path.

    Returns
    -------
    str
        Project root path.
    """
    return str(Path(__file__).parent.parent.parent)


@st.cache_data(ttl=300)
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
        return load_data(project_root)
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
    config_streamlit = toml.load(
        Path(get_project_root()) / f"config/{config_streamlit_filename}"
    )
    config_instructions = toml.load(
        Path(get_project_root()) / f"config/{config_instructions_filename}"
    )
    config_readme = toml.load(
        Path(get_project_root()) / f"config/{config_readme_filename}"
    )
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
