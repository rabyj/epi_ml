"""Test if all main scripts imports run without errors."""
# pylint: disable=unused-import, import-outside-toplevel
from __future__ import annotations


def import_test():
    """Test if all main scripts imports run without errors."""
    import argparse
    import glob
    import json
    import os
    import sys
    import warnings
    from functools import partial
    from pathlib import Path
    from typing import Dict

    import comet_ml  # needed because special snowflake # pylint: disable=unused-import
    import numpy as np
    import pytorch_lightning as pl  # in case GCC or CUDA needs it # pylint: disable=unused-import
    import pytorch_lightning.callbacks as pl_callbacks
    import torch
    from pytorch_lightning import loggers as pl_loggers
    from torch.utils.data import DataLoader, TensorDataset

    import src.python.core.estimators as estimators
    from src.python.argparseutils.DefaultHelpParser import (
        DefaultHelpParser as ArgumentParser,
    )
    from src.python.argparseutils.directorychecker import DirectoryChecker
    from src.python.core import analysis, data, metadata
    from src.python.core.data import DataSet, UnknownData
    from src.python.core.data_source import EpiDataSource
    from src.python.core.epiatlas_treatment import EpiAtlasFoldFactory
    from src.python.core.hdf5_loader import Hdf5Loader
    from src.python.core.lgbm import tune_lgbm
    from src.python.core.model_pytorch import LightningDenseClassifier
    from src.python.core.trainer import MyTrainer, define_callbacks
    from src.python.utils.check_dir import create_dirs
    from src.python.utils.time import time_now
