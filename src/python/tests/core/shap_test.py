"""Test SHAP related modules."""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Tuple

warnings.filterwarnings("ignore", message=".*IPython display.*")

import numpy as np
import pytest

from epi_ml.core.data import DataSet, UnknownData
from epi_ml.core.hdf5_loader import Hdf5Loader
from epi_ml.core.model_pytorch import LightningDenseClassifier
from epi_ml.core.shap_values import SHAP_Analyzer, SHAP_Handler


class Test_SHAP_Handler:
    """Class to test SHAP_Handler class."""

    @pytest.fixture
    def logdir(self, mk_logdir) -> Path:
        """Test logdir"""
        return mk_logdir("shap")

    @pytest.fixture
    def handler(self, logdir: Path, test_NN_model) -> SHAP_Handler:
        """SHAP_Handler instance"""
        return SHAP_Handler(test_NN_model, logdir)

    @pytest.fixture
    def mock_shap_values(self, test_epiatlas_dataset: DataSet) -> List[np.ndarray]:
        """Mock shape values for evaluation on two examples."""
        shap_values = [
            np.zeros(test_epiatlas_dataset.validation.signals.shape)
            for _ in test_epiatlas_dataset.classes
        ]
        return shap_values

    @pytest.fixture
    def fake_ids(self, test_epiatlas_dataset: DataSet):
        """Fake signal ids"""
        num_signals = test_epiatlas_dataset.validation.num_examples
        return [f"id{i}" for i in range(num_signals)]

    def test_compute_NN(self, handler: SHAP_Handler, test_epiatlas_dataset: DataSet):
        """Test shapes of return of compute_NN method."""
        dset = test_epiatlas_dataset
        _, shap_values = handler.compute_NN(
            background_dset=dset.train, evaluation_dset=dset.validation, save=False  # type: ignore
        )
        print(f"len(shap_values) = {len(shap_values)}")
        print(f"shap_values[0].shape = {shap_values[0].shape }")

        n_signals, n_dims = dset.validation.signals.shape[:]
        assert shap_values[0].shape == (n_signals, n_dims)

    def test_save_load_csv(self, handler: SHAP_Handler, mock_shap_values, fake_ids):
        """Test pickle save/load methods."""
        shaps = mock_shap_values[0]
        path = handler.save_to_csv(shaps, fake_ids, name="test")

        data = handler.load_from_csv(path)
        assert list(data.index) == fake_ids
        assert np.array_equal(shaps, data.values)

    def test_save_to_csv_list_input(
        self, handler: SHAP_Handler, mock_shap_values, fake_ids
    ):
        """Test effect of list input."""
        shap_values_matrix = [mock_shap_values[0]]
        name = "test_csv"

        with pytest.raises(ValueError):
            handler.save_to_csv(shap_values_matrix, fake_ids, name)  # type: ignore

    def test_create_filename(self, handler: SHAP_Handler):
        """Test filename creation method. Created by GPT4 lol."""
        ext = "pickle"
        name = "test_name"

        filename = handler._create_filename(ext, name)  # pylint: disable=protected-access
        assert filename.name.startswith(f"shap_{name}_")
        assert filename.name.endswith(f".{ext}")
        assert filename.parent == Path(handler.logdir)


class Test_SHAP_Analyzer:
    """Class to test SHAP_Analyzer class."""

    @pytest.fixture()
    def test_folder(self, mk_logdir) -> Path:
        """Return temp shap test folder."""
        return mk_logdir("shap_test")

    @pytest.fixture()
    def saccer3_dir(self) -> Path:
        """saccer3 params dir"""
        return Path(__file__).parent.parent / "fixtures" / "saccer3"

    @pytest.fixture
    def saccer3_model(self, saccer3_dir: Path) -> LightningDenseClassifier:
        """saccer3 test model"""
        saccer3_model_dir = saccer3_dir / "model"
        return LightningDenseClassifier.restore_model(saccer3_model_dir)

    @pytest.fixture
    def saccer3_signals(self, saccer3_dir: Path) -> Dict:
        """saccer3 epigenetic signals"""
        chrom_file = saccer3_dir / "saccer3.can.chrom.sizes"
        hdf5_filelist = saccer3_dir / "hdf5_10kb_all_none.list"
        hdf5_loader = Hdf5Loader(chrom_file=chrom_file, normalization=True)
        hdf5_loader.load_hdf5s(hdf5_filelist, strict=True)
        return hdf5_loader.signals

    @pytest.fixture
    def test_dsets(self, saccer3_signals: Dict) -> Tuple[UnknownData, UnknownData]:
        """Return background and evaluation datasets."""
        background_signals = list(saccer3_signals.values())[0:12]
        eval_signals = background_signals[10:12]

        background_dset = UnknownData(
            ids=range(len(background_signals)),
            x=background_signals,
            y=np.zeros(len(background_signals)),
            y_str=["NA" for _ in range(len(background_signals))],
        )

        eval_dset = UnknownData(
            ids=[-i for i in range(len(eval_signals))],
            x=eval_signals,
            y=np.zeros(len(eval_signals)),
            y_str=["NA" for _ in range(len(eval_signals))],
        )

        return background_dset, eval_dset

    def test_verify_shap_values_coherence(
        self,
        saccer3_model: LightningDenseClassifier,
        test_folder: Path,
        test_dsets: Tuple[UnknownData, UnknownData],
    ):
        """Test compute_shap_values method."""
        shap_handler = SHAP_Handler(model=saccer3_model, logdir=test_folder)

        explainer, shap_values = shap_handler.compute_NN(
            background_dset=test_dsets[0],
            evaluation_dset=test_dsets[1],
            save=True,
            name="test",
        )

        shap_analyzer = SHAP_Analyzer(saccer3_model, explainer)
        shap_analyzer.verify_shap_values_coherence(shap_values, test_dsets[1])
        assert True