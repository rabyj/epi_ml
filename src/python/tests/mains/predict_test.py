"""Test module for predict.py."""
import sys
from pathlib import Path

import pytest

from epi_ml.predict import main as main_module


@pytest.fixture(name="test_dir")
def fixture_test_dir(mk_logdir) -> Path:
    """Make temp logdir for tests."""
    return mk_logdir("predict")


def test_training(test_dir: Path):
    """Test if basic training succeeds."""
    current_dir = Path(__file__).parent.resolve()

    fixtures_dir = current_dir.parent / "fixtures" / "saccer3"
    file_list = fixtures_dir / "hdf5_10kb_all_none.list"
    chroms = fixtures_dir / "saccer3.can.chrom.sizes"

    sys.argv = [
        "predict.py",
        str(file_list),
        str(chroms),
        str(test_dir),  # logdir
        "--offline",
        "--model",
        str(fixtures_dir),
    ]
    main_module()
