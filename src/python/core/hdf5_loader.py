"""Module for hdf5 loading handling."""
# pylint: disable=unexpected-keyword-arg
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path
from typing import Dict

import h5py
import numpy as np


class Hdf5Loader(object):
    """Handles loading/creating signals from hdf5 files"""

    def __init__(self, chrom_file, normalization: bool):
        self._normalization = normalization
        self._chroms = Hdf5Loader.load_chroms(chrom_file)
        self._files = {}
        self._signals = {}

    @property
    def loaded_files(self) -> Dict[str, Path]:
        """Return a {md5:path} dict with last loaded files."""
        return self._files

    @property
    def signals(self) -> Dict[str, np.ndarray]:
        """Return a {md5:signal dict} with the last loaded signals,
        where the signal has concanenated chromosomes, and is normalized if set so.
        """
        return self._signals

    @staticmethod
    def load_chroms(chrom_file):
        """Return sorted chromosome names list."""
        with open(chrom_file, "r", encoding="utf-8") as file:
            chroms = []
            for line in file:
                line = line.rstrip()
                if line:
                    chroms.append(line.split()[0])
        chroms.sort()
        return chroms

    @staticmethod
    def read_list(data_file: Path) -> Dict[str, Path]:
        """Return {md5:file} dict from file of paths list."""
        with open(data_file, "r", encoding="utf-8") as file_of_paths:
            files = {}
            for path in file_of_paths:
                path = Path(path.rstrip())
                files[Hdf5Loader.extract_md5(path)] = path
        return files

    def load_hdf5s(
        self, data_file: Path, md5s=None, verbose=True, strict=False
    ) -> Hdf5Loader:
        """Load hdf5s from path list file, into self.signals
        If a list of md5s is given, load only the corresponding files.
        Normalize if internal flag set so.

        If strict, will raise OSError if an hdf5 cannot be opened.

        Loads them as float32.
        """
        files = self.read_list(data_file)

        new_parent = os.getenv("HDF5_PARENT", "hdf5s")
        files = Hdf5Loader.adapt_to_environment(files, new_parent)
        self._files = files

        # Remove undesired files
        if md5s is not None:
            chosen_md5s = set(md5s)
            # fmt: off
            files = {
                md5: path for md5, path in files.items()
                if md5 in chosen_md5s
                }  # fmt: on

            absent_md5s = chosen_md5s - set(files.keys())
            if absent_md5s and verbose:
                print("Following given md5s are absent of hdf5 list")
                for md5 in absent_md5s:
                    print(md5)

        # Load hdf5s and concatenate chroms into signals
        signals = {}
        for md5, file in files.items():

            # Trying to open hdf5 file.
            with h5py.File(file, "r") as f:
                try:
                    signals[md5] = self._normalize(self._read_hdf5(f, md5))
                except OSError as err:
                    print(f"Error occured with {md5}: {file}. {err}", file=sys.stderr)
                    if strict:
                        print(
                            "Strict hdf5 loading policy true, raising original error.",
                            file=sys.stderr,
                        )
                        raise err from None
                    else:
                        continue

        self._signals = signals

        return self

    def _read_hdf5(self, file: h5py.File, md5: str) -> np.ndarray:
        """Read and return concatenated genome signal for open hdf5 file."""
        try:
            hdf5_data = file[md5]
        except KeyError:
            header = list(file.keys())[0]
            warnings.warn(
                f"Cannot read file directly with {md5}, header is different. Using header {header}."
            )
            hdf5_data = file[header]

        chrom_signals = [hdf5_data[chrom][...] for chrom in self._chroms]  # type: ignore
        return np.concatenate(chrom_signals, dtype=np.float32)  # type: ignore

    def _normalize(self, array: np.ndarray) -> np.ndarray:
        if self._normalization:
            return (array - array.mean()) / array.std()
        else:
            return array

    @staticmethod
    def extract_md5(file_name: Path):
        """Extract the md5 string from file path with specific naming convention."""
        return file_name.name.split("_")[0]

    @staticmethod
    def adapt_to_environment(
        files: Dict[str, Path], new_parent="hdf5s"
    ) -> Dict[str, Path]:
        """Change files paths if they exist on cluster scratch.

        Files : {md5:path} dict.
        new_parent : directory after $SLURM_TMPDIR.
        """
        local_tmp = Path(os.getenv("SLURM_TMPDIR", "./bleh")) / new_parent

        if local_tmp.exists():
            print(f"Using files in {local_tmp}")
            for md5, path in list(files.items()):
                files[md5] = local_tmp / Path(path).name

        return files
