"""Compute PCA for some hardcoded datasets. (dataset selection done in .sh script)"""
# pylint: disable=import-error, redefined-outer-name, use-dict-literal, too-many-lines, unused-import, unused-argument, too-many-branches
from __future__ import annotations

import argparse
import os
import warnings
from importlib import metadata
from pathlib import Path

import numpy as np
import skops.io as skio
from sklearn.decomposition import IncrementalPCA

from epi_ml.core.hdf5_loader import Hdf5Loader


def parse_arguments() -> argparse.Namespace:
    """argument parser for command line"""
    # fmt: on
    arg_parser = argparse.ArgumentParser(
        description="Compute UMAP embeddings for hdf5 files. Files need to be stored in /tmp or $SLURM_TMPDIR."
    )
    arg_parser.add_argument(
        "chromsize",
        type=Path,
        help="A file with chrom sizes.",
    )
    arg_parser.add_argument(
        "output",
        type=Path,
        default=None,
        help="Directory to save embeddings in. Saves in home directory if not provided.",
    )
    # fmt: on
    return arg_parser.parse_args()


def main():
    """Run the main function."""
    cli = parse_arguments()

    if cli.output is not None:
        output_dir = cli.output
        try:
            output_dir.mkdir(exist_ok=True)
        except FileNotFoundError:
            output_dir = Path.home()
    else:
        output_dir = Path.home()

    chromsize_path = cli.chromsize
    hdf5_loader = Hdf5Loader(chrom_file=chromsize_path, normalization=True)

    # Find all hdf5 files
    hdf5_input_dir = Path(os.environ.get("SLURM_TMPDIR", "/tmp"))

    all_paths = list(hdf5_input_dir.rglob("*.hdf5"))
    if not all_paths:
        raise FileNotFoundError(f"No hdf5 files found in {hdf5_input_dir}.")
    print(f"Found {len(all_paths)} hdf5 files.")

    hdf5_paths_list_path = output_dir / f"{output_dir.name}_umap_files.list"
    with open(hdf5_paths_list_path, "w", encoding="utf8") as f:
        for path in all_paths:
            f.write(f"{path}\n")
    print(f"Saved hdf5 files used to: {hdf5_paths_list_path}")

    # Load relevant files
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Cannot read file directly with")
        hdf5_dict = hdf5_loader.load_hdf5s(
            data_file=hdf5_paths_list_path,
            verbose=True,
            strict=False,
        ).signals

    print(f"Loaded {len(hdf5_dict)}/{len(all_paths)} files.")
    file_names = list(hdf5_dict.keys())
    data = np.array(list(hdf5_dict.values()), dtype=np.float32)
    del hdf5_dict
    N_files = len(file_names)

    # PCA computation
    n_components = 3
    ipca = IncrementalPCA(n_components=n_components, batch_size=int(N_files / 5))
    X_ipca = ipca.fit_transform(data)

    # Save the PCA model and the transformed data
    fit_name = f"IPCA_fit_n{N_files}.skops"
    X_name = f"X_IPCA_n{N_files}.skops"
    dump_fit = {"file_names": file_names, "ipca_fit": ipca}
    dump_transformed_data = {"file_names": file_names, "X_ipca": X_ipca}
    skio.dump(dump_fit, output_dir / fit_name)
    skio.dump(dump_transformed_data, output_dir / X_name)

    # Save requirements saved file still usable in the future
    dists = metadata.distributions()
    req_file_name = "IPCA_saved_files_requirements.txt"
    with open(output_dir / req_file_name, "w", encoding="utf8") as f:
        for dist in dists:
            name = dist.metadata["Name"]
            version = dist.version
            f.write(f"{name}=={version}\n")

    print(f"Saved IPCA fit to: {output_dir / fit_name}")
    print(f"Saved transformed data to: {output_dir / X_name}")
    print(f"Saved requirements to: {output_dir / req_file_name}")


if __name__ == "__main__":
    main()