import argparse
from pathlib import Path

from src.python.argparseutils.DefaultHelpParser import DefaultHelpParser as ArgumentParser
from src.python.argparseutils.directorychecker import DirectoryChecker
from src.python.core import metadata
from src.python.core.analysis import SHAP_Handler
from src.python.core.data import DataSetFactory, KnownData
from src.python.core.data_source import EpiDataSource
from src.python.core.model_pytorch import LightningDenseClassifier
from src.python.utils.time import time_now


def parse_arguments() -> argparse.Namespace:
    """argument parser for command line"""
    arg_parser = ArgumentParser()

    # fmt: off
    arg_parser.add_argument(
        "category", type=str, help="The metatada category to analyse.",
    )
    arg_parser.add_argument(
        "hdf5", type=Path, help="A file with hdf5 filenames. Use absolute path!",
    )
    arg_parser.add_argument(
        "chromsize", type=Path, help="A file with chrom sizes.",
    )
    arg_parser.add_argument(
        "metadata", type=Path, help="A metadata JSON file.",
    )
    arg_parser.add_argument(
        "logdir", type=DirectoryChecker(), help="Directory for the output logs.",
    )
    arg_parser.add_argument(
        "model", type=DirectoryChecker(), help="Directory where to load the classifier.",
    )
    # fmt: on
    return arg_parser.parse_args()


def benchmark(metadata: metadata.Metadata, datasource: EpiDataSource, model):
    """Benchmark how much time shap values computations take according to size of background dataset."""
    full_data = DataSetFactory.from_epidata(
        datasource=datasource,
        label_category="assay",
        metadata=metadata,
        min_class_size=1,
        test_ratio=0,
        validation_ratio=0,
        oversample=False,
    )

    for n in [250]:
        train_data = full_data.train.subsample(list(range(n)))
        shap_computer = SHAP_Handler(model=model, logdir=None)

        eval_size = 25
        evaluation_data = full_data.train.subsample(list(range(n, n + eval_size)))
        t_a = time_now()
        shap_computer.compute_NN(
            background_dset=train_data,
            evaluation_dset=evaluation_data,
            save=False,
        )
        print(f"Time taken with n={n}: {time_now() - t_a}")


def test_background_effect(my_metadata, my_datasource, my_model, logdir):
    # --- Prefilter metadata ---
    my_metadata.display_labels("assay")
    my_metadata.select_category_subsets("track_type", ["pval", "Unique_plusRaw"])

    assay_list = ["h3k9me3", "h3k36me3", "rna_seq"]
    my_metadata.select_category_subsets("assay", assay_list)

    md5_per_classes = my_metadata.md5_per_class("assay")
    background_1_md5s = md5_per_classes["h3k9me3"][0:10]
    background_2_md5s = md5_per_classes["rna_seq"][0:10]

    evaluation_md5s = (
        md5_per_classes["h3k9me3"][10:20]
        + md5_per_classes["rna_seq"][10:20]
        + md5_per_classes["h3k36me3"][0:10]
    )
    all_md5s = set(background_1_md5s + background_2_md5s + evaluation_md5s)

    for md5 in list(my_metadata.md5s):
        if md5 not in all_md5s:
            del my_metadata[md5]

    full_data = DataSetFactory.from_epidata(
        datasource=my_datasource,
        label_category="assay",
        metadata=my_metadata,
        min_class_size=1,
        test_ratio=0,
        validation_ratio=0,
        oversample=False,
    )

    background_1_idxs = [
        i
        for i, signal_id in enumerate(full_data.train.ids)
        if signal_id in set(background_1_md5s)
    ]
    background_2_idxs = [
        i
        for i, signal_id in enumerate(full_data.train.ids)
        if signal_id in set(background_2_md5s)
    ]
    evaluation_idxs = list(
        set(range(full_data.train.num_examples))
        - set(background_1_idxs + background_2_idxs)
    )

    assert isinstance(full_data.train, KnownData)
    background_1_data = full_data.train.subsample(background_1_idxs)
    background_2_data = full_data.train.subsample(background_2_idxs)

    evaluation_data = full_data.train.subsample(evaluation_idxs)

    for background_data in [background_1_data, background_2_data]:
        shap_computer = SHAP_Handler(model=my_model, logdir=logdir)
        shap_computer.compute_NN(
            background_dset=background_data,
            evaluation_dset=evaluation_data,
            save=True,
            name="background_effect_test",
        )


def main():
    cli = parse_arguments()

    category = cli.category

    my_datasource = EpiDataSource(cli.hdf5, cli.chromsize, cli.metadata)

    my_metadata = metadata.Metadata(my_datasource.metadata_file)

    logdir = cli.logdir
    model_dir = logdir
    if cli.model is not None:
        model_dir = cli.model
    my_model = LightningDenseClassifier.restore_model(model_dir)

    benchmark(my_metadata, my_datasource, my_model)


if __name__ == "__main__":
    main()
