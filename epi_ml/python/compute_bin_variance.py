import tensorflow as tf #import first because of library linking (cuda) reasons

import argparse
import os.path
import sys

import numpy as np

from argparseutils.directorytype import DirectoryType
from core import data
from core import analysis

def parse_arguments(args: list) -> argparse.Namespace:
    """argument parser for command line"""
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('hdf5', type=argparse.FileType('r'), help='A file with hdf5 filenames. Use absolute path!')
    arg_parser.add_argument('chromsize', type=argparse.FileType('r'), help='A file with chrom sizes.')
    arg_parser.add_argument('metadata', type=argparse.FileType('r'), help='A metadata JSON file.')
    arg_parser.add_argument('logdir', type=DirectoryType(), help='A directory for the logs.')
    return arg_parser.parse_args(args)

def compute_variance(hdf5s):
    """Return array of variance per signal bin from hdf5 signals dict."""
    return np.var(list(hdf5s.values()), axis=0)

def main(args):
    """main called from command line, edit to change behavior"""
    # parse params
    epiml_options = parse_arguments(args)

    # load external files
    my_datasource = data.EpiDataSource(
        epiml_options.hdf5,
        epiml_options.chromsize,
        epiml_options.metadata
        )

    # load useful info
    hdf5_resolution = my_datasource.hdf5_resolution()
    chroms = my_datasource.load_chrom_sizes()

    # md5:signal dict
    hdf5s = data.Hdf5Loader(my_datasource.chromsize_file,
                            my_datasource.hdf5_file,
                            normalization=True).hdf5s

    variance = compute_variance(hdf5s)

    bedgraph_path = os.path.join(epiml_options.logdir, "variance.bedgraph")
    analysis.values_to_bedgraph(variance, chroms, hdf5_resolution, bedgraph_path)

if __name__ == "__main__":
    main(sys.argv[1:])