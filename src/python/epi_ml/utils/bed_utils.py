"""
This module provides a collection of utilities for manipulating and analyzing genomic data.
The utilities are useful for performing operations such as mapping between genomic ranges and bins,
and writing genomic data to bedgraph or .bed files.

The module provides functions to:
- Compute the size of a concatenated genome based on the resolution of each chromosome.
- Verify if a given resolution is coherent with the input size of the network.
- Convert values to a bedgraph format.
- Write given bed ranges to a .bed file.
- Compute the cumulative bin positions at the start of each chromosome.
- Convert multiple global genome bins to chromosome ranges.
- Convert multiple chromosome ranges to global genome bins.

Please note:
The function values_to_bedgraph() is not yet implemented and will raise a NotImplementedError when invoked.
"""
from __future__ import annotations

import itertools
from pathlib import Path
from typing import List, Tuple


def predict_concat_size(chroms, resolution):
    """Compute the size of a concatenated genome from the resolution of each chromosome."""
    concat_size = 0
    for _, size in chroms:
        size_of_mean = size // resolution
        if size_of_mean % resolution == 0:
            concat_size += size_of_mean
        else:
            concat_size += size_of_mean + 1

    return concat_size


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


def assert_correct_resolution(chroms, resolution, signal_length):
    """Raise AssertionError if the given resolution is not coherent with
    the input size of the network.
    """
    if predict_concat_size(chroms, resolution) != signal_length:
        raise AssertionError(
            f"Signal_length not coherent with given resolution of {resolution}."
        )


def values_to_bedgraph(values, chroms, resolution, bedgraph_path):
    """Write a bedgraph from a full genome values iterable (e.g. importance).
    The chromosome coordinates are zero-based, half-open (from 0 to N-1).
    """
    raise NotImplementedError
    # i = 0
    # with open(bedgraph_path, "w", encoding="utf-8") as my_bedgraph:
    #     for name, size in chroms:
    #         positions = itertools.chain(range(0, size, resolution), [size - 1])

    #         for pos1, pos2 in pairwise(positions):
    #             line = [name, pos1, pos2, values[i]]
    #             my_bedgraph.write("{}\t{}\t{}\t{}\n".format(*line))
    #             i += 1


def write_to_bed(
    bed_ranges: List[Tuple[str, int, int]], bed_path: str | Path, verbose: bool = False
) -> None:
    """Writes the given bed ranges to a .bed file.

    Args:
        bed_ranges (List[Tuple[str, int, int]]): List of tuples, each containing
            (chromosome name, start position, end position).
        bed_path (str): The path where the .bed file should be written.

    Note:
        The function doesn't return anything. It writes directly to a file.
    """
    with open(bed_path, "w", encoding="utf8") as file:
        for bed_range in bed_ranges:
            file.write(f"{bed_range[0]}\t{bed_range[1]}\t{bed_range[2]}\n")
    if verbose:
        print(f"Bed file written to {bed_path}")


def compute_cumulative_bins(chroms: List[Tuple[str, int]], resolution: int) -> List[int]:
    """Compute the cumulative bin positions at the start of each chromosome.

    Args:
        chroms (List[Tuple[str, int]]): List of tuples (ordered by bedsort chromosome order),
            where each tuple contains a chromosome name and its length in base pairs.
        resolution (int): The size of each bin.

    Returns:
        List[int]: List of cumulative bin positions.
    """
    if resolution % 10 != 0:
        raise ValueError("Resolution must be a multiple of 10.")

    cumulative_bins = [0]
    for _, chrom_size in chroms:
        bins_in_chrom = chrom_size // resolution + (chrom_size % resolution > 0)
        cumulative_bins.append(cumulative_bins[-1] + bins_in_chrom)

    return cumulative_bins


def bins_to_bed_ranges(
    bin_indexes: List[int], chroms: List[Tuple[str, int]], resolution: int
) -> List[Tuple[str, int, int]]:
    """Convert multiple global genome bins to chromosome ranges.

    Args:
        bin_indexes (List[int]): List of bin indexes in the genome.
        chroms (List[Tuple[str, int]]): List of tuples (ordered by chromosome order),
            where each tuple contains a chromosome name and its length in base pairs.
        resolution (int): The size of each bin.

    Returns:
        List[Tuple[str, int, int]]: List of tuples, each containing (chromosome name, start position, end position).

    Raises:
        IndexError: If any bin index is not in any chromosome,
        i.e., it's greater than the total number of bins in the genome.

    Note:
        The function assumes that chromosomes in `chroms` are ordered as they appear in the genome.
        The functions assumes that the binning was done per chromosome and then joined.
        The bin indexes are zero-based and span the entire genome considering the resolution.
        The returned ranges are half-open intervals [start, end).
    """
    bin_indexes = sorted(bin_indexes)
    bin_ranges = []

    cumulative_bins = compute_cumulative_bins(chroms, resolution)

    for bin_index in bin_indexes:
        # Find the chromosome that contains this bin
        for chrom_index, (chrom_start_bin, chrom_end_bin) in enumerate(
            zip(cumulative_bins[:-1], cumulative_bins[1:])
        ):
            if chrom_start_bin <= bin_index < chrom_end_bin:
                # The bin is in this chromosome
                bin_in_chrom = bin_index - chrom_start_bin
                start = bin_in_chrom * resolution
                end = min((bin_in_chrom + 1) * resolution, chroms[chrom_index][1])
                bin_ranges.append((chroms[chrom_index][0], start, end))
                break
        else:
            # The bin index is out of range
            raise IndexError("bin_index out of range")

    return bin_ranges


def bed_ranges_to_bins(
    ranges: List[Tuple[str, int, int]], chroms: List[Tuple[str, int]], resolution: int
) -> List[int]:
    """Convert multiple chromosome ranges to global genome bins.

    Args:
        ranges (List[Tuple[str, int, int]]): List of tuples, each containing (chromosome name, start position, end position).
        chroms (List[Tuple[str, int]]): List of tuples (ordered by chromosome order),
            where each tuple contains a chromosome name and its length in base pairs.
        resolution (int): The size of each bin.

    Returns:
        List[int]: List of bin indexes in the genome.

    Raises:
        IndexError: If any range is not in any chromosome.

    Note:
        The function assumes that chromosomes in `chroms` are ordered as they appear in the genome.
        The functions assumes that the binning was done per chromosome and then joined.
        The ranges are half-open intervals [start, end).
        The returned bin indexes are zero-based and span the entire genome considering the resolution.
    """
    cumulative_bins = compute_cumulative_bins(chroms, resolution)

    bin_indexes = []
    for chrom_range in ranges:
        chrom_name, start, end = chrom_range
        # Find the chromosome that contains this range
        for chrom_index, (chrom_start_bin, _) in enumerate(
            zip(cumulative_bins[:-1], cumulative_bins[1:])
        ):
            if chroms[chrom_index][0] == chrom_name:
                # The range is in this chromosome
                start_bin_in_chrom = start // resolution
                end_bin_in_chrom = (
                    end + resolution - 1
                ) // resolution  # This ensures we cover the entire range
                # Convert bins in chromosome to global bin indexes
                start_bin_global = chrom_start_bin + start_bin_in_chrom
                end_bin_global = chrom_start_bin + end_bin_in_chrom
                bin_indexes.extend(range(start_bin_global, end_bin_global))
                break
        else:
            # The chromosome name is not found
            raise IndexError("chromosome name not found")

    return bin_indexes