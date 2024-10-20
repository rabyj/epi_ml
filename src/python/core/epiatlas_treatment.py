"""Functions to split epiatlas datasets properly, keeping track types together in the different sets."""
# TODO: Proper Data vs TestData typing
from __future__ import annotations

import collections
import copy
import itertools
import warnings
from typing import Dict, Generator, Iterable, List

import numpy as np
from sklearn.model_selection import StratifiedKFold

from src.python.core import data
from src.python.core.data_source import EpiDataSource
from src.python.core.hdf5_loader import Hdf5Loader
from src.python.core.metadata import Metadata

TRACKS_MAPPING = {
    "raw": ["pval", "fc"],
    "ctl_raw": [],
    "Unique_plusRaw": ["Unique_minusRaw"],
    "gembs_pos": ["gembs_neg"],
}

ACCEPTED_TRACKS = list(TRACKS_MAPPING.keys()) + list(
    itertools.chain.from_iterable(TRACKS_MAPPING.values())
)

LEADER_TRACKS = frozenset(["raw", "Unique_plusRaw", "gembs_pos"])


class EpiAtlasDataset:
    """Class that handles how epiatlas data signals are linked together.

    Parameters
    ----------
    datasource : EpiDataSource
        Where everything is read from.
    label_category : str
        The target category of labels to use.
    label_list : List[str], optional
        List of labels/classes to include from given category
    test_ratio : float, optional
        Ratio of data kept for test (not used for training or validation)
    min_class_size : int, optional
        Minimum number of samples per class.
    my_metadata : Metadata, optional
        Metadata to use, if the complete source should not be used. (e.g. more complex pre-filtering)
    """

    def __init__(
        self,
        datasource: EpiDataSource,
        label_category: str,
        label_list: List[str] | None = None,
        min_class_size: int = 10,
        test_ratio: float = 0,
        metadata: Metadata | None = None,
    ):
        self._datasource = datasource
        self._label_category = label_category
        self._label_list = label_list

        if metadata is not None:
            self._metadata = metadata
        else:
            self._metadata = Metadata(self.datasource.metadata_file)

        self._filter_metadata(min_class_size, verbose=True)

        self._raw_to_others = EpiAtlasDataset.epiatlas_prepare_split(self._metadata)

        # Load files
        self._raw_dset = self._create_raw_dataset(test_ratio, min_class_size)
        self._other_tracks = self._load_other_tracks()

        self._classes = self._raw_dset.classes

    @property
    def datasource(self) -> EpiDataSource:
        """Return given datasource."""
        return self._datasource

    @property
    def target_category(self) -> str:
        """Return given label category (e.g. assay)"""
        return self._label_category

    @property
    def label_list(self) -> List[str] | None:
        """Return given target labels inclusion list."""
        return self._label_list

    @property
    def classes(self) -> List[str]:
        """Return target classes"""
        return self._classes

    @property
    def raw_dataset(self) -> data.DataSet:
        """Return dataset of unmatched signals created during init."""
        return self._raw_dset

    @property
    def group_mapper(self) -> Dict[str, Dict[str, str]]:
        """Return md5sum track_type mapping dict.

        e.g. 1 entry { raw_md5sum : {"pval":md5sum, "fc":md5sum} }
        """
        return self._raw_to_others

    @property
    def metadata(self) -> Metadata:
        """Return a copy of current metadata held"""
        return copy.deepcopy(self._metadata)

    def _filter_metadata(self, min_class_size, verbose: bool) -> None:
        """Filter entry metadata for assay list and label_category."""
        if self.label_list is not None:
            self._metadata.select_category_subsets(self.target_category, self.label_list)
        self._metadata.remove_small_classes(min_class_size, self.target_category, verbose)

    def _create_raw_dataset(self, test_ratio: float, min_class_size: int) -> data.DataSet:
        """Create a dataset with raw+ctl_raw signals, all in the training set."""
        print("Creating epiatlas 'raw' signal training dataset")
        meta = self.metadata

        print("Theoretical maximum with complete dataset:")
        meta.display_labels(self.target_category)
        meta.display_labels("track_type")

        print("raw dataset: Selected signals in accordance with metadata:")
        meta.select_category_subsets("track_type", list(TRACKS_MAPPING.keys()))
        meta.display_labels("track_type")

        # important to not oversample now, because the train would bleed into valid during kfold.
        print("'Raw' dataset before oversampling and adding associated signals:")
        my_data = data.DataSetFactory.from_epidata(
            self.datasource,
            meta,
            self.target_category,
            min_class_size=min_class_size,
            validation_ratio=0,
            test_ratio=test_ratio,
            onehot=False,
            oversample=False,
        )
        meta.display_labels(self.target_category)

        return my_data

    @staticmethod
    def epiatlas_prepare_split(metadata: Metadata) -> Dict[str, Dict[str, str]]:
        """Return track_type mapping dict assuming the datasource is complete.

        Assumption/Condition: Only one file per track type, for a given uuid.

        e.g. { raw_md5sum : {"pval":md5sum, "fc":md5sum} }
        """
        # { uuid : {track_type1:md5sum, track_type2:md5sum, ...} }
        uuid_to_md5s = collections.defaultdict(dict)
        for dset in metadata.datasets:
            uuid = dset["uuid"]
            uuid_to_md5s[uuid].update({dset["track_type"]: dset["md5sum"]})

        raw_to_others = {}
        for tracks_to_md5 in uuid_to_md5s.values():

            for lead_track in LEADER_TRACKS:
                if lead_track in tracks_to_md5:

                    non_lead_tracks = set(TRACKS_MAPPING[lead_track]) & set(
                        tracks_to_md5.keys()
                    )
                    lead_md5 = tracks_to_md5[lead_track]

                    raw_to_others[lead_md5] = {
                        track: tracks_to_md5[track] for track in non_lead_tracks
                    }

        return raw_to_others

    def _load_other_tracks(self) -> Dict[str, np.ndarray]:
        """Return Hdf5Loader.signals for md5s of other (e.g. fc and pval) signals"""
        hdf5_loader = Hdf5Loader(self.datasource.chromsize_file, normalization=True)

        md5s = itertools.chain.from_iterable(
            [other_dict.values() for _, other_dict in self._raw_to_others.items()]
        )

        hdf5_loader.load_hdf5s(
            self.datasource.hdf5_file, md5s=md5s, verbose=False, strict=True
        )
        return hdf5_loader.signals

    def add_other_tracks(
        self, selected_positions: Iterable[int], dset: data.KnownData, resample: bool
    ) -> data.KnownData:
        """Return a modified dset object with added tracks (e.g. pval + fc) for selected signals.
        The matching tracks will be put right after the selected tracks.

        dset needs to be composed only of "leader" tracks, otherwise it will fail.
        """
        new_signals, new_str_labels, new_encoded_labels, new_md5s = [], [], [], []

        raw_dset = dset
        idxs = collections.Counter(i for i in np.arange(raw_dset.num_examples))
        if resample:
            resampled_X, resampled_y, idxs = data.EpiData.oversample_data(
                dset.signals, dset.encoded_labels
            )
            raw_dset = data.KnownData(
                ids=np.take(dset.ids, idxs),
                x=resampled_X,
                y=resampled_y,
                y_str=np.take(dset.original_labels, idxs),
                metadata=dset.metadata,
            )
            idxs = collections.Counter(i for i in idxs)

        for selected_index in selected_positions:
            og_dset_metadata = raw_dset.get_metadata(selected_index)
            chosen_md5 = raw_dset.get_id(selected_index)
            label = raw_dset.get_original_label(selected_index)
            encoded_label = raw_dset.get_encoded_label(selected_index)
            signal = raw_dset.get_signal(selected_index)

            track_type = og_dset_metadata["track_type"]

            if chosen_md5 != og_dset_metadata["md5sum"]:
                raise Exception("You dun fucked up")

            # oversampling specific to each "leader" signal
            for _ in range(idxs[selected_index]):

                if track_type in LEADER_TRACKS:

                    other_md5s = list(self._raw_to_others[chosen_md5].values())

                    other_signals = [self._other_tracks[md5] for md5 in other_md5s]

                    # order important, leader track first, order used for find_other_tracks.
                    new_md5s.extend([chosen_md5] + other_md5s)
                    new_signals.extend([signal] + other_signals)
                    new_str_labels.extend([label for _ in range(len(other_md5s) + 1)])
                    new_encoded_labels.extend(
                        [encoded_label for _ in range(len(other_md5s) + 1)]
                    )

                elif track_type == "ctl_raw":
                    new_md5s.append(chosen_md5)
                    new_signals.append(signal)
                    new_encoded_labels.append(encoded_label)
                    new_str_labels.append(label)
                else:
                    raise Exception("You dun fucked up")

        new_dset = data.KnownData(
            new_md5s, new_signals, new_encoded_labels, new_str_labels, dset.metadata
        )

        return new_dset

    def create_total_data(self, oversampling=False) -> data.KnownData:
        """Return a data set with all signals."""
        return self.add_other_tracks(
            range(self._raw_dset.train.num_examples),
            self._raw_dset.train,  # type: ignore
            resample=oversampling,
        )


class EpiAtlasFoldFactory:
    """Class that handles how epiatlas data is split into training and testing sets.

    Parameters
    ----------
    epiatlas_dataset : EpiAtlasDataset
        Source container for epiatlas data.
    n_fold : int, optional
        Number of folds for cross-validation.
    """

    def __init__(
        self,
        epiatlas_dataset: EpiAtlasDataset,
        n_fold: int = 10,
    ):
        self.k = n_fold
        if n_fold < 2:
            raise ValueError(
                f"Need at least two folds for cross-validation. Got {n_fold}."
            )

        self._epiatlas_dataset = epiatlas_dataset
        self.classes = self._epiatlas_dataset.classes

        self._add_other_tracks = epiatlas_dataset.add_other_tracks

    @classmethod
    def from_datasource(
        cls,
        datasource: EpiDataSource,
        label_category: str,
        label_list: List[str] | None = None,
        min_class_size: int = 10,
        test_ratio: float = 0,
        metadata: Metadata | None = None,
        n_fold: int = 10,
    ):
        epiatlas_dataset = EpiAtlasDataset(
            datasource,
            label_category,
            label_list,
            min_class_size,
            test_ratio,
            metadata,
        )
        return cls(epiatlas_dataset, n_fold)

    @property
    def n_fold(self) -> int:
        """Returns expected number of folds."""
        return self.k

    @property
    def epiatlas_dataset(self) -> EpiAtlasDataset:
        """Returns source EpiAtlasDataset."""
        return self._epiatlas_dataset

    def yield_split(self) -> Generator[data.DataSet, None, None]:
        """Yield train and valid tensor datasets for one split.

        Depends on given init parameters.
        """
        skf = StratifiedKFold(n_splits=self.k, shuffle=False)

        raw_dset = self.epiatlas_dataset.raw_dataset
        for train_idxs, valid_idxs in skf.split(
            np.zeros((raw_dset.train.num_examples, len(self.classes))),
            list(raw_dset.train.encoded_labels),
        ):
            new_datasets = data.DataSet.empty_collection()

            new_train = copy.deepcopy(raw_dset.train)
            new_train = self._add_other_tracks(train_idxs, new_train, resample=True)  # type: ignore

            new_valid = copy.deepcopy(raw_dset.train)
            new_valid = self._add_other_tracks(valid_idxs, new_valid, resample=False)  # type: ignore

            new_datasets.set_train(new_train)
            new_datasets.set_validation(new_valid)

            yield new_datasets

    def _correct_signal_group(self, md5s: List, verbose=True):
        """Return md5s corresponding to signal having the same EpiRR and assay (same group, like a pair or trio).
        as the first md5 (expected to be leader track), if they are contiguous with first signal.
        """
        info = [
            (
                self.epiatlas_dataset.metadata[md5]["EpiRR"],
                self.epiatlas_dataset.metadata[md5]["assay"],
            )
            for md5 in md5s
        ]
        if len(set(info)) != 1:
            if verbose:
                warnings.warn(
                    "Signals not from the same group in function _correct_signal_group. md5s: {md5s}. Returning subset."
                )

            if info[0] == info[1]:
                return md5s[0:2]

            if verbose:
                warnings.warn("No matching signals. Returning first md5.")
            return md5s[0]

        return md5s[:]

    def _find_other_tracks(
        self, selected_positions, dset: data.KnownData, resample: bool, md5_mapping: dict
    ) -> list[int]:
        """Return indexes that sample from complete data, i.e. all signals with their match next to them.
        Uses logic from create_total_data and add_other_tracks.

        md5_mapping : total data signal position dict of format {md5sum:i}
        """
        raw_dset = dset
        idxs = collections.Counter(i for i in np.arange(raw_dset.num_examples))
        index_mapping = {v: k for k, v in md5_mapping.items()}

        if resample:
            _, _, idxs = data.EpiData.oversample_data(
                np.zeros(shape=dset.signals.shape), dset.encoded_labels
            )
            repetitions = collections.Counter(i for i in idxs)
        else:
            repetitions = idxs

        new_selected_positions = []
        for selected_index in selected_positions:
            og_dset_metadata = raw_dset.get_metadata(selected_index)
            chosen_md5 = raw_dset.get_id(selected_index)
            track_type = og_dset_metadata["track_type"]

            if chosen_md5 != og_dset_metadata["md5sum"]:
                raise Exception("You dun fucked up")

            # oversampling specific to each "leader" signal
            rep = repetitions[selected_index]

            # number of matching signals (is it alone (ctl_raw), a pair, or a "fc,pval,raw" trio)
            other_nb = len(TRACKS_MAPPING[track_type])

            # add each group of indexes the required number of times (oversampling)
            if track_type in TRACKS_MAPPING:

                pos1 = md5_mapping[chosen_md5]
                all_match_indexes = list(range(pos1, pos1 + other_nb + 1))

                # check if all expected md5s have same epiRR and assay. correct if needed.
                if other_nb != 0:
                    md5s = [index_mapping[i] for i in all_match_indexes]
                    md5s = self._correct_signal_group(md5s)
                    if len(md5s) - 1 != other_nb:
                        other_nb = len(md5s) - 1
                        all_match_indexes = list(range(pos1, pos1 + other_nb + 1))

                new_selected_positions.extend(all_match_indexes * rep)

            else:
                raise Exception("You dun fucked up")

        return new_selected_positions

    # pylint: disable=unused-argument
    def split(
        self,
        total_data: data.KnownData,
        X=None,
        y=None,
        groups=None,
    ) -> Generator[tuple[List, List], None, None]:
        """Generate indices to split total data into training and validation set.

        Indexes match positions in output of create_total_data()
        X, y and groups :
            Always ignored, exist for compatibility.
        """
        md5_mapping = {md5: i for i, md5 in enumerate(total_data.ids)}

        raw_dset = self.epiatlas_dataset.raw_dataset
        skf = StratifiedKFold(n_splits=self.k, shuffle=False)
        for train_idxs, valid_idxs in skf.split(
            np.zeros((raw_dset.train.num_examples, len(self.classes))),
            list(raw_dset.train.encoded_labels),
        ):

            # The "complete" refers to the fact that the indexes are sampling over total data.
            complete_train_idxs = self._find_other_tracks(
                train_idxs, self._raw_dset.train, resample=True, md5_mapping=md5_mapping  # type: ignore
            )

            complete_valid_idxs = self._find_other_tracks(
                valid_idxs, self._raw_dset.train, resample=False, md5_mapping=md5_mapping  # type: ignore
            )

            yield complete_train_idxs, complete_valid_idxs

    def yield_subsample_validation(self, chosen_split: int, nb_split: int):
        """Will use StratifiedKFold to further subsample one of the validation
        splits normally generated into a training (no oversampling) and validation set.

        chosen_split (int): Split to subsample, as generated by yield_split.
        nb_split (int): How many splits to make into the chosen split.

        Created for SHAP values calculation sampling.
        """
        raw_dset = self.epiatlas_dataset.raw_dataset
        assert isinstance(raw_dset.train, data.KnownData)

        skf1 = StratifiedKFold(n_splits=self.k, shuffle=False)
        skf2 = StratifiedKFold(n_splits=nb_split, shuffle=False)
        idxs_gen = skf1.split(
            np.zeros((raw_dset.train.num_examples, len(self.classes))),
            list(raw_dset.train.encoded_labels),
        )

        try:
            _, valid_idxs = list(idxs_gen)[chosen_split]
        except IndexError as e:
            raise IndexError(
                f"Chosen split {chosen_split} out of range of initial {self.k} folds."
            ) from e

        chosen_total_dataset = raw_dset.train.subsample(list(valid_idxs))

        for train_subsplit_idxs, valid_subsplit_idxs in skf2.split(
            np.zeros((chosen_total_dataset.num_examples, len(self.classes))),
            list(chosen_total_dataset.encoded_labels),
        ):
            new_datasets = data.DataSet.empty_collection()

            new_train = self._add_other_tracks(train_subsplit_idxs, chosen_total_dataset, resample=False)  # type: ignore
            new_valid = self._add_other_tracks(valid_subsplit_idxs, chosen_total_dataset, resample=False)  # type: ignore

            new_datasets.set_train(new_train)
            new_datasets.set_validation(new_valid)

            yield new_datasets
