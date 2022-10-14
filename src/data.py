import random
from typing import Callable, Collection, Dict, Iterator, List, Tuple, Type

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from math import ceil


class WordVectorModel:
    def train(self, data: pd.DataFrame, stimuli: Dict[int, List[str]]):
        pass

    def get_word_vectors(self, words: List[str]) -> List[torch.Tensor]:
        raise NotImplementedError()

    def dimensions(self) -> int:
        raise NotImplementedError()


class NoWordVectors(WordVectorModel):
    def get_word_vectors(self, words: List[str]) -> List[torch.Tensor]:
        return [torch.tensor([]) for word in words]

    def dimensions(self) -> int:
        return 0


class BertWordVectors(WordVectorModel):
    def __init__(self):
        import bert

        self._bert = bert

    def get_word_vectors(self, words: List[str]) -> List[torch.Tensor]:
        sentence = " ".join(words)  # TODO: Correct approach for Chinese tokenization?
        return [self._bert.get_word_vector(sentence, i) for i, word in enumerate(words)]

    def dimensions(self) -> int:
        return 768


class BertMeandiffWordVectors(WordVectorModel):
    def __init__(self, **kwargs):
        import meandiff

        self._meandiff = meandiff
        self._bert = BertWordVectors()
        self._encoder_kwargs = kwargs

    def train(self, data: pd.DataFrame, stimuli: Dict[int, List[str]]):
        self._encoder = self._meandiff.train_encoder(
            data, stimuli, **self._encoder_kwargs
        )

    def get_word_vectors(self, words: List[str]) -> List[torch.Tensor]:
        bert_vectors = self._bert.get_word_vectors(words)
        return [self._encoder.encode(vector).detach() for vector in bert_vectors]

    def dimensions(self) -> int:
        return self._encoder.encoding_size


class BertPCAWordVectors(WordVectorModel):
    def __init__(self, num_components: int = 20, **kwargs):
        from sklearn.decomposition import PCA

        self._bert = BertWordVectors()
        self._pca = PCA(num_components, **kwargs)

    def train(self, data: pd.DataFrame, stimuli: Dict[int, List[str]]):
        bert_vectors = []
        for sn in stimuli:
            bert_vectors.extend(self._bert.get_word_vectors(stimuli[sn]))
        X = torch.stack(bert_vectors)
        print("Training PCA...")
        self._pca.fit(X)

    def get_word_vectors(self, words: List[str]) -> List[torch.Tensor]:
        bert_vectors = torch.stack(self._bert.get_word_vectors(words))
        reduced_vectors = torch.from_numpy(self._pca.transform(bert_vectors)).float()
        return list(reduced_vectors)

    def dimensions(self) -> int:
        return self._pca.n_components


LinguisticFeature = Callable[[Tuple[str]], Tuple[torch.Tensor]]


class LingFeatWordVectors(WordVectorModel):
    def __init__(self, features: List[LinguisticFeature] = None):
        import ling, surprisal

        self._ling = ling
        self._surprisal = surprisal
        if features is not None:
            self.features = features
        else:
            self.features = [
                surprisal.surprisal,
                ling.pos,
                ling.dep,
                ling.depth,
                ling.character_frequency,
                ling.word_frequency,
            ]

        dummy_word = "中文"
        self._dimensions = len(self.get_word_vectors([dummy_word])[0])

    def get_word_vectors(self, words: List[str]) -> List[torch.Tensor]:
        words = tuple(words)
        values = [feature(words) for feature in self.features]
        return [torch.cat(v) for v in zip(*values)]

    def dimensions(self) -> int:
        return self._dimensions


def apply_standardization(x, m, sd):
    nonzero_sd = sd.clone()
    nonzero_sd[sd == 0] = 1
    res = (x.unsqueeze(1) - m.unsqueeze(0)) / nonzero_sd.unsqueeze(0)
    res = res.squeeze(1)
    return res


class EyetrackingDataPreprocessor:
    def __init__(
        self,
        features: Collection[str],
        *,
        word_vector_type: Type[WordVectorModel] = NoWordVectors,
        word_vector_kwargs: dict = None,
        preprocessing: Dict[str, Callable] = None,
        num_folds: float = 10,
    ):
        data = pd.read_csv("../data/saccades.csv", index_col=[0])
        data["wn"] = data["wn"] - 1  # Correct to 0-based indexing
        data["group"] = data["group"].map(lambda x: int(x + 0.5))

        stimuli_data = pd.read_csv("../data/all_stimuli.csv")
        self._stimuli = {}
        for sn in stimuli_data["sn"].unique():
            stimulus_data = stimuli_data[stimuli_data["sn"] == sn].sort_values("wn")
            assert stimulus_data["nw"].unique().item() == len(stimulus_data)
            self._stimuli[sn] = list(stimulus_data["item"])

        self._word_vector_type = word_vector_type
        self._word_vector_kwargs = word_vector_kwargs or {}

        self._features = features
        if preprocessing is None:
            preprocessing = {}

        self._data = pd.DataFrame()
        # Add sentence IDs, word numbers, and subject IDs
        self._data["sn"] = data["sn"]
        self._data["wn"] = data["wn"]
        self._data["subj"] = data["subj"]
        # Add labels
        self._data["group"] = data["group"]
        # Add eye-tracking features
        for feature in features:
            preprocess = preprocessing.get(feature, lambda x: x)
            self._data[feature] = data[feature].apply(preprocess)

        # Drop rows without stimulus data
        rows_before = len(self._data)
        self._data = self._data[self._data["sn"].isin(self._stimuli.keys())]
        rows_after = len(self._data)
        if rows_after != rows_before:
            print(
                f"Dropped {rows_before - rows_after} rows due to missing stimulus data"
            )
        print(f"DataFrame contains {rows_after} rows")

        # Distribute subjects across stratified folds
        self._num_folds = num_folds
        self._folds = [[] for _ in range(num_folds)]
        dyslexic_subjects = self._data[self._data["group"] == 1]["subj"].unique()
        control_subjects = self._data[self._data["group"] == 0]["subj"].unique()
        random.shuffle(dyslexic_subjects)
        random.shuffle(control_subjects)
        for i, subj in enumerate(dyslexic_subjects):
            self._folds[i % num_folds].append(subj)
        for i, subj in enumerate(control_subjects):
            self._folds[num_folds - 1 - i % num_folds].append(subj)
        for fold in self._folds:
            random.shuffle(fold)

    def _iter_trials(self, folds: Collection[int]) -> Iterator[pd.DataFrame]:
        # Iterate over all folds
        for fold in folds:
            # Iterate over all subjects in the fold
            for subj in self._folds[fold]:
                subj_data = self._data[self._data["subj"] == subj]
                # Iterate over all sentences this subject read
                for sn in subj_data["sn"].unique():
                    trial_data = subj_data[subj_data["sn"] == sn]
                    yield trial_data

    def train_word_vector_model(self, folds: Collection[int]) -> WordVectorModel:
        word_vector_model = self._word_vector_type(**self._word_vector_kwargs)
        word_vector_model.train(self._data, self._stimuli)
        return word_vector_model

    def iter_folds(
        self, folds: Collection[int], word_vector_model: WordVectorModel
    ) -> Iterator[Tuple[torch.Tensor, torch.Tensor, int]]:
        for trial_data in self._iter_trials(folds):
            sn = trial_data["sn"].unique().item()
            word_vectors = word_vector_model.get_word_vectors(self._stimuli[sn])
            feature_vectors = []
            for wn in range(len(word_vectors)):
                rows = trial_data[trial_data["wn"] == wn]
                if len(rows) == 0:
                    eyetracking_vector = torch.zeros((self.num_features,))
                else:
                    assert len(rows) == 1, f"More than 1 row for word {wn} (sn={sn})"
                    _, row = next(rows.iterrows())
                    eyetracking_vector = torch.tensor(
                        list(row[self._features]), dtype=torch.float
                    )
                feature_vectors.append(
                    torch.concat(
                        (
                            eyetracking_vector,
                            word_vectors[wn],
                        )
                    )
                )
            label = trial_data["group"].unique().item()
            subj = trial_data["subj"].unique().item()
            #  X = (time_steps, features)
            X = torch.stack(feature_vectors)
            y = torch.tensor(label, dtype=torch.float)
            yield X, y, subj

    @property
    def num_features(self) -> int:
        """Number of features per word (excluding word vector dimensions)."""
        return len(self._features)

    @property
    def max_sentence_length(self):
        data_copy = self._data.copy()
        max_s_len = max(data_copy.groupby(by=["subj", "sn"]).size())
        return max_s_len

    @property
    def max_number_of_sentences(self):
        data_copy = self._data.copy()
        max_s_count = data_copy.groupby(by="subj").sn.unique()
        return max([len(x) for x in max_s_count])


class PretrainingDataPreprocessor(EyetrackingDataPreprocessor):
    def __init__(
        self,
        features: Collection[str],
        *,
        word_vector_type: Type[WordVectorModel] = NoWordVectors,
        word_vector_kwargs: dict = None,
        preprocessing: Dict[str, Callable] = None,
    ):
        data = pd.read_csv("../data/bsc/bsc_saccades.csv", index_col=[0])
        data["wn"] = data["wn"] - 1  # Correct to 0-based indexing
        data = data.astype({"wn": int, "sn": int})
        # sents per subjects
        # print(data.groupby('subj')['sn'].nunique())
        stimuli_data = pd.read_csv("../data/bsc/bsc_stimuli_info.csv", index_col=None)
        stimuli_data.columns = stimuli_data.columns.str.lower()
        stimuli_data.rename(columns={"nw": "wn", "len": "WL", "word": "item"}, inplace=True)
        stimuli_data = stimuli_data.astype({"wn": int, "sn": int})
        self._stimuli = {}
        for sn in stimuli_data["sn"].unique():
            stimulus_data = stimuli_data[stimuli_data["sn"] == sn].sort_values("wn")
            self._stimuli[sn] = list(stimulus_data["item"])

        self._word_vector_type = word_vector_type
        self._word_vector_kwargs = word_vector_kwargs or {}

        self._features = features
        if preprocessing is None:
            preprocessing = {}

        self._data = pd.DataFrame()
        # Add sentence IDs, word numbers, and subject IDs
        self._data["sn"] = data["sn"]
        self._data["wn"] = data["wn"]
        self._data["subj"] = data["subj"]
        # Add labels
        # self._data["group"] = data["group"]
        # Add eye-tracking features
        for feature in features:
            preprocess = preprocessing.get(feature, lambda x: x)
            self._data[feature] = data[feature].apply(preprocess)

        # number of subjects = 60
        self._subjects = data['subj'].unique()

    @property
    def num_features(self) -> int:
        """Number of features per word (excluding word vector dimensions)."""
        return len(self._features)

    def train_word_vector_model(self, folds: Collection[int]) -> WordVectorModel:
        word_vector_model = self._word_vector_type(**self._word_vector_kwargs)
        word_vector_model.train(self._data, self._stimuli)
        return word_vector_model

    def _iter_trials(self) -> Iterator[pd.DataFrame]:
        for subj in self._subjects:
            subj_data = self._data[self._data["subj"] == subj]
            # Iterate over all sentences this subject reads
            for sn in subj_data["sn"].unique():
                trial_data = subj_data[subj_data["sn"] == sn]
                yield trial_data

    def iter_data(
            self, word_vector_model: WordVectorModel
    ) -> Iterator[Tuple[torch.Tensor, torch.Tensor, int]]:
        for trial_data in self._iter_trials():
            sn = trial_data["sn"].unique().item()
            # word_vectors = [torch.zeros((len(word_vector_model),)) for _ in self._stimuli[sn]]
            word_vectors = word_vector_model.get_word_vectors(self._stimuli[sn])
            # word_vectors = word_vector_model.get_word_vectors(self._stimuli[sn])
            feature_vectors = []
            for wn in range(len(word_vectors)):
                rows = trial_data[trial_data["wn"] == wn]
                if len(rows) == 0:
                    # pad not fixated words with 0
                    eyetracking_vector = torch.zeros((self.num_features,))
                else:
                    assert len(rows) == 1, f"More than 1 row for word {wn} (sn={sn})"
                    _, row = next(rows.iterrows())
                    eyetracking_vector = torch.tensor(
                        list(row[self._features]), dtype=torch.float
                    )
                feature_vectors.append(
                    torch.concat(
                        (
                            eyetracking_vector,
                            word_vectors[wn],
                        )
                    )
                )
            # dummy label, will be masked later
            label = 0
            subj = trial_data["subj"].unique().item()
            X = torch.stack(feature_vectors)
            y = torch.tensor(label, dtype=torch.float)
            yield X, y, subj

    @property
    def max_number_of_sentences(self):
        data_copy = self._data.copy()
        max_s_count = data_copy.groupby(by="subj").sn.unique()
        return max([len(x) for x in max_s_count])


class EyetrackingDataset(Dataset):
    def __init__(
        self,
        preprocessor: EyetrackingDataPreprocessor,
        word_vector_model: WordVectorModel,
        folds: Collection[int],
        batch_subjects: bool = False,
    ):
        self.sentences = list(preprocessor.iter_folds(folds, word_vector_model))
        self._subjects = list(np.unique([subj for _, _, subj in self.sentences]))
        self.num_features = preprocessor.num_features + word_vector_model.dimensions()
        self.batch_subjects = batch_subjects
        self.max_sentence_length = preprocessor.max_sentence_length
        self.max_number_of_sentences = preprocessor.max_number_of_sentences

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        if self.batch_subjects:
            subject = self._subjects[index]
            subject_sentences = [
                (X, y, subj) for X, y, subj in self.sentences if subj == subject
            ]
            X = torch.stack(
                [
                    F.pad(X, (0, 0, 0, self.max_sentence_length - X.size(0)))
                    for X, _, _ in subject_sentences
                ]
            )
            X = F.pad(X, (0, 0, 0, 0, 0, self.max_number_of_sentences - X.size(0)))
            y = torch.stack([y for _, y, _ in subject_sentences]).unique()
            return X, y, subject

        else:
            X, y, subj = self.sentences[index]
            X = F.pad(X, (0, 0, 0, self.max_sentence_length - X.size(0)))
            return X, y, subj

    def __len__(self) -> int:
        if self.batch_subjects:
            return len(self._subjects)
        else:
            return len(self.sentences)

    def standardize(self, mean: torch.Tensor, sd: torch.Tensor):
        self.sentences = [
            (apply_standardization(X, mean, sd), y, subj)
            for X, y, subj in self.sentences
        ]


class PretrainingDataset(EyetrackingDataset):
    def __init__(
        self,
        preprocessor: PretrainingDataPreprocessor,
        word_vector_model: WordVectorModel,
    ):
        self.sentences = list(preprocessor.iter_data(word_vector_model))
        self.num_et_features = preprocessor.num_features
        self.num_features = preprocessor.num_features + word_vector_model.dimensions()
        self.masked_sentences = self.mask_sentences()
        self._subjects = list(np.unique([subj for _, _, subj in self.sentences]))
        self.max_sentence_length = preprocessor.max_sentence_length
        self.max_number_of_sentences = preprocessor.max_number_of_sentences

    def mask_sentences(self):
        masked_sentences = []
        for sentence in self.sentences:
            X, y, subj = sentence
            t_mask = random.sample(range(0, X.shape[0]), 2)
            for t in t_mask:
                X_masked = X.clone()
                y_masked = X_masked[t, 0:self.num_et_features].clone()
                X_masked[t, 0:self.num_et_features] = 0
                masked_sentences.append((X_masked, y_masked, subj))
        return masked_sentences

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        X, y, subj = self.masked_sentences[index]
        X = F.pad(X, (0, 0, 0, self.max_sentence_length - X.size(0)))
        return X, y, subj

    def __len__(self) -> int:
        return len(self.sentences)
