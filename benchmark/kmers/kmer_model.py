#!/usr/bin/env python3
# coding: utf-8

import contextlib

import numpy as np
import pandas as pd
from itertools import product

from typing import List, Optional, Literal

from sklearn.linear_model import Ridge


@contextlib.contextmanager
def numpy_temp_seed(seed):
    state = np.random.get_state()
    if seed is not None:
        np.random.seed(seed)
    try:
        yield
    finally:
        if seed is not None:
            np.random.set_state(state)


class KmerEncoder:
    def __init__(self, k: int):
        self.alphabet = ("A", "C", "G", "T")
        self.complement = {"A": "T", "C": "G",
                           "G": "C", "T": "A"}
        self.encode_dict = {k: i for i, k in enumerate(self.alphabet)}
        self.decode_dict = {v: k for k, v in self.encode_dict.items()}
        self.k = k

    def encode(self, kmer: str) -> int:
        code = 0
        for i in range(0, self.k, 1):
            itercode = self.encode_dict[kmer[i]]
            code = (code << 2) + itercode
        return code

    def decode(self, code: int) -> str:
        kmer = []
        code_residual = code
        for i in range(self.k - 1, -1, -1):
            ex = 4 ** i
            code_residual, itercode = code_residual % ex, code_residual // ex
            kmer.append(self.decode_dict[itercode])
        if code_residual != 0:
            raise ValueError(f"Wrong code for k={self.k}: {code} (residual: {code_residual})")
        return "".join(kmer)

    def revcomp(self, kmer: str) -> str:
        return "".join((self.complement[x] for x in reversed(kmer)))

    def get_all_kmers(self):
        kdict = list()
        for kmer in product(self.alphabet, repeat=self.k):
            kmer = "".join(kmer)
            kdict.append(kmer)
        return kdict

    def get_rc_kmers(self):
        kdict = dict()
        for kmer in product(self.alphabet, repeat=self.k):
            kmer = "".join(kmer)
            rc = self.revcomp(kmer)
            if rc in kdict:
                kdict[kmer] = False
            else:
                kdict[kmer] = True
        return kdict


class KmerLinearRegressor:
    """ A model utilizing k-mer frequencies against background """

    def __init__(
        self,
        complement: bool = True,
        kmer_length: int = 4,
        linreg_kws: Optional[dict] = None,
    ):
        """
        Initializes a `DifferentialKmerModel` with parameters.

        Parameters
        ----------
        complement : bool
            States whether the reverse-complement sequences should be scored identically.
            E.g., if 'complement' is True, then the sequences "GGAAAAA" and "TTTTTCC" will
            be treated as identical by the model.
            Default: True
        kmer_length: int
            Specifies k-mer length for the scoring algorithm.
            Default: 4
        linreg_kws: dict
            Additional arguments for Scikit-Learn Ridge regression
        """
        self.complement = complement

        self.k = kmer_length
        self.kmer_encoder = KmerEncoder(k=self.k)
        if linreg_kws is None:
            linreg_kws = dict()
        self.linear_regression = Ridge(**linreg_kws)
        if self.complement:
            self.kmer_complement = self.kmer_encoder.get_rc_kmers()
            self.kmers = sorted(self.kmer_complement.keys())
        else:
            self.kmers = sorted(self.kmer_encoder.get_all_kmers())

    @property
    def kmer_scores(self):
        return pd.Series(self.linear_regression.coef_, index=self.kmers)

    def fit(
        self,
        sequences: List[str],
        y: List[float],
        _return_kmer_df: bool = False,
    ) -> None | pd.DataFrame:
        sequences = pd.Series(sequences)

        kmer_df = self.count_kmers(sequences)
        if self.complement:
            kmer_df = self.invert_kmers(kmer_df)

        self.linear_regression.fit(kmer_df, y)

        if _return_kmer_df:
            return kmer_df
        else:
            return self

    def predict(self, sequences: List[str]):
        # if self.kmer_scores is None:
        #     raise ValueError(f"This '{self.__class__.__name__}' instance is not fitted yet."
        #                      "Call 'fit' with appropriate arguments before using this estimator.")
        exp_kmer_df = self.count_kmers(sequences)
        if self.complement:
            exp_kmer_df = self.invert_kmers(kmer_df)
        scores = self._predict_kmer_df(exp_kmer_df)
        return scores

    def _predict_kmer_df(self, exp_kmer_df: pd.DataFrame):
        scores = self.linear_regression.predict(exp_kmer_df)
        return scores

    def fit_predict(self, sequences: List[str], y: List[float]):
        exp_kmer_df = self.fit(sequences, y, _return_kmer_df=True)
        scores = self._predict_kmer_df(exp_kmer_df)
        return scores

    def invert_kmers(self, data, dtype="int64"):
        if isinstance(data, pd.DataFrame):
            new_data = pd.DataFrame(dtype=dtype)
        elif isinstance(data, pd.Series):
            new_data = pd.DataFrame(dtype=dtype)
        else:
            raise ValueError("'data' must be DataFrame or Series")
        for kmer in self.kmer_complement:
            if self.kmer_complement[kmer]:
                rc = self.kmer_encoder.revcomp(kmer)
                # palindromes are counted twice
                # because they occur on both strains
                new_data[kmer] = data[kmer] + data[rc]
                new_data[rc] = new_data[kmer]
            else:
                continue
        if isinstance(data, pd.DataFrame):
            new_data = new_data.reindex(columns=self.kmers, fill_value=0)
        elif isinstance(data, pd.Series):
            new_data = new_data.reindex(self.kmers, fill_value=0)
        return new_data

    def count_kmers(
        self,
        seqs: List[str]
    ):
        unique_seqs = pd.Series(seqs).drop_duplicates()
        letters = np.asarray(unique_seqs.apply(lambda x: [self.kmer_encoder.encode_dict[sym] for sym in x]).tolist())
        cp = letters.copy()
        for roll in range(1, self.k):
            letters = (letters << 2) + np.roll(cp, -roll, axis=1)

        if self.k == 1:
            letters = pd.DataFrame(letters, index=unique_seqs.to_numpy())
        else:
            letters = pd.DataFrame(letters[:, :-self.k + 1], index=unique_seqs.to_numpy())
        letters.rename_axis("seq", axis=0, inplace=True)
        letters.reset_index(inplace=True)
        codes = letters.melt(id_vars=["seq"], var_name="position", value_name="kmer")
        kmer_counts = pd.pivot_table(codes, values="position", index="seq", columns="kmer",
                                     aggfunc="count", fill_value=0)
        kmer_counts.rename(columns=self.kmer_encoder.decode, inplace=True, errors="raise")
        kmer_counts = kmer_counts.reindex(columns=self.kmers, fill_value=0)
        kmer_counts = kmer_counts.reindex(index=seqs)
        return kmer_counts

