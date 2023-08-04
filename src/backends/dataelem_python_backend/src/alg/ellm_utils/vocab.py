# flake8: noqa
import collections
import io
import json
import os
import warnings

import numpy as np


class Vocab(object):
    def __init__(self,
                 counter=None,
                 max_size=None,
                 min_freq=1,
                 token_to_idx=None,
                 unk_token=None,
                 pad_token=None,
                 bos_token=None,
                 eos_token=None,
                 **kwargs):
        # Handle special tokens
        combs = (
            ('unk_token', unk_token),
            ('pad_token', pad_token),
            ('bos_token', bos_token),
            ('eos_token', eos_token),
        )
        for name, value in combs:
            kwargs[name] = value
        special_tokens = []
        special_iter = kwargs.keys()
        # sort alphabetically
        special_iter = sorted(special_iter)
        for special_token_name in special_iter:
            # Test if kwarg specifies a special token
            if not special_token_name.endswith('_token'):
                raise ValueError(
                    '{} is invalid. Only keyword arguments '
                    "that end in '_token' are supported "
                    'to declare special tokens.'.format(special_token_name))

            special_token = kwargs[special_token_name]
            if special_token is not None and special_token not in special_tokens:
                special_tokens.append(special_token)

        if counter is None:
            # use token_to_idx as dict to import pretrained vocabulary
            assert token_to_idx, 'token_to_idx should not be None when counter is None'
            for special_token in special_tokens:
                assert special_token in token_to_idx, '{} is not in token_to_idx'.format(
                    special_token)
            self._token_to_idx = token_to_idx
            self._idx_to_token = {
                idx: token
                for token, idx in token_to_idx.items()
            }
            if unk_token:
                unk_index = self._token_to_idx[unk_token]
                self._token_to_idx = collections.defaultdict(lambda: unk_index)
                self._token_to_idx.update(token_to_idx)
        else:
            self._idx_to_token = {
                idx: special_token
                for idx, special_token in enumerate(special_tokens)
            }
            self._token_to_idx = collections.defaultdict()
            self._token_to_idx.update(
                (token, idx) for idx, token in self._idx_to_token.items())
            self._index_counter_keys(counter, special_tokens, max_size,
                                     min_freq)
            if token_to_idx:
                self._sort_index_according_to_user_specification(token_to_idx)
            if unk_token:
                self._token_to_idx.default_factory = lambda: self._token_to_idx[
                    unk_token]

        # _expose_tokens_as_attributes
        self._identifiers_to_tokens = kwargs
        for identifier, token in kwargs.items():
            if identifier.startswith('_'):
                raise ValueError(
                    'It is not allowed to use identifiers starting with '
                    'underscore. In Python identifier names beginning with '
                    'underscore are internal.')
            if hasattr(self, identifier):
                raise ValueError(
                    'vocab.{} already exists. '
                    'Please choose a different identifier for token {}'.format(
                        identifier, token))
            setattr(self, identifier, token)

    def _index_counter_keys(self, counter, special_tokens, max_size, min_freq):
        # sort by frequency, then alphabetically
        token_freqs = sorted(counter.items(), key=lambda x: x[0])
        token_freqs.sort(key=lambda x: x[1], reverse=True)
        # frequencies of special tokens are not counted when building vocabulary
        # in frequency order
        special_tokens = set(special_tokens)
        max_size = None if max_size is None else max_size + len(special_tokens)
        for token, freq in token_freqs:
            if freq < min_freq or len(self._idx_to_token) == max_size:
                break
            if token not in special_tokens:
                self._idx_to_token[max(list(self._idx_to_token.keys()) + [-1])
                                   + 1] = token
                self._token_to_idx[token] = max(self._idx_to_token.keys())

    def _sort_index_according_to_user_specification(self, token_to_idx):
        # Sanity checks
        if not set(token_to_idx.keys()).issubset(self.token_to_idx.keys()):
            raise ValueError(
                'User-specified token_to_idx mapping can only contain '
                'tokens that will be part of the vocabulary.')
        if len(set(token_to_idx.values())) != len(token_to_idx):
            raise ValueError(
                'User-specified indices must not contain duplicates.')
        if min(token_to_idx.values()) < 0 or max(token_to_idx.values()) >= len(
                self.token_to_idx):
            raise ValueError(
                'User-specified indices must not be < 0 or >= the number of tokens '
                'that will be in the vocabulary. The current vocab contains {}'
                'tokens.'.format(len(self.token_to_idx)))

        # Update index ordering
        for token, new_idx in token_to_idx.items():
            old_idx = self.token_to_idx[token]
            ousted_token = self.idx_to_token[new_idx]

            self.token_to_idx[token] = new_idx
            self.token_to_idx[ousted_token] = old_idx
            self.idx_to_token[old_idx] = ousted_token
            self.idx_to_token[new_idx] = token

    def to_tokens(self, indices):
        to_reduce = False
        if not isinstance(indices, (list, tuple, np.ndarray)):
            indices = [indices]
            to_reduce = True
        if isinstance(indices, (list, tuple)):
            indices = np.asarray(indices)

        if isinstance(indices, (np.ndarray)) and len(indices.shape) > 1:
            raise ValueError(
                'Token indices is invalid. Expected 1D array, but received {}D array. '
                .format(len(indices.shape)))

        tokens = []
        for idx in indices:
            if not isinstance(idx, (int, np.integer)):
                warnings.warn(
                    "The type of `to_tokens()`'s input `indices` is not `int` which will be forcibly transfered to `int`. "
                )
                idx = int(idx)

            try:
                tokens.append(self._idx_to_token[idx])
            except KeyError:
                raise ValueError(
                    'Token index {} in the provided `indices` is invalid.'.
                    format(idx))

        return tokens[0] if to_reduce else tokens

    def to_indices(self, tokens):
        return self[tokens]

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self._token_to_idx[
                tokens] if tokens in self._token_to_idx else self._token_to_idx[
                    self.unk_token]
        else:
            return [
                self._token_to_idx[token] if token in self._token_to_idx else
                self._token_to_idx[self.unk_token] for token in tokens
            ]

    def __len__(self):
        return len(self._idx_to_token)

    def __contains__(self, token):
        return token in self._token_to_idx

    def __call__(self, tokens):
        """
        Maps the input tokens into indices. Its function is the same as the
        :meth:`to_indices` method.

        See detail at `to_indices`.
        """
        return self[tokens]

    @property
    def idx_to_token(self):
        # Returns index-token dict
        return self._idx_to_token

    @property
    def token_to_idx(self):
        # Return token-index dict
        return self._token_to_idx

    def to_json(self, path=None):
        vocab_dict = {}
        vocab_dict['idx_to_token'] = dict(self.idx_to_token)
        vocab_dict['token_to_idx'] = dict(self.token_to_idx)
        vocab_dict['unk_token'] = self.unk_token
        vocab_dict['identifiers_to_tokens'] = self._identifiers_to_tokens
        json_str = json.dumps(vocab_dict)
        if path:
            with io.open(path, 'w', encoding='utf-8') as f:
                f.write(json_str)
        return json_str

    @classmethod
    def from_json(cls, json_str):
        if os.path.isfile(json_str):
            with io.open(json_str, 'r', encoding='utf-8') as f:
                vocab_dict = json.load(f)
        else:
            vocab_dict = json.loads(json_str)
        token_to_idx = vocab_dict.get('token_to_idx')
        unk_token = vocab_dict.get('unk_token')
        identifiers_to_tokens = vocab_dict.get('identifiers_to_tokens', dict())
        if 'unk_token' in identifiers_to_tokens:
            del identifiers_to_tokens['unk_token']
        vocab = cls(counter=None,
                    token_to_idx=token_to_idx,
                    unk_token=unk_token,
                    **identifiers_to_tokens)
        return vocab

    @classmethod
    def from_dict(cls,
                  token_to_idx,
                  unk_token=None,
                  pad_token=None,
                  bos_token=None,
                  eos_token=None,
                  **kwargs):
        vocab = cls(
            counter=None,
            token_to_idx=token_to_idx,
            unk_token=unk_token,
            pad_token=pad_token,
            bos_token=bos_token,
            eos_token=eos_token,
            **kwargs,
        )
        return vocab

    @staticmethod
    def build_vocab(iterator,
                    max_size=None,
                    min_freq=1,
                    token_to_idx=None,
                    unk_token=None,
                    pad_token=None,
                    bos_token=None,
                    eos_token=None,
                    **kwargs):
        counter = collections.Counter()
        for tokens in iterator:
            counter.update(tokens)
        vocab = Vocab(
            counter,
            max_size=max_size,
            min_freq=min_freq,
            token_to_idx=token_to_idx,
            unk_token=unk_token,
            pad_token=pad_token,
            bos_token=bos_token,
            eos_token=eos_token,
            **kwargs,
        )
        return vocab

    @staticmethod
    def load_vocabulary(filepath,
                        unk_token=None,
                        pad_token=None,
                        bos_token=None,
                        eos_token=None,
                        **kwargs):
        token_to_idx = {}
        with io.open(filepath, 'r', encoding='utf-8') as f:
            for index, line in enumerate(f):
                token = line.rstrip('\n')
                token_to_idx[token] = int(index)
        vocab = Vocab.from_dict(token_to_idx,
                                unk_token=unk_token,
                                pad_token=pad_token,
                                bos_token=bos_token,
                                eos_token=eos_token,
                                **kwargs)
        return vocab

    def save_vocabulary(self, filepath):
        with open(filepath, 'w') as f:
            for idx in range(len(self._idx_to_token)):
                f.write(self._idx_to_token[idx] + '\n')

    def get_unk_token_id(self):
        return self._token_to_idx[
            self.unk_token] if self.unk_token is not None else self.unk_token

    def get_bos_token_id(self):
        return self._token_to_idx[
            self.bos_token] if self.bos_token is not None else self.bos_token

    def get_eos_token_id(self):
        return self._token_to_idx[
            self.eos_token] if self.eos_token is not None else self.eos_token

    def get_pad_token_id(self):
        return self._token_to_idx[
            self.pad_token] if self.pad_token is not None else self.pad_token
