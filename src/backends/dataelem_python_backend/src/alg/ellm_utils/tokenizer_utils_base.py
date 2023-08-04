# flake8: noqa
import copy
import io
import json
import os
import warnings
from collections import OrderedDict, UserDict
from dataclasses import dataclass, field
from enum import Enum
from typing import (Any, Dict, List, NamedTuple, Optional, Sequence, Tuple,
                    Union)

import numpy as np

from .env import MODEL_HOME
from .log import logger


@dataclass(frozen=True, eq=True)
class AddedToken:
    """
    AddedToken represents a token to be added to a Tokenizer An AddedToken can have special options defining the
    way it should behave.
    """

    content: str = field(default_factory=str)
    single_word: bool = False
    lstrip: bool = False
    rstrip: bool = False
    normalized: bool = True

    def __getstate__(self):
        return self.__dict__

    def __str__(self):
        return self.content


@dataclass
class FastEncoding:
    """This is dummy class reserved for fast tokenizer"""

    pass


class ExplicitEnum(Enum):
    """
    Enum with more explicit error message for missing values.
    """
    @classmethod
    def _missing_(cls, value):
        raise ValueError(
            f'{value} is not a valid {cls.__name__}, please select one of {list(cls._value2member_map_.keys())}'
        )


class PaddingStrategy(ExplicitEnum):
    """
    Possible values for the `padding` argument in [`PretrainedTokenizerBase.__call__`]. Useful for tab-completion in an
    IDE.
    """

    LONGEST = 'longest'
    MAX_LENGTH = 'max_length'
    DO_NOT_PAD = 'do_not_pad'


class TensorType(ExplicitEnum):
    """
    Possible values for the `return_tensors` argument in [`PretrainedTokenizerBase.__call__`]. Useful for
    tab-completion in an IDE.
    """

    PADDLE = 'pd'
    NUMPY = 'np'


VERY_LARGE_INTEGER = int(
    1e30
)  # This is used to set the max input length for a model with infinite size input
LARGE_INTEGER = int(
    1e20
)  # This is used when we need something big but slightly smaller than VERY_LARGE_INTEGER

# Define type aliases and NamedTuples
TextInput = str
PreTokenizedInput = List[str]
EncodedInput = List[int]
TextInputPair = Tuple[str, str]
PreTokenizedInputPair = Tuple[List[str], List[str]]
EncodedInputPair = Tuple[List[int], List[int]]

SPECIAL_TOKENS_MAP_FILE = 'special_tokens_map.json'
ADDED_TOKENS_FILE = 'added_tokens.json'
TOKENIZER_CONFIG_FILE = 'tokenizer_config.json'


def _is_numpy(x):
    return isinstance(x, np.ndarray)


class TruncationStrategy(ExplicitEnum):
    """
    Possible values for the `truncation` argument in [`PretrainedTokenizerBase.__call__`]. Useful for tab-completion in
    an IDE.
    """

    ONLY_FIRST = 'only_first'
    ONLY_SECOND = 'only_second'
    LONGEST_FIRST = 'longest_first'
    DO_NOT_TRUNCATE = 'do_not_truncate'


class CharSpan(NamedTuple):
    """
    Character span in the original string.

    Args:
        start (`int`): Index of the first character in the original string.
        end (`int`): Index of the character following the last character in the original string.
    """

    start: int
    end: int


class TokenSpan(NamedTuple):
    """
    Token span in an encoded string (list of tokens).

    Args:
        start (`int`): Index of the first token in the span.
        end (`int`): Index of the token following the last token in the span.
    """

    start: int
    end: int


class BatchEncoding(UserDict):
    def __init__(
        self,
        data: Optional[Dict[str, Any]] = None,
        encoding: Optional[Union[FastEncoding, Sequence[FastEncoding]]] = None,
        tensor_type: Union[None, str] = None,
        prepend_batch_axis: bool = False,
        n_sequences: Optional[int] = None,
    ):
        super().__init__(data)

        if isinstance(encoding, FastEncoding):
            encoding = [encoding]

        self._encodings = encoding

        if n_sequences is None and encoding is not None and len(encoding):
            n_sequences = encoding[0].n_sequences

        self._n_sequences = n_sequences

        self.convert_to_tensors(tensor_type=tensor_type,
                                prepend_batch_axis=prepend_batch_axis)

    @property
    def n_sequences(self) -> Optional[int]:
        """
        `Optional[int]`: The number of sequences used to generate each sample from the batch encoded in this
        [`BatchEncoding`]. Currently can be one of `None` (unknown), `1` (a single sentence) or `2` (a pair of
        sentences)
        """
        return self._n_sequences

    @property
    def is_fast(self) -> bool:
        """
        `bool`: Indicate whether this [`BatchEncoding`] was generated from the result of a [`PretrainedFastTokenizer`]
        or not.
        """
        return self._encodings is not None

    def __getitem__(self, item: Union[int, str]) -> Union[Any, FastEncoding]:
        """
        If the key is a string, returns the value of the dict associated to `key` ('input_ids', 'attention_mask',
        etc.).

        If the key is an integer, get the `Encoding` for batch item with index `key`.
        """
        if isinstance(item, str):
            return self.data[item]
        elif self._encodings is not None:
            return self._encodings[item]
        else:
            raise KeyError(
                'Indexing with integers is not available when using tokenizer.__call__()'
                ' with return_dict=True. Please set return_dict to False to use integer indexing.'
            )

    def __getattr__(self, item: str):
        try:
            return self.data[item]
        except KeyError:
            raise AttributeError

    def __getstate__(self):
        return {'data': self.data, 'encodings': self._encodings}

    def __setstate__(self, state):
        if 'data' in state:
            self.data = state['data']

        if 'encodings' in state:
            self._encodings = state['encodings']

    def keys(self):
        return self.data.keys()

    def values(self):
        return self.data.values()

    def items(self):
        return self.data.items()

    # After this point:
    # Extended properties and methods only available for fast tokenizers
    # not yet supported

    @property
    def encodings(self) -> Optional[List[FastEncoding]]:
        """
        `Optional[List[FastEncoding]]`: The list all encodings from the tokenization process. Returns `None` if
        the input was tokenized through Python (i.e., not a fast) tokenizer.
        """
        return self._encodings

    def tokens(self, batch_index: int = 0) -> List[str]:
        """
        Return the list of tokens (sub-parts of the input strings after word/subword splitting and before conversion to
        integer indices) at a given batch index (only works for the output of a fast tokenizer).

        Args:
            batch_index (`int`, *optional*, defaults to 0): The index to access in the batch.

        Returns:
            `List[str]`: The list of tokens at that index.
        """
        if not self._encodings:
            raise ValueError(
                'tokens() is not available when using Python-based tokenizers')
        return self._encodings[batch_index].tokens

    def sequence_ids(self, batch_index: int = 0) -> List[Optional[int]]:
        if not self._encodings:
            raise ValueError(
                'sequence_ids() is not available when using Python-based tokenizers'
            )
        return self._encodings[batch_index].sequence_ids

    def words(self, batch_index: int = 0) -> List[Optional[int]]:
        if not self._encodings:
            raise ValueError(
                'words() is not available when using Python-based tokenizers')
        warnings.warn(
            '`BatchEncoding.words()` property is deprecated and should be replaced with the identical, '
            'but more self-explanatory `BatchEncoding.word_ids()` property.',
            FutureWarning,
        )
        return self.word_ids(batch_index)

    def word_ids(self, batch_index: int = 0) -> List[Optional[int]]:
        if not self._encodings:
            raise ValueError(
                'word_ids() is not available when using Python-based tokenizers'
            )
        return self._encodings[batch_index].word_ids

    def token_to_sequence(self,
                          batch_or_token_index: int,
                          token_index: Optional[int] = None) -> int:
        if not self._encodings:
            raise ValueError(
                'token_to_sequence() is not available when using Python based tokenizers'
            )
        if token_index is not None:
            batch_index = batch_or_token_index
        else:
            batch_index = 0
            token_index = batch_or_token_index
        if batch_index < 0:
            batch_index = self._batch_size + batch_index
        if token_index < 0:
            token_index = self._seq_len + token_index
        return self._encodings[batch_index].token_to_sequence(token_index)

    def token_to_word(self,
                      batch_or_token_index: int,
                      token_index: Optional[int] = None) -> int:
        if not self._encodings:
            raise ValueError(
                'token_to_word() is not available when using Python based tokenizers'
            )
        if token_index is not None:
            batch_index = batch_or_token_index
        else:
            batch_index = 0
            token_index = batch_or_token_index
        if batch_index < 0:
            batch_index = self._batch_size + batch_index
        if token_index < 0:
            token_index = self._seq_len + token_index
        return self._encodings[batch_index].token_to_word(token_index)

    def word_to_tokens(self,
                       batch_or_word_index: int,
                       word_index: Optional[int] = None,
                       sequence_index: int = 0) -> Optional[TokenSpan]:
        if not self._encodings:
            raise ValueError(
                'word_to_tokens() is not available when using Python based tokenizers'
            )
        if word_index is not None:
            batch_index = batch_or_word_index
        else:
            batch_index = 0
            word_index = batch_or_word_index
        if batch_index < 0:
            batch_index = self._batch_size + batch_index
        if word_index < 0:
            word_index = self._seq_len + word_index
        span = self._encodings[batch_index].word_to_tokens(
            word_index, sequence_index)
        return TokenSpan(*span) if span is not None else None

    def token_to_chars(self,
                       batch_or_token_index: int,
                       token_index: Optional[int] = None) -> CharSpan:
        if not self._encodings:
            raise ValueError(
                'token_to_chars() is not available when using Python based tokenizers'
            )
        if token_index is not None:
            batch_index = batch_or_token_index
        else:
            batch_index = 0
            token_index = batch_or_token_index
        return CharSpan(
            *(self._encodings[batch_index].token_to_chars(token_index)))

    def char_to_token(self,
                      batch_or_char_index: int,
                      char_index: Optional[int] = None,
                      sequence_index: int = 0) -> int:
        if not self._encodings:
            raise ValueError(
                'char_to_token() is not available when using Python based tokenizers'
            )
        if char_index is not None:
            batch_index = batch_or_char_index
        else:
            batch_index = 0
            char_index = batch_or_char_index
        return self._encodings[batch_index].char_to_token(
            char_index, sequence_index)

    def word_to_chars(self,
                      batch_or_word_index: int,
                      word_index: Optional[int] = None,
                      sequence_index: int = 0) -> CharSpan:
        if not self._encodings:
            raise ValueError(
                'word_to_chars() is not available when using Python based tokenizers'
            )
        if word_index is not None:
            batch_index = batch_or_word_index
        else:
            batch_index = 0
            word_index = batch_or_word_index
        return CharSpan(*(self._encodings[batch_index].word_to_chars(
            word_index, sequence_index)))

    def char_to_word(self,
                     batch_or_char_index: int,
                     char_index: Optional[int] = None,
                     sequence_index: int = 0) -> int:
        if not self._encodings:
            raise ValueError(
                'char_to_word() is not available when using Python based tokenizers'
            )
        if char_index is not None:
            batch_index = batch_or_char_index
        else:
            batch_index = 0
            char_index = batch_or_char_index
        return self._encodings[batch_index].char_to_word(
            char_index, sequence_index)

    def convert_to_tensors(self,
                           tensor_type: Optional[Union[str,
                                                       TensorType]] = None,
                           prepend_batch_axis: bool = False):
        if tensor_type is None:
            return self

        if not isinstance(tensor_type, TensorType):
            tensor_type = TensorType(tensor_type)
        as_tensor = np.asarray
        is_tensor = _is_numpy

        for key, value in self.items():
            try:
                if prepend_batch_axis:
                    value = [value]

                if not is_tensor(value):
                    tensor = as_tensor(value)

                    self[key] = tensor
            except:  # noqa E722
                if key == 'overflowing_tokens':
                    raise ValueError(
                        'Unable to create tensor returning overflowing tokens of different lengths. '
                        'Please see if a fast version of this tokenizer is available to have this feature available.'
                    )
                raise ValueError(
                    'Unable to create tensor, you should probably activate truncation and/or padding '
                    "with 'padding=True' 'truncation=True' to have batched tensors with the same length."
                )

        return self


class SpecialTokensMixin:
    SPECIAL_TOKENS_ATTRIBUTES = [
        'bos_token',
        'eos_token',
        'unk_token',
        'sep_token',
        'pad_token',
        'cls_token',
        'mask_token',
        'additional_special_tokens',
    ]

    def __init__(self, verbose=True, **kwargs):
        self._bos_token = getattr(self, '_bos_token', None)
        self._eos_token = getattr(self, '_eos_token', None)
        self._unk_token = getattr(self, '_unk_token', None)
        self._sep_token = getattr(self, '_sep_token', None)
        self._pad_token = getattr(self, '_pad_token', None)
        self._cls_token = getattr(self, '_cls_token', None)
        self._mask_token = getattr(self, '_mask_token', None)
        self._pad_token_type_id = getattr(self, '_pad_token_type_id', 0)
        self._additional_special_tokens = getattr(
            self, '_additional_special_tokens', [])
        self.verbose = verbose
        for key, value in kwargs.items():
            if value is None:
                continue
            if key in self.SPECIAL_TOKENS_ATTRIBUTES:
                if key == 'additional_special_tokens':
                    assert isinstance(
                        value,
                        (list, tuple)), f'Value {value} is not a list or tuple'
                    assert all(
                        isinstance(t, (str, AddedToken)) for t in value
                    ), 'One of the tokens is not a string or an AddedToken'
                    setattr(self, key, value)
                elif isinstance(value, (str, AddedToken)):
                    setattr(self, key, value)
                else:
                    raise TypeError(
                        f'special token {key} has to be either str or AddedToken but got: {type(value)}'
                    )

    def sanitize_special_tokens(self) -> int:
        return self.add_tokens(self.all_special_tokens_extended,
                               special_tokens=True)

    def add_special_tokens(
            self, special_tokens_dict: Dict[str, Union[str,
                                                       AddedToken]]) -> int:
        if not special_tokens_dict:
            return 0

        added_tokens = 0
        for key, value in special_tokens_dict.items():
            assert key in self.SPECIAL_TOKENS_ATTRIBUTES, f'Key {key} is not a special token'

            if self.verbose:
                logger.info(
                    f'Assigning {value} to the {key} key of the tokenizer')
            setattr(self, key, value)

            if key == 'additional_special_tokens':
                assert isinstance(value, (list, tuple)) and all(
                    isinstance(t, (str, AddedToken)) for t in value
                ), f'Tokens {value} for key {key} should all be str or AddedToken instances'
                added_tokens += self.add_tokens(value, special_tokens=True)
            else:
                assert isinstance(
                    value, (str, AddedToken)
                ), f'Token {value} for key {key} should be a str or an AddedToken instance'
                added_tokens += self.add_tokens([value], special_tokens=True)

        return added_tokens

    def add_tokens(self,
                   new_tokens: Union[str, AddedToken, List[Union[str,
                                                                 AddedToken]]],
                   special_tokens: bool = False) -> int:
        if not new_tokens:
            return 0

        if not isinstance(new_tokens, (list, tuple)):
            new_tokens = [new_tokens]

        return self._add_tokens(new_tokens, special_tokens=special_tokens)

    def _add_tokens(self,
                    new_tokens: Union[List[str], List[AddedToken]],
                    special_tokens: bool = False) -> int:
        raise NotImplementedError

    @property
    def bos_token(self) -> str:
        """
        `str`: Beginning of sentence token. Log an error if used while not having been set.
        """
        if self._bos_token is None and self.verbose:
            logger.error('Using bos_token, but it is not set yet.')
            return None
        return str(self._bos_token)

    @property
    def eos_token(self) -> str:
        """
        `str`: End of sentence token. Log an error if used while not having been set.
        """
        if self._eos_token is None and self.verbose:
            logger.error('Using eos_token, but it is not set yet.')
            return None
        return str(self._eos_token)

    @property
    def unk_token(self) -> str:
        """
        `str`: Unknown token. Log an error if used while not having been set.
        """
        if self._unk_token is None and self.verbose:
            logger.error('Using unk_token, but it is not set yet.')
            return None
        return str(self._unk_token)

    @property
    def sep_token(self) -> str:
        """
        `str`: Separation token, to separate context and query in an input sequence. Log an error if used while not
        having been set.
        """
        if self._sep_token is None and self.verbose:
            logger.error('Using sep_token, but it is not set yet.')
            return None
        return str(self._sep_token)

    @property
    def pad_token(self) -> str:
        """
        `str`: Padding token. Log an error if used while not having been set.
        """
        if self._pad_token is None and self.verbose:
            logger.error('Using pad_token, but it is not set yet.')
            return None
        return str(self._pad_token)

    @property
    def cls_token(self) -> str:
        """
        `str`: Classification token, to extract a summary of an input sequence leveraging self-attention along the full
        depth of the model. Log an error if used while not having been set.
        """
        if self._cls_token is None and self.verbose:
            logger.error('Using cls_token, but it is not set yet.')
            return None
        return str(self._cls_token)

    @property
    def mask_token(self) -> str:
        """
        `str`: Mask token, to use when training a model with masked-language modeling. Log an error if used while not
        having been set.
        """
        if self._mask_token is None and self.verbose:
            logger.error('Using mask_token, but it is not set yet.')
            return None
        return str(self._mask_token)

    @property
    def additional_special_tokens(self) -> List[str]:
        """
        `List[str]`: All the additional special tokens you may want to use. Log an error if used while not having been
        set.
        """
        if self._additional_special_tokens is None and self.verbose:
            logger.error(
                'Using additional_special_tokens, but it is not set yet.')
            return None
        return [str(tok) for tok in self._additional_special_tokens]

    @bos_token.setter
    def bos_token(self, value):
        self._bos_token = value

    @eos_token.setter
    def eos_token(self, value):
        self._eos_token = value

    @unk_token.setter
    def unk_token(self, value):
        self._unk_token = value

    @sep_token.setter
    def sep_token(self, value):
        self._sep_token = value

    @pad_token.setter
    def pad_token(self, value):
        self._pad_token = value

    @cls_token.setter
    def cls_token(self, value):
        self._cls_token = value

    @mask_token.setter
    def mask_token(self, value):
        self._mask_token = value

    @additional_special_tokens.setter
    def additional_special_tokens(self, value):
        self._additional_special_tokens = value

    @property
    def bos_token_id(self) -> Optional[int]:
        """
        `Optional[int]`: Id of the beginning of sentence token in the vocabulary. Returns `None` if the token has not
        been set.
        """
        if self._bos_token is None:
            return None
        return self.convert_tokens_to_ids(self.bos_token)

    @property
    def eos_token_id(self) -> Optional[int]:
        """
        `Optional[int]`: Id of the end of sentence token in the vocabulary. Returns `None` if the token has not been
        set.
        """
        if self._eos_token is None:
            return None
        return self.convert_tokens_to_ids(self.eos_token)

    @property
    def unk_token_id(self) -> Optional[int]:
        """
        `Optional[int]`: Id of the unknown token in the vocabulary. Returns `None` if the token has not been set.
        """
        if self._unk_token is None:
            return None
        return self.convert_tokens_to_ids(self.unk_token)

    @property
    def sep_token_id(self) -> Optional[int]:
        """
        `Optional[int]`: Id of the separation token in the vocabulary, to separate context and query in an input
        sequence. Returns `None` if the token has not been set.
        """
        if self._sep_token is None:
            return None
        return self.convert_tokens_to_ids(self.sep_token)

    @property
    def pad_token_id(self) -> Optional[int]:
        """
        `Optional[int]`: Id of the padding token in the vocabulary. Returns `None` if the token has not been set.
        """
        if self._pad_token is None:
            return None
        return self.convert_tokens_to_ids(self.pad_token)

    @property
    def pad_token_type_id(self) -> int:
        """
        `int`: Id of the padding token type in the vocabulary.
        """
        return self._pad_token_type_id

    @property
    def cls_token_id(self) -> Optional[int]:
        """
        `Optional[int]`: Id of the classification token in the vocabulary, to extract a summary of an input sequence
        leveraging self-attention along the full depth of the model.

        Returns `None` if the token has not been set.
        """
        if self._cls_token is None:
            return None
        return self.convert_tokens_to_ids(self.cls_token)

    @property
    def mask_token_id(self) -> Optional[int]:
        """
        `Optional[int]`: Id of the mask token in the vocabulary, used when training a model with masked-language
        modeling. Returns `None` if the token has not been set.
        """
        if self._mask_token is None:
            return None
        return self.convert_tokens_to_ids(self.mask_token)

    @property
    def additional_special_tokens_ids(self) -> List[int]:
        """
        `List[int]`: Ids of all the additional special tokens in the vocabulary. Log an error if used while not having
        been set.
        """
        return self.convert_tokens_to_ids(self.additional_special_tokens)

    @bos_token_id.setter
    def bos_token_id(self, value):
        self._bos_token = self.convert_ids_to_tokens(
            value) if value is not None else None

    @eos_token_id.setter
    def eos_token_id(self, value):
        self._eos_token = self.convert_ids_to_tokens(
            value) if value is not None else None

    @unk_token_id.setter
    def unk_token_id(self, value):
        self._unk_token = self.convert_ids_to_tokens(
            value) if value is not None else None

    @sep_token_id.setter
    def sep_token_id(self, value):
        self._sep_token = self.convert_ids_to_tokens(
            value) if value is not None else None

    @pad_token_id.setter
    def pad_token_id(self, value):
        self._pad_token = self.convert_ids_to_tokens(
            value) if value is not None else None

    @cls_token_id.setter
    def cls_token_id(self, value):
        self._cls_token = self.convert_ids_to_tokens(
            value) if value is not None else None

    @mask_token_id.setter
    def mask_token_id(self, value):
        self._mask_token = self.convert_ids_to_tokens(
            value) if value is not None else None

    @additional_special_tokens_ids.setter
    def additional_special_tokens_ids(self, values):
        self._additional_special_tokens = [
            self.convert_ids_to_tokens(value) for value in values
        ]

    @property
    def special_tokens_map(self) -> Dict[str, Union[str, List[str]]]:
        """
        `Dict[str, Union[str, List[str]]]`: A dictionary mapping special token class attributes (`cls_token`,
        `unk_token`, etc.) to their values (`'<unk>'`, `'<cls>'`, etc.).

        Convert potential tokens of `AddedToken` type to string.
        """
        set_attr = {}
        for attr in self.SPECIAL_TOKENS_ATTRIBUTES:
            attr_value = getattr(self, '_' + attr)
            if attr_value:
                set_attr[attr] = (type(attr_value)(
                    str(attr_value_sub)
                    for attr_value_sub in attr_value) if isinstance(
                        attr_value, (list, tuple)) else str(attr_value))
        return set_attr

    @property
    def special_tokens_map_extended(
        self
    ) -> Dict[str, Union[str, AddedToken, List[Union[str, AddedToken]]]]:
        set_attr = {}
        for attr in self.SPECIAL_TOKENS_ATTRIBUTES:
            attr_value = getattr(self, '_' + attr)
            if attr_value:
                set_attr[attr] = attr_value
        return set_attr

    @property
    def all_special_tokens(self) -> List[str]:
        all_toks = [str(s) for s in self.all_special_tokens_extended]
        return all_toks

    @property
    def all_special_tokens_extended(self) -> List[Union[str, AddedToken]]:
        all_toks = []
        set_attr = self.special_tokens_map_extended
        for attr_value in set_attr.values():
            all_toks = all_toks + (list(attr_value) if isinstance(
                attr_value, (list, tuple)) else [attr_value])
        all_toks = list(OrderedDict.fromkeys(all_toks))
        return all_toks

    @property
    def all_special_ids(self) -> List[int]:
        all_toks = self.all_special_tokens
        all_ids = self.convert_tokens_to_ids(all_toks)
        return all_ids


class PretrainedTokenizerBase(SpecialTokensMixin):
    resource_files_names: Dict[str, str] = {}
    pretrained_resource_files_map: Dict[str, Dict[str, str]] = {}
    pretrained_init_configuration: Dict[str, Dict[str, Any]] = {}
    max_model_input_sizes: Dict[str, Optional[int]] = {}
    _auto_class: Optional[str] = None
    tokenizer_config_file = 'tokenizer_config.json'

    model_input_names: List[str] = ['input_ids', 'token_type_ids']
    padding_side: str = 'right'
    truncation_side: str = 'right'
    slow_tokenizer_class = None

    def __init__(self, **kwargs):
        self.init_inputs = ()

        self.init_kwargs = getattr(self, 'init_kwargs',
                                   None) or copy.deepcopy(kwargs)
        self.name_or_path = kwargs.pop('name_or_path', '')
        self._processor_class = kwargs.pop('processor_class', None)

        model_max_length = kwargs.pop('model_max_length',
                                      kwargs.pop('max_len', None))
        self.model_max_length = model_max_length if model_max_length is not None else VERY_LARGE_INTEGER
        self.padding_side = kwargs.pop('padding_side', self.padding_side)
        if self.padding_side not in ['right', 'left']:
            raise ValueError(
                f"Padding side should be selected between 'right' and 'left', current value: {self.padding_side}"
            )

        self.truncation_side = kwargs.pop('truncation_side',
                                          self.truncation_side)
        if self.truncation_side not in ['right', 'left']:
            raise ValueError(
                f"Padding side should be selected between 'right' and 'left', current value: {self.truncation_side}"
            )

        self.model_input_names = kwargs.pop('model_input_names',
                                            self.model_input_names)

        self.deprecation_warnings = (
            {}
        )  # Use to store when we have already noticed a deprecation warning (avoid overlogging).

        super().__init__(**kwargs)

    @property
    def max_len_single_sentence(self) -> int:
        """
        `int`: The maximum length of a sentence that can be fed to the model.
        """
        return self.model_max_length - self.num_special_tokens_to_add(
            pair=False)

    @property
    def max_len_sentences_pair(self) -> int:
        """
        `int`: The maximum combined length of a pair of sentences that can be fed to the model.
        """
        return self.model_max_length - self.num_special_tokens_to_add(
            pair=True)

    @max_len_single_sentence.setter
    def max_len_single_sentence(self, value) -> int:
        # For backward compatibility, allow to try to setup 'max_len_single_sentence'.
        if value == self.model_max_length - self.num_special_tokens_to_add(
                pair=False) and self.verbose:
            if not self.deprecation_warnings.get('max_len_single_sentence',
                                                 False):
                warnings.warn(
                    "Setting 'max_len_single_sentence' is now deprecated. "
                    'This value is automatically set up.')
            self.deprecation_warnings['max_len_single_sentence'] = True
        else:
            raise ValueError(
                "Setting 'max_len_single_sentence' is now deprecated. "
                'This value is automatically set up.')

    @max_len_sentences_pair.setter
    def max_len_sentences_pair(self, value) -> int:
        # For backward compatibility, allow to try to setup 'max_len_sentences_pair'.
        if value == self.model_max_length - self.num_special_tokens_to_add(
                pair=True) and self.verbose:
            if not self.deprecation_warnings.get('max_len_sentences_pair',
                                                 False):
                warnings.warn(
                    "Setting 'max_len_sentences_pair' is now deprecated. "
                    'This value is automatically set up.')
            self.deprecation_warnings['max_len_sentences_pair'] = True
        else:
            raise ValueError(
                "Setting 'max_len_sentences_pair' is now deprecated. "
                'This value is automatically set up.')

    def _set_processor_class(self, processor_class: str):
        """Sets processor class as an attribute."""
        self._processor_class = processor_class

    def __repr__(self) -> str:
        return (
            f"{'PretrainedTokenizer'}(name_or_path='{self.name_or_path}', "
            f'vocab_size={self.vocab_size}, model_max_len={self.model_max_length}, '
            f"padding_side='{self.padding_side}', truncation_side='{self.truncation_side}', special_tokens={self.special_tokens_map_extended})"
        )

    def get_vocab(self) -> Dict[str, int]:
        raise NotImplementedError()

    @classmethod
    def from_pretrained(cls,
                        pretrained_model_name_or_path,
                        *args,
                        from_hf_hub=False,
                        **kwargs):
        pretrained_model_name_or_path = str(pretrained_model_name_or_path)
        vocab_files = {}
        init_configuration = {}

        additional_files_names = {
            'added_tokens_file': ADDED_TOKENS_FILE,
            'special_tokens_map_file': SPECIAL_TOKENS_MAP_FILE,
            'tokenizer_config_file': TOKENIZER_CONFIG_FILE,
        }

        vocab_files_target = {
            **cls.resource_files_names,
            **additional_files_names
        }
        vocab_files_target['tokenizer_config_file'] = cls.tokenizer_config_file
        for file_id, file_name in vocab_files_target.items():
            full_file_name = os.path.join(pretrained_model_name_or_path,
                                          file_name)
            if os.path.isfile(full_file_name):
                vocab_files[file_id] = full_file_name

        default_root = os.path.join(MODEL_HOME, pretrained_model_name_or_path)
        resolved_vocab_files = {}
        for file_id, file_path in vocab_files.items():
            if file_path is None or os.path.isfile(file_path):
                resolved_vocab_files[file_id] = file_path
                continue

            path = os.path.join(default_root, file_path.split('/')[-1])
            resolved_vocab_files[file_id] = path

        has_tokenizer_file = resolved_vocab_files.get('tokenizer_file',
                                                      None) is not None
        tokenizer_config_file = resolved_vocab_files.pop(
            'tokenizer_config_file', None)
        if tokenizer_config_file is not None:
            with io.open(tokenizer_config_file, encoding='utf-8') as f:
                init_kwargs = json.load(f)
        else:
            init_kwargs = init_configuration

        init_args = init_kwargs.pop('init_args', ())
        init_kwargs.pop('init_class', None)

        init_args = init_args if not args else args
        init_kwargs.update(kwargs)

        def convert_added_tokens(obj):
            if isinstance(
                    obj, dict
            ) and '__type' in obj and obj['__type'] == 'AddedToken':
                obj.pop('__type')
                return AddedToken(**obj)
            elif isinstance(obj, (list, tuple)):
                return list(convert_added_tokens(o) for o in obj)
            elif isinstance(obj, dict):
                return {k: convert_added_tokens(v) for k, v in obj.items()}
            return obj

        init_kwargs = convert_added_tokens(init_kwargs)
        if pretrained_model_name_or_path in cls.max_model_input_sizes:
            model_max_length = cls.max_model_input_sizes[
                pretrained_model_name_or_path]
            if model_max_length is not None and isinstance(
                    model_max_length, (int, float)):
                init_kwargs['model_max_length'] = min(
                    init_kwargs.get('model_max_length', int(1e30)),
                    model_max_length)

        added_tokens_file = resolved_vocab_files.pop('added_tokens_file', None)
        for args_name, file_path in resolved_vocab_files.items():
            if args_name not in init_kwargs:
                init_kwargs[args_name] = file_path
            elif not os.path.isfile(init_kwargs[args_name]
                                    or '') and os.path.isfile(file_path):
                init_kwargs[args_name] = file_path
        tokenizer = cls(*init_args, **init_kwargs)
        special_tokens_map_file = resolved_vocab_files.pop(
            'special_tokens_map_file', None)
        if special_tokens_map_file is not None:
            with open(special_tokens_map_file,
                      encoding='utf-8') as special_tokens_map_handle:
                special_tokens_map = json.load(special_tokens_map_handle)
            for key, value in special_tokens_map.items():
                if key in kwargs and kwargs[key]:
                    # This value has already been redefined by the kwargs
                    # We keep this new value and ignore the one stored in the special_tokens_map_file

                    continue

                if isinstance(value, dict):
                    value = AddedToken(**value)
                elif isinstance(value, list):
                    value = [
                        AddedToken(
                            **token) if isinstance(token, dict) else token
                        for token in value
                    ]
                setattr(tokenizer, key, value)

        special_tokens = tokenizer.all_special_tokens
        if added_tokens_file is not None:
            with open(added_tokens_file,
                      encoding='utf-8') as added_tokens_handle:
                added_tok_encoder = json.load(added_tokens_handle)

            added_tok_encoder_sorted = list(
                sorted(added_tok_encoder.items(), key=lambda x: x[1]))
            for token, index in added_tok_encoder_sorted:
                if has_tokenizer_file and index != len(
                        tokenizer) and tokenizer.convert_tokens_to_ids(
                            token) != index:
                    # index is the current length of the tokenizer (not in vocabulary)
                    raise ValueError(
                        f'Wrong index found for {token}: should be {tokenizer.convert_tokens_to_ids(token)} but found '
                        f'{index}.')
                elif not has_tokenizer_file and index != len(tokenizer):
                    # Tokenizer slow: added token cannot already be in the vocabulary so its index needs to be the
                    # current length of the tokenizer.
                    raise ValueError(
                        f"Non-consecutive added token '{token}' found. "
                        f'Should have index {len(tokenizer)} but has index {index} in saved vocabulary.'
                    )

                tokenizer.add_tokens(
                    token, special_tokens=bool(token in special_tokens))
        added_tokens = tokenizer.sanitize_special_tokens()
        return tokenizer

    def _get_padding_truncation_strategies(self,
                                           padding=False,
                                           truncation=False,
                                           max_length=None,
                                           pad_to_multiple_of=None,
                                           verbose=True,
                                           **kwargs):
        old_truncation_strategy = kwargs.pop('truncation_strategy',
                                             'do_not_truncate')
        old_pad_to_max_length = kwargs.pop('pad_to_max_seq_len', False)

        # Backward compatibility for previous behavior, maybe we should deprecate it:
        # If you only set max_length, it activates truncation for max_length
        if max_length is not None and padding is False and truncation is False:
            if verbose:
                if not self.deprecation_warnings.get(
                        'Truncation-not-explicitly-activated', False):
                    warnings.warn(
                        'Truncation was not explicitly activated but `max_length` is provided a specific value, '
                        'please use `truncation=True` to explicitly truncate examples to max length. '
                        "Defaulting to 'longest_first' truncation strategy. "
                        'If you encode pairs of sequences (GLUE-style) with the tokenizer you can select this strategy '
                        'more precisely by providing a specific strategy to `truncation`.'
                    )
                self.deprecation_warnings[
                    'Truncation-not-explicitly-activated'] = True
            truncation = 'longest_first'

        # Get padding strategy
        if padding is False and old_pad_to_max_length:
            if verbose:
                warnings.warn(
                    'The `pad_to_max_length` argument is deprecated and will be removed in a future version, '
                    "use `padding=True` or `padding='longest'` to pad to the longest sequence in the batch, or "
                    "use `padding='max_length'` to pad to a max length. In this case, you can give a specific "
                    'length with `max_length` (e.g. `max_length=45`) or leave max_length to None to pad to the '
                    'maximal input size of the model (e.g. 512 for Bert).',
                    FutureWarning,
                )
            if max_length is None:
                padding_strategy = PaddingStrategy.LONGEST
            else:
                padding_strategy = PaddingStrategy.MAX_LENGTH
        elif padding is not False:
            if padding is True:
                if verbose:
                    if max_length is not None and (truncation is False
                                                   or truncation
                                                   == 'do_not_truncate'):
                        warnings.warn(
                            '`max_length` is ignored when `padding`=`True` and there is no truncation strategy. '
                            "To pad to max length, use `padding='max_length'`."
                        )
                    if old_pad_to_max_length is not False:
                        warnings.warn(
                            'Though `pad_to_max_length` = `True`, it is ignored because `padding`=`True`.'
                        )
                # Default to pad to the longest sequence in the batch
                padding_strategy = PaddingStrategy.LONGEST
            elif not isinstance(padding, PaddingStrategy):
                padding_strategy = PaddingStrategy(padding)
            elif isinstance(padding, PaddingStrategy):
                padding_strategy = padding
        else:
            padding_strategy = PaddingStrategy.DO_NOT_PAD

        # Get truncation strategy
        if truncation is False and old_truncation_strategy != 'do_not_truncate':
            if verbose:
                warnings.warn(
                    'The `truncation_strategy` argument is deprecated and will be removed in a future version, '
                    'use `truncation=True` to truncate examples to a max length. You can give a specific '
                    'length with `max_length` (e.g. `max_length=45`) or leave max_length to None to truncate to the '
                    'maximal input size of the model (e.g. 512 for Bert). '
                    ' If you have pairs of inputs, you can give a specific truncation strategy selected among '
                    "`truncation='only_first'` (will only truncate the first sentence in the pairs) "
                    "`truncation='only_second'` (will only truncate the second sentence in the pairs) "
                    "or `truncation='longest_first'` (will iteratively remove tokens from the longest sentence in the pairs).",
                    FutureWarning,
                )
            truncation_strategy = TruncationStrategy(old_truncation_strategy)
        elif truncation is not False:
            if truncation is True:
                truncation_strategy = (
                    TruncationStrategy.LONGEST_FIRST
                )  # Default to truncate the longest sequences in pairs of inputs
            elif not isinstance(truncation, TruncationStrategy):
                truncation_strategy = TruncationStrategy(truncation)
            elif isinstance(truncation, TruncationStrategy):
                truncation_strategy = truncation
        else:
            truncation_strategy = TruncationStrategy.DO_NOT_TRUNCATE

        # Set max length if needed
        if max_length is None:
            if padding_strategy == PaddingStrategy.MAX_LENGTH:
                if self.model_max_length > LARGE_INTEGER:
                    if verbose:
                        if not self.deprecation_warnings.get(
                                'Asking-to-pad-to-max_length', False):
                            warnings.warn(
                                'Asking to pad to max_length but no maximum length is provided and the model has no predefined maximum length. '
                                'Default to no padding.')
                        self.deprecation_warnings[
                            'Asking-to-pad-to-max_length'] = True
                    padding_strategy = PaddingStrategy.DO_NOT_PAD
                else:
                    max_length = self.model_max_length

            if truncation_strategy != TruncationStrategy.DO_NOT_TRUNCATE:
                if self.model_max_length > LARGE_INTEGER:
                    if verbose:
                        if not self.deprecation_warnings.get(
                                'Asking-to-truncate-to-max_length', False):
                            warnings.warn(
                                'Asking to truncate to max_length but no maximum length is provided and the model has no predefined maximum length. '
                                'Default to no truncation.')
                        self.deprecation_warnings[
                            'Asking-to-truncate-to-max_length'] = True
                    truncation_strategy = TruncationStrategy.DO_NOT_TRUNCATE
                else:
                    max_length = self.model_max_length

        # Test if we have a padding token
        if padding_strategy != PaddingStrategy.DO_NOT_PAD and (
                not self.pad_token or self.pad_token_id < 0):
            raise ValueError(
                'Asking to pad but the tokenizer does not have a padding token. '
                'Please select a token to use as `pad_token` `(tokenizer.pad_token = tokenizer.eos_token e.g.)` '
                "or add a new pad token via `tokenizer.add_special_tokens({'pad_token': '[PAD]'})`."
            )

        # Check that we will truncate to a multiple of pad_to_multiple_of if both are provided
        if (truncation_strategy != TruncationStrategy.DO_NOT_TRUNCATE
                and padding_strategy != PaddingStrategy.DO_NOT_PAD
                and pad_to_multiple_of is not None and max_length is not None
                and (max_length % pad_to_multiple_of != 0)):
            raise ValueError(
                f'Truncation and padding are both activated but '
                f'truncation length ({max_length}) is not a multiple of pad_to_multiple_of ({pad_to_multiple_of}).'
            )

        return padding_strategy, truncation_strategy, max_length, kwargs

    def __call__(self,
                 text: Union[str, List[str], List[List[str]]],
                 text_pair: Optional[Union[str, List[str],
                                           List[List[str]]]] = None,
                 max_length: Optional[int] = None,
                 stride: int = 0,
                 is_split_into_words: Union[bool, str] = False,
                 padding: Union[bool, str, PaddingStrategy] = False,
                 truncation: Union[bool, str, TruncationStrategy] = False,
                 return_position_ids: bool = False,
                 return_token_type_ids: Optional[bool] = None,
                 return_attention_mask: Optional[bool] = None,
                 return_length: bool = False,
                 return_overflowing_tokens: bool = False,
                 return_special_tokens_mask: bool = False,
                 return_dict: bool = True,
                 return_offsets_mapping: bool = False,
                 add_special_tokens: bool = True,
                 pad_to_multiple_of: Optional[int] = None,
                 return_tensors: Optional[Union[str, TensorType]] = None,
                 verbose: bool = True,
                 **kwargs):
        # Input type checking for clearer error
        def _is_valid_text_input(t):
            if isinstance(t, str):
                # Strings are fine
                return True
            elif isinstance(t, (list, tuple)):
                # List are fine as long as they are...
                if len(t) == 0:
                    # ... empty
                    return True
                elif isinstance(t[0], str):
                    # ... list of strings
                    return True
                elif isinstance(t[0], (list, tuple)):
                    # ... list with an empty list or with a list of strings
                    return len(t[0]) == 0 or isinstance(t[0][0], str)
                else:
                    return False
            else:
                return False

        if not _is_valid_text_input(text):
            raise ValueError(
                'text input must of type `str` (single example), `List[str]` (batch or single pretokenized example) '
                'or `List[List[str]]` (batch of pretokenized examples).')

        if text_pair is not None and not _is_valid_text_input(text_pair):
            raise ValueError(
                'text input must of type `str` (single example), `List[str]` (batch or single pretokenized example) '
                'or `List[List[str]]` (batch of pretokenized examples).')

        # check `split_into_words` value
        if isinstance(is_split_into_words,
                      str) and is_split_into_words != 'token':
            raise ValueError(
                "the value of `is_split_into_words` should be one of: {True, False, 'token'} but receive: <%s>",
                is_split_into_words,
            )

        if is_split_into_words:
            is_batched = isinstance(text,
                                    (list, tuple)) and text and isinstance(
                                        text[0], (list, tuple))
        else:
            is_batched = isinstance(text, (list, tuple))

        if is_batched:
            if isinstance(text_pair, str):
                raise TypeError(
                    'when tokenizing batches of text, `text_pair` must be a list or tuple with the same length as `text`.'
                )
            if text_pair is not None and len(text) != len(text_pair):
                raise ValueError(
                    f'batch length of `text`: {len(text)} does not match batch length of `text_pair`: {len(text_pair)}.'
                )
            batch_text_or_text_pairs = list(zip(
                text, text_pair)) if text_pair is not None else text
            return self.batch_encode(
                batch_text_or_text_pairs=batch_text_or_text_pairs,
                max_length=max_length,
                stride=stride,
                is_split_into_words=is_split_into_words,
                padding=padding,
                truncation=truncation,
                return_position_ids=return_position_ids,
                return_token_type_ids=return_token_type_ids,
                return_attention_mask=return_attention_mask,
                return_length=return_length,
                return_overflowing_tokens=return_overflowing_tokens,
                return_special_tokens_mask=return_special_tokens_mask,
                return_dict=return_dict,
                return_offsets_mapping=return_offsets_mapping,
                add_special_tokens=add_special_tokens,
                pad_to_multiple_of=pad_to_multiple_of,
                return_tensors=return_tensors,
                verbose=verbose,
                **kwargs,
            )
        else:
            return self.encode(
                text=text,
                text_pair=text_pair,
                max_length=max_length,
                stride=stride,
                is_split_into_words=is_split_into_words,
                padding=padding,
                truncation=truncation,
                return_position_ids=return_position_ids,
                return_token_type_ids=return_token_type_ids,
                return_attention_mask=return_attention_mask,
                return_length=return_length,
                return_overflowing_tokens=return_overflowing_tokens,
                return_special_tokens_mask=return_special_tokens_mask,
                return_offsets_mapping=return_offsets_mapping,
                add_special_tokens=add_special_tokens,
                pad_to_multiple_of=pad_to_multiple_of,
                return_tensors=return_tensors,
                verbose=verbose,
                **kwargs,
            )

    def encode(self,
               text,
               text_pair=None,
               add_special_tokens=True,
               padding: Union[bool, str, PaddingStrategy] = False,
               truncation: Union[bool, str, TruncationStrategy] = False,
               max_length: Optional[int] = None,
               stride: int = 0,
               is_split_into_words: bool = False,
               pad_to_multiple_of: Optional[int] = None,
               return_tensors: Optional[Union[str, TensorType]] = None,
               return_token_type_ids: Optional[bool] = None,
               return_attention_mask: Optional[bool] = None,
               return_overflowing_tokens: bool = False,
               return_special_tokens_mask: bool = False,
               return_offsets_mapping: bool = False,
               return_length: bool = False,
               verbose: bool = True,
               return_position_ids=False,
               **kwargs) -> BatchEncoding:
        """
        Tokenize and prepare for the model a sequence or a pair of sequences.

        Args:
            text (`str`, `List[str]` or `List[int]`):
                The first sequence to be encoded. This can be a string, a list of strings (tokenized string using the
                `tokenize` method) or a list of integers (tokenized string ids using the `convert_tokens_to_ids`
                method).
            text_pair (`str`, `List[str]` or `List[int]`, *optional*):
                Optional second sequence to be encoded. This can be a string, a list of strings (tokenized string using
                the `tokenize` method) or a list of integers (tokenized string ids using the `convert_tokens_to_ids`
                method).
        """
        # Backward compatibility for 'max_seq_len'
        old_max_seq_len = kwargs.get('max_seq_len', None)
        if max_length is None and old_max_seq_len:
            if verbose:
                warnings.warn(
                    'The `max_seq_len` argument is deprecated and will be removed in a future version, '
                    'please use `max_length` instead.',
                    FutureWarning,
                )
            max_length = old_max_seq_len
        # Backward compatibility for 'truncation_strategy', 'pad_to_max_length'
        padding_strategy, truncation_strategy, max_length, kwargs = self._get_padding_truncation_strategies(
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            verbose=verbose,
            **kwargs,
        )

        return self._encode_plus(
            text=text,
            text_pair=text_pair,
            add_special_tokens=add_special_tokens,
            padding_strategy=padding_strategy,
            truncation_strategy=truncation_strategy,
            max_length=max_length,
            stride=stride,
            is_split_into_words=is_split_into_words,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors=return_tensors,
            return_position_ids=return_position_ids,
            return_token_type_ids=return_token_type_ids,
            return_attention_mask=return_attention_mask,
            return_overflowing_tokens=return_overflowing_tokens,
            return_special_tokens_mask=return_special_tokens_mask,
            return_offsets_mapping=return_offsets_mapping,
            return_length=return_length,
            verbose=verbose,
            **kwargs,
        )

    def _encode_plus(
            self,
            text: Union[TextInput, PreTokenizedInput, EncodedInput],
            text_pair: Optional[Union[TextInput, PreTokenizedInput,
                                      EncodedInput]] = None,
            add_special_tokens: bool = True,
            padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
            truncation_strategy: TruncationStrategy = TruncationStrategy.
        DO_NOT_TRUNCATE,
            max_length: Optional[int] = None,
            stride: int = 0,
            is_split_into_words: bool = False,
            pad_to_multiple_of: Optional[int] = None,
            return_position_ids: Optional[bool] = None,
            return_tensors: Optional[Union[str, TensorType]] = None,
            return_token_type_ids: Optional[bool] = None,
            return_attention_mask: Optional[bool] = None,
            return_overflowing_tokens: bool = False,
            return_special_tokens_mask: bool = False,
            return_offsets_mapping: bool = False,
            return_length: bool = False,
            verbose: bool = True,
            **kwargs) -> BatchEncoding:
        raise NotImplementedError

    def batch_encode(
            self,
            batch_text_or_text_pairs: Union[List[TextInput],
                                            List[TextInputPair],
                                            List[PreTokenizedInput],
                                            List[PreTokenizedInputPair],
                                            List[EncodedInput],
                                            List[EncodedInputPair], ],
            max_length=None,
            stride: int = 0,
            is_split_into_words: bool = False,
            padding: Union[bool, str, PaddingStrategy] = False,
            truncation: Union[bool, str, TruncationStrategy] = False,
            return_position_ids=False,
            # TODO(wj-mcat): keep align with `encode` method
            return_token_type_ids=None,
            return_attention_mask=None,
            return_length=False,
            return_overflowing_tokens=False,
            return_special_tokens_mask=False,
            return_dict=True,
            return_offsets_mapping=False,
            add_special_tokens=True,
            pad_to_multiple_of: Optional[int] = None,
            return_tensors: Optional[Union[str, TensorType]] = None,
            verbose: bool = True,
            **kwargs) -> BatchEncoding:
        old_max_seq_len = kwargs.get('max_seq_len', None)
        if max_length is None and old_max_seq_len:
            if verbose:
                warnings.warn(
                    'The `max_seq_len` argument is deprecated and will be removed in a future version, '
                    'please use `max_length` instead.',
                    FutureWarning,
                )
            max_length = old_max_seq_len
        # Backward compatibility for 'truncation_strategy', 'pad_to_max_length'
        padding_strategy, truncation_strategy, max_length, kwargs = self._get_padding_truncation_strategies(
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            verbose=verbose,
            **kwargs,
        )

        return self._batch_encode_plus(
            batch_text_or_text_pairs=batch_text_or_text_pairs,
            add_special_tokens=add_special_tokens,
            padding_strategy=padding_strategy,
            truncation_strategy=truncation_strategy,
            max_length=max_length,
            stride=stride,
            is_split_into_words=is_split_into_words,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors=return_tensors,
            return_position_ids=return_position_ids,
            return_token_type_ids=return_token_type_ids,
            return_attention_mask=return_attention_mask,
            return_overflowing_tokens=return_overflowing_tokens,
            return_special_tokens_mask=return_special_tokens_mask,
            return_dict=return_dict,
            return_offsets_mapping=return_offsets_mapping,
            return_length=return_length,
            verbose=verbose,
            **kwargs,
        )

    def _batch_encode_plus(
            self,
            batch_text_or_text_pairs: Union[List[TextInput],
                                            List[TextInputPair],
                                            List[PreTokenizedInput],
                                            List[PreTokenizedInputPair],
                                            List[EncodedInput],
                                            List[EncodedInputPair], ],
            add_special_tokens: bool = True,
            padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
            truncation_strategy: TruncationStrategy = TruncationStrategy.
        DO_NOT_TRUNCATE,
            max_length: Optional[int] = None,
            stride: int = 0,
            is_split_into_words: bool = False,
            pad_to_multiple_of: Optional[int] = None,
            return_position_ids: Optional[bool] = None,
            return_tensors: Optional[Union[str, TensorType]] = None,
            return_token_type_ids: Optional[bool] = None,
            return_attention_mask: Optional[bool] = None,
            return_overflowing_tokens: bool = False,
            return_special_tokens_mask: bool = False,
            return_dict: bool = True,
            return_offsets_mapping: bool = False,
            return_length: bool = False,
            verbose: bool = True,
            **kwargs) -> BatchEncoding:
        raise NotImplementedError

    def pad(
        self,
        encoded_inputs: Union[BatchEncoding, List[BatchEncoding],
                              Dict[str,
                                   EncodedInput], Dict[str,
                                                       List[EncodedInput]],
                              List[Dict[str, EncodedInput]], ],
        padding: Union[bool, str, PaddingStrategy] = True,
        max_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        return_attention_mask: Optional[bool] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        verbose: bool = True,
    ) -> BatchEncoding:
        # If we have a list of dicts, let's convert it in a dict of lists
        if isinstance(encoded_inputs,
                      (list, tuple)) and isinstance(encoded_inputs[0],
                                                    (dict, BatchEncoding)):
            encoded_inputs = {
                key: [example[key] for example in encoded_inputs]
                for key in encoded_inputs[0].keys()
            }

        # The model's main input name, usually `input_ids`, has be passed for padding
        if self.model_input_names[0] not in encoded_inputs:
            raise ValueError(
                'You should supply an encoding or a list of encodings to this method '
                f'that includes {self.model_input_names[0]}, but you provided {list(encoded_inputs.keys())}'
            )

        required_input = encoded_inputs[self.model_input_names[0]]

        if not required_input:
            if return_attention_mask:
                encoded_inputs['attention_mask'] = []
            return encoded_inputs

        # If we have Paddle/NumPy tensors/arrays as inputs, we cast them as python objects
        # and rebuild them afterwards if no return_tensors is specified

        first_element = required_input[0]
        if isinstance(first_element, (list, tuple)):
            # first_element might be an empty list/tuple in some edge cases so we grab the first non empty element.
            for item in required_input:
                if len(item) != 0:
                    first_element = item[0]
                    break

        padding_strategy, _, max_length, _ = self._get_padding_truncation_strategies(
            padding=padding, max_length=max_length, verbose=verbose)

        required_input = encoded_inputs[self.model_input_names[0]]
        if required_input and not isinstance(required_input[0], (list, tuple)):
            encoded_inputs = self._pad(
                encoded_inputs,
                max_length=max_length,
                padding_strategy=padding_strategy,
                pad_to_multiple_of=pad_to_multiple_of,
                return_attention_mask=return_attention_mask,
            )
            return BatchEncoding(encoded_inputs, tensor_type=return_tensors)

        batch_size = len(required_input)
        assert all(
            len(v) == batch_size for v in encoded_inputs.values()
        ), 'Some items in the output dictionary have a different batch size than others.'

        if padding_strategy == PaddingStrategy.LONGEST:
            max_length = max(len(inputs) for inputs in required_input)
            padding_strategy = PaddingStrategy.MAX_LENGTH

        batch_outputs = {}
        for i in range(batch_size):
            inputs = dict((k, v[i]) for k, v in encoded_inputs.items())
            outputs = self._pad(
                inputs,
                max_length=max_length,
                padding_strategy=padding_strategy,
                pad_to_multiple_of=pad_to_multiple_of,
                return_attention_mask=return_attention_mask,
            )

            for key, value in outputs.items():
                if key not in batch_outputs:
                    batch_outputs[key] = []
                batch_outputs[key].append(value)

        return BatchEncoding(batch_outputs, tensor_type=return_tensors)

    def create_token_type_ids_from_sequences(
            self,
            token_ids_0: List[int],
            token_ids_1: Optional[List[int]] = None) -> List[int]:
        """
        Create the token type IDs corresponding to the sequences passed. [What are token type
        IDs?](../glossary#token-type-ids)

        Should be overridden in a subclass if the model has a special way of building those.

        Args:
            token_ids_0 (`List[int]`): The first tokenized sequence.
            token_ids_1 (`List[int]`, *optional*): The second tokenized sequence.

        Returns:
            `List[int]`: The token type ids.
        """
        if token_ids_1 is None:
            return len(token_ids_0) * [0]
        return [0] * len(token_ids_0) + [1] * len(token_ids_1)

    def build_inputs_with_special_tokens(
            self,
            token_ids_0: List[int],
            token_ids_1: Optional[List[int]] = None) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens.

        This implementation does not add special tokens and this method should be overridden in a subclass.

        Args:
            token_ids_0 (`List[int]`): The first tokenized sequence.
            token_ids_1 (`List[int]`, *optional*): The second tokenized sequence.

        Returns:
            `List[int]`: The model input with special tokens.
        """
        if token_ids_1 is None:
            return token_ids_0
        return token_ids_0 + token_ids_1

    def build_offset_mapping_with_special_tokens(self,
                                                 offset_mapping_0,
                                                 offset_mapping_1=None):
        """
        Build offset map from a pair of offset map by concatenating and adding offsets of special tokens.

        Should be overridden in a subclass if the model has a special way of building those.

        Args:
            offset_mapping_0 (List[tuple]):
                List of char offsets to which the special tokens will be added.
            offset_mapping_1 (List[tuple], optional):
                Optional second list of char offsets for offset mapping pairs.

        Returns:
            List[tuple]: List of char offsets with the appropriate offsets of special tokens.
        """
        if offset_mapping_1 is None:
            return offset_mapping_0

        return offset_mapping_0 + offset_mapping_1

    def prepare_for_model(self,
                          ids,
                          pair_ids=None,
                          padding: Union[bool, str, PaddingStrategy] = False,
                          truncation: Union[bool, str,
                                            TruncationStrategy] = False,
                          max_length: Optional[int] = None,
                          stride: int = 0,
                          pad_to_multiple_of: Optional[int] = None,
                          return_tensors: Optional[Union[str,
                                                         TensorType]] = None,
                          return_position_ids=False,
                          return_token_type_ids: Optional[bool] = None,
                          return_attention_mask: Optional[bool] = None,
                          return_length=False,
                          return_overflowing_tokens=False,
                          return_special_tokens_mask=False,
                          return_offsets_mapping=False,
                          add_special_tokens=True,
                          verbose: bool = True,
                          prepend_batch_axis: bool = False,
                          **kwargs):
        """
        Performs tokenization and uses the tokenized tokens to prepare model
        inputs. It supports sequence or sequence pair as input, and batch input
        is not allowed.
        """
        padding_strategy, truncation_strategy, max_length, kwargs = self._get_padding_truncation_strategies(
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            verbose=verbose,
            **kwargs,
        )

        pair = bool(pair_ids is not None)
        len_ids = len(ids)
        len_pair_ids = len(pair_ids) if pair else 0

        if return_token_type_ids and not add_special_tokens:
            raise ValueError(
                'Asking to return token_type_ids while setting add_special_tokens to False '
                'results in an undefined behavior. Please set add_special_tokens to True or '
                'set return_token_type_ids to None.')

        if (return_overflowing_tokens
                and truncation_strategy == TruncationStrategy.LONGEST_FIRST
                and pair_ids is not None):
            raise ValueError(
                'Not possible to return overflowing tokens for pair of sequences with the '
                '`longest_first`. Please select another truncation strategy than `longest_first`, '
                'for instance `only_second` or `only_first`.')

        # Load from model defaults
        if return_token_type_ids is None:
            return_token_type_ids = 'token_type_ids' in self.model_input_names
        if return_attention_mask is None:
            return_attention_mask = 'attention_mask' in self.model_input_names

        encoded_inputs = {}
        # Truncation: Handle max sequence length
        total_len = len_ids + len_pair_ids + (self.num_special_tokens_to_add(
            pair=pair) if add_special_tokens else 0)

        overflowing_tokens = []

        if truncation_strategy != TruncationStrategy.DO_NOT_TRUNCATE and max_length and total_len > max_length:
            ids, pair_ids, overflowing_tokens = self.truncate_sequences(
                ids,
                pair_ids=pair_ids,
                num_tokens_to_remove=total_len - max_length,
                truncation_strategy=truncation_strategy,
                stride=stride,
            )
        if return_overflowing_tokens:
            encoded_inputs['overflowing_tokens'] = overflowing_tokens
            encoded_inputs['num_truncated_tokens'] = total_len - max_length

        # Add special tokens
        if add_special_tokens:
            sequence = self.build_inputs_with_special_tokens(ids, pair_ids)
            token_type_ids = self.create_token_type_ids_from_sequences(
                ids, pair_ids)
        else:
            sequence = ids + pair_ids if pair else ids
            token_type_ids = [0] * len(ids) + ([0] *
                                               len(pair_ids) if pair else [])

        # Build output dictionnary
        encoded_inputs['input_ids'] = sequence
        if return_token_type_ids:
            encoded_inputs['token_type_ids'] = token_type_ids
        if return_special_tokens_mask:
            if add_special_tokens:
                encoded_inputs[
                    'special_tokens_mask'] = self.get_special_tokens_mask(
                        ids, pair_ids)
            else:
                encoded_inputs['special_tokens_mask'] = [0] * len(sequence)

        if return_offsets_mapping and 'text' in kwargs and 'text_pair' in kwargs:
            text = kwargs.pop('text')
            text_pair = kwargs.pop('text_pair')

            token_offset_mapping = self.get_offset_mapping(text)
            token_pair_offset_mapping = self.get_offset_mapping(
                text_pair) if text_pair is not None else None
            if max_length and total_len > max_length:
                token_offset_mapping, token_pair_offset_mapping, _ = self.truncate_sequences(
                    token_offset_mapping,
                    pair_ids=token_pair_offset_mapping,
                    num_tokens_to_remove=total_len - max_length,
                    truncation_strategy=truncation_strategy,
                    stride=stride,
                )
            if add_special_tokens:
                offset_mapping = self.build_offset_mapping_with_special_tokens(
                    token_offset_mapping, token_pair_offset_mapping)
            else:
                offset_mapping = (token_offset_mapping +
                                  token_pair_offset_mapping
                                  if token_pair_offset_mapping else
                                  token_offset_mapping)
            encoded_inputs['offset_mapping'] = offset_mapping

        self._eventual_warn_about_too_long_sequence(
            encoded_inputs['input_ids'], max_length, verbose)

        if return_position_ids:
            encoded_inputs['position_ids'] = list(
                range(len(encoded_inputs['input_ids'])))

        if padding_strategy != PaddingStrategy.DO_NOT_PAD or return_attention_mask:
            encoded_inputs = self.pad(
                encoded_inputs,
                max_length=max_length,
                padding=padding_strategy.value,
                pad_to_multiple_of=pad_to_multiple_of,
                return_attention_mask=return_attention_mask,
            )

        if return_length:
            encoded_inputs['length'] = len(encoded_inputs['input_ids'])
            # for compatibility
            encoded_inputs['seq_len'] = encoded_inputs['length']

        batch_outputs = BatchEncoding(encoded_inputs,
                                      tensor_type=return_tensors,
                                      prepend_batch_axis=prepend_batch_axis)

        return batch_outputs

    def truncate_sequences(
        self,
        ids: List[int],
        pair_ids: Optional[List[int]] = None,
        num_tokens_to_remove: int = 0,
        truncation_strategy: Union[str, TruncationStrategy] = 'longest_first',
        stride: int = 0,
    ) -> Tuple[List[int], List[int], List[int]]:
        if num_tokens_to_remove <= 0:
            return ids, pair_ids, []

        if not isinstance(truncation_strategy, TruncationStrategy):
            truncation_strategy = TruncationStrategy(truncation_strategy)

        overflowing_tokens = []
        if truncation_strategy == TruncationStrategy.ONLY_FIRST or (
                truncation_strategy == TruncationStrategy.LONGEST_FIRST
                and pair_ids is None):
            if len(ids) > num_tokens_to_remove:
                window_len = min(len(ids), stride + num_tokens_to_remove)
                if self.truncation_side == 'left':
                    overflowing_tokens = ids[:window_len]
                    ids = ids[num_tokens_to_remove:]
                elif self.truncation_side == 'right':
                    overflowing_tokens = ids[-window_len:]
                    ids = ids[:-num_tokens_to_remove]
                else:
                    raise ValueError(
                        f"invalid truncation strategy: {self.truncation_side}, use 'left' or 'right'."
                    )

            else:
                error_msg = (
                    f'We need to remove {num_tokens_to_remove} to truncate the input '
                    f'but the first sequence has a length {len(ids)}. ')
                if truncation_strategy == TruncationStrategy.ONLY_FIRST:
                    error_msg = (
                        error_msg +
                        'Please select another truncation strategy than '
                        f"{truncation_strategy}, for instance 'longest_first' or 'only_second'."
                    )
                logger.error(error_msg)
        elif truncation_strategy == TruncationStrategy.LONGEST_FIRST:
            warnings.warn(
                f'Be aware, overflowing tokens are not returned for the setting you have chosen,'
                f" i.e. sequence pairs with the '{TruncationStrategy.LONGEST_FIRST.value}' "
                f'truncation strategy. So the returned list will always be empty even if some '
                f'tokens have been removed.')
            for _ in range(num_tokens_to_remove):
                if pair_ids is None or len(ids) > len(pair_ids):
                    if self.truncation_side == 'right':
                        ids = ids[:-1]
                    elif self.truncation_side == 'left':
                        ids = ids[1:]
                    else:
                        raise ValueError('invalid truncation strategy:' +
                                         str(self.truncation_side))
                else:
                    if self.truncation_side == 'right':
                        pair_ids = pair_ids[:-1]
                    elif self.truncation_side == 'left':
                        pair_ids = pair_ids[1:]
                    else:
                        raise ValueError('invalid truncation strategy:' +
                                         str(self.truncation_side))
        elif truncation_strategy == TruncationStrategy.ONLY_SECOND and pair_ids is not None:
            if len(pair_ids) > num_tokens_to_remove:
                window_len = min(len(pair_ids), stride + num_tokens_to_remove)
                if self.truncation_side == 'right':
                    overflowing_tokens = pair_ids[-window_len:]
                    pair_ids = pair_ids[:-num_tokens_to_remove]
                elif self.truncation_side == 'left':
                    overflowing_tokens = pair_ids[:window_len]
                    pair_ids = pair_ids[num_tokens_to_remove:]
                else:
                    raise ValueError('invalid truncation strategy:' +
                                     str(self.truncation_side))
            else:
                logger.error(
                    f'We need to remove {num_tokens_to_remove} to truncate the input '
                    f'but the second sequence has a length {len(pair_ids)}. '
                    f'Please select another truncation strategy than {truncation_strategy}, '
                    f"for instance 'longest_first' or 'only_first'.")

        return (ids, pair_ids, overflowing_tokens)

    def _pad(
        self,
        encoded_inputs: Union[Dict[str, EncodedInput], BatchEncoding],
        max_length: Optional[int] = None,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        pad_to_multiple_of: Optional[int] = None,
        return_attention_mask: Optional[bool] = None,
    ) -> dict:
        if return_attention_mask is None:
            return_attention_mask = 'attention_mask' in self.model_input_names or 'attention_mask' in encoded_inputs

        required_input = encoded_inputs[self.model_input_names[0]]

        if padding_strategy == PaddingStrategy.LONGEST:
            max_length = len(required_input)

        if max_length is not None and pad_to_multiple_of is not None and (
                max_length % pad_to_multiple_of != 0):
            max_length = (
                (max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of

        needs_to_be_padded = padding_strategy != PaddingStrategy.DO_NOT_PAD and len(
            required_input) != max_length

        # Initialize attention mask if not present.
        if return_attention_mask and 'attention_mask' not in encoded_inputs:
            encoded_inputs['attention_mask'] = [1] * len(required_input)

        if needs_to_be_padded:
            difference = max_length - len(required_input)

            if self.padding_side == 'right':
                if return_attention_mask:

                    encoded_inputs['attention_mask'] = encoded_inputs[
                        'attention_mask'] + [0] * difference
                if 'token_type_ids' in encoded_inputs:
                    encoded_inputs['token_type_ids'] = (
                        encoded_inputs['token_type_ids'] +
                        [self.pad_token_type_id] * difference)
                if 'special_tokens_mask' in encoded_inputs:
                    encoded_inputs['special_tokens_mask'] = encoded_inputs[
                        'special_tokens_mask'] + [1] * difference
                if 'offset_mapping' in encoded_inputs:
                    encoded_inputs['offset_mapping'] = encoded_inputs[
                        'offset_mapping'] + [(0, 0)] * difference
                if 'position_ids' in encoded_inputs:
                    encoded_inputs['position_ids'] = encoded_inputs[
                        'position_ids'] + [0] * difference
                encoded_inputs[self.model_input_names[
                    0]] = required_input + [self.pad_token_id] * difference
            elif self.padding_side == 'left':
                if return_attention_mask:
                    encoded_inputs['attention_mask'] = [
                        0
                    ] * difference + encoded_inputs['attention_mask']
                if 'token_type_ids' in encoded_inputs:
                    encoded_inputs['token_type_ids'] = [
                        self.pad_token_type_id
                    ] * difference + encoded_inputs['token_type_ids']
                if 'special_tokens_mask' in encoded_inputs:
                    encoded_inputs['special_tokens_mask'] = [
                        1
                    ] * difference + encoded_inputs['special_tokens_mask']
                if 'offset_mapping' in encoded_inputs:
                    encoded_inputs['offset_mapping'] = [
                        (0, 0)
                    ] * difference + encoded_inputs['offset_mapping']
                if 'position_ids' in encoded_inputs:
                    encoded_inputs['position_ids'] = [
                        0
                    ] * difference + encoded_inputs['position_ids']
                encoded_inputs[self.model_input_names[
                    0]] = [self.pad_token_id] * difference + required_input
            else:
                raise ValueError('Invalid padding strategy:' +
                                 str(self.padding_side))

        return encoded_inputs

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """
        Converts a sequence of tokens in a single string. The most simple way to do it is `" ".join(tokens)` but we
        often want to remove sub-word tokenization artifacts at the same time.

        Args:
            tokens (`List[str]`): The token to join in a string.

        Returns:
            `str`: The joined tokens.
        """
        raise NotImplementedError

    def batch_decode(self,
                     sequences: Union[List[int], List[List[int]],
                                      'np.ndarray'],
                     skip_special_tokens: bool = False,
                     clean_up_tokenization_spaces: bool = True,
                     **kwargs) -> List[str]:
        return [
            self.decode(
                seq,
                skip_special_tokens=skip_special_tokens,
                clean_up_tokenization_spaces=clean_up_tokenization_spaces,
                **kwargs,
            ) for seq in sequences
        ]

    def decode(self,
               token_ids: Union[int, List[int], 'np.ndarray'],
               skip_special_tokens: bool = False,
               clean_up_tokenization_spaces: bool = True,
               **kwargs) -> str:
        # Convert inputs to python lists
        # token_ids = to_py_obj(token_ids)

        return self._decode(
            token_ids=token_ids,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            **kwargs,
        )

    def _decode(self,
                token_ids: Union[int, List[int]],
                skip_special_tokens: bool = False,
                clean_up_tokenization_spaces: bool = True,
                **kwargs) -> str:
        raise NotImplementedError

    def get_special_tokens_mask(
            self,
            token_ids_0: List[int],
            token_ids_1: Optional[List[int]] = None,
            already_has_special_tokens: bool = False) -> List[int]:
        """
        Retrieves sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer `prepare_for_model` or `encode_plus` methods.

        Args:
            token_ids_0 (`List[int]`):
                List of ids of the first sequence.
            token_ids_1 (`List[int]`, *optional*):
                List of ids of the second sequence.
            already_has_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not the token list is already formatted with special tokens for the model.

        Returns:
            A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """
        assert already_has_special_tokens and token_ids_1 is None, (
            'You cannot use ``already_has_special_tokens=False`` with this tokenizer. '
            'Please use a slow (full python) tokenizer to activate this argument. '
            'Or set `return_special_tokens_mask=True` when calling the encoding method '
            'to get the special tokens mask in any tokenizer. ')

        all_special_ids = self.all_special_ids  # cache the property

        special_tokens_mask = [
            1 if token in all_special_ids else 0 for token in token_ids_0
        ]

        return special_tokens_mask

    @staticmethod
    def clean_up_tokenization(out_string: str) -> str:
        """
        Clean up a list of simple English tokenization artifacts like spaces before punctuations and abbreviated forms.

        Args:
            out_string (`str`): The text to clean up.

        Returns:
            `str`: The cleaned-up string.
        """
        out_string = (out_string.replace(' .', '.').replace(' ?', '?').replace(
            ' !', '!').replace(' ,', ',').replace(" ' ", "'").replace(
                " n't",
                "n't").replace(" 'm", "'m").replace(" 's", "'s").replace(
                    " 've", "'ve").replace(" 're", "'re"))
        return out_string

    def _eventual_warn_about_too_long_sequence(self, ids: List[int],
                                               max_length: Optional[int],
                                               verbose: bool):
        """
        Depending on the input and internal state we might trigger a warning about a sequence that is too long for its
        corresponding model

        Args:
            ids (`List[str]`): The ids produced by the tokenization
            max_length (`int`, *optional*): The max_length desired (does not trigger a warning if it is set)
            verbose (`bool`): Whether or not to print more information and warnings.

        """
        if max_length is None and len(ids) > self.model_max_length and verbose:
            if not self.deprecation_warnings.get(
                    'sequence-length-is-longer-than-the-specified-maximum',
                    False):
                logger.warning(
                    'Token indices sequence length is longer than the specified maximum sequence length '
                    f'for this model ({len(ids)} > {self.model_max_length}). Running this sequence through the model '
                    'will result in indexing errors')
            self.deprecation_warnings[
                'sequence-length-is-longer-than-the-specified-maximum'] = True
