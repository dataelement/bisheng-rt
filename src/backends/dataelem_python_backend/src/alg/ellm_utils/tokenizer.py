# flake8: noqa
import os
import unicodedata
from typing import List, Optional

import sentencepiece as spm

from .tokenizer_utils import PretrainedTokenizer
from .tokenizer_utils_base import AddedToken

__all__ = [
    'AutoTokenizer',
]

SPIECE_UNDERLINE = '▁'


class ErnieLayoutTokenizer(PretrainedTokenizer):
    resource_files_names = {
        'sentencepiece_model_file': 'sentencepiece.bpe.model',
        'vocab_file': 'vocab.txt',
    }

    def __init__(self,
                 vocab_file,
                 sentencepiece_model_file,
                 do_tokenize_postprocess=False,
                 sep_token='[SEP]',
                 cls_token='[CLS]',
                 unk_token='[UNK]',
                 pad_token='[PAD]',
                 mask_token='[MASK]',
                 **kwargs):
        mask_token = AddedToken(mask_token,
                                lstrip=True, rstrip=False) if isinstance(
                                    mask_token, str) else mask_token
        self._sep_token = sep_token
        self._cls_token = cls_token
        self._unk_token = unk_token
        self._pad_token = pad_token
        self._mask_token = mask_token
        self.sp_model = spm.SentencePieceProcessor()
        self.vocab_file = vocab_file
        self.sentencepiece_model_file = sentencepiece_model_file
        if os.path.isfile(sentencepiece_model_file):
            self.sp_model.Load(sentencepiece_model_file)
        self.vocab_file = vocab_file
        self.do_tokenize_postprocess = do_tokenize_postprocess
        self.tokens_to_ids = {'[CLS]': 0, '[PAD]': 1, '[SEP]': 2, '[UNK]': 3}
        self.offset = 1
        self.tokens_to_ids['[MASK]'] = len(self.sp_model) + self.offset
        self.ids_to_tokens = {v: k for k, v in self.tokens_to_ids.items()}
        self.SP_CHAR_MAPPING = {}

        for ch in range(65281, 65375):
            if ch in [ord('～')]:
                self.SP_CHAR_MAPPING[chr(ch)] = chr(ch)
                continue
            self.SP_CHAR_MAPPING[chr(ch)] = chr(ch - 65248)

    def build_inputs_with_special_tokens(
            self,
            token_ids_0: List[int],
            token_ids_1: Optional[List[int]] = None) -> List[int]:
        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        return cls + token_ids_0 + sep + sep + token_ids_1 + sep

    def build_offset_mapping_with_special_tokens(self,
                                                 offset_mapping_0,
                                                 offset_mapping_1=None):
        if offset_mapping_1 is None:
            return [(0, 0)] + offset_mapping_0 + [(0, 0)]

        return [(0, 0)] + offset_mapping_0 + [(0, 0), (0, 0)
                                              ] + offset_mapping_1 + [(0, 0)]

    def get_special_tokens_mask(
            self,
            token_ids_0: List[int],
            token_ids_1: Optional[List[int]] = None,
            already_has_special_tokens: bool = False) -> List[int]:
        if already_has_special_tokens:
            if token_ids_1 is not None:
                raise ValueError(
                    'You should not supply a second sequence if the provided sequence of '
                    'ids is already formatted with special tokens for the model.'
                )
            return list(
                map(
                    lambda x: 1
                    if x in [self.sep_token_id, self.cls_token_id] else 0,
                    token_ids_0))

        if token_ids_1 is None:
            return [1] + ([0] * len(token_ids_0)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1, 1] + (
            [0] * len(token_ids_1)) + [1]

    def create_token_type_ids_from_sequences(
            self,
            token_ids_0: List[int],
            token_ids_1: Optional[List[int]] = None) -> List[int]:
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]

        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        return len(cls + token_ids_0 + sep + sep + token_ids_1 + sep) * [0]

    def get_offset_mapping(self, text):
        split_tokens = self._tokenize(text)
        normalized_text, char_mapping = '', []

        for i, ch in enumerate(text):

            if ch in self.SP_CHAR_MAPPING:
                ch = self.SP_CHAR_MAPPING.get(ch)
            else:
                ch = unicodedata.normalize('NFKC', ch)
            if self.is_whitespace(ch):
                continue
            normalized_text += ch
            char_mapping.extend([i] * len(ch))

        text, token_mapping, offset = normalized_text, [], 0
        for token in split_tokens:
            if token[:1] == '▁':
                token = token[1:]
                if not token:
                    continue
            start = text[offset:].index(token) + offset
            end = start + len(token)

            token_mapping.append(
                (char_mapping[start], char_mapping[end - 1] + 1))
            offset = end
        return token_mapping

    @property
    def vocab_size(self):
        return len(self.sp_model) + self.offset + 1  # Add the <mask> token

    def get_vocab(self):
        vocab = {
            self.convert_ids_to_tokens(i): i
            for i in range(self.vocab_size)
        }
        vocab.update(self.added_tokens_encoder)
        return vocab

    def _tokenize(self, text):
        """Tokenize a string."""
        pieces = self.sp_model.EncodeAsPieces(text)
        if self.do_tokenize_postprocess:
            new_pieces = []
            for piece in pieces:
                if piece == SPIECE_UNDERLINE:
                    continue
                lst_i = 0
                for i, c in enumerate(piece):
                    if c == SPIECE_UNDERLINE:
                        continue
                    if self.is_ch_char(c) or self.is_punct(c):
                        if i > lst_i and piece[lst_i:i] != SPIECE_UNDERLINE:
                            new_pieces.append(piece[lst_i:i])
                        new_pieces.append(c)
                        lst_i = i + 1
                    elif c.isdigit() and i > 0 and not piece[i - 1].isdigit():
                        if i > lst_i and piece[lst_i:i] != SPIECE_UNDERLINE:
                            new_pieces.append(piece[lst_i:i])
                        lst_i = i
                    elif not c.isdigit() and i > 0 and piece[i - 1].isdigit():
                        if i > lst_i and piece[lst_i:i] != SPIECE_UNDERLINE:
                            new_pieces.append(piece[lst_i:i])
                        lst_i = i
                if len(piece) > lst_i:
                    new_pieces.append(piece[lst_i:])
            pieces = new_pieces
        return pieces

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        if token in self.tokens_to_ids:
            return self.tokens_to_ids[token]
        spm_id = self.sp_model.PieceToId(token)

        # Need to return unknown token if the SP model returned 0
        return spm_id + self.offset if spm_id else self.unk_token_id

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        if index in self.ids_to_tokens:
            return self.ids_to_tokens[index]
        return self.sp_model.IdToPiece(index - self.offset)

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (strings for sub-words) in a single string."""
        out_string = ''.join(tokens).replace(SPIECE_UNDERLINE, ' ').strip()
        return out_string

    def num_special_tokens_to_add(self, pair=False):
        token_ids_0 = []
        token_ids_1 = []
        return len(
            self.build_inputs_with_special_tokens(
                token_ids_0, token_ids_1 if pair else None))

    def is_ch_char(self, char):
        """
        is_ch_char
        """
        if '\u4e00' <= char <= '\u9fff':
            return True
        return False

    def is_alpha(self, char):
        """
        is_alpha
        """
        if 'a' <= char <= 'z':
            return True
        if 'A' <= char <= 'Z':
            return True
        return False

    def is_punct(self, char):
        """
        is_punct
        """
        if char in ',;:.?!~，；：。？！《》【】':
            return True
        return False

    def is_whitespace(self, char):
        """
        is whitespace
        """
        if char == ' ' or char == '\t' or char == '\n' or char == '\r':
            return True
        if len(char) == 1:
            cat = unicodedata.category(char)
            if cat == 'Zs':
                return True
        return False
