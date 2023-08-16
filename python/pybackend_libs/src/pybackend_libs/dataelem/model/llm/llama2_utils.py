# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import json
import os
import sys
import time
from pathlib import Path
from typing import List, Literal, Optional, Tuple, TypedDict

import torch
import torch.nn.functional as F

Role = Literal['system', 'user', 'assistant']


class Message(TypedDict):
    role: Role
    content: str


class CompletionPrediction(TypedDict, total=False):
    generation: str
    tokens: List[str]  # not required
    logprobs: List[float]  # not required


class ChatPrediction(TypedDict, total=False):
    generation: Message
    tokens: List[str]  # not required
    logprobs: List[float]  # not required


Dialog = List[Message]

B_INST, E_INST = '[INST]', '[/INST]'
B_SYS, E_SYS = '<<SYS>>\n', '\n<</SYS>>\n\n'

SPECIAL_TAGS = [B_INST, E_INST, '<<SYS>>', '<</SYS>>']
UNSAFE_ERROR = 'Error: special tags are not allowed as part of the prompt.'


class LlamaTokenizerHelper:
    def __init__(self, tokenizer):
        self.sp_model = tokenizer.sp_model
        # BOS / EOS token IDs
        self.n_words: int = self.sp_model.vocab_size()
        self.bos_id: int = self.sp_model.bos_id()
        self.eos_id: int = self.sp_model.eos_id()
        self.pad_id: int = self.sp_model.pad_id()
        assert self.sp_model.vocab_size() == self.sp_model.get_piece_size()

    def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
        assert type(s) is str
        t = self.sp_model.encode(s)
        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]
        return t

    def decode(self, t: List[int]) -> str:
        return self.sp_model.decode(t)

    def text_completion(self, prompts: List[str]):
        prompt_tokens = [self.encode(x, bos=True, eos=False) for x in prompts]
        return torch.LongTensor(prompt_tokens)

    def chat_completion(self, dialogs: List[Dialog]):
        prompt_tokens = []
        unsafe_requests = []
        for dialog in dialogs:
            unsafe_requests.append(
                any([
                    tag in msg['content'] for tag in SPECIAL_TAGS
                    for msg in dialog
                ]))
            if dialog[0]['role'] == 'system':
                dialog = [{
                    'role':
                    dialog[1]['role'],
                    'content':
                    B_SYS + dialog[0]['content'] + E_SYS +
                    dialog[1]['content'],
                }] + dialog[2:]
            assert all([
                msg['role'] == 'user' for msg in dialog[::2]
            ]) and all([msg['role'] == 'assistant' for msg in dialog[1::2]]), (
                "model only supports 'system', 'user' and 'assistant' roles, "
                "starting with 'system', then 'user' and alternating (u/a/u/a/u...)"
            )
            dialog_tokens: List[int] = sum(
                [
                    self.encode(
                        f"{B_INST} {(prompt['content']).strip()} {E_INST} {(answer['content']).strip()} ",
                        bos=True,
                        eos=True,
                    ) for prompt, answer in zip(
                        dialog[::2],
                        dialog[1::2],
                    )
                ],
                [],
            )
            assert (
                dialog[-1]['role'] == 'user'
            ), f"Last message must be from user, got {dialog[-1]['role']}"
            dialog_tokens += self.encode(
                f"{B_INST} {(dialog[-1]['content']).strip()} {E_INST}",
                bos=True,
                eos=False,
            )
            prompt_tokens.append(dialog_tokens)

        return torch.LongTensor(prompt_tokens)
