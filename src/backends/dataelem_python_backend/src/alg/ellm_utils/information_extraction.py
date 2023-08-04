# flake8: noqa
import abc
import base64
import json
import math
import os
import re
from io import BytesIO
from multiprocessing import cpu_count

import numpy as np
import triton_python_backend_utils as pb_utils
from PIL import Image

from .doc_parser import DocParser
from .env import CONFIG_NAME
from .image_utils import NormalizeImage, Permute, ResizeImage
from .tokenizer import ErnieLayoutTokenizer
from .tools import get_bool_ids_greater_than, get_span
from .utils import SchemaTree, cut_chinese_sent, dbc2sbc, get_id_and_prob
from torch.utils.dlpack import from_dlpack

def pb_tensor_to_numpy(pb_tensor):
    if pb_tensor.is_cpu():
        return pb_tensor.as_numpy()
    else:
        pytorch_tensor = from_dlpack(pb_tensor.to_dlpack())
        return pytorch_tensor.detach().cpu().numpy()

def map_offset(ori_offset, offset_mapping):
    """
    map ori offset to token offset
    """
    for index, span in enumerate(offset_mapping):
        if span[0] <= ori_offset < span[1]:
            return index
    return -1


resize_func = ResizeImage(target_size=224, interp=1)
norm_func = NormalizeImage(is_channel_first=False,
                           mean=[123.675, 116.280, 103.530],
                           std=[58.395, 57.120, 57.375])
permute_func = Permute(to_bgr=False)


def pad_image_data(image_data):
    if not image_data:
        image = np.zeros([3, 224, 224])
        return image
    # decode image
    data = np.frombuffer(bytearray(image_data), dtype='uint8')
    image = np.array(Image.open(BytesIO(data)).convert('RGB'))
    sample = {'image': image}
    # resize image
    sample = resize_func(sample)
    # norm image
    sample = norm_func(sample)
    # permute
    sample = permute_func(sample)
    return sample['image']


class UIETask(metaclass=abc.ABCMeta):
    resource_files_names = {
        'model_state': 'model_state.pdparams',
        'config': 'config.json',
        'vocab_file': 'vocab.txt',
        'special_tokens_map': 'special_tokens_map.json',
        'tokenizer_config': 'tokenizer_config.json',
    }

    def __init__(self, model, schema=None, **kwargs):
        self.kwargs = kwargs
        self._max_seq_len = kwargs.get('max_seq_len', 512)
        self._batch_size = kwargs.get('batch_size', 16)
        self._split_sentence = kwargs.get('split_sentence', False)
        self._position_prob = kwargs.get('position_prob', 0.5)
        self._lazy_load = kwargs.get('lazy_load', False)
        self._num_workers = kwargs.get('num_workers', 0)
        self._use_fast = kwargs.get('use_fast', False)
        self._layout_analysis = kwargs.get('layout_analysis', False)
        self._ocr_lang = kwargs.get('ocr_lang', 'ch')
        self._schema_lang = kwargs.get('schema_lang', 'ch')
        self._is_trt = kwargs.get("is_trt", False)
        self._expand_to_a4_size = False
        self.resource_files_names[
            'sentencepiece_model_file'] = 'sentencepiece.bpe.model'

        self._num_threads = kwargs[
            'num_threads'] if 'num_threads' in kwargs else math.ceil(
                cpu_count() / 2)
        self._task_path = kwargs['task_path']

        self.dep_model_name = kwargs.get('graph_name', 'ELLM_graph')
        self.dep_model_version = -1
        self.dep_model_inputs = [
            'input_ids', 'token_type_ids', 'position_ids', 'attention_mask',
            'bbox', 'image'
        ]
        self.dep_model_outputs = ['sigmoid_1.tmp_0', 'sigmoid_2.tmp_0']

        with open(os.path.join(self._task_path, CONFIG_NAME)) as f:
            self._init_class = json.load(f)['architectures'].pop()

        self._is_en = True if model in [
            'uie-base-en'
        ] or self._schema_lang == 'en' else False

        if self._init_class in ['UIEX']:
            self._summary_token_num = 4  # [CLS] prompt [SEP] [SEP] text [SEP] for UIE-X
        else:
            self._summary_token_num = 3  # [CLS] prompt [SEP] text [SEP]

        self._parser_map = {
            'ch': None,  # OCR-CH
            'en': None,  # OCR-EN
            'ch-layout': None,  # Layout-CH
            'en-layout': None,  # Layout-EN
        }

        if schema is not None:
            self.set_schema(schema)
        self._construct_tokenizer()

    def __call__(self, *args):
        inputs = self._preprocess(*args)
        outputs = self._run_model(inputs)
        results = self._postprocess(outputs)
        return results

    def set_argument(self, argument: dict):
        for k, v in argument.items():
            if k == 'input':
                continue
            setattr(self, f'_{k}', v)

    def set_schema(self, schema):
        if isinstance(schema, dict) or isinstance(schema, str):
            schema = [schema]
        self._schema_tree = self._build_tree(schema)

    def _check_input_text(self, inputs):
        """
        Check whether the input text meet the requirement.
        """
        inputs = inputs[0]
        if isinstance(inputs, str):
            if len(inputs) == 0:
                raise ValueError(
                    'Invalid inputs, input text should not be empty text, please check your input.'
                )
            inputs = [inputs]
        elif isinstance(inputs, list):
            if not (isinstance(inputs[0], str) and len(inputs[0].strip()) > 0):
                raise TypeError(
                    'Invalid inputs, input text should be list of str, and first element of list should not be empty text.'
                )
        else:
            raise TypeError(
                'Invalid inputs, input text should be str or list of str, but type of {} found!'
                .format(type(inputs)))
        return inputs

    def _auto_splitter(self,
                       input_texts,
                       max_text_len,
                       bbox_list=None,
                       split_sentence=False):
        input_mapping = {}
        short_input_texts = []
        cnt_org = 0
        cnt_short = 0
        with_bbox = False
        if bbox_list:
            with_bbox = True
            short_bbox_list = []
            if split_sentence:
                split_sentence = False

        for idx in range(len(input_texts)):
            if not split_sentence:
                sens = [input_texts[idx]]
            else:
                sens = cut_chinese_sent(input_texts[idx])
            for sen in sens:
                lens = len(sen)
                if lens <= max_text_len:
                    short_input_texts.append(sen)
                    if with_bbox:
                        short_bbox_list.append(bbox_list[idx])
                    input_mapping.setdefault(cnt_org, []).append(cnt_short)
                    cnt_short += 1
                else:
                    temp_text_list = [
                        sen[i:i + max_text_len]
                        for i in range(0, lens, max_text_len)
                    ]
                    short_input_texts.extend(temp_text_list)
                    if with_bbox:
                        temp_bbox_list = [
                            bbox_list[idx][i:i + max_text_len]
                            for i in range(0, lens, max_text_len)
                        ]
                        short_bbox_list.extend(temp_bbox_list)
                    short_idx = cnt_short
                    cnt_short += math.ceil(lens / max_text_len)
                    temp_text_id = [
                        short_idx + i for i in range(cnt_short - short_idx)
                    ]
                    input_mapping.setdefault(cnt_org, []).extend(temp_text_id)
            cnt_org += 1
        if with_bbox:
            return short_input_texts, short_bbox_list, input_mapping
        else:
            return short_input_texts, input_mapping

    def _auto_joiner(self, short_results, input_mapping, is_dict=False):
        concat_results = []
        elem_type = {} if is_dict else []
        for k, vs in input_mapping.items():
            single_results = elem_type
            for v in vs:
                if len(single_results) == 0:
                    single_results = short_results[v]
                elif isinstance(elem_type, list):
                    single_results.extend(short_results[v])
                elif isinstance(elem_type, dict):
                    for sk in single_results.keys():
                        if isinstance(single_results[sk], str):
                            single_results[sk] += short_results[v][sk]
                        else:
                            single_results[sk].extend(short_results[v][sk])
                else:
                    raise ValueError(
                        'Invalid element type, the type of results '
                        'for each element should be list of dict, '
                        'but {} received.'.format(type(single_results)))
            concat_results.append(single_results)
        return concat_results

    def _construct_tokenizer(self):
        """
        Construct the tokenizer for the predictor.
        """
        self._tokenizer = ErnieLayoutTokenizer.from_pretrained(
            self._task_path, use_fast=False, from_hf_hub=False)

    def _preprocess(self, inputs):
        """
        Transform the raw text to the model inputs, two steps involved:
           1) Transform the raw text to token ids.
           2) Generate the other model inputs from the raw text and token ids.
        """
        inputs = self._check_input_text(inputs)
        outputs = {}
        outputs['text'] = inputs
        return outputs

    def _check_input_text(self, inputs):
        """
        Check whether the input meet the requirement.
        """
        self._ocr_lang_choice = (
            self._ocr_lang +
            '-layout') if self._layout_analysis else self._ocr_lang
        inputs = inputs[0]
        if isinstance(inputs, dict) or isinstance(inputs, str):
            inputs = [inputs]
        if isinstance(inputs, list):
            input_list = []
            for example in inputs:
                data = {}
                if isinstance(example, dict):
                    if 'doc' in example.keys():
                        if not self._parser_map[self._ocr_lang_choice]:
                            self._parser_map[
                                self._ocr_lang_choice] = DocParser(
                                    ocr_lang=self._ocr_lang,
                                    layout_analysis=self._layout_analysis)
                        if 'layout' in example.keys():
                            data = self._parser_map[
                                self._ocr_lang_choice].parse(
                                    {'doc': example['doc']},
                                    do_ocr=False,
                                    expand_to_a4_size=self._expand_to_a4_size)
                            data['layout'] = example['layout']
                        else:
                            data = self._parser_map[
                                self._ocr_lang_choice].parse(
                                    {'doc': example['doc']},
                                    expand_to_a4_size=self._expand_to_a4_size)
                    elif 'text' in example.keys():
                        if not isinstance(example['text'], str):
                            raise TypeError(
                                'Invalid inputs, the input text should be string. but type of {} found!'
                                .format(type(example['text'])))
                        data['text'] = example['text']
                    else:
                        raise ValueError(
                            'Invalid inputs, the input should contain a doc or a text.'
                        )
                    input_list.append(data)
                elif isinstance(example, str):
                    input_list.append(example)
                else:
                    raise TypeError(
                        'Invalid inputs, the input should be dict or list of dict, but type of {} found!'
                        .format(type(example)))
        else:
            raise TypeError('Invalid input format!')
        return input_list

    def _single_stage_predict(self, inputs):
        input_texts = [d['text'] for d in inputs]
        prompts = [d['prompt'] for d in inputs]

        max_predict_len = self._max_seq_len - len(
            max(prompts)) - self._summary_token_num

        if self._init_class in ['UIEX']:
            bbox_list = [d['bbox'] for d in inputs]
            short_input_texts, short_bbox_list, input_mapping = self._auto_splitter(
                input_texts,
                max_predict_len,
                bbox_list=bbox_list,
                split_sentence=self._split_sentence)
        else:
            short_input_texts, input_mapping = self._auto_splitter(
                input_texts,
                max_predict_len,
                split_sentence=self._split_sentence)

        short_texts_prompts = []
        for k, v in input_mapping.items():
            short_texts_prompts.extend([prompts[k] for _ in range(len(v))])
        if self._init_class in ['UIEX']:
            image_list = []
            for k, v in input_mapping.items():
                image_list.extend([inputs[k]['image'] for _ in range(len(v))])
            short_inputs = [{
                'text': short_input_texts[i],
                'prompt': short_texts_prompts[i],
                'bbox': short_bbox_list[i],
                'image': image_list[i],
            } for i in range(len(short_input_texts))]
        else:
            short_inputs = [{
                'text': short_input_texts[i],
                'prompt': short_texts_prompts[i]
            } for i in range(len(short_input_texts))]

        def doc_reader(inputs, pad_id=1, c_sep_id=2):
            def _process_bbox(tokens, bbox_lines, offset_mapping, offset_bias):
                bbox_list = [[0, 0, 0, 0] for x in range(len(tokens))]

                for index, bbox in enumerate(bbox_lines):
                    index_token = map_offset(index + offset_bias,
                                             offset_mapping)
                    if 0 <= index_token < len(bbox_list):
                        bbox_list[index_token] = bbox
                return bbox_list

            def _encode_doc(tokenizer, offset_mapping, last_offset, prompt,
                            this_text_line, inputs_ids, q_sep_index,
                            max_seq_len):
                if len(offset_mapping) == 0:
                    content_encoded_inputs = tokenizer(
                        text=[prompt],
                        text_pair=[this_text_line],
                        max_seq_len=max_seq_len,
                        return_dict=False,
                        return_offsets_mapping=True,
                    )

                    content_encoded_inputs = content_encoded_inputs[0]
                    inputs_ids = content_encoded_inputs['input_ids'][:-1]
                    sub_offset_mapping = [
                        list(x)
                        for x in content_encoded_inputs['offset_mapping']
                    ]
                    q_sep_index = content_encoded_inputs['input_ids'].index(
                        2, 1)

                    bias = 0
                    for index in range(len(sub_offset_mapping)):
                        if index == 0:
                            continue
                        mapping = sub_offset_mapping[index]
                        if mapping[0] == 0 and mapping[1] == 0 and bias == 0:
                            bias = sub_offset_mapping[index - 1][-1] + 1
                        if mapping[0] == 0 and mapping[1] == 0:
                            continue
                        if mapping == sub_offset_mapping[index - 1]:
                            continue
                        sub_offset_mapping[index][0] += bias
                        sub_offset_mapping[index][1] += bias

                    offset_mapping = sub_offset_mapping[:-1]
                    last_offset = offset_mapping[-1][-1]
                else:
                    content_encoded_inputs = tokenizer(
                        text=this_text_line,
                        max_seq_len=max_seq_len,
                        return_dict=False,
                        return_offsets_mapping=True)

                    inputs_ids += content_encoded_inputs['input_ids'][1:-1]
                    sub_offset_mapping = [
                        list(x)
                        for x in content_encoded_inputs['offset_mapping']
                    ]

                    for i, sub_list in enumerate(sub_offset_mapping[1:-1]):
                        if i == 0:
                            org_offset = sub_list[1]
                        else:
                            if sub_list[0] != org_offset and sub_offset_mapping[
                                    1:-1][i - 1] != sub_list:
                                last_offset += 1
                            org_offset = sub_list[1]
                        offset_mapping += [[
                            last_offset,
                            sub_list[1] - sub_list[0] + last_offset
                        ]]
                        last_offset = offset_mapping[-1][-1]
                return offset_mapping, last_offset, q_sep_index, inputs_ids

            for example in inputs:
                content = example['text']
                prompt = example['prompt']
                bbox_lines = example.get('bbox', None)
                image_buff_string = example.get('image', None)
                # Text
                if bbox_lines is None:
                    encoded_inputs = self._tokenizer(
                        text=[example['prompt']],
                        text_pair=[example['text']],
                        truncation=True,
                        max_seq_len=self._max_seq_len,
                        pad_to_max_seq_len=True,
                        return_attention_mask=True,
                        return_position_ids=True,
                        return_offsets_mapping=True,
                        return_dict=False,
                    )

                    encoded_inputs = encoded_inputs[0]

                    inputs_ids = encoded_inputs['input_ids']
                    position_ids = encoded_inputs['position_ids']
                    attention_mask = encoded_inputs['attention_mask']

                    q_sep_index = inputs_ids.index(2, 1)
                    c_sep_index = attention_mask.index(0)

                    offset_mapping = [
                        list(x) for x in encoded_inputs['offset_mapping']
                    ]

                    bbox_list = [[0, 0, 0, 0] for x in range(len(inputs_ids))]
                    token_type_ids = [
                        1 if token_index <= q_sep_index
                        or token_index > c_sep_index else 0
                        for token_index in range(self._max_seq_len)
                    ]
                    padded_image = np.zeros([3, 224, 224])
                # Doc
                else:
                    inputs_ids = []
                    prev_bbox = [-1, -1, -1, -1]
                    this_text_line = ''
                    q_sep_index = -1
                    offset_mapping = []
                    last_offset = 0
                    for char_index, (char, bbox) in enumerate(
                            zip(content, bbox_lines)):
                        if char_index == 0:
                            prev_bbox = bbox
                            this_text_line = char
                            continue

                        if all([bbox[x] == prev_bbox[x] for x in range(4)]):
                            this_text_line += char
                        else:
                            offset_mapping, last_offset, q_sep_index, inputs_ids = _encode_doc(
                                self._tokenizer,
                                offset_mapping,
                                last_offset,
                                prompt,
                                this_text_line,
                                inputs_ids,
                                q_sep_index,
                                self._max_seq_len,
                            )
                            this_text_line = char
                        prev_bbox = bbox
                    if len(this_text_line) > 0:
                        offset_mapping, last_offset, q_sep_index, inputs_ids = _encode_doc(
                            self._tokenizer,
                            offset_mapping,
                            last_offset,
                            prompt,
                            this_text_line,
                            inputs_ids,
                            q_sep_index,
                            self._max_seq_len,
                        )
                    if len(inputs_ids) > self._max_seq_len:
                        inputs_ids = inputs_ids[:(self._max_seq_len -
                                                  1)] + [c_sep_id]
                        offset_mapping = offset_mapping[:(self._max_seq_len -
                                                          1)] + [[0, 0]]
                    else:
                        inputs_ids += [c_sep_id]
                        offset_mapping += [[0, 0]]

                    if len(offset_mapping) > 1:
                        offset_bias = offset_mapping[q_sep_index - 1][-1] + 1
                    else:
                        offset_bias = 0

                    seq_len = len(inputs_ids)
                    inputs_ids += [pad_id] * (self._max_seq_len - seq_len)
                    token_type_ids = [1] * (q_sep_index + 1) + [0] * (
                        seq_len - q_sep_index - 1)
                    token_type_ids += [pad_id] * (self._max_seq_len - seq_len)

                    bbox_list = _process_bbox(inputs_ids, bbox_lines,
                                              offset_mapping, offset_bias)

                    offset_mapping += [[0, 0]] * (self._max_seq_len - seq_len)

                    # Reindex the text
                    text_start_idx = offset_mapping[1:].index(
                        [0, 0]) + self._summary_token_num - 1
                    for idx in range(text_start_idx, self._max_seq_len):
                        offset_mapping[idx][0] -= offset_bias
                        offset_mapping[idx][1] -= offset_bias

                    position_ids = list(range(seq_len))

                    position_ids = position_ids + [0] * (self._max_seq_len -
                                                         seq_len)
                    attention_mask = [1] * seq_len + [0] * (self._max_seq_len -
                                                            seq_len)

                    image_data = base64.b64decode(
                        image_buff_string.encode('utf8'))
                    padded_image = pad_image_data(image_data)

                input_list = [
                    inputs_ids,
                    token_type_ids,
                    position_ids,
                    attention_mask,
                    bbox_list,
                    padded_image,
                    offset_mapping,
                ]
                input_list = [
                    inputs_ids, token_type_ids, position_ids, attention_mask,
                    bbox_list
                ]
                return_list = [np.array(x, dtype='int64') for x in input_list]
                return_list.append(np.array(padded_image, dtype='float32'))
                return_list.append(np.array(offset_mapping, dtype='int64'))
                assert len(inputs_ids) == self._max_seq_len
                assert len(token_type_ids) == self._max_seq_len
                assert len(position_ids) == self._max_seq_len
                assert len(attention_mask) == self._max_seq_len
                assert len(bbox_list) == self._max_seq_len
                yield tuple(return_list)

        sentence_ids = []
        probs = []
        for batch in doc_reader(short_inputs):
            inputs = [np.expand_dims(x, axis=0) for x in batch[:-1]]
            if self._is_trt:
                inputs[:5] = [x.astype(np.int32) for x in inputs[:5]]
            offset_maps = np.expand_dims(batch[-1], axis=0)
            input_tensors = []
            for index, input_ in enumerate(inputs):
                in_tensor = pb_utils.Tensor(self.dep_model_inputs[index],
                                            input_)
                input_tensors.append(in_tensor)

            infer_request = pb_utils.InferenceRequest(
                model_name=self.dep_model_name,
                model_version=self.dep_model_version,
                requested_output_names=self.dep_model_outputs,
                inputs=input_tensors)

            infer_response = infer_request.exec()
            if infer_response.has_error():
                raise pb_utils.TritonModelException(
                    infer_response.error().message())

            graph_outputs = []
            for index, output_ in enumerate(self.dep_model_outputs):
                pb_tensor = pb_utils.get_output_tensor_by_name(
                    infer_response, output_)
                graph_outputs.append(pb_tensor_to_numpy(pb_tensor))

            start_prob = graph_outputs[0].tolist()
            end_prob = graph_outputs[1].tolist()

            start_ids_list = get_bool_ids_greater_than(
                start_prob, limit=self._position_prob, return_prob=True)
            end_ids_list = get_bool_ids_greater_than(end_prob,
                                                     limit=self._position_prob,
                                                     return_prob=True)

            for start_ids, end_ids, offset_map in zip(start_ids_list,
                                                      end_ids_list,
                                                      offset_maps.tolist()):
                span_set = get_span(start_ids, end_ids, with_prob=True)
                sentence_id, prob = get_id_and_prob(span_set, offset_map)
                sentence_ids.append(sentence_id)
                probs.append(prob)
        results = self._convert_ids_to_results(short_inputs, sentence_ids,
                                               probs)
        results = self._auto_joiner(results, short_input_texts, input_mapping)
        return results

    def _auto_joiner(self, short_results, short_inputs, input_mapping):
        concat_results = []
        is_cls_task = False
        for short_result in short_results:
            if short_result == []:
                continue
            elif 'start' not in short_result[0].keys(
            ) and 'end' not in short_result[0].keys():
                is_cls_task = True
                break
            else:
                break
        for k, vs in input_mapping.items():
            if is_cls_task:
                cls_options = {}
                single_results = []
                for v in vs:
                    if len(short_results[v]) == 0:
                        continue
                    if short_results[v][0]['text'] not in cls_options.keys():
                        cls_options[short_results[v][0]['text']] = [
                            1, short_results[v][0]['probability']
                        ]
                    else:
                        cls_options[short_results[v][0]['text']][0] += 1
                        cls_options[short_results[v][0]['text']][
                            1] += short_results[v][0]['probability']
                if len(cls_options) != 0:
                    cls_res, cls_info = max(cls_options.items(),
                                            key=lambda x: x[1])
                    concat_results.append([{
                        'text':
                        cls_res,
                        'probability':
                        cls_info[1] / cls_info[0]
                    }])
                else:
                    concat_results.append([])
            else:
                offset = 0
                single_results = []
                for v in vs:
                    if v == 0:
                        single_results = short_results[v]
                        offset += len(short_inputs[v])
                    else:
                        for i in range(len(short_results[v])):
                            if 'start' not in short_results[v][
                                    i] or 'end' not in short_results[v][i]:
                                continue
                            short_results[v][i]['start'] += offset
                            short_results[v][i]['end'] += offset
                        offset += len(short_inputs[v])
                        single_results.extend(short_results[v])
                concat_results.append(single_results)
        return concat_results

    def _run_model(self, inputs):
        raw_inputs = inputs['text']
        _inputs = self._parse_inputs(raw_inputs)
        results = self._multi_stage_predict(_inputs)
        inputs['result'] = results
        return inputs

    def _parse_inputs(self, inputs):
        _inputs = []
        for d in inputs:
            if isinstance(d, dict):
                if 'doc' in d.keys():
                    text = ''
                    bbox = []
                    img_w, img_h = d['img_w'], d['img_h']
                    offset_x, offset_y = d['offset_x'], d['offset_x']
                    for segment in d['layout']:
                        org_box = segment[0]  # bbox before expand to A4 size
                        box = [
                            org_box[0] + offset_x,
                            org_box[1] + offset_y,
                            org_box[2] + offset_x,
                            org_box[3] + offset_y,
                        ]
                        box = self._parser_map[
                            self._ocr_lang_choice]._normalize_box(
                                box, [img_w, img_h], [1000, 1000])
                        box = np.clip(box, 0, 1000).tolist()
                        text += segment[1]
                        bbox.extend([box] * len(segment[1]))
                    _inputs.append({
                        'text': text,
                        'bbox': bbox,
                        'image': d['image'],
                        'layout': d['layout']
                    })
                else:
                    _inputs.append({
                        'text': d['text'],
                        'bbox': None,
                        'image': None
                    })
            else:
                _inputs.append({'text': d, 'bbox': None, 'image': None})
        return _inputs

    def _multi_stage_predict(self, data):
        """
        Traversal the schema tree and do multi-stage prediction.

        Args:
            data (list): a list of strings

        Returns:
            list: a list of predictions, where the list's length
                equals to the length of `data`
        """
        results = [{} for _ in range(len(data))]
        # Input check to early return
        if len(data) < 1 or self._schema_tree is None:
            return results

        schema_list = self._schema_tree.children[:]
        while len(schema_list) > 0:
            node = schema_list.pop(0)
            #print(schema_list, node)
            examples = []
            input_map = {}
            cnt = 0
            idx = 0
            if not node.prefix:
                for one_data in data:
                    examples.append({
                        'text': one_data['text'],
                        'bbox': one_data['bbox'],
                        'image': one_data['image'],
                        'prompt': dbc2sbc(node.name),
                    })
                    input_map[cnt] = [idx]
                    idx += 1
                    cnt += 1
            else:
                for pre, one_data in zip(node.prefix, data):
                    if len(pre) == 0:
                        input_map[cnt] = []
                    else:
                        for p in pre:
                            if self._is_en:
                                if re.search(r'\[.*?\]$', node.name):
                                    prompt_prefix = node.name[:node.name.find(
                                        '[', 1)].strip()
                                    cls_options = re.search(
                                        r'\[.*?\]$', node.name).group()
                                    prompt = prompt_prefix + p + ' ' + cls_options
                                else:
                                    prompt = node.name + p
                            else:
                                prompt = p + node.name
                            examples.append({
                                'text': one_data['text'],
                                'bbox': one_data['bbox'],
                                'image': one_data['image'],
                                'prompt': dbc2sbc(prompt),
                            })
                        input_map[cnt] = [i + idx for i in range(len(pre))]
                        idx += len(pre)
                    cnt += 1
            if len(examples) == 0:
                result_list = []
            else:
                result_list = self._single_stage_predict(examples)

            if not node.parent_relations:
                relations = [[] for i in range(len(data))]
                for k, v in input_map.items():
                    for idx in v:
                        if len(result_list[idx]) == 0:
                            continue
                        if node.name not in results[k].keys():
                            results[k][node.name] = result_list[idx]
                        else:
                            results[k][node.name].extend(result_list[idx])
                    if node.name in results[k].keys():
                        relations[k].extend(results[k][node.name])
            else:
                relations = node.parent_relations
                for k, v in input_map.items():
                    for i in range(len(v)):
                        if len(result_list[v[i]]) == 0:
                            continue
                        if 'relations' not in relations[k][i].keys():
                            relations[k][i]['relations'] = {
                                node.name: result_list[v[i]]
                            }
                        elif node.name not in relations[k][i][
                                'relations'].keys():
                            relations[k][i]['relations'][
                                node.name] = result_list[v[i]]
                        else:
                            relations[k][i]['relations'][node.name].extend(
                                result_list[v[i]])
                new_relations = [[] for i in range(len(data))]
                for i in range(len(relations)):
                    for j in range(len(relations[i])):
                        if 'relations' in relations[i][j].keys(
                        ) and node.name in relations[i][j]['relations'].keys():
                            for k in range(
                                    len(relations[i][j]['relations'][
                                        node.name])):
                                new_relations[i].append(
                                    relations[i][j]['relations'][node.name][k])
                relations = new_relations

            prefix = [[] for _ in range(len(data))]
            for k, v in input_map.items():
                for idx in v:
                    for i in range(len(result_list[idx])):
                        if self._is_en:
                            prefix[k].append(' of ' +
                                             result_list[idx][i]['text'])
                        else:
                            prefix[k].append(result_list[idx][i]['text'] + '的')

            for child in node.children:
                child.prefix = prefix
                child.parent_relations = relations
                schema_list.append(child)
        results = self._add_bbox_info(results, data)
        return results

    def _add_bbox_info(self, results, data):
        def _add_bbox(result, char_boxes):
            for vs in result.values():
                for v in vs:
                    if 'start' in v.keys() and 'end' in v.keys():
                        boxes = []
                        for i in range(v['start'], v['end']):
                            cur_box = char_boxes[i][1]
                            if i == v['start']:
                                box = cur_box
                                continue
                            _, cur_y1, cur_x2, cur_y2 = cur_box
                            if cur_y1 == box[1] and cur_y2 == box[3]:
                                box[2] = cur_x2
                            else:
                                boxes.append(box)
                                box = cur_box
                        if box:
                            boxes.append(box)
                        boxes = [[int(b) for b in box] for box in boxes]
                        v['bbox'] = boxes
                    if v.get('relations'):
                        _add_bbox(v['relations'], char_boxes)
            return result

        new_results = []
        for result, one_data in zip(results, data):
            if 'layout' in one_data.keys():
                layout = one_data['layout']
                char_boxes = []
                for segment in layout:
                    sbox = segment[0]
                    text_len = len(segment[1])
                    if text_len == 0:
                        continue
                    if len(segment) == 2 or (len(segment) == 3
                                             and segment[2] != 'table'):
                        char_w = (sbox[2] - sbox[0]) * 1.0 / text_len
                        for i in range(text_len):
                            cbox = [
                                sbox[0] + i * char_w, sbox[1],
                                sbox[0] + (i + 1) * char_w, sbox[3]
                            ]
                            char_boxes.append((segment[1][i], cbox))
                    else:
                        cell_bbox = [(segment[1][i], sbox)
                                     for i in range(text_len)]
                        char_boxes.extend(cell_bbox)

                result = _add_bbox(result, char_boxes)
            new_results.append(result)
        return new_results

    def _convert_ids_to_results(self, examples, sentence_ids, probs):
        """
        Convert ids to raw text in a single stage.
        """
        results = []
        for example, sentence_id, prob in zip(examples, sentence_ids, probs):
            if len(sentence_id) == 0:
                results.append([])
                continue
            result_list = []
            text = example['text']
            prompt = example['prompt']
            for i in range(len(sentence_id)):
                start, end = sentence_id[i]
                if start < 0 and end >= 0:
                    continue
                if end < 0:
                    start += len(prompt) + 1
                    end += len(prompt) + 1
                    result = {
                        'text': prompt[start:end],
                        'probability': prob[i]
                    }
                    result_list.append(result)
                else:
                    result = {
                        'text': text[start:end],
                        'start': start,
                        'end': end,
                        'probability': prob[i]
                    }
                    result_list.append(result)
            results.append(result_list)
        return results

    @classmethod
    def _build_tree(cls, schema, name='root'):
        """
        Build the schema tree.
        """
        schema_tree = SchemaTree(name)
        for s in schema:
            if isinstance(s, str):
                schema_tree.add_child(SchemaTree(s))
            elif isinstance(s, dict):
                for k, v in s.items():
                    if isinstance(v, str):
                        child = [v]
                    elif isinstance(v, list):
                        child = v
                    else:
                        raise TypeError(
                            'Invalid schema, value for each key:value pairs should be list or string'
                            'but {} received'.format(type(v)))
                    schema_tree.add_child(cls._build_tree(child, name=k))
            else:
                raise TypeError(
                    'Invalid schema, element should be string or dict, '
                    'but {} received'.format(type(s)))
        return schema_tree

    def _postprocess(self, inputs):
        """
        This function will convert the model output to raw text.
        """
        return inputs['result']
