import asyncio
from functools import partial

import cv2
import numpy as np
import triton_python_backend_utils as pb_utils

from .alg import ALGORITHM_REGISTRY, AlgorithmWithGraph, pb_tensor_to_numpy


# utility functions
def print_mat(arr, name='', ch_split=False):
    def str_(li):
        return '[' + ','.join(map(str, li)) + ']'

    def calc_sum_(arr, ch_split):
        result = []
        if ch_split:
            ch = arr.shape[-1]
            for i in range(ch):
                result.append(np.sum(arr[:, :, i]))
        else:
            result = np.sum(arr)
        return result

    print('<<<<<<<<<<<<<<<<<<<<')
    print('mat.name:', name)
    print('mat::type:', arr.dtype)
    print('mat::sum:', calc_sum_(arr, ch_split))
    print('mat.dims=[', arr.shape, ']')

    arr_flat = arr.flatten()
    n = min(16, len(arr_flat))
    print('mat::data:', str_(arr_flat[:n]))


def resize_image(img,
                 H=32,
                 W_min=40,
                 W_max=1000,
                 leftmargin=2,
                 rightmargin=32,
                 uppermargin=1,
                 lowermargin=1,
                 is_grayscale=False,
                 downsample_rate=8,
                 tail_padding=False,
                 extra_padding_length=88):
    assert W_min % downsample_rate == 0 and W_max % downsample_rate == 0, \
        'W_min and W_max must be mulitple of downsample_rate'
    h, w, c = img.shape
    if is_grayscale:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        c = 1

    h2 = H - uppermargin - lowermargin
    w2 = max(int(w * h2 / h), 1)
    W0 = w2 + leftmargin + rightmargin
    W = int(np.ceil(W0 / downsample_rate) * downsample_rate)
    w2 = w2 + W - W0
    if W <= W_max and W >= W_min:
        img1 = cv2.resize(img, (w2, h2))
        if is_grayscale:
            img1 = np.expand_dims(img1, -1)
        canvas = np.zeros((H, W, c))
        canvas[uppermargin:H - lowermargin,
               leftmargin:W - rightmargin, :] = img1
    elif W < W_min:
        rightmargin = rightmargin + W_min - W
        W = W_min
        img1 = cv2.resize(img, (w2, h2))
        if is_grayscale:
            img1 = np.expand_dims(img1, -1)
        canvas = np.zeros((H, W, c))
        canvas[uppermargin:H - lowermargin,
               leftmargin:W - rightmargin, :] = img1
    else:
        W = W_max
        canvas = np.zeros((H, W, c))
        w2 = W - leftmargin - rightmargin
        h2 = int(w2 / w * h)
        img1 = cv2.resize(img, (w2, h2))
        if is_grayscale:
            img1 = np.expand_dims(img1, -1)
        remainder = (H - h2) % 2
        uppermargin = lowermargin = int((H - h2) / 2)
        lowermargin += remainder
        canvas[uppermargin:H - lowermargin,
               leftmargin:W - rightmargin, :] = img1
    if tail_padding:
        tail = np.zeros((H, extra_padding_length, c))
        canvas = np.concatenate([canvas, tail], axis=1)
    return np.asarray(canvas, 'uint8')


def img_padding(img, W_max):
    h, w, c = img.shape
    tail = np.zeros((h, W_max - w, c))
    img = np.concatenate([img, tail], axis=1)
    return img


def preprocess_recog_batch_v2(images,
                              IMAGE_HEIGHT=32,
                              MIN_WIDTH=40,
                              channels=3,
                              downsample_rate=8,
                              max_img_side=800):
    # batching mode
    # images list of np.array
    assert channels in [1, 3], 'chanels must be 1 or 3. Gray or BGR'
    # todo: extra_padding_length should be associated with the backbone type
    if downsample_rate == 8:
        extra_padding_length = 108
    elif downsample_rate == 4:
        extra_padding_length = 108
    imgs = []
    shapes = []
    W_max = 0
    for img in images:
        img0 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img1 = resize_image(img0,
                            H=IMAGE_HEIGHT,
                            W_min=MIN_WIDTH,
                            W_max=max_img_side,
                            leftmargin=0,
                            rightmargin=0,
                            uppermargin=0,
                            lowermargin=0,
                            is_grayscale=channels == 1,
                            downsample_rate=downsample_rate,
                            tail_padding=True,
                            extra_padding_length=extra_padding_length)
        img1 = img1 / 255.
        imgs.append(img1)
        shapes.append((img1.shape[0], img1.shape[1] - extra_padding_length))
        W_max = max(W_max, img1.shape[1])
    imgs = np.array(list(map(partial(img_padding, W_max=W_max), imgs)),
                    np.float32)
    shapes = np.asarray(shapes, np.int32)
    return imgs, shapes


@ALGORITHM_REGISTRY.register_module()
class OcrTransformer(AlgorithmWithGraph):
    def __init__(self, alg_params, alg_inputs, alg_ouputs, **kwargs):
        super(OcrTransformer, self).__init__(alg_params, alg_inputs,
                                             alg_ouputs, **kwargs)
        self.dep_model_inputs = ['image', 'image_shape']
        self.dep_model_outputs = ['while/Exit_1']

        self.fixed_height = int(alg_params['fixed_height']['string_value']
                                ) if 'fixed_height' in alg_params else 32
        self.batch_size = int(alg_params['batch_size']['string_value']
                              ) if 'batch_size' in alg_params else 32
        self.input_channels = int(alg_params['input_channels']['string_value']
                                  ) if 'input_channels' in alg_params else 1
        self.downsample_rate = int(
            alg_params['downsample_rate']
            ['string_value']) if 'downsample_rate' in alg_params else 8
        self.W_min = int(alg_params['W_min']
                         ['string_value']) if 'W_min' in alg_params else 40
        self.W_max = int(alg_params['W_max']
                         ['string_value']) if 'W_max' in alg_params else 800

    def preprocess(self, context, inputs):
        """OcrTransformer preprocess

        Args:
            context (dict): OcrTransformer global information
            inputs (list[np.array]): [images, images_shape],
                                      inputs of algorithm
        Returns:
            context (dict): OcrTransformer global information
            pre_outputs (list[np.array]): [batch_imgs, batch_imgs_shape],
                                          outputs of preprocess
        """
        images = inputs[0]
        images_shape = inputs[1]
        img_num, _, _, _ = images.shape
        images_no_padding = []
        widths = []
        for index, image in enumerate(images):
            images_no_padding.append(image[:, :images_shape[index, 1], :])
            widths.append([images_shape[index, 1], index])
        # descending
        widths = sorted(widths, key=lambda x: -x[0])

        outputs = []
        outputs_shape = []
        batchs = int(np.ceil(img_num * 1.0 / self.batch_size))
        for k in range(batchs):
            s = k * self.batch_size
            e = img_num if k == batchs - 1 else (k + 1) * self.batch_size

            batch_image = []
            for i in range(s, e):
                batch_image.append(images_no_padding[widths[i][1]])

            batch_image, batch_image_shape = preprocess_recog_batch_v2(
                batch_image,
                IMAGE_HEIGHT=self.fixed_height,
                MIN_WIDTH=self.W_min,
                channels=self.input_channels,
                downsample_rate=self.downsample_rate,
                max_img_side=self.W_max)
            outputs.append(batch_image)
            outputs_shape.append(batch_image_shape)

        pre_outputs = [outputs, outputs_shape]
        context['widths'] = widths
        return context, pre_outputs

    async def infer(self, context, inputs):
        """algorithm infer

        Args:
            context (dict): OcrTransformer global information
            inputs (list[np.array]): [batch_imgs, batch_imgs_shape],
                                      inputs of infer (outputs of preprocess)
        Returns:
            context (dict): OcrTransformer global information
            infer_outputs (list[np.array]): [img_string], outputs of infer
                                             (inputs of postprocess)
        """
        batch_inputs = inputs[0]
        batch_inputs_shape = inputs[1]

        output_texts = []
        inference_response_awaits = []
        for index, batch_input in enumerate(batch_inputs):
            graph_inputs = [batch_input, batch_inputs_shape[index]]
            # graph infer
            input_tensors = []
            for tensor_index, input_ in enumerate(graph_inputs):
                in_tensor = pb_utils.Tensor(
                    self.dep_model_inputs[tensor_index], input_)
                input_tensors.append(in_tensor)
            infer_request = pb_utils.InferenceRequest(
                model_name=self.dep_model_name,
                model_version=self.dep_model_version,
                requested_output_names=self.dep_model_outputs,
                inputs=input_tensors)
            # Store the awaitable inside the array. We don't need
            # the inference response immediately so we do not `await` here.
            inference_response_awaits.append(infer_request.async_exec())

        inference_responses = await asyncio.gather(*inference_response_awaits)
        for infer_response in inference_responses:
            if infer_response.has_error():
                raise pb_utils.TritonModelException(
                    infer_response.error().message())

            graph_outputs = []
            for tensor_index, output_ in enumerate(self.dep_model_outputs):
                pb_tensor = pb_utils.get_output_tensor_by_name(
                    infer_response, output_)
                graph_outputs.append(pb_tensor_to_numpy(pb_tensor))

            output_texts.extend(graph_outputs[0])

        infer_outputs = [np.asarray(output_texts)]
        return context, infer_outputs

    def postprocess(self, context, inputs):
        """OcrTransformer postprocess

        Args:
            context (dict): OcrTransformer global information
            inputs (list[np.array]): [img_string], outputs of infer
        Returns:
            context (dict): OcrTransformer global information
            post_outputs (list[np.array]): [img_string], outputs of algorithm
        """
        texts = inputs[0]
        widths = context['widths']
        reorder_texts = [
            None,
        ] * len(texts)
        for k, v in zip(widths, texts):
            reorder_texts[k[1]] = v

        post_outputs = [np.asarray(reorder_texts)]
        return context, post_outputs


@ALGORITHM_REGISTRY.register_module()
class OcrTransformerTrt(OcrTransformer):
    def __init__(self, alg_params, alg_inputs, alg_ouputs, **kwargs):
        super(OcrTransformerTrt, self).__init__(alg_params, alg_inputs,
                                                alg_ouputs, **kwargs)
        self.dep_model_inputs = ['inputs', 'inputs_shape']
        self.dep_model_outputs = [
            'output_ids', 'parent_ids', 'sequence_length'
        ]
        self.post_model = 'ocr_transformer_trt_post'
        self.post_model_inputs = [
            'output_ids', 'parent_ids', 'sequence_length'
        ]
        self.post_model_outputs = ['while/Exit_1']

        self.fixed_height = int(alg_params['fixed_height']['string_value']
                                ) if 'fixed_height' in alg_params else 32
        self.batch_size = int(alg_params['batch_size']['string_value']
                              ) if 'batch_size' in alg_params else 32
        self.input_channels = int(alg_params['input_channels']['string_value']
                                  ) if 'input_channels' in alg_params else 1
        self.downsample_rate = int(
            alg_params['downsample_rate']
            ['string_value']) if 'downsample_rate' in alg_params else 8
        self.W_min = int(alg_params['W_min']
                         ['string_value']) if 'W_min' in alg_params else 40
        self.W_max = int(alg_params['W_max']
                         ['string_value']) if 'W_max' in alg_params else 800

    async def infer(self, context, inputs):
        """algorithm infer

        Args:
            context (dict): OcrTransformer global information
            inputs (list[np.array]): [batch_imgs, batch_imgs_shape],
                                      inputs of infer (outputs of preprocess)
        Returns:
            context (dict): OcrTransformer global information
            infer_outputs (list[np.array]): [img_string], outputs of infer
                                            (inputs of postprocess)
        """
        batch_inputs = inputs[0]
        batch_inputs_shape = inputs[1]

        output_texts = []
        inference_response_awaits = []
        for index, batch_input in enumerate(batch_inputs):
            # print(batch_input.shape, batch_inputs_shape[index].shape)
            graph_inputs = [batch_input, batch_inputs_shape[index]]
            print_mat(batch_input, 'batch_input')
            print_mat(batch_inputs_shape[index], 'batch_inputs_shape[index]')

            # graph infer
            input_tensors = []
            for tensor_index, input_ in enumerate(graph_inputs):
                in_tensor = pb_utils.Tensor(
                    self.dep_model_inputs[tensor_index], input_)
                input_tensors.append(in_tensor)
            infer_request = pb_utils.InferenceRequest(
                model_name=self.dep_model_name,
                model_version=self.dep_model_version,
                requested_output_names=self.dep_model_outputs,
                inputs=input_tensors)
            # Store the awaitable inside the array. We don't need
            # the inference response immediately so we do not `await` here.
            inference_response_awaits.append(infer_request.async_exec())

        inference_responses = await asyncio.gather(*inference_response_awaits)
        for infer_response in inference_responses:
            if infer_response.has_error():
                raise pb_utils.TritonModelException(
                    infer_response.error().message())

            graph_outputs = []
            for tensor_index, output_ in enumerate(self.dep_model_outputs):
                pb_tensor = pb_utils.get_output_tensor_by_name(
                    infer_response, output_)
                graph_outputs.append(pb_tensor_to_numpy(pb_tensor))

            # trt transformer post
            # print(graph_outputs[0].shape, graph_outputs[1].shape)
            output_ids = pb_utils.Tensor(self.post_model_inputs[0],
                                         graph_outputs[0])
            parent_ids = pb_utils.Tensor(self.post_model_inputs[1],
                                         graph_outputs[1])
            sequence_length = pb_utils.Tensor(self.post_model_inputs[2],
                                              graph_outputs[2])
            post_infer_request = pb_utils.InferenceRequest(
                model_name=self.post_model,
                model_version=-1,
                requested_output_names=self.post_model_outputs,
                inputs=[output_ids, parent_ids, sequence_length])

            post_infer_response = post_infer_request.exec()
            if post_infer_response.has_error():
                raise pb_utils.TritonModelException(
                    post_infer_response.error().message())

            texts_tensor = pb_utils.get_output_tensor_by_name(
                post_infer_response, self.post_model_outputs[0])
            output_texts.extend(pb_tensor_to_numpy(texts_tensor))

        infer_outputs = [np.asarray(output_texts)]
        return context, infer_outputs
