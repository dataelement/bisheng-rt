import argparse
import os
import sys

import cv2
import numpy as np
import tritongrpcclient


def preprocess_recog_batch(images,
                           IMAGE_HEIGHT=32,
                           MIN_WIDTH=40,
                           channels=1,
                           downsample_rate=8,
                           max_img_side=800):
    # batching mode
    # images list of np.array
    assert channels in [1, 3], print('chanels must be 1 or 3. Gray or BGR')
    bs = len(images)
    shapes = np.array(list(map(lambda x: x.shape[:2], images)))
    # widths = np.round(IMAGE_HEIGHT/shapes[:,0]*shapes[:,1]).reshape([bs, 1])
    widths = np.array(
        np.ceil(IMAGE_HEIGHT / shapes[:, 0] * shapes[:, 1] / downsample_rate) *
        downsample_rate,
        dtype=np.int32).reshape([bs, 1])
    widths = np.minimum(widths, max_img_side)
    heights = np.ones([bs, 1]) * IMAGE_HEIGHT
    shapes = np.asarray(np.concatenate([heights, widths], axis=1), np.int32)
    w_max = np.max(widths)
    if w_max < MIN_WIDTH:
        w_max = MIN_WIDTH
    max_im_w = int(w_max + IMAGE_HEIGHT)
    img_canvas = np.zeros([bs, IMAGE_HEIGHT, max_im_w, channels],
                          dtype=np.float32)

    for i, img in enumerate(images):
        if channels == 1:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = shapes[i]
        img = cv2.resize(img, (w, h))
        if channels == 1:
            img = np.expand_dims(img, -1)
        if w < MIN_WIDTH:
            diff = MIN_WIDTH - w
            pad_left = pad_right = int(diff / 2)
            if diff % 2 == 1:
                pad_right += 1
            img = np.pad(img, [(0, 0), (pad_left, pad_right), (0, 0)],
                         'constant',
                         constant_values=255)
            w = MIN_WIDTH
            shapes[i][1] = MIN_WIDTH
        img_canvas[i, :, :w, :] = img / 255
    return img_canvas, shapes


def load_files(im_dir):
    names = os.listdir(im_dir)
    return [xx for xx in names if not xx.startswith('.')]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name',
                        type=str,
                        required=False,
                        default='ocr_transformer_graph',
                        help='Model name')
    parser.add_argument('--image_folder',
                        type=str,
                        required=True,
                        help='Path to the image folder')
    parser.add_argument(
        '--url',
        type=str,
        required=False,
        default='localhost:8101',
        help='Inference server URL. Default is localhost:8001.')
    parser.add_argument('-v',
                        '--verbose',
                        action='store_true',
                        required=False,
                        default=False,
                        help='Enable verbose output')
    args = parser.parse_args()

    try:
        triton_client = tritongrpcclient.InferenceServerClient(
            url=args.url, verbose=args.verbose)
    except Exception as e:
        print('channel creation failed: ' + str(e))
        sys.exit(1)

    input_names = ['image', 'image_shape']
    output_names = ['while/Exit_1']

    names = load_files(args.image_folder)
    # sort image list by ratio
    imgs = [
        cv2.imread(os.path.join(args.image_folder, name)) for name in names
    ]
    imgs_ratio = [img.shape[1] / img.shape[0] for img in imgs]
    idxs = np.argsort(imgs_ratio)
    names = np.array(names)[idxs]

    batch_size = 64
    NB = int(len(names) / batch_size)
    N = len(names) - NB * batch_size
    batch_names = []
    for i in range(NB):
        batch_names.append(names[i * batch_size:i * batch_size + batch_size])
    if N > 0:
        batch_names.append(names[NB * batch_size:])

    cnt = 0
    for batch_name in batch_names:
        im_list = []
        N = len(batch_name)
        for name in batch_name:
            im = cv2.imread(os.path.join(args.image_folder, name))
            im_list.append(im)

        images, images_shape = preprocess_recog_batch(im_list)

        inputs = []
        outputs = []
        inputs.append(
            tritongrpcclient.InferInput(input_names[0], images.shape, 'FP32'))
        inputs.append(
            tritongrpcclient.InferInput(input_names[1], images_shape.shape,
                                        'INT32'))
        inputs[0].set_data_from_numpy(images.astype(np.float32))
        inputs[1].set_data_from_numpy(images_shape.astype(np.int32))

        for output_name in output_names:
            outputs.append(tritongrpcclient.InferRequestedOutput(output_name))

        results = triton_client.infer(model_name=args.model_name,
                                      inputs=inputs,
                                      outputs=outputs)
        outputs_ = results.as_numpy(output_names[0])
        outputs_ = list(map(lambda x: x.decode(), outputs_))
        print(f'batch:{cnt}', outputs_)
