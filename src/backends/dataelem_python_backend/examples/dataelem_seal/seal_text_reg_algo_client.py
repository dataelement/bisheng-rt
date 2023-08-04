import argparse
import copy
import os
import sys
import time

import cv2
import numpy as np
import tritongrpcclient


def convert_to_image_tensor(images, H=32):
    # images: list of np.array
    _, _, channel = images[0].shape
    num_img = len(images)
    shapes = np.array(list(map(lambda x: x.shape[:2], images)))
    widths = np.round(H / shapes[:, 0] * shapes[:, 1]).reshape([num_img, 1])
    heights = np.ones([num_img, 1]) * H
    shapes = np.asarray(np.concatenate([heights, widths], axis=1), np.int32)
    w_max = int(np.max(widths))

    img_canvas = np.zeros([num_img, H, w_max, channel], dtype=np.int32)
    for i, img in enumerate(images):
        h, w = shapes[i]
        img = cv2.resize(img, (w, h))
        img_canvas[i, :, :w, :] = img
    return img_canvas, shapes


def load_files(im_dir):
    names = os.listdir(im_dir)
    return [xx for xx in names if not xx.startswith('.')]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name',
                        type=str,
                        required=False,
                        default='seal_text_reg_algo',
                        help='Model name')
    parser.add_argument('--image_folder',
                        type=str,
                        required=True,
                        help='Path to the image folder')
    parser.add_argument(
        '--url',
        type=str,
        required=False,
        default='localhost:3001',
        help='Inference server URL. Default is localhost:3001.')
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

    reg_res = 'socr_res_32.txt'
    input_names = ['image', 'image_shape']
    output_names = ['while/Exit_1']

    names = load_files(args.image_folder)
    origin_names = copy.deepcopy(names)
    origin_imgs = [
        cv2.imread(os.path.join(args.image_folder, name)) for name in names
    ]

    totalHost = 0
    start_cnt = False
    cnt = 0
    with open(reg_res, 'w') as f:
        while origin_imgs:
            image_names = origin_names[:1000]
            images, images_shape = convert_to_image_tensor(origin_imgs[:1000])
            origin_imgs = origin_imgs[1000:]
            origin_names = origin_names[1000:]

            inputs = []
            outputs = []
            inputs.append(
                tritongrpcclient.InferInput(input_names[0], images.shape,
                                            'FP32'))
            inputs.append(
                tritongrpcclient.InferInput(input_names[1], images_shape.shape,
                                            'INT32'))
            inputs[0].set_data_from_numpy(images.astype(np.float32))
            inputs[1].set_data_from_numpy(images_shape.astype(np.int32))

            for output_name in output_names:
                outputs.append(
                    tritongrpcclient.InferRequestedOutput(output_name))

            t0 = time.time()
            results = triton_client.infer(model_name=args.model_name,
                                          inputs=inputs,
                                          outputs=outputs)
            outputs_ = results.as_numpy(output_names[0])
            outputs_ = list(map(lambda x: x.decode(), outputs_))
            print(outputs_)
            totalHost += time.time() - t0

            for index, name in enumerate(image_names):
                f.write(name + ':' + outputs_[index] + '\n')

    print(f'total:{totalHost}, cnt:{len(names)}, t/per:{totalHost/len(names)}')
