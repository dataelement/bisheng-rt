import argparse
import os
import sys
import time

import cv2
import numpy as np
import tritongrpcclient


def load_image(img_path: str):
    """
    Loads an encoded image as an array of bytes.

    """
    return np.fromfile(img_path, dtype='uint8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name',
                        type=str,
                        required=False,
                        default='ocr_mrcnn_trt_graph',
                        help='Model name')
    parser.add_argument('--image_folder',
                        type=str,
                        required=True,
                        help='Path to the image_folder')
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

    longer_edge_size = 1600
    totalHost = 0
    start_cnt = False
    cnt = 0
    image_files = os.listdir(args.image_folder)
    for file in image_files:
        inputs = []
        outputs = []

        img = cv2.imread(os.path.join(args.image_folder, file))
        orig_shape = img.shape[:2]
        h = orig_shape[0]
        w = orig_shape[1]

        scale = longer_edge_size * 1.0 / max(h, w)
        if h > w:
            newh, neww = longer_edge_size, scale * w
        else:
            newh, neww = scale * h, longer_edge_size

        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        resized_img = cv2.resize(img, dsize=(neww, newh))
        resized_h, resized_w, channel = resized_img.shape

        pad_img = np.zeros([1, longer_edge_size, longer_edge_size, channel],
                           dtype=np.int32)
        pad_img[0, :resized_h, :resized_w, :] = resized_img

        inputs.append(
            tritongrpcclient.InferInput('image', pad_img.shape, 'FP32'))
        inputs[0].set_data_from_numpy(pad_img.astype(np.float32))

        output_names = ['output_detections', 'output_masks']
        for output_name in output_names:
            outputs.append(tritongrpcclient.InferRequestedOutput(output_name))

        t0 = time.time()
        results = triton_client.infer(model_name=args.model_name,
                                      inputs=inputs,
                                      outputs=outputs)
        output_detections = results.as_numpy(output_names[0])
        # print(output_detections, output_detections.shape)
        output_masks = results.as_numpy(output_names[1])
        # print(output_masks, output_masks.shape)
        totalHost += time.time() - t0
        cnt += 1
        if (cnt == 100) and (not start_cnt):
            start_cnt = True
            cnt = 0
            totalHost = 0
        if start_cnt and cnt > 0 and cnt % 50 == 0:
            print(
                f'totalHost:{totalHost}, cnt:{cnt}, t/per_img:{totalHost/cnt}')

    print(f'totalHost:{totalHost}, cnt:{cnt}, t/per_img:{totalHost/cnt}')
