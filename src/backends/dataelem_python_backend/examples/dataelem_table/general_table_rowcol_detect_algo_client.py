import argparse
import json
import os
import shutil
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
                        default='general_table_rowcol_detect_algo',
                        help='Model name')
    parser.add_argument('--image_folder',
                        type=str,
                        required=True,
                        help='Path to the image')
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

    res_folder = 'res'
    if not os.path.exists(res_folder):
        os.makedirs(res_folder)
    else:
        shutil.rmtree(res_folder)
        os.makedirs(res_folder)

    totalHost = 0
    start_cnt = False
    cnt = 0
    # image_data = load_image(args.image)
    image_files = os.listdir(args.image_folder)
    for file in image_files:
        inputs = []
        outputs = []

        image_data = cv2.imread(os.path.join(args.image_folder, file))
        params = {'longer_edge_size': 1600}
        params_byte = json.dumps(params).encode('utf-8')

        inputs.append(
            tritongrpcclient.InferInput('image', image_data.shape, 'FP32'))
        inputs.append(tritongrpcclient.InferInput('params', [1], 'BYTES'))
        inputs[0].set_data_from_numpy(image_data.astype(np.float32))
        inputs[1].set_data_from_numpy(
            np.array([params_byte]).astype(np.object_))

        output_names = ['boxes', 'scores', 'labels']
        for output_name in output_names:
            outputs.append(tritongrpcclient.InferRequestedOutput(output_name))

        t0 = time.time()
        results = triton_client.infer(model_name=args.model_name,
                                      inputs=inputs,
                                      outputs=outputs)
        boxes = results.as_numpy(output_names[0])
        scores = results.as_numpy(output_names[1])
        labels = results.as_numpy(output_names[2])
        print('boxes:', boxes)
        print('scores:', scores)
        print('labels:', labels)
        totalHost += time.time() - t0
        cnt += 1
        if (cnt == 100) and (not start_cnt):
            start_cnt = True
            cnt = 0
            totalHost = 0
        if start_cnt and cnt > 0 and cnt % 50 == 0:
            print(
                f'totalHost:{totalHost}, cnt:{cnt}, t/per_img:{totalHost/cnt}')

        # with open(os.path.join(
        #     res_folder, os.path.splitext(file)[0] + '.txt'), 'w') as f:
        #     for box in boxes:
        #         bbox = box.reshape(-1)
        #         bbox = [str(i) for i in bbox]
        #         line = ','.join(bbox)
        #         f.write(line + '\n')

    print(f'totalHost:{totalHost}, cnt:{cnt}, t/per_img:{totalHost/cnt}')
