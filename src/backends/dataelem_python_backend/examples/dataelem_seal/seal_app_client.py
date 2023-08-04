import argparse
import base64
import json
import os
import shutil
import sys
import time

import numpy as np
import tritongrpcclient


def convert_b64(file):
    if os.path.isfile(file):
        with open(file, 'rb') as fh:
            x = base64.b64encode(fh.read())
            return x
    else:
        return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name',
                        type=str,
                        required=False,
                        default='seal_app',
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
        image_b64 = convert_b64(os.path.join(args.image_folder, file))
        params = {
            # 'debug': True,
            # 'image_name': file,
            # 'seal_det_longer_edge_size': 1600,
            # 'seal_curve_text_det_longer_edge_size': 640,
            # 'seal_quad_text_det_longer_edge_size': 256,
        }
        params_byte = json.dumps(params).encode('utf-8')

        inputs.append(tritongrpcclient.InferInput('image_b64', [1], 'BYTES'))
        inputs.append(tritongrpcclient.InferInput('params', [1], 'BYTES'))
        inputs[0].set_data_from_numpy(np.array([image_b64]).astype(np.object_))
        inputs[1].set_data_from_numpy(
            np.array([params_byte]).astype(np.object_))

        output_names = ['seal_results']
        for output_name in output_names:
            outputs.append(tritongrpcclient.InferRequestedOutput(output_name))

        t0 = time.time()
        results = triton_client.infer(model_name=args.model_name,
                                      inputs=inputs,
                                      outputs=outputs)
        seal_results = results.as_numpy(output_names[0])
        seal_results = json.loads(seal_results[0].decode('utf-8'))
        print(seal_results)
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
