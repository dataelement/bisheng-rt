import argparse
import base64
import json
import os
import shutil
import sys
import time

import cv2
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
                        default='general_table_cell_app',
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
        # if not file.startswith('00093'):
        #     continue
        inputs = []
        outputs = []
        image_data = cv2.imread(os.path.join(args.image_folder, file))
        params = {
            # 'debug': True,
            # 'table_det_longer_edge_size': 1120,
            # 'table_rowcol_det_longer_edge_size': 1600
        }
        params_byte = json.dumps(params).encode('utf-8')

        ocr_file = os.path.join(os.path.dirname(args.image_folder),
                                'ocr_results',
                                os.path.splitext(file)[0] + '.json')
        with open(ocr_file, 'r') as f:
            ocr_results = json.load(f)
        ocr_results_byte = json.dumps(ocr_results).encode('utf-8')

        inputs.append(
            tritongrpcclient.InferInput('image', image_data.shape, 'UINT8'))
        inputs.append(tritongrpcclient.InferInput('ocr_result', [1], 'BYTES'))
        inputs.append(tritongrpcclient.InferInput('params', [1], 'BYTES'))
        inputs[0].set_data_from_numpy(image_data)
        inputs[1].set_data_from_numpy(
            np.array([ocr_results_byte]).astype(np.object_))
        inputs[2].set_data_from_numpy(
            np.array([params_byte]).astype(np.object_))

        output_names = ['table_result']
        for output_name in output_names:
            outputs.append(tritongrpcclient.InferRequestedOutput(output_name))

        t0 = time.time()
        results = triton_client.infer(model_name=args.model_name,
                                      inputs=inputs,
                                      outputs=outputs)
        table_result = results.as_numpy(output_names[0])
        table_result = json.loads(table_result[0].decode('utf-8'))
        print(table_result['raw_result'])
        excel_list = table_result['resultFile']
        excel_bytes = np.fromstring(base64.b64decode(excel_list), np.uint8)
        with open(
                os.path.join(res_folder,
                             os.path.splitext(file)[0] + '.xlsx'), 'wb') as f:
            f.write(excel_bytes)
        img_bin = base64.b64decode(table_result['resultImg'])
        with open(os.path.join(res_folder,
                               os.path.splitext(file)[0] + '.png'), 'wb') as f:
            f.write(img_bin)

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
