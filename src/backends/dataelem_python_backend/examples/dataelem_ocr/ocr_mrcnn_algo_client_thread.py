import argparse
import concurrent.futures
import json
import os
import shutil
import sys
import time

import cv2
import numpy as np
import tqdm
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
                        default='ocr_mrcnn_algo',
                        help='Model name')
    parser.add_argument('--image_folder',
                        type=str,
                        required=True,
                        help='Path to the image')
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

    res_folder = 'res'
    if not os.path.exists(res_folder):
        os.makedirs(res_folder)
    else:
        shutil.rmtree(res_folder)
        os.makedirs(res_folder)

    # image_data = load_image(args.image)
    image_files = os.listdir(args.image_folder)
    images = [
        cv2.imread(os.path.join(args.image_folder, file))
        for file in image_files
    ]

    def predict(img):
        try:
            triton_client = tritongrpcclient.InferenceServerClient(
                url=args.url, verbose=args.verbose)
        except Exception as e:
            print('channel creation failed: ' + str(e))
            sys.exit(1)

        inputs = []
        outputs = []

        image_data = img
        params = {'longer_edge_size': 1600}
        params_byte = json.dumps(params).encode('utf-8')

        inputs.append(
            tritongrpcclient.InferInput('image', image_data.shape, 'FP32'))
        inputs.append(tritongrpcclient.InferInput('params', [1], 'BYTES'))
        inputs[0].set_data_from_numpy(image_data.astype(np.float32))
        inputs[1].set_data_from_numpy(
            np.array([params_byte]).astype(np.object_))

        output_names = ['boxes', 'box_scores']
        for output_name in output_names:
            outputs.append(tritongrpcclient.InferRequestedOutput(output_name))

        results = triton_client.infer(model_name=args.model_name,
                                      inputs=inputs,
                                      outputs=outputs)
        # boxes = results.as_numpy(output_names[0])
        box_scores = results.as_numpy(output_names[1])

        return box_scores

    rets = []
    t0 = time.time()
    # 多线程
    with tqdm.tqdm(total=len(image_files)) as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures_data = [executor.submit(predict, img) for img in images]
            for future in concurrent.futures.as_completed(futures_data):
                rets.append(future.result())
                pbar.update(1)

    # # 多进程
    # pool = multiprocessing.Pool(processes=10)
    # pbar = tqdm.tqdm(total=len(image_files))
    # multi_res = []
    # for img in images:
    #     # 非阻塞调用，pool同时处理10个任务，哪个任务结束了，该进程就执行下个任务
    #     multi_res.append(
    #         pool.apply_async(
    #             predict, (img, ),
    #             callback=lambda *a: pbar.update(1)))
    # pool.close()
    # # 在当前位置阻塞主进程，等待子进程结束
    # pool.join()
    # for i in range(len(img)):
    #     rets.append(multi_res[i].get())

    t1 = time.time()
    print('time', (t1 - t0) / len(image_files))
    print(rets[0])
