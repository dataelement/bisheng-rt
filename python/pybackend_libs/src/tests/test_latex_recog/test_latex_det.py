import base64
import time

import cv2
# import numpy as np
from pybackend_libs.dataelem.model.ocr.latex_det import LatexDetection


def test_preprocess():
    kwargs = {
        'model_path': '/public/bisheng/latex_data',
        'devices': ''
    }
    latex_det = LatexDetection(**kwargs)

    input_image = '/public/bisheng/latex_data/zh5.jpg'
    img = cv2.imread(input_image, cv2.IMREAD_COLOR)
    outs = latex_det.preprocess({}, [img])
    assert outs[0].shape
    # np.testing.assert_almost_equal(x, outs[0])


def test_det():
    kwargs = {
        'model_path': (
          '/public/bisheng/model_repository/graphs/latex_det_graph_v1/1'),
        'devices': '0'
    }
    latex_det = LatexDetection(**kwargs)
    image_file = '/public/bisheng/latex_data/zh5.jpg'
    # image_file = '/public/bisheng/latex_data/eng2.png'
    # image_file = '/public/bisheng/latex_data/eng1.png'

    inputs = {
      'b64_image': base64.b64encode(open(image_file, 'rb').read()).decode()
    }
    out1 = latex_det.predict(inputs)
    print('result', out1)

    image_file = '/public/bisheng/latex_data/zh6.jpg'
    # image_file = '/public/bisheng/latex_data/eng2.png'

    inputs = {
      'b64_image': base64.b64encode(open(image_file, 'rb').read()).decode()
    }

    tic = time.time()
    for _ in range(10):
        out1 = latex_det.predict(inputs)
        print('result', out1)
    print('elapse', time.time() - tic)


# test_preprocess()
test_det()
