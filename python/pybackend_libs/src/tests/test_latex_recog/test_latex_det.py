import base64

import cv2
import numpy as np
# import onnxruntime
import torch
from pybackend_libs.dataelem.model.ocr.latex_det import LatexDetection


def test():
    with open('/public/bisheng/latex_data/x1.npy', 'rb') as f:
        x = np.load(f)
    x = torch.from_numpy(x)
    model = torch.jit.load('/public/bisheng/latex_data/model.torchscript')
    model.eval()
    with torch.no_grad():
        y = model(x)
        print(np.sum(y.numpy()), y.shape, y.tolist())


def test_preprocess():
    kwargs = {
        'model_path': '/public/bisheng/latex_data',
        'devices': ''
    }
    latex_det = LatexDetection(**kwargs)

    input_image = '/public/bisheng/latex_data/zh5.jpg'
    img = cv2.imread(input_image, cv2.IMREAD_COLOR)
    outs = latex_det.preprocess({}, [img])

    with open('data/x1.npy', 'rb') as f:
        x = np.load(f)

    np.testing.assert_almost_equal(x, outs[0])


def test_all():
    kwargs = {
        'model_path': '/public/bisheng/latex_data',
        'devices': ''
    }
    latex_det = LatexDetection(**kwargs)
    image_file = '/public/bisheng/latex_data/zh5.jpg'
    inputs = {
      'b64_image': base64.b64encode(open(image_file, 'rb').read()).decode()
    }
    out1 = latex_det.predict(inputs)
    print(out1)


# test_preprocess()
# test()
test_all()
