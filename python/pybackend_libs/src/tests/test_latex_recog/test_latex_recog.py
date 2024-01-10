import base64

from pybackend_libs.dataelem.model.ocr.latex_recog import LatexRecog


def test_export_onnx():
    kwargs = {
        'model_path': '/public/bisheng/model_repository/latex_recog/',
        'devices': ''
    }
    latex_recog = LatexRecog(**kwargs)
    image_file = '/public/bisheng/latex_data/formula.jpg'
    # save_path = '/public/bisheng/model_repository/latex_recog/model.onnx'
    save_path = '/public/bisheng/model_repository/latex_recog/model.pt'
    inputs = {
      'b64_image': base64.b64encode(open(image_file, 'rb').read()).decode()
    }
    latex_recog.export_onnx(inputs, save_path)

    save_path = (
     '/public/bisheng/model_repository/latex_recog/pytorch_model.bin')
    latex_recog.export_onnx(inputs, save_path, True)


def test_latex_recog_model():
    kwargs = {
        'model_path': '/public/bisheng/model_repository/latex_recog/',
        'devices': '',
        'use_onnx_encoder': False,
    }
    latex_recog = LatexRecog(**kwargs)
    image_files = [
        '/public/bisheng/latex_data/formula.jpg',
        '/public/bisheng/latex_data/formula2.jpg',
        '/public/bisheng/latex_data/formula3.jpg',
    ]

    for image_file in image_files:
        inputs = {
          'b64_image': base64.b64encode(open(image_file, 'rb').read()).decode()
        }
        out1 = latex_recog.predict(inputs)
        print(out1)


def test_safe_recog_model():
    kwargs = {
        'model_path': '/public/bisheng/model_repository/latex_recog/',
        'devices': '',
        'use_onnx_encoder': True,
    }
    latex_recog = LatexRecog(**kwargs)
    image_files = [
        '/public/bisheng/latex_data/formula.jpg',
        '/public/bisheng/latex_data/formula2.jpg',
        '/public/bisheng/latex_data/formula3.jpg',
    ]

    for image_file in image_files:
        inputs = {
          'b64_image': base64.b64encode(open(image_file, 'rb').read()).decode()
        }
        out1 = latex_recog.predict(inputs)
        print(out1)


# test_export_onnx()
test_safe_recog_model()
# test_latex_recog_model()
