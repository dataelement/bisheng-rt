import base64

from pybackend_libs.dataelem.model.ocr.latex_recog import LatexRec

# import json


def test_latex_recog_model():
    kwargs = {
        'model_path': '/public/bisheng/model_repository/latex_recog/',
        'devices': ''
    }
    latex_recog = LatexRec(**kwargs)
    image_file = '/public/bisheng/latex_data/formular.jpg'
    inputs = {
      'b64_image': base64.b64encode(open(image_file, 'rb').read()).decode()
    }
    out1 = latex_recog.predict(inputs)
    print(out1)


test_latex_recog_model()
