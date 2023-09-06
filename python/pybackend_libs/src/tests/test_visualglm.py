from pybackend_libs.dataelem.model.mmu.visualglm import VisualGLM
from pybackend_libs.dataelem.utils import convert_file_to_base64


def test_visualglm_6b():
    params = {
        'pretrain_path': '/home/public/models/visualglm-6b',
        'devices': '6',
        'gpu_memory': 20,
    }

    model = VisualGLM(**params)
    image_path = '../data/maoxuan_mulu.jpg'
    b64_image = convert_file_to_base64(image_path)

    inp = {'b64_image': b64_image}
    outp = model.predict(inp)
    print(outp)


test_visualglm_6b()
