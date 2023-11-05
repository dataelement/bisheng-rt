from pybackend_libs.dataelem.model import get_model


def test1():
    cls_type = get_model('MrcnnTableDetect')
    assert cls_type is not None
    cls_type = get_model('LayoutMrcnn')
    assert cls_type is not None
    cls_type = get_model('Llama2Chat')
    assert cls_type is not None
    cls_type = get_model('ME5Embedding')
    assert cls_type is not None
    cls_type = get_model('VisualGLM')
    assert cls_type is not None


test1()
