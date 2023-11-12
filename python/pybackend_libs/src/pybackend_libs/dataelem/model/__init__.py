from pybackend_libs.dataelem.utils.lazy_loader import lazy

bge = lazy('pybackend_libs.dataelem.model.embedding.bge')
gte = lazy('pybackend_libs.dataelem.model.embedding.gte')
me5 = lazy('pybackend_libs.dataelem.model.embedding.me5')
baichuan = lazy('pybackend_libs.dataelem.model.llm.baichuan')
chatglm2 = lazy('pybackend_libs.dataelem.model.llm.chatglm2')
code_geex2 = lazy('pybackend_libs.dataelem.model.llm.code_geex2')
internlm = lazy('pybackend_libs.dataelem.model.llm.internlm')
llama2 = lazy('pybackend_libs.dataelem.model.llm.llama2')
qwen = lazy('pybackend_libs.dataelem.model.llm.qwen')
xverse = lazy('pybackend_libs.dataelem.model.llm.xverse')
visualglm = lazy('pybackend_libs.dataelem.model.mmu.visualglm')
vllm_model = lazy('pybackend_libs.dataelem.model.vllm.vllm_model')
layout_mrcnn = lazy('pybackend_libs.dataelem.model.layout.layout_mrcnn')
table_mrcnn = lazy('pybackend_libs.dataelem.model.table.table_mrcnn')
table_app = lazy('pybackend_libs.dataelem.model.table.table_app')


def get_model(name: str):
    model_name_mapping = {
        'ChatGLM2': chatglm2,
        'BaichuanChat': baichuan,
        'QwenChat': qwen,
        'Llama2Chat': llama2,
        'VisualGLM': visualglm,
        'XverseChat': xverse,
        'InternLM': internlm,
        'ME5Embedding': me5,
        'BGEZhEmbedding': bge,
        'GTEEmbedding': gte,
        'LayoutMrcnn': layout_mrcnn,
        'TableCellApp': table_app,
        'TableRowColApp': table_app,
        'MrcnnTableDetect': table_mrcnn,
        'VLLMModel': vllm_model,
    }
    assert name in model_name_mapping, f'Unknown model name: {name}'
    return getattr(model_name_mapping[name], name)
