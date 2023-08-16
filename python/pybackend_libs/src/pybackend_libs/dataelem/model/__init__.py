from .embedding.bge import BGEZhEmbedding
from .embedding.gte import GTEEmbedding
from .embedding.me5 import ME5Embedding
from .layout.layout_mrcnn import LayoutMrcnn
from .llm.baichuan import BaichuanChat
from .llm.chatglm2 import ChatGLM2
from .llm.llama2 import Llama2Chat
from .llm.qwen import QwenChat

__all__ = [
    'ChatGLM2', 'BaichuanChat', 'QwenChat', 'Llama2Chat', 'ME5Embedding',
    'BGEZhEmbedding', 'GTEEmbedding', 'LayoutMrcnn'
]


def get_model(name: str):
    model_name_mapping = {
        'ChatGLM2': ChatGLM2,
        'BaichuanChat': BaichuanChat,
        'QwenChat': QwenChat,
        'Llama2Chat': Llama2Chat,
        'ME5Embedding': ME5Embedding,
        'BGEZhEmbedding': BGEZhEmbedding,
        'GTEEmbedding': GTEEmbedding,
        'LayoutMrcnn': LayoutMrcnn
    }

    return model_name_mapping.get(name, None)
