from .baichuan import BaichuanChat
from .bge import BGEZhEmbedding
from .chatglm2 import ChatGLM2
from .gte import GTEEmbedding
from .llama2 import Llama2Chat
from .me5 import ME5Embedding
from .qwen import QwenChat

__all__ = [
    'ChatGLM2', 'BaichuanChat', 'QwenChat', 'Llama2Chat', 'ME5Embedding',
    'BGEZhEmbedding', 'GTEEmbedding'
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
    }

    return model_name_mapping.get(name, None)
