import torch
import torch.nn.functional as F

from .embedding import BaseEmbedding, EmbResponse, mean_pooling, torch_gc


class JINAEmbedding(BaseEmbedding):
    def __init__(self, **kwargs):
        pretrain_path = kwargs.get('pretrain_path')
        precision = kwargs.get('precision', 'fp16')
        gpu_memory = kwargs.get('gpu_memory')
        devices = kwargs.get('devices').split(',')
        self.batch_size = int(kwargs.get('batch_size', '32'))
        self.devices = devices
        self.default_device = f'cuda:{devices[0]}'
        self._load(pretrain_path, precision, devices, gpu_memory)

    def predict(self, kwargs):
        model = kwargs.get('model')
        input_texts = kwargs.get('texts')

        def infer_handler(input_texts):
            # Tokenize the input texts
            batch_dict = self.tokenizer(input_texts,
                                        max_length=512,
                                        padding=True,
                                        truncation=True,
                                        return_tensors='pt')

            input_ids = batch_dict['input_ids'].to(self.default_device)
            attention_mask = batch_dict['attention_mask'].to(
                self.default_device)
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids,
                                     attention_mask=attention_mask)

            embeddings = mean_pooling(
                outputs.last_hidden_state, attention_mask)
            embeddings = F.normalize(embeddings, p=2, dim=1).cpu().numpy()
            return embeddings.tolist()

        embs = self._batch_predict(self.batch_size, input_texts, infer_handler)
        torch_gc(self.devices)
        return EmbResponse(model=model, embeddings=embs).dict()
