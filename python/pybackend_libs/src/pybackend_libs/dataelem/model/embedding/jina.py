import torch
import torch.nn.functional as F
import math
import os
import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import numpy as np
from transformers.utils import (logging)
from transformers import AutoTokenizer

from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    NextSentencePredictorOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)


from .embedding import BaseEmbedding, EmbResponse, torch_gc
# from .configuration_bert import JinaBertConfig
from .jina_util import JinaBertEmbeddings, JinaBertEncoder, JinaBertPooler, JinaBertPreTrainedModel, JinaBertConfig


logger = logging.get_logger(__name__)

class JINAEmbedding(JinaBertPreTrainedModel):
    def __init__(self, config=JinaBertConfig(),add_pooling_layer=True, **kwargs):
        super().__init__(config)
        # config = 
        self.config = JinaBertConfig()
        print("----- self.config: ", self.config)
        self.emb_pooler = config.emb_pooler
        self._name_or_path = kwargs.get('pretrain_path')
        # if self.emb_pooler:
        #     from transformers import AutoTokenizer

        pretrain_path = kwargs.get('pretrain_path')
        precision = kwargs.get('precision', 'fp16')
        gpu_memory = kwargs.get('gpu_memory')
        devices = kwargs.get('devices').split(',')
        self.batch_size = int(kwargs.get('batch_size', '32'))
        self.devices = devices
        self.default_device = f'cuda:{devices[0]}'
        # self._load(pretrain_path, precision, devices, gpu_memory,jina_mode=True)

        self.tokenizer = AutoTokenizer.from_pretrained(self._name_or_path)
        # self.eval
        # print("===")
        self.embeddings = JinaBertEmbeddings(config)
        # print("=== self.embeddings: ",self.embeddings)
        self.encoder = JinaBertEncoder(config)
        self.pooler = JinaBertPooler(config) if add_pooling_layer else None
        self.post_init()

    def mean_pooling(
        self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor
    ):
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        # print("--- output_attentions: ",output_attentions)
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        # print("--- output_hidden_states: ",output_hidden_states)
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        # print("=== return_dict:",return_dict)

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False
        # print("=== use_cache:",use_cache)
        # input_ids = input_ids.to(self.default_device)
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            # self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")
        # print("=== input_shape:",input_shape)

        batch_size, seq_length = input_shape
        # device = input_ids.device if input_ids is not None else inputs_embeds.device
        device = self.default_device

        # past_key_values_length
        past_key_values_length = (
            past_key_values[0][0].shape[2] if past_key_values is not None else 0
        )
        # print("=== past_key_values_length:",past_key_values_length)

        if attention_mask is None:
            attention_mask = torch.ones(
                ((batch_size, seq_length + past_key_values_length)), device=device
            )
        # print("=== attention_mask:",attention_mask)
        # print("token_type_ids:",token_type_ids)
        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(
                    batch_size, seq_length
                )
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(
                    input_shape, dtype=torch.long, device=device
                )
        # print("token_type_ids:",token_type_ids)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(
            attention_mask, input_shape
        )
        # print("==== extended_attention_mask:",extended_attention_mask)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            (
                encoder_batch_size,
                encoder_sequence_length,
                _,
            ) = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(
                encoder_attention_mask
            )
        else:
            encoder_extended_attention_mask = None
        # print("=== encoder_extended_attention_mask:",encoder_extended_attention_mask)

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
        # print("=== head_mask:",head_mask)
        """
        if input_ids is not None:
            input_ids = input_ids.to(self.default_device)
        if position_ids is not None:
            position_ids = position_ids.to(self.default_device)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(self.default_device)
        if inputs_embeds is not None:
            inputs_embeds = inputs_embeds.to(self.default_device)
        """
        # print("======input_ids: ",input_ids)
        # print("======position_ids: ",position_ids)
        # print("======token_type_ids: ",token_type_ids)
        print("======inputs_embeds: ",inputs_embeds)
        # print("======past_key_values_length: ",past_key_values_length)
        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )
        print("=== embedding_output:",embedding_output)
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = (
            self.pooler(sequence_output) if self.pooler is not None else None
        )

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )

    @torch.inference_mode()
    def encode(
        self: 'JINAEmbedding',
        sentences: Union[str, List[str]],
        batch_size: int = 32,
        show_progress_bar: Optional[bool] = None,
        output_value: str = 'sentence_embedding',
        convert_to_numpy: bool = True,
        convert_to_tensor: bool = False,
        device: Optional[torch.device] = None,
        normalize_embeddings: bool = False,
        **tokenizer_kwargs,
    ) -> Union[List[torch.Tensor], np.ndarray, torch.Tensor]:

        if not self.emb_pooler:
            warnings.warn("No emb_pooler specified, defaulting to mean pooling.")
            self.emb_pooler = 'mean'
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(self._name_or_path)
        is_training = self.training
        self.eval()

        if convert_to_tensor:
            convert_to_numpy = False

        if output_value != 'sentence_embedding':
            convert_to_tensor = False
            convert_to_numpy = False

        input_was_string = False
        if isinstance(sentences, str) or not hasattr(sentences, '__len__'):
            sentences = [sentences]
            input_was_string = True

        if device is not None:
            self.to(device)

        # print("--- sentences: ", sentences)
        # TODO: Maybe use better length heuristic?
        permutation = np.argsort([-len(i) for i in sentences])
        inverse_permutation = np.argsort(permutation)
        sentences = [sentences[idx] for idx in permutation]
        print("--- sentences: ", sentences)

        tokenizer_kwargs['padding'] = tokenizer_kwargs.get('padding', True)
        tokenizer_kwargs['max_length'] = tokenizer_kwargs.get('max_length', 8192)
        tokenizer_kwargs['truncation'] = tokenizer_kwargs.get('truncation', True)
        print("--- tokenizer_kwargs: ",tokenizer_kwargs)

        all_embeddings = []


        range_iter = range(0, len(sentences), batch_size)

        for i in range_iter:
            encoded_input = self.tokenizer(
                sentences[i : i + batch_size],
                return_tensors='pt',
                **tokenizer_kwargs,
            ).to(self.device)
            # print("sentences[i : i + batch_size]: ",sentences[i : i + batch_size])
            # encoded_input.update({"token_type_ids": torch.zeros(
            #         len(sentences[i : i + batch_size]), dtype=torch.long, device=device)})
            # print("--- encoded_input: ",encoded_input)
            token_embs = self.forward(**encoded_input)[0]
            print("--- token_embs:",token_embs)

            # Accumulate in fp32 to avoid overflow
            token_embs = token_embs.float()

            if output_value == 'token_embeddings':
                raise NotImplementedError
            elif output_value is None:
                raise NotImplementedError
            else:
                embeddings = self.mean_pooling(
                    token_embs, encoded_input['attention_mask']
                )

                if normalize_embeddings:
                    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

                if convert_to_numpy:
                    embeddings = embeddings.cpu()
            all_embeddings.extend(embeddings)

        all_embeddings = [all_embeddings[idx] for idx in inverse_permutation]

        if convert_to_tensor:
            all_embeddings = torch.stack(all_embeddings)
        elif convert_to_numpy:
            all_embeddings = np.asarray([emb.numpy() for emb in all_embeddings])

        if input_was_string:
            all_embeddings = all_embeddings[0]

        self.train(is_training)
        return all_embeddings

    def predict(self, kwargs):
        model = kwargs.get('model')
        sentences = kwargs.get('texts')

        embs = self.encode(sentences,
                    self.batch_size,
                    None,
                    'sentence_embedding',
                    True,
                    False,
                    self.default_device,
                    False).tolist()



        # permutation = np.argsort([-len(i) for i in sentences])
        # inverse_permutation = np.argsort(permutation)
        # sentences = [sentences[idx] for idx in permutation]

        # all_embeddings = []
        # range_iter = range(0, len(sentences), self.batch_size)
        # for i in range_iter:
        #     encoded_input = self.tokenizer(
        #         sentences[i : i + self.batch_size],
        #         return_tensors='pt',
        #         max_length=512,
        #         padding=True,
        #         truncation=True,
        #     )#.to(self.default_device)
        #     token_embs = self.forward(**encoded_input)[0]

        #     # Accumulate in fp32 to avoid overflow
        #     token_embs = token_embs.float()

        #     # if output_value == 'token_embeddings':
        #     #     raise NotImplementedError
        #     # elif output_value is None:
        #     #     raise NotImplementedError
        #     if 0:
        #         pass
        #     else:
        #         embeddings = mean_pooling(
        #             token_embs, encoded_input['attention_mask']
        #         )

        #         # if normalize_embeddings:
        #         embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        #         # if convert_to_numpy:
        #         embeddings = embeddings.cpu()
        #         print("---embeddings:",embeddings)
        #     all_embeddings.extend(embeddings)
        # all_embeddings = [all_embeddings[idx] for idx in inverse_permutation]
        # print("--- all_embeddings[0]: ",type(all_embeddings[0].tolist()))
        # embs = [emb.tolist() for emb in all_embeddings]

        # def infer_handler(input_texts):
        #     # Tokenize the input texts
        #     batch_dict = self.tokenizer(input_texts,
        #                                 max_length=512,
        #                                 padding=True,
        #                                 truncation=True,
        #                                 return_tensors='pt')

        #     input_ids = batch_dict['input_ids'].to(self.default_device)
        #     attention_mask = batch_dict['attention_mask'].to(
        #         self.default_device)
            
        #     output_attentions = attention_mask

        #     with torch.no_grad():
        #         outputs = self.model(input_ids=input_ids,
        #                              attention_mask=attention_mask)
                
        #     output_hidden_states = outputs.last_hidden_state

        #     embeddings = mean_pooling(
        #         outputs.last_hidden_state, attention_mask)
        #     embeddings = F.normalize(embeddings, p=2, dim=1).cpu().numpy()
        #     return embeddings.tolist()

        # embs = self._batch_predict(self.batch_size, sentences, infer_handler)
        torch_gc(self.devices)
        return EmbResponse(model=model, embeddings=embs).dict()


