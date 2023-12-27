# flake8: noqa
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from timm.models.layers import StdConv2dSame
from timm.models.resnetv2 import ResNetV2
from timm.models.vision_transformer import VisionTransformer
from timm.models.vision_transformer_hybrid import HybridEmbed
from x_transformers import Decoder, Encoder, TransformerWrapper
from x_transformers.autoregressive_wrapper import (AutoregressiveWrapper,
                                                   top_k, top_p)


class CustomVisionTransformer(VisionTransformer):
    def __init__(self, img_size=224, patch_size=16, *args, **kwargs):
        super(CustomVisionTransformer, self).__init__(
            img_size=img_size, patch_size=patch_size, *args, **kwargs)
        self.height, self.width = img_size
        self.patch_size = patch_size

    def forward_features(self, x):
        B, c, h, w = x.shape
        x = self.patch_embed(x)

        # stole cls_tokens impl from Phil Wang, thanks
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        h, w = h//self.patch_size, w//self.patch_size
        pos_emb_ind = (repeat(
            torch.arange(h)*(self.width//self.patch_size-w), 'h -> (h w)', w=w)
          + torch.arange(h*w))
        pos_emb_ind = torch.cat((torch.zeros(1), pos_emb_ind+1), dim=0).long()
        x += self.pos_embed[:, pos_emb_ind]
        # x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x


class HybridEncoder(object):

    @staticmethod
    def get_encoder(args):
        backbone = ResNetV2(
            layers=args.backbone_layers,
            num_classes=0, global_pool='', in_chans=args.channels,
            preact=False, stem_type='same', conv_layer=StdConv2dSame)
        min_patch_size = 2**(len(args.backbone_layers)+1)

        def embed_layer(**x):
            ps = x.pop('patch_size', min_patch_size)
            assert ps % min_patch_size == 0 and ps >= min_patch_size, (
              'patch_size needs to be multiple of '
              '%i with current backbone configuration' % min_patch_size)
            return HybridEmbed(
                **x, patch_size=ps//min_patch_size, backbone=backbone)

        encoder = CustomVisionTransformer(
            img_size=(args.max_height, args.max_width),
            patch_size=args.patch_size,
            in_chans=args.channels,
            num_classes=0,
            embed_dim=args.dim,
            depth=args.encoder_depth,
            num_heads=args.heads,
            embed_layer=embed_layer)
        return encoder


class ViTransformerWrapper(nn.Module):
    def __init__(
        self,
        *,
        max_width,
        max_height,
        patch_size,
        attn_layers,
        channels=1,
        num_classes=None,
        dropout=0.,
        emb_dropout=0.
    ):
        super().__init__()
        assert isinstance(attn_layers, Encoder), (
            'attention layers must be an Encoder')
        assert max_width % patch_size == 0 and max_height % patch_size == 0, (
            'image dimensions must be divisible by the patch size')
        dim = attn_layers.dim
        num_patches = (max_width // patch_size)*(max_height // patch_size)
        patch_dim = channels * patch_size ** 2

        self.patch_size = patch_size
        self.max_width = max_width
        self.max_height = max_height

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.attn_layers = attn_layers
        self.norm = nn.LayerNorm(dim)
        # self.mlp_head = (
        #  FeedForward(dim, dim_out = num_classes, dropout = dropout)
        #  if exists(num_classes) else None)

    def forward(self, img, **kwargs):
        p = self.patch_size

        x = rearrange(
            img, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p, p2=p)
        x = self.patch_to_embedding(x)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        h, w = torch.tensor(img.shape[2:])//p
        pos_emb_ind = (repeat(
            torch.arange(h)*(self.max_width//p-w), 'h -> (h w)', w=w) +
            torch.arange(h*w))

        pos_emb_ind = torch.cat((torch.zeros(1), pos_emb_ind+1), dim=0).long()
        x += self.pos_embedding[:, pos_emb_ind]
        x = self.dropout(x)

        x = self.attn_layers(x, **kwargs)
        x = self.norm(x)

        return x


class VitEncoder(object):

    @staticmethod
    def get_encoder(args):
        return ViTransformerWrapper(
            max_width=args.max_width,
            max_height=args.max_height,
            channels=args.channels,
            patch_size=args.patch_size,
            emb_dropout=args.get('emb_dropout', 0),
            attn_layers=Encoder(
                dim=args.dim,
                depth=args.encoder_depth,
                heads=args.heads,
            )
        )


class CustomARWrapper(AutoregressiveWrapper):
    def __init__(self, *args, **kwargs):
        super(CustomARWrapper, self).__init__(*args, **kwargs)

    @torch.no_grad()
    def generate(
            self,
            start_tokens, seq_len=256, eos_token=None, temperature=1.,
            filter_logits_fn=top_k, filter_thres=0.9, **kwargs):
        # device = start_tokens.device
        was_training = self.net.training
        num_dims = len(start_tokens.shape)

        if num_dims == 1:
            start_tokens = start_tokens[None, :]

        b, t = start_tokens.shape

        self.net.eval()
        out = start_tokens
        mask = kwargs.pop('mask', None)
        if mask is None:
            mask = torch.full_like(
                out, True, dtype=torch.bool, device=out.device)

        for _ in range(seq_len):
            x = out[:, -self.max_seq_len:]
            mask = mask[:, -self.max_seq_len:]
            # print('arw:',out.shape)
            logits = self.net(x, mask=mask, **kwargs)[:, -1, :]

            if filter_logits_fn in {top_k, top_p}:
                filtered_logits = filter_logits_fn(logits, thres=filter_thres)
                probs = F.softmax(filtered_logits / temperature, dim=-1)

            sample = torch.multinomial(probs, 1)

            out = torch.cat((out, sample), dim=-1)
            mask = F.pad(mask, (0, 1), value=True)

            if eos_token is not None and (
                    torch.cumsum(out == eos_token, 1)[:, -1] >= 1).all():
                break

        out = out[:, t:]

        if num_dims == 1:
            out = out.squeeze(0)

        self.net.train(was_training)
        return out


class TransformerDecoder(object):
    @staticmethod
    def get_decoder(args):
        return CustomARWrapper(
            TransformerWrapper(
                num_tokens=args.num_tokens,
                max_seq_len=args.max_seq_len,
                attn_layers=Decoder(
                    dim=args.dim,
                    depth=args.num_layers,
                    heads=args.heads,
                    **args.decoder_args
                )),
            pad_value=args.pad_token)


class LatexRecogModel(nn.Module):
    """model and implementation from
        https://github.com/lukas-blecher/LaTeX-OCR/tree/main
      post/prep from https://github.com/breezedeus/Pix2Text/tree/main
    """

    def __init__(self, encoder, decoder, args):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.args = args

    def data_parallel(
            self, x: torch.Tensor, device_ids, output_device=None, **kwargs):
        if not device_ids or len(device_ids) == 1:
            return self(x, **kwargs)
        if output_device is None:
            output_device = device_ids[0]

        replicas = nn.parallel.replicate(self, device_ids)
        # Slices tensors into approximately equal chunks and distributes them
        # across given GPUs.
        inputs = nn.parallel.scatter(x, device_ids)

        # Duplicates references to objects that are not tensors.
        kwargs = nn.parallel.scatter(kwargs, device_ids)
        replicas = replicas[:len(inputs)]
        kwargs = kwargs[:len(inputs)]
        outputs = nn.parallel.parallel_apply(replicas, inputs, kwargs)
        return nn.parallel.gather(outputs, output_device).mean()

    def forward(self, x: torch.Tensor, tgt_seq: torch.Tensor,  **kwargs):
        encoded = self.encoder(x)
        out = self.decoder(tgt_seq, context=encoded, **kwargs)
        return out

    @torch.no_grad()
    def generate(self, x: torch.Tensor, temperature: float = 0.25):
        return self.decoder.generate(
            (torch.LongTensor([self.args.bos_token] * len(x))[:, None]).to(
                x.device),
            self.args.max_seq_len,
            eos_token=self.args.eos_token,
            context=self.encoder(x),
            temperature=temperature)

    @classmethod
    def create_model(cls, args):
        if args.encoder_structure.lower() == 'vit':
            encoder = VitEncoder.get_encoder(args)
        elif args.encoder_structure.lower() == 'hybrid':
            encoder = HybridEncoder.get_encoder(args)
        else:
            raise NotImplementedError(
                'Encoder structure '
                '"%s" not supported.' % args.encoder_structure)
        decoder = TransformerDecoder.get_decoder(args)
        encoder.to(args.device)
        decoder.to(args.device)
        model = LatexRecogModel(encoder, decoder, args)
        return model
