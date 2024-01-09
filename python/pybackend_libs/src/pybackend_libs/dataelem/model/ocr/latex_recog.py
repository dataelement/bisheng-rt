# flake8: noqa
import base64
import io
import json
import os
import re
from typing import Any, Dict, Optional, Tuple

import albumentations as alb
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from einops import rearrange, repeat
from PIL import Image, ImageOps
from timm.models.layers import StdConv2dSame
from timm.models.resnetv2 import ResNetV2
from timm.models.vision_transformer import VisionTransformer
from timm.models.vision_transformer_hybrid import HybridEmbed
from transformers import PreTrainedTokenizerFast
from x_transformers import Decoder, Encoder, TransformerWrapper
from x_transformers.autoregressive_wrapper import (AutoregressiveWrapper,
                                                   top_k, top_p)

# The Implementaion are from
#  main framework: https://github.com/lukas-blecher/LaTeX-OCR/tree/main
#  extra post: https://github.com/breezedeus/Pix2Text/tree/main

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
        layers = args['backbone_layers']
        channels = args['channels']
        max_height = args['max_height']
        max_width = args['max_width']
        patch_size = args['patch_size']
        dim = args['dim']
        encoder_depth = args['encoder_depth']
        heads = args['heads']

        backbone = ResNetV2(
            layers=layers,
            num_classes=0, global_pool='', in_chans=channels,
            preact=False, stem_type='same', conv_layer=StdConv2dSame)
        min_patch_size = 2**(len(layers) + 1)

        def embed_layer(**x):
            ps = x.pop('patch_size', min_patch_size)
            assert ps % min_patch_size == 0 and ps >= min_patch_size, (
              'patch_size needs to be multiple of '
              '%i with current backbone configuration' % min_patch_size)
            return HybridEmbed(
                **x, patch_size=ps//min_patch_size, backbone=backbone)

        encoder = CustomVisionTransformer(
            img_size=(max_height, max_width),
            patch_size=patch_size,
            in_chans=channels,
            num_classes=0,
            embed_dim=dim,
            depth=encoder_depth,
            num_heads=heads,
            embed_layer=embed_layer)
        return encoder


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
            # print('arw:', out.shape)
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
        num_tokens = args['num_tokens']
        max_seq_len = args['max_seq_len']
        dim = args['dim']
        num_layers = args['num_layers']
        heads = args['heads']
        decoder_args = args['decoder_args']
        pad_token = args['pad_token']

        return CustomARWrapper(
            TransformerWrapper(
                num_tokens=num_tokens,
                max_seq_len=max_seq_len,
                attn_layers=Decoder(
                    dim=dim,
                    depth=num_layers,
                    heads=heads,
                    **decoder_args
                )),
            pad_value=pad_token)


class EncoderDecoder(nn.Module):
    """model and implementation from
        https://github.com/lukas-blecher/LaTeX-OCR/tree/main

      post/prep from https://github.com/breezedeus/Pix2Text/tree/main
    """

    def __init__(self, **args):
        super().__init__()
        encoder_structure = args['encoder_structure']
        device = args['device']
        self.bos_token = args['bos_token']
        self.eos_token = args['eos_token']
        self.max_seq_len = args['max_seq_len']
        if encoder_structure.lower() == 'vit':
            raise Exception('vit encoder not supported')
        elif encoder_structure.lower() == 'hybrid':
            self.encoder = HybridEncoder.get_encoder(args)
        else:
            raise NotImplementedError(
                'Encoder structure '
                '"%s" not supported.' % encoder_structure)


        self.decoder = TransformerDecoder.get_decoder(args)
        self.encoder.to(device)
        self.decoder.to(device)
        self.encoder.eval()
        self.decoder.eval()

         # Compatibility updates
        for m in self.encoder.modules():
            print('---m', type(m))

        print('---enocder', dir(self.encoder), self.encoder.training)

    def forward(self, x: torch.Tensor, tgt_seq: torch.Tensor,  **kwargs):
        encoded = self.encoder(x)
        out = self.decoder(tgt_seq, context=encoded, **kwargs)
        return out

    @torch.no_grad()
    def generate(self, x: torch.Tensor, temperature: float = 0.25):
        start_tokens = (
            torch.LongTensor([self.bos_token])[:, None]).to(x.device)
        print('---encoder, input->output', x.size(), self.encoder(x).size())
        return self.decoder.generate(
            start_tokens,
            self.max_seq_len,
            eos_token=self.eos_token,
            context=self.encoder(x),
            temperature=temperature)

    def export_enc_onnx(self, x, save_onnx_path):
        torch.onnx.export(
            self.encoder,
            x,
            save_onnx_path,
            export_params=True,
            opset_version=17,
            verbose=False,
            input_names=['input'],
            output_names=['output'],
            do_constant_folding=True,
            dynamic_axes={
                'input': {2: 'height', 3: 'width'},
                'output': {1: 'context1', 2: 'context2'},
            },
        )

    def export_decoder_weights(self, save_path):
        torch.save(self.decoder.state_dict(), save_path)


class EncoderDecoderV2(nn.Module):
    """model and implementation from
        https://github.com/lukas-blecher/LaTeX-OCR/tree/main

      post/prep from https://github.com/breezedeus/Pix2Text/tree/main
    """

    def __init__(self, **args):
        super().__init__()
        device = args['device']
        devices = args['devices']

        self.bos_token = args['bos_token']
        self.eos_token = args['eos_token']
        self.max_seq_len = args['max_seq_len']

        encoder_model_path = args['encoder_model_path']
        decode_model_path = args['decoder_model_path']

        from pybackend_libs.dataelem.framework.onnx_graph import ONNXGraph
        self.encoder = ONNXGraph(encoder_model_path, devices[0])
        self.decoder = TransformerDecoder.get_decoder(args)
        self.decoder.load_state_dict(
            torch.load(decode_model_path, map_location=device))
        self.decoder.eval()
        self.decoder.to(device)

    @torch.no_grad()
    def generate(self, x: torch.Tensor, temperature: float = 0.25):
        context = self.encoder.run([x.cpu().numpy()])[0]
        context = torch.from_numpy(context).to(x.device)
        start_tokens = (
            torch.LongTensor([self.bos_token])[:, None]).to(x.device)

        return self.decoder.generate(
            start_tokens,
            self.max_seq_len,
            eos_token=self.eos_token,
            context=context,
            temperature=temperature)


def get_image_resizer_model(args):
    resizer_checkpoint = args['resizer_checkpoint']
    max_dimensions = args['max_dimensions']
    device = args['device']

    image_resizer = ResNetV2(
        layers=[2, 3, 3],
        num_classes=max(max_dimensions) // 32,
        global_pool='avg',
        in_chans=1,
        drop_rate=0.05,
        preact=True,
        stem_type='same',
        conv_layer=StdConv2dSame,
    ).to(device)

    image_resizer.load_state_dict(
        torch.load(
            resizer_checkpoint,
            map_location=device,
        )
    )
    image_resizer.eval()
    return image_resizer


def minmax_size(
img: Image,
    max_dimensions: Tuple[int, int] = None,
    min_dimensions: Tuple[int, int] = None,
) -> Image:
    """Resize or pad an image to fit into given dimensions

    Args:
        img (Image): Image to scale up/down.
        max_dimensions (Tuple[int, int], optional):
          Maximum dimensions. Defaults to None.
        min_dimensions (Tuple[int, int], optional):
          Minimum dimensions. Defaults to None.
    Returns:
        Image: Image with correct dimensionality
    """
    if max_dimensions is not None:
        ratios = [a / b for a, b in zip(img.size, max_dimensions)]
        if any([r > 1 for r in ratios]):
            size = np.array(img.size) // max(ratios)
            img = img.resize(size.astype(int), Image.BILINEAR)
    if min_dimensions is not None:
        # hypothesis: there is a dim in img smaller than min_dimensions,
        # and return a proper dim >= min_dimensions
        padded_size = [
            max(img_dim, min_dim) for img_dim, min_dim
            in zip(img.size, min_dimensions)
        ]
        if padded_size != list(img.size):  # assert hypothesis
            padded_im = Image.new('L', padded_size, 255)
            padded_im.paste(img, img.getbbox())
            img = padded_im
    return img


def find_all_left_or_right(latex, left_or_right='left'):
    left_bracket_infos = []
    prefix_len = len(left_or_right) + 1
    # 匹配出latex中所有的 '\left' 后面跟着的第一个非空格字符，定位它们所在的位置
    for m in re.finditer(rf'\\{left_or_right}\s*\S', latex):
        start, end = m.span()
        # 如果最后一个字符为 "\"，则往前继续匹配，直到匹配到一个非字母的字符
        # 如 "\left \big("
        while latex[end - 1] in ('\\', ' '):
            end += 1
            while end < len(latex) and latex[end].isalpha():
                end += 1
        ori_str = latex[start + prefix_len : end].strip()
        # FIXME: ori_str中可能出现多个 '\left'，此时需要分隔开

        left_bracket_infos.append({'str': ori_str, 'start': start, 'end': end})
        left_bracket_infos.sort(key=lambda x: x['start'])
    return left_bracket_infos


def match_left_right(left_str, right_str):
    """匹配左右括号，如匹配 `\left(` 和 `\right)`。"""
    left_str = left_str.strip().replace(' ', '')[len('left') + 1 :]
    right_str = right_str.strip().replace(' ', '')[len('right') + 1 :]
    # 去掉开头的相同部分
    while left_str and right_str and left_str[0] == right_str[0]:
        left_str = left_str[1:]
        right_str = right_str[1:]

    match_pairs = [
        ('', ''),
        ('(', ')'),
        ('\{', '.'),  # 大括号那种
        ('⟮', '⟯'),
        ('[', ']'),
        ('⟨', '⟩'),
        ('{', '}'),
        ('⌈', '⌉'),
        ('┌', '┐'),
        ('⌊', '⌋'),
        ('└', '┘'),
        ('⎰', '⎱'),
        ('lt', 'gt'),
        ('lang', 'rang'),
        (r'langle', r'rangle'),
        (r'lbrace', r'rbrace'),
        ('lBrace', 'rBrace'),
        (r'lbracket', r'rbracket'),
        (r'lceil', r'rceil'),
        ('lcorner', 'rcorner'),
        (r'lfloor', r'rfloor'),
        (r'lgroup', r'rgroup'),
        (r'lmoustache', r'rmoustache'),
        (r'lparen', r'rparen'),
        (r'lvert', r'rvert'),
        (r'lVert', r'rVert'),
    ]
    return (left_str, right_str) in match_pairs


def post_post_process_latex(latex: str) -> str:
    """对识别结果做进一步处理和修正。"""
    # 把latex中的中文括号全部替换成英文括号
    latex = latex.replace('（', '(').replace('）', ')')
    # 把latex中的中文逗号全部替换成英文逗号
    latex = latex.replace('，', ',')

    left_bracket_infos = find_all_left_or_right(latex, left_or_right='left')
    right_bracket_infos = find_all_left_or_right(latex, left_or_right='right')
    # left 和 right 找配对，left找位置比它靠前且最靠近他的right配对
    for left_bracket_info in left_bracket_infos:
        for right_bracket_info in right_bracket_infos:
            if (
                not right_bracket_info.get('matched', False)
                and right_bracket_info['start'] > left_bracket_info['start']
                and match_left_right(
                    right_bracket_info['str'], left_bracket_info['str']
                )
            ):
                left_bracket_info['matched'] = True
                right_bracket_info['matched'] = True
                break

    for left_bracket_info in left_bracket_infos:
        # 把没有匹配的 '\left'替换为等长度的空格
        left_len = len('left') + 1
        if not left_bracket_info.get('matched', False):
            start_idx = left_bracket_info['start']
            end_idx = start_idx + left_len
            latex = (
                latex[: left_bracket_info['start']]
                + ' ' * (end_idx - start_idx)
                + latex[end_idx:]
            )
    for right_bracket_info in right_bracket_infos:
        # 把没有匹配的 '\right'替换为等长度的空格
        right_len = len('right') + 1
        if not right_bracket_info.get('matched', False):
            start_idx = right_bracket_info['start']
            end_idx = start_idx + right_len
            latex = (
                latex[: right_bracket_info['start']]
                + ' ' * (end_idx - start_idx)
                + latex[end_idx:]
            )

    # 把 latex 中的连续空格替换为一个空格
    latex = re.sub(r'\s+', ' ', latex)
    return latex


def pad(img: Image, divable: int = 32) -> Image:
    """Pad an Image to the next full divisible value of `divable`.
       Also normalizes the image and invert if needed.

    Args:
        img (PIL.Image): input image
        divable (int, optional): . Defaults to 32.

    Returns:
        PIL.Image
    """
    threshold = 128
    data = np.array(img.convert('LA'))
    if data[..., -1].var() == 0:
        data = (data[..., 0]).astype(np.uint8)
    else:
        data = (255-data[..., -1]).astype(np.uint8)
    data = (data-data.min())/(data.max()-data.min())*255
    if data.mean() > threshold:
        # To invert the text to white
        gray = 255*(data < threshold).astype(np.uint8)
    else:
        gray = 255*(data > threshold).astype(np.uint8)
        data = 255-data

    coords = cv2.findNonZero(gray)  # Find all non-zero points (text)
    a, b, w, h = cv2.boundingRect(coords)  # Find minimum spanning bounding box
    rect = data[b:b+h, a:a+w]
    im = Image.fromarray(rect).convert('L')
    dims = []
    for x in [w, h]:
        div, mod = divmod(x, divable)
        dims.append(divable*(div + (1 if mod > 0 else 0)))
    padded = Image.new('L', dims, 255)
    padded.paste(im, (0, 0, im.size[0], im.size[1]))
    return padded


def post_process(s: str):
    """Remove unnecessary whitespace from LaTeX code.

    Args:
        s (str): Input string

    Returns:
        str: Processed image
    """
    text_reg = r'(\\(operatorname|mathrm|text|mathbf)\s?\*? {.*?})'
    letter = '[a-zA-Z]'
    noletter = '[\W_^\d]'
    names = [x[0].replace(' ', '') for x in re.findall(text_reg, s)]
    s = re.sub(text_reg, lambda match: str(names.pop(0)), s)
    news = s
    while True:
        s = news
        news = re.sub(
            r'(?!\\ )(%s)\s+?(%s)' % (noletter, noletter), r'\1\2', s)
        news = re.sub(
            r'(?!\\ )(%s)\s+?(%s)' % (noletter, letter), r'\1\2', news)
        news = re.sub(r'(%s)\s+?(%s)' % (letter, noletter), r'\1\2', news)
        if news == s:
            break
    return s


def token2str(tokens, tokenizer) -> list:
    if len(tokens.shape) == 1:
        tokens = tokens[None, :]
    dec = [tokenizer.decode(tok) for tok in tokens]
    return [''.join(detok.split(' ')).replace('Ġ', ' ').replace(
                '[EOS]', '').replace('[BOS]', '').replace('[PAD]', '').strip()
            for detok in dec]


class LatexRecog(object):
    """Get a prediction of an image in the easiest way"""
    image_resizer = None

    def __init__(self, **kwargs):
        """Initialize a LatexOCR model
        """

        pretrain_path = kwargs.get('model_path')
        devices = kwargs.get('devices').split(',')
        device = torch.device(f'cuda:{devices[0]}' if devices[0] else 'cpu')
        use_onnx_encoder = kwargs.get('use_onnx_encoder', False)

        args_file = os.path.join(pretrain_path, 'model_config.json')
        args = json.load(open(args_file))
        args.update(
            resizer_checkpoint=os.path.join(pretrain_path, 'image_resizer.pth'))
        args.update(tokenizer=os.path.join(pretrain_path, 'tokenizer.json'))
        args.update(
            mfr_checkpoint=os.path.join(pretrain_path, 'p2t-mfr-20230702.pth'))
        args.update(device=device)

        if use_onnx_encoder:
            args['devices'] = devices
            args['encoder_model_path'] = pretrain_path
            args['decoder_model_path'] = os.path.join(
                pretrain_path, 'pytorch_model.bin')
            self.model = EncoderDecoderV2(**args)
        else:
            self.model = EncoderDecoder(**args)
            mfr_checkpoint = args['mfr_checkpoint']
            device = args['device']
            self.model.load_state_dict(
                torch.load(mfr_checkpoint, map_location=device))
            self.model.eval()

        self.max_dimensions = args['max_dimensions']
        self.min_dimensions = args['min_dimensions']
        self.temperature = args['temperature']
        self.device = device

        self.image_resizer = get_image_resizer_model(args)
        self.tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=args['tokenizer'])

        self.test_transform = alb.Compose(
                [
                    alb.ToGray(always_apply=True),
                    alb.Normalize(
                        (0.7931, 0.7931, 0.7931), (0.1738, 0.1738, 0.1738)),
                    # alb.Sharpen()
                    ToTensorV2(),
                ]
            )

    def preprocess(self, img, resize=True):
        assert img is not None, 'empty image'
        img = minmax_size(
            pad(img),
            self.max_dimensions,
            self.min_dimensions)

        input_image = img.convert('RGB').copy()
        r, w, h = 1, input_image.size[0], input_image.size[1]
        for _ in range(10):
            sampling_type = (
                Image.Resampling.BILINEAR
                if r > 1 else Image.Resampling.LANCZOS)
            h = int(h * r)  # height to resize
            resized_image = input_image.resize((w, h), sampling_type)
            img = pad(minmax_size(resized_image,
                    self.max_dimensions,
                    self.min_dimensions))
            t = self.test_transform(image=np.array(img.convert('RGB')))
            t = t['image'][:1].unsqueeze(0)
            w = self.image_resizer(t.to(self.device)).argmax(-1).item()
            w = (w + 1) * 32
            if w == img.size[0]:
                break

            r = w / img.size[0]

        im = t.to(self.device)
        return im

    def postprocess(self, dec):
        pred = post_process(token2str(dec, self.tokenizer)[0])
        pred = post_post_process_latex(pred)
        return pred


    def read_image(self, path):
        img = Image.open(path)
        img = ImageOps.exif_transpose(img).convert('RGB')
        return img

    def predict(self, inp) -> Dict[str, Any]:
        img = inp.get('b64_image')
        resize = inp.get('resize', True)
        b64_img = base64.b64decode(img)
        img = self.read_image(io.BytesIO(b64_img))
        im = self.preprocess(img, resize)
        dec = self.model.generate(im, temperature=self.temperature)
        pred = self.postprocess(dec)
        return pred

    def export_onnx(self, inp, save_path, save_decoder=False):
        img = inp.get('b64_image')
        resize = inp.get('resize', True)
        b64_img = base64.b64decode(img)
        img = self.read_image(io.BytesIO(b64_img))
        im = self.preprocess(img, resize)
        if save_decoder:
            self.model.export_decoder_weights(save_path)
        else:
            self.model.export_enc_onnx(im, save_path)
