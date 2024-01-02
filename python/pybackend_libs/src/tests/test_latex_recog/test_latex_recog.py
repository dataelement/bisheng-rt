import json

import numpy as np
import torch
from PIL import Image, ImageOps
from recog_model import EncoderDecoder, LatexOCRAlg


def read_image(path):
    img = Image.open(path)
    img = ImageOps.exif_transpose(img).convert('RGB')
    return img


def test_recog_model():
    args_file = './data/args.json'
    args = json.load(open(args_file, 'r'))
    model = EncoderDecoder(**args)

    mfr_checkpoint = args['mfr_checkpoint']
    device = args['device']
    model.load_state_dict(torch.load(mfr_checkpoint, map_location=device))

    with open('./data/recog_in.npy', 'rb') as fin:
        im = np.load(fin)

    device = torch.device('cpu')
    x = torch.from_numpy(im).to(device)
    pred = model.generate(x, temperature=0.20)
    print('---pred', pred.size(), np.sum(pred.numpy()))


def test_latex_ocr_alg():
    args_file = './data/args.json'
    model = LatexOCRAlg(args_file)

    img = read_image('./data/zh1.jpg')
    res = model.infer(img)
    print('res', res)


test_latex_ocr_alg()
# test_recog_model()
