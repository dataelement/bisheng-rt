import base64
import os
from inspect import getsourcefile

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

this_dir = os.path.dirname(os.path.abspath(getsourcefile(lambda: 0)))

# try:
#     vis_font = ImageFont.truetype(
#         os.path.join(this_dir, '../socr_data/data/Alibaba-PuHuiTi-Bold.ttf'),
#         50)
# except Exception:
#     vis_font = ImageFont.load_default()

vis_font = ImageFont.truetype(os.path.join(this_dir, 'simsun.ttc'), 50)


class Box(object):
    def __init__(self, box):
        if isinstance(box, list):
            box = np.array(box, dtype=np.int32)
        self.box = box

    def get_box_size(self):
        w1 = np.sqrt(np.sum(np.square(self.box[1] - self.box[0])))
        h1 = np.sqrt(np.sum(np.square(self.box[3] - self.box[0])))
        return w1, h1

    def get_box_center(self):
        return (self.box[0] + self.box[2]) / 2


def generate_word_patch(text, box, img_shape):
    img_height, img_width = img_shape
    width, height = vis_font.getsize(text)
    box_w, box_h = Box(box).get_box_size()
    text_image = Image.new('RGBA', (width, height), (0, 0, 128, 50))
    draw_text = ImageDraw.Draw(text_image)
    draw_text.text((0, 0), text=text, font=vis_font, fill=(0, 255, 128, 128))

    text_image = text_image.resize((int(box_w), int(box_h)),
                                   resample=Image.BILINEAR)
    pt1 = np.float32([[0, 0], [box_w, 0], [box_w, box_h], [0, box_h]])
    pt2 = np.float32(box)
    M = cv2.getPerspectiveTransform(pt1, pt2)
    text_image = np.asarray(text_image)
    patch = cv2.warpPerspective(text_image, M, (img_width, img_height))

    return patch


def ocr_visual(image, res, draw_number, rotate_angle=0, rotateupright=True):

    if isinstance(image, np.ndarray):
        img_array = image
        is_arr = True
    else:
        img_array = np.fromstring(image, np.uint8)
        img_array = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        is_arr = False

    if rotateupright:
        if rotate_angle == 90:
            img_array = cv2.rotate(img_array, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif rotate_angle == -90:
            img_array = cv2.rotate(img_array, cv2.ROTATE_90_CLOCKWISE)
        elif rotate_angle == 180:
            img_array = cv2.rotate(img_array, cv2.ROTATE_180)

    img_pil = Image.fromarray(np.uint8(img_array))

    _boxes = res['bboxes']['position'] if isinstance(res['bboxes'],
                                                     dict) else res['bboxes']
    patchs = np.zeros((img_array.shape[0], img_array.shape[1], 4),
                      dtype=np.uint8)
    for ind, (box, text) in enumerate(zip(_boxes, res['texts'])):
        patch = generate_word_patch(text, box, img_array.shape[:2])
        # patch = Image.fromarray(patch)
        patchs = cv2.add(patch, patchs)

    patchs = Image.fromarray(patchs)
    img_pil.paste(patchs, (0, 0), mask=patchs)
    img_array = np.asarray(img_pil)
    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)

    for ind, (box, text) in enumerate(zip(_boxes, res['texts'])):
        box = np.array(box)
        cv2.polylines(img_array, [box.astype(np.int32).reshape((-1, 1, 2))],
                      True,
                      color=(255, 255, 0),
                      thickness=2,
                      lineType=cv2.LINE_AA)
        if draw_number:
            ind_str = ','.join(
                list(map(str, [
                    ind,
                ] + res['row_col_info'][ind]))
            ) if 'row_col_info' in res else str(ind)
            text_size = cv2.getTextSize(ind_str,
                                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                        fontScale=0.5,
                                        thickness=1)[0]
            cv2.putText(img_array,
                        ind_str,
                        org=(int(box[0][0] - text_size[0]),
                             int(box[0][1] + text_size[1] / 2)),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5,
                        thickness=1,
                        color=(0, 0, 255))

    if is_arr:
        return img_array
    else:
        # 将图片编码成流数据，放到内存缓存中，然后转化成string格式
        buffer = cv2.imencode('.jpg', img_array)[1]
        pic_str = base64.b64encode(buffer)
        pic_str = pic_str.decode()
        return pic_str


def convert_base64(image):
    image_binary = cv2.imencode('.jpg', image)[1].tobytes()
    x = base64.b64encode(image_binary)
    return x.decode('ascii').replace('\n', '')


def draw_box_on_img(image, boxes_group, color=(0, 0, 255), thickness=3):
    for index, box in enumerate(boxes_group):
        cv2.polylines(image, [box.astype(np.int32).reshape((-1, 1, 2))],
                      True,
                      color=color,
                      thickness=thickness)
        cv2.circle(image, (int(float(box[0, 0])), int(float(box[0, 1]))), 4,
                   (255, 0, 0), thickness)
        cv2.putText(image,
                    str(index), (int(float(box[1, 0])), int(float(box[1, 1]))),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5,
                    thickness=1,
                    color=(0, 0, 255))
    return convert_base64(image)


def draw_text_on_image(image, text, pos, color=(0, 0, 255), font_size=12):
    image = Image.fromarray(image.astype('uint8')).convert('RGB')
    draw = ImageDraw.Draw(image)
    this_dir = os.path.dirname(os.path.abspath(getsourcefile(lambda: 0)))
    try:
        font = ImageFont.truetype(
            os.path.join(this_dir, 'Alibaba-PuHuiTi-Bold.ttf'), font_size)
    except Exception:
        font = ImageFont.load_default()
    draw.text(pos, text, fill=color, font=font)
    return np.array(image)


def draw_box(image, box, color=(0, 0, 255)):
    box = np.array(box)
    cv2.line(image, tuple(box[0].astype(np.int32)),
             tuple(box[1].astype(np.int32)), color)
    cv2.line(image, tuple(box[1].astype(np.int32)),
             tuple(box[2].astype(np.int32)), color)
    cv2.line(image, tuple(box[2].astype(np.int32)),
             tuple(box[3].astype(np.int32)), color)
    cv2.line(image, tuple(box[3].astype(np.int32)),
             tuple(box[0].astype(np.int32)), color)
