import base64
import copy
import json
# import pdb
from io import BytesIO

import cv2
import numpy as np
import PIL
import torch
import torch.nn.functional as tnf
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import triton_python_backend_utils as pb_utils
from PIL import Image, ImageFile
from skimage import img_as_float
from skimage.exposure import rescale_intensity
from skimage.filters import threshold_yen
from torch.utils.dlpack import from_dlpack

np.set_printoptions(threshold=np.inf)
ImageFile.LOAD_TRUNCATED_IMAGES = True

resize = 512
transform_pil = transforms.Compose([
    transforms.ToPILImage(),
])


def img_to_tensor(im, normalize=None):
    tensor = torch.from_numpy(
        np.moveaxis(im / (255.0 if im.dtype == np.uint8 else 1), -1,
                    0).astype(np.float32))
    if normalize is not None:
        return F.normalize(tensor, **normalize)
    return tensor


def padding_image(img, h, w):
    bottom, right = 0, 0
    if h % 16:
        new_h = 16 - h % 16 + h
        bottom = new_h - h
    if w % 16:
        new_w = 16 - w % 16 + w
        right = new_w - w

    img = cv2.copyMakeBorder(img,
                             0,
                             bottom,
                             0,
                             right,
                             cv2.BORDER_CONSTANT,
                             value=(255, 255, 255))

    return img


def padding_detect_res(img, max_score):
    x, y = 0, 0
    w, h = img.shape[1], 50

    cv2.rectangle(img, (x, y), (x + w, y + h), (204, 232, 207), -1)

    text = 'ps rate: {:.4f}'.format(max_score)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = h / 50
    font_thickness = 1
    (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale,
                                          font_thickness)
    text_x = int((w - text_w) / 2)
    text_y = int((h + text_h) / 2)
    cv2.putText(img, text, (text_x, text_y), font, font_scale, (0, 0, 0),
                font_thickness)
    return img


def pb_tensor_to_numpy(pb_tensor):
    if pb_tensor.is_cpu():
        return pb_tensor.as_numpy()
    else:
        pytorch_tensor = from_dlpack(pb_tensor.to_dlpack())
        return pytorch_tensor.detach().cpu().numpy()


def exec_graph(model_name, output_names, input_names, raw_inputs):
    inputs = []
    for name, inp in zip(input_names, raw_inputs):
        inputs.append(pb_utils.Tensor(name, inp))

    inference_request = pb_utils.InferenceRequest(
        model_name=model_name,
        requested_output_names=output_names,
        inputs=inputs)

    # return inference_request.async_exec()
    infer_response = inference_request.exec()
    if infer_response.has_error():
        raise pb_utils.TritonModelException(infer_response.error().message())

    graph_outputs = []
    for index, output_ in enumerate(output_names):
        pb_tensor = pb_utils.get_output_tensor_by_name(infer_response, output_)
        graph_outputs.append(pb_tensor_to_numpy(pb_tensor))
    return graph_outputs


def Corner_point_detection(mode, img):
    if mode == 'harris':
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = np.float32(gray)
        dst = cv2.cornerHarris(gray, 2, 3, 0.04)
        # result用于标记角点，并不重要
        dst = cv2.dilate(dst, None)
        # 最佳值的阈值，它可能因图像而异。
        img[dst > 0.01 * dst.max()] = [0, 0, 255]
    elif mode == 'shi_Tomasi':
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners = cv2.goodFeaturesToTrack(gray, 45, 0.01, 10)
        corners = np.int0(corners)
        # img3 = cv2.drawKeypoints(img,corners,None,color=(0,0,255))
        for i in corners:
            x, y = i.ravel()
            cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
    elif mode == 'fast':
        fast = cv2.FastFeatureDetector_create()
        kp = fast.detect(img, None)
        img = cv2.drawKeypoints(img, kp, None, color=(0, 0, 255))
    return img


def boxfilter(Iarr, r):

    M, N = Iarr.shape
    dest = np.zeros((M, N))
    sumY = np.cumsum(Iarr, axis=0)
    dest[:r + 1] = sumY[r:2 * r + 1]  # top r+1 lines
    dest[r + 1:M - r] = sumY[2 * r + 1:] - sumY[:M - 2 * r - 1]
    dest[-r:] = np.tile(
        sumY[-1], (r, 1)) - sumY[M - 2 * r - 1:M - r - 1]  # bottom r lines
    sumX = np.cumsum(dest, axis=1)
    dest[:, :r + 1] = sumX[:, r:2 * r + 1]  # left r+1 columns
    dest[:, r + 1:N - r] = sumX[:, 2 * r + 1:] - sumX[:, :N - 2 * r - 1]
    dest[:, -r:] = np.tile(
        sumX[:, -1][:, None],
        (1, r)) - sumX[:, N - 2 * r - 1:N - r - 1]  # right r columns
    return dest


def resize_pil_image(pil_image, max_size):
    im = pil_image
    ratio = 1
    if im.width > max_size:
        desired_width = max_size
        desired_height = 1.0 * im.height / (1.0 * im.width / max_size)
        ratio = 1.0 * max_size / im.width
    else:
        desired_height = im.height
        desired_width = im.width
    # convert back to integer
    desired_height = int(desired_height)
    desired_width = int(desired_width)
    im_resized = im.resize((desired_width, desired_height),
                           resample=Image.BILINEAR)
    return im_resized, ratio


def blank(shape, dtype=np.uint8, filler='0'):
    if filler == '0':
        blank = np.zeros(shape, dtype)

    elif filler == '1':
        blank = np.ones(shape, dtype)
    else:
        return "BAD FILLER VALUE; MUST BE STRINGS OF '0' OR '1'"

    return blank


def resize(img, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = img.shape[:2]

    if width is None and height is None:
        return img

    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)

    else:
        r = width / float(w)
        dim = (width, int(h * r))

    resized = cv2.resize(img, dim, interpolation=inter)
    return resized


def order_points(pts):
    # automatically convert a python list to a numpy array
    if str(type(pts)) != "<class 'numpy.ndarray'>":
        pts = np.array(pts)

    corners = np.zeros((4, 2), dtype='float32')
    sums = pts.sum(axis=1)  # sum up all numbers horizontally
    corners[0] = pts[np.argmin(sums)]  # find out the TOP-LEFT coordinate
    corners[2] = pts[np.argmax(sums)]  # find out the BOTTOM-RIGHT coordinate

    diffs = np.diff(pts, axis=1)
    corners[1] = pts[np.argmin(diffs)]
    corners[3] = pts[np.argmax(diffs)]

    return corners


def perspective_transform(img, pts):
    corners_old = order_points(pts)
    tl, tr, br, bl = corners_old

    distT = np.sqrt(((tr[0] - tl[0])**2) + ((tr[1] - tl[1])**2))
    distB = np.sqrt(((br[0] - bl[0])**2) + ((br[1] - bl[1])**2))
    maxW = max(int(distT), int(distB))

    distL = np.sqrt(((tl[0] - bl[0])**2) + ((tl[1] - bl[1])**2))
    distR = np.sqrt(((tr[0] - br[0])**2) + ((tr[1] - br[1])**2))
    maxH = max(int(distL), int(distR))
    corners_corrected = np.array(
        [[0, 0], [maxW - 1, 0], [maxW - 1, maxH - 1], [0, maxH - 1]],
        dtype='float32')

    matrix = cv2.getPerspectiveTransform(corners_old, corners_corrected)
    img_corrected = cv2.warpPerspective(img, matrix, (maxW, maxH))

    return img_corrected


def preprocess(img):
    img_adj = cv2.convertScaleAbs(img, alpha=float(1.56), beta=float(-60))

    scale = img_adj.shape[0] / 500.0
    img_scaled = resize(img_adj, height=500)
    img_gray = cv2.cvtColor(img_scaled, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.GaussianBlur(img_gray, (11, 11), 0)
    img_edge = cv2.Canny(img_gray, 60, 245)

    dkernel = np.ones((3, 3), np.uint8)
    img_edge = cv2.dilate(img_edge, dkernel, iterations=1)

    return img_adj, scale, img_scaled, img_edge


def gethull(img_edge):
    img_prehull = img_edge.copy()
    images, outlines, _ = cv2.findContours(img_prehull, cv2.RETR_LIST,
                                           cv2.CHAIN_APPROX_SIMPLE)
    img_hull = blank(img_prehull.shape, img_prehull.dtype, '0')

    for outline in range(len(outlines)):
        hull = cv2.convexHull(outlines[outline])
        cv2.drawContours(img_hull, [hull], 0, 255, 3)

    ekernel = np.ones((3, 3), np.uint8)
    img_hull = cv2.erode(img_hull, ekernel, iterations=1)
    return img_hull


def getcorners(img_hull):
    img_outlines = img_hull.copy()
    images, outlines, _ = cv2.findContours(img_outlines, cv2.RETR_LIST,
                                           cv2.CHAIN_APPROX_SIMPLE)
    outlines = sorted(outlines, key=cv2.contourArea, reverse=True)[:4]

    corners = None
    for outline in outlines:
        perimeter = cv2.arcLength(outline, True)
        approx = cv2.approxPolyDP(outline, 0.02 * perimeter, True)
        if len(approx) == 4:
            corners = approx
            break

    return corners


def enhance_constract_v1(image):
    image = img_as_float(image)

    yen_threshold = threshold_yen(image)
    bright = rescale_intensity(image, (0, yen_threshold), (0, 255))
    return bright


def enhance_constract_v2(image, clip_hist_percent=1):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate grayscale histogram
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist_size = len(hist)

    # Calculate cumulative distribution from the histogram
    accumulator = []
    accumulator.append(float(hist[0]))
    for index in range(1, hist_size):
        accumulator.append(accumulator[index - 1] + float(hist[index]))

    # Locate points to clip
    maximum = accumulator[-1]
    clip_hist_percent *= (maximum / 100.0)
    clip_hist_percent /= 2.0

    # Locate left cut
    minimum_gray = 0
    while accumulator[minimum_gray] < clip_hist_percent:
        minimum_gray += 1

    # Locate right cut
    maximum_gray = hist_size - 1
    while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
        maximum_gray -= 1

    # Calculate alpha and beta values
    alpha = 255 / (maximum_gray - minimum_gray)
    beta = -minimum_gray * alpha

    auto_result = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    return auto_result


def enhance_constract_v3(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)

    # Applying CLAHE to L-channel
    # feel free to try different values for the limit and grid size:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)

    # merge the CLAHE enhanced L-channel with the a and b channel
    limg = cv2.merge((cl, a, b))

    # Converting image from LAB Color model to BGR color spcae
    enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return enhanced_img


def enhance_constract_v4(image, bins=256):
    image_flattened = image.flatten()
    image_hist = np.zeros(bins)

    # frequency count of each pixel
    for pix in image:
        image_hist[pix] += 1

    # cummulative sum
    cum_sum = np.cumsum(image_hist)
    norm = (cum_sum - cum_sum.min()) * 255
    # normalization of the pixel values
    n_ = cum_sum.max() - cum_sum.min()
    uniform_norm = norm / n_
    uniform_norm = uniform_norm.astype('int')

    # flat histogram
    image_eq = uniform_norm[image_flattened]
    # reshaping the flattened matrix to its original shape
    image_eq = np.reshape(a=image_eq, newshape=image.shape)

    return image_eq


def image_sharpen(image,
                  kernel_size=(5, 5),
                  sigma=1.0,
                  amount=1.0,
                  threshold=0):

    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened


def perspective_correct(image):
    ori_image = copy.deepcopy(image)
    img_adj, scale, img_scaled, img_edge = preprocess(image)

    img_hull = gethull(img_edge)
    corners = getcorners(img_hull)

    if corners is None:
        print('INFO: No suitable contours were found')
        return ori_image

    corners = corners.reshape(4, 2) * scale
    img_corrected = perspective_transform(img_adj, corners)
    return img_corrected


def cut_border(color_image,
               gray,
               mode='black',
               threshold_black=20,
               threshold_white=235):
    # color_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    if mode == 'black':
        _, thresh = cv2.threshold(gray, threshold_black, 255,
                                  cv2.THRESH_BINARY)
    elif mode == 'white':
        _, thresh = cv2.threshold(gray, threshold_white, 255,
                                  cv2.THRESH_BINARY_INV)
    else:
        raise pb_utils.TritonModelException(
            'cut_border mode must be black or white')

    images, contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
    biggest = np.array([])
    max_area = 0
    for cntrs in contours:
        area = cv2.contourArea(cntrs)
        peri = cv2.arcLength(cntrs, True)
        approx = cv2.approxPolyDP(cntrs, 0.02 * peri, True)
        if area > max_area and len(approx) == 4:
            biggest = approx
            max_area = area
    cnt = biggest
    x, y, w, h = cv2.boundingRect(cnt)
    crop = color_image[y:y + h, x:x + w]
    return crop


def method_binarize_otsu(gray):
    ret, threshold_img = cv2.threshold(gray, 0, 255,
                                       cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return threshold_img


def method_binarize_adapt(gray, blockSize=11, C=2):
    threshold_img = cv2.adaptiveThreshold(gray, 255,
                                          cv2.ADAPTIVE_THRESH_MEAN_C,
                                          cv2.THRESH_BINARY, blockSize, C)
    return threshold_img


def method_binarize_sauv(gray, window_size=15, k=0.2, R=128):
    # 得到行和列
    rows, cols = gray.shape
    threshold_img = np.zeros((rows, cols), np.uint8)
    for r in range(rows):
        for c in range(cols):
            # 计算局部区域内的像素值均值和标准差
            x1 = max(0, r - window_size)
            x2 = min(rows - 1, r + window_size)
            y1 = max(0, c - window_size)
            y2 = min(cols - 1, c + window_size)
            mean, stddev = cv2.meanStdDev(gray[x1:x2, y1:y2])
            # 计算局部阈值
            threshold = mean * (1 + k * ((stddev / R) - 1))
            # 如果像素值大于阈值，则将该像素设为255，否则为0
            if gray[r, c] > threshold:
                threshold_img[r, c] = 255
            else:
                threshold_img[r, c] = 0
    return threshold_img


def method_images_fix(pil_image, bin_image):
    img_type = pil_image.format
    if img_type == 'GIF':
        img = pil_image.convert('RGBA')
        x, y = img.size
        background = PIL.Image.new('RGBA', img.size, (255, 255, 255))
        background.paste(img, (0, 0, x, y), img)
        pil_image = background.convert('RGB')

    if img_type == 'TIFF':
        offset = pil_image.tag_v2[513]
        bytecount = pil_image.tag_v2[514]
        img = bin_image[offset:(offset + bytecount)]
        pil_image = Image.open(BytesIO(img))

    return pil_image


def method_images_compress(pil_image, image_size):
    if image_size > 2000000:
        w, h = pil_image.size
        max_size = 2000
        if w < max_size and h < max_size:
            pass
        else:
            pil_image, ratio = resize_pil_image(pil_image, max_size)
        return pil_image, ratio
    else:
        return pil_image, 1.


class EraseWaterMarkModel(object):
    def __init__(self, params={}):
        self.params = params

    def prep(self, image):
        """
        预处理图像归一化, bbox_ndarray去掉固长的padding并转化为list
        Args:
            - image: (-1,-1,3) BGR格式 cv2读取图片的风格
            - bbox_ndarray: ndarray, (n, -1), 印章检测器输出的印章坐标ndarray
        Rets
            - image: paddle tensor (1, 3, -1, -1)
            - seal_cnts: list
        """
        image = image.transpose((2, 0, 1))
        image = image.astype('float32')
        image /= 255
        image = np.expand_dims(image, axis=0)
        return image

    # async def infer(self, image, seal_cnts):
    def infer(self, image):
        """
        根据坐标把每个印章裁剪出来，送入模型推理，再把擦除后结果贴回原图
        Args
            - image: paddle tensor (1, 3, -1, -1)
            - seal_cnts: list
        Rets:
            - res: paddle tensor (1, 3, -1, -1)
        """
        res = copy.deepcopy(image)
        _, _, h, w = image.shape
        padder_size = 128
        if h <= 512 and w <= 512:  # h < 1600 and w < 1600
            _, _, H, W = image.shape
            mod_pad_h = (padder_size - H % padder_size) % padder_size
            mod_pad_w = (padder_size - W % padder_size) % padder_size
            clip = np.pad(image,
                          ((0, 0), (0, 0), (0, mod_pad_h), (0, mod_pad_w)),
                          mode='constant')
            tt = self.model(clip)
            res = tt
            res = res[:, :, :H, :W]
        else:
            step = min(h, w, 512)  # 512
            res = np.zeros_like(image, dtype='float32')
            for i in range(0, h, step):
                for j in range(0, w, step):
                    clip = image[:, :, i:(i + step), j:(j + step)]
                    _, _, H, W = clip.shape
                    mod_pad_h = (padder_size - H % padder_size) % padder_size
                    mod_pad_w = (padder_size - W % padder_size) % padder_size
                    clip = np.pad(clip, ((0, 0), (0, 0), (0, mod_pad_h),
                                         (0, mod_pad_w)),
                                  mode='constant')

                    tt = exec_graph('erase_net_wm', ['conv2d_143.tmp_1'],
                                    ['x'], [clip])[0]
                    g_images_clip = tt
                    g_images_clip = g_images_clip[:, :, :H, :W]
                    res[:, :, i:(i + step), j:(j + step)] = g_images_clip
        return res

    def post(self, image):
        """
        后处理, 逆归一化并转化为numpy,返回
        Args
            - image: paddle tensor (1, 3, -1, -1)
        Rets:
            - res: numpy array (-1, -1, 3) BGR格式 需可视化直接用cv2保存
        """
        image = np.clip(image * 255, 0, 255)
        image = np.round(image).astype(np.uint8)
        image = image[0].transpose((1, 2, 0))
        return image

    def execute(self, image):
        """
        Args:
            - image: (-1,-1,3) BGR格式 cv2读取图片风格
        Rets
            - a new image (-1,-1,3) BGR格式 cv2读取图片风格
        """
        image = self.prep(image)
        res = self.infer(image)
        res = self.post(res)
        return res


class EraseSealModel(object):
    def __init__(self, params={}):
        self.params = params

    def prep(self, image, bbox_ndarray):
        """
        预处理图像归一化, bbox_ndarray去掉固长的padding并转化为list
        Args:
            - image: (-1,-1,3) BGR格式 cv2读取图片的风格
            - bbox_ndarray: ndarray, (n, -1), 印章检测器输出的印章坐标ndarray
        Rets
            - image: paddle tensor (1, 3, -1, -1)
            - seal_cnts: list
        """
        image = image.transpose((2, 0, 1))
        # image = paddle.to_tensor(image).astype('float32')
        image = image.astype('float32')
        image /= 255
        # image = paddle.unsqueeze(image, 0)
        image = np.expand_dims(image, axis=0)

        seal_cnts = bbox_ndarray.tolist()
        for index, cnt in enumerate(seal_cnts):
            # delete padding
            if -100 in cnt:
                find_index = cnt.index(-100)
                seal_cnts[index] = cnt[:find_index]
        return image, seal_cnts

    def infer(self, image, seal_cnts):
        """
        根据坐标把每个印章裁剪出来，送入模型推理，再把擦除后结果贴回原图
        Args
            - image: paddle tensor (1, 3, -1, -1)
            - seal_cnts: list
        Rets:
            - res: paddle tensor (1, 3, -1, -1)
        """
        # res = image.clone()
        res = copy.deepcopy(image)
        _, _, h, w = image.shape
        for coor_list in seal_cnts:
            for i, c in enumerate(coor_list):
                coor_list[i] = float(coor_list[i])
                if len(coor_list) % 2 != 0:
                    print('warning: 坐标xy点数量不能整除2!')
                    continue
            x_list = coor_list[::2]
            y_list = coor_list[1::2]
            left_top = [min(x_list), min(y_list)]
            right_bottom = [max(x_list), max(y_list)]
            x_extend = (right_bottom[0] - left_top[0]) / 8
            y_extend = (right_bottom[1] - left_top[1]) / 8
            left_top[0] = int(np.clip(np.round((left_top[0] - x_extend)), 0,
                                      w))
            left_top[1] = int(np.clip(np.round((left_top[1] - y_extend)), 0,
                                      h))
            right_bottom[0] = int(
                np.clip(np.round((right_bottom[0] + x_extend)), 0, w))
            right_bottom[1] = int(
                np.clip(np.round((right_bottom[1] + y_extend)), 0, h))
            ins_local = copy.deepcopy(image[:, :,
                                            left_top[1]:right_bottom[1] + 1,
                                            left_top[0]:right_bottom[0] + 1])
            # with paddle.no_grad():
            padder_size = 128
            _, _, H, W = ins_local.shape
            mod_pad_h = (padder_size - H % padder_size) % padder_size
            mod_pad_w = (padder_size - W % padder_size) % padder_size
            # ins_local = F.pad(ins_local, [0, mod_pad_w, 0, mod_pad_h])
            ins_local = np.pad(ins_local, ((0, 0), (0, 0), (0, mod_pad_h),
                                           (0, mod_pad_w)),
                               mode='constant')
            # ins_local_infer = model(ins_local)
            ins_local_infer = exec_graph('erase_net', ['conv2d_143.tmp_1'],
                                         ['x'], [ins_local])[0]

            ins_local_infer = ins_local_infer[:, :, :H, :W]
            res[:, :, left_top[1]:right_bottom[1] + 1,
                left_top[0]:right_bottom[0] + 1] = ins_local_infer
        return res

    def post(self, image):
        """
        后处理, 逆归一化并转化为numpy,返回
        Args
            - image: paddle tensor (1, 3, -1, -1)
        Rets:
            - res: numpy array (-1, -1, 3) BGR格式 需可视化直接用cv2保存
        """
        # image = paddle.clip(image*255, 0, 255)
        # image = paddle.round(image)
        image = np.clip(image * 255, 0, 255)
        image = np.round(image).astype(np.uint8)
        image = image[0].transpose((1, 2, 0))
        # image = image.numpy()
        return image

    def execute(self, image, boxes):
        """
        Args:
            - image: (-1,-1,3) BGR格式 cv2读取图片风格
            - bbox_ndarray: ndarray, (n, -1) 印章检测器输出的印章坐标ndarray
        Rets
            - a new image (-1,-1,3) BGR格式 cv2读取图片风格
        """
        image, seal_cnts = self.prep(image, boxes)
        res = self.infer(image, seal_cnts)
        res = self.post(res)
        return res


def detect_ps(image):
    fake_size = image.shape
    fake_ = cv2.resize(image, (512, 512))

    img = fake_.reshape(
        (-1, fake_.shape[-3], fake_.shape[-2], fake_.shape[-1]))

    normalize = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}

    # if len(img) != 1:
    #     pdb.set_trace()
    img = img_to_tensor(img[0], normalize).unsqueeze(0).numpy()

    res = exec_graph('detect_ps_net',
                     ['noise_extractor.model.fc.bias', '1410'],
                     ['model.conv1.weight'], [img])
    seg = res[1]
    seg = torch.sigmoid(torch.from_numpy(seg)).detach().cpu()
    if torch.isnan(seg).any() or torch.isinf(seg).any():
        max_score = 0.0
    else:
        max_score = torch.max(seg).numpy()

    seg = [np.array(transform_pil(seg[i])) for i in range(len(seg))]

    # if len(seg) != 1:
    #     pdb.set_trace()
    # else:
    #     fake_seg = seg[0]
    fake_seg = seg[0]

    fake_seg = cv2.resize(fake_seg, (fake_size[1], fake_size[0]))
    image = fake_seg.astype(np.uint8)

    image = padding_detect_res(image, max_score)
    return image


def erase_moire(image):
    h, w = image.shape[:2]
    if h % 16 or w % 16:
        image = padding_image(image, h, w)

    image = image.astype(np.float32) / 255.0
    image = image.reshape((1, ) + image.shape)
    _gt = exec_graph('erase_moire_net', ['depth2_space_3/DepthToSpace'],
                     ['input_1'], [image])[0]
    _gt = _gt.copy()
    _gt = _gt[0]
    _gt[_gt > 1] = 1
    _gt[_gt < 0] = 0
    _gt = _gt * 255.0
    _gt = np.round(_gt).astype(np.uint8)
    return _gt


def cutedge(image):
    im_ori = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    im_ori = im_ori.astype(np.float32) / 255.
    h, w, _ = im_ori.shape
    im = cv2.resize(im_ori, (256, 256))
    im = im.transpose(2, 0, 1)
    img = torch.from_numpy(im).float().unsqueeze(0)

    bm = exec_graph('cut_edge_model', ['module.update_block.mask.2.bias'],
                    ['module.imcnn.conv1.weight'], [img.numpy()])[0]

    bm0 = cv2.resize(bm[0, 0], (w, h))  # x flow
    bm1 = cv2.resize(bm[0, 1], (w, h))  # y flow
    bm0 = cv2.blur(bm0, (3, 3))
    bm1 = cv2.blur(bm1, (3, 3))
    lbl = torch.from_numpy(np.stack([bm0, bm1],
                                    axis=2)).unsqueeze(0)  # h * w * 2
    out = tnf.grid_sample(torch.from_numpy(im_ori).permute(
        2, 0, 1).unsqueeze(0).float(),
                          lbl,
                          align_corners=True)
    out = ((out[0] * 255).permute(1, 2,
                                  0).numpy())[:, :, ::-1].astype(np.uint8)
    return out


class GeneralPreprocess(object):
    def __init__(self, params={}):
        self.params = params
        self.seal_model = params.get('seal_model', 'seal_detect_algo')

    def execute(self, bin_image, params):
        cv_image = None
        ratio = 1.

        image_fix_flag = params.get('image_fix', False)
        image_compress_flag = params.get('compress_image', False)
        binarize_method = params.get('binarize_method', None)
        cut_border_mode = params.get('cut_border', None)

        if image_fix_flag or image_compress_flag:
            pil_image, image_size = self.convert_image_bin2pil(bin_image)

            if image_fix_flag:
                pil_image = method_images_fix(pil_image, bin_image)

            if image_compress_flag:
                pil_image, ratio = method_images_compress(
                    pil_image, image_size)
            np_image = np.array(pil_image)
            cv_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)
        else:
            cv_image = cv2.imdecode(np.frombuffer(bin_image, np.uint8),
                                    cv2.IMREAD_COLOR)

        if params.get('enable_gray', False):
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        if params.get('perspective_correct', False):
            # cv_image = perspective_correct(cv_image)
            cv_image = cutedge(cv_image)

        if cut_border_mode:
            if len(cv_image.shape) == 2:  # 灰度图
                gray = cv_image
            else:
                gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

            cv_image = cut_border(cv_image, gray, mode=cut_border_mode)

        if params.get('enhance_constract_clarity', False):
            cv_image = enhance_constract_v2(cv_image)
            cv_image = image_sharpen(cv_image)

        if binarize_method:
            if len(cv_image.shape) == 2:  # 灰度图
                gray = cv_image
            else:
                gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

            if binarize_method == 'otsu':
                cv_image = method_binarize_otsu(gray)

            elif binarize_method == 'adapt':
                cv_image = method_binarize_adapt(gray)

            elif binarize_method == 'sauv':
                cv_image = method_binarize_sauv(gray)

        if params.get('erase_watermark', False):
            erase_watermark_model = EraseWaterMarkModel(self.params)
            cv_image = erase_watermark_model.execute(cv_image)

        if params.get('erase_seal', False):
            erase_seal_model = EraseSealModel(self.params)
            image = cv_image.astype(np.float32)
            boxes = exec_graph(self.seal_model,
                               ['boxes', 'labels', 'boxes_cos', 'boxes_sin'],
                               ['image'], [image])[0]
            cv_image = erase_seal_model.execute(image, boxes)

        if params.get('remove_moire', False):
            cv_image = erase_moire(cv_image)

        if params.get('detect_ps', False):
            cv_image = detect_ps(cv_image)

        h, w = cv_image.shape[:2]
        prep_param = np.array([
            json.dumps({
                'resize_scale': 1.0 / ratio,
                'height': h,
                'width': w
            })
        ],
                              dtype=np.object_)

        return cv_image, prep_param

    def convert_image_bin2pil(self, bin_image):
        try:
            pil_image = PIL.Image.open(BytesIO(bin_image))
            image_size = len(bin_image)
        except Exception as e:
            print('e:', str(e))
            raise pb_utils.TritonModelException(str(e))
        return pil_image, image_size


class GeneralPostprocess(object):
    def __init__(self, params={}):
        self.params = params

    def execute(self, bboxes, params):
        if 'resize_scale' in params:
            resize_scale = params.get('resize_scale')
            if np.abs(resize_scale - 1.0) > 1e-6:
                bboxes = bboxes * resize_scale

        return bboxes


class TritonPythonModel:
    def initialize(self, args):
        self.model_config = json.loads(args['model_config'])
        params = self.model_config['parameters']
        self.output_names = ['prep_image', 'prep_params']
        self.prep_model = GeneralPreprocess(params)

    async def execute(self, requests):
        try:

            def _get_np_input(request, name, has_batch=True):
                return pb_utils.get_input_tensor_by_name(request,
                                                         name).as_numpy()

            def _get_optional_params(request, name):
                tensor = pb_utils.get_input_tensor_by_name(request, name)
                return json.loads(tensor.as_numpy()[0]) if tensor else {}

            responses = []
            for request in requests:
                bin_images = _get_np_input(request, 'bin_images')
                params = _get_optional_params(request, 'params')
                outputs = []
                if not bin_images:
                    raise pb_utils.TritonModelException(
                        'input bin_image is empty')
                outputs = self.prep_model.execute(
                    base64.b64decode(bin_images[0]), params)
                outputs = [outputs[0], outputs[1]]
        except Exception:
            import traceback
            output = dict()
            output['code'] = 300
            output['message'] = traceback.format_exc()
            outputs = [
                np.zeros([0, 0, 3]),
                np.array([json.dumps(output)], dtype=np.object_)
            ]

        tensors = []
        for out, out_name in zip(outputs, self.output_names):
            tensors.append(pb_utils.Tensor(out_name, out))
        inference_response = pb_utils.InferenceResponse(output_tensors=tensors)
        responses.append(inference_response)

        return responses
