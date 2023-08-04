import numpy as np


def sort_segment_num(image_size):
    pre_nms_topk = 6000
    anchor_ratio_num = 9
    stage2 = image_size / 4
    stage3 = image_size / 8
    stage4 = image_size / 16
    stage5 = image_size / 32
    stage6 = image_size / 64

    anchor_num = (stage2 * stage2 + stage3 * stage3 +
                 stage4 * stage4 + stage5 * stage5 + stage6 * stage6) * anchor_ratio_num

    segment = np.sqrt(anchor_num / pre_nms_topk)
    for i in range(int(segment), 0, -1):
        if anchor_num % i == 0:
            break

    print('anchor_num:', anchor_num, 'segment:', i)

sort_segment_num(1600)

