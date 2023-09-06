import os
import json
import cv2
import numpy as np
import base64

def convert_b64(file):
    if os.path.isfile(file):
        with open(file, 'rb') as fh:
            x = base64.b64encode(fh.read())
            x = x.decode()
            return x
    else:
        return None

det_image_folder = '/opt/workspace/datasets/all_kinds_train_images_angle/val'
image_files = os.listdir(det_image_folder)

# json_data = dict()
# json_data["data"] = []
# for index, file in enumerate(image_files):
#     print('image index:', index)
#     if index == 100:
#         break
#     img = cv2.imread(os.path.join(det_image_folder, file))
#     img = img.astype(np.float32)
#     h, w, c = img.shape
#     inputs = dict()
#     inputs["image"] = dict()
#     inputs["image"]["content"] = img.reshape(-1).tolist()
#     inputs["image"]["shape"] = [h, w, c]
#     inputs["params"] = dict()
#     inputs["params"]["content"] = ['{}']
#     inputs["params"]["shape"] = [1]
#     json_data["data"].append(inputs)

# with open(os.path.join(os.path.dirname(det_image_folder), 'val_data.json'), 'w') as f:
#     json.dump(json_data, f, ensure_ascii=False)


json_data = dict()
json_data["data"] = []
for index, file in enumerate(image_files):
    print('image index:', index)
    image_b64 = convert_b64(os.path.join(det_image_folder, file))
    inputs = dict()

    inputs["image_b64"] = dict()
    inputs["image_b64"]["content"] = [image_b64]
    inputs["image_b64"]["shape"] = [1]

    inputs["params"] = dict()
    parameters = {'longer_edge_size': 1600, 'refine_boxes': True, 'padding': True}
    parameters = json.dumps(parameters)
    inputs["params"]["content"] = [parameters]
    inputs["params"]["shape"] = [1]

    json_data["data"].append(inputs)

with open(os.path.join(
    os.path.dirname(det_image_folder),
    'val_data_padding_base64.json'), 'w') as f:
    json.dump(json_data, f, ensure_ascii=False)

