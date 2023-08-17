import base64
import cv2


def decode_image_from_b64(b64_image):
    img = base64.b64decode(b64_image)
    img = np.fromstring(img, np.uint8)
    return cv2.imdecode(img, cv2.IMREAD_COLOR)


def convert_image_to_base64(image):
    image_binary = cv2.imencode('.jpg', image)[1].tobytes()
    x = base64.b64encode(image_binary)
    return x.decode('ascii').replace('\n', '')
