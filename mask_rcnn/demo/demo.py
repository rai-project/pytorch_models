from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
import cv2
from torchvision import transforms as T

from utils import resize_image
from utils import Masker

device = 'cpu'
min_image_size = 800
confidence_threshold = 0.7
show_mask_heatmaps = False
palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
mask_threshold = 0.5

INPUT_TO_BGR255 = True
INPUT_PIXEL_MEAN = [102.9801, 115.9465, 122.7717]
INPUT_PIXEL_STD = [1., 1., 1.]
DATALOADER_SIZE_DIVISIBILITY = 32
MODEL_MASK_ON = False
MODEL_KEYPOINT_ON = False

CATEGORIES = [
        "__background",
        "person",
        "bicycle",
        "car",
        "motorcycle",
        "airplane",
        "bus",
        "train",
        "truck",
        "boat",
        "traffic light",
        "fire hydrant",
        "stop sign",
        "parking meter",
        "bench",
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
        "backpack",
        "umbrella",
        "handbag",
        "tie",
        "suitcase",
        "frisbee",
        "skis",
        "snowboard",
        "sports ball",
        "kite",
        "baseball bat",
        "baseball glove",
        "skateboard",
        "surfboard",
        "tennis racket",
        "bottle",
        "wine glass",
        "cup",
        "fork",
        "knife",
        "spoon",
        "bowl",
        "banana",
        "apple",
        "sandwich",
        "orange",
        "broccoli",
        "carrot",
        "hot dog",
        "pizza",
        "donut",
        "cake",
        "chair",
        "couch",
        "potted plant",
        "bed",
        "dining table",
        "toilet",
        "tv",
        "laptop",
        "mouse",
        "remote",
        "keyboard",
        "cell phone",
        "microwave",
        "oven",
        "toaster",
        "sink",
        "refrigerator",
        "book",
        "clock",
        "vase",
        "scissors",
        "teddy bear",
        "hair drier",
        "toothbrush",
    ]


def build_transform():
    if INPUT_TO_BGR255:
    	to_bgr_transform = T.Lambda(lambda x: x * 255)
    else:
    	to_bgr_transform = T.Lambda(lambda x: x[[2, 1, 0]])

    normalize_transform = T.Normalize(
        mean=INPUT_PIXEL_MEAN, std=INPUT_PIXEL_STD
    )

    transform = T.Compose(
        [
            T.ToPILImage(),
            T.Resize(min_image_size),
            T.ToTensor(),
            to_bgr_transform,
            normalize_transform,
        ]
    )
    return transform

def run_prediction(image):
	#image_list = image
	result = image.copy()

	image = transforms(image)

	#image_list = to_image_list(image, DATALOADER_SIZE_DIVISIBILITY)
	image_list = resize_image(image, DATALOADER_SIZE_DIVISIBILITY)
	image_list = image_list.to(device)
	with torch.no_grad():
		predictions = model(image_list)

	predictions = [o.to("cpu") for o in predictions]

	# always single image is passed at a time
	prediction = predictions[0]

	height, width = result.shape[:-1]
	prediction = prediction.resize((width, height))

	if prediction.has_field("mask"):
		# if we have masks, paste the masks in the right position
	    # in the image, as defined by the bounding boxes
		masks = prediction.get_field("mask")
		# always single image is passed at a time
		masks = masker([masks], [prediction])[0]
		prediction.add_field("mask", masks)

	top_predictions = select_top_predictions(prediction, confidence_threshold)

	result = overlay_boxes(result, top_predictions)

	if MODEL_MASK_ON:
	    result = overlay_mask(result, top_predictions)

	result = overlay_class_names(result, top_predictions)
	return result


def select_top_predictions(predictions, confidence_threshold):
    """
    Select only predictions which have a `score` > self.confidence_threshold,
    and returns the predictions in descending order of score

    Arguments:
        predictions (BoxList): the result of the computation by the model.
            It should contain the field `scores`.

    Returns:
        prediction (BoxList): the detected objects. Additional information
            of the detection properties can be found in the fields of
            the BoxList via `prediction.fields()`
    """
    scores = predictions.get_field("scores")
    keep = torch.nonzero(scores > confidence_threshold).squeeze(1)
    predictions = predictions[keep]
    scores = predictions.get_field("scores")
    _, idx = scores.sort(0, descending=True)

    return predictions[idx]

def overlay_boxes(image, predictions):
    """
    Adds the predicted boxes on top of the image

    Arguments:
        image (np.ndarray): an image as returned by OpenCV
        predictions (BoxList): the result of the computation by the model.
            It should contain the field `labels`.
    """
    labels = predictions.get_field("labels")
    boxes = predictions.bbox

    colors = compute_colors_for_labels(labels).tolist()

    for box, color in zip(boxes, colors):
        box = box.to(torch.int64)
        top_left, bottom_right = box[:2].tolist(), box[2:].tolist()
        image = cv2.rectangle(image, tuple(top_left), tuple(bottom_right), tuple(color), 1)

    return image


def compute_colors_for_labels(labels):
    """
    Simple function that adds fixed colors depending on the class
    """
    colors = labels[:, None] * palette
    colors = (colors % 255).numpy().astype("uint8")
    return colors

def overlay_mask(image, predictions):
    """
    Adds the instances contours for each predicted object.
    Each label has a different color.

    Arguments:
        image (np.ndarray): an image as returned by OpenCV
        predictions (BoxList): the result of the computation by the model.
            It should contain the field `mask` and `labels`.
    """
    masks = predictions.get_field("mask").numpy()
    labels = predictions.get_field("labels")

    colors = compute_colors_for_labels(labels).tolist()

    for mask, color in zip(masks, colors):
        thresh = mask[0, :, :, None]
        contours, hierarchy = cv2_util.findContours(
            thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        image = cv2.drawContours(image, contours, -1, color, 3)

    composite = image

    return composite

def overlay_class_names(image, predictions):
    """
    Adds detected class names and scores in the positions defined by the
    top-left corner of the predicted bounding box

    Arguments:
        image (np.ndarray): an image as returned by OpenCV
        predictions (BoxList): the result of the computation by the model.
            It should contain the field `scores` and `labels`.
    """
    scores = predictions.get_field("scores").tolist()
    labels = predictions.get_field("labels").tolist()
    labels = [CATEGORIES[i] for i in labels]
    boxes = predictions.bbox

    template = "{}: {:.2f}"
    for box, score, label in zip(boxes, scores, labels):
        x, y = box[:2]
        s = template.format(label, score)
        cv2.putText(
            image, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1
        )

    return image

if __name__ == '__main__':
    model_path = '../pretrain/e2e_mask_rcnn_R_50_FPN_1x_caffe2.pth'
    image_path = "./input.jpg"
    output_path = "./"

    model = torch.load(model_path)
    model.eval()
    model.to(device)
    transforms = build_transform()
    masker = Masker(threshold=mask_threshold, padding=1)


    image = Image.open(image_path)
    image = np.array(image)[:, :, [2, 1, 0]]

    prediction = run_prediction(image)
    prediction = Image.fromarray(prediction[:, :, [2, 1, 0]])
    prediction.save(output_path+'output.jpg')