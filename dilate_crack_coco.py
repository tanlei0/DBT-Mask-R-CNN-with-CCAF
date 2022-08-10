import argparse
import cv2
import os
import numpy as np
from pycocotools.coco import COCO
from pycocotools import mask as coco_mask
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from skimage import measure
import json
from tqdm import tqdm
def close_contour(contour):
    if not np.array_equal(contour[0], contour[-1]):
        contour = np.vstack((contour, contour[0]))
    return contour

def binary_mask_to_polygon(binary_mask, tolerance=0):
    """Converts a binary mask to COCO polygon representation
    Args:
    binary_mask: a 2D binary numpy array where '1's represent the object
    tolerance: Maximum distance from original points of polygon to approximated
    polygonal chain. If tolerance is 0, the original coordinate array is returned.
    """
    
    polygons = []
    # pad mask to close contours of shapes which start and end at an edge
    padded_binary_mask = np.pad(binary_mask, pad_width=1, mode='constant', constant_values=0)
    contours = measure.find_contours(padded_binary_mask, 0.5)
    contours = np.subtract(contours, 1)
    for contour in contours:
        contour = close_contour(contour)
        contour = measure.approximate_polygon(contour, tolerance)
        if len(contour) < 3:
            continue
        contour = np.flip(contour, axis=1)
        segmentation = contour.ravel().tolist()
        # after padding and subtracting 1 we may get -0.5 points in our segmentation
        segmentation = [0 if i < 0 else i for i in segmentation]
        polygons.append(segmentation)
    return polygons


def dilate_process_2(mask, k=2,h=1024, w=1024, kernel = np.ones((3, 3), dtype=np.uint8)):
    img_h = h
    img_w = w
    # kernel may need predefine ?
    rles = coco_mask.frPyObjects(mask, img_h, img_w)
    mask = coco_mask.decode(rles)
    if len(mask.shape) < 3:
        mask = mask[..., None]
    mask = mask.any(axis=2)
    for i in range(k):
        dilated = cv2.dilate(mask.astype(np.uint8), kernel, 1)
        mask = dilated
    
    return dilated


def dilate_crack(coco_json, save_path,k,kernel = np.ones((3, 3), dtype=np.uint8)):
    

    json_path = coco_json

    coco = COCO(annotation_file=json_path)
    # overwrite the coco var
    for ann_id, ann in tqdm(coco.anns.items()):
        img_id = ann["image_id"]
        img = coco.imgs[img_id]
        h, w = img["height"], img["width"]
        dilate2 = dilate_process_2(ann["segmentation"],k=k,kernel=kernel,h=h,w=w)
        dilate_ploy = binary_mask_to_polygon(dilate2)
        ann["segmentation"] = dilate_ploy
        dilate2_rle = coco_mask.encode(np.asfortranarray(dilate2))
        bbox = coco_mask.toBbox(dilate2_rle)
        ann["bbox"] = list(bbox)

    # write the json file
    data_coco = {}
    data_coco["images"] = list(coco.imgs.values())
    data_coco["categories"] = list(coco.cats.values())
    data_coco["annotations"] = list(coco.anns.values())

    
    with open(save_path, "w+") as fp:
        json.dump(data_coco, fp, indent=4)
    print("write the json file: ",save_path)



if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="dilate crack x times"
    )
    parser.add_argument(
        "--coco",
        help="The coco json file",
        type=str,
    )
    parser.add_argument(
        "--save",
        help="The save path",
        type=str,
    )
    parser.add_argument(
        "--k",
        help="dilate k times",
        type=int,
        default=2,
    )
    args = parser.parse_args()
    dilate_crack(args.coco, args.save, args.k)
    
