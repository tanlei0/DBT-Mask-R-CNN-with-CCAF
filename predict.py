import os
import time
import json
import cv2
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torchvision import transforms

from network_files import MaskRCNN
from backbone import resnet50_fpn_backbone
from draw_box_utils import draw_objs
import pandas as pd

def create_model(num_classes, box_thresh=0.5,use_best_thr = False):
    backbone = resnet50_fpn_backbone()
    model = MaskRCNN(backbone,
                     num_classes=num_classes,
                     rpn_score_thresh=box_thresh,
                     box_score_thresh=box_thresh,
                     use_best_thr=use_best_thr)

    return model

def erode_for_mask(mask, k=2,kernel = np.ones((3, 3), dtype=np.uint8)):
    non_eroded = mask
    for i in range(k):
        eroded = cv2.erode(non_eroded.astype(np.uint8), kernel, 1)
        non_eroded = eroded
    
    return eroded

def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()

def main(args):
    num_classes = 1  # 不包含背景
    box_thresh = 0.5
    weight_path = args.weight_path
    img_path = args.img_path
    category_index = {"1":"crack"}

    use_best_thr = args.DBT

    # get devices
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # create model 
    model = create_model(num_classes=num_classes + 1, box_thresh=box_thresh,use_best_thr=use_best_thr)

    # load train weights
    assert os.path.exists(weight_path), "{} file dose not exist.".format(weight_path)
    weights_dict = torch.load(weight_path, map_location='cpu')
    weights_dict = weights_dict["model"] if "model" in weights_dict else weights_dict
    model.load_state_dict(weights_dict)
    model.to(device)


    # load image
    assert os.path.exists(img_path), f"{img_path} does not exits."
    original_img = Image.open(img_path).convert('RGB')

    # from pil image to tensor, do not normalize image
    data_transform = transforms.Compose([transforms.ToTensor()])
    img = data_transform(original_img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    model.eval() 
    with torch.no_grad():
        # init
        img_height, img_width = img.shape[-2:]
        init_img = torch.zeros((1, 3, img_height, img_width), device=device)
        model(init_img)

        t_start = time_synchronized()
        predictions = model(img.to(device))[0]
        t_end = time_synchronized()
        print("inference+NMS time: {}".format(t_end - t_start))


        predict_boxes = predictions["boxes"].to("cpu").numpy()
        predict_classes = predictions["labels"].to("cpu").numpy()
        predict_scores = predictions["scores"].to("cpu").numpy()
        predict_mask = predictions["masks"].to("cpu").numpy()
        predict_mask = np.squeeze(predict_mask, axis=1)  # [batch, 1, h, w] -> [batch, h, w]
        if use_best_thr:
            predict_thr = predictions["thr"].to("cpu").numpy()
            print("predict_thr:", predict_thr)
        else:
            predict_thr = 0.5

        
        if len(predict_boxes) == 0:
            print("No object!")
            return
        

        plot_img = draw_objs(original_img,
                             boxes=predict_boxes,
                             classes=predict_classes,
                             scores=predict_scores,
                             masks=predict_mask,
                             category_index=category_index,
                             line_thickness=3,
                             font='arial.ttf',
                             font_size=20,
                             mask_thresh=predict_thr,
                             is_gt=False,
                             )
        plt.imshow(plot_img)
        plt.show()

        plot_img.save(os.path.join('./results',img_path.split("/")[-1]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__)

    parser.add_argument('--weight_path', help='weight', type=str)
    parser.add_argument('--img_path', help='image path',type=str)
    parser.add_argument('--DBT', help='Swith for DBT branch',type=bool,default=False)
    parser.add_argument('--k', help='k2 for CCAF',type=int, default=0)
    args = parser.parse_args()

    main(args)

