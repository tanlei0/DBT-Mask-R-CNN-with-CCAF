import json
import copy
import cv2
import torch
import numpy as np
import os
from pycocotools.coco import COCO

from train_utils.mycocoeval import COCOeval
import pycocotools.mask as mask_util


def erode_for_mask(mask, k=2,kernel = np.ones((3, 3), dtype=np.uint8)):
    non_eroded = mask
    for i in range(k):
        eroded = cv2.erode(non_eroded.astype(np.uint8), kernel, 1)
        non_eroded = eroded
    
    return eroded


class EvalCOCOPOOL:
    def __init__(self,
                 coco: COCO = None,
                 iou_type: str = None,
                 results_file_name: str = "predict_results.json",
                 classes_mapping: dict = None):
        self.coco = copy.deepcopy(coco)
        self.img_ids = []  # 记录每个进程处理图片的ids
        self.results = []
        self.aggregation_results = None
        self.classes_mapping = classes_mapping
        self.coco_evaluator = None
        assert iou_type in ["bbox", "segm", "keypoints"]
        self.iou_type = iou_type
        self.results_file_name = results_file_name


    def prepare_for_coco_segmentation(self, targets, outputs, seg_thr=0.5,k=None):
        """将预测的结果转换成COCOeval指定的格式，针对实例分割任务"""
        # 遍历每张图像的预测结果
        for target, output in zip(targets, outputs):
            if len(output) == 0:
                continue

            img_id = int(target["image_id"])


            self.img_ids.append(img_id)
            per_image_masks = output["masks"]
            per_image_classes = output["labels"].tolist()
            per_image_scores = output["scores"].tolist()

            # TODO: the mask threshold 
            masks = (per_image_masks > seg_thr).int()
            if k is not None and k !=0:
                for i, mask in enumerate(masks):
                    masks[i] = torch.Tensor(erode_for_mask(mask.squeeze().numpy(),k=k),device=masks.device)

            res_list = []
            # 遍历每个目标的信息
            for mask, label, score in zip(masks, per_image_classes, per_image_scores):
                rle = mask_util.encode(np.array(mask[0, :, :, np.newaxis], dtype=np.uint8, order="F"))[0]
                rle["counts"] = rle["counts"].decode("utf-8")

                class_idx = int(label)
                if self.classes_mapping is not None:
                    class_idx = int(self.classes_mapping[str(class_idx)])

                res = {"image_id": img_id,
                       "category_id": class_idx,
                       "segmentation": rle,
                       "score": round(score, 3)}
                res_list.append(res)
            self.results.append(res_list)

    def update(self, targets, outputs, seg_thr=0.5, k=None):
        if self.iou_type == "bbox":
            pass
        elif self.iou_type == "segm":
            self.prepare_for_coco_segmentation(targets, outputs, seg_thr,k=k)
        else:
            raise KeyError(f"not support iou_type: {self.iou_type}")

    def synchronize_results(self, result_file_name=None):
        if result_file_name:
            self.results_file_name = result_file_name

        json_str = json.dumps(self.results, indent=4)
        with open(self.results_file_name, 'w') as json_file:
            json_file.write(json_str)

    def evaluate(self, is_my_coco_eval=False):
        # 只在主进程上评估即可
      
        # accumulate predictions from all images
        coco_true = self.coco
        coco_pre = coco_true.loadRes(self.results_file_name)

        self.coco_evaluator = COCOeval(cocoGt=coco_true, cocoDt=coco_pre, iouType=self.iou_type)
        my_metric = None
        if is_my_coco_eval:
            my_metric = self.coco_evaluator.my_evaluate()
        else:
            self.coco_evaluator.evaluate()
        self.coco_evaluator.accumulate()
        #print(f"IoU metric: {self.iou_type}")
        self.coco_evaluator.summarize(is_print=False)
        coco_info = self.coco_evaluator.stats.tolist()  # numpy to list

        if my_metric:
            coco_info.extend(list(my_metric.values()))
        
        return coco_info


    def release(self):
        self.img_ids = []
        self.results = []
        if os.path.isfile(self.results_file_name):
            os.remove(self.results_file_name)
            print("delete the file: {}", self.results_file_name)
        