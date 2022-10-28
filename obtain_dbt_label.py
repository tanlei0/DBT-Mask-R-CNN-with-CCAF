
import os
import torch
from tqdm import tqdm
import numpy as np
import pickle
import datetime
import transforms
from backbone import resnet50_fpn_backbone
from network_files import MaskRCNN
from train_utils import EvalCOCOMetricSingleImage
from dataset.crack500dataset import Crack500Instance


def summarize(self, catId=None):
    """
    Compute and display summary metrics for evaluation results.
    Note this functin can *only* be applied on the default parameter setting
    """

    def _summarize(ap=1, iouThr=None, areaRng='all', maxDets=100):
        p = self.params
        iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
        titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
        typeStr = '(AP)' if ap == 1 else '(AR)'
        iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
            if iouThr is None else '{:0.2f}'.format(iouThr)

        aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
        mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]

        if ap == 1:
            # dimension of precision: [TxRxKxAxM]
            s = self.eval['precision']
            # IoU
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]

            if isinstance(catId, int):
                s = s[:, :, catId, aind, mind]
            else:
                s = s[:, :, :, aind, mind]

        else:
            # dimension of recall: [TxKxAxM]
            s = self.eval['recall']
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]

            if isinstance(catId, int):
                s = s[:, catId, aind, mind]
            else:
                s = s[:, :, aind, mind]

        if len(s[s > -1]) == 0:
            mean_s = -1
        else:
            mean_s = np.mean(s[s > -1])

        print_string = iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s)
        return mean_s, print_string

    stats, print_list = [0] * 12, [""] * 12
    stats[0], print_list[0] = _summarize(1)
    stats[1], print_list[1] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2])
    stats[2], print_list[2] = _summarize(1, iouThr=.75, maxDets=self.params.maxDets[2])
    stats[3], print_list[3] = _summarize(1, areaRng='small', maxDets=self.params.maxDets[2])
    stats[4], print_list[4] = _summarize(1, areaRng='medium', maxDets=self.params.maxDets[2])
    stats[5], print_list[5] = _summarize(1, areaRng='large', maxDets=self.params.maxDets[2])
    stats[6], print_list[6] = _summarize(0, maxDets=self.params.maxDets[0])
    stats[7], print_list[7] = _summarize(0, maxDets=self.params.maxDets[1])
    stats[8], print_list[8] = _summarize(0, maxDets=self.params.maxDets[2])
    stats[9], print_list[9] = _summarize(0, areaRng='small', maxDets=self.params.maxDets[2])
    stats[10], print_list[10] = _summarize(0, areaRng='medium', maxDets=self.params.maxDets[2])
    stats[11], print_list[11] = _summarize(0, areaRng='large', maxDets=self.params.maxDets[2])

    print_info = "\n".join(print_list)

    if not self.eval:
        raise Exception('Please run accumulate() first')

    return stats, print_info


def save_info(coco_evaluators,
              category_index: dict,
              save_name: str = "record_mAP.txt",
              extra_metric_info = None):
    record_lines = []
    for coco_evaluator in coco_evaluators:
        iou_type = coco_evaluator.params.iouType
        print(f"IoU metric: {iou_type}")
        # calculate COCO info for all classes
        coco_stats, print_coco = summarize(coco_evaluator)

        # calculate voc info for every classes(IoU=0.5)
        classes = [v for v in category_index.values() if v != "N/A"]
        voc_map_info_list = []
        for i in range(len(classes)):
            stats, _ = summarize(coco_evaluator, catId=i)
            voc_map_info_list.append(" {:15}: {}".format(classes[i], stats[1]))

        print_voc = "\n".join(voc_map_info_list)
        print(print_voc)
        record_lines.extend([print_coco,""])
    line = "MUCov: "+str(extra_metric_info[-2])+" MWCov: "+str(extra_metric_info[-1])
    record_lines.extend([line,""])
   
    with open(save_name, "w") as f:
        f.write("\n".join(record_lines))


def main(parser_data):
    device = torch.device(parser_data.device if torch.cuda.is_available() else "cpu")
    print("Using {} device training.".format(device.type))

    data_transform = transforms.Compose([transforms.ToTensor()])
    
    batch_size = parser_data.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using %g dataloader workers' % nw)

    # load validation data set
    set_type = parser_data.dataset
    data_path = parser_data.data_path
    select_set = parser_data.set
    if set_type == "crack500":
        dataset = Crack500Instance(data_path, dataset=select_set,transforms=data_transform)

    
    dataset_loader = torch.utils.data.DataLoader(dataset,
                                                     batch_size=batch_size,
                                                     shuffle=False,
                                                     pin_memory=True,
                                                     num_workers=nw,
                                                     collate_fn=dataset.collate_fn)


    # create model
    backbone = resnet50_fpn_backbone()
    model = MaskRCNN(backbone, num_classes=args.num_classes + 1)

    # 载入你自己训练好的模型权重
    weights_path = parser_data.weights_path
    assert os.path.exists(weights_path), "not found {} file.".format(weights_path)
    model.load_state_dict(torch.load(weights_path, map_location='cpu')['model'])
    # print(model)
    model.to(device)

    # evaluate on the val dataset
    cpu_device = torch.device("cpu")
    now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tmp_json = "seg_"+__file__.split("/")[-1]+f"_{now}.json"
    
    seg_metric = EvalCOCOMetricSingleImage(dataset.coco, "segm", tmp_json)
    model.eval()
    results = {}  
    thr_range = np.linspace(0.01,1, 100)

    
    
    
    skip_flag = False
    with torch.no_grad():
        for i, (image, targets) in tqdm(enumerate(dataset_loader), desc="validation..."):
            
            image = list(img.to(device) for img in image)
            image_id = int(targets[0]["image_id"])
            # inference
            outputs = model(image)

            outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
            for output in outputs:
                if skip_flag:
                    break
                else:
                    for k,v in output.items():
                        if v.numel() == 0:
                            results[(image_id,0,0)] = 0
                            print(f"{image_id} is skipped")
                            skip_flag = True
                            break
            if skip_flag:
                skip_flag = False
                continue
    
            # NOTE: k is not relevant to OTM, used for other experiments, and can be ignored here
            k = 0
            for thr in thr_range:
                thr = np.round(thr, 2)
                seg_metric.update(targets, outputs, seg_thr=thr,k=k)
                seg_metric.synchronize_results()
                seg_info = seg_metric.evaluate(is_my_coco_eval=True)
                results[(image_id,k, thr)] =  {"MUCov": seg_info[-2], "MWCov":seg_info[-1]}
                print("results[({},{},{})]: {}".format(image_id, k,thr, results[(image_id, k, thr)]))
                seg_metric.release()
        
    with open(args.save_pkl, "wb") as fp:
        pickle.dump(results, fp)
    print("save the ", args.save_pkl)

    try:
        os.remove(tmp_json)
        print(f"remove {tmp_json}")
    except:
        print(f"remove {tmp_json} fail" )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__)

    parser.add_argument('--device', default='cuda:1', help='device')
    parser.add_argument('--num-classes', type=int, default=1, help='number of classes')
    parser.add_argument('--data-path', default='', help='dataset root')
    parser.add_argument('--weights-path', type=str, help='training weights')

    # batch size(set to 1, don't change)
    parser.add_argument('--batch-size', default=1, type=int, metavar='N',
                        help='batch size when validation.')
    parser.add_argument('--label-json-path', type=str, default="coco91_indices.json")
    parser.add_argument('--save_pkl', type=str, help="the save pkl file")
    parser.add_argument("--dataset", type=str, default="crack500")
    parser.add_argument("--set", type=str, default="train")
    args = parser.parse_args()

    main(args)
