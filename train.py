import os
import datetime

import torch
from torchvision.ops.misc import FrozenBatchNorm2d

import transforms
from network_files import MaskRCNN
from backbone import resnet50_fpn_backbone
from train_utils import train_eval_utils as utils
from train_utils import GroupedBatchSampler, create_aspect_ratio_groups
from dataset.crack500dataset import Crack500Instance
from dataset.gapsv2dataset import gapsV2Dataset
from dataset.crackls315 import CrackLS315

def create_model(num_classes, load_pretrain_weights=True, pretrain_root= ""):

    # resnet50 imagenet weights url: https://download.pytorch.org/models/resnet50-0676ba61.pth
    backbone = resnet50_fpn_backbone(pretrain_path=os.path.join(pretrain_root,"resnet50.pth"), trainable_layers=5,norm_layer=FrozenBatchNorm2d)

    model = MaskRCNN(backbone, num_classes=num_classes, use_best_thr=False)
    # print(model)
    if load_pretrain_weights:
        # coco weights url: "https://download.pytorch.org/models/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth"
        weights_dict = torch.load(os.path.join(pretrain_root,"maskrcnn_resnet50_fpn_coco.pth"), map_location="cpu")
        for k in list(weights_dict.keys()):
            if ("box_predictor" in k) or ("mask_fcn_logits" in k):
                del weights_dict[k]

        print(model.load_state_dict(weights_dict, strict=False))

    return model


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("Using {} device training.".format(device.type))

    now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    
    det_results_file = f"det_results{now}.txt"
    seg_results_file = f"seg_results{now}.txt"

    data_transform = {
        "train": transforms.Compose([transforms.ToTensor(),
                                     transforms.RandomHorizontalFlip(0.5)]),
        "val": transforms.Compose([transforms.ToTensor()])
    }

    data_root = args.data_path
    imp = args.imp
    DBT = args.DBT
    if DBT is True:
        othrs_file = args.othrs
    else:
        othrs_file = ""

    if imp is not None:
        if othrs_file is not "":
            train_dataset = Crack500Instance(data_root, dataset="train_imp",transforms=data_transform["train"], best_thr_file=othrs_file)  
        else:
            train_dataset = Crack500Instance(data_root, dataset="train_imp",transforms=data_transform["train"])  
    else:  
        if othrs_file is not "":
            train_dataset = Crack500Instance(data_root, dataset="train",transforms=data_transform["train"],best_thr_file=othrs_file)
        else:
            train_dataset = Crack500Instance(data_root, dataset="train",transforms=data_transform["train"])
    
  

    if args.aspect_ratio_group_factor >= 0:
        train_sampler = torch.utils.data.RandomSampler(train_dataset)
  
        group_ids = create_aspect_ratio_groups(train_dataset, k=args.aspect_ratio_group_factor)
   
        train_batch_sampler = GroupedBatchSampler(train_sampler, group_ids, args.batch_size)

    
    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using %g dataloader workers' % nw)

            
    if train_sampler:
        train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                        batch_sampler=train_batch_sampler,
                                                        pin_memory=True,
                                                        num_workers=nw,
                                                        collate_fn=train_dataset.collate_fn)
    else:
        train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                        batch_size=batch_size,
                                                        shuffle=True,
                                                        pin_memory=True,
                                                        num_workers=nw,
                                                        collate_fn=train_dataset.collate_fn)
    
    # load validation data set
    if imp is not None:
        val_dataset = Crack500Instance(data_root, dataset="test_imp",transforms=data_transform["val"])  
    else:  
        val_dataset = Crack500Instance(data_root, dataset="test",transforms=data_transform["val"])
    
    val_data_loader = torch.utils.data.DataLoader(val_dataset,
                                                  batch_size=1,
                                                  shuffle=False,
                                                  pin_memory=True,
                                                  num_workers=nw,
                                                  collate_fn=train_dataset.collate_fn)

    # create model num_classes equal background + classes
    model = create_model(num_classes=args.num_classes + 1, load_pretrain_weights=args.pretrain)
    model.to(device)
    
    train_loss = []
    learning_rate = []

    # define optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    # learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        milestones=args.lr_steps,
                                                        gamma=args.lr_gamma)

    if args.resume:
        # If map_location is missing, torch.load will first load the module to CPU
        # and then copy each parameter to where it was saved,
        # which would result in all processes on the same machine using the same set of devices.
        checkpoint = torch.load(args.resume, map_location='cpu')  
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        if args.amp and "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])

    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch, printing every 50 iterations
        mean_loss, lr = utils.train_one_epoch(model, optimizer, train_data_loader,
                                              device, epoch, print_freq=50,
                                              warmup=True, scaler=scaler)
        train_loss.append(mean_loss.item())
        learning_rate.append(lr)

        # update the learning rate
        lr_scheduler.step()

        # evaluate on the test dataset
        det_info, seg_info = utils.evaluate(model, val_data_loader, device=device)

        # write detection into txt
        with open(det_results_file, "a") as f:
          
            result_info = [f"{i:.4f}" for i in det_info + [mean_loss.item()]] + [f"{lr:.6f}"]
            txt = "epoch:{} {}".format(epoch, '  '.join(result_info))
            f.write(txt + "\n")

        # write seg into txt
        with open(seg_results_file, "a") as f:
           
            result_info = [f"{i:.4f}" for i in seg_info + [mean_loss.item()]] + [f"{lr:.6f}"]
            txt = "epoch:{} {}".format(epoch, '  '.join(result_info))
            f.write(txt + "\n")


        # save weights
        save_files = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch}
        if args.amp:
            save_files["scaler"] = scaler.state_dict()
        torch.save(save_files, "./save_weights/model_{}.pth".format(epoch))

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__)

  
    parser.add_argument('--device', default='cuda', help='device')

    parser.add_argument('--data-path', help='dataset')

    parser.add_argument('--num-classes', default=1, type=int, help='num_classes')

    parser.add_argument('--output-dir', default='./save_weights', help='path where to save')

    parser.add_argument('--resume', default='', type=str, help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')

    parser.add_argument('--epochs', default=26, type=int, metavar='N',
                        help='number of total epochs to run')

    parser.add_argument('--lr', default=0.004, type=float,
                        help='initial learning rate, 0.02 is the default value for training '
                             'on 8 gpus and 2 images_per_gpu')

    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')

    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')

    parser.add_argument('--lr-steps', default=[16, 22], nargs='+', type=int,
                        help='decrease lr every step-size epochs')

    parser.add_argument('--lr-gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')

    parser.add_argument('--batch_size', default=2, type=int, metavar='N',
                        help='batch size when training.')
    parser.add_argument('--aspect-ratio-group-factor', default=3, type=int)
    parser.add_argument("--pretrain", type=bool, default=False, help="load COCO pretrain weights.")

    parser.add_argument("--amp", default=False, help="Use torch.cuda.amp for mixed precision training")
    parser.add_argument("--imp", default=False, type=bool, help="Switch for CCAF")
    parser.add_argument("--DBT", default=True, type=bool, help="Switch for DBT")
    parser.add_argument("--othrs", default="", type=str, help="The DBT label file")
    args = parser.parse_args()
    print(args)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    main(args)
