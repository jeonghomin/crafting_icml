import argparse

import os
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from dataloader import *
from model import *
from dataset import *
from util import *
import logging
from mmcv.runner import get_time_str
import matplotlib.pyplot as plt
import yaml
import pickle
from utils.builder import build_pipeline, PIPELINES
from utils.builder import build_metrics
from utils.builder import build_models
from torchvision import transforms
from utils import calculate_PSNR_SSIM
logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser(description="Regression Tasks such as inpainting, denoising, and super_resolution",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--mode", default="test", choices=["train", "test"], type=str, dest="mode")
parser.add_argument("--train_continue", default="on", choices=["on", "off"], type=str, dest="train_continue")

parser.add_argument("--lr", default=1e-4, type=float, dest="lr")
parser.add_argument("--batch_size", default=8, type=int, dest="batch_size")
parser.add_argument("--num_epoch", default=500, type=int, dest="num_epoch")

parser.add_argument("--data_dir", default="./datasets/BSR/BSDS500/data/images", type=str, dest="data_dir")
parser.add_argument("--ckpt_dir", default="./checkpoint/srresnet/super_resolution", type=str, dest="ckpt_dir")
parser.add_argument("--log_dir", default="./log/srresnet/super_resolution", type=str, dest="log_dir")
parser.add_argument("--result_dir", default="./result/srresnet/super_resolution", type=str, dest="result_dir")
parser.add_argument("--test_dir", default="/nas/k8s/dev/research/intern/jhmin/test_project/despeckle/CTD/PIPAL/test/", type=str)

parser.add_argument("--task", default="super_resolution", choices=["inpainting", "denoising", "super_resolution"], type=str, dest="task")
parser.add_argument('--opts', nargs='+', default=['bilinear', 4.0, 0], dest='opts')

parser.add_argument("--ny", default=320, type=int, dest="ny")
parser.add_argument("--nx", default=480, type=int, dest="nx")
parser.add_argument("--nch", default=3, type=int, dest="nch")
parser.add_argument("--nker", default=64, type=int, dest="nker")

parser.add_argument("--network", default="srresnet", choices=["unet", "hourglass", "resnet", "srresnet"], type=str, dest="network")
parser.add_argument("--learning_type", default="plain", choices=["plain", "residual"], type=str, dest="learning_type")


parser.add_argument("--weight_path", default=None, type=str, help = "weight ( probabilistic distribution of each bins)")
parser.add_argument("--type", type=str, default=None, help = "div2k, pipal, etc .." )
args = parser.parse_args()


mode = args.mode
train_continue = args.train_continue

lr = args.lr
batch_size = args.batch_size
num_epoch = args.num_epoch



task = args.task
opts = [args.opts[0], np.asarray(args.opts[1:]).astype(np.float)]

ny = args.ny
nx = args.nx
nch = args.nch
nker = args.nker

network = args.network
learning_type = args.learning_type

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


args.config = "./degradation.yaml"

with open(args.config) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
  
    if args.weight_path is not None: 
        with open(args.weight_path, 'rb') as f:
            prob = pickle.load(f) # 75
        # args.result_dir = os.path.join(args.result_dir,f"ours_{get_time_str()}/") 
        # args.ckpt_dir = os.path.join(args.ckpt_dir,f"ours_{get_time_str()}/") 
        # args.log_dir = os.path.join(args.log_dir,f"ours_{get_time_str()}/") 
    else:
        prob = []
        for _ in range(75): 
            prob.append(np.float64(1/75))
        # args.result_dir = os.path.join(args.result_dir,f"baseline_{get_time_str()}/") 
        # args.ckpt_dir = os.path.join(args.ckpt_dir,f"baseline_{get_time_str()}/") 
        # args.log_dir = os.path.join(args.log_dir,f"baseline_{get_time_str()}/") 
data_dir = args.data_dir
ckpt_dir = args.ckpt_dir
log_dir = args.log_dir
result_dir = args.result_dir
print(args)

result_dir_train = os.path.join(result_dir, 'train')
result_dir_val = os.path.join(result_dir, 'val')
result_dir_test = os.path.join(result_dir, 'test')

if not os.path.exists(result_dir):
    os.makedirs(os.path.join(result_dir_train, 'png'))
    # os.makedirs(os.path.join(result_dir_train, 'numpy'))

    os.makedirs(os.path.join(result_dir_val, 'png'))
    # os.makedirs(os.path.join(result_dir_val, 'numpy'))

    os.makedirs(os.path.join(result_dir_test, 'png'))
    os.makedirs(os.path.join(result_dir_test, 'numpy'))


if mode == 'train':
    # transform_train = transforms.Compose([RandomCrop(shape=(ny, nx)), Normalization(mean=0.5, std=0.5), RandomFlip()])
    # transform_val = transforms.Compose([RandomCrop(shape=(ny, nx)), Normalization(mean=0.5, std=0.5)])

    transform_train = transforms.Compose([RandomCrop(shape=(288,288)), RandomFlip()])
    transform_val = transforms.Compose([])

    dataset_train = Dataset(data_dir=data_dir, 
                            transform=transform_train, task=task, opts=opts, pipelines=config["degradations"], weight =prob)
    
    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=8)

    dataset_val = TestDataset(data_dir=args.test_dir, transform=transform_val)
    loader_val = DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=8)


    num_data_train = len(dataset_train)
    num_data_val = len(dataset_val)

    num_batch_train = np.ceil(num_data_train / batch_size)
    num_batch_val = np.ceil(num_data_val / batch_size)
else:
    transform_test = transforms.Compose([])

    # transform_test = transforms.Compose([RandomCrop(shape=(ny, nx))])

    dataset_test = TestDataset(data_dir=args.test_dir, transform=transform_test)
    loader_test = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=8)

    
    num_data_test = len(dataset_test)

    num_batch_test = np.ceil(num_data_test / batch_size)


if network == "unet":
    net = UNet(in_channels=nch, out_channels=nch, nker=nker, learning_type=learning_type).to(device)
elif network == "hourglass":
    net = Hourglass(in_channels=nch, out_channels=nch, nker=nker, learning_type=learning_type).to(device)
elif network == "resnet":
    net = ResNet(in_channels=nch, out_channels=nch, nker=nker, learning_type=learning_type).to(device)
elif network == "srresnet":
    net = SRResNet(in_channels=nch, out_channels=nch, nker=nker, learning_type=learning_type).to(device)


# fn_loss = nn.BCEWithLogitsLoss().to(device)
fn_loss = nn.MSELoss().to(device)


optim = torch.optim.Adam(net.parameters(), lr=lr, betas=(0.5, 0.999))

fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
fn_denorm = lambda x, mean, std: (x * std) + mean
fn_class = lambda x: 1.0 * (x > 0.5)

cmap = None


writer_train = SummaryWriter(log_dir=os.path.join(log_dir, 'train'))
writer_val = SummaryWriter(log_dir=os.path.join(log_dir, 'val'))


st_epoch = 0

# TRAIN MODE
if mode == 'train':

    best_psnr = 0
    if train_continue == "on":
        net, optim, st_epoch = load(ckpt_dir=ckpt_dir, net=net, optim=optim)

    for epoch in range(st_epoch + 1, num_epoch + 1):
        net.train()
        loss_mse = []

        for batch, data in enumerate(loader_train, 1):
          
            label = data['label'].to(device)
            input = data['input'].to(device)

            output = net(input)

          
            optim.zero_grad()

            loss = fn_loss(output, label)
            loss.backward()

            optim.step()

        
            loss_mse += [loss.item()]

            print("TRAIN: EPOCH %04d / %04d | BATCH %04d / %04d | LOSS %.4f" %
                  (epoch, num_epoch, batch, num_batch_train, np.mean(loss_mse)))

            if epoch %50 == 0:
              if batch % 10 ==0:
                # Tensorboard 저장하기
                # label = fn_tonumpy(fn_denorm(label, mean=0.5, std=0.5))
                # input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5))
                # output = fn_tonumpy(fn_denorm(output, mean=0.5, std=0.5))

                label = fn_tonumpy(label)
                input = fn_tonumpy(input)
                output = fn_tonumpy(output)

                label = np.clip(label, a_min=0, a_max=1)
                input = np.clip(input, a_min=0, a_max=1)
                output = np.clip(output, a_min=0, a_max=1)

                id = num_batch_train * (epoch - 1) + batch
                
                plt.imsave(os.path.join(result_dir_train, '%04d_label.png' % id), label[0].squeeze(), cmap=None)
                plt.imsave(os.path.join(result_dir_train, '%04d_input.png' % id), input[0].squeeze(), cmap=None)
                plt.imsave(os.path.join(result_dir_train, '%04d_output.png' % id), output[0].squeeze(), cmap=None)

          

        writer_train.add_scalar('loss', np.mean(loss_mse), epoch)
      
        with torch.no_grad():
            PSNR = []
            SSIM = []
            
            net.eval()
            loss_mse = []
        
            for batch, data in enumerate(loader_val, 1):
          
                label = data['label'].to(device)
                input = data['input'].to(device)
              
                output = net(input)
          
                loss = fn_loss(output, label)
                loss_mse += [loss.item()]
                
                print("VALID: EPOCH %04d / %04d | BATCH %04d / %04d | LOSS %.4f" %
                      (epoch, num_epoch, batch, num_batch_val, np.mean(loss_mse)))
            
                if epoch > 500:
                    if batch %1 ==0:
             
                    # label = fn_tonumpy(fn_denorm(label, mean=0.5, std=0.5))
                    # input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5))
                    # output = fn_tonumpy(fn_denorm(output, mean=0.5, std=0.5))
                        to_y = True
                        output_img = calculate_PSNR_SSIM.tensor2img(output)
                        gt = calculate_PSNR_SSIM.tensor2img(label)
                        output_img = output_img[:, :, [2, 1, 0]]
                        gt = gt[:, :, [2, 1, 0]]
                        output_img = output_img.astype(np.float32) / 255.0
                        gt = gt.astype(np.float32) / 255.0

                        if to_y:
                            output_img = calculate_PSNR_SSIM.bgr2ycbcr(output_img, only_y=to_y)
                            gt = calculate_PSNR_SSIM.bgr2ycbcr(gt, only_y=to_y)

                        psnr = calculate_PSNR_SSIM.calculate_psnr(output_img * 255, gt * 255)
                        ssim = calculate_PSNR_SSIM.calculate_ssim(output_img * 255, gt * 255)

                        PSNR.append(psnr)
                        SSIM.append(ssim)


                        
                        label = fn_tonumpy(label)
                        input = fn_tonumpy(input)
                        output = fn_tonumpy(output)

                        label = np.clip(label, a_min=0, a_max=1)
                        input = np.clip(input, a_min=0, a_max=1)
                        output = np.clip(output, a_min=0, a_max=1)

                        id = num_batch_val * (epoch - 1) + batch
                        if epoch % 50 == 0 :
                            plt.imsave(os.path.join(result_dir_val, '%04d_label.png' % id), label[0].squeeze(), cmap=None)
                            plt.imsave(os.path.join(result_dir_val, '%04d_input.png' % id), input[0].squeeze(), cmap=None)
                            plt.imsave(os.path.join(result_dir_val, '%04d_output.png' % id), output[0].squeeze(), cmap=None)

                        # writer_val.add_image('label', label, id, dataformats='NHWC')
                        # writer_val.add_image('input', input, id, dataformats='NHWC')
                        # writer_val.add_image('output', output, id, dataformats='NHWC')
            psnr = np.mean(PSNR)
        writer_val.add_scalar('psnr', psnr, epoch)    
        writer_val.add_scalar('loss', np.mean(loss_mse), epoch)
        
        
        if epoch > 500:
            if epoch % 20 == 0:
                save(ckpt_dir=ckpt_dir, net=net, optim=optim, epoch=epoch)
            if psnr > best_psnr:
                best_psnr = psnr
                best_ckpt = os.path.join(ckpt_dir,"best")
                save(ckpt_dir=best_ckpt, net=net, optim=optim, epoch=0)
    writer_train.close()
    writer_val.close()

# TEST MODE
else:
    PSNR = []
    SSIM = []
    net, optim, st_epoch = load(ckpt_dir=ckpt_dir, net=net, optim=optim)

    with torch.no_grad():
        net.eval()
        loss_mse = []

        for batch, data in enumerate(loader_test, 1):
    
            label = data['label'].to(device)
            input = data['input'].to(device)

            output = net(input)

   
            loss = fn_loss(output, label)
            

            to_y = True
            output_img = calculate_PSNR_SSIM.tensor2img(output)
            gt = calculate_PSNR_SSIM.tensor2img(label)
            output_img = output_img[:, :, [2, 1, 0]]
            gt = gt[:, :, [2, 1, 0]]
            output_img = output_img.astype(np.float32) / 255.0
            gt = gt.astype(np.float32) / 255.0

            if to_y:
                output_img = calculate_PSNR_SSIM.bgr2ycbcr(output_img, only_y=to_y)
                gt = calculate_PSNR_SSIM.bgr2ycbcr(gt, only_y=to_y)

            psnr = calculate_PSNR_SSIM.calculate_psnr(output_img * 255, gt * 255)
            ssim = calculate_PSNR_SSIM.calculate_ssim(output_img * 255, gt * 255)
            PSNR.append(psnr)
            SSIM.append(ssim)
            logging.info('psnr: %.6f    ssim: %.6f' % (psnr, ssim))

            # path = os.path.join(save_path, image_name)
            # out_img = output[0].flip(dims=(0,)).clamp(0., 1.)
            # gt_img = label[0].flip(dims=(0,))
            
            loss_mse += [loss.item()]

            print("TEST: BATCH %04d / %04d | LOSS %.4f" %
                    (batch, num_batch_test, np.mean(loss_mse)))

            label = fn_tonumpy(label)
            input = fn_tonumpy(input)
            output = fn_tonumpy(output)

            for j in range(label.shape[0]):
                id = batch_size * (batch - 1) + j

                label_ = label[j]
                input_ = input[j]
                output_ = output[j]

           

                label_ = np.clip(label_, a_min=0, a_max=1)
                input_ = np.clip(input_, a_min=0, a_max=1)
                output_ = np.clip(output_, a_min=0, a_max=1)

                plt.imsave(os.path.join(result_dir_test, 'png', '%04d_label.png' % id), label_, cmap=cmap)
                plt.imsave(os.path.join(result_dir_test, 'png', '%04d_input.png' % id), input_, cmap=cmap)
                plt.imsave(os.path.join(result_dir_test, 'png', '%04d_output.png' % id), output_, cmap=cmap)
    PSNR = np.mean(PSNR)
    SSIM = np.mean(SSIM)
    logging.info('--------- average PSNR: %.06f,  SSIM: %.06f' % (PSNR, SSIM))
    # print("AVERAGE TEST: BATCH %04d / %04d | LOSS %.4f" %
    #       (batch, num_batch_test, np.mean(loss_mse)))

