
import os
from skimage import io
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import segmentation_models_pytorch as smp
import PIL
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import argparse
import random
import itertools
from datetime import datetime
import data as dataset
import itertools


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--data_root', type=str, default='D:/gq/dataset/height/data')
parser.add_argument('--resume', type=int, default=1)
parser.add_argument('--model_dir', type=str, default='save_model', help='path of save model')
parser.add_argument('--model_name', type=str, default='VNY_1', help='model name')
parser.add_argument('--epochs', type=int, default=100000, help='number of training epochs')
parser.add_argument('--batch_size', type=int, default=4, help='number of batch size')
parser.add_argument('--val_frequency', type=int, default=100)
parser.add_argument('--summary_frequency', type=int, default=100)
parser.add_argument('--save_frequency', type=int, default=100)
parser.add_argument('--gpu_id', type=str, default='0')


args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id


def get_inputs(id, loaders, iters):
    inputs = None
    try:
        inputs = next(iters[id])
    except StopIteration:
        iters[id] = iter(loaders[id])
        inputs = next(iters[id])

    return inputs


def vis(height):
    mean = height.mean(dim=[1, 2], keepdim=True)
    return (height - mean) * 0.035 + 0.5


def put(height):
    height_in = height - height.mean(dim=[1, 2, 3], keepdim=True)
    return height_in


def val_step(data_loader, T_model, H_model, writer, prefix='prefix', step=0):
    T_model.eval()
    H_model.eval()

    test_loss, accuracy, test_rmse, test_zncc = [], [], [], []
    test_data = next(iter(data_loader))
    test_images, test_height = test_data['image'], test_data['height']
    with torch.no_grad():
        test_T = T_model.forward(test_images)
        test_H = H_model.forward(test_T)
    h_in = put(test_height)
    batch_loss = torch.pow(h_in - test_H, 2).mean()
    test_loss.append(batch_loss.item())
    accuracy.append(1 - torch.mean(torch.abs(test_H - h_in)).item())

    rmse = torch.sqrt(torch.pow(h_in - test_H, 2).mean(dim=[1, 2, 3])).mean()
    mu_in = test_height.mean(dim=[1, 2, 3], keepdim=True)
    mu_out = test_H.mean(dim=[1, 2, 3], keepdim=True)
    std_in = torch.sqrt(torch.pow(test_height - mu_in, 2).mean(dim=[1, 2, 3], keepdim=True))
    std_out = torch.sqrt(torch.pow(test_H - mu_out, 2).mean(dim=[1, 2, 3], keepdim=True))
    zncc = ((test_height - mu_in) * (test_H - mu_out) / (std_in * std_out)).mean()

    test_rmse.append(rmse.item())
    test_zncc.append(zncc.item())

    writer.add_scalar('val/%s/test_loss_H' % prefix, sum(test_loss) / len(test_loss), global_step=step)
    writer.add_scalar('val/%s/test_acc' % prefix, sum(accuracy) / len(accuracy), global_step=step)
    writer.add_scalar('val/%s/test_rmse' % prefix, sum(test_rmse) / len(test_rmse), global_step=step)
    writer.add_scalar('val/%s/test_zncc' % prefix, sum(test_zncc) / len(test_zncc), global_step=step)

    writer.add_image('val/%s/image' % prefix, test_images[0], global_step=step)
    writer.add_image('val/%s/labels_vis' % prefix, vis(test_height[0]), global_step=step)
    writer.add_image('val/%s/T_out' % prefix, test_T[0], global_step=step)
    writer.add_image('val/%s/P_height_vis' % prefix, vis(test_H[0]), global_step=step)


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_data_vai = dataset.Data(os.path.join(args.data_root, 'Vaihingen'),
                                  mode='train', device=device, augment=False, caching=True)
    train_loader_vai = DataLoader(train_data_vai, batch_size=args.batch_size, shuffle=True, drop_last=True)
    train_data_pot = dataset.Data(os.path.join(args.data_root, 'Potsdam'),
                                  mode='train', device=device, augment=False, caching=True)
    train_loader_pot = DataLoader(train_data_pot, batch_size=args.batch_size, shuffle=True, drop_last=True)
    train_data_syn = dataset.Data(os.path.join(args.data_root, 'Synthetic'),
                                  mode='train', device=device, augment=False, caching=True)
    train_loader_syn = DataLoader(train_data_syn, batch_size=args.batch_size, shuffle=True, drop_last=True)

    val_data_vai = dataset.Data(os.path.join(args.data_root, 'Vaihingen'),
                                mode='val', device=device, augment=False)
    val_loader_vai = DataLoader(val_data_vai, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_data_pot = dataset.Data(os.path.join(args.data_root, 'Potsdam'),
                                mode='val', device=device, augment=False)
    val_loader_pot = DataLoader(val_data_pot, batch_size=args.batch_size, shuffle=True, drop_last=False)

    train_loaders = [train_loader_vai, train_loader_pot, train_loader_syn]
    train_iters = [iter(train_loader_pot), iter(train_loader_vai), iter(train_loader_syn)]

    T_model = smp.Unet('mobilenet_v2', classes=3, activation='sigmoid')
    D_model = smp.Unet('mobilenet_v2', classes=3)
    H_model = smp.Unet('mobilenet_v2', classes=1)
    zeros = torch.zeros([args.batch_size, 512, 512], dtype=torch.long, device=device)
    ones = torch.ones([args.batch_size, 512, 512], dtype=torch.long, device=device)
    twos = torch.ones([args.batch_size, 512, 512], dtype=torch.long, device=device) + 1
    style_labels = [zeros, ones, twos]

    save_dir = os.path.join(args.model_dir, args.model_name)
    T_model_path = os.path.join(save_dir, 'T_model.pth')
    D_model_path = os.path.join(save_dir, 'D_model.pth')
    H_model_path = os.path.join(save_dir, 'H_model.pth')
    if args.resume > 0:
        if os.path.exists(T_model_path):
            T_model.load_state_dict(torch.load(T_model_path))
        if os.path.exists(H_model_path):
            H_model.load_state_dict(torch.load(H_model_path))
        if os.path.exists(D_model_path):
            D_model.load_state_dict(torch.load(D_model_path))

    optimizer_D = torch.optim.Adam(D_model.parameters(), lr=0.003)
    optimizer_TH = torch.optim.Adam(itertools.chain(T_model.parameters(), H_model.parameters()),
                                    lr=0.001)

    T_model.to(device)
    D_model.to(device)
    H_model.to(device)

    date_time = datetime.now().strftime("_%Y_%m_%d_%H_%M_%S")
    writer = SummaryWriter(os.path.join('runs', args.model_name + date_time))
    writer.add_text('args', str(args), 0)

    train_loss_T, train_loss_D, train_loss_H = [1], [1], [1]

    for steps in range(args.epochs):

        T_model.train()
        D_model.train()
        H_model.train()

        data_id = random.randint(0, 1)
        data_id = [0, 2][data_id]
        inputs = get_inputs(data_id, train_loaders, train_iters)
        train_images, train_height = inputs['image'], inputs['height']

        T_out = T_model.forward(train_images)
        D_out = D_model.forward(T_out)
        H_out = H_model.forward(T_out)

        loss_T = 0.0
        loss_H = 0.0
        loss_D = nn.functional.cross_entropy(D_out, style_labels[data_id])

        if data_id != 0:
            loss_T = nn.functional.cross_entropy(D_out, style_labels[0])
            train_loss_T.append(loss_T.item())
        if data_id != 1:
            loss_H = torch.pow(put(train_height) - H_out, 2).mean()
            train_loss_H.append(loss_H.item())

        loss_TH = loss_H + 0.5 * loss_T
        optimizer_TH.zero_grad()
        loss_TH.backward(retain_graph=True)
        optimizer_TH.step()

        optimizer_D.zero_grad()
        loss_D.backward()
        optimizer_D.step()
        train_loss_D.append(loss_D.item())

        if steps % args.val_frequency == 0:
            val_step(val_loader_vai, T_model, H_model, writer, prefix='vai', step=steps)
            val_step(val_loader_pot, T_model, H_model, writer, prefix='pot', step=steps)

        if steps % args.save_frequency == 0:
            os.makedirs(save_dir, exist_ok=True)
            torch.save(T_model.state_dict(), T_model_path)
            torch.save(D_model.state_dict(), D_model_path)
            torch.save(H_model.state_dict(), H_model_path)

        if steps % args.summary_frequency == 0:

            writer.add_scalar('loss/train_loss_H', sum(train_loss_H) / len(train_loss_H), global_step=steps)
            writer.add_scalar('loss/train_loss_T', sum(train_loss_T) / len(train_loss_T), global_step=steps)
            writer.add_scalar('loss/train_loss_D', sum(train_loss_D) / len(train_loss_D), global_step=steps)

            writer.add_image('train/images', train_images[0], global_step=steps)
            writer.add_image('train/labels_vis', vis(train_height[0]), global_step=steps)
            writer.add_image('train/P_height_vis', vis(H_out[0]), global_step=steps)
            writer.add_image('train/T_out', T_out[0], global_step=steps)
            writer.add_image('train/D_out', nn.functional.softmax(D_out, dim=1)[0], global_step=steps)

            current_time = datetime.now().strftime("%H:%M:%S")
            print(f"Steps {steps+1}/{args.epochs}.. "   
                  f"Train loss_T: {sum(train_loss_T)/len(train_loss_T):.5f}.. "
                  f"Train loss_D: {sum(train_loss_D)/len(train_loss_D):.5f}.. " 
                  f"Train loss_H: {sum(train_loss_H)/len(train_loss_H):.5f}.. " +
                  current_time)

            train_loss_T, train_loss_D, train_loss_H = [1], [1], [1]


if __name__ == '__main__':
    main()

