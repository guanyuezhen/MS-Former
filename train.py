import sys

sys.path.insert(0, '.')
import torch
import torchvision.transforms.functional as ttf
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.nn.parallel import gather
import torch.optim.lr_scheduler
import datasets.dataset as myDataLoader
import datasets.Transforms as myTransforms
from utils.metric_tool import ConfuseMatrixMeter
import os, time
import numpy as np
from argparse import ArgumentParser
from models.model import get_model
from utils.torchutils import make_numpy_grid, de_norm
import matplotlib.pyplot as plt


def seg_loss(pre, gts):
    bce = F.binary_cross_entropy(pre, gts)

    return bce


def reg_loss(pre, gts):
    reg = F.mse_loss(pre, gts)

    return reg


@torch.no_grad()
def val(args, val_loader, model):
    model.eval()

    cd_evaluation = ConfuseMatrixMeter(n_class=2)

    epoch_loss = []

    total_batches = len(val_loader)
    print(len(val_loader))
    for iter, batched_inputs in enumerate(val_loader):

        img, patch_target, target = batched_inputs

        start_time = time.time()

        if args.onGPU == True:
            img = img.cuda()
            patch_target = patch_target.cuda()
            target = target.cuda()

        img_var = torch.autograd.Variable(img).float()
        patch_target_var = torch.autograd.Variable(patch_target).float()
        target_var = torch.autograd.Variable(target).float()  # only used for evaluation

        # run the mdoel
        change_mask = model(img_var)

        pred = torch.where(change_mask > 0.5, torch.ones_like(change_mask), torch.zeros_like(change_mask)).long()

        #
        loss = F.binary_cross_entropy(change_mask, target_var)

        # torch.cuda.synchronize()
        time_taken = time.time() - start_time

        epoch_loss.append(loss.data.item())

        # compute the confusion matrix
        if args.onGPU and torch.cuda.device_count() > 1:
            output = gather(pred, 0, dim=0)
        # salEvalVal.addBatch(pred, target_var)
        f1 = cd_evaluation.update_cm(pr=pred.cpu().numpy(), gt=target_var.cpu().numpy())
        if iter % 5 == 0:
            print('\r[%d/%d] F1: %3f loss: %.3f time: %.3f' % (iter, total_batches, f1, loss.data.item(), time_taken),
                  end='')

    average_epoch_loss_val = sum(epoch_loss) / len(epoch_loss)
    scores = cd_evaluation.get_scores()

    return average_epoch_loss_val, scores


def train(args, train_loader, model, optimizer, epoch, max_batches, cur_iter=0, lr_factor=1.):
    # switch to train mode
    model.train()

    cd_evaluation = ConfuseMatrixMeter(n_class=2)
    epoch_loss = []

    for iter, batched_inputs in enumerate(train_loader):

        img, patch_target, target = batched_inputs
        #
        start_time = time.time()

        if args.onGPU == True:
            img = img.cuda()
            patch_target = patch_target.cuda()
            target = target.cuda()

        img_var = torch.autograd.Variable(img).float()
        patch_target_var = torch.autograd.Variable(patch_target).float()
        target_var = torch.autograd.Variable(target).float()  # only used for evaluation

        # adjust the learning rate
        lr = adjust_learning_rate(args, optimizer, iter + cur_iter, max_batches, lr_factor=lr_factor)

        # run the model
        change_mask, change_mask_aux, region_mask = model(img_var, patch_target_var)

        # loss
        size = img_var.size()[2:]
        target_var_down = F.adaptive_max_pool2d(patch_target_var,
                                                (size[0] // args.patch_size, size[1] // args.patch_size))

        seg_loss_1 = seg_loss(change_mask_aux, target_var_down)
        seg_loss_2 = torch.abs(change_mask - patch_target_var)
        seg_loss_2 = torch.mean(seg_loss_2[patch_target_var == 0])
        b, n, h, w = region_mask.size()
        seg_loss_3 = 0
        for i in range(n):
            seg_loss_3 += seg_loss(region_mask[:, i:i + 1, :, :], target_var_down)

        loss = seg_loss_1 + seg_loss_2 + seg_loss_3

        #
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss.append(loss.data.item())
        time_taken = time.time() - start_time
        res_time = (max_batches * args.max_epochs - iter - cur_iter) * time_taken / 3600

        #
        pred = torch.where(change_mask > 0.5, torch.ones_like(change_mask), torch.zeros_like(change_mask)).long()
        if args.onGPU and torch.cuda.device_count() > 1:
            output = gather(pred, 0, dim=0)

        # Computing F-measure and IoU on GPU
        with torch.no_grad():
            f1 = cd_evaluation.update_cm(pr=pred.cpu().numpy(), gt=target_var.cpu().numpy())

        if iter % 5 == 0:
            print('\riteration: [%d/%d] f1: %.3f lr: %.7f loss: %.3f time:%.3f h' % (
                iter + cur_iter, max_batches * args.max_epochs, f1, lr, loss.data.item(),
                res_time), end='')

        # if np.mod(iter, 200) == 1:
        #     vis_input = make_numpy_grid(de_norm(img_var[0:8, 0:3]))
        #     vis_input2 = make_numpy_grid(de_norm(img_var[0:8, 3:6]))
        #     vis_pred = make_numpy_grid(pred[0:8])
        #     vis_gt = make_numpy_grid(target_var[0:8])
        #     vis = np.concatenate([vis_input, vis_input2, vis_pred, vis_gt], axis=0)
        #     vis = np.clip(vis, a_min=0.0, a_max=1.0)
        #     file_name = os.path.join(
        #         args.vis_dir, 'train_' + str(epoch) + '_' + str(iter) + '.jpg')
        #     plt.imsave(file_name, vis)

    average_epoch_loss_train = sum(epoch_loss) / len(epoch_loss)
    scores = cd_evaluation.get_scores()

    return average_epoch_loss_train, scores, lr


def adjust_learning_rate(args, optimizer, iter, max_batches, lr_factor=1):
    max_iter = max_batches * args.max_epochs
    warm_up_iter = np.floor(max_iter * 0.1)
    if args.lr_mode == 'poly':
        cur_iter = iter - warm_up_iter
        max_iter = max_iter - warm_up_iter
        lr = args.lr * (1 - cur_iter * 1.0 / max_iter) ** 0.9
    else:
        raise ValueError('Unknown lr mode {}'.format(args.lr_mode))
    if iter < warm_up_iter:
        lr = args.lr * 0.9 * (iter + 1) / warm_up_iter + 0.1 * args.lr  # warm_up
    lr *= lr_factor
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def train_val_change_detection(args):
    torch.backends.cudnn.benchmark = True
    SEED = 2023
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    model = get_model(args.patch_size, args.memory_length, args.depth)

    args.save_dir = args.save_dir + args.file_root + '_iter_' + str(args.max_steps) + '_lr_' + str(
        args.lr) + '_p_' + str(args.patch_size) + '_m_' + str(args.memory_length) + '_d_' + str(args.depth) + '/'
    args.vis_dir = args.save_dir + '/Vis/'

    if args.file_root == 'LEVIR':
        args.train_data_root = '/mnt/disk_d/Change Detection/Datasets_BCD_2/LEVIR'
        args.test_data_root = '/mnt/disk_d/Change Detection/Datasets_BCD_2/LEVIR'
    elif args.file_root == 'GVLM':
        args.train_data_root = '/mnt/disk_d/Change Detection/Datasets_BCD_2/GVLM_In_Domain'
        args.test_data_root = '/mnt/disk_d/Change Detection/Datasets_BCD_2/GVLM_In_Domain'
    elif args.file_root == 'SYSU':
        args.train_data_root = '/mnt/disk_d/Change Detection/Datasets_BCD_2/SYSU'
        args.test_data_root = '/mnt/disk_d/Change Detection/Datasets_BCD_2/SYSU'
    elif args.file_root == 'BGMix':
        args.train_data_root = '/mnt/disk_d/Change Detection/Datasets_BCD_2/BCDD_BGMix'
        args.test_data_root = '/mnt/disk_d/Change Detection/Datasets_BCD_2/BCDD_BGMix'
    else:
        raise TypeError('%s has not defined' % args.file_root)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    if not os.path.exists(args.vis_dir):
        os.makedirs(args.vis_dir)

    if args.onGPU:
        model = model.cuda()

    total_params = sum([np.prod(p.size()) for p in model.parameters()])
    print('Total network parameters (excluding idr): ' + str(total_params))

    mean = [0.406, 0.456, 0.485, 0.406, 0.456, 0.485]
    std = [0.225, 0.224, 0.229, 0.225, 0.224, 0.229]

    # compose the data with transforms
    trainDataset_main = myTransforms.Compose([
        myTransforms.Normalize(mean=mean, std=std),
        myTransforms.Scale(args.inWidth, args.inHeight),
        myTransforms.RandomFlip(),
        myTransforms.RandomExchange(),
        myTransforms.ToTensor()
    ])

    valDataset = myTransforms.Compose([
        myTransforms.Normalize(mean=mean, std=std),
        myTransforms.Scale(args.inWidth, args.inHeight),
        myTransforms.ToTensor()
    ])

    train_data = myDataLoader.Dataset(
        "train", file_root=args.train_data_root,
        transform=trainDataset_main, label_patch_size=args.patch_size
    )
    trainLoader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=False, drop_last=True
    )

    test_data = myDataLoader.Dataset(
        "test", file_root=args.test_data_root, transform=valDataset
    )
    testLoader = torch.utils.data.DataLoader(
        test_data, shuffle=False,
        batch_size=args.batch_size // 4, num_workers=args.num_workers, pin_memory=False
    )

    max_batches = len(trainLoader)
    print('For each epoch, we have {} batches'.format(max_batches))

    if args.onGPU:
        cudnn.benchmark = True

    args.max_epochs = int(np.ceil(args.max_steps / max_batches))
    start_epoch = 0
    cur_iter = 0
    loss_lowest = 10

    logFileLoc = args.save_dir + args.logFile
    if os.path.isfile(logFileLoc):
        logger = open(logFileLoc, 'a')
    else:
        logger = open(logFileLoc, 'w')
        logger.write("Parameters: %s" % (str(total_params)))
        logger.write(
            "\n%s\t%s\t%s\t%s\t%s\t%s" % ('Epoch', 'Kappa (Tr)', 'IoU (Tr)', 'F1 (Tr)', 'R (Tr)', 'P (Tr)'))
    logger.flush()

    optimizer = torch.optim.Adam(model.parameters(), args.lr, (0.9, 0.99), eps=1e-08, weight_decay=1e-4)

    for epoch in range(start_epoch, args.max_epochs):
        lossTr, score_tr, lr = train(args, trainLoader, model, optimizer, epoch, max_batches, cur_iter)
        cur_iter += len(trainLoader)

        torch.cuda.empty_cache()

        logger.write("\n%d\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f" % (epoch, score_tr['Kappa'], score_tr['IoU'],
                                                                       score_tr['F1'], score_tr['recall'],
                                                                       score_tr['precision']))
        logger.flush()

        torch.save({
            'epoch': epoch + 1,
            'arch': str(model),
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lossTr': lossTr,
            'lossVal': lossTr,
            'F_Tr': score_tr['F1'],
            'lr': lr
        }, args.save_dir + 'checkpoint.pth.tar')

        # save the model
        model_file_name = args.save_dir + 'best_model.pth'
        if loss_lowest >= lossTr:
            loss_lowest = lossTr
            torch.save(model.state_dict(), model_file_name)

        print("Epoch " + str(epoch) + ': Details')
        print("\nEpoch No. %d:\tTrain Loss = %.4f\tF1(tr) = %.4f" \
              % (epoch, lossTr, score_tr['F1'])
              )

    # save the model of last epoch
    lost_model_file_name = args.save_dir + 'last_model.pth'
    torch.save(model.state_dict(), lost_model_file_name)
    #
    state_dict = torch.load(model_file_name)
    model.load_state_dict(state_dict)

    #
    loss_test, score_test = val(args, testLoader, model)
    torch.cuda.empty_cache()
    print("\nTest :\t Kappa (te) = %.4f\t IoU (te) = %.4f\t F1 (te) = %.4f\t R (te) = %.4f\t P (te) = %.4f" \
          % (score_test['Kappa'], score_test['IoU'], score_test['F1'], score_test['recall'],
             score_test['precision']))
    logger.write("\n%s\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f" % ('Test',
                                                                   score_test['Kappa'],
                                                                   score_test['IoU'],
                                                                   score_test['F1'],
                                                                   score_test['recall'],
                                                                   score_test['precision']))
    logger.flush()
    logger.close()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--file_root', default="LEVIR", help='Data directory')
    parser.add_argument('--inWidth', type=int, default=256, help='Width of RGB image')
    parser.add_argument('--inHeight', type=int, default=256, help='Height of RGB image')
    parser.add_argument('--patch_size', type=int, default=64, help='size of label patch')
    parser.add_argument('--memory_length', type=int, default=128, help='size of label patch')
    parser.add_argument('--depth', type=int, default=2, help='size of label patch')
    parser.add_argument('--max_steps', type=int, default=20000, help='Max. number of iterations')
    parser.add_argument('--num_workers', type=int, default=4, help='No. of parallel threads')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--lr', type=float, default=5e-4, help='Initial learning rate')
    parser.add_argument('--lr_mode', default='poly', help='Learning rate policy')
    parser.add_argument('--save_dir', default='./weights/', help='Directory to save the results')
    parser.add_argument('--logFile', default='trainValLog.txt',
                        help='File that stores the training and validation logs')
    parser.add_argument('--onGPU', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help='Run on CPU or GPU. If TRUE, then GPU.')
    parser.add_argument('--weight', default='', type=str, help='pretrained weight, can be a non-strict copy')
    parser.add_argument('--ms', type=int, default=0, help='apply multi-scale training, default False')

    args = parser.parse_args()
    print('Called with args:')
    print(args)

    train_val_change_detection(args)
