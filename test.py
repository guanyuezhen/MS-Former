import sys

sys.path.insert(0, '.')
import torch
import torchvision.transforms.functional as ttf
import scipy.io as scio
import torch.backends.cudnn as cudnn
from torch.nn.parallel import gather
import torch.optim.lr_scheduler
import datasets.dataset as myDataLoader
import datasets.Transforms as myTransforms
from utils.metric_tool import ConfuseMatrixMeter
from PIL import Image
import os, time
import numpy as np
from argparse import ArgumentParser
from models.model import get_model


@torch.no_grad()
def val(args, val_loader, model, vis_dir):
    model.eval()

    cd_evaluation = ConfuseMatrixMeter(n_class=2)

    total_batches = len(val_loader)
    print(len(val_loader))

    for iter, batched_inputs in enumerate(val_loader):

        img, patch_target, target = batched_inputs
        img_name = val_loader.sampler.data_source.file_list[iter]
        start_time = time.time()

        if args.onGPU == True:
            img = img.cuda()
            patch_target = patch_target.cuda()
            target = target.cuda()

        img_var = torch.autograd.Variable(img).float()
        patch_target_var = torch.autograd.Variable(patch_target).float()
        target_var = torch.autograd.Variable(target).float()  # only used for evaluation

        # run the model
        if args.file_root == 'LEVIR':
            B, C, H, W = img_var.size()
            change_mask = torch.ones(B, 1, H, W).cuda()
            for patch_h in range(0, H, 256):
                for patch_w in range(0, W, 256):
                    patch_mask = model(img_var[:, :, patch_h: patch_h + 256, patch_w: patch_w + 256])
                    change_mask[:, :, patch_h: patch_h + 256, patch_w: patch_w + 256] = patch_mask
        else:
            change_mask = model(img_var)

        pred = torch.where(change_mask > 0.5, torch.ones_like(change_mask), torch.zeros_like(change_mask)).long()

        # torch.cuda.synchronize()
        time_taken = time.time() - start_time

        # compute the confusion matrix
        if args.onGPU and torch.cuda.device_count() > 1:
            pred = gather(pred, 0, dim=0)

        # save change maps
        pr = pred[0, 0].cpu().numpy()
        gt = target_var[0, 0].cpu().numpy()
        index_tp = np.where(np.logical_and(pr == 1, gt == 1))
        index_fp = np.where(np.logical_and(pr == 1, gt == 0))
        index_tn = np.where(np.logical_and(pr == 0, gt == 0))
        index_fn = np.where(np.logical_and(pr == 0, gt == 1))
        #
        map = np.zeros([gt.shape[0], gt.shape[1], 3])
        map[index_tp] = [255, 255, 255]  # white
        map[index_fp] = [255, 0, 0]  # red
        map[index_tn] = [0, 0, 0]  # black
        map[index_fn] = [0, 255, 255]  # Cyan

        change_map = Image.fromarray(np.array(map, dtype=np.uint8))
        change_map.save(vis_dir + img_name)

        f1 = cd_evaluation.update_cm(pr, gt)

        if iter % 5 == 0:
            print('\r[%d/%d] F1: %3f time: %.3f' % (iter, total_batches, f1, time_taken),
                  end='')

    scores = cd_evaluation.get_scores()

    return scores


def val_change_detection(args):
    torch.backends.cudnn.benchmark = True
    SEED = 2023
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    model = get_model(args.patch_size, args.memory_length, args.depth)

    args.save_dir = args.save_dir + args.file_root + '_iter_' + str(args.max_steps) + '_lr_' + str(
        args.lr) + '_p_' + str(args.patch_size) + '_m_' + str(args.memory_length) + '_d_' + str(args.depth) + '/'

    args.vis_dir = './predict/' + args.file_root + '_patch_' + str(args.patch_size) + '/'

    if args.file_root == 'LEVIR':
        args.train_data_root = '/mnt/disk_d/Change Detection/Datasets_BCD_2/LEVIR'
        args.test_data_root = '/mnt/disk_d/Change Detection/Datasets_BCD_2/LEVIR-1024'
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
    valDataset = myTransforms.Compose([
        myTransforms.Normalize(mean=mean, std=std),
        myTransforms.Scale(args.inWidth, args.inHeight),
        myTransforms.ToTensor()
    ])

    test_data = myDataLoader.Dataset("test", file_root=args.test_data_root, transform=valDataset)
    testLoader = torch.utils.data.DataLoader(
        test_data, shuffle=False,
        batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=False)

    if args.onGPU:
        cudnn.benchmark = True

    logFileLoc = args.save_dir + args.logFile
    if os.path.isfile(logFileLoc):
        logger = open(logFileLoc, 'a')
    else:
        logger = open(logFileLoc, 'w')
        logger.write("Parameters: %s" % (str(total_params)))
        logger.write(
            "\n%s\t%s\t%s\t%s\t%s\t%s" % ('Epoch', 'Kappa', 'IoU', 'F1', 'R', 'P'))
    logger.flush()

    # load the model
    model_file_name = args.save_dir + 'best_model.pth'
    state_dict = torch.load(model_file_name)
    model.load_state_dict(state_dict)

    score_test = val(args, testLoader, model, args.vis_dir)
    torch.cuda.empty_cache()
    print("\nLEVIR_Test :\t Kappa (te) = %.4f\t IoU (te) = %.4f\t F1 (te) = %.4f\t R (te) = %.4f\t P (te) = %.4f" \
          % (score_test['Kappa'], score_test['IoU'], score_test['F1'], score_test['recall'], score_test['precision']))
    logger.write("\n%s\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f" % ('LEVIR_Test',
                                                                   score_test['Kappa'],
                                                                   score_test['IoU'],
                                                                   score_test['F1'],
                                                                   score_test['recall'],
                                                                   score_test['precision']))
    logger.flush()
    scio.savemat(args.vis_dir + 'results.mat', score_test)

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

    val_change_detection(args)
