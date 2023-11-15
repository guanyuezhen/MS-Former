import cv2
import numpy
import torch.utils.data


class Dataset(torch.utils.data.Dataset):
    '''
    Class to load the dataset
    '''

    def __init__(self, dataset, file_root='data/', transform=None, label_patch_size=None):
        """
        dataset: dataset name, e.g. NJU2K_NLPR_train
        file_root: root of data_path, e.g. ./data/
        """
        self.file_list = open(file_root + '/' + dataset + '/list/' + dataset + '.txt').read().splitlines()
        self.pre_images = [file_root + '/' + dataset + '/A/' + x for x in self.file_list]
        self.post_images = [file_root + '/' + dataset + '/B/' + x for x in self.file_list]
        if label_patch_size == 32:
            self.patch_gts = [file_root + '/' + dataset + '/label_32/' + x for x in self.file_list]
            self.gts = [file_root + '/' + dataset + '/label/' + x for x in self.file_list]
        elif label_patch_size == 64:
            self.patch_gts = [file_root + '/' + dataset + '/label_64/' + x for x in self.file_list]
            self.gts = [file_root + '/' + dataset + '/label/' + x for x in self.file_list]
        elif label_patch_size == 128:
            self.patch_gts = [file_root + '/' + dataset + '/label_128/' + x for x in self.file_list]
            self.gts = [file_root + '/' + dataset + '/label/' + x for x in self.file_list]
        elif label_patch_size == 256:
            self.patch_gts = [file_root + '/' + dataset + '/label_256/' + x for x in self.file_list]
            self.gts = [file_root + '/' + dataset + '/label/' + x for x in self.file_list]
        else:
            self.patch_gts = [file_root + '/' + dataset + '/label/' + x for x in self.file_list]
            self.gts = [file_root + '/' + dataset + '/label/' + x for x in self.file_list]
        self.transform = transform

    def __len__(self):
        return len(self.pre_images)

    def __getitem__(self, idx):
        pre_image_name = self.pre_images[idx]
        post_image_name = self.post_images[idx]
        patch_label_name = self.patch_gts[idx]
        label_name = self.gts[idx]
        pre_image = cv2.imread(pre_image_name)
        post_image = cv2.imread(post_image_name)
        patch_label = cv2.imread(patch_label_name, 0)
        label = cv2.imread(label_name, 0)
        img = numpy.concatenate((pre_image, post_image), axis=2)
        if self.transform:
            [img, patch_label, label] = self.transform(img, patch_label, label)

        return img, patch_label, label

    def get_img_info(self, idx):
        img = cv2.imread(self.pre_images[idx])
        return {"height": img.shape[0], "width": img.shape[1]}
