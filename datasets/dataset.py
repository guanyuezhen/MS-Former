import cv2
import numpy as np
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
        self.label_patch_size = label_patch_size
        self.gts = [file_root + '/' + dataset + '/label/' + x for x in self.file_list]
        self.transform = transform

    def __len__(self):
        return len(self.pre_images)
    
    @staticmethod
    def crop_image_into_patches(image, patch_size):
        image = np.ceil(image / 255)
        width, height = image.shape # height = width
        # Reshape the image into image_patches
        image_patches = image.reshape(width//patch_size, patch_size, -1, patch_size).swapaxes(1,2)
        # Check if there is any 1 in each patch
        change_exists = np.any(image_patches, axis=(2,3))
        # Set elements to 1 in each patch where 1 exists
        new_image = np.repeat(np.repeat(change_exists, patch_size, axis=0), patch_size, axis=1).astype(int)
        new_image = new_image * 255

        return new_image

    def __getitem__(self, idx):
        pre_image_name = self.pre_images[idx]
        post_image_name = self.post_images[idx]
        label_name = self.gts[idx]
        pre_image = cv2.imread(pre_image_name)
        post_image = cv2.imread(post_image_name)
        label = cv2.imread(label_name, 0)
        patch_label = self.crop_image_into_patches(label, self.label_patch_size)
        img = np.concatenate((pre_image, post_image), axis=2)
        if self.transform:
            [img, patch_label, label] = self.transform(img, patch_label, label)

        return img, patch_label, label

    def get_img_info(self, idx):
        img = cv2.imread(self.pre_images[idx])
        return {"height": img.shape[0], "width": img.shape[1]}
