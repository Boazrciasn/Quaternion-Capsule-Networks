from __future__ import print_function
import torch.utils.data as data
from PIL import Image
import os
import os.path
import errno
import numpy as np
import torch
import torchvision.transforms.functional as TF
from torchvision import transforms
import struct


class NORB(data.Dataset):
    """`NORB <https://cs.nyu.edu/~ylclab/data/norb-v1.0/>`_ Dataset.
    The training set is composed of 5 instances of each category (instances 4, 6, 7, 8 and 9),
    and the test set of the remaining 5 instances (instances 0, 1, 2, 3, and 5).
    The image pairs were taken by two cameras. Labels and additional info are the same
    for a stereo image pair. More details about the data collection can be found in
    `http://leon.bottou.org/publications/pdf/cvpr-2004.pdf`.

    Args:
        root (string): Root directory of dataset where ``processed/training.pt``
            and  ``processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    raw_folder = 'raw'
    processed_folder = 'processed'
    training_file = 'training.pt'
    test_file = 'test.pt'
    types = ['dat', 'cat', 'info']
    urls = {}

    def __init__(self, root, train=True, download=False, input_size=32, mode='azimuth'):
        if len(self.urls) == 0:
            for k in self.types:
                self.urls['test_{}'.format(k)] = \
                         ['https://cs.nyu.edu/~ylclab/data/norb-v1.0/norb-5x01235x9x18x6x2x108x108-testing-{:02d}-{}.mat.gz' \
                         .format(x+1, k) for x in range(2)]
                self.urls['train_{}'.format(k)] = \
                         ['https://cs.nyu.edu/~ylclab/data/norb-v1.0/norb-5x46789x9x18x6x2x108x108-training-{:02d}-{}.mat.gz'\
                         .format(x+1, k) for x in range(10)]
        self.root = os.path.expanduser(root)
        self.train = train  # training set or test set
        self.input_size = input_size
        self.ColorJitter = transforms.ColorJitter(brightness=32. / 255, contrast=0.5)
        self.mode = mode
        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        if mode == 'default':
            # image pairs stored in [i, :, :] and [i+1, :, :]
            # they are sharing the same labels and info
            if self.train:
                self.train_data, self.train_labels, self.train_info = torch.load(
                    os.path.join(self.root, self.processed_folder, self.training_file))
                size = len(self.train_labels)
                assert size == len(self.train_info)
                assert size*2 == len(self.train_data)
                self.train_labels = self.train_labels.view(size, 1).repeat(1, 2).view(2*size, 1)
                self.train_info = self.train_info.repeat(1, 2).view(2*size, 4)
                self.train_data = self.train_data.reshape(24300, 2, 96, 96)
                self.train_labels = self.train_labels.reshape(24300, 2)[:, 0, None]
                self.train_info = self.train_info.reshape(24300, 2, 4)[:, 0, :]

            else:
                self.test_data, self.test_labels, self.test_info = torch.load(
                    os.path.join(self.root, self.processed_folder, self.test_file))
                size = len(self.test_labels)
                assert size == len(self.test_info)
                assert size*2 == len(self.test_data)
                self.test_labels = self.test_labels.view(size, 1).repeat(1, 2).view(2*size, 1)
                self.test_info = self.test_info.repeat(1, 2).view(2*size, 4)
                self.test_data = self.test_data.reshape(24300, 2, 96, 96)
                self.test_labels = self.test_labels.reshape(24300, 2)[:, 0, None]
                self.test_info = self.test_info.reshape(24300, 2, 4)[:, 0, :]

        elif mode == 'azimuth':
            train_data, train_labels, train_info = torch.load(
                os.path.join(root, self.processed_folder, self.training_file))
            size = len(train_labels)
            assert size == len(train_info)
            assert size * 2 == len(train_data)
            train_labels = train_labels.view(size, 1).repeat(1, 2).view(2 * size, 1)
            train_info = train_info.repeat(1, 2).view(2 * size, 4)
            train_data = train_data.reshape(24300, 2, 96, 96)
            train_labels = train_labels.reshape(24300, 2)[:, 0, None]
            train_info = train_info.reshape(24300, 2, 4)[:, 0, :]


            test_data, test_labels, test_info = torch.load(
                os.path.join(root, self.processed_folder, self.test_file))
            size = len(test_labels)
            assert size == len(test_info)
            assert size * 2 == len(test_data)
            test_labels = test_labels.view(size, 1).repeat(1, 2).view(2 * size, 1)
            test_info = test_info.repeat(1, 2).view(2 * size, 4)
            test_data = test_data.reshape(24300, 2, 96, 96)
            test_labels = test_labels.reshape(24300, 2)[:, 0, None]
            test_info = test_info.reshape(24300, 2, 4)[:, 0, :]

            total_set_data = torch.cat([train_data, test_data], dim=0)
            total_set_labels = torch.cat([train_labels, test_labels], dim=0)
            total_set_info = torch.cat([train_info, test_info], dim=0)

            if train:

                train_mask =    (total_set_info[:, 2] == 34) + (total_set_info[:, 2] == 32) + \
                                (total_set_info[:, 2] == 30) + (total_set_info[:, 2] == 0) + \
                                (total_set_info[:, 2] == 2) + (total_set_info[:, 2] == 4)



                self.train_data = total_set_data[train_mask]
                self.train_labels = total_set_labels[train_mask]
                self.train_info = total_set_info[train_mask]

            else:
                test_mask =     (total_set_info[:, 2] == 6) + (total_set_info[:, 2] == 8) + \
                                (total_set_info[:, 2] == 10) + (total_set_info[:, 2] == 12) + \
                                (total_set_info[:, 2] == 14) + (total_set_info[:, 2] == 16) + \
                                (total_set_info[:, 2] == 18) + (total_set_info[:, 2] == 20) + \
                                (total_set_info[:, 2] == 22) + (total_set_info[:, 2] == 24) + \
                                (total_set_info[:, 2] == 26) + (total_set_info[:, 2] == 28)

                self.test_data = total_set_data[test_mask]
                self.test_labels = total_set_labels[test_mask]
                self.test_info = total_set_info[test_mask]

        elif mode == 'elevation':

            train_data, train_labels, train_info = torch.load(
                os.path.join(root, self.processed_folder, self.training_file))
            size = len(train_labels)
            assert size == len(train_info)
            assert size * 2 == len(train_data)
            train_labels = train_labels.view(size, 1).repeat(1, 2).view(2 * size, 1)
            train_info = train_info.repeat(1, 2).view(2 * size, 4)
            train_data = train_data.reshape(24300, 2, 96, 96)
            train_labels = train_labels.reshape(24300, 2)[:, 0, None]
            train_info = train_info.reshape(24300, 2, 4)[:, 0, :]

            test_data, test_labels, test_info = torch.load(
                os.path.join(root, self.processed_folder, self.test_file))
            size = len(test_labels)
            assert size == len(test_info)
            assert size * 2 == len(test_data)
            test_labels = test_labels.view(size, 1).repeat(1, 2).view(2 * size, 1)
            test_info = test_info.repeat(1, 2).view(2 * size, 4)
            test_data = test_data.reshape(24300, 2, 96, 96)
            test_labels = test_labels.reshape(24300, 2)[:, 0, None]
            test_info = test_info.reshape(24300, 2, 4)[:, 0, :]

            total_set_data = torch.cat([train_data, test_data], dim=0)
            total_set_labels = torch.cat([train_labels, test_labels], dim=0)
            total_set_info = torch.cat([train_info, test_info], dim=0)

            if train:

                train_mask = (total_set_info[:, 1] == 0) + (total_set_info[:, 1] == 1) + \
                            (total_set_info[:, 1] == 2)


                self.train_data = total_set_data[train_mask]
                self.train_labels = total_set_labels[train_mask]
                self.train_info = total_set_info[train_mask]

            else:
                test_mask = (total_set_info[:, 1] == 3) + (total_set_info[:, 1] == 4) + \
                            (total_set_info[:, 1] == 5) + (total_set_info[:, 1] == 6) + \
                            (total_set_info[:, 1] == 7) + (total_set_info[:, 1] == 8)

                self.test_data = total_set_data[test_mask]
                self.test_labels = total_set_labels[test_mask]
                self.test_info = total_set_info[test_mask]

        elif mode == 'visualize':
            train_data, train_labels, train_info = torch.load(
                os.path.join(root, self.processed_folder, self.training_file))
            size = len(train_labels)
            assert size == len(train_info)
            assert size * 2 == len(train_data)
            train_labels = train_labels.view(size, 1).repeat(1, 2).view(2 * size, 1)
            train_info = train_info.repeat(1, 2).view(2 * size, 4)
            train_data = train_data.reshape(24300, 2, 96, 96)
            train_labels = train_labels.reshape(24300, 2)[:, 0, None]
            train_info = train_info.reshape(24300, 2, 4)[:, 0, :]


            test_data, test_labels, test_info = torch.load(
                os.path.join(root, self.processed_folder, self.test_file))
            size = len(test_labels)
            assert size == len(test_info)
            assert size * 2 == len(test_data)
            test_labels = test_labels.view(size, 1).repeat(1, 2).view(2 * size, 1)
            test_info = test_info.repeat(1, 2).view(2 * size, 4)
            test_data = test_data.reshape(24300, 2, 96, 96)
            test_labels = test_labels.reshape(24300, 2)[:, 0, None]
            test_info = test_info.reshape(24300, 2, 4)[:, 0, :]

            total_set_data = torch.cat([train_data, test_data], dim=0)
            total_set_labels = torch.cat([train_labels, test_labels], dim=0)
            total_set_info = torch.cat([train_info, test_info], dim=0)

            sorted_labels, index = total_set_labels.sort(0)
            sorted_info = torch.index_select(total_set_info, 0, index.flatten())
            sorted_data = torch.index_select(total_set_data, 0, index.flatten())
            total_info = torch.cat([total_set_labels.int(), total_set_info], dim=1)
            total_info, _ = total_info.sort(dim=0)

            info_0 = sorted_info[0:9719]
            data_0 = sorted_data[0:9719]
            label_0 = sorted_labels[0:9719]

            info_0_instance = info_0[:, 0]
            sorted_instance_0, inds = info_0_instance.sort()
            info_inst_0 = torch.index_select(info_0, dim=0, index=inds)[0:972]
            data_inst_0 = torch.index_select(data_0, dim=0, index=inds)[0:972]
            label_inst_0 = torch.index_select(label_0, dim=0, index=inds)[0:972]

            info_elev_0 = info_inst_0[:, 1]
            sorted_elev_0, inds = info_elev_0.sort()
            info_elev_0 = torch.index_select(info_inst_0, dim=0, index=inds)[0:108]
            data_elev_0 = torch.index_select(data_inst_0, dim=0, index=inds)[0:108]
            label_elev_0 = torch.index_select(label_inst_0, dim=0, index=inds)[0:108]

            info_light_0 = info_elev_0[:, 3]
            sorted_light_0, inds = info_light_0.sort()
            info_light_0 = torch.index_select(info_elev_0, dim=0, index=inds)[0:18]
            data_light_0 = torch.index_select(data_elev_0, dim=0, index=inds)[0:18]
            label_light_0 = torch.index_select(label_elev_0, dim=0, index=inds)[0:18]

            info_azm_0 = info_light_0[:, 2]
            sorted_azm_0, inds = info_azm_0.sort()
            info_0_azm_sorted = torch.index_select(info_light_0, dim=0, index=inds)
            data_0_azm_sorted = torch.index_select(data_light_0, dim=0, index=inds)
            label_0_azm_sorted = torch.index_select(label_light_0, dim=0, index=inds)


            self.train_data = data_0_azm_sorted
            self.train_labels = label_0_azm_sorted
            self.train_info = info_0_azm_sorted

            self.test_data = data_0_azm_sorted
            self.test_labels = label_0_azm_sorted
            self.test_info = info_0_azm_sorted


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Note that additional info is not used in this experiment.

        Returns:
            tuple: (image, target)
            where target is index of the target class and info contains
            ...
        """
        if self.train:
            img1, img2, target, info = Image.fromarray(self.train_data[index][0].numpy()), Image.fromarray(self.train_data[index][1].numpy()), int(self.train_labels[index]), self.train_info[index]
            img1 = TF.resize(img1, 48)
            img2 = TF.resize(img2, 48)

            if self.mode== 'visualize':
                img1 = TF.center_crop(img1, output_size=(self.input_size, self.input_size))
                img2 = TF.center_crop(img2, output_size=(self.input_size, self.input_size))
                img1 = transforms.ToTensor()(img1)
                img2 = transforms.ToTensor()(img2)
                img1 = transforms.Normalize(mean=[img1.mean()], std=[img1.std()])(img1)
                img2 = transforms.Normalize(mean=[img2.mean()], std=[img2.std()])(img2)
            else:
                i, j, h, w = transforms.RandomCrop.get_params(img1, output_size=(self.input_size, self.input_size))
                img1 = TF.crop(img1,  i, j, h, w)
                img2 = TF.crop(img2, i, j, h, w)

                img1 = transforms.ToTensor()(img1)
                img2 = transforms.ToTensor()(img2)
                img1 = transforms.Normalize(mean=[img1.mean()], std=[img1.std()])(img1)
                img2 = transforms.Normalize(mean=[img2.mean()], std=[img2.std()])(img2)

            sample = torch.cat((img1, img2), dim=0)

        else:
            img1, img2, target, info = Image.fromarray(self.test_data[index][0].numpy()), Image.fromarray(self.test_data[index][1].numpy()), self.test_labels[index], self.test_info[index]
            img1 = TF.resize(img1, 48)
            img2 = TF.resize(img2, 48)
            img1 = TF.center_crop(img1, output_size=(self.input_size, self.input_size))
            img2 = TF.center_crop(img2, output_size=(self.input_size, self.input_size))
            img1 = transforms.ToTensor()(img1)
            img2 = transforms.ToTensor()(img2)
            img1 = transforms.Normalize(mean=[img1.mean()], std=[img1.std()])(img1)
            img2 = transforms.Normalize(mean=[img2.mean()], std=[img2.std()])(img2)
            sample = torch.cat((img1, img2), dim=0)

        if self.mode == 'visualize':
            return sample, target, info
        else:
            return sample, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.processed_folder, self.training_file)) and \
            os.path.exists(os.path.join(self.root, self.processed_folder, self.test_file))

    def download(self):

        from six.moves import urllib
        import gzip

        if self._check_exists():
            return

        # download files
        try:
            os.makedirs(os.path.join(self.root, self.raw_folder))
            os.makedirs(os.path.join(self.root, self.processed_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        for k in self.urls:
            for url in self.urls[k]:
                print('Downloading ' + url)
                data = urllib.request.urlopen(url)
                filename = url.rpartition('/')[2]
                file_path = os.path.join(self.root, self.raw_folder, filename)
                with open(file_path, 'wb') as f:
                    f.write(data.read())
                with open(file_path.replace('.gz', ''), 'wb') as out_f, \
                        gzip.GzipFile(file_path) as zip_f:
                    out_f.write(zip_f.read())
                os.unlink(file_path)

        # process and save as torch files
        print('Processing...')

        parsed = {}
        for k in self.urls:
            op = get_op(k)
            for url in self.urls[k]:
                filename = url.rpartition('/')[2].replace('.gz', '')
                path = os.path.join(self.root, self.raw_folder, filename)
                print(path)
                if k not in parsed:
                    parsed[k] = op(path)
                else:
                    parsed[k] = torch.cat([parsed[k], op(path)], dim=0)

        training_set = (
            parsed['train_dat'],
            parsed['train_cat'],
            parsed['train_info']
        )
        test_set = (
            parsed['test_dat'],
            parsed['test_cat'],
            parsed['test_info']
        )
        with open(os.path.join(self.root, self.processed_folder, self.training_file), 'wb') as f:
            torch.save(training_set, f)
        with open(os.path.join(self.root, self.processed_folder, self.test_file), 'wb') as f:
            torch.save(test_set, f)

        print('Done!')

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


class smallNORB(NORB):
    """`smallNORB <https://cs.nyu.edu/%7Eylclab/data/norb-v1.0-small/>`_ Dataset.

    Args:
        root (string): Root directory of dataset where ``processed/training.pt``
            and  ``processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """
    urls = {
        'train_dat': ['https://cs.nyu.edu/%7Eylclab/data/norb-v1.0-small/smallnorb-5x46789x9x18x6x2x96x96-training-dat.mat.gz'],
        'train_cat': ['https://cs.nyu.edu/%7Eylclab/data/norb-v1.0-small/smallnorb-5x46789x9x18x6x2x96x96-training-cat.mat.gz'],
        'train_info': ['https://cs.nyu.edu/%7Eylclab/data/norb-v1.0-small/smallnorb-5x46789x9x18x6x2x96x96-training-info.mat.gz'],
        'test_dat': ['https://cs.nyu.edu/%7Eylclab/data/norb-v1.0-small/smallnorb-5x01235x9x18x6x2x96x96-testing-dat.mat.gz'],
        'test_cat': ['https://cs.nyu.edu/%7Eylclab/data/norb-v1.0-small/smallnorb-5x01235x9x18x6x2x96x96-testing-cat.mat.gz'],
        'test_info': ['https://cs.nyu.edu/%7Eylclab/data/norb-v1.0-small/smallnorb-5x01235x9x18x6x2x96x96-testing-info.mat.gz'],
    }


def magic2type(magic):
    m2t = {'1E3D4C51': 'single precision matrix',
           '1E3D4C52': 'packed matrix',
           '1E3D4C53': 'double precision matrix',
           '1E3D4C54': 'integer matrix',
           '1E3D4C55': 'byte matrix',
           '1E3D4C56': 'short matrix'}
    m = bytearray(reversed(magic)).hex().upper()
    return m2t[m]

def parse_header(fd):
    magic = struct.unpack('<BBBB', fd.read(4))
    ndim, = struct.unpack('<i', fd.read(4))
    dim = []
    for _ in range(ndim):
        dim += struct.unpack('<i', fd.read(4))

    header = {'magic': magic,
              'type': magic2type(magic),
              'dim': dim}
    return header

def parse_cat_file(path):
    """
        -cat file stores corresponding category of images
        Return:
            ByteTensor of shape (N,)
    """
    with open(path, 'rb') as f:
        header = parse_header(f)
        num, = header['dim']
        struct.unpack('<BBBB', f.read(4))
        struct.unpack('<BBBB', f.read(4))

        labels = np.zeros(shape=num, dtype=np.int32)
        for i in range(num):
            labels[i], = struct.unpack('<i', f.read(4))

        return torch.from_numpy(labels).long()

def parse_dat_file(path):
    """
        -dat file stores N image pairs. Each image pair,
        [i, :, :] and [i+1, :, :], includes two images
        taken from two cameras. They share the category
        and additional information.

        Return:
            ByteTensor of shape (2*N, 96, 96)
    """
    with open(path, 'rb') as f:
        header = parse_header(f)
        num, c, h, w = header['dim']
        imgs = np.zeros(shape=(num * c, h, w), dtype=np.uint8)

        for i in range(num * c):
            img = struct.unpack('<' + h * w * 'B', f.read(h * w))
            imgs[i] = np.uint8(np.reshape(img, newshape=(h, w)))

        return torch.from_numpy(imgs)

def parse_info_file(path):
    """
        -info file stores the additional info for each image.
        The specific meaning of each dimension is:

            (:, 0): 10 instances
            (:, 1): 9 elevation
            (:, 2): 18 azimuth
            (:, 3): 6 lighting conditions

        Return:
            ByteTensor of shape (N, 4)
    """
    with open(path, 'rb') as f:
        header = parse_header(f)
        num, num_info = header['dim']
        struct.unpack('<BBBB', f.read(4))
        info = np.zeros(shape=(num, num_info), dtype=np.int32)
        for r in range(num):
            for c in range(num_info):
                info[r, c], = struct.unpack('<i', f.read(4))

        return torch.from_numpy(info)

def get_op(key):
    op_dic = {
        'train_dat': parse_dat_file,
        'train_cat': parse_cat_file,
        'train_info': parse_info_file,
        'test_dat': parse_dat_file,
        'test_cat': parse_cat_file,
        'test_info': parse_info_file
    }
    return op_dic[key]



if __name__ == "__main__":

    import matplotlib.pyplot as plt
    num_class = 5
    path = "smallNORB/"
    Dataset_train = torch.utils.data.DataLoader(smallNORB(path, train=True, download=True, mode="visualize"),
                                                batch_size=4,
                                                shuffle=False)

    for batch_idx, (X, Y, info) in enumerate(Dataset_train):
        print(info)
