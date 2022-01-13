import os
import cv2
import numpy as np
import torch
import random
import math
import json
import collections
dirname = os.path.dirname(__file__)
dirname = os.path.dirname(dirname)
from torch.utils.data import Dataset
from utils.util import get_files
from torchvision import transforms


# ========================================
# Massachusetts Roads Dataset
# - Train: [sat, map]
# - Valid: [sat, map]
# - Test:  [sat, map]
# ========================================
class RoadDataset(Dataset):
    def __init__(self, data_dir, mode="train", ratio=0.5, augmentation=None, transform=None):
        assert mode in ['train', 'valid', 'test'], "wrong set:{}".format(mode)
        if mode == 'train':
            self.training = True
        else:
            self.training = False
        self.ratio = ratio
        self.augmentation = augmentation
        self.transform = transform
        self.base_path = os.path.join(dirname, data_dir)
        self.sat_ids = get_files(os.path.join(self.base_path, os.path.join(mode, 'sat/')), format='png')
        self.map_ids = get_files(os.path.join(self.base_path, os.path.join(mode, 'map/')), format='png')
        self.partial_ids = get_files(os.path.join(self.base_path, os.path.join(mode, 'map_erase/')), format='png')
        self.edge_ids = get_files(os.path.join(self.base_path, os.path.join(mode, 'edge/')), format='png')
        assert len(self.sat_ids) == len(self.map_ids) and len(self.sat_ids) == len(self.partial_ids) and len(self.sat_ids) == len(self.edge_ids), "lengths of satellite and map images are different"

    def __len__(self):
        return len(self.sat_ids)

    def __getitem__(self, index):
        sat_id = self.sat_ids[index]
        map_id = self.map_ids[index]
        partial_id = self.partial_ids[index]
        edge_id = self.edge_ids[index]

        # load image
        img_sat = cv2.imread(sat_id, cv2.IMREAD_COLOR)
        img_sat = cv2.cvtColor(img_sat, cv2.COLOR_BGR2RGB)

        img_map = cv2.imread(map_id, 0)
        _, img_map = cv2.threshold(img_map, 127, 255, cv2.THRESH_BINARY)

        img_partial = cv2.imread(partial_id, 0)
        _, img_partial = cv2.threshold(img_partial, 127, 255, cv2.THRESH_BINARY)

        img_edge = cv2.imread(edge_id, 0)
        _, img_edge = cv2.threshold(img_edge, 127, 255, cv2.THRESH_BINARY)

        if self.augmentation:
            sample = self.augmentation(image=img_sat, mask=img_map, mask_partial=img_partial, mask_edge=img_edge)
            img_sat, img_map, img_partial, img_edge = sample['image'], sample['mask'], sample['mask_partial'], sample['mask_edge']

        if self.transform:
            img_sat = self.transform(img_sat)
            img_map = self.transform(img_map)
            img_partial = self.transform(img_partial)
            img_edge = self.transform(img_edge)
        return img_sat, img_partial, img_map, img_edge


# ========================================
# DeepGlobe Roads Dataset
# - Train: [sat, map]
# - Valid: [sat] (no map, so merge valid and test to make a new test set and split train set for validation)
# - Test:  [sat]
# ========================================
class DeepGlobeDataset(Dataset):
    def __init__(self, data_dir, mode="train", augmentation=None, transform=None):
        assert mode in ['train', 'valid', 'test'], "wrong set:{}".format(mode)
        if mode == 'train':
            self.training = True
        else:
            self.training = False
        self.augmentation = augmentation
        self.transform = transform
        self.base_path = os.path.join(dirname, data_dir)
        self.mode = mode
        self.sat_ids = get_files(os.path.join(self.base_path, os.path.join(mode, 'sat/')), format='png')
        self.map_ids = get_files(os.path.join(self.base_path, os.path.join(mode, 'map/')), format='png')
        self.partial_ids = get_files(os.path.join(self.base_path, os.path.join(mode, 'map_erase/')), format='png')
        self.edge_ids = get_files(os.path.join(self.base_path, os.path.join(mode, 'edge/')), format='png')
        assert len(self.sat_ids) == len(self.map_ids) or len(self.map_ids) == 0, "lengths of satellite and map images are different"

    def __len__(self):
        return len(self.sat_ids)

    def __getitem__(self, index):
        sat_id = self.sat_ids[index]

        # load image
        img_sat = cv2.imread(sat_id, cv2.IMREAD_COLOR)
        img_sat = cv2.cvtColor(img_sat, cv2.COLOR_BGR2RGB)

        img_map = None
        img_partial = None
        img_edge = None
        if self.mode == 'train':
            map_id = self.map_ids[index]
            partial_id = self.partial_ids[index]
            edge_id = self.edge_ids[index]

            img_map = cv2.imread(map_id, 0)
            _, img_map = cv2.threshold(img_map, 127, 255, cv2.THRESH_BINARY)

            img_partial = cv2.imread(partial_id, 0)
            _, img_partial = cv2.threshold(img_partial, 127, 255, cv2.THRESH_BINARY)

            img_edge = cv2.imread(edge_id, 0)
            _, img_edge = cv2.threshold(img_edge, 127, 255, cv2.THRESH_BINARY)

            if self.augmentation:
                sample = self.augmentation(image=img_sat, mask=img_map, mask_partial=img_partial, mask_edge=img_edge)
                img_sat, img_map, img_partial, img_edge = sample['image'], sample['mask'], sample['mask_partial'], sample['mask_edge']

            if self.transform:
                img_sat = self.transform(img_sat)
                img_map = self.transform(img_map)
                img_partial = self.transform(img_partial)
                img_edge = self.transform(img_edge)
        else:
            if self.transform:
                img_sat = self.transform(img_sat)
        return img_sat, img_partial, img_map, img_edge


# ========================================
# SpaceNet Road Dataset
# - Train: [sat, map]
# - Valid: [sat, map]
# - Test:  [sat, map]
# ========================================
class SpaceNetDataset(Dataset):
    def __init__(self, data_dir, mode="train", ratio=0.75, augmentation=None, transform=None):
        assert ratio in [0.0, 0.25, 0.50, 0.75, 'mix']
        if mode == 'train':
            self.training = True
        else:
            self.training = False
        self.ratio = ratio
        self.augmentation = augmentation
        self.transform = transform
        self.base_path = os.path.join(dirname, data_dir)

        vegas_path = os.path.join(self.base_path, 'Vegas/')
        paris_path = os.path.join(self.base_path, 'Paris/')
        shanghai_path = os.path.join(self.base_path, 'Shanghai/')
        khartoum_path = os.path.join(self.base_path, 'Khartoum/')
        sat_vegas = get_files(os.path.join(vegas_path, 'sats/'), format='png')
        sat_paris = get_files(os.path.join(paris_path, 'sats/'), format='png')
        sat_shanghai = get_files(os.path.join(shanghai_path, 'sats/'), format='png')
        sat_khartoum = get_files(os.path.join(khartoum_path, 'sats/'), format='png')
        map_vegas = get_files(os.path.join(vegas_path, 'maps/'), format='png')
        map_paris = get_files(os.path.join(paris_path, 'maps/'), format='png')
        map_shanghai = get_files(os.path.join(shanghai_path, 'maps/'), format='png')
        map_khartoum = get_files(os.path.join(khartoum_path, 'maps/'), format='png')
        self.sat_ids = sat_vegas
        self.sat_ids.extend(sat_paris)
        self.sat_ids.extend(sat_shanghai)
        self.sat_ids.extend(sat_khartoum)
        self.map_ids = map_vegas
        self.map_ids.extend(map_paris)
        self.map_ids.extend(map_shanghai)
        self.map_ids.extend(map_khartoum)

        # never mind we set ratio to 0.75 here, when ratio is 0, we would create an empty mask later
        if ratio == 'mix':
            self.partial_ids = []
            par25_vegas = get_files(os.path.join(vegas_path, 'maps_{}/'.format(int(100 * 0.25))), format='png')
            par25_paris = get_files(os.path.join(paris_path, 'maps_{}/'.format(int(100 * 0.25))), format='png')
            par25_shanghai = get_files(os.path.join(shanghai_path, 'maps_{}/'.format(int(100 * 0.25))), format='png')
            par25_khartoum = get_files(os.path.join(khartoum_path, 'maps_{}/'.format(int(100 * 0.25))), format='png')
            par50_vegas = get_files(os.path.join(vegas_path, 'maps_{}/'.format(int(100 * 0.5))), format='png')
            par50_paris = get_files(os.path.join(paris_path, 'maps_{}/'.format(int(100 * 0.5))), format='png')
            par50_shanghai = get_files(os.path.join(shanghai_path, 'maps_{}/'.format(int(100 * 0.5))), format='png')
            par50_khartoum = get_files(os.path.join(khartoum_path, 'maps_{}/'.format(int(100 * 0.5))), format='png')
            par75_vegas = get_files(os.path.join(vegas_path, 'maps_{}/'.format(int(100 * 0.75))), format='png')
            par75_paris = get_files(os.path.join(paris_path, 'maps_{}/'.format(int(100 * 0.75))), format='png')
            par75_shanghai = get_files(os.path.join(shanghai_path, 'maps_{}/'.format(int(100 * 0.75))), format='png')
            par75_khartoum = get_files(os.path.join(khartoum_path, 'maps_{}/'.format(int(100 * 0.75))), format='png')
            for i in range(len(par25_vegas)):
                self.partial_ids.append([par25_vegas[i], par50_vegas[i], par75_vegas[i]])
            for i in range(len(par25_paris)):
                self.partial_ids.append([par25_paris[i], par50_paris[i], par75_paris[i]])
            for i in range(len(par25_shanghai)):
                self.partial_ids.append([par25_shanghai[i], par50_shanghai[i], par75_shanghai[i]])
            for i in range(len(par25_khartoum)):
                self.partial_ids.append([par25_khartoum[i], par50_khartoum[i], par75_khartoum[i]])
            # 0 => 25%
            # 1 => 50%
            # 2 => 75%
            self.random_pars = np.random.randint(0, 3, len(self.partial_ids))

            # save the mix information
            mix_info_file = os.path.join(self.base_path, "mix_info.json")
            assert len(self.partial_ids) == len(self.random_pars)
            if not os.path.exists(mix_info_file):
                print("mix dataset information does not exist, create one...")
                info = {}
                for _, (partial_id_3, idx) in enumerate(zip(self.partial_ids, self.random_pars)):
                    file_name = partial_id_3[idx].split("spacenet\\")[-1].replace("/", "\\")
                    splits = file_name.split("\\")
                    file_name = "{}\\{}".format(splits[0], splits[2])
                    if file_name not in info:
                        partial = 25
                        if idx == 0:
                            partial = 25
                        elif idx == 1:
                            partial = 50
                        elif idx == 2:
                            partial = 75
                        info[file_name] = partial
                    else:
                        print("Duplicate...")
                with open(mix_info_file, "w") as f:
                    json.dump(info, f)
        else:
            if ratio == 0.0:
                ratio = 0.75
            par_vegas = get_files(os.path.join(vegas_path, 'maps_{}/'.format(int(100 * ratio))), format='png')
            par_paris = get_files(os.path.join(paris_path, 'maps_{}/'.format(int(100 * ratio))), format='png')
            par_shanghai = get_files(os.path.join(shanghai_path, 'maps_{}/'.format(int(100 * ratio))), format='png')
            par_khartoum = get_files(os.path.join(khartoum_path, 'maps_{}/'.format(int(100 * ratio))), format='png')
            self.partial_ids = par_vegas
            self.partial_ids.extend(par_paris)
            self.partial_ids.extend(par_shanghai)
            self.partial_ids.extend(par_khartoum)
        assert len(self.sat_ids) == len(self.map_ids) and len(self.sat_ids) == len(self.partial_ids), "lengths of satellite and map images are different"

    def __len__(self):
        return len(self.sat_ids)

    def __getitem__(self, index):
        sat_id = self.sat_ids[index]
        map_id = self.map_ids[index]
        partial_id = self.partial_ids[index]

        # load image
        img_sat = cv2.imread(sat_id, cv2.IMREAD_COLOR)
        img_sat = cv2.cvtColor(img_sat, cv2.COLOR_BGR2RGB)

        img_map = cv2.imread(map_id, 0)
        _, img_map = cv2.threshold(img_map, 127, 255, cv2.THRESH_BINARY)

        # create an all-zero partial map when the ratio is 0
        if self.ratio == 0.0:
            img_partial = np.zeros(img_map.shape, dtype=np.float32)
        elif self.ratio == 'mix':
            partial_id = partial_id[self.random_pars[index]]
            img_partial = cv2.imread(partial_id, 0)
            _, img_partial = cv2.threshold(img_partial, 127, 255, cv2.THRESH_BINARY)
        else:
            img_partial = cv2.imread(partial_id, 0)
            _, img_partial = cv2.threshold(img_partial, 127, 255, cv2.THRESH_BINARY)

        if self.augmentation:
            sample = self.augmentation(image=img_sat, mask=img_map, mask_partial=img_partial)
            img_sat, img_map, img_partial = sample['image'], sample['mask'], sample['mask_partial']

        if self.transform:
            img_sat = self.transform(img_sat)
            img_map = self.transform(img_map)
            img_partial = self.transform(img_partial)
        return img_sat, img_partial, img_map, sat_id


# ========================================
# Test OSM Dataset
# - Test:  [sat, osm_map, spacenet_map]
# ========================================
class TestOSMDataset(Dataset):
    def __init__(self, data_dir, partial='osm', augmentation=None, transform=None):
        assert partial in ['osm', 'spacenet'], "the partial map should be osm or spacenet"
        self.training = False
        self.augmentation = augmentation
        self.transform = transform
        self.base_path = os.path.join(dirname, data_dir)

        sat_ids = get_files(os.path.join(self.base_path, 'sats/'), format='png')
        osm_ids = get_files(os.path.join(self.base_path, 'maps_osm/' if partial == 'osm' else 'maps_spacenet_partial'), format='png')
        spacenet_ids = get_files(os.path.join(self.base_path, 'maps_spacenet/'), format='png')

        self.sat_ids = sat_ids
        self.osm_ids = osm_ids
        self.spacenet_ids = spacenet_ids
        assert len(self.sat_ids) == len(self.osm_ids) and len(self.sat_ids) == len(self.spacenet_ids), "lengths of satellite and map images are different"

    def __len__(self):
        return len(self.sat_ids)

    def __getitem__(self, index):
        sat_id = self.sat_ids[index]
        osm_id = self.osm_ids[index]
        spacenet_id = self.spacenet_ids[index]

        # load image
        img_sat = cv2.imread(sat_id, cv2.IMREAD_COLOR)
        img_sat = cv2.cvtColor(img_sat, cv2.COLOR_BGR2RGB)

        img_osm = cv2.imread(osm_id, 0)
        _, img_osm = cv2.threshold(img_osm, 127, 255, cv2.THRESH_BINARY)

        img_spacenet = cv2.imread(spacenet_id, 0)
        _, img_spacenet = cv2.threshold(img_spacenet, 127, 255, cv2.THRESH_BINARY)

        if self.augmentation:
            sample = self.augmentation(image=img_sat, mask=img_spacenet, mask_partial=img_osm)
            img_sat, img_spacenet, img_osm = sample['image'], sample['mask'], sample['mask_partial']

        if self.transform:
            img_sat = self.transform(img_sat)
            img_osm = self.transform(img_osm)
            img_spacenet = self.transform(img_spacenet)
        return img_sat, img_osm, img_spacenet


# ========================================
# Road Dataset (Road Connectivity)
# - Train: [sat, map]
# - Valid: [sat, map]
# - Test:  [sat, map]
# ========================================
class RoadConnectivityRoadDataset(Dataset):
    def __init__(self, config, dataset_name, seed=7, multi_scale_pred=False, is_train=True):
        # Seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)

        self.threshold = config["thresh"]

        self.split = config["mode"]
        self.config = config
        # paths
        self.dir = self.config[dataset_name]["dir"]

        self.img_root = os.path.join(self.dir, "images/")
        self.gt_root = os.path.join(self.dir, "gt/")
        self.image_list = self.config[dataset_name]["file"]

        # list of all images
        self.images = [line.rstrip("\n") for line in open(self.image_list)]
        num_images = len(self.images)
        val_test_ratio = 0.5
        val_num = int(num_images * val_test_ratio)
        if self.split == "val":
            self.images = self.images[:val_num]
        elif self.split == "test":
            self.images = self.images[val_num:]

        # augmentations
        self.augmentation = self.config["augmentation"]
        self.crop_size = [
            self.config[dataset_name]["crop_size"],
            self.config[dataset_name]["crop_size"],
        ]
        self.multi_scale_pred = multi_scale_pred

        # preprocess
        self.angle_theta = self.config["angle_theta"]
        self.mean_bgr = np.array(eval(self.config["mean"]))
        self.deviation_bgr = np.array(eval(self.config["std"]))
        self.normalize_type = self.config["normalize_type"]

        # to avoid Deadloack  between CV Threads and Pytorch Threads caused in resizing
        cv2.setNumThreads(0)

        self.files = collections.defaultdict(list)
        for f in self.images:
            self.files[self.split].append(
                {
                    "img": self.img_root
                           + f
                           + self.config[dataset_name]["image_suffix"],
                    "lbl": self.gt_root + f + self.config[dataset_name]["gt_suffix"],
                }
            )

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.files[self.split])

    def getRoadData(self, index):
        image_dict = self.files[self.split][index]
        # read each image in list
        if os.path.isfile(image_dict["img"]):
            image = cv2.imread(image_dict["img"]).astype(np.float)
        else:
            print("ERROR: couldn't find image -> ", image_dict["img"])

        if os.path.isfile(image_dict["lbl"]):
            gt = cv2.imread(image_dict["lbl"], 0).astype(np.float)
        else:
            print("ERROR: couldn't find image -> ", image_dict["lbl"])

        if self.split == "train":
            image, gt = self.random_crop(image, gt, self.crop_size)
        else:
            # image = cv2.resize(
            #     image,
            #     (self.crop_size[0], self.crop_size[1]),
            #     interpolation=cv2.INTER_LINEAR,
            # )
            # gt = cv2.resize(
            #     gt,
            #     (self.crop_size[0], self.crop_size[1]),
            #     interpolation=cv2.INTER_LINEAR,
            # )
            image, gt = self.center_crop(image, gt, self.crop_size)

        if self.split == "train" and index == len(self.files[self.split]) - 1:
            np.random.shuffle(self.files[self.split])

        h, w, c = image.shape
        if self.augmentation == 1:
            flip = np.random.choice(2) * 2 - 1
            image = np.ascontiguousarray(image[:, ::flip, :])
            gt = np.ascontiguousarray(gt[:, ::flip])
            rotation = np.random.randint(4) * 90
            M = cv2.getRotationMatrix2D((w / 2, h / 2), rotation, 1)
            image = cv2.warpAffine(image, M, (w, h))
            gt = cv2.warpAffine(gt, M, (w, h))

        # image = self.reshape(image)
        image = self.transform(np.uint8(image))
        gt = self.transform(np.uint8(gt))
        gt[gt < self.threshold] = 0
        gt[gt >= self.threshold] = 1
        return image, gt

    def getCorruptRoad(self, road_gt, height, width, artifacts_shape="linear", element_counts=8):
        # False Negative Mask
        FNmask = np.ones((height, width), np.float)
        # False Positive Mask
        FPmask = np.zeros((height, width), np.float)
        indices = np.where(road_gt == 1)

        if artifacts_shape == "square":
            shapes = [[16, 16], [32, 32]]
            ##### FNmask
            if len(indices[0]) == 0:  ### no road pixel in GT
                pass
            else:
                for c_ in range(element_counts):
                    c = np.random.choice(len(shapes), 1)[
                        0
                    ]  ### choose random square size
                    shape_ = shapes[c]
                    ind = np.random.choice(len(indices[0]), 1)[
                        0
                    ]  ### choose a random road pixel as center for the square
                    row = indices[0][ind]
                    col = indices[1][ind]

                    FNmask[
                    row - shape_[0] / 2: row + shape_[0] / 2,
                    col - shape_[1] / 2: col + shape_[1] / 2,
                    ] = 0
            #### FPmask
            for c_ in range(element_counts):
                c = np.random.choice(len(shapes), 2)[0]  ### choose random square size
                shape_ = shapes[c]
                row = np.random.choice(height - shape_[0] - 1, 1)[
                    0
                ]  ### choose random pixel
                col = np.random.choice(width - shape_[1] - 1, 1)[
                    0
                ]  ### choose random pixel
                FPmask[
                row - shape_[0] / 2: row + shape_[0] / 2,
                col - shape_[1] / 2: col + shape_[1] / 2,
                ] = 1

        elif artifacts_shape == "linear":
            ##### FNmask
            if len(indices[0]) == 0:  ### no road pixel in GT
                pass
            else:
                for c_ in range(element_counts):
                    c1 = np.random.choice(len(indices[0]), 1)[
                        0
                    ]  ### choose random 2 road pixels to draw a line
                    c2 = np.random.choice(len(indices[0]), 1)[0]
                    cv2.line(
                        FNmask,
                        (indices[1][c1], indices[0][c1]),
                        (indices[1][c2], indices[0][c2]),
                        0,
                        self.angle_theta * 2,
                    )
            #### FPmask
            for c_ in range(element_counts):
                row1 = np.random.choice(height, 1)
                col1 = np.random.choice(width, 1)
                row2, col2 = (
                    row1 + np.random.choice(50, 1),
                    col1 + np.random.choice(50, 1),
                )
                cv2.line(FPmask, (col1, row1), (col2, row2), 1, self.angle_theta * 2)

        erased_gt = (road_gt * FNmask) + FPmask
        erased_gt[erased_gt > 0] = 1

        return erased_gt

    def reshape(self, image):
        if self.normalize_type == "Std":
            image = (image - self.mean_bgr) / (3 * self.deviation_bgr)
        elif self.normalize_type == "MinMax":
            image = (image - self.min_bgr) / (self.max_bgr - self.min_bgr)
            image = image * 2 - 1
        elif self.normalize_type == "Mean":
            image -= self.mean_bgr
        else:
            image = (image / 255.0) * 2 - 1

        image = image.transpose(2, 0, 1)
        return image

    def random_crop(self, image, gt, size):
        w, h, _ = image.shape
        crop_h, crop_w = size

        start_x = np.random.randint(0, w - crop_w)
        start_y = np.random.randint(0, h - crop_h)

        image = image[start_x: start_x + crop_w, start_y: start_y + crop_h, :]
        gt = gt[start_x: start_x + crop_w, start_y: start_y + crop_h]

        return image, gt

    def center_crop(self, image, gt, size):
        w, h, _ = image.shape
        crop_h, crop_w = size

        start_x = int((w - crop_w) / 2)
        start_y = int((h - crop_h) / 2)

        image = image[start_x: start_x + crop_w, start_y: start_y + crop_h, :]
        gt = gt[start_x: start_x + crop_w, start_y: start_y + crop_h]

        return image, gt


# ========================================
# SpaceNet Road Dataset (Road Connectivity)
# - Train: [sat, map]
# - Valid: [sat, map]
# - Test:  [sat, map]
# ========================================
class SpaceNetRoadConnectivityDataset(RoadConnectivityRoadDataset):
    def __init__(self, config, multi_scale_pred=False, is_train=True):
        super(SpaceNetRoadConnectivityDataset, self).__init__(
            config, "spacenet", multi_scale_pred, is_train
        )

    def __getitem__(self, index):
        image, gt = self.getRoadData(index)
        return image, gt, gt, gt


# ========================================
# OSM Road Dataset
# - Train: [sat, map]
# - Valid: [sat, map]
# - Test:  [sat, map]
# ========================================
class OSMDataset(Dataset):
    def __init__(self, data_dir, mode="train", file_list="../data/osm/train.txt", ratio=0.75, augmentation=None, transform=None):
        assert ratio in [0.0, 0.25, 0.50, 0.75, 'mix']
        if mode == 'train':
            self.training = True
        else:
            self.training = False
        self.ratio = ratio
        self.augmentation = augmentation
        self.transform = transform
        self.base_path = os.path.join(dirname, data_dir)
        mix_info_file = os.path.join(self.base_path, "mix_info.json")

        self.file_list = file_list
        if not file_list:
            sat_ids = get_files(os.path.join(self.base_path, 'imagery/'), format='png')
            map_ids = get_files(os.path.join(self.base_path, 'masks/'), format='png')
            self.sat_ids = sat_ids
            self.map_ids = map_ids

            # never mind we set ratio to 0.75 here, when ratio is 0, we would create an empty mask later
            if ratio == 'mix':
                self.partial_ids = []
                par25_ids = get_files(os.path.join(self.base_path, 'masks_{}/'.format(int(100 * 0.25))), format='png')
                par50_ids = get_files(os.path.join(self.base_path, 'masks_{}/'.format(int(100 * 0.5))), format='png')
                par75_ids = get_files(os.path.join(self.base_path, 'masks_{}/'.format(int(100 * 0.75))), format='png')
                for i in range(len(par25_ids)):
                    self.partial_ids.append([par25_ids[i], par50_ids[i], par75_ids[i]])
                # 0 => 25%
                # 1 => 50%
                # 2 => 75%
                self.random_pars = np.random.randint(0, 3, len(self.partial_ids))

                # save the mix information
                assert len(self.partial_ids) == len(self.random_pars)
                if not os.path.exists(mix_info_file):
                    print("mix dataset information does not exist, create one...")
                    info = {}
                    for _, (partial_id_3, idx) in enumerate(zip(self.partial_ids, self.random_pars)):
                        file_name = partial_id_3[idx].split("osm\\")[-1].replace("/", "\\")
                        splits = file_name.split("\\")
                        file_name = "{}\\{}".format(splits[0], splits[2])
                        if file_name not in info:
                            partial = 25
                            if idx == 0:
                                partial = 25
                            elif idx == 1:
                                partial = 50
                            elif idx == 2:
                                partial = 75
                            info[file_name] = partial
                        else:
                            print("Duplicate...")
                    with open(mix_info_file, "w") as f:
                        json.dump(info, f)
            else:
                if ratio == 0.0:
                    ratio = 0.75
                par_ids = get_files(os.path.join(self.base_path, 'masks_{}/'.format(int(100 * ratio))), format='png')
                self.partial_ids = par_ids
        else:
            self.image_ids = [line.rstrip("\n") for line in open(file_list)]
            self.sat_ids = []
            self.map_ids = []
            self.partial_ids = []
            if ratio == "mix":
                with open(mix_info_file, "r") as f:
                    mix_infos = json.load(f)
                    for image in self.image_ids:
                        image = image.replace("\\", "/")
                        mask = image.replace("imagery", "masks")
                        self.sat_ids.append(os.path.join(self.base_path, image))
                        self.map_ids.append(os.path.join(self.base_path, mask))

                        image_name = image.replace("imagery/", "")
                        partial = mix_infos[image_name]
                        par = image.replace("imagery", "masks_{}".format(partial))
                        self.partial_ids.append(os.path.join(self.base_path, par))
            else:
                if ratio == 0.0:
                    ratio = 0.75
                for image in self.image_ids:
                    image = image.replace("\\", "/")
                    mask = image.replace("imagery", "masks")
                    par = image.replace("imagery", "masks_{}".format(int(100 * ratio)))
                    self.sat_ids.append(os.path.join(self.base_path, image))
                    self.map_ids.append(os.path.join(self.base_path, mask))
                    self.partial_ids.append(os.path.join(self.base_path, par))

        assert len(self.sat_ids) == len(self.map_ids) and len(self.sat_ids) == len(self.partial_ids), "lengths of satellite and map images are different"

    def __len__(self):
        return len(self.sat_ids)

    def __getitem__(self, index):
        sat_id = self.sat_ids[index]
        map_id = self.map_ids[index]
        partial_id = self.partial_ids[index]

        # load image
        img_sat = cv2.imread(sat_id, cv2.IMREAD_COLOR)
        img_sat = cv2.cvtColor(img_sat, cv2.COLOR_BGR2RGB)

        img_map = cv2.imread(map_id, 0)
        _, img_map = cv2.threshold(img_map, 127, 255, cv2.THRESH_BINARY)

        # create an all-zero partial map when the ratio is 0
        if self.ratio == 0.0:
            img_partial = np.zeros(img_map.shape, dtype=np.float32)
        elif self.ratio == 'mix':
            if not self.file_list:
                partial_id = partial_id[self.random_pars[index]]
            img_partial = cv2.imread(partial_id, 0)
            _, img_partial = cv2.threshold(img_partial, 127, 255, cv2.THRESH_BINARY)
        else:
            img_partial = cv2.imread(partial_id, 0)
            _, img_partial = cv2.threshold(img_partial, 127, 255, cv2.THRESH_BINARY)

        if self.augmentation:
            sample = self.augmentation(image=img_sat, mask=img_map, mask_partial=img_partial)
            img_sat, img_map, img_partial = sample['image'], sample['mask'], sample['mask_partial']

        if self.transform:
            img_sat = self.transform(img_sat)
            img_map = self.transform(img_map)
            img_partial = self.transform(img_partial)
        return img_sat, img_partial, img_map, sat_id
