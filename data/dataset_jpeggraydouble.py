import random
import numpy as np
import torch
import torch.utils.data as data
import utils.utils_image as util
import cv2

class DatasetJPEG(data.Dataset):

    def __init__(self, opt):
        super(DatasetJPEG, self).__init__()
        self.opt = opt
        self.n_channels = opt['n_channels'] if opt['n_channels'] else 3
        self.patch_size = self.opt['H_size'] if opt['H_size'] else 64

        # -------------------------------------
        # get the path of H, return None if input is None
        # -------------------------------------
        self.paths_H = util.get_image_paths(opt['dataroot_H'])

        self.count = 0
        self.use_double = False

    def __getitem__(self, index):
        # -------------------------------------
        # get H image
        # -------------------------------------
        H_path = self.paths_H[index]
        img_H = util.imread_uint(H_path, self.n_channels)

        L_path = H_path

        if self.opt['phase'] == 'train':
            """
            # --------------------------------
            # get L/H/M patch pairs
            # --------------------------------
            """
            H, W = img_H.shape[:2]
            self.patch_size_plus16 = self.patch_size+16
            self.patch_size_plus8 = self.patch_size+8
            # ---------------------------------
            # randomly crop the patch
            # ---------------------------------
            rnd_h = random.randint(0, max(0, H - self.patch_size_plus16))
            rnd_w = random.randint(0, max(0, W - self.patch_size_plus16))
            patch_H = img_H[rnd_h:rnd_h + self.patch_size_plus16, rnd_w:rnd_w + self.patch_size_plus16, :]

            # ---------------------------------
            # augmentation - flip, rotate
            # ---------------------------------
            mode = random.randint(0, 7)
            patch_H = util.augment_img(patch_H, mode=mode)
#            if random.random() > 0.5:
#                patch_H = self.jitter(util.uint2tensor4(patch_H))
#                patch_H = util.tensor2uint(patch_H)

            # ---------------------------------
            # HWC to CHW, numpy(uint) to tensor
            # ---------------------------------
#            img_H = util.uint2tensor3(patch_H)
#            img_L = img_H.clone()
            img_L = patch_H.copy()
            

            # ---------------------------------
            # single JPEG
            # ---------------------------------

            if random.random() > 0.75:
                quality_factor = random.randint(5, 95)
            else:
                quality_factor = random.choice([10,20,30,40,50,60,70])

            noise_level = (100-quality_factor)/100.0
            #img_L = cv2.cvtColor(img_L, cv2.COLOR_RGB2BGR)

            if random.random() > 0.25:
              img_L = util.rgb2ycbcr(img_L)
            else:
              img_L = cv2.cvtColor(img_L, cv2.COLOR_RGB2GRAY)

            img_H = img_L.copy()
            result, encimg = cv2.imencode('.jpg', img_L, [int(cv2.IMWRITE_JPEG_QUALITY), quality_factor])
            img_L = cv2.imdecode(encimg, 0)
            #img_L = cv2.cvtColor(img_L, cv2.COLOR_BGR2RGB)

            noise_level = torch.FloatTensor([noise_level])

            # ---------------------------------
            # double JPEG
            # ---------------------------------

            if self.count % self.opt['dataloader_batch_size'] == 0:
                self.count += 1
                self.use_double = random.choice([False, True])

            if self.use_double:
                H, W = img_H.shape[:2]
                rnd_h = random.randint(0, max(0, H - self.patch_size_plus8))
                rnd_w = random.randint(0, max(0, W - self.patch_size_plus8))
                img_H = img_H[rnd_h:rnd_h + self.patch_size_plus8, rnd_w:rnd_w + self.patch_size_plus8]
                img_L = img_L[rnd_h:rnd_h + self.patch_size_plus8, rnd_w:rnd_w + self.patch_size_plus8]
                noise_level = torch.tensor(float('nan'))
                if random.random() > 0.5:
                    quality_factor = random.randint(5, 95)
                else:
                    quality_factor = random.choice([10,20,30,40,50,60,70])
               # noise_level2 = (100-quality_factor)/100.0
              #  noise_level2 = torch.FloatTensor([noise_level2])
                
              #  noise_level = torch.max(noise_level2,noise_level)
                
                result, encimg = cv2.imencode('.jpg', img_L, [int(cv2.IMWRITE_JPEG_QUALITY), quality_factor])
                img_L = cv2.imdecode(encimg, 0)

            H, W = img_H.shape[:2]
            if random.random() > 0.5:
                rnd_h = random.randint(0, max(0, H - self.patch_size))
                rnd_w = random.randint(0, max(0, W - self.patch_size))
            else:
                rnd_h = 0
                rnd_w = 0
            img_H = img_H[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size]
            img_L = img_L[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size]

          #  noise_level_map = torch.ones((1, img_L.shape[0], img_L.shape[0])).mul_(noise_level).float() 
            img_L, img_H = util.uint2tensor3(img_L), util.uint2tensor3(img_H)

            # ---------------------------------
            # add noise
            # ---------------------------------
#            noise = torch.randn(img_L.size()).mul_(noise_level).float()
#            img_L.add_(noise)

        else:
            """
            # --------------------------------
            # get L/H/M image pairs
            # --------------------------------
            """
            H_path = self.paths_H[index]
            img_H = cv2.imread(H_path, cv2.IMREAD_UNCHANGED) 
            L_path = H_path

            img_L = img_H.copy()
            grayscale = True if img_L.ndim == 2 else False

            quality_factor = 10
            noise_level = (100-quality_factor)/100.0
            img_H = img_L.copy()
            result, encimg = cv2.imencode('.jpg', img_L, [int(cv2.IMWRITE_JPEG_QUALITY), quality_factor])
            img_L = cv2.imdecode(encimg, 0)
            #img_L = cv2.cvtColor(img_L, cv2.COLOR_BGR2RGB)

            noise_level = torch.FloatTensor([noise_level])

          #  noise_level_map = torch.ones((1, img_L.shape[0], img_L.shape[0])).mul_(noise_level).float()  # torch.full((1, img_L.size(1), img_L.size(2)), noise_level)

            img_L, img_H = util.uint2tensor3(img_L[..., np.newaxis]), util.uint2tensor3(img_H[..., np.newaxis])

        return {'L': img_L, 'H': img_H, 'qf': noise_level, 'L_path': L_path, 'H_path': H_path}

    def __len__(self):
        return len(self.paths_H)
