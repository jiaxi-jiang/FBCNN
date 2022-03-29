import os.path
import random
import numpy as np
import torch
import torch.utils.data as data
import utils.utils_image as util
from PIL import Image
from PIL import JpegPresets
from io import BytesIO
from IPython import embed
import cv2
import lmdb
import mmcv
from torch.utils import data as data

from basicsr.data.transforms import augment, paired_random_crop, totensor
from basicsr.data.util import (paired_paths_from_folder,
                               paired_paths_from_lmdb,
                               paired_paths_from_meta_info_file)
from basicsr.utils import FileClient
import multiprocessing
#CompressBuffer = BytesIO()

class DatasetFBAR(data.Dataset):
    """
    # -----------------------------------------
    # Get Compressed/Original (C/O) images.
    # Only dataroot_O is needed.
    # -----------------------------------------
    # e.g., ResUNet
    # -----------------------------------------
    """

    def __init__(self, opt):
        super(DatasetFBAR, self).__init__()
        print('Dataset: Deblocking JPEG images.')
        self.opt = opt
        self.n_channels = opt['n_channels'] if opt['n_channels'] else 3
        self.patch_size = opt['H_size'] if opt['H_size'] else 64

        # ------------------------------------
        # get path of Original images
        # return None if input is None
        # ------------------------------------

        self.need_preproc_train = opt['need_preproc']
        self.from_lmdb = opt['from_lmdb']
        self.qf = opt['qf'] # for non-blinding
#        self.QF = 10
        self.complex = 0
        self.count = 0
        if self.from_lmdb:
            self.io_backend_opt = {'type':'lmdb', 'db_paths': [opt['dataroot_L'], opt['dataroot_H']], 'client_keys': ['lq', 'gt']}
            self.paths = paired_paths_from_lmdb([opt['dataroot_L'], opt['dataroot_H']], ['lq', 'gt'])
            self.file_client = None
        else:
            self.paths_H = util.get_image_paths(opt['dataroot_H'])
            self.paths_L = util.get_image_paths(opt['dataroot_L'])
            self.paths = self.paths_H


    def __getitem__(self, index):



        if self.opt['phase'] == 'train':


            if self.from_lmdb:

                if self.file_client is None:
                    self.file_client = FileClient(
                    self.io_backend_opt.pop('type'), **self.io_backend_opt)

                H_path = self.paths[index]['gt_path']
                img_bytes = self.file_client.get(H_path, 'gt')
                img_gt = mmcv.imfrombytes(img_bytes)#[:,:,0]
                img_gt = util.rgb2ycbcr(img_gt)

                L_path = self.paths[index]['lq_path']
                img_bytes = self.file_client.get(L_path, 'lq')
                img_lq = mmcv.imfrombytes(img_bytes)#[:,:,0]
                img_lq = util.rgb2ycbcr(img_lq)
                QF = 1-int(L_path[-2:])/100.0
                # --------------------------------
                # augmentation - flip, rotate
                # --------------------------------
                mode = np.random.randint(0, 8)
                img_H = util.augment_img(img_gt, mode=mode)
                img_L = util.augment_img(img_lq, mode=mode)
                # --------------------------------

                img_H = util.uint2tensor3(img_H)
                img_L = util.uint2tensor3(img_L)

            else:

                # ------------------------------------
                # get H image
                # ------------------------------------
                H_path = self.paths_H[index]

                """
                # --------------------------------
                # get L/H patch pairs
                # --------------------------------
                """

                if not self.need_preproc_train:

 #                   C_path = self.paths_C[index] if self.paths_C[index] else O_path

                    img_H = util.imread_uint(H_path, self.n_channels)

                    L_path = self.paths_L[index]
                    img_L = util.imread_uint(L_path, self.n_channels)
                    QF = int(L_path[-6:-4])/100.0
                #    embed()
#                    print(QF)
                    #img_O = util.rgb2ycbcr(img_O)
                    #img_C = util.rgb2ycbcr(img_C)
                    H, W, C = img_O.shape
                    rnd = np.random.randint(0,3)
                    if rnd == 0:
                        rnd_h = random.randint(0, max(0, H - self.patch_size))
                        rnd_w = random.randint(0, max(0, W - self.patch_size))
                    else:
                        rnd_h, rnd_w = 0, 0
                    img_H = img_O[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size,...]
                    img_L = img_C[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size,...]

                    # --------------------------------
                    # augmentation - flip, rotate
                    # --------------------------------
                    mode = np.random.randint(0, 8)
                    img_H = util.augment_img(img_H, mode=mode)
                    img_L = util.augment_img(img_L, mode=mode)
                    # --------------------------------

                    img_H = util.uint2tensor3(img_H)
                    img_L = util.uint2tensor3(img_L)

                else:
                #    process_id = multiprocessing.current_process()._identity[0]
                #    if self.count % (self.opt['dataloader_batch_size']) == 0:
                #        self.complex = random.randint(1,3)# if process_id < 20 else 0
                   # print(self.count, self.complex)
                #    self.count += 1
                    L_path = H_path
#                    QF = int(C_path[-6:-4])/100.0
                    img_H = util.imread_uint(H_path, 1)
                    H, W, C = img_H.shape
                  #  img_O = img_O[:,:,0]
                    if C == 3:
                        img_H = util.rgb2ycbcr(img_H)
#                        print(img_O.shape)
#                        img_O, _, _ = Image.fromarray(img_O).convert('YCbCr').split()
#                        img_O = np.array(img_O)
                    # --------------------------------
                    # JPEG compression
                    # --------------------------------
                    rnd_h = random.randint(0, max(0, H - 96))
                    rnd_w = random.randint(0, max(0, W - 96))
                    img_O = img_O[rnd_h:rnd_h + 96, rnd_w:rnd_w + 96,...]

            #        rnd = random.randint(0,10)
                 #   qf = random.choice([10,20,30,40,50]) if rnd else random.randint(10,90)

              #      if rnd <4:
                    rnd = random.randint(0,5)
                    qf = random.randint(10,40) if rnd else random.randint(10,90)
              #      elif rnd <7:
              #          qf = random.choice([30,40])
              #      else:
              #          qf =random.randint(10,90)

                    _, encimg = cv2.imencode('.jpg', img_O, [int(cv2.IMWRITE_JPEG_QUALITY), qf])
                    img_C = cv2.imdecode(encimg, 0) 

                    h,w = 0,0
                 #   if self.complex == 1:
                  #      h = random.randint(0,7)
                   #     w = random.randint(0,7)
                    #    qf2 = random.randint(qf, 99)
                      #  _, encimg = cv2.imencode('.jpg', img_C[h:,w:,...], [int(cv2.IMWRITE_JPEG_QUALITY), qf2])
                     #   img_C = cv2.imdecode(encimg, 0) 

                    QF = qf/100.0
                    # --------------------------------
                    # randomly crop the patch
                    # --------------------------------
               #     rndd = random.randint(0,1)
               #     if rndd:
               #         rnd_h,rnd_w = 0,0
               #     else:
                    rnd_h = random.randint(0, max(0, 96 - h - self.patch_size))
                    rnd_w = random.randint(0, max(0, 96 - w - self.patch_size))
                    img_O = img_O[h+rnd_h:h+rnd_h + self.patch_size, w+rnd_w:w+rnd_w + self.patch_size,...]
                    img_C = img_C[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size,...]


#                    patch_O = util.rgb2ycbcr(patch_O)
#                    patch_C = util.rgb2ycbcr(patch_C)



#                    tmp = Image.fromarray(patch_O)
#                    CompressBuffer.seek(0)
#                    tmp.save(CompressBuffer, "JPEG", quality=qf)
#                    CompressBuffer.seek(0)
#                    patch_C = np.array(Image.open(CompressBuffer))

                    # --------------------------------
                    # augmentation - flip, rotate
                    # --------------------------------
                    mode = np.random.randint(0, 8)
                    img_H = util.augment_img(img_H, mode=mode)
                    img_L = util.augment_img(img_L, mode=mode)
#

                    # --------------------------------
                    # HWC to CHW, numpy(uint) to tensor
                    # --------------------------------
                    img_H = util.uint2tensor3(img_H)
                    img_L = util.uint2tensor3(img_L)



        else: ### test
            self.complex = 0
            H_path = self.paths_H[index]
            L_path = self.paths_L[index] if self.paths_L[index] else H_path

            """
            # --------------------------------
            # get C/O image pairs
            # --------------------------------
            """
            img_H = util.imread_uint(H_path, 1)
            img_L = util.imread_uint(L_path, 1)

            img_H = util.uint2tensor3(img_H)
            img_L = util.uint2tensor3(img_L)

            QF = 0.1
        return {'L': img_L, 'H': img_H, 'H_path': H_path, 'L_path': L_path, 'QF': torch.tensor(QF).unsqueeze(0), 'complex':self.complex}

    def __len__(self):
        return len(self.paths)
