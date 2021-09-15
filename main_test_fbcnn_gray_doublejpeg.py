import os.path
import logging
import numpy as np
from datetime import datetime
from collections import OrderedDict
import torch
import cv2
from utils import utils_logger
from utils import utils_image as util
import requests

def main():

    quality_factor_list = [10, 30, 50]
    testset_name = 'LIVE1_gray' # 'LIVE1_gray' 'Classic5' 'BSDS500_gray'
    n_channels = 1            # set 1 for grayscale image, set 3 for color image
    model_name = 'fbcnn_gray_double.pth'
    nc = [64,128,256,512]
    nb = 4
    show_img = False                 # default: False
    testsets = 'testsets' 
    results = 'test_results'    

    for qf1 in quality_factor_list:
        for qf2 in quality_factor_list:
            result_name = testset_name + '_' + model_name[:-4]
            H_path = os.path.join(testsets, testset_name)
            E_path = os.path.join(results, result_name, str(qf1)+str(qf2))   # E_path, for Estimated images
            util.mkdir(E_path)

            model_pool = 'model_zoo'  # fixed
            model_path = os.path.join(model_pool, model_name)
            if os.path.exists(model_path):
                print(f'loading model from {model_path}')
            else:
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                url = 'https://github.com/jiaxi-jiang/FBCNN/releases/download/v1.0/{}'.format(os.path.basename(model_path))
                r = requests.get(url, allow_redirects=True)
                print(f'downloading model {model_path}')
                open(model_path, 'wb').write(r.content)
            
            logger_name = result_name + '_qf_' + str(qf1)+str(qf2)
            utils_logger.logger_info(logger_name, log_path=os.path.join(E_path, logger_name+'.log'))
            logger = logging.getLogger(logger_name)
            logger.info('--------------- QF1={:d}, QF2={:d} ---------------'.format(qf1,qf2))

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            border = 0


            # ----------------------------------------
            # load model
            # ----------------------------------------

            from models.network_fbcnn import FBCNN as net
            model = net(in_nc=n_channels, out_nc=n_channels, nc=nc, nb=nb, act_mode='R')
            model.load_state_dict(torch.load(model_path), strict=True)
            model.eval()
            for k, v in model.named_parameters():
                v.requires_grad = False
            model = model.to(device)
            logger.info('Model path: {:s}'.format(model_path))

            test_results = OrderedDict()
            test_results['psnr'] = []
            test_results['ssim'] = []
            test_results['psnrb'] = []

            H_paths = util.get_image_paths(H_path)
            for idx, img in enumerate(H_paths):

                # ------------------------------------
                # (1) img_L
                # ------------------------------------
                img_name, ext = os.path.splitext(os.path.basename(img))
                logger.info('{:->4d}--> {:>10s}'.format(idx+1, img_name+ext))
                img_L = util.imread_uint(img, n_channels=n_channels)

                if n_channels == 3:
                    img_L = cv2.cvtColor(img_L, cv2.COLOR_RGB2BGR)                
                _, encimg = cv2.imencode('.jpg', img_L, [int(cv2.IMWRITE_JPEG_QUALITY), qf1])
                img_L = cv2.imdecode(encimg, 0) if n_channels == 1 else cv2.imdecode(encimg, 3)

                shift_h, shift_w = 4, 4
                _, encimg = cv2.imencode('.jpg', img_L[shift_h:,shift_w:], [int(cv2.IMWRITE_JPEG_QUALITY), qf2])
                img_L = cv2.imdecode(encimg, 0) if n_channels == 1 else cv2.imdecode(encimg, 3)

                if n_channels == 3:
                    img_L = cv2.cvtColor(img_L, cv2.COLOR_BGR2RGB)               
                img_L = util.uint2tensor4(img_L)
                img_L = img_L.to(device)

                # ------------------------------------
                # (2) img_E
                # ------------------------------------

                # img_E,QF = model(img_L, torch.tensor([[0.6]]))      
                img_E,QF = model(img_L)
                QF = 1 - QF

                img_E = util.tensor2single(img_E)
                img_E = util.single2uint(img_E)
                img_H = util.imread_uint(H_paths[idx], n_channels=n_channels).squeeze()[shift_h:,shift_w:]
                # --------------------------------
                # PSNR and SSIM, PSNRB
                # --------------------------------

                psnr = util.calculate_psnr(img_E, img_H, border=border)
                ssim = util.calculate_ssim(img_E, img_H, border=border)
                psnrb = util.calculate_psnrb(img_H, img_E, border=border)
                test_results['psnr'].append(psnr)
                test_results['ssim'].append(ssim)
                test_results['psnrb'].append(psnrb)
                logger.info('{:s} - PSNR: {:.2f} dB; SSIM: {:.3f}; PSNRB: {:.2f} dB.'.format(img_name+ext, psnr, ssim, psnrb))
                logger.info('predicted quality factor: {:d}'.format(round(float(QF*100))))

                util.imshow(np.concatenate([img_E, img_H], axis=1), title='Recovered / Ground-truth') if show_img else None
                util.imsave(img_E, os.path.join(E_path, img_name+'.png'))

            ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
            ave_ssim = sum(test_results['ssim']) / len(test_results['ssim'])
            ave_psnrb = sum(test_results['psnrb']) / len(test_results['psnrb'])
            logger.info(
                'Average PSNR/SSIM/PSNRB - {} -: {:.2f}$\\vert${:.4f}$\\vert${:.2f}.'.format(result_name+'_qf1_'+str(qf1)+'_qf2_'+str(qf2), ave_psnr, ave_ssim, ave_psnrb))


if __name__ == '__main__':
    main()
