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

    testset_name = 'Real'  # folder name of real images
    n_channels = 3            # set 1 for grayscale image, set 3 for color image
    model_name = 'fbcnn_color.pth'
    nc = [64,128,256,512]
    nb = 4
    testsets = 'testsets'    
    results = 'test_results'     

    do_flexible_control = True
    QF_control = [5,10,30,50,70,90] # adjust qf as input to provide different results

    result_name = testset_name + '_' + model_name[:-4]
    L_path = os.path.join(testsets, testset_name)
    E_path = os.path.join(results, result_name)   # E_path, for Estimated images
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
    
    logger_name = result_name
    utils_logger.logger_info(logger_name, log_path=os.path.join(E_path, logger_name+'.log'))
    logger = logging.getLogger(logger_name)

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

    L_paths = util.get_image_paths(L_path)
    for idx, img in enumerate(L_paths):

        # ------------------------------------
        # (1) img_L
        # ------------------------------------
        img_name, ext = os.path.splitext(os.path.basename(img))
        logger.info('{:->4d}--> {:>10s}'.format(idx+1, img_name+ext))
        img_L = util.imread_uint(img, n_channels=n_channels)
       
        img_L = util.uint2tensor4(img_L)
        img_L = img_L.to(device)

        # ------------------------------------
        # (2) img_E
        # ------------------------------------
     
        #img_E,QF = model(img_L, torch.tensor([[0.6]]))      
        img_E,QF = model(img_L)
        QF = 1- QF
        img_E = util.tensor2single(img_E)
        img_E = util.single2uint(img_E)
        logger.info('predicted quality factor: {:d}'.format(round(float(QF*100))))
        util.imsave(img_E, os.path.join(E_path, img_name+'.png'))

        if do_flexible_control:
            for QF_set in QF_control:
                logger.info('Flexible control by QF = {:d}'.format(QF_set))
            #    from IPython import embed; embed()
                qf_input = torch.tensor([[1-QF_set/100]]).cuda() if device == torch.device('cuda') else torch.tensor([[1-QF_set/100]])
                img_E,QF = model(img_L, qf_input)  
                QF = 1- QF
                img_E = util.tensor2single(img_E)
                img_E = util.single2uint(img_E)
                util.imsave(img_E, os.path.join(E_path, img_name + '_qf_'+ str(QF_set)+'.png'))


if __name__ == '__main__':
    main()
