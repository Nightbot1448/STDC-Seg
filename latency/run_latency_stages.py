from __future__ import division

import os
import sys
import logging
import torch
import numpy as np

from thop import profile
sys.path.append("../")

from utils.darts_utils import create_exp_dir, plot_op, plot_path_width, objective_acc_lat
# try:
#     from utils.darts_utils import compute_latency_ms_tensorrt as compute_latency
#     print("use TensorRT for latency test")
# except:
from utils.darts_utils import compute_latency_ms_pytorch as compute_latency
print("use PyTorch for latency test")


from models.model_stages_trt import BiSeNet
import DDRNet
import DDRNet_HDB_B2N
import ddrnet_hbd_b2n_dgt

def main():
    
    print("begin")
    # preparation ################
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    seed = 12345
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    # Configuration ##############
    use_boundary_2 = False
    use_boundary_4 = False
    use_boundary_8 = True
    use_boundary_16 = False
    use_conv_last = False
    n_classes = 19
    
    #  1050Ti is 45 - 60 times slower than 1080Ti

    # # STDC1Seg-50 250.4 FPS on NVIDIA GTX 1080Ti
    # # 5.04 - 5.41 FPS on NVIDIA GTX 1050Ti 
    # # 1050Ti is 46 - 50 times slower than 1080Ti
    # backbone = 'STDCNet813'
    # methodName = 'STDC1-Seg'
    # inputSize = 512
    # inputScale = 50
    # inputDimension = (1, 3, 512, 1024)

    # # STDC1Seg-75 126.7 FPS on NVIDIA GTX 1080Ti
    # # 1.92 - 2.21 FPS on NVIDIA GTX 1050Ti
    # # 1050Ti is 57 - 66 times slower than 1080Ti
    # backbone = 'STDCNet813'
    # methodName = 'STDC1-Seg'
    # inputSize = 768
    # inputScale = 75
    # inputDimension = (1, 3, 768, 1536)

    # # STDC2Seg-50 188.6 FPS on NVIDIA GTX 1080Ti
    # # 3.35 - 3.83 FPS on NVIDIA GTX 1050Ti
    # # 1050Ti is 49 - 56 times slower than 1080Ti
    # backbone = 'STDCNet1446'
    # methodName = 'STDC2-Seg'
    # inputSize = 512
    # inputScale = 50
    # inputDimension = (1, 3, 512, 1024)

    # # STDC2Seg-75 97.0 FPS on NVIDIA GTX 1080Ti
    # # 1.55 - 1.68 FPS on NVIDIA GTX 1050Ti
    # # 1050Ti is 58 - 62 times slower than 1080Ti
    # backbone = 'STDCNet1446'
    # methodName = 'STDC2-Seg'
    # inputSize = 768
    # inputScale = 75
    # inputDimension = (1, 3, 768, 1536)

    backbone = None
    methodName = 'DDRNet'
    inputSize = 768
    # inputScale = 100
    # inputScale = 101
    # inputScale = 102
    inputScale = 103
    inputDimension = (1, 3, 768, 1536)



    model = None
    if methodName != 'DDRNet':
        model = BiSeNet(backbone=backbone, n_classes=n_classes, 
        use_boundary_2=use_boundary_2, use_boundary_4=use_boundary_4, 
        use_boundary_8=use_boundary_8, use_boundary_16=use_boundary_16, 
        input_size=inputSize, use_conv_last=use_conv_last)
    else:
        if inputScale == 100:
            # DDRNet 1.00-1.15 FPS on NVIDIA GTX 1050Ti
            model = DDRNet.DualResNet(DDRNet.BasicBlock, [2, 2, 2, 2], num_classes=19, planes=64, 
                         spp_planes=128, head_planes=128, augment=False)
        elif inputScale == 101:
            # 2.05 - 2.36
            model = DDRNet_HDB_B2N.get_seg_model(False)
        elif inputScale == 102:
            model = DDRNet_HDB_B2N.get_seg_model(True)
        elif inputScale == 103:
            model = ddrnet_hbd_b2n_dgt.get_seg_model()
        else:
            print("wrong model")

    
    print('loading parameters...')
    respth = '../checkpoints/{}/'.format(methodName)
    save_pth = os.path.join(respth, 'model_maxmIOU{}.pth'.format(inputScale))
    model.load_state_dict(torch.load(save_pth))
    model = model.cuda()
    #####################################################

    latency = compute_latency(model, inputDimension)
    print("{}{} FPS:".format(methodName, inputScale) + str(1000./latency))
    logging.info("{}{} FPS:".format(methodName, inputScale) + str(1000./latency))

    # calculate FLOPS and params
    '''
    model = model.cpu()
    flops, params = profile(model, inputs=(torch.randn(inputDimension),), verbose=False)
    print("params = {}MB, FLOPs = {}GB".format(params / 1e6, flops / 1e9))
    logging.info("params = {}MB, FLOPs = {}GB".format(params / 1e6, flops / 1e9))
    '''


if __name__ == '__main__':
    main() 
