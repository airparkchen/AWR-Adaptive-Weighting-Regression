import os
import os.path as osp
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
from torchnet import meter

from model.resnet_deconv import get_deconv_net
from model.hourglass import PoseNet
from model.loss import My_SmoothL1Loss
from dataloader.nyu_loader import NYU
from util.feature_tool import FeatureModule
from util.eval_tool import EvalUtil
from util.vis_tool import VisualUtil
from config import opt
from test import Trainer 

class DemoTrainer(Trainer):

    def __init__(self, config):
        super(DemoTrainer, self).__init__(config)
        
    def predict_single_image(self, img_path):
        self.net.eval()
        input_image = Image.open(img_path).convert('L')  # 轉為灰度圖像
        # input_image = Image.open(img_path)
        preprocess = transforms.Compose([
            transforms.Resize((self.config.img_size, self.config.img_size)),
            transforms.ToTensor(),
        ])
        input_tensor = preprocess(input_image)
        input_batch = input_tensor.unsqueeze(0).cuda()
        
        with torch.no_grad():
            if 'hourglass' in self.config.net:
                offset_pred = self.net(input_batch)[-1]  # 使用 hourglass的最後一個輸出
            else:
                offset_pred = self.net(input_batch)
            jt_uvd_pred = self.FM.offset2joint_softmax(offset_pred, input_batch, self.config.kernel_size)
        
        jt_uvd_pred = jt_uvd_pred.detach().cpu().numpy()

        # Visualization
        result_path = os.path.join(self.result_dir, f"{osp.basename(img_path)}_result.png")
        self.visualizer.plot(input_tensor.squeeze().cpu().numpy(), result_path, jt_uvd_pred[0])
        print(jt_uvd_pred[0])
        print(f"Prediction saved at {result_path}")


if __name__ == '__main__':
    img_path = "./data/test/depth_1_0000008.png"
    demo_trainer = DemoTrainer(opt)
    demo_trainer.predict_single_image(img_path)