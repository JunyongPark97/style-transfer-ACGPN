import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from util.gramMatrix import StyleLoss
import torchvision


class STNModel(BaseModel):
    def name(self):
        return 'STNModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['stn_l1', 'stn_mask_l1', 'stn']
        # specify the images G_A'you want to save/display. The program will call base_model.get_current_visuals
        visual_names_A = ['image_mask', 'cloth_mask', 'real_warped', 'real_image_mask']

        self.visual_names = visual_names_A
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.isTrain:
            self.model_names = ['STN']
        else:  # during test time, only load Gs
            self.model_names = ['STN']

        self.tanh = torch.nn.Tanh()

        # load/define networks
        self.netSTN = networks.define_UnetMask(4, self.gpu_ids)

        if self.isTrain:
            # define loss functions
            self.criterionL1 = torch.nn.L1Loss().to(self.device)

            # initialize optimizers
            self.optimizer = torch.optim.Adam(self.netSTN.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))

            self.optimizers = []
            self.optimizers.append(self.optimizer)

    def set_input(self, input):
        self.real_image = input['base_image'].to(self.device)
        self.real_image_mask = input['base_image_mask'].to(self.device)
        self.real_cloth = input['base_cloth'].to(self.device)
        self.real_cloth_mask = input['base_cloth_mask'].to(self.device)
        self.input_cloth = input['input_cloth'].to(self.device)
        self.input_cloth_mask = input['input_cloth_mask'].to(self.device)

    def forward(self):
        self.image_mask = self.real_image.mul(self.real_image_mask)
        self.cloth_mask = self.real_cloth.mul(self.real_cloth_mask)
        self.input_mask = self.input_cloth.mul(self.input_cloth_mask)

        # Real Image STN
        self.real_warp_conv, self.real_warped, self.real_warped_mask, self.real_rx, self.real_ry, self.real_cx, self.real_cy, self.real_rg, self.real_cg = self.netSTN(self.cloth_mask, self.real_image_mask, self.real_cloth_mask)

    def backward(self):
        #STN loss
        self.loss_stn_l1 = self.criterionL1(self.image_mask, self.real_warped)
        self.loss_stn_mask_l1 = self.criterionL1(self.real_image_mask, self.real_warped_mask)
        #combined loss
        self.loss_stn = self.loss_stn_l1 + self.loss_stn_mask_l1 + torch.mean(self.real_rx + self.real_ry + self.real_cx + self.real_cy + self.real_rg + self.real_cg)
        self.loss_stn.backward(retain_graph=True)

    def optimize_parameters(self):
        # forward
        self.forward()
        # G_A and G_B
        self.set_requires_grad([self.netSTN], True)
        self.optimizer.zero_grad()
        self.backward()
        self.optimizer.step()
