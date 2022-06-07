import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from util.gramMatrix import StyleLoss
import torchvision
import os


class STNTransferModel(BaseModel):
    def name(self):
        return 'STNTransferModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['style_vgg', 'content_vgg', 'G_A', 'D_A']
        # specify the images G_A'you want to save/display. The program will call base_model.get_current_visuals
        visual_names_A = ['image_mask', 'input_mask', 'warped', 'fake_image', 'final_image']

        self.visual_names = visual_names_A
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.isTrain:
            self.model_names = ['STN', 'G_A', 'D_A']
        else:  # during test time, only load Gs
            self.model_names = ['STN', 'G_A']

        self.tanh = torch.nn.Tanh()

        # load pretrained STN networks
        self.netSTN = networks.define_UnetMask(4, self.gpu_ids)

        load_filename = '%s_net_%s.pth' % ('latest', 'STN')
        load_path = os.path.join(self.save_dir, load_filename)
        net = getattr(self, 'net' + 'STN')
        if isinstance(net, torch.nn.DataParallel):
            net = net.module
        print('loading the model from %s' % load_path)
        # if you are using PyTorch newer than 0.4 (e.g., built from
        # GitHub source), you can remove str() on self.device
        state_dict = torch.load(load_path, map_location=str(self.device))
        if hasattr(state_dict, '_metadata'):
            del state_dict._metadata
        print(state_dict.keys())
        # patch InstanceNorm checkpoints prior to 0.4
        for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
            self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))
        self.netSTN.load_state_dict(state_dict)

        # define networks
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                         not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.vgg19 = networks.VGG19(requires_grad=False).cuda()
        use_sigmoid = opt.no_lsgan
        self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                         opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, opt.init_gain,
                                         self.gpu_ids)
        if self.isTrain:
            # define loss functions
            self.criterionStyleTransfer = networks.StyleTransferLoss().to(self.device)
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan).to(self.device)
            self.criterionL1 = torch.nn.L1Loss().to(self.device)
            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(self.netG_A.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD_A.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))

            self.optimizers = []
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        self.real_image = input['base_image'].to(self.device)
        self.real_image_mask = input['base_image_mask'].to(self.device)
        self.real_cloth = input['base_cloth'].to(self.device)
        self.real_cloth_mask = input['base_cloth_mask'].to(self.device)
        self.input_cloth = input['input_cloth'].to(self.device)
        self.input_cloth_mask = input['input_cloth_mask'].to(self.device)

    def get_vgg_loss(self):
        image_features = self.vgg19(self.image_mask)
        input_features = self.vgg19(self.input_mask)
        fake_features = self.vgg19(self.fake_image)
        return self.criterionStyleTransfer(image_features, input_features, fake_features)

    def forward(self):
        self.image_mask = self.real_image.mul(self.real_image_mask)
        self.cloth_mask = self.real_cloth.mul(self.real_cloth_mask)
        self.input_mask = self.input_cloth.mul(self.input_cloth_mask)

       # Fake Image STN
        self.warp_conv, self.warped, self.warped_mask, self.rx, self.ry, self.cx, self.cy, self.rg, self.cg = self.netSTN(self.input_mask, self.real_image_mask, self.input_cloth_mask)
        self.warp_conv = self.warp_conv[:, 0:3, :, :]
        self.warp_conv = self.tanh(self.warp_conv)
        self.fake_image = self.netG_A(torch.cat([self.warped, self.image_mask], dim=1))

        self.empty_image = torch.sub(self.real_image, self.image_mask)
        self.final_image = torch.add(self.empty_image, self.fake_image)

    def backward_D_basic(self, netD, real_image, fake_image):
        # Real
        pred_real = netD(torch.cat([real_image], dim=1))
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(torch.cat([fake_image], dim=1))
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss
        loss_D_pos = loss_D_real * 0.5
        loss_D_neg = loss_D_fake * 0.5
        loss_D = loss_D_pos + loss_D_neg
        # backward
        loss_D.backward(retain_graph=True)
        return loss_D

    def backward_D_A(self):
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_image, self.final_image)

    def backward_G(self):
        # GAN loss D_A(G_A(A))
        self.loss_content_vgg, self.loss_style_vgg = self.get_vgg_loss()
        # GAN loss D_B(G_B(B))
        self.loss_G_A = self.criterionGAN(self.netD_A(torch.cat([self.final_image], dim=1)), True)
        # combined loss
        self.loss_G = self.loss_G_A + self.loss_content_vgg + self.loss_style_vgg
        self.loss_G.backward(retain_graph=True)

    def optimize_parameters(self):
        # forward
        self.forward()
        # G_A and G_B
        self.set_requires_grad([self.netD_A], False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        # D_A and D_B
        self.set_requires_grad([self.netD_A], True)
        self.optimizer_D.zero_grad()
        self.backward_D_A()
        # # self.backward_D_B()
        self.optimizer_D.step()
