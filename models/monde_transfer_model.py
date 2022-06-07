import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from util.gramMatrix import StyleLoss
import torchvision


class MondeTransferModel(BaseModel):
    def name(self):
        return 'MondeTransferModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        # default CycleGAN did not use dropout
        parser.set_defaults(no_dropout=True)
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            # parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            # parser.add_argument('--lambda_identity', type=float, default=1.0, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')

        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['content_vgg', 'G_A', 'D_A']#'cycle_A',
        # specify the images G_A'you want to save/display. The program will call base_model.get_current_visuals
        visual_names_A = ['image_mask', 'input_mask', 'fake_image', 'empty_image', 'final_image']#, 'cloth_mask', 'rec_image'
        # visual_names_B = ['real_B', 'fake_A', 'rec_B']
        # if self.isTrain and self.opt.lambda_identity > 0.0:
            # visual_names_A.append('idt_A')
            # visual_names_B.append('idt_B')

        self.visual_names = visual_names_A
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.isTrain:
            self.model_names = ['G_A', 'D_A']#
        else:  # during test time, only load Gs
            self.model_names = ['G_A']

        # load/define networks
        # The naming conversion is different from those used in the paper
        # Code (paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                         not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.vgg19 = networks.VGG19(requires_grad=False).cuda()
        use_sigmoid = opt.no_lsgan
        self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                         opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, opt.init_gain,
                                         self.gpu_ids)
        # self.netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
        #                                not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        if self.isTrain:
            # self.fake_A_pool = ImagePool(opt.pool_size)
            # self.fake_B_pool = ImagePool(opt.pool_size)
            # define loss functions
            # self.criterionCycle = torch.nn.L1Loss()
            self.criterionStyleTransfer = networks.StyleTransferLoss().to(self.device)
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan).to(self.device)
            # self.criterionIdt = torch.nn.L1Loss()
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
        # self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def get_vgg_loss(self):
        image_features = self.vgg19(self.image_mask)
        input_features = self.vgg19(self.input_mask)
        fake_features = self.vgg19(self.fake_image)
        return self.criterionStyleTransfer(image_features, input_features, fake_features)

    def forward(self):
        self.image_mask = self.real_image.mul(self.real_image_mask)
        self.cloth_mask = self.real_cloth.mul(self.real_cloth_mask)
        self.input_mask = self.input_cloth.mul(self.input_cloth_mask)
        self.fake_image = self.netG_A(torch.cat([self.image_mask, self.input_cloth], dim=1))
        # self.rec_image = self.netG_A(torch.cat([self.fake_image, self.cloth_mask], dim=1))

        self.empty_image = torch.sub(self.real_image, self.image_mask)
        self.final_image = torch.add(self.empty_image, self.fake_image)

        # self.fake_A = self.netG_B(self.real_B)
        # self.rec_B = self.netG_A(self.fake_A)
    def backward_D_basic(self, netD, base_cloth, input_cloth, real_image, fake_image):#, rec_image
        # Real
        pred_real = netD(torch.cat([real_image.detach(), base_cloth], dim=1))
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(torch.cat([fake_image.detach(), input_cloth], dim=1))
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Rec
        # pred_rec = netD(torch.cat([rec_image.detach(), base_cloth], dim=1))
        # loss_D_rec = self.criterionGAN(pred_rec, False)
        # Combined loss
        loss_D_pos = loss_D_real * 0.5
        loss_D_neg = loss_D_fake * 0.5
        loss_D = loss_D_pos + loss_D_neg
        # backward
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        # rec_image = self.fake_B_pool.query(self.rec_image)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.cloth_mask, self.input_mask, self.image_mask, self.fake_image)

    def backward_G(self):
        # lambda_idt = self.opt.lambda_identity
        # lambda_A = self.opt.lambda_A
        # lambda_B = self.opt.lambda_B
        ## Identity loss
        #if lambda_idt > 0:
        #    # G_A should be identity if real_B is fed.
        #    self.idt_A = self.netG_A(self.real_B)
        #    self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
        #    # G_B should be identity if real_A is fed.
        #    self.idt_B = self.netG_B(self.real_A)
        #    self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        #else:
        #    self.loss_idt_A = 0
        #    self.loss_idt_B = 0

        # GAN loss D_A(G_A(A))
        self.loss_content_vgg, self.loss_style_vgg = self.get_vgg_loss()
        # GAN loss D_B(G_B(B))

        self.loss_G_A = self.criterionGAN(self.netD_A(torch.cat([self.fake_image, self.input_mask], dim=1)), True)
        # self.loss_G_A_2 = self.criterionGAN(self.netD_A(torch.cat([self.rec_image, self.cloth_mask], dim=1)), True)

        # self.loss_G_B = self.criterionGAN(self.netD_B(self.rec_image), True)
        # Forward cycle loss
        # self.loss_cycle_A = self.criterionCycle(self.image_mask, self.rec_image) * lambda_A
        # # Backward cycle loss
        # self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B

        # combined loss
        self.loss_G = self.loss_G_A + self.loss_content_vgg
        self.loss_G.backward()

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
