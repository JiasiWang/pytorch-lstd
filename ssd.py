import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from layers import *
from data import v2
import os
#from layers.roi_pooling.roi_pool import RoIPoolFunction
from layers.functions.post_rois import Post_rois
#from layers.roi_pooling import roi_pooling as _roi_pooling
from lib.model.roi_pooling.modules.roi_pool import _RoIPooling
import cv2

class SSD(nn.Module):
    """Single Shot Multibox Architecture
    The network is composed of a base VGG network followed by the
    added multibox conv layers.  Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.

    Args:
        phase: (string) Can be "test" or "train"
        base: VGG16 layers for input, size of either 300 or 500
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    def __init__(self, phase, base, extras, head, extras_lstd, num_classes):
        super(SSD, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        # TODO: implement __call__ in PriorBox
        self.priorbox = PriorBox(v2)
        self.priors = Variable(self.priorbox.forward(), volatile=True)
        self.size = 300

        # SSD network
        self.vgg = nn.ModuleList(base)
        # Layer learns to scale the l2 normalized features from conv4_3
        self.L2Norm = L2Norm(512, 20)
        self.extras = nn.ModuleList(extras)

        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])
        '''
        self.extras_lstd = nn.ModuleList(extras_lstd)
        self.classifier = nn.ModuleList([nn.Linear(256*3*3, 21)])
        '''
        self.classifier = nn.ModuleList([nn.Linear(1024 * 5 * 5, 4096),
            nn.ReLU(True),
            nn.Dropout(p = 0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 21)])

        self.softmax = nn.Softmax(dim=-1)
        self.post_rois = Post_rois(num_classes, 0, 100, 0, 0.65)

        self.detect = Detect(num_classes, 0, 100, 0, 0.65)
        #self.ROI_POOL = RoIPoolFunction(5, 5, 1.0/19.0)
        #self.roi_pooling = _roi_pooling       
        self.roi_pool = _RoIPooling(5, 5, 1.0/16.0)


    def forward(self, x, im_ids):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3,300,300].

        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """
        def vis(img, rois, idx):
            print(rois.size())
            im = img.cpu().data
            im = im.numpy()
            im = im.transpose(1,2,0)
            img = im[:, :, (2, 1, 0)]
            im = img.copy()
            #for i in range(decoded_boxes.size(0)):
            for i in range(10):
                write_flag = True
                cv2.rectangle(im, (int(rois[i][1]), int(rois[i][2])),(int(rois[i][3]),int(rois[i][3])), (255,0,0),1)
                #cv2.putText(im, str(confs[i]), (int(decoded_boxes[i][0]*300) + int((decoded_boxes[i][2]*300 - decoded_boxes[i][0]*300)/2) ,int(decoded_boxes[i][1]*300) + int((decoded_boxes[i][3]*300 - decoded_boxes[i][1]*300)/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255))
            if write_flag:
                cv2.imwrite('./vis/'+str(idx)+'.jpg', im)
                #cv2.imshow('./vis/'+str(idx)+'.jpg', im)


        img = x.clone()

        sources = list()
        loc = list()
        conf = list()
        
        # apply vgg up to conv4_3 relu
        for k in range(23):
            x = self.vgg[k](x)

        s = self.L2Norm(x)
        sources.append(s)

        # apply vgg up to fc7
        for k in range(23, len(self.vgg)):
            x = self.vgg[k](x)
        sources.append(x)

        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                sources.append(x)

        # apply multibox head to source layers
        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        
        rpn_output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
                self.priors
        )
        rois = self.post_rois(img,  
                loc.view(loc.size(0), -1, 4),                   # loc preds
                conf.view(conf.size(0), -1, self.num_classes),                         # conf preds
                self.priors.type(type(x.data))                  # default boxes
        ) # rois.size (32,1,100,5) 5:

        # print(conf.size()) (1, 8732*2)
        rois = Variable(rois.data)
        rois = rois.contiguous().view(-1,5)
        rois[:,1:] = rois[:,1:]*300
    
        x = self.roi_pool(sources[1], rois)

        #for k, v in enumerate(self.extras_lstd):
        #    x = v(x)
      
        x = x.view(x.size(0), -1)
        
        for k, v in enumerate(self.classifier):
             x = v(x)
        
        #cls_output = self.classifier[0](x)

        if self.phase == "train": 
            return rpn_output, x, rois
        else:
            
            x = self.softmax(x.view(x.size(0), 21))
            if 0:
                for n_im in range(img.size(0)):
                    vis(img[n_im],rois[n_im*100:(n_im*100 + 100)],n_im)
            return x, rois

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file,
                       map_location=lambda storage, loc: storage))
 
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')



# This function is derived from torchvision VGG make_layers()
# https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
def vgg(cfg, i, batch_norm=False):
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6,
               nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers


def add_extras(cfg, i, batch_norm=False):
    # Extra layers added to VGG for feature scaling
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                layers += [nn.Conv2d(in_channels, cfg[k + 1],
                           kernel_size=(1, 3)[flag], stride=2, padding=1)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
            flag = not flag
        in_channels = v
    return layers

def add_lstd_extras(i):
    # Extra layers added to VGG for feature scaling
    layers = []
    in_channels = i
    conv_add1 = nn.Conv2d(in_channels, 256,
                           kernel_size=3, stride=1, padding=1)

    in_channels = 256
    batchnorm_add1 = nn.BatchNorm2d(in_channels)
    conv_add2 = nn.Conv2d(in_channels, 256,
                           kernel_size=3, stride=2, padding=1)

    batchnorm_add2 = nn.BatchNorm2d(in_channels)
    #bbox_score_voc = nn.Linear(256, 21)

    layers += [conv_add1, batchnorm_add1, nn.ReLU(inplace=True), conv_add2, batchnorm_add2, nn.ReLU(inplace=True)]
    return layers


def multibox(vgg, extra_layers, cfg, num_classes):
    loc_layers = []
    conf_layers = []
    vgg_source = [21, -2]
    for k, v in enumerate(vgg_source):
        loc_layers += [nn.Conv2d(vgg[v].out_channels,
                                 cfg[k] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(vgg[v].out_channels,
                        cfg[k] * num_classes, kernel_size=3, padding=1)]
    for k, v in enumerate(extra_layers[1::2], 2):
        loc_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                 * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                  * num_classes, kernel_size=3, padding=1)]
    return vgg, extra_layers, (loc_layers, conf_layers)


base = {
    '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
    '512': [],
}
extras = {
    '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
    '512': [],
}
mbox = {
    '300': [4, 6, 6, 6, 4, 4],  # number of boxes per feature map location
    '512': [],
}


def build_ssd(phase, size=300, num_classes=21):
    if phase != "test" and phase != "train":
        print("Error: Phase not recognized")
        return
    if size != 300:
        print("Error: Sorry only SSD300 is supported currently!")
        return
    base_, extras_, head_ = multibox(vgg(base[str(size)], 3),
                                     add_extras(extras[str(size)], 1024),
                                     mbox[str(size)], num_classes)
    extras_lstd_ = add_lstd_extras(1024)
    #return SSD(phase, base_, extras_, head_, num_classes)
    return SSD(phase, base_, extras_, head_, extras_lstd_, num_classes)
