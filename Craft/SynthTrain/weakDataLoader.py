import h5py 
import os
import os.path as osp 

import torch 
from torch.utils import data
import cv2 
import numpy as np 

import sys 
sys.path.append('..')
from imgproc import loadImage, normalizeMeanVariance

def resizeMap(image, canvas_size, pad = False):
    """
        image : np array (h,w,3)
 
    """
    ratio =  max(image.shape[:2]) / canvas_size
    img_h, img_w = image.shape[:2]
    resized_image = cv2.resize(image, (int(img_w/ratio),int(img_h/ratio)),interpolation=cv2.INTER_LINEAR)

    if pad :
        pad_resize =np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
        pad_resize[0:resized_image.shape[0], 0:resized_image.shape[1],:] = resized_image
        resized_image, origin_resize = pad_resize, resized_image


    return resized_image, ratio


class WeakData(data.Dataset):

    def __init__(self, dataset_path, canvas_size, cut_size):
        
        # TODO : multi root
        self.list_img = []
        for name in os.listdir(dataset_path):
            if not name.endswith('.txt') :
                self.list_img.append(osp.join(dataset_path,name))

        self.cut_size = cut_size
        self.canvas_size = canvas_size

    def __len__(self):
        return len(self.list_img)    

    def __getitem__(self, idx):

        image = loadImage(self.list_img[idx])

        image, ratio = resizeMap(image, self.canvas_size, True)

        with open(self.list_img[idx].split('.')[0]+'.txt') as F :
            nps = []
            ncuts = []
            ndst = []
            label_counter = []
            for line in F.readlines():
                if line == '':
                    continue 
                label = ','.join(line[8:])
                line = line.split(',')[:8]
                line = [float(x) / ratio for x in line]


                # TODO : whether use resize ? 

                ps = np.array([line[0:2],line[2:4],line[4:6],line[6:8]], dtype=np.float32)
                cut_h, cut_w = np.linalg.norm(ps[0]-ps[-1]), np.linalg.norm(ps[0]-ps[1])
                dst = np.array([[0, 0],[0, cut_h],[cut_w, cut_h],[cut_w, 0]], dtype=np.float32)
                cut_ratio = max(cut_h, cut_w) / self.cut_size

                dst = dst / cut_ratio

                
                M = cv2.getPerspectiveTransform(ps, dst)
                cut = cv2.warpPerspective(image, M, (self.cut_size, self.cut_size))
                

                cut = normalizeMeanVariance(cut)
                cut = torch.from_numpy(cut).permute(2, 0, 1)

                ncuts.append(cut)
                ndst.append([dst])
                nps.append([ps])
                label_counter.append(len(label))
    
        nps = np.concatenate(nps)
        ndst = np.concatenate(nps)

        image = normalizeMeanVariance(image)
        image = torch.from_numpy(image).permute(2, 0, 1)    # [h, w, c] to [c, h, w]

        ncuts = torch.cat([x.unsqueeze(0) for x in ncuts], dim=0)
        return image, ncuts, nps, ndst, label_counter 
        

def weakCollate(batch):
    """
    images : tensor(b,c,h,w)
    ncuts : tensor(n,c,h,w)  # n = n1+n2+..nb
    nps : list 
    ndst : list
    label_counter :  
    """

    batch = filter(lambda x: x is not None, batch)
    images, ncuts, nps, ndst, label_counter = zip(*batch)
    
    images = torch.cat([x.unsqueeze(0) for x in images], dim=0)
    ncuts = torch.cat([x for x in ncuts], dim=0)
    nps = list(nps)
    ndst = list(ndst)
    label_counter = list(label_counter)

    return images, ncuts, nps, ndst, label_counter


def getPseudoGts(model, images, ncuts, nps, ndst, label_counter):
    

if __name__ == '__main__':

    dataset = WeakData('/ai/local/menglc/LSVT/total_data/train', 1024, 512)

    image, ncut, nps, *_ = dataset[1]

    print('dataset len', len(dataset))
    print('image shape', image.shape)
    print('ncut shape', ncut[0].shape)
    print('nps shape', nps.shape)

    loader = data.DataLoader(dataset, batch_size=4 ,collate_fn = weakCollate)

    print('test data loader ')
    for (image, ncut, nps, ndst, label_counter) in loader :
        print('image shape', image.shape)
        print('ncut shape', ncut.shape)
        # print('nps shape', nps.shape)

