import torch
import h5py  
import os 
import os.path as osp
from torch.utils import data
import random

import sys 
sys.path.append('..')
from imgproc import normalizeMeanVariance


class SynthData(data.DataLoader):

    def __init__(self, dataset_path ):
        # super(SynthData, self).__init__()
        # self.dataset = h5py.File(dataset_path, 'r')
        
        self.list_dataset_path = []
        self.list_keys = []
        
        for name in os.listdir(dataset_path):
            if name.endswith('.h5'):
                name = osp.join(dataset_path,name)
                idx = len(self.list_dataset_path)
                with h5py.File(name, 'r') as F :
                    self.list_keys += [[idx, x ] for x in F.keys()]
                self.list_dataset_path.append(name)
                

        random.shuffle(self.list_keys)
        # self.list_dataset_keys = []

        # for name in self.list_dataset_path:
        #     with h5py.File(name, 'r') as F :
        #         self.list_dataset_keys.append(list(F.keys()))
        
        # self.list_iters = [iter(x) for x in self.list_dataset_keys]


        print('success open')


    def peek(self, idx):
        keys = self.list_keys[idx]

        with h5py.File(self.list_dataset_path[keys[0]], 'r') as F :
            image = F[keys[1]]['image'][...]
            label = F[keys[1]]['label'][...]

        import cv2 
        from imgproc import normalizeMeanVariance, cvt2HeatmapImg

        print(keys)
        cv2.imwrite('test_{}.jpg'.format(idx), image[:,:,::-1])
        cv2.imwrite('test_{}_region.jpg'.format(idx), cvt2HeatmapImg(label[:,:,0]))
        cv2.imwrite('test_{}_link.jpg'.format(idx), cvt2HeatmapImg(label[:,:,1]))
        
        

    def __len__(self):
        # count = 0 
        # for ks in self.list_dataset_keys:
        #     count += len(ks)
        
        # return count 
        return len(self.list_keys)

    def __getitem__(self, idx):
        
        idx = idx % len(self)

        # choice = idx % len(self.list_dataset_path)
        

        # try :
        #     key  = next(self.list_iters[choice])
        # except :
        #     self.list_iters[choice] = iter(self.list_dataset_keys[choice])
        #     key = next(self.list_iters[choice])

        # with h5py.File(self.list_dataset_path[choice], 'r') as F :
        #     image = F[key]['image'][...]
        #     label = F[key]['label'][...]

        set_id, key = self.list_keys[idx]

        with h5py.File(self.list_dataset_path[set_id], 'r') as F :
            image = F[key]['image'][...]
            label = F[key]['label'][...]

        image = normalizeMeanVariance(image)
        image = torch.from_numpy(image).permute(2, 0, 1)    # [h, w, c] to [c, h, w]

        label = torch.from_numpy(label)

        return image, label 

if __name__ == '__main__':


    import cv2 
    # dataset = SynthData('/ai/local/menglc/CRAFT_dataset/train')
    dataset = SynthData('/ai/local/menglc/SynthData/test')
    

    for i in range(10):
        dataset.peek(i)


    # loader = data.DataLoader(dataset ,batch_size = 32 ,  shuffle= False, num_workers= 4)

    # print('dataset len' , len(dataset))

    # for (img, label) in loader:

    #     print(img.shape)

    #     print((img[0]-img[1]).abs().max(), (img[0]-img[1]).shape)
    #     break 
    # image, label = dataset[0]
    # image = image.numpy()
    # label = label.numpy()

    # print('dataset len {}'.format(len(dataset)))


    # from PIL import Image 
    # from imgproc import cvt2HeatmapImg



    # cv2.imwrite('test_region.jpg', cvt2HeatmapImg(label[:,:,0]))
    # cv2.imwrite('test_affinity.jpg', cvt2HeatmapImg(label[:,:,1]))


    # a = Image.fromarray(image)

    # Image.save('tmp/test.jpg', a )


        