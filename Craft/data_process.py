import numpy as np 
import cv2 
import h5py 

from .imgproc import cvt2HeatmapImg, loadImage

DEBUG = False

BOX_EXPAND = 1.3

def getIntersect(rect):

    x1,y1,x3,y3,x2,y2,x4,y4 = rect.reshape(-1)

    d = ((x1 - x2) * (y3 - y4)) - ((y1 - y2) * (x3 - x4))

    if d :
        x = ((x1*y2 - y1*x2) * (x3 - x4) - (x1 - x2) * (x3*y4 - y3*x4)) / d   
        y = ((x1*y2 - y1*x2) * (y3 - y4) - (y1 - y2) * (x3*y4 - y3*x4)) / d
        return np.array([x,y], dtype=np.float32)
    else :
        print('rect :', rect)
        return rect[0]

class LabelTransformer():
    """
    根据字符级标注得到图片的region_map和affinity_map 

    gt 格式为每行字符级别标注 p1_x0,p1_y0 - p1_x3,p1_y3,...,pn_x0 - pn_y3, line_txt 
    """

    def __init__(self, kernel_size = 51, sigma=-1):
        self.kernel = self.get2dGuassionKernel(kernel_size, sigma)
        self.dst = np.array([
            [0, 0],
            [self.kernel.shape[1] - 1, 0],
            [self.kernel.shape[1] - 1, self.kernel.shape[0] - 1],
            [0, self.kernel.shape[0] - 1]], dtype=np.float32)

    def get2dGuassionKernel(self, kernel_size, sigma):
        kx = cv2.getGaussianKernel(kernel_size,sigma)
        ky = cv2.getGaussianKernel(kernel_size,sigma)
        kernel=np.multiply(kx,np.transpose(ky)) 
        norm_k = cv2.normalize(kernel ,None , 0, 1, cv2.NORM_MINMAX)
        
        return norm_k 

    def warpTransformKernel(self, image, pts):
        
        
        max_y, max_x = image.shape[:2]
        pts  = pts.astype(np.float32)
        intersect = getIntersect(pts)
        pts = (pts- intersect) *BOX_EXPAND + intersect

        M = cv2.getPerspectiveTransform(self.dst, pts)
        warped = cv2.warpPerspective(self.kernel, M, (max_x, max_y))

        return warped

    def procPolyScore(self, image, rects ):
        
        region_score = np.zeros(image.shape[:2], dtype=np.float32)
        
        for rect in rects :
            warp_gausion = self.warpTransformKernel(image, rect)

            # region_score = np.concatenate([[region_score, warp_gausion]], axis=0).max(axis=0)
            region_score = np.max([region_score, warp_gausion], axis=0)
        
        return region_score

    def procAffinityPolyLine(self, polys):

        """
        polys: nparray(n,4,2) n个字符框

        return : nparray(n-1,4,2) Affinity框
        """
        
        n = polys.shape[0]
        poly_centers = np.zeros((n,2), dtype=np.float32)
        for i in range(n):
            poly_centers[i] = getIntersect(polys[i])


        tri_centers = np.zeros((n,2,2), dtype=np.float32)
        up_polys = polys[:,0:2,:]
        down_polys = polys[:,2:4,:]

        tri_centers[:,0,:] = (up_polys.sum(axis=1) + poly_centers) / 3 
        tri_centers[:,1,:] = (down_polys.sum(axis=1) + poly_centers) / 3 

        
        affinity_rects = np.zeros((n-1,4,2), dtype=np.float32)

        affinity_rects[:,0,:] = tri_centers[:n-1,0,:]
        affinity_rects[:,1,:] = tri_centers[1:,0,:]
        affinity_rects[:,2,:] = tri_centers[1:,1,:]
        affinity_rects[:,3,:] = tri_centers[:n-1,1,:]

        return affinity_rects 

    def procPicture(self, image, region_poly_lines ):

        affinity_lines = []
        for line in region_poly_lines :
            affi_line = self.procAffinityPolyLine(line)
            affinity_lines.append(affi_line)

        affinity_polys = np.concatenate(affinity_lines, axis=0)
        region_polys = np.concatenate(region_poly_lines, axis=0)


        region_score = self.procPolyScore(image, region_polys)
        affinity_score = self.procPolyScore(image, affinity_polys)

        if DEBUG:
            region_score = cvt2HeatmapImg(region_score)
            affinity_score = cvt2HeatmapImg(affinity_score)



        return region_score, affinity_score


def getGtRects(gt_path):
    gts = open(gt_path, encoding='utf-8').read().split('\n')
    rects = []
    for line in gts :
        if line == '':
            continue 
        line = line.split(',')
        line = [int(x) for x in line ]
        # rect = [line[0:2],line[2:4],line[4:6],line[6:8]]
        rect = np.array(line, dtype=np.float32)
        rect = rect.reshape(-1,4,2)
        rects.append(rect)
        

    rects = np.array(rects)

    return rects 

def resizeMap(image, region_score, affinity_score, canvas_size):
    
    ratio =  max(image.shape[:2]) / canvas_size
    img_h, img_w = image.shape[:2]
    resized_image = cv2.resize(image, (int(img_w/ratio),int(img_h/ratio)),interpolation=cv2.INTER_LINEAR)
    resized_region = cv2.resize(region_score, (int(img_w/2/ratio),int(img_h/2/ratio)),interpolation=cv2.INTER_LINEAR)
    resized_affinity = cv2.resize(affinity_score, (int(img_w/2/ratio),int(img_h/2/ratio)),interpolation=cv2.INTER_LINEAR)

    return resized_image, resized_region, resized_affinity

    

def convertCTW(img_root ,save_path ,canvas_size = 1024):

    import os 
    import os.path as osp 
    

    label_size = canvas_size // 2 

    dataset_root = h5py.File(save_path, 'w')
    img_list = os.listdir(img_root)
    lt = LabelTransformer()


    counter = 0
    for i, name in enumerate(img_list):
        if not name.endswith('.jpg'):
            continue 

        counter +=1 
        print('{}|{}'.format(counter,name))


        
        name_raw = name.split('.')[0]
        name_group = dataset_root.create_group(name_raw)
        image = loadImage(osp.join(img_root, name))
        
        gt_rects = getGtRects(osp.join(img_root, name_raw+'.txt'))


        region_score, affinity_score = lt.procPicture(image, gt_rects)

        resize_image = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
        resize_label = np.zeros((label_size, label_size, 2), dtype= np.float32)

        image, region_score, affinity_score = resizeMap(image, region_score, affinity_score, canvas_size)

        resize_image[0:image.shape[0],0:image.shape[1],:] = image 
        resize_label[0:region_score.shape[0],0:region_score.shape[1],0] = region_score
        resize_label[0:affinity_score.shape[0],0:affinity_score.shape[1],1] = affinity_score

        name_group.create_dataset('image', data= resize_image)
        name_group.create_dataset('label', data= resize_label)

        # cv2.imwrite('test.jpg', cv2.cvtColor(resize_image, cv2.COLOR_RGB2BGR))
        # cv2.imwrite('test_region.jpg', cvt2HeatmapImg(resize_label[:,:,0]))
        # cv2.imwrite('test_affinity.jpg', cvt2HeatmapImg(resize_label[:,:,1]))
        

        # if flag :
            
        #     cv2.imwrite(osp.join(save_path, 'test.jpg'),cv2.cvtColor(resize_image, cv2.COLOR_RGB2BGR))
        #     cv2.imwrite(osp.join(save_path, 'test_region.jpg'),cvt2HeatmapImg(resize_label[:,:,0]))
        #     cv2.imwrite(osp.join(save_path, 'test_affinity.jpg'),cvt2HeatmapImg(resize_label[:,:,1]))
            
        #     return 



        
        

if __name__ == '__main__':
    import os 
    import os.path as osp 
    img_root = '/ai/local/menglc/ctw/tmp'
    img_list = os.listdir(img_root)
    lt = LabelTransformer()


    # name = '0001579.jpg'
    # name_raw = '0001579'
    # image = loadImage(osp.join(img_root, name))
    # gt_rects = getGtRects(osp.join(img_root, name_raw+'.txt'))
    # region_score, affinity_score = lt.procPicture(image, gt_rects)
    # print(region_score.shape)

    convertCTW('/ai/local/menglc/ctw/train', '/ai/local/menglc/CRAFT_dataset/ctw_train.h5')
    # convertCTW('/ai/local/menglc/SynthData/tmp', '/ai/local/menglc/SynthData/test/ST_train.h5')


    # for name in img_list:
    #     if name.endswith('.txt'):
    #         continue 
    #     print(name)
    #     image = cv2.imread(os.path.join(img_root,name))
        

    #     gts = open(os.path.join(img_root,name.split('.')[0]+'.txt'), encoding='utf-8').read().split('\n')
    #     rects = []
    #     for line in gts :
    #         if line == '':
    #             continue 
    #         line = line.split(',')[:-1]
    #         line = [int(x) for x in line ]
    #         # rect = [line[0:2],line[2:4],line[4:6],line[6:8]]
    #         rect = np.array(line, dtype=np.float32)
    #         rect = rect.reshape(-1,4,2)
    #         rects.append(rect)

    #     rects = np.array(rects)

        
    #     region_img, affinity_img = lt.procPicture(image, rects)  
    #     print( np.abs(region_img- affinity_img).max())
    #     cv2.imwrite(os.path.join(img_root,'{}_reg.jpg'.format(name.split('.')[0])), region_img)
    #     cv2.imwrite(os.path.join(img_root,'{}_aff.jpg'.format(name.split('.')[0])), affinity_img)

    #     mask_img = 0.8*image + 0.1*region_img + 0.1*affinity_img 
    #     cv2.imwrite(os.path.join(img_root,'{}_res.jpg'.format(name.split('.')[0])), mask_img)
