import os
import numpy as np
import glob
import PIL.Image as Image
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import random
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

#%%
class FUNDUS_RETINA(torch.utils.data.Dataset):
    def __init__(self, dataset = "train", transform = [False,False],
                 data_path = "data/Fundus_Retina/dataset_normal/"):
        """ dataset = 'train','val','test'
        
        transform = [Bool,Bool], first argument is translation+scaling
                                 second is rotation
        """
        data_path = data_path+dataset
        self.data_path = data_path
        self.image_paths = glob.glob(data_path + "/images" + "/*.png")
        self.seg_paths = glob.glob(data_path + "/segmentations" + "/*.png")
        self.transform = transform[0]
        self.rot_transform = transform[1]
        return None

    def __len__(self):
        'Returns the total number of samples'
        return len(self.image_paths)
    
    def rotation_transform(self, image, segmentation):
        """ User-defined augmentation which is applied to both the 
            image and segmentation
        """
        if random.random() <= 0.9:
            angle = random.randint(-180, 180)
            image = TF.rotate(image, angle)
            segmentation = TF.rotate(segmentation, angle)
        return image, segmentation
    
    def my_segmentation_transforms(self, image, segmentation):
        """ User-defined augmentation which is applied to both the 
            image and segmentation
        """
        if random.random() <= 0.2:
          translate1 = 0.1
          translate2 = 0.1
          max_dx = translate1 * image.size[0]
          max_dy = translate2 * image.size[1]
          translations = (np.round(np.random.uniform(-max_dx, max_dx)),
                          np.round(np.random.uniform(-max_dy, max_dy)))
          image = TF.affine(image,angle = 0, translate=translations, scale = 1, shear = 0)
          segmentation = TF.affine(segmentation,angle = 0, translate=translations, scale = 1, shear = 0)
        
        if random.random() <= 0.2:
            scale = random.uniform(0.9, 1.1)
            
            image = TF.affine(image,angle = 0, scale=scale, shear=0,translate=(0,0))
            segmentation = TF.affine(segmentation,angle= 0, scale = scale,shear=0,translate=(0,0))
            
        if random.random() <= 0.2:
            image = transforms.ColorJitter(brightness=[0.9,1.1],
                                        contrast = [0.9,1.1],
                                        hue = 0.1)(image)
            
        return image, segmentation 

    def __getitem__(self, idx):
        'Generates one sample of data'
        img_path = self.image_paths[idx]
        seg_path = self.seg_paths[idx]
        
        # Read files
        image = Image.open(img_path)
        segmentation = Image.open(seg_path)

        # Perform transformation to both images
        if self.transform == True:
            image,segmentation = self.my_segmentation_transforms(image,segmentation) 

        if self.rot_transform == True:
            image,segmentation = self.rotation_transform(image,segmentation) 
            
        # Perform transformation
        y = transforms.ToTensor()(segmentation)
        X = transforms.ToTensor()(image)
        return X, y
    
if __name__ == '__main__':
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
    
    batch_size = 64
    data_path = "data/Fundus_Retina/dataset_2000samples/"
    
    trainset = FUNDUS_RETINA(dataset = "train", transform = [True,False], data_path=data_path)
    valset = FUNDUS_RETINA(dataset = "val", transform = [False,False], data_path=data_path)
    testset = FUNDUS_RETINA(dataset = "test", transform = [False,False], data_path=data_path)
    
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(valset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=0)

    X,y = next(iter(test_loader))
    
    # Show 6 or less images
    N_imgs = X.shape[0] if X.shape[0] < 6 else 6
    imgsize = N_imgs*3 if N_imgs*3 < 14 else 14

    f,ax = plt.subplots(2,N_imgs,figsize=(imgsize,5))
    ax = ax.flatten()
    for i in range(N_imgs):
        ax[i].imshow(np.moveaxis(X[i].numpy(),0,2))
        ax[i+N_imgs].imshow(y[i,0].numpy(),'gray')
    plt.show()
    
    
    # #%%
    # img_path = "data/Fundus_Retina/Data_Raw/images/image0243.png"
    # seg_path = "data/Fundus_Retina/Data_Raw/segmentations/segmentation0243.png"
    # # Read files
    # image = Image.open(img_path)
    # segmentation = Image.open(seg_path)
    
    # I = np.array(image)
    # S = np.array(segmentation)
    
    # f,ax = plt.subplots(1,2, figsize=(12,7))
    # ax[0].imshow(I)
    # ax[1].imshow(S,'gray')
    
    # for i in range(2):
    #     ax[i].axis('off')
    #     ax[i].set_aspect('equal')
    # plt.subplots_adjust(wspace=0, hspace=0)
    # plt.show()
    # f.savefig('image3.png', bbox_inches='tight',pad_inches = 0,dpi=300)
    
    
    
    
    
    
    