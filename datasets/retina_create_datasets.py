import os
import numpy as np
import glob
import PIL.Image as Image
import skimage.io
from tqdm import tqdm
import sys 
if sys.path[-1]=='..':
    sys.path = sys.path[:-1]
sys.path.append('..')
import lib.utils as util

def create_retina_dataset(IMG_SIZE, N_train = 1000,
                   split = [0.7, 0.15, 0.15],
                   orientation_threshold = 0.45,
                   data_path = "data/Fundus_Retina/Data_Unsorted/",
                   DATASET_PATH = "data/Fundus_Retina/dataset_normal/"
                   ):
    """ This function creates the trainable-datasets
    Give it a required image size and number of total training images
    along with the split of datasets

    Parameters
    ----------
    IMG_SIZE : TYPE
        DESCRIPTION.
    N_train : scalar, optional
        DESCRIPTION. The default is 1000.
    split : list, optional
        DESCRIPTION. The default is [0.7, 0.15, 0.15].
    orientation_threshold : scalar, optional
        DESCRIPTION. The default is 0.45.
    data_path : string, optional
        DESCRIPTION. The default is "data/Fundus_Retina/Data_Raw/".
    DATASET_PATH : string, optional
        DESCRIPTION. The default is "data/Fundus_Retina/dataset_normal/".

    Returns
    -------
    None

    """
    
    # Get paths to data
    image_paths = glob.glob(data_path + "images/*")
    seg_paths = glob.glob(data_path + "segmentations/*")
    N_images = len(image_paths)    
    # Sizes of full images:
    # (2336, 3504, 3) has 44 images
    # (1000, 1154, 3) has 99 images
    # (960, 999, 3)   has 27 images
    # (584, 565, 3)   has 39 images
    # (605, 700, 3)   has 19 images
    # (1500, 2100, 3) has 14 images
    
    # Define splits for new dataset
    splits = ["train/", "val/", "test/"]
    
    # Split the raw images
    all_idx = np.arange(0,N_images)
    np.random.shuffle(all_idx)
    train_idx = all_idx[:int(split[0]*N_images)]
    val_idx = all_idx[int(split[0]*N_images):int((split[0]+split[1])*N_images)]
    test_idx = all_idx[int((split[0]+split[1])*N_images):]
    
    # Number of images in validation and test sets
    N_val = int(N_train*split[1]/split[0])
    N_test = int(N_train*split[2]/split[0])
    if N_val < 200 or N_test < 200:
        N_val = 200
        N_test = 200

    for split_set in splits:
        
        # Create directories
        if not os.path.exists(DATASET_PATH+split_set):
            os.makedirs(DATASET_PATH+split_set + "images")
            os.makedirs(DATASET_PATH+split_set + "segmentations")
        
        # Remove all previous files in target folder
        image_files = glob.glob(DATASET_PATH+split_set + "images/*")
        segmentation_files = glob.glob(DATASET_PATH+split_set + "segmentations/*")
        if image_files:
            for f in image_files:
                os.remove(f)
        if segmentation_files:
            for f in segmentation_files:
                os.remove(f)
        
        # Get correct indicies for split
        if split_set == "train/":
            split_idx = train_idx
            N_total = N_train
        elif split_set == "val/":
            split_idx = val_idx
            N_total = N_val
        elif split_set == "test/":
            split_idx = test_idx
            N_total = N_test
        else:
            raise ValueError("Split name doesn't match")
           
        pbar = tqdm(total=N_total, desc=split_set[:-1]+"set", 
                    position=0, leave=True)
        num_img = 0
        while num_img < N_total:
            
            # Loop over images in set
            for idx in split_idx:
                # break
                img_path = image_paths[idx]
                seg_path = seg_paths[idx]
                
                # Read image
                Large_I = skimage.io.imread(img_path)
                Large_S = skimage.io.imread(seg_path)
                
                # Make sure the image and segmentation correspond
                assert Large_I.shape[:-1] == Large_S.shape
                
                # Crop parameters
                large_im_shape = np.array(list(Large_I.shape[:2]))
                margin = large_im_shape-IMG_SIZE
                
                # Number of patches to take from each image
                # Take more from larger images
                if N_total > 500:
                    n_patch = Large_I.shape[1]//400
                else:
                    n_patch = 1
                
                for _ in range(n_patch):
                    # Select a random crop from large volume
                    crop = np.random.randint([0,0],margin+1)
                    # Get patch
                    I, S = get_patch(Large_I, Large_S, IMG_SIZE, crop) 
                    
                    # Skip boring mostly black images
                    if np.sum(I == 0)/np.prod(I.shape) > 0.4:
                        continue
                    
                    # Get orientation angles from segmentation
                    angle_totals, _, _ = util.get_dominant_orientation(S ,n_bins=120)
                    # Get dominant orientation
                    dom_angle = np.argmax(angle_totals)
                    
                    # How dominant is that angle?
                    if np.sum(angle_totals) == 0:
                        continue
                    dom_degree = angle_totals[dom_angle]/np.sum(angle_totals)
                    
                    # Criteria to not use the found image
                    condition1 = dom_degree <= orientation_threshold
                    condition2 = (split_set != "test/") and (dom_angle not in [0,1])
                    condition3 = (split_set == "test/") and (dom_angle not in [2,3])
                    if any([condition1,condition2,condition3]):
                        continue
                    
                    # Save images
                    im = Image.fromarray(I)
                    im.save(DATASET_PATH+split_set + "images/" + f"image{num_img:06d}.png")
                    seg = Image.fromarray(S)
                    seg.save(DATASET_PATH+split_set + "segmentations/" + f"segmentation{num_img:06d}.png")
                    num_img += 1
                    pbar.update(1)
                    if num_img >= N_total:
                        break
                if num_img >= N_total:
                    break
            if num_img >= N_total:
                break
        pbar.close()
    return None

def get_patch(image, segmentation, IMG_SIZE, crop):
    """ Takes a patch from an image and its corresponding segmentation """
    I = image[crop[0]:crop[0]+IMG_SIZE,
                    crop[1]:crop[1]+IMG_SIZE]
    S = segmentation[crop[0]:crop[0]+IMG_SIZE,
                crop[1]:crop[1]+IMG_SIZE] 
    return I,S


if __name__ == '__main__':
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
    
    # Set seed
    seed = 50
    np.random.seed(seed)
    
    # Define parameters for dataset
    IMG_SIZE = 128 # (IMG_SIZE,IMG_SIZE,3) is the shape of images
    N_train = 2000 # Number of images in training set
    split = [0.7, 0.15, 0.15] # train/val/test split
    orientation_threshold = 0.45
    raw_data_path = "data/Fundus_Retina/Data_Raw/"
    DATASET_PATH = "data/Fundus_Retina/dataset_2000samples/"
    

    # Create dataset
    create_retina_dataset(IMG_SIZE, N_train, split,
                              orientation_threshold, raw_data_path, DATASET_PATH)

