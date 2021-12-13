import numpy as np
import skimage.io
import skimage.transform
import scipy.io
import scipy.ndimage
import matplotlib.pyplot as plt
import glob
"""
This function contains structure tensor relates functions and tests.
The purpose of this is to determine orientation in images.
@author: s173910@student.dtu.dk
"""

def structure_tensor(image, sigma, rho, normalization="none",truncate = 4.0):
    """ Structure tensor for 2D image data
    Arguments:
        image: a 2D array of size N = rows*columns
        sigma: a noise scale, structures smaller than sigma will be 
            removed by smoothing
        rho: an integration scale giving the size over the neighborhood in 
            which the orientation is to be analysed
    Returns:
        an array with shape (3,N) containing elements of structure tensor 
            s_xx, s_yy, s_xy ordered acording to image.ravel()
    Author: vand@dtu.dk, 2019
    """
    
    # computing derivatives (scipy implementation truncates filter at 4 sigma)
    image = image.astype(float);

    Ix = scipy.ndimage.gaussian_filter(image, sigma, order=[1,0], mode='nearest',truncate=truncate)
    Iy = scipy.ndimage.gaussian_filter(image, sigma, order=[0,1], mode='nearest',truncate=truncate)
    
    # integrating elements of structure tensor (scipy uses sequence of 1D)
    Jxx = scipy.ndimage.gaussian_filter(Ix**2, rho, mode='nearest',truncate=truncate)
    Jyy = scipy.ndimage.gaussian_filter(Iy**2, rho, mode='nearest',truncate=truncate)
    Jxy = scipy.ndimage.gaussian_filter(Ix*Iy, rho, mode='nearest',truncate=truncate)
    S = np.vstack((Jxx.ravel(), Jyy.ravel(), Jxy.ravel()));
    if (normalization == "norm"):
        a = S
        b = np.sqrt(S[0,:]**2+2*(S[2,:]**2)+S[1,:]**2)
        S = np.divide(a, b, out=np.zeros_like(a), where=b!=0)
    return S
def eig_special(S):
    """ Eigensolution for symmetric real 2-by-2 matrices
    Arguments:
        S: an array with shape (3,N) containing structure tensor
    Returns:
        val: an array with shape (2,N) containing sorted eigenvalues
        vec: an array with shape (2,N) containing eigenvector corresponding
            to the smallest eigenvalue (the other is orthogonal to the first)
    More:
        See https://en.wikipedia.org/wiki/Eigenvalue_algorithm for calculation
    Author: vand@dtu.dk, 2019
    """

    val = 0.5*(S[0]+S[1]+np.outer(np.array([-1,1]), np.sqrt((S[0]-S[1])**2+4*S[2]**2)))
    vec = np.vstack((-S[2],S[0]-val[0])) # y will be positive
    aligned = S[2]==0 # dealing with diagonal matrices
    vec[:,aligned] = 1-np.argsort(S[0:2,aligned], axis=0)
    vec = vec/np.sqrt(vec[0]**2+vec[1]**2) # normalizing
    return val, vec
def plot_orientations(ax, image, vec, s = 5):
    """ Helping function for adding orientation-quiver to the plot.
    Arguments: plot axes, image shape, orientation, arrow spacing.
    """
    dim = image.shape
    ax.imshow(image,cmap=plt.cm.gray)
    vx = vec[0].reshape(dim)
    vy = vec[1].reshape(dim)
    xmesh, ymesh = np.meshgrid(np.arange(dim[0]), np.arange(dim[1]), indexing='ij')
    ax.quiver(ymesh[s//2::s,s//2::s],xmesh[s//2::s,s//2::s],vy[s//2::s,s//2::s],vx[s//2::s,s//2::s],color='r',angles='xy')
    ax.quiver(ymesh[s//2::s,s//2::s],xmesh[s//2::s,s//2::s],-vy[s//2::s,s//2::s],-vx[s//2::s,s//2::s],color='r',angles='xy')
    return None

def angleHistogram(ax,vec,n_bins=120, vlines = False):
    """
    Plots a histogram of angles from an array of unit vectors
    Parameters
    ----------
    ax : pyplot axis
    vec : numpy (2,N) array
        array of unit vectors
    n_bins : Scalar, optional
        Number of bins in histogram. The default is 120.
    Returns
    -------
    None.
    """
    # Compute angles from orientation vectors, angles are in range 0 to pi
    angles = np.arctan2(vec[1], vec[0]) # angles from 0 to pi
    distribution = np.histogram(angles, bins=n_bins, range=(0.0, np.pi))[0]
    
    bin_centers = (np.arange(n_bins)+0.5)*np.pi/n_bins # half circle (180 deg)
    bin_centers = np.roll(bin_centers,len(bin_centers)//2,axis=0)
    colors = plt.cm.hsv(bin_centers/np.pi)
    colors = np.roll(colors,len(distribution)//2,axis=0)
    
    ax.bar(bin_centers, distribution, width = np.pi/n_bins, color = colors)
    ax.set_xlabel('angle')
    ax.set_xlim([0,np.pi])
    ax.set_aspect(np.pi/ax.get_ylim()[1])
    ax.set_xticks([0,np.pi/4,np.pi/2,3*np.pi/4,np.pi])
    ax.set_xticklabels(['0',r'$\frac{\pi}{4}$',r'$\frac{\pi}{2}$',
                        r'$\frac{3\pi}{4}$',r'$\pi$'])
    ax.set_ylabel('count')
    ax.set_title('Histogram over angles', pad=20)
    plt.gca().invert_xaxis()
    
    ymax = np.max(distribution)+0.1*np.max(distribution)+1
    ax.set_ylim([0,ymax])
    
    if vlines:
        ax.vlines(np.pi/8,0,ymax,linewidth=3, color='k')
        ax.vlines(3*np.pi/8,0,ymax,linewidth=3, color='k')
        ax.vlines(5*np.pi/8,0,ymax,linewidth=3, color='k')
        ax.vlines(7*np.pi/8,0,ymax,linewidth=3, color='k')    
    return None
    
def get_range(angle, c = 8):
    """ Creates a range in the interval [0,pi] that is covered
    with an angle of [angle - pi/c , angle + pi/c]
    NB, when angle is too close to 0 or pi it returns a list with a 
    different length - this is an inellegant quick fix.
    """
    if angle-np.pi/c < 0:
        angle_range = [[0,angle+np.pi/c],
                       [np.pi+angle-np.pi/c,np.pi],[]]
    elif angle+np.pi/8 > np.pi:
        angle_range = [[angle-np.pi/c,np.pi],
                       [0,-(np.pi-(angle+np.pi/c))],[]]
    else:
        angle_range = [angle-np.pi/c,angle+np.pi/c]
    return angle_range

def get_dominant_orientation(segmentation, n_bins = 120):
    """ Gets dominant orientation of a segmentation given a vector
    of structure tensor orientations with 
    pi/n_cutoff different angles.
    
    Assumes the segmentation to contain [0,255]
    """
    
    if all(np.unique(segmentation) != [0,255]):
        raise ValueError("Image does contain other values than [0,255]")
    
    # ST parameters
    truncate = 4.0
    sigma = 0.5 # sigma = 0.5 -> 5x5 kernel
    rho = 10
    
    # Compute structure tensors
    S = structure_tensor(segmentation, sigma, rho,"norm",truncate)
    val,vec = eig_special(S)
    
    # Remove orientation from background
    vx = vec[0].reshape(segmentation.shape)
    vy = vec[1].reshape(segmentation.shape)
    vx = vx[segmentation==255]
    vy = vy[segmentation==255]
    vec_bk_removed = np.concatenate((vx[None,:],vy[None,:]),axis=0)
    
    # Compute angles from orientation vectors, angles are in range 0 to pi
    angles = np.arctan2(vec_bk_removed[1], vec_bk_removed[0])
    
    # Get bins from angles
    distribution = np.histogram(angles, bins=n_bins, range=(0.0, np.pi))[0]
    distribution = np.roll(distribution,len(distribution)//2,axis=0)
    bin_centers = (np.arange(n_bins)+0.5)*np.pi/n_bins # half circle (180 deg)
    
    # We split orientations into 4 main ranges:
    range1 = ((0 <= bin_centers) & (bin_centers <= 1*np.pi/8)) | ((7*np.pi/8 <= bin_centers) & (bin_centers <= 8*np.pi/8))
    range2 = (1*np.pi/8 <= bin_centers) & (bin_centers <= 3*np.pi/8)
    range3 = (3*np.pi/8 <= bin_centers) & (bin_centers <= 5*np.pi/8)
    range4 = (5*np.pi/8 <= bin_centers) & (bin_centers <= 7*np.pi/8)
    ranges = [range1,range2,range3,range4]
    
    # Find out how many bins fall into each range
    angle_totals = np.zeros((4,))
    for i,range_ in enumerate(ranges):
        angle_totals[i] = np.sum(distribution[range_])
    return angle_totals, vec, vec_bk_removed

def print_dom_angle(angle_totals):
    """ Prints the dominant angle 
    """
    dom_angle = np.argmax(angle_totals)
    dom_degree = round(angle_totals[dom_angle]/np.sum(angle_totals),3)
    
    message = "Dominant angle is the range "
    if dom_angle == 0:
        message += "[0,pi/8] U [7pi/8,pi]"
    elif dom_angle ==1:
        message += "[pi/8,3pi/8]"
    elif dom_angle ==2:
        message += "[3pi/8,5pi/8]"
    else:
        message += "[5pi/8,7pi/8]"
    message += "\nWith a dominance degree of " + str(dom_degree)
    print(message)
    return None

if __name__ == '__main__':   
    # Load some image to test functions
    data_path = '../../datasets/data/Fundus_Retina/Data_Unsorted/'
    # image_paths = glob.glob(data_path + "images/*")
    # seg_paths = glob.glob(data_path + "segmentations/*")
    
    # # Load random segmentation
    # i = np.random.randint(0,len(seg_paths),(1,))[0]
    # segmentation = skimage.io.imread(seg_paths[i])
    
    # n_bins = 120    
    # angle_totals, vec, vec_bk_removed = get_dominant_orientation(segmentation, n_bins)

    # # visualization
    # f,ax = plt.subplots(1,3,figsize=(15,10))
    # ax = ax.flatten()
    # ax[0].imshow(segmentation,cmap=plt.cm.gray)
    # ax[0].set_title('Segmentation')
    # angleHistogram(ax[1],vec_bk_removed,n_bins, vlines = True)
    # intensity_rgba = plt.cm.gray(segmentation)
    # orientation_st_rgba = plt.cm.hsv((np.arctan2(vec[1], vec[0])/np.pi).reshape(segmentation.shape))
    # im = (0.5+0.5*intensity_rgba)*orientation_st_rgba
    # im[segmentation==0,:3] = 0
    # ax[2].imshow(im)
    # ax[2].set_title("Orientation as color on segmentation")
    # plt.show() 
    
    # print_dom_angle(angle_totals)
