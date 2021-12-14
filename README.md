# Group-Equivarient-UNet-Segmentation
Master-Course project in image segmentations using group equivariant convolutional neural networks. Specifically the U-Net
architecture is used. Project was carried out in the autumn semester of 2021 at the Technical University of Denmark.


The project is carried out by [Rasmus Tuxen](https://github.com/RTuxen) and supervised by:
- [Anders Bjorholm Dahl](https://orbit.dtu.dk/en/persons/anders-bjorholm-dahl)
- [Vedrana Andersen Dahl](https://orbit.dtu.dk/en/persons/vedrana-andersen-dahl)
- [Patrick Møller Jensen](https://orbit.dtu.dk/en/persons/patrick-m%C3%B8ller-jensen)

Project uses the [General E(2)-Equivariant Steerable CNNs](https://github.com/QUVA-Lab/e2cnn) implementations of group convolutions.

## Collaborators
[Rasmus Tuxen](https://github.com/RTuxen)

## Dataset
The data examined in this project is of Fundus Retina images. The original images and segmentations are from the PRIME-FP20, STARE, DRIVE and CHASEDB1 datasets.
Examples of images with their corresponding segmentations:
![Alt text](experiments/figures/data_image1.png?raw=true "Title")
![Alt text](experiments/figures/data_image2.png?raw=true "Title")

The full pre-processed dataset can be found using the following [link](https://drive.google.com/file/d/178nqFih4HpvjPt4olFQaeLWGIGjfUHIo/view?usp=sharing)

## Experimental Results
E(2)-steerable convolutions can be used as a drop in replacement for the conventional convolutions used in CNNs. Keeping the same training setup and without performing hyperparameter tuning, this leads to significant performance boosts compared to CNN baselines, especially for lower training data:
![Alt text](experiments/figures/Loss.png?raw=true "Title")

The models were designed for a fair comparison such  that the number of parameters of is approximately preserved across all models. A segmentation output of the models look like:
![Alt text](experiments/figures/model_outputs.png?raw=true "Title")


## References
O. Ronneberger et al. U-Net: Convolutional Networks for Biomedical Image Segmentation, [https://arxiv.org/abs/1505.04597](https://arxiv.org/abs/1505.04597)

 
M. Weiler & G. Cesa. General E(2) - Equivariant Steerable CNNs, [https://arxiv.org/abs/1911.08251](https://arxiv.org/abs/1911.08251)

T. S. Cohen & M. Welling. Group Equivariant Convolutional Networks, [https://arxiv.org/abs/1602.07576](https://arxiv.org/abs/1602.07576)

T. S. Cohen & M. Welling. STEERABLE CNNS, [https://arxiv.org/abs/1612.08498](https://arxiv.org/abs/1612.08498)

M. Winkels & T. S. Cohen. 3D G-CNNs for Pulmonary Nodule Detection, [https://arxiv.org/abs/1804.04656](https://arxiv.org/abs/1804.04656)


