# Group-Equivarient-UNet-Segmentation
Master-Course project in image segmentations using group equivariant convolutional neural networks. Specifically the U-Net ([paper](https://arxiv.org/abs/1505.04597))
architecture is used. Project was made in the autumn semester of 2021 at the Technical University of Denmark.


The project is carried out by [Rasmus Tuxen](https://github.com/RTuxen) and supervised by:
- [Anders Bjorholm Dahl](https://orbit.dtu.dk/en/persons/anders-bjorholm-dahl)
- [Vedrana Andersen Dahl](https://orbit.dtu.dk/en/persons/vedrana-andersen-dahl)
- [Patrick Møller Jensen](https://orbit.dtu.dk/en/persons/patrick-m%C3%B8ller-jensen)

Project uses the [General E(2)-Equivariant Steerable CNNs](https://github.com/QUVA-Lab/e2cnn) implementations of group convolutions.

## Collaborators
[Rasmus Tuxen](https://github.com/RTuxen)

## Dataset
The data examined in this project is of Fundus Retina images. The original images and segmentations are from the PRIME-FP20, STARE, DRIVE and CHASEDB1 datasets.
![Alt text](experiments/figures/image2.png?raw=true "Title")
![Alt text](experiments/figures/image3.png?raw=true "Title")

The full pre-processed dataset can be found using the following [link](https://drive.google.com/file/d/178nqFih4HpvjPt4olFQaeLWGIGjfUHIo/view?usp=sharing)

## Experimental Results
E(2)-steerable convolutions can be used as a drop in replacement for the conventional convolutions used in CNNs. Keeping the same training setup and without performing hyperparameter tuning, this leads to significant performance boosts compared to CNN baselines, especially for lower training data:
![Alt text](experiments/figures/Loss.png?raw=true "Title")

The models were designed for a fair comparison such  that the number of parameters of is approximately preserved across all models. A segmentation output of the models look like:
![Alt text](experiments/figures/model_outputs.png?raw=true "Title")




## References
O. Ronneberger et al. U-Net: Convolutional Networks for Biomedical Image Segmentation![image](https://user-images.githubusercontent.com/39836677/146071346-608fa249-91ea-484e-aca0-a4f6fc6d812c.png)
 
M. Weiler & G. Cesa. General E(2) - Equivariant Steerable CNNs![image](https://user-images.githubusercontent.com/39836677/146071289-6040ea8f-ccb2-411e-bb2f-1a0514a2687d.png)

T. S. Cohen & M. Welling. Group Equivariant Convolutional Networks![image](https://user-images.githubusercontent.com/39836677/146071362-5304b8c3-b6bf-4f68-ad5f-218a75f863f2.png)

T. S. Cohen & M. Welling. STEERABLE CNNS![image](https://user-images.githubusercontent.com/39836677/146071373-0ab9a0d9-7bc5-4867-b932-10427926a62e.png)

M. Winkels & T. S. Cohen. 3D G-CNNs for Pulmonary Nodule Detection![image](https://user-images.githubusercontent.com/39836677/146071382-291cb277-2a54-4e55-9308-592237c6f98f.png)

