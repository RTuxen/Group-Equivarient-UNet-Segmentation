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
The data examined in this project is of Fundus Retina images
- Indsæt billede af retina

## Experimental Results

E(2)-steerable convolutions can be used as a drop in replacement for the conventional convolutions used in CNNs. Keeping the same training setup and without performing hyperparameter tuning, this leads to significant performance boosts compared to CNN baselines (values are test errors in percent):

The models without * are for a fair comparison designed such that the number of parameters of the baseline is approximately preserved while models with * preserve the number of channels, and hence compute. For more details


![Alt text](experiments/figures/model_outputs.png?raw=true "Title")

- Indsæt link til dataen
![plot](experiments/figures/Loss.png?raw=true "Title")

## References
Diederik P. Kingma & Max Welling: An Introduction to Variational Autoencoders, arXiv:1906.02691

Carl Doersch: Tutorial on Variational Autoencoders, arXiv:1606.05908

Casper Kaae Sønderby, Tapani Raiko, Lars Maaløe, Søren Kaae Sønderby & Ole Winther, Ladder Variational Autoencoders, arXiv:1602.02282

Karol Gregor, Ivo Danihelka, Alex Graves, Danilo Jimenez Rezende & Daan Wierstra: DRAW A Recurrent Neural Network For Image Generation, arXiv:1502.04623

