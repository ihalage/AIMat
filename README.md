# AIMat
### An AI based software to predict the dielectric properties of composite materials. Implemented for 5G materials discovery.

The goal is to predict the dielectric constant, Quality factor (1/loss tangent) and the temperature coefficient of resonance frequency of `(1-x)A - xB` type alloyed compositions where A and B are two known dielectric materials and x is the mole fraction of B. Predictions are made by three combined deep neural networks (see `images` directory). 5G materials require a high quality factor (low loss) and moderate dielectric constant (around 5), and a temperature coefficient near zero. A user friendly software is developed so those who are interested can choose constituent materials from hundreds of available materials in our database and get an idea of what will the properties look like when the selected two materials are mixed at a given ratio. Additionally, users can set target properties and apply genetic algorithm optimisation provided in the repository to inversely discover new materials that fit their purpose.

![AIMat_demo](https://user-images.githubusercontent.com/32927933/132994013-9162b87c-aa73-4500-83de-b0e1b3d7e26f.gif)

# Dependencies
Since this is a project done back in 2018-2019, it requires python-2.7. 
* tensorflow
* keras
* PyQt4
* Tkinter
* numpy
* pandas
* sklearn
* matplotlib

In order to launch AIMat GUI, please run;

    python aimat_gui.py
