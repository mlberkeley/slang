# System for Learned Artificial Narrative Generation
The SLANG project is a project with the goal of achieving a program that can write acceptable fiction by using deep learning techniques on colloquia of narrative texts.

## Requirements
Python 3.X
Tensorflow 1.0+
Numpy
NLTK

## Instructions for use
In the src folder, to run the vsem autoencoder, type in the command `python3 fivesent.py`. To modify paramaters such as whether to load or train, the save directory for the model, and model hyperparameters, open up the `fivesent.py` file and modifiy the `params` dictionary at the top. (*TODO*: move the dictionary out as a seperate yaml file)

To run the bstm autoencoder, follow the same commands as above, instead calling `python3 fivesent_bstm.py`, etc.
