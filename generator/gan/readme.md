Install environment from `utrgan_environment_new.yml`

The usage of this method is hampered by the implementations of UTRGAN being with TensorFlow (+Keras), while PARADE is implemented with PyTorch (+Lightning).

To obtain generated sequences for 5' or 3'UTRs with this method, you are to do the following steps:

1. Train the GAN model on UTR dataset -- use `1_UTRGAN_[53]_train.ipynb` notebook (skip it if you want to use pre-trained weights)
2. Generate the sequences using the trained GAN -- use `2_UTRGAN_[53]_generate.ipynb` notebook
3. Score the generated sequences using PARADE -- use `3_sequence_scoring_UTR[53].ipynb` notebook
4. Optimize the GAN input vector -- use `4_RNAregressor_optimization_UTR[53].ipynb` notebook
