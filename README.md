# Readme
* `classifierclass_usage.ipynb` is a jupyter-notebook, it has tensorboard visualization and some links for other sources. Use this on a personal machine.

* `classifierclass_usage_keras_main_script.py` and `classifierclass_usage_numpy_main_script.py` are script files to be used when running on a cluster.

* **data/** folder contains data for `EMNIST` and `MNIST`, if you don't find data in data folder then download it from internet and place it in the data folder, if the folder doesn't exist create one.
 * **EMNSIT** can be found [at](http://greg-cohen.com/datasets/emnist/)
 * **MNIST** can be found [at](http://deeplearning.net/tutorial/gettingstarted.html)
 
# Suggested Exercises

## Exercise 1

* Try to run the notebooks `classifierclass_usage.ipynb`, `train_using_numpy_arrays.ipynb' and complete the exrercises given within.

# Known issues with windows
* All the codes were tested with `tensorflow 1.15` and `Keras 2.2.4` in `Python 2.7` and `Python 3.7.5` on **Linux** however when you try to run on windows you might encounter issues like below:

## NBEXTENSION ISSUE:
* If you don't have `nbextension` tab visible in your notebook then open anaconda prompt and run the following

 * `pip install jupyter_contrib_nbextensions`
 * `jupyter contrib nbextension install --user`
 * `jupyter nbextension enable varInspector/main`

* Enable table of contents from nbextension tab once you start your notebook

## H5PY ISSUE:
* Another problem you might have is:

* `AttributeError: module 'h5py' has no attribute 'Group'` when you try to run something with keras in windows

* So, run the following to fix this by opening anaconda command prompt in admin mode

 * `pip uninstall h5py` 
 * `pip install h5py==2.10.0`


## VISUALIZE TENSORBOARD

* Go to the directory where you're logging your file by using the command `cd "your/logging/directory"`. Replace `your/logging/directory"` with the path to your tensorflow log files. 

* And then run the following `tensorboard --logdir "your logging directory name here" `

* And then tenosrboard will print a link to the screen that might looks like  http://ENG40xxx:6006/
* Copy the link and paste it in your browser then you should be able to see the various trends for the models that
are being trained.

