=======================================
NBLE: Natural Background Level estimator
=======================================


**NBLE** is a Python source code for estimating the natural background levels of geochemical species with multiple sources. Basically, the code performs two main tasks: 

1. Produces the morphorlogical structure of the geochemical data using the Kernel Density Estimation (KDE) approach as presented below;

.. image:: https://raw.githubusercontent.com/cosmo-ethz/hide/master/docs/simdata.png
   :alt: Simulated time-ordered-data with **HIDE**.
   :align: center

2. Estimates the NBL by using the Gaussian Mixture Model to decompose the geochemical data into components as displayed below; 

.. image:: https://raw.githubusercontent.com/cosmo-ethz/hide/master/docs/simdata.png
   :alt: Simulated time-ordered-data with **HIDE**.
   :align: center


To run the <<model_util>> script you may install **Anaconda** from the `Official link <https://www.anaconda.com/products/individual>`_. Click on this `template <https://nbviewer.jupyter.org/urls/dl.dropbox.com/s/od7hlgp3101o4wa/nble.ipynb>`_ to obtain the plots above. 