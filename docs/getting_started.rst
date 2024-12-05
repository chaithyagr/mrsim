Getting Started
===============

Installing MRSim
-----------------

MRSim is available on PyPi

.. code-block:: sh

    pip install mrsim

Development Version
~~~~~~~~~~~~~~~~~~~

If you want to modifiy the mrsim code base

.. code-block:: sh

    git clone https://github.com/INFN-MRI/mrsim
    pip install -e ./mrsim[test, dev, doc]


Basic Usage
===========
Using MRSim, we can quickly implement and run MR simulations.
We also provide pre-defined simulators for several applications:

.. code-block:: python
    
    import numpy as np
    import mrsim
    
    # generate a flip angle pattern
    flip = np.concatenate((np.linspace(5, 60.0, 300), np.linspace(60.0, 2.0, 300), np.ones(280)*2.0))
    sig, jac = mrsim.mrf_sim(flip=flip, TR=10.0, T1=1000.0, T2=100.0, diff=("T1","T2"))
    
This way we obtained the forward pass signal (``sig``) as well as the jacobian
calculated with respect to ``T1`` and ``T2``.
    

