MRSim
=====

MRSim is a pure Pytorch-based MR simulator, including analytical and EPG model.

|Coverage| |CI| |CD| |License| |Codefactor| |Sphinx| |PyPi| |Black| |PythonVersion|

.. |Coverage| image:: https://infn-mri.github.io/mrsim/_static/coverage_badge.svg
   :target: https://infn-mri.github.io/mrsim

.. |CI| image:: https://github.com/INFN-MRI/mrsim/workflows/CI/badge.svg
   :target: https://github.com/INFN-MRI/mrsim

.. |CD| image:: https://github.com/INFN-MRI/mrsim/workflows/CD/badge.svg
   :target: https://github.com/INFN-MRI/mrsim

.. |License| image:: https://img.shields.io/github/license/INFN-MRI/mrsim
   :target: https://github.com/INFN-MRI/mrsim/blob/main/LICENSE.txt

.. |Codefactor| image:: https://www.codefactor.io/repository/github/INFN-MRI/mrsim/badge
   :target: https://www.codefactor.io/repository/github/INFN-MRI/mrsim

.. |Sphinx| image:: https://img.shields.io/badge/docs-Sphinx-blue
   :target: https://infn-mri.github.io/mrsim

.. |PyPi| image:: https://img.shields.io/pypi/v/mrsim
   :target: https://pypi.org/project/mrsim

.. |Black| image:: https://img.shields.io/badge/style-black-black

.. |PythonVersion| image:: https://img.shields.io/badge/Python-%3E=3.10-blue?logo=python&logoColor=white
   :target: https://python.org

Features
--------
MRSim contains tools to implement parallelized and differentiable MR simulators. Specifically, we provide

1. Automatic vectorization of across multiple atoms (e.g., voxels).
2. Automatic generation of forward and jacobian methods (based on forward-mode autodiff) to be used in parameter fitting or model-based reconstructions.
3. Support for custom manual defined jacobian methods to override auto-generated jacobian.
4. Support for advanced signal models, including diffusion, flow, magnetization transfer and chemical exchange.
5. GPU support.

Installation
------------

MRSim can be installed via pip as:

.. code-block:: bash

    pip install mrsim

Basic Usage
-----------

Development
~~~~~~~~~~~

If you are interested in improving this project, install MRSim in editable mode:

.. code-block:: bash

    git clone git@github.com:INFN-MRI/mrsim
    cd mrsim
    pip install -e .[dev,test,doc]


Related projects
----------------

This package is inspired by the following excellent projects:

- epyg <https://github.com/brennerd11/EpyG>
- sycomore <https://github.com/lamyj/sycomore/>
- mri-sim-py <https://somnathrakshit.github.io/projects/project-mri-sim-py-epg/>
- ssfp <https://github.com/mckib2/ssfp>
- erwin <https://github.com/lamyj/erwin>

