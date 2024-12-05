API References
==============

Base
----
Base classes and routines for MR simulator implementation.

.. autosummary::
   :toctree: generated
   :nosignatures:

   mrsim.base.AbstractModel
   mrsim.base.autocast
   
Parameter configuration
~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated
   :nosignatures:
   
   mrsim.base.prepare_environmental_parameters
   mrsim.base.prepare_single_pool
   mrsim.base.prepare_two_pool_bm
   mrsim.base.prepare_two_pool_mt
   mrsim.base.prepare_three_pool

Extended Phase Graphs
---------------------
Subroutines for Extended Phase Graphs based simulators.

.. autosummary::
   :toctree: generated
   :nosignatures:

   mrsim.epg.states_matrix
   mrsim.epg.get_signal
   mrsim.epg.get_demodulated_signal

RF Pulses
~~~~~~~~~

.. autosummary::
   :toctree: generated
   :nosignatures:

   mrsim.epg.rf_pulse_op
   mrsim.epg.phased_rf_pulse_op
   mrsim.epg.multidrive_rf_pulse_op
   mrsim.epg.phased_multidrive_rf_pulse_op
   mrsim.epg.rf_pulse
   
   mrsim.epg.initialize_mt_sat
   mrsim.epg.mt_sat_op
   mrsim.epg.multidrive_mt_sat_op
   mrsim.epg.mt_sat
    
Relaxation and Exchange
~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated
   :nosignatures:

   mrsim.epg.longitudinal_relaxation_op
   mrsim.epg.longitudinal_relaxation
   mrsim.epg.longitudinal_relaxation_exchange_op
   mrsim.epg.longitudinal_relaxation_exchange
   mrsim.epg.transverse_relaxation_op
   mrsim.epg.transverse_relaxation
   mrsim.epg.transverse_relaxation_exchange_op
   mrsim.epg.transverse_relaxation_exchange
   
Gradient Dephasing
~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated
   :nosignatures:

   mrsim.epg.shift
   mrsim.epg.spoil  
   
Magnetization Prep
~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated
   :nosignatures:
   
    mrsim.epg.adiabatic_inversion   
   
Flow and Diffusion
~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated
   :nosignatures:

   mrsim.epg.diffusion_op
   mrsim.epg.diffusion
   mrsim.epg.flow_op
   mrsim.epg.flow   
   
Signal Models
-------------
Pre-defined signal models.

Analytical
~~~~~~~~~~

.. autosummary::
   :toctree: generated
   :nosignatures:  
   
   mrsim.models.bSSFPModel
   mrsim.models.SPGRModel
   
Iterative
~~~~~~~~~

.. autosummary::
   :toctree: generated
   :nosignatures:  
   
   mrsim.models.FSEModel
   mrsim.models.MP2RAGEModel
   mrsim.models.MPnRAGEModel
   mrsim.models.MRFModel
   
Functional
----------
Functional wrappers for signal models.

Analytical
~~~~~~~~~~

.. autosummary::
   :toctree: generated
   :nosignatures:  
   
   mrsim.bssfp_sim
   mrsim.spgr_sim
    
Iterative
~~~~~~~~~

.. autosummary::
   :toctree: generated
   :nosignatures:  
   
   mrsim.fse_sim
   mrsim.mp2rage_sim
   mrsim.mpnrage_sim
   mrsim.mrf_sim
    
Miscellaneous
-------------
Other simulation utilities.

.. autosummary::
   :toctree: generated
   :nosignatures:

   mrsim.utils.b1rms
   mrsim.utils.slice_prof


