"""MPRAGE simulator"""

__all__ = ["mprage"]

import warnings
import numpy as np

import dacite
from dacite import Config

from mrinufft._array_compat import with_torch

from .. import blocks
from .. import ops
from . import base


@with_torch
def mprage(
    nshots,
    flip,
    TR,
    T1,
    T2,
    spoil_inc=117.0,
    sliceprof=False,
    diff=None,
    device="cpu",
    TI=0.0,
    **kwargs,
):
    """
    Simulate a Magnetization Prepared (MP) Rapid Gradient Echo sequence.

    Parameters
    ----------
    nshots : int
        Number of pulses in the inversion block.
    flip : float
        Flip angle in degrees, of shape `(nmodes,)`.
    TR : float
        Repetition time in milliseconds.
    T1 : float or ArrayLike
        Longitudinal relaxation time in milliseconds.
    T2 : float or ArrayLike
        Transverse relaxation time in milliseconds.
    sliceprof : bool or ArrayLike, optional
        Excitation slice profile (i.e., flip angle scaling across the slice).
        If `False`, pulses are non-selective.
        If `True`, pulses are selective but assume an ideal profile.
        If an array, simulates flip angle scaling along the slice. Defaults to `False`.
    spoil_inc : float, optional
        RF spoiling increment in degrees. Defaults to `117.0`.
    diff : str, tuple of str, or None, optional
        Arguments to compute the signal derivative with respect to. Defaults to `None` (no differentiation).
    device : str, optional
        Computational device. Defaults to `"cpu"`.
    TI : float, optional
        Inversion time in milliseconds. Defaults to `0.0`.

    Other Parameters
    ----------------
    simulation_kwargs : dict, optional
        Additional keyword arguments for simulation:

        - nstates : int, optional
            Maximum number of EPG states to retain during simulation. Higher values improve accuracy but reduce performance. Defaults to `10`.
        - max_chunk_size : int, optional
            Maximum number of atoms to simulate in parallel. Larger values increase speed and memory usage. Defaults to `natoms`.
        - nlocs : int, optional
            Maximum number of spatial locations to simulate (e.g., for slice profile effects). Defaults to `15` for slice-selective acquisitions and `1` for non-selective acquisitions.
        - verbose : bool, optional
            If `True`, prints execution time for signal and gradient calculations.

    sequence_kwargs : dict, optional
        Additional keyword arguments for sequence settings:

        - TE : float, optional
            Echo time in milliseconds. Defaults to `0.0`.
        - B1sqrdTau : float
            Pulse energy in `uT**2 * ms`.
        - global_inversion : bool
            Assumes non-selective (`True`) or selective (`False`) inversion. Defaults to `True`.
        - inv_B1sqrdTau : float
            Inversion pulse energy in `uT**2 * ms`.
        - grad_tau : float
            Gradient lobe duration in milliseconds.
        - grad_amplitude : float, optional
            Gradient amplitude in the unbalanced direction, in `mT/m`. Used to compute diffusion and flow effects if `grad_dephasing` is not provided.
        - grad_dephasing : float, optional
            Total gradient-induced dephasing across a voxel in the gradient direction. Used if `grad_amplitude` is not provided.
        - voxelsize : str or array-like, optional
            Voxel size `(dx, dy, dz)` in millimeters. If scalar, assumes isotropic voxel size. Defaults to `None`.
        - grad_orient : str or array-like, optional
            Gradient orientation (`"x"`, `"y"`, `"z"`, or a vector). Defaults to `"z"`.
        - slice_orient : str or array-like, optional
            Slice orientation (`"x"`, `"y"`, `"z"`, or a vector). Ignored if pulses are non-selective. Defaults to `"z"`.

    system_kwargs : dict, optional
        Additional keyword arguments for system settings:

        - B1 : float or array-like, optional
            Flip angle scaling factor (`1.0` corresponds to the nominal flip angle). Defaults to `None`.
        - B0 : float or array-like, optional
            Bulk off-resonance in Hertz. Defaults to `None`.
        - B1Tx2 : float or array-like, optional
            Flip angle scaling factor for the secondary RF mode (`1.0` corresponds to nominal flip angle). Defaults to `None`.
        - B1phase : float or array-like, optional
            B1 relative phase in degrees (`0.0` corresponds to nominal RF phase). Defaults to `None`.

    main_pool_kwargs : dict, optional
        Additional keyword arguments for the main pool:

        - T2star : float or array-like, optional
            Effective relaxation time for the main pool in milliseconds. Defaults to `None`.
        - D : float or array-like, optional
            Apparent diffusion coefficient in `um**2/ms`. Defaults to `None`.
        - v : float or array-like, optional
            Spin velocity in `cm/s`. Defaults to `None`.
        - chemshift : float, optional
            Chemical shift for the main pool in Hertz. Defaults to `None`.

    bloch_mcconnell_kwargs : dict, optional
        Additional keyword arguments for Bloch-McConnell modeling:

        - T1bm : float or array-like, optional
            Longitudinal relaxation time for the secondary pool in milliseconds. Defaults to `None`.
        - T2bm : float or array-like, optional
            Transverse relaxation time for the secondary pool in milliseconds. Defaults to `None`.
        - kbm : float or array-like, optional
            Nondirectional exchange rate between the main and secondary pools, in Hertz. Defaults to `None`.
        - weight_bm : float or array-like, optional
            Relative fraction of the secondary pool. Defaults to `None`.
        - chemshift_bm : float, optional
            Chemical shift for the secondary pool in Hertz. Defaults to `None`.

    magnetization_transfer_kwargs : dict, optional
        Additional keyword arguments for magnetization transfer effects:

        - kmt : float or array-like, optional
            Nondirectional exchange rate between the free and bound pools, in Hertz. Defaults to `None`. If a secondary pool is defined, exchange is between secondary and bound pools (e.g., myelin water and macromolecular pools); otherwise, exchange is between the main and bound pools.
        - weight_mt : float or array-like, optional
            Relative fraction of the bound pool. Defaults to `None`.
    """
    # constructor
    init_params = {
        "flip": flip,
        "TR": TR,
        "T1": T1,
        "T2": T2,
        "diff": diff,
        "device": device,
        "TI": TI,
        **kwargs,
    }

    # get TE
    if "TE" not in init_params:
        TE = 0.0
    else:
        TE = init_params["TE"]

    # get verbosity
    if "verbose" in init_params:
        verbose = init_params["verbose"]
    else:
        verbose = False

    # get selectivity:
    if sliceprof:
        selective_exc = True
    else:
        selective_exc = False

    # add moving pool if required
    if selective_exc and "v" in init_params:
        init_params["moving"] = True

    # check for global inversion
    if "global_inversion" in init_params:
        selective_inv = not (init_params["global_inversion"])
    else:
        selective_inv = False

    # check for conflicts in inversion selectivity
    if selective_exc is False and selective_inv is True:
        warnings.warn("3D acquisition - forcing inversion pulse to global.")
        selective_inv = False

    # inversion pulse properties
    if TI is None:
        inv_props = {}
    else:
        inv_props = {"slice_selective": selective_inv}

    if "inv_B1sqrdTau" in kwargs:
        inv_props["b1rms"] = kwargs["inv_B1sqrdTau"] ** 0.5
        inv_props["duration"] = 1.0

    # excitation pulse properties
    rf_props = {"slice_selective": selective_exc}
    if "B1sqrdTau" in kwargs:
        inv_props["b1rms"] = kwargs["B1sqrdTau"] ** 0.5
        inv_props["duration"] = 1.0

    if np.isscalar(sliceprof) is False:
        rf_props["slice_profile"] = kwargs["sliceprof"]

    # get nlocs
    if "nlocs" in init_params:
        nlocs = init_params["nlocs"]
    else:
        if selective_exc:
            nlocs = 15
        else:
            nlocs = 1

    # interpolate slice profile:
    if "slice_profile" in rf_props:
        nlocs = min(nlocs, len(rf_props["slice_profile"]))
    else:
        nlocs = 1

    # assign nlocs
    init_params["nlocs"] = nlocs

    # unbalanced gradient properties
    grad_props = {}
    if "grad_tau" in kwargs:
        grad_props["duration"] = kwargs["grad_tau"]
    if "grad_dephasing" in kwargs:
        grad_props["total_dephasing"] = kwargs["grad_dephasing"]
    if "voxelsize" in kwargs:
        grad_props["voxelsize"] = kwargs["voxelsize"]
    if "grad_amplitude" in kwargs:
        grad_props["grad_amplitude"] = kwargs["grad_amplitude"]
    if "grad_orient" in kwargs:
        grad_props["grad_direction"] = kwargs["grad_orient"]
    if "slice_orient" in kwargs:
        grad_props["slice_direction"] = kwargs["slice_orient"]

    # check for possible inconsistencies:
    if "total_dephasing" in rf_props and "grad_amplitude" in rf_props:
        warnings.warn(
            "Both total_dephasing and grad_amplitude are provided - using the first"
        )

    # put all properties together
    props = {
        "inv_props": inv_props,
        "rf_props": rf_props,
        "grad_props": grad_props,
        "nshots": nshots,
        "spoil_inc": spoil_inc,
    }

    # initialize simulator
    simulator = dacite.from_dict(MPRAGE, init_params, config=Config(check_types=False))

    # run simulator
    if diff:
        # actual simulation
        sig, dsig = simulator(flip=flip, TR=TR, TI=TI, TE=TE, props=props)

        # prepare info
        info = {"trun": simulator.trun, "tgrad": simulator.tgrad}
        if verbose:
            return sig, dsig, info
        else:
            return sig, dsig
    else:
        # actual simulation
        sig = simulator(flip=flip, TR=TR, TI=TI, TE=TE, props=props)

        # prepare info
        info = {"trun": simulator.trun}
        if verbose:
            return sig, info
        else:
            return sig


# %% utils
spin_defaults = {"T2star": None, "D": None, "v": None}


class MPRAGE(base.BaseSimulator):
    """Class to simulate inversion-prepared Rapid Gradient Echo."""

    @staticmethod
    def sequence(
        flip,
        TR,
        TI,
        TE,
        props,
        T1,
        T2,
        B1,
        df,
        weight,
        k,
        chemshift,
        D,
        v,
        states,
        signal,
    ):
        # parsing pulses and grad parameters
        inv_props = props["inv_props"]
        rf_props = props["rf_props"]
        grad_props = props["grad_props"]
        spoil_inc = props["spoil_inc"]
        npulses = props["nshots"]

        # define preparation
        Prep = blocks.InversionPrep(TI, T1, T2, weight, k, inv_props)

        # prepare RF pulse
        RF = blocks.ExcPulse(states, B1, rf_props)

        # prepare free precession period
        X, XS = blocks.SSFPFidStep(
            states, TE, TR, T1, T2, weight, k, chemshift, D, v, grad_props
        )

        # initialize phase
        phi = 0
        dphi = 0

        # magnetization prep
        states = Prep(states)

        # actual sequence loop
        for n in range(npulses):

            # update phase
            dphi = (phi + spoil_inc) % 360.0
            phi = (phi + dphi) % 360.0

            # apply pulse
            states = RF(states, flip, phi)

            # relax, recover and record signal for each TE
            states = X(states)
            signal[n] = ops.observe(states, RF.phi)

            # relax, recover and spoil
            states = XS(states)

        return ops.susceptibility(signal, TE, df)
