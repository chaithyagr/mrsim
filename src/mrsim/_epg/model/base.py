"""Base Simulation Class"""

__all__ = ["BaseSimulator"]

import gc
import inspect
import time
import typing as t

from functools import partial, wraps

import numpy.typing as npt
import torch
from torch.func import jacfwd, vmap

from .. import ops

eps = torch.finfo(torch.float32).eps
spin_properties = (
    "T1",
    "T2",
    "T2star",
    "chemshift",
    "D",
    "v",
    "T1bm",
    "T2bm",
    "chemshift_bm",
    "kbm",
    "weight_bm",
    "kmt",
    "weight_mt",
    "B0",
    "B1",
    "B1Tx2",
    "B1phase",
)
allowed = ("T1", "T2", "k", "weight", "chemshift", "D", "v", "B1", "df", "props")


def inspect_signature(input):
    return list(inspect.signature(input).parameters)


def jacadapt(func):
    @wraps(func)
    def wrapper(*args):
        # Replace real values with complex ones for signal and states
        args = list(args)
        args[-1] = real2complex(args[-1], "signal")
        args[-2] = real2complex(args[-2], "states")

        # Call the original function
        output = func(*args)

        # Convert the output back to real values
        return complex2real(output)

    return wrapper


def real2complex(input, what):
    if what == "signal":
        return input["real"] + 1j * input["imag"]
    elif what == "states":
        F = input["F"]["real"] + 1j * input["F"]["imag"]
        Z = input["Z"]["real"] + 1j * input["Z"]["imag"]
        out = {"F": F, "Z": Z}

        if "moving" in input:
            Fmoving = input["moving"]["F"]["real"] + 1j * input["moving"]["F"]["imag"]
            Zmoving = input["moving"]["Z"]["real"] + 1j * input["moving"]["Z"]["imag"]
            out["moving"] = {}
            out["moving"]["F"] = Fmoving
            out["moving"]["Z"] = Zmoving

        if "Zbound" in input:
            Zbound = input["Zbound"]["real"] + 1j * input["Zbound"]["imag"]
            out["Zbound"] = Zbound

        return out


def complex2real(input):
    return torch.stack((input.real, input.imag), dim=-1)


def _sort_signature(input, reference):
    out = {k: input[k] for k in reference if k in input}
    return list(out.values()), list(out.keys())


class BaseSimulator:
    """
    Base class for Bloch simulators.

    This class manages the setup for tissue and sequence parameters as well as simulation configuration
    like the computational device and state size. Users can extend this class to define specific sequence
    behaviors while benefiting from automatic handling of tissue properties like T1, T2, etc.

    Parameters
    ----------
    use_sequence : bool, optional
        A toggle that chooses whether to use sequence or tissue parameters (default is True).
    nstates : int, optional
        The maximum number of EPG states to be retained during simulation (default is 10).
    max_chunk_size : int, optional
        The maximum number of atoms to simulate in parallel. Defaults to None.
    device : str, optional
        The computational device (e.g., "cpu", "cuda"). Defaults to "cpu".

    Attributes
    ----------
    use_sequence : bool
        Whether to use sequence parameters (True) or tissue parameters (False).
    nstates : int
        The maximum number of EPG states to be retained during simulation.
    max_chunk_size : int, optional
        The maximum number of atoms to simulate in parallel.
    device : str
        The computational device (e.g., "cpu", "cuda").
    T1 : Optional[Union[float, torch.Tensor]]
        The longitudinal relaxation time [ms].
    T2 : Optional[Union[float, torch.Tensor]]
        The transverse relaxation time [ms].
    diff : Optional[Union[str, Tuple[str]]]
        Parameters to compute derivatives with respect to. Defaults to None.
    B1 : Optional[Union[float, torch.Tensor]]
        The flip angle scaling factor (1.0 = nominal flip angle).
    B0 : Optional[Union[float, torch.Tensor]]
        The bulk off-resonance in Hz.
    B1Tx2 : Optional[Union[float, torch.Tensor]]
        Flip angle scaling factor for secondary RF mode.
    B1phase : Optional[Union[float, torch.Tensor]]
        B1 relative phase in degrees.

    Methods
    -------
    set_callable_functions() :
        Sets the functions that will be used for simulation (fun, jac, etc).

    """

    @staticmethod
    def sequence():  # noqa
        """Base method to be overridden to define new signal simulators."""
        ...

    def __init__(
        self,
        nlocs: t.Optional[int] = 1,
        nstates: t.Optional[int] = 10,
        max_chunk_size: t.Optional[int] = None,
        diff: t.Optional[str | tuple[str]] = None,
        device: t.Optional[str] = "cpu",
    ):
        # Only initialize performance-related parameters
        self.nlocs = nlocs
        self.nstates = nstates
        self.max_chunk_size = max_chunk_size
        self.diff = diff
        self.device = device

        # Initialize placeholders for the simulation parameters
        # Main properties
        self.T1 = None
        self.T2 = None
        self.B1 = None
        self.B0 = None

        # Other main pool properties
        self.T2star = None
        self.D = None
        self.v = None
        self.moving = False
        self.chemshift = None

        # Bloch-mcconnell parameters
        self.T1bm = None
        self.T2bm = None
        self.kbm = None
        self.weight_bm = None
        self.chemshift_bm = None

        # Bloch-mt parameters
        self.kmt = None
        self.weight_mt = None

        # Fields
        self.B1Tx2 = None
        self.B1phase = None
        self.model = None

        # Set default callable functions
        self.fun = None
        self.jac = None
        self._spin_params = None
        self._sequence_params = None

    def initialize_spin_params(
        self,
        T1: float | torch.FloatTensor | npt.NDArray[float],
        T2: float | torch.FloatTensor | npt.NDArray[float],
        B1: t.Optional[float | torch.FloatTensor | npt.NDArray[float]] = None,
        B0: t.Optional[float | torch.FloatTensor | npt.NDArray[float]] = None,
        T2star: t.Optional[float | torch.FloatTensor | npt.NDArray[float]] = None,
        D: t.Optional[float | torch.FloatTensor | npt.NDArray[float]] = None,
        v: t.Optional[float | torch.FloatTensor | npt.NDArray[float]] = None,
        moving: t.Optional[bool] = False,
        chemshift: t.Optional[float | torch.FloatTensor | npt.NDArray[float]] = None,
        T1bm: t.Optional[float | torch.FloatTensor | npt.NDArray[float]] = None,
        T2bm: t.Optional[float | torch.FloatTensor | npt.NDArray[float]] = None,
        kbm: t.Optional[float | torch.FloatTensor | npt.NDArray[float]] = None,
        weight_bm: t.Optional[float | torch.FloatTensor | npt.NDArray[float]] = None,
        chemshift_bm: t.Optional[float | torch.FloatTensor | npt.NDArray[float]] = None,
        kmt: t.Optional[float | torch.FloatTensor | npt.NDArray[float]] = None,
        weight_mt: t.Optional[float | torch.FloatTensor | npt.NDArray[float]] = None,
        B1Tx2: t.Optional[float | torch.FloatTensor | npt.NDArray[float]] = None,
        B1phase: t.Optional[float | torch.FloatTensor | npt.NDArray[float]] = None,
    ):
        """
        Initialize all the properties that define the simulation parameters.

        This is called separately after instantiating the simulator object.
        """
        self.T1 = T1
        self.T2 = T2
        self.B1 = B1
        self.B0 = B0
        self.T2star = T2star
        self.D = D
        self.moving = moving
        self.v = v
        self.chemshift = chemshift
        self.T1bm = T1bm
        self.T2bm = T2bm
        self.kbm = kbm
        self.weight_bm = weight_bm
        self.chemshift_bm = chemshift_bm
        self.kmt = kmt
        self.weight_mt = weight_mt
        self.B1Tx2 = B1Tx2
        self.B1phase = B1phase

        # Initialize properties and model
        self._cast(spin_properties)
        self._initialize_model()
        self._initialize_fieldmap()
        self._initialize_chemshift()
        self._initialize_differentiation()

        self._get_sim_inputs()

    def _cast(self, prop_dict):
        """Initialize properties and cast them to tensors."""
        props = {}
        for fname in dir(self):
            if fname in prop_dict:
                fvalue = getattr(self, fname)
                props[fname] = fvalue

        props = {
            k: torch.as_tensor(v, dtype=torch.float32, device=self.device)
            for k, v in props.items()
            if v is not None
        }

        # Expand and broadcast tensors
        props = {
            k: torch.atleast_1d(v.squeeze())[..., None] + eps for k, v in props.items()
        }
        bprops = torch.broadcast_tensors(*props.values())
        props = dict(zip(props.keys(), bprops, strict=False))

        # Replace properties with broadcasted values
        for fname in props.keys():
            setattr(self, fname, props[fname])

    def _initialize_model(self):
        """Initialize the model type based on available parameters."""
        if self.T1bm is not None:
            self.model = "bm"
            assert self.T2bm is not None, "T2 for secondary free pool not provided!"
            assert (
                self.kbm is not None
            ), "Exchange rate for secondary free pool not provided!"
            assert (
                self.weight_bm is not None
            ), "Weight for secondary free pool not provided!"
            self.T1 = torch.cat((self.T1, self.T1bm), axis=-1) + eps
        if self.T2bm is not None:
            assert self.T1bm is not None, "T1 for secondary free pool not provided!"
            self.T2 = torch.cat((self.T2, self.T2bm), axis=-1) + eps

        if self.kmt is not None:
            if self.model is not None:
                self.model = "bm-mt"
            else:
                self.model = "mt"
            assert self.weight_mt is not None, "Weight for bound pool not provided!"
        if self.weight_mt is not None:
            assert self.kmt is not None, "Exchange rate for bound pool not provided!"

    def _initialize_chemshift(self):
        """Initialize the chemical shift for the model."""
        if self.model and "bm" in self.model:
            if self.chemshift_bm is None and self.chemshift is not None:
                self.chemshift_bm = self.chemshift
            elif self.chemshift_bm is not None and self.chemshift is None:
                self.chemshift = self.chemshift_bm

    def _initialize_fieldmap(self):
        """Initialize the field maps."""
        if self.B0 is None:
            self.B0 = torch.zeros(
                self.T1.shape, dtype=torch.float32, device=self.device
            )

        # total (complex) field variation
        if self.T2star is not None and self.model is not None:
            R2prime = 1 / self.T2star - 1 / self.T2[..., -1]
            T2prime = 1 / R2prime
            T2prime = torch.nan_to_num(T2prime, posinf=0.0, neginf=0.0) + eps
            self.df = R2prime + 1j * 2 * torch.pi * self.B0
        elif self.T2star is None and self.model is None:
            self.df = 1j * 2 * torch.pi * (self.B0 + self.chemshift)
        elif self.T2star is not None and self.model is None:
            R2star = 1 / self.T2star
            R2star = torch.nan_to_num(R2star, posinf=0.0, neginf=0.0) + eps
            self.df = R2star + 1j * 2 * torch.pi * (self.B0 + self.chemshift)
        else:
            self.df = 1j * 2 * torch.pi * self.B0
        self.df = torch.stack((self.df.real, self.df.imag), axis=-1)

        # B1
        if self.B1Tx2 is not None:
            assert self.B1 is not None, "B1 not provided!"
            if self.B1phase is None:
                self.B1phase = 0.0 * self.B1
            else:
                self.B1phase = torch.deg2rad(self.B1phase)
            self.B1 = torch.cat(
                (self.B1, self.B1Tx2 * torch.exp(1j * self.B1phase)), axis=-1
            )

    def _initialize_differentiation(self):
        """Initialize the field maps."""
        if self.diff is None:
            self.diff = []
        for fname in dir(self):
            if fname in self.diff:
                fvalue = getattr(self, fname)  # get current value
                if fvalue.requires_grad is False:
                    fvalue.requires_grad = True
                    setattr(self, fname, fvalue)

    def _initialize_buffer(self):  # noqa
        """Initialize EPG matrix buffer."""
        # get sizes
        self.batch_size, self.npools = self.T1.shape

        return ops.EPGstates(
            self.device,
            self.batch_size,
            self.nstates,
            self.nlocs,
            self.seqlength,
            self.npools,
            self.weight,
            self.model,
            self.moving,
        )

    def _get_sim_inputs(self):  # noqa
        """
        Prepare inputs for the simulation based on the sequence or tissue parameters.
        """
        # inspect signature
        modelparams = inspect_signature(self.sequence)
        assert (
            "signal" in modelparams
        ), "Error! Please design the model to accept a 'signal' argument."
        assert (
            "states" in modelparams
        ), "Error! Please design the model to accept a 'states' argument."

        # get sequence and tissue parameters
        output = {
            "T1": self.T1,
            "T2": self.T2,
            "k": self.k,
            "weight": self.weight,
            "chemshift": self.chemshift,
            "df": self.df,
            "D": self.D,
            "v": self.v,
            "B1": self.B1,
        }

        # clean up
        self._spin_params = {k: v for k, v in output.items() if k in modelparams}

    def _reformat(self, input):  # noqa
        # handle tuples
        if isinstance(input, (list, tuple)):
            output = [item[..., 0, :] + 1j * item[..., -1, :] for item in input]
            # output = [torch.diagonal(item, dim1=-2, dim2=-1) if len(item.shape) == 4 else item for item in output]
            output = [
                item.reshape(*item.shape[:2], -1) if len(item.shape) == 4 else item
                for item in output
            ]

            # stack
            if len(output) == 1:
                output = output[0]
            else:
                output = torch.concatenate(output, dim=-1)
                output = output.permute(2, 0, 1)
        else:
            output = input[..., 0] + 1j * input[..., -1]

        return output

    def _set_callable_functions(self):
        """
        Assign the `fun` and `jac` functions based on the toggle `use_sequence`.
        """
        if self.use_sequence:
            self.fun = self._fun_sequence
            self.jac = self._jac_sequence
        else:
            self.fun = self._fun_tissue
            self.jac = self._jac_tissue

    def _fun_sequence(self, seqparams=None, *tissue_params):
        """
        Sequence computation function. Uses sequence parameters to perform the computation.
        """
        return None

    def _fun_tissue(self, tissueparams=None, *seq_params):
        """
        Tissue computation function. Uses tissue parameters for mapping.
        """
        return None

    def _jac_sequence(self, seqparams=None, *tissue_params):
        """
        Jacobian computation using sequence parameters.
        """
        return None

    def _jac_tissue(self, tissueparams=None, *seq_params):
        """
        Jacobian computation using tissue parameters.
        """
        return None

    @property
    def xdata(self):
        """
        Return post-initialized properties for sequence or tissue parameters as a tensor.
        """
        if self._spin_params is not None:
            return torch.cat(
                [
                    self._spin_params
                    for k in self._spin_params.keys()
                    if self._spin_params[k] is not None
                ],
                dim=-1,
            )
        return None
