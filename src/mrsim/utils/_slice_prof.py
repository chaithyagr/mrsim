"""Slice Profile estimation routines."""

__all__ = ["slice_prof"]


import numpy.typing as npt
import numpy as np

import scipy


def slice_prof(
    rf_envelope: npt.ArrayLike,
    nlocations: int = 10,
    osf: int = 100,
) -> npt.ArrayLike:
    """
    Calculate slice profile using Fourier approximation.

    Parameters
    ----------
    rf_envelope : npt.ArrayLike
        Temporal envelope of RF waveform.
    nlocations : int, optional
        Number of spatial locations. The default is 10.
    osf : int, optional
        Oversampling factor for FFT interpolation. The default is 100.

    Returns
    -------
    profile : npt.ArrayLike
        B1 profile along slice.
        This is normalized so that center of the slice has nominal
        flip angle.

    """
    npts = len(rf_envelope)

    # oversampled fft
    freq = np.fft.fft(rf_envelope, n=int(osf * npts))
    freq = abs(freq)
    freq = freq / freq[0]  # normalize to DC

    # get first point lower than 0.1 maximum size
    idx = np.where(freq <= 0.1)[0][0]

    # cut frequency profile
    profile = freq[:idx]
    profile = _spline(
        np.linspace(0.0, 1.0, len(profile)), profile, np.linspace(0.0, 1.0, nlocations)
    )

    return profile


# %%
def _spline(x, y, xq):
    """
    Same as MATLAB cubic spline interpolation.
    """
    # interpolate
    cs = scipy.interpolate.InterpolatedUnivariateSpline(x, y)
    return cs(xq)
