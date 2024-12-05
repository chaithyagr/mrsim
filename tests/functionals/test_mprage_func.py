"""MPRAGE tests."""

from torchsim import mprage_sim


def test_scalar_forward():
    sig = mprage_sim(
        nshots=100,
        TI=200.0,
        flip=5.0,
        TRspgr=10.0,
        T1=1000.0,
    )
    assert sig.shape == ()


def test_multiple_forward():
    sig = mprage_sim(
        nshots=100,
        TI=200.0,
        flip=5.0,
        TRspgr=10.0,
        T1=(200, 500, 1000.0),
    )
    assert sig.shape == (3,)
