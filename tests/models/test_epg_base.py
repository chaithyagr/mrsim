
"""Test base EPG sequence"""

import torch

from mrsim.epg.model import BaseSimulator

# Test Case 1: Test Class Initialization with Default Parameters
def test_initialization_default():
    simulator = BaseSimulator()
    
    # Assert default values
    assert simulator.nstates == 10
    assert simulator.max_chunk_size is None
    assert simulator.device == "cpu"
    assert simulator.T1 is None
    assert simulator.T2 is None
    assert simulator.B1 is None
    assert simulator.B0 is None
    assert simulator.B1Tx2 is None
    assert simulator.B1phase is None


# Test Case 2: Test Class Initialization with Custom Parameters
def test_initialization_custom():    
    simulator = BaseSimulator(nstates=20, max_chunk_size=500, device="cuda")
    
    # Assert custom values
    assert simulator.nstates == 20
    assert simulator.max_chunk_size == 500
    assert simulator.device == "cuda"


# Test Case 3: Test callable functions initialization
def test_callable_functions():
    simulator = BaseSimulator()
    
    # Test the callable functions are set to None initially
    assert simulator.fun is None
    assert simulator.jac is None
    assert simulator.xdata is None


# Test Case 5: Test T1 and T2 Assignment
def test_t1_t2_assignment():
    T1_value = torch.tensor([500.0, 600.0])
    T2_value = torch.tensor([50.0, 60.0])

    simulator = BaseSimulator()
    simulator.T1 = T1_value
    simulator.T2 = T2_value

    # Assert that T1 and T2 are set correctly
    assert torch.equal(simulator.T1, T1_value)
    assert torch.equal(simulator.T2, T2_value)


# Test Case 7: Test max_chunk_size
def test_max_chunk_size():
    simulator = BaseSimulator(max_chunk_size=1000)
    assert simulator.max_chunk_size == 1000

    simulator = BaseSimulator(max_chunk_size=None)
    assert simulator.max_chunk_size is None


# Test Case 8: Test device assignment
def test_device_assignment():
    simulator = BaseSimulator(device="cuda")
    assert simulator.device == "cuda"
    
    simulator = BaseSimulator(device="cpu")
    assert simulator.device == "cpu"