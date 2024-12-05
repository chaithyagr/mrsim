"""Test abstract model."""

import torch

from torchsim.base import AbstractModel  # Update with actual path


# Mock subclass of AbstractModel for testing purposes
class MyModel(AbstractModel):
    def set_properties(self, param):
        self.properties.param = param

    def set_sequence(self):
        pass

    @staticmethod
    def _engine(param):
        return torch.tensor([1.0, 2.0, 3.0])


# Test initialization and attribute assignment
def test_initialization():
    model = MyModel(chunk_size=10, device="cuda", diff="param")

    assert model.chunk_size == 10
    assert model.device == "cuda"
    assert model.diff == "param"
    assert model.broadcastable_params == ["param"]


# Test forward function
def test_forward():
    model = MyModel(chunk_size=10, diff="param")

    forward_fn = model.forward()

    # Test the returned forward function
    output = forward_fn(torch.tensor([2.0]))  # Passing tensor as input
    assert isinstance(output, torch.Tensor)  # Check that output is a tensor


# Test jacobian function
def test_jacobian():
    model = MyModel(chunk_size=10, diff="param")

    jacobian_fn = model.jacobian()

    # Test the returned jacobian function
    jacobian_output = jacobian_fn(torch.tensor([2.0]))  # Passing tensor as input
    assert isinstance(jacobian_output, torch.Tensor)  # Check that output is a tensor


# Test __call__ method
def test_call():
    model = MyModel(chunk_size=10, diff="param")
    model.set_properties(torch.tensor([1.0]))

    output, jacobian_output = model()

    # Test that the __call__ method returns both outputs
    assert isinstance(output, torch.Tensor)
    assert isinstance(jacobian_output, torch.Tensor)
