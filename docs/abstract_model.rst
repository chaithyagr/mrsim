AbstractModel: Explanation
==========================

Context and Purpose
-------------------

The `AbstractModel` class is designed to serve as a foundation for building MRI simulation models, especially those that require computational efficiency and the ability to compute derivatives such as gradients and Jacobians. It is meant to help streamline and automate common tasks in MRI simulations, such as managing simulation parameters, handling differentiation, and running computations efficiently on different devices.

MRI simulations often involve working with multiple parameters that need to be adjusted during computation. These parameters can affect the simulation's output and the computational cost, especially when computations need to be performed on large datasets or over multiple iterations. The `AbstractModel` class helps manage these complexities by defining a clear structure for handling parameters, computations, and gradients in a scalable way.

Why Use This Class?
-------------------

The need for such an abstract class arises because MRI simulation models tend to be highly parameterized, with many different settings that can impact both the simulation’s accuracy and computational performance. Traditional model-building approaches may involve manually managing these parameters, setting them in each simulation, and calculating the Jacobians (or gradients) for optimization purposes. This can become cumbersome and error-prone as models grow in complexity.

`AbstractModel` automates these tasks in several key ways:

1. **Efficient Parameter Management**: It automatically segregates parameters into broadcastable and non-broadcastable groups, ensuring that computations can be efficiently parallelized and distributed across devices (e.g., CPU or GPU). This reduces the burden on the user to manually manage parameters.

2. **Automatic Differentiation**: The class facilitates the automatic calculation of derivatives, such as the Jacobian, which is essential in gradient-based optimization algorithms. By integrating this functionality directly into the base class, the user is spared from needing to explicitly implement gradient computation methods.

3. **Parallel and Vectorized Computation**: The class integrates with PyTorch's `vmap`, which enables automatic vectorization of computations. This allows for batch processing and parallel computation of multiple samples, making the simulation more efficient, especially when working with large datasets or complex models.

How Does It Work?
-----------------

### 1. **Initialization and Parameter Handling**

The constructor (`__init__`) of the `AbstractModel` class initializes the model with the necessary parameters, including the computational device (CPU, CUDA-enabled GPU, etc.), the chunk size for batching computations, and the set of parameters to differentiate with respect to (for Jacobian computation). 

- **Device**: The `device` argument determines where the computations will occur (e.g., `'cpu'`, `'cuda'` for GPU). This enables the model to run on different hardware environments with minimal changes to the code.
- **Chunk Size**: The `chunk_size` controls how many samples will be processed simultaneously during computations, making the model adaptable for both small and large datasets.


### 2. **Forward Computation and Jacobian Calculation**

The `AbstractModel` is designed to handle both **forward computations** and **Jacobian computations**:

- **Forward Computation (`forward`)**: The `forward` method provides a way to compute the model's output based on the input parameters. This method is designed to be vectorized using `torch.vmap`, which allows it to efficiently handle multiple inputs at once. The user can specify the inputs, and the method will automatically apply the model's logic across multiple input samples in parallel, improving computation efficiency.

- **Jacobian Computation (`jacobian`)**: The Jacobian is a matrix of all first-order partial derivatives of the model's outputs with respect to its inputs. This is crucial in optimization algorithms like gradient descent, where gradients are needed to update parameters. The `jacobian` method handles the automatic computation of this matrix using PyTorch's `jacfwd` function (a forward-mode automatic differentiation tool). The class integrates this capability into the model’s interface, allowing users to easily compute gradients without needing to implement the differentiation logic themselves.

### 3. ** Engine Creation**

The `_engine` method represents the actual simulation step. The user should focus on the implementation of the simulator in this section, with minimal boilerplate to achieve
parallelization and enable automatic differentiation. 

This flexibility allows the class to support a wide range of use cases, from simple parameter setups to more complex models with specific computational needs.

Why is `set_properties` and `set_sequence` abstract?
---------------------------------------------------

The methods `set_properties` and `set_sequence` are marked as abstract because they define the spin/environment parameters and sequence parameters that are specific to the MRI simulation model being created. These methods must be implemented by subclasses of `AbstractModel`, which gives the user flexibility to define model-specific properties. 

This ensures that the base class provides a structure and automatic handling for common operations, while still allowing subclasses to define the behavior and properties unique to their particular simulation model.

Conclusion
----------

The `AbstractModel` class is an essential tool for creating flexible and efficient MRI simulation models. By automating the handling of parameters, computations, and gradients, it allows the user to focus on the unique aspects of their simulation. With built-in support for parallel computations, automatic differentiation, and mixed-precision optimizations, it provides a powerful framework for handling complex MRI simulation tasks.

Ultimately, the purpose of `AbstractModel` is to provide a high-level structure that simplifies common MRI simulation tasks while giving users the flexibility to define and optimize their own models within that structure. By abstracting away complex tasks such as parameter broadcasting and gradient computation, the class makes it easier to develop high-performance simulations without dealing with low-level details.

