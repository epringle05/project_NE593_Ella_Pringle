# NE593 Final Project
Ella Pringle and Patrick Normantas

Compares a standard ReLU network against a heterogeneous network that mixes
four biologically-inspired neuron types: standard ReLU, polynomial/quadratic,
dendritic (multi-branch), and spiking (leaky integrate-and-fire).

1. Standard Neuron - A conventional neuron: linear projection followed by a ReLU activation.
 
    Computes:
        output = ReLU(x @ W^T + b)
     It is included here as the baseline neuron type inside the heterogeneous layer.
    Args:
        in_features  (int): Dimensionality of the input vector.
        out_features (int): Number of neurons (output dimensionality).

2. Polynomial Neuron - A quadratic (polynomial) neuron based on the formulation of Fan et al.
 
    Instead of a simple weighted sum, this neuron computes pairwise feature
    interactions, allowing it to model relationships like x_i * x_j directly.
    This pushes nonlinearity inside the neuron rather than applying it only
    at the activation stage.
 
    Computes:
        output = (x @ W_r^T) * (x @ W_g^T) + (x^2) @ W_b^T
 
    Where:
        W_r, W_g  capture multiplicative feature interactions (the quadratic term)
        W_b       captures element-wise squared input contributions
 
    Parameter cost is O(3n) (three times a standard neuron) but representational
    capacity is strictly higher. By the fundamental theorem of algebra, quadratic
    neurons are sufficient building blocks for polynomial function approximation.
 
    Args:
        in_features  (int): Dimensionality of the input vector.
        out_features (int): Number of neurons (output dimensionality).
3. Dendritic Neuron - A multi-branch dendritic neuron inspired by the two-stage computation     of biological neurons.
 
    Real dendrites are not passive wires, local patches of the dendritic tree
    compute nonlinear functions of nearby synapses before the results reach the
    cell body (soma). This module models that by running several independent
    linear branches, applying a local tanh nonlinearity to each, and summing
    the results (soma integration).
 
    A single dendritic neuron can specialize different branches for different
    input patterns, supporting richer per-neuron computation than a point neuron.
 
    Computes (for B branches):
        output = sum_b( tanh(x @ W_b^T + bias_b) )  for b in 1..B
 
    Args:
        in_features  (int): Dimensionality of the input vector.
        out_features (int): Number of output units.
        branches     (int): Number of dendritic branches. Default: 3.
4. Spiking Neuron - A differentiable leaky integrate-and-fire (LIF) spiking neuron.
 
    Biological neurons communicate through discrete voltage spikes rather than
    continuous values. The LIF model captures this with three dynamics:
        1. Integration  — input current charges a membrane potential
        2. Leakage      — the potential decays over time toward zero
        3. Firing       — when the potential exceeds a threshold, a spike is emitted
                          and the potential partially resets
 
    Because the true spike (a hard threshold step) has zero gradient almost
    everywhere, training via backpropagation is impossible directly. We use a
    *surrogate gradient*: during the forward pass the sigmoid approximates the
    spike; during the backward pass its smooth gradient flows through normally.
 
    The neuron is simulated for `time_steps` steps; the output is the average
    spike rate, converting the temporal spike train into a single continuous value
    compatible with standard layers downstream.
 
    Computes over T time steps:
        I         = x @ W^T + b          (input current, fixed across steps)
        mem_t     = decay * mem_{t-1} + I (leaky integration)
        spike_t   = sigmoid(beta * (mem_t - threshold))  (surrogate spike)
        mem_t     = mem_t * (1 - spike_t)                (soft reset)
        output    = mean(spike_t)  over t = 1..T
 
    Args:
        in_features  (int):   Dimensionality of the input vector.
        out_features (int):   Number of output units.
        threshold    (float): Membrane potential required to trigger a spike. Default: 1.0.
        decay        (float): Fraction of membrane potential retained each step (leak). Default: 0.9.
        time_steps   (int):   Number of simulation steps per forward pass. Default: 10.
        beta         (float): Sharpness of the surrogate sigmoid; higher = closer to a true spike. Default: 10.0.

Both models are trained on the two-moons binary classification benchmark and
evaluated on training loss, test accuracy, and decision boundary geometry.

References:
    Fan et al., "Expressivity of Quadratic Neurons" (polynomial neuron formulation)
    Chrysos et al., "Polynomial Networks" (deep polynomial architectures)
    Fan, FL., Li, Y., Zeng, T. et al. Towards NeuroAI: introducing neuronal diversity into artificial neural networks. Med-X 3, 2 (2025). https://doi.org/10.1007/s44258-024-00042-2
    LLM Use: Claude and ChatGPT - Used for debugging and code comments/docstrings

Dependencies:
    torch, numpy, matplotlib, scikit-learn
