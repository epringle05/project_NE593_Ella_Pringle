# project_NE593_Ella_Pringle_and_Patrick_Normantas

Compares a standard ReLU network against a heterogeneous network that mixes
four biologically-inspired neuron types: standard ReLU, polynomial/quadratic,
dendritic (multi-branch), and spiking (leaky integrate-and-fire).

Both models are trained on the two-moons binary classification benchmark and
evaluated on training loss, test accuracy, and decision boundary geometry.

Reference:
    Fan et al., "Expressivity of Quadratic Neurons" (polynomial neuron formulation)
    Chrysos et al., "Polynomial Networks" (deep polynomial architectures)
    Fan, FL., Li, Y., Zeng, T. et al. Towards NeuroAI: introducing neuronal diversity into       artificial neural networks. Med-X 3, 2 (2025). https://doi.org/10.1007/s44258-024-00042-2
Dependencies:
    torch, numpy, matplotlib, scikit-learn
