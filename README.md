# Gated-XNOR
Gated XNOR Networks: Deep Neural Networks with Ternary Weights and Activations under a Unified Discretization Framework
(https://arxiv.org/abs/1705.09283)

By Lei Deng, Peng Jiao, Jing Pei, Zhenzhi Wu, Guoqi Li

Introduction

There is a pressing need to build an architecture that could subsume these networks undera unified framework that achieves both higher performance and less overhead. To this end, two fundamental issues are yet to be addressed. The first one is how to implement the back propagation when neuronal activations are discrete. The second one is how to remove the full-precision hidden weights in the training phase to break the bottlenecks of memory/computation consumption. To address the first issue, we present a multistep neuronal activation discretization method and a derivative approximation technique that enable the implementing the back propagation algorithm on discrete DNNs. While for the second issue, we propose a discrete state transition (DST) methodology to constrain the weights in a discrete space without saving the hidden weights. In this way, we build a unified framework that subsumes the binary or ternary networks as its special cases.More particularly, we find that when both the weights and activations become ternary values, the DNNs can be reduced to gated XNOR networks (or sparse binary networks) since only the event of non-zero weight and non-zero activation enables the control gate to start the XNOR logic operations in the original binary networks. This promises the event-driven hardware design for efficient mobile intelligence. We achieve advanced performance compared with state-of-the-art algorithms. Furthermore,the computational sparsity and the number of states in the discrete space can be flexibly modified to make it suitable for various hardware platforms.
