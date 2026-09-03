NumPy-Native Recurrent Neural Network

A recurrent neural network built entirely from scratch in NumPy — no Keras, no PyTorch, no autograd. The forward pass, backpropagation through time (BPTT), and gradient updates are all implemented manually.

Why This Project

Most RNN implementations call a single library function and never expose what happens underneath. The goal here was the opposite: to implement every step by hand in order to understand recurrent architectures at the level of the math — how the hidden state carries information across timesteps, how gradients flow backward through an unrolled network, and why vanishing/exploding gradients arise.

What It Does

The model predicts tomorrow's maximum temperature from a sequence of prior daily values.

Implementation Details
Forward pass — hidden state computed at each timestep from the current input and the previous hidden state.
Backpropagation Through Time (BPTT) — the network is unrolled across the input sequence and gradients are accumulated across all timesteps for the shared weights.
Gradient updates — implemented manually without an optimizer library.
No deep learning framework — only NumPy for all matrix operations.
Tech Stack

Python, NumPy
