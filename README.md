# DummyBird

Some dummy code for three attention patterns in BigBird model.

## Motivation
In order to use pre-tuned operators, network models often include a large number of data layouts and format conversions, which will generate large memory access costs, and some of them may be redundant. For example, in the BigBird model, converting irregular sparse attention computations into matrix multiplications suitable for running on the GPU requires a lot of Reshape, Concatenates, and Gathers operations.

We want to know whether a unified semantic model can be used to represent various data access operations in deep learning applications. We can simplify and optimize data access at the semantic model level, and offload it to the underlying primitives for different target platforms.

In this repo, we will not use the primitives provided by the deep learning framework, but use simple for loops and ordinary data access statements to implement the attention patterns in BigBird. We hope to find some evidence to support our hypothesis and to inspire ourselves.