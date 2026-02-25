# Understanding BDH: The Baby Dragon Hatchling Architecture

[![YouTube Playlist](https://img.shields.io/badge/YouTube-Watch_the_Series-FF0000?style=for-the-badge&logo=youtube)](#)
[![Website](https://img.shields.io/badge/Website-Interactive_Tutorials-blue?style=for-the-badge)](#)
[![Pathway](https://img.shields.io/badge/Original_Code-Pathway-green?style=for-the-badge)](#)

Welcome to **Understanding BDH**, a comprehensive video tutorial series and resource hub dedicated to demystifying the **Baby Dragon Hatchling (BDH)** language model architecture. 

If you are looking to understand how we can move beyond the quadratic memory constraints of standard Transformers by leveraging biologically inspired, scale-free graph dynamics, you are in the right place.

## What is BDH?
BDH (Baby Dragon Hatchling) is a novel architecture that bridges the gap between the distributed, sparse physics of biological brains and the dense, tensor-based efficiency of modern GPUs. By replacing the traditional $O(N^2)$ global matrix attention and infinitely growing KV-caches with a fixed-size state matrix, high-dimensional sparsity, and local graph dynamics, BDH achieves Transformer-level performance with linear $O(T)$ scaling.

This repository serves as the companion guide to our 6-part YouTube masterclass on the BDH paper.

---

## Video Series Index

### [🎥 #1 From Transformers to BDH | What is Attention?](https://www.youtube.com/watch?v=aFU5szokm5k&list=PLS6rVTBBmT8sIcPJ8fPWlKRXvMj_fHq6G&index=1)
**Setting the Stage.** We break down the fundamentals of the standard Transformer architecture, explain the mechanics (and the bottlenecks) of global softmax Attention, and introduce the core motivations behind the Baby Dragon Hatchling (BDH) architecture. 

### [🎥 #2 BDH-Graph | Beyond the Matrix: Distributed Graph Intelligence](https://www.youtube.com/watch?v=9IPLT4XzDs0&list=PLS6rVTBBmT8sIcPJ8fPWlKRXvMj_fHq6G&index=2)
**Section 2 of the Paper.** We dive into the theoretical math of BDH. Learn how to replace monolithic matrix operations with local graph dynamics. We cover the 4-phase reasoning cycle (Read, Write, Filter, Send), Hebbian learning, and Replicator Dynamics.

### [🎥 #3 BDH-GPU | A Tensor-Friendly Version of the BDH Architecture](https://www.youtube.com/watch?v=2JSm78o65R8&list=PLS6rVTBBmT8sIcPJ8fPWlKRXvMj_fHq6G&index=3)
**Section 3 of the Paper.** How do we run organic graph physics on a GPU? We explore the low-rank factorizations that translate $O(N^2)$ graph connections into efficient tensor operations, proving that continual, synaptic reasoning is compatible with modern deep learning stacks.

### [🎥 #4 Bridging Biological Graph Dynamics with Tensor Efficiency](https://www.youtube.com/watch?v=2YKTmu9MQYw&list=PLS6rVTBBmT8sIcPJ8fPWlKRXvMj_fHq6G&index=4)
**Section 4 of the Paper.** The engineering reality. We break down the implementation details: how BDH uses just three shared matrices ($E, D_x, D_y$), how it maintains a constant memory footprint, and how the 5% activation sparsity drives computational efficiency and $O(T)$ linear scaling laws.

### [🎥 #5 Analysis: Emergence of Modularity and Scale-Free Structure](https://www.youtube.com/watch?v=Y3X_UiCOkMo&list=PLS6rVTBBmT8sIcPJ8fPWlKRXvMj_fHq6G&index=5)
**Section 5 of the Paper.** Why does this architecture matter? We explore "Axiomatic AI" and how imposing structural constraints (bottlenecks, sparsity, thresholds) forces the network to spontaneously evolve biological traits—like core-periphery hubs, monosemantic synapses, and composability.

### [🎥 #6 Official Pathway `bdh.py` Code Explained!](https://www.youtube.com/watch?v=lnivP1jU2SM&list=PLS6rVTBBmT8sIcPJ8fPWlKRXvMj_fHq6G&index=6)
**Code Walkthrough.** We pop the hood and analyze the official open-source PyTorch implementation from Pathway. We walk through the exact mechanics of the massive sparse latent expansion, the positive orthant gating, and Rotary Positional Embeddings (RoPE) integration.

---

## Resources

* **[Original BDH Paper](https://arxiv.org/abs/2509.26507):** Read the foundational research.
* **[Pathway GitHub Repo](https://github.com/pathwaycom/bdh):** The official source code for BDH.


## 📬 Connect with Us
* Subscribe on [YouTube](#)


---

