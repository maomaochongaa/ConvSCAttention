<div align="center">
<h1 align="center">ConvSCAttention</h1>

<h3>CSCA-MobileNet: A Cashmere and Wool Fiber Recognition Network Based on Convolutional Spatial-Channel Attention Mechanism</h3>
    
[**Overview**](#overview) | [**Get Started**](#%EF%B8%8Flets-get-started)

</div>

## 🛎️Updates
* **` February 15th, 2025`**: The code for ConvSCAttention has been organized and uploaded. You are welcome to use them!!

## 🔭Overview
This repository provides the official implementation of ConvSCAttention.

## 🗝️Let's Get Started!
### Requirements
• PyTorch version >= 1.10.0

• Python version >= 3.9

### Usage Code
```python
    import torch
    from ConvSCAtt import ConvSCAtt

    # Create a test input tensor (batch_size, channels, height, width)
    x = torch.randn(1, 32, 256, 256).to(device)
    # Initialize the ConvSCAtt module
    csca = ConvSCAtt(dim=32, kernel_size=3)
    print(csca)

    output = csca(x)

    # Print the shapes of input and output tensors
    print("Input tensor shape:", x.shape)
    print("Output tensor shape:", output.shape)
