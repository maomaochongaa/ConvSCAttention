# ConvSCAttention
This repository provides the official implementation of ConvSCAttention.

# Requirements
• PyTorch version >= 1.10.0

• Python version >= 3.9

# Usage Code
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
