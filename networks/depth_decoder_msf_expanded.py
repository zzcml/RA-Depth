# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.


# depth_decoder_msf_expanded - Fully expanded version with all operators explicitly defined
from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict


class DepthDecoder_MSF_Expanded(nn.Module):
    """
    Fully expanded DepthDecoder_MSF with all submodules explicitly named.
    All F.interpolate operations are replaced with ConvTranspose2d (deconvolution).
    All operators are defined directly in __init__ without nested helper functions.
    """
    
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True):
        super(DepthDecoder_MSF_Expanded, self).__init__()

        self.num_output_channels = num_output_channels
        self.scales = scales
        self.num_ch_enc = num_ch_enc  # features in encoder, e.g., [64, 18, 36, 72, 144]
        self.use_skips = use_skips

        # ========== Stage 0: Parallel convolutions ==========
        # parallel_conv layer 0, branch 1
        self.parallel_conv_0_1_conv = nn.Conv2d(num_ch_enc[1], num_ch_enc[1], kernel_size=3, stride=1, padding=1)
        self.parallel_conv_0_1_elu = nn.ELU(inplace=True)
        
        # parallel_conv layer 0, branch 2
        self.parallel_conv_0_2_conv = nn.Conv2d(num_ch_enc[2], num_ch_enc[2], kernel_size=3, stride=1, padding=1)
        self.parallel_conv_0_2_elu = nn.ELU(inplace=True)
        
        # parallel_conv layer 0, branch 3
        self.parallel_conv_0_3_conv = nn.Conv2d(num_ch_enc[3], num_ch_enc[3], kernel_size=3, stride=1, padding=1)
        self.parallel_conv_0_3_elu = nn.ELU(inplace=True)
        
        # parallel_conv layer 0, branch 4
        self.parallel_conv_0_4_conv = nn.Conv2d(num_ch_enc[4], num_ch_enc[4], kernel_size=3, stride=1, padding=1)
        self.parallel_conv_0_4_elu = nn.ELU(inplace=True)

        # ========== Stage 0: 1x1 convolutions for feature alignment ==========
        # conv1x1: 0, 2->1 (scale factor 2)
        self.conv1x1_0_2_1_conv = nn.Conv2d(num_ch_enc[2], num_ch_enc[1], kernel_size=1, stride=1)
        self.conv1x1_0_2_1_elu = nn.ELU(inplace=True)
        self.upsample_0_2_1 = nn.ConvTranspose2d(num_ch_enc[2], num_ch_enc[2], kernel_size=4, stride=2, padding=1)
        
        # conv1x1: 0, 3->2 (scale factor 2)
        self.conv1x1_0_3_2_conv = nn.Conv2d(num_ch_enc[3], num_ch_enc[2], kernel_size=1, stride=1)
        self.conv1x1_0_3_2_elu = nn.ELU(inplace=True)
        self.upsample_0_3_2 = nn.ConvTranspose2d(num_ch_enc[3], num_ch_enc[3], kernel_size=4, stride=2, padding=1)
        
        # conv1x1: 0, 3->1 (scale factor 4)
        self.conv1x1_0_3_1_conv = nn.Conv2d(num_ch_enc[3], num_ch_enc[1], kernel_size=1, stride=1)
        self.conv1x1_0_3_1_elu = nn.ELU(inplace=True)
        self.upsample_0_3_1 = nn.ConvTranspose2d(num_ch_enc[3], num_ch_enc[3], kernel_size=8, stride=4, padding=2)
        
        # conv1x1: 0, 4->3 (scale factor 2)
        self.conv1x1_0_4_3_conv = nn.Conv2d(num_ch_enc[4], num_ch_enc[3], kernel_size=1, stride=1)
        self.conv1x1_0_4_3_elu = nn.ELU(inplace=True)
        self.upsample_0_4_3 = nn.ConvTranspose2d(num_ch_enc[4], num_ch_enc[4], kernel_size=4, stride=2, padding=1)
        
        # conv1x1: 0, 4->2 (scale factor 4)
        self.conv1x1_0_4_2_conv = nn.Conv2d(num_ch_enc[4], num_ch_enc[2], kernel_size=1, stride=1)
        self.conv1x1_0_4_2_elu = nn.ELU(inplace=True)
        self.upsample_0_4_2 = nn.ConvTranspose2d(num_ch_enc[4], num_ch_enc[4], kernel_size=8, stride=4, padding=2)
        
        # conv1x1: 0, 4->1 (scale factor 8)
        self.conv1x1_0_4_1_conv = nn.Conv2d(num_ch_enc[4], num_ch_enc[1], kernel_size=1, stride=1)
        self.conv1x1_0_4_1_elu = nn.ELU(inplace=True)
        self.upsample_0_4_1 = nn.ConvTranspose2d(num_ch_enc[4], num_ch_enc[4], kernel_size=16, stride=8, padding=4)

        # ========== Stage 1: Parallel convolutions ==========
        # parallel_conv layer 1, branch 1
        self.parallel_conv_1_1_conv = nn.Conv2d(num_ch_enc[1], num_ch_enc[1], kernel_size=3, stride=1, padding=1)
        self.parallel_conv_1_1_elu = nn.ELU(inplace=True)
        
        # parallel_conv layer 1, branch 2
        self.parallel_conv_1_2_conv = nn.Conv2d(num_ch_enc[2], num_ch_enc[2], kernel_size=3, stride=1, padding=1)
        self.parallel_conv_1_2_elu = nn.ELU(inplace=True)
        
        # parallel_conv layer 1, branch 3
        self.parallel_conv_1_3_conv = nn.Conv2d(num_ch_enc[3], num_ch_enc[3], kernel_size=3, stride=1, padding=1)
        self.parallel_conv_1_3_elu = nn.ELU(inplace=True)

        # ========== Stage 1: 1x1 convolutions for feature alignment ==========
        # conv1x1: 1, 2->1 (scale factor 2)
        self.conv1x1_1_2_1_conv = nn.Conv2d(num_ch_enc[2], num_ch_enc[1], kernel_size=1, stride=1)
        self.conv1x1_1_2_1_elu = nn.ELU(inplace=True)
        self.upsample_1_2_1 = nn.ConvTranspose2d(num_ch_enc[2], num_ch_enc[2], kernel_size=4, stride=2, padding=1)
        
        # conv1x1: 1, 3->2 (scale factor 2)
        self.conv1x1_1_3_2_conv = nn.Conv2d(num_ch_enc[3], num_ch_enc[2], kernel_size=1, stride=1)
        self.conv1x1_1_3_2_elu = nn.ELU(inplace=True)
        self.upsample_1_3_2 = nn.ConvTranspose2d(num_ch_enc[3], num_ch_enc[3], kernel_size=4, stride=2, padding=1)
        
        # conv1x1: 1, 3->1 (scale factor 4)
        self.conv1x1_1_3_1_conv = nn.Conv2d(num_ch_enc[3], num_ch_enc[1], kernel_size=1, stride=1)
        self.conv1x1_1_3_1_elu = nn.ELU(inplace=True)
        self.upsample_1_3_1 = nn.ConvTranspose2d(num_ch_enc[3], num_ch_enc[3], kernel_size=8, stride=4, padding=2)

        # ========== Stage 2: Parallel convolutions ==========
        # parallel_conv layer 2, branch 1
        self.parallel_conv_2_1_conv = nn.Conv2d(num_ch_enc[1], num_ch_enc[1], kernel_size=3, stride=1, padding=1)
        self.parallel_conv_2_1_elu = nn.ELU(inplace=True)
        
        # parallel_conv layer 2, branch 2
        self.parallel_conv_2_2_conv = nn.Conv2d(num_ch_enc[2], num_ch_enc[2], kernel_size=3, stride=1, padding=1)
        self.parallel_conv_2_2_elu = nn.ELU(inplace=True)

        # ========== Stage 2: 1x1 convolutions for feature alignment ==========
        # conv1x1: 2, 2->1 (scale factor 2)
        self.conv1x1_2_2_1_conv = nn.Conv2d(num_ch_enc[2], num_ch_enc[1], kernel_size=1, stride=1)
        self.conv1x1_2_2_1_elu = nn.ELU(inplace=True)
        self.upsample_2_2_1 = nn.ConvTranspose2d(num_ch_enc[2], num_ch_enc[2], kernel_size=4, stride=2, padding=1)

        # ========== Stage 3: Parallel convolutions ==========
        # parallel_conv layer 3, branch 0
        self.parallel_conv_3_0_conv = nn.Conv2d(num_ch_enc[0], num_ch_enc[0], kernel_size=3, stride=1, padding=1)
        self.parallel_conv_3_0_elu = nn.ELU(inplace=True)
        
        # parallel_conv layer 3, branch 1
        self.parallel_conv_3_1_conv = nn.Conv2d(num_ch_enc[1], num_ch_enc[1], kernel_size=3, stride=1, padding=1)
        self.parallel_conv_3_1_elu = nn.ELU(inplace=True)

        # ========== Stage 3: 1x1 convolutions for feature alignment ==========
        # conv1x1: 3, 1->0 (scale factor 2)
        self.conv1x1_3_1_0_conv = nn.Conv2d(num_ch_enc[1], num_ch_enc[0], kernel_size=1, stride=1)
        self.conv1x1_3_1_0_elu = nn.ELU(inplace=True)
        self.upsample_3_1_0 = nn.ConvTranspose2d(num_ch_enc[1], num_ch_enc[1], kernel_size=4, stride=2, padding=1)

        # ========== Stage 4: Final decoding layers ==========
        # parallel_conv layer 4, branch 0: num_ch_enc[0] -> 32
        self.parallel_conv_4_0_conv = nn.Conv2d(num_ch_enc[0], 32, kernel_size=3, stride=1, padding=1)
        self.parallel_conv_4_0_elu = nn.ELU(inplace=True)
        self.upsample_4_0 = nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1)
        
        # parallel_conv layer 5, branch 0: 32 -> 16
        self.parallel_conv_5_0_conv = nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1)
        self.parallel_conv_5_0_elu = nn.ELU(inplace=True)
        
        # dispconv: 16 -> num_output_channels
        self.dispconv_0_conv = nn.Conv2d(16, num_output_channels, kernel_size=3, stride=1, padding=1)
        
        # Sigmoid activation for disparity output
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features):
        """
        Forward pass with explicit operator calls.
        input_features: list of encoder features [e0, e1, e2, e3, e4]
        """
        outputs = {}

        # Extract encoder features
        e4 = input_features[4]
        e3 = input_features[3]
        e2 = input_features[2]
        e1 = input_features[1]
        e0 = input_features[0]

        # ========== Stage 0: Parallel convolutions ==========
        out = self.parallel_conv_0_1_conv(e1)
        d0_1 = self.parallel_conv_0_1_elu(out)
        
        out = self.parallel_conv_0_2_conv(e2)
        d0_2 = self.parallel_conv_0_2_elu(out)
        
        out = self.parallel_conv_0_3_conv(e3)
        d0_3 = self.parallel_conv_0_3_elu(out)
        
        out = self.parallel_conv_0_4_conv(e4)
        d0_4 = self.parallel_conv_0_4_elu(out)

        # ========== Stage 0: Upsampling via deconvolution and 1x1 convolutions ==========
        # Upsample d0_2 by scale factor 2
        out = self.upsample_0_2_1(d0_2)
        out = self.conv1x1_0_2_1_conv(out)
        d0_2_1 = self.conv1x1_0_2_1_elu(out)
        
        # Upsample d0_3 by scale factor 2
        out = self.upsample_0_3_2(d0_3)
        out = self.conv1x1_0_3_2_conv(out)
        d0_3_2 = self.conv1x1_0_3_2_elu(out)
        
        # Upsample d0_3 by scale factor 4
        out = self.upsample_0_3_1(d0_3)
        out = self.conv1x1_0_3_1_conv(out)
        d0_3_1 = self.conv1x1_0_3_1_elu(out)
        
        # Upsample d0_4 by scale factor 2
        out = self.upsample_0_4_3(d0_4)
        out = self.conv1x1_0_4_3_conv(out)
        d0_4_3 = self.conv1x1_0_4_3_elu(out)
        
        # Upsample d0_4 by scale factor 4
        out = self.upsample_0_4_2(d0_4)
        out = self.conv1x1_0_4_2_conv(out)
        d0_4_2 = self.conv1x1_0_4_2_elu(out)
        
        # Upsample d0_4 by scale factor 8
        out = self.upsample_0_4_1(d0_4)
        out = self.conv1x1_0_4_1_conv(out)
        d0_4_1 = self.conv1x1_0_4_1_elu(out)

        # ========== Stage 0: Multi-scale fusion ==========
        d0_1_msf = d0_1 + d0_2_1 + d0_3_1 + d0_4_1
        d0_2_msf = d0_2 + d0_3_2 + d0_4_2
        d0_3_msf = d0_3 + d0_4_3

        # ========== Stage 1: Parallel convolutions ==========
        out = self.parallel_conv_1_1_conv(d0_1_msf)
        d1_1 = self.parallel_conv_1_1_elu(out)
        
        out = self.parallel_conv_1_2_conv(d0_2_msf)
        d1_2 = self.parallel_conv_1_2_elu(out)
        
        out = self.parallel_conv_1_3_conv(d0_3_msf)
        d1_3 = self.parallel_conv_1_3_elu(out)

        # ========== Stage 1: Upsampling via deconvolution and 1x1 convolutions ==========
        # Upsample d1_2 by scale factor 2
        out = self.upsample_1_2_1(d1_2)
        out = self.conv1x1_1_2_1_conv(out)
        d1_2_1 = self.conv1x1_1_2_1_elu(out)
        
        # Upsample d1_3 by scale factor 2
        out = self.upsample_1_3_2(d1_3)
        out = self.conv1x1_1_3_2_conv(out)
        d1_3_2 = self.conv1x1_1_3_2_elu(out)
        
        # Upsample d1_3 by scale factor 4
        out = self.upsample_1_3_1(d1_3)
        out = self.conv1x1_1_3_1_conv(out)
        d1_3_1 = self.conv1x1_1_3_1_elu(out)

        # ========== Stage 1: Multi-scale fusion ==========
        d1_1_msf = d1_1 + d1_2_1 + d1_3_1
        d1_2_msf = d1_2 + d1_3_2

        # ========== Stage 2: Parallel convolutions ==========
        out = self.parallel_conv_2_1_conv(d1_1_msf)
        d2_1 = self.parallel_conv_2_1_elu(out)
        
        out = self.parallel_conv_2_2_conv(d1_2_msf)
        d2_2 = self.parallel_conv_2_2_elu(out)

        # ========== Stage 2: Upsampling via deconvolution and 1x1 convolution ==========
        # Upsample d2_2 by scale factor 2
        out = self.upsample_2_2_1(d2_2)
        out = self.conv1x1_2_2_1_conv(out)
        d2_2_1 = self.conv1x1_2_2_1_elu(out)

        # ========== Stage 2: Multi-scale fusion ==========
        d2_1_msf = d2_1 + d2_2_1

        # ========== Stage 3: Parallel convolutions ==========
        out = self.parallel_conv_3_0_conv(e0)
        d3_0 = self.parallel_conv_3_0_elu(out)
        
        out = self.parallel_conv_3_1_conv(d2_1_msf)
        d3_1 = self.parallel_conv_3_1_elu(out)

        # ========== Stage 3: Upsampling via deconvolution and 1x1 convolution ==========
        # Upsample d3_1 by scale factor 2
        out = self.upsample_3_1_0(d3_1)
        out = self.conv1x1_3_1_0_conv(out)
        d3_1_0 = self.conv1x1_3_1_0_elu(out)

        # ========== Stage 3: Multi-scale fusion ==========
        d3_0_msf = d3_0 + d3_1_0

        # ========== Stage 4: Final decoding ==========
        out = self.parallel_conv_4_0_conv(d3_0_msf)
        d4_0 = self.parallel_conv_4_0_elu(out)
        
        # Upsample d4_0 by scale factor 2
        d4_0 = self.upsample_4_0(d4_0)
        
        out = self.parallel_conv_5_0_conv(d4_0)
        d5 = self.parallel_conv_5_0_elu(out)
        
        out = self.dispconv_0_conv(d5)
        disp = self.sigmoid(out)
        
        outputs[("disp", 0)] = disp

        return outputs

    def export_to_onnx(self, onnx_file_path, input_shape=(1, 3, 192, 640), opset_version=11, simplify=True):
        """
        Export the model to ONNX format and optionally simplify it using onnxsim.
        
        Args:
            onnx_file_path: Path to save the ONNX model
            input_shape: Tuple (batch_size, channels, height, width) for dummy input
            opset_version: ONNX opset version (default: 11)
            simplify: Whether to simplify the model using onnxsim (default: True)
        """
        import os
        import torch.onnx
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(onnx_file_path) if os.path.dirname(onnx_file_path) else '.', exist_ok=True)
        
        # Create dummy input
        dummy_input = [torch.randn(*input_shape) for _ in range(5)]
        
        # Export to ONNX
        self.eval()
        torch.onnx.export(
            self,
            dummy_input,
            onnx_file_path,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['e0', 'e1', 'e2', 'e3', 'e4'],
            output_names=['disp_0'],
            dynamic_axes={
                'e0': {0: 'batch', 2: 'height', 3: 'width'},
                'e1': {0: 'batch', 2: 'height', 3: 'width'},
                'e2': {0: 'batch', 2: 'height', 3: 'width'},
                'e3': {0: 'batch', 2: 'height', 3: 'width'},
                'e4': {0: 'batch', 2: 'height', 3: 'width'},
                'disp_0': {0: 'batch', 2: 'height', 3: 'width'}
            }
        )
        
        print(f"Model exported to {onnx_file_path}")
        
        # Get file size before simplification
        size_before = os.path.getsize(onnx_file_path)
        print(f"ONNX file size before simplification: {size_before / 1024:.2f} KB")
        
        # Simplify the model if requested
        if simplify:
            try:
                import onnxsim
                import onnx
                
                # Load the ONNX model
                onnx_model = onnx.load(onnx_file_path)
                
                # Simplify the model
                model_simp, check = onnxsim.simplify(onnx_model)
                
                # Verify the simplified model
                assert check, "Simplified model check failed!"
                
                # Save the simplified model
                onnx.save(model_simp, onnx_file_path)
                
                # Get file size after simplification
                size_after = os.path.getsize(onnx_file_path)
                print(f"ONNX file size after simplification: {size_after / 1024:.2f} KB")
                print(f"Size reduction: {(size_before - size_after) / 1024:.2f} KB ({100 * (size_before - size_after) / size_before:.2f}%)")
                print("Model simplified successfully using onnxsim!")
                
            except ImportError:
                print("Warning: onnxsim not installed. Install it with: pip install onnxsim")
                print("Skipping model simplification.")
            except Exception as e:
                print(f"Warning: Failed to simplify model: {e}")
        
        # Example usage for ONNX Runtime verification (commented out)
        """
        # To verify with ONNX Runtime:
        import onnxruntime as ort
        sess = ort.InferenceSession(onnx_file_path)
        inputs = {
            'e0': dummy_input[0].numpy(),
            'e1': dummy_input[1].numpy(),
            'e2': dummy_input[2].numpy(),
            'e3': dummy_input[3].numpy(),
            'e4': dummy_input[4].numpy()
        }
        outputs = sess.run(None, inputs)
        print(f"ONNX Runtime output shape: {outputs[0].shape}")
        """


def create_depth_decoder_msf_expanded(num_ch_enc=[64, 18, 36, 72, 144], num_output_channels=1):
    """
    Factory function to create a DepthDecoder_MSF_Expanded instance.
    
    Args:
        num_ch_enc: List of encoder channel dimensions [ch0, ch1, ch2, ch3, ch4]
                   Default: [64, 18, 36, 72, 144] for HRNet-18
        num_output_channels: Number of output channels (default: 1 for disparity)
    
    Returns:
        DepthDecoder_MSF_Expanded model instance
    """
    return DepthDecoder_MSF_Expanded(
        num_ch_enc=num_ch_enc,
        scales=range(4),
        num_output_channels=num_output_channels,
        use_skips=True
    )


if __name__ == "__main__":
    # Test the expanded model
    print("Testing DepthDecoder_MSF_Expanded...")
    
    # Example encoder channels for HRNet-18
    num_ch_enc = [64, 18, 36, 72, 144]
    
    # Create model
    model = create_depth_decoder_msf_expanded(num_ch_enc=num_ch_enc)
    model.eval()
    
    print(f"Model created with encoder channels: {num_ch_enc}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create dummy input features (simulating encoder output)
    batch_size = 1
    base_height = 192
    base_width = 640
    
    # Encoder features at different scales
    # e0: 1/1 scale, e1: 1/2, e2: 1/4, e3: 1/8, e4: 1/16
    input_features = [
        torch.randn(batch_size, num_ch_enc[0], base_height, base_width),       # e0
        torch.randn(batch_size, num_ch_enc[1], base_height // 2, base_width // 2),  # e1
        torch.randn(batch_size, num_ch_enc[2], base_height // 4, base_width // 4),  # e2
        torch.randn(batch_size, num_ch_enc[3], base_height // 8, base_width // 8),  # e3
        torch.randn(batch_size, num_ch_enc[4], base_height // 16, base_width // 16) # e4
    ]
    
    # Forward pass
    with torch.no_grad():
        outputs = model(input_features)
    
    disp_output = outputs[("disp", 0)]
    print(f"Input features shapes: {[f.shape for f in input_features]}")
    print(f"Disparity output shape: {disp_output.shape}")
    print(f"Disparity output range: [{disp_output.min():.4f}, {disp_output.max():.4f}]")
    print("\nTest passed successfully!")
    
    # Example: Export to ONNX
    print("\n--- ONNX Export Example ---")
    print("To export the model to ONNX, use:")
    print("  model.export_to_onnx('depth_decoder_msf_expanded.onnx', input_shape=(1, 3, 192, 640))")
    print("\nOr run this script with:")
    print("  python depth_decoder_msf_expanded.py")
