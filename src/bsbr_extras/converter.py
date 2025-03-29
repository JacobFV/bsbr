import torch
import torch.nn as nn
import math
from typing import Optional, Dict, Any, Union

from transformers import PreTrainedModel, AutoModel, GPT2Config
from transformers.models.gpt2.modeling_gpt2 import GPT2Model, GPT2Attention

from bsbr.bsbr import BSBRAttention, BSBRLayer, BSBRModel, PositionalEncoding


class TransformerToBSBRConverter:
    """
    Utility to convert vanilla transformer models (especially GPT-style models)
    to Block Sparse Attention with Block Retrieval (BSBR) transformers.
    
    This converter allows reusing pre-trained weights while benefiting from the
    efficiency of BSBR for processing longer sequences.
    """
    
    def __init__(self, chunk_size: int = 128, compression_factor: Optional[int] = None):
        """
        Initialize the converter with BSBR-specific parameters.
        
        Args:
            chunk_size: Size of chunks for BSBR processing
            compression_factor: Optional factor to compress state vectors
        """
        self.chunk_size = chunk_size
        self.compression_factor = compression_factor
    
    def convert_gpt2_attention_to_bsbr(self, 
                                      gpt_attention: GPT2Attention, 
                                      hidden_dim: int,
                                      num_heads: int) -> BSBRAttention:
        """
        Convert a GPT2 attention layer to a BSBR attention layer.
        
        Args:
            gpt_attention: The GPT2 attention layer to convert
            hidden_dim: Hidden dimension size
            num_heads: Number of attention heads
            
        Returns:
            A BSBR attention layer with weights initialized from the GPT2 layer
        """
        bsbr_attention = BSBRAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            chunk_size=self.chunk_size,
            dropout=gpt_attention.attn_dropout.p,
            compression_factor=self.compression_factor
        )
        
        # Handle GPT2's Conv1D vs Linear difference
        # In GPT2, Conv1D has shape (out_features, in_features) and is transposed compared to nn.Linear
        # c_attn is a single matrix for q, k, v concatenated, shape is (hidden_dim, 3*hidden_dim)
        
        # For q, k, v projections, we need to handle the transposed nature of Conv1D vs nn.Linear
        # The Conv1D weight is already transposed compared to nn.Linear, so we need to transpose it back
        if hasattr(gpt_attention, 'c_attn'):
            # Split the weights for Q, K, V
            qkv_weight = gpt_attention.c_attn.weight
            qkv_bias = gpt_attention.c_attn.bias
            
            # Check if qkv_weight has the expected shape (hidden_dim, 3*hidden_dim) or (3*hidden_dim, hidden_dim)
            if qkv_weight.shape[0] == hidden_dim:
                # Weights have shape (hidden_dim, 3*hidden_dim)
                # Need to transpose for nn.Linear which expects (out_features, in_features)
                q_weight = qkv_weight[:, :hidden_dim].t()
                k_weight = qkv_weight[:, hidden_dim:2*hidden_dim].t()
                v_weight = qkv_weight[:, 2*hidden_dim:3*hidden_dim].t()
                
                # Biases don't need transposition
                q_bias = qkv_bias[:hidden_dim] if qkv_bias is not None else None
                k_bias = qkv_bias[hidden_dim:2*hidden_dim] if qkv_bias is not None else None
                v_bias = qkv_bias[2*hidden_dim:3*hidden_dim] if qkv_bias is not None else None
            else:
                # Weights have shape (3*hidden_dim, hidden_dim)
                q_weight = qkv_weight[:hidden_dim, :]
                k_weight = qkv_weight[hidden_dim:2*hidden_dim, :]
                v_weight = qkv_weight[2*hidden_dim:3*hidden_dim, :]
                
                # Biases
                q_bias = qkv_bias[:hidden_dim] if qkv_bias is not None else None
                k_bias = qkv_bias[hidden_dim:2*hidden_dim] if qkv_bias is not None else None
                v_bias = qkv_bias[2*hidden_dim:3*hidden_dim] if qkv_bias is not None else None
        else:
            # If c_attn doesn't exist, try to find separate q, k, v projections
            q_weight = gpt_attention.q_proj.weight if hasattr(gpt_attention, 'q_proj') else None
            k_weight = gpt_attention.k_proj.weight if hasattr(gpt_attention, 'k_proj') else None
            v_weight = gpt_attention.v_proj.weight if hasattr(gpt_attention, 'v_proj') else None
            
            q_bias = gpt_attention.q_proj.bias if hasattr(gpt_attention, 'q_proj') else None
            k_bias = gpt_attention.k_proj.bias if hasattr(gpt_attention, 'k_proj') else None
            v_bias = gpt_attention.v_proj.bias if hasattr(gpt_attention, 'v_proj') else None
        
        # Copy weights to BSBR attention
        if q_weight is not None:
            bsbr_attention.q_proj.weight.data = q_weight
        if k_weight is not None:
            bsbr_attention.k_proj.weight.data = k_weight
        if v_weight is not None:
            bsbr_attention.v_proj.weight.data = v_weight
        
        # Copy biases
        if q_bias is not None:
            bsbr_attention.q_proj.bias.data = q_bias
        if k_bias is not None:
            bsbr_attention.k_proj.bias.data = k_bias
        if v_bias is not None:
            bsbr_attention.v_proj.bias.data = v_bias
        
        # Handle output projection
        if hasattr(gpt_attention, 'c_proj'):
            if gpt_attention.c_proj.weight.shape[0] == hidden_dim:
                # Weight is transposed compared to nn.Linear
                bsbr_attention.out_proj.weight.data = gpt_attention.c_proj.weight.t()
            else:
                bsbr_attention.out_proj.weight.data = gpt_attention.c_proj.weight
                
            if hasattr(gpt_attention.c_proj, 'bias') and gpt_attention.c_proj.bias is not None:
                bsbr_attention.out_proj.bias.data = gpt_attention.c_proj.bias
        
        # Initialize meta projections as combinations of existing projections
        # This is a heuristic approach to provide a reasonable starting point
        bsbr_attention.meta_r_proj.weight.data = (bsbr_attention.q_proj.weight.data + 
                                                bsbr_attention.k_proj.weight.data) / 2
        bsbr_attention.meta_h_proj.weight.data = (bsbr_attention.k_proj.weight.data + 
                                                bsbr_attention.v_proj.weight.data) / 2
        
        if hasattr(bsbr_attention.q_proj, 'bias') and bsbr_attention.q_proj.bias is not None:
            bsbr_attention.meta_r_proj.bias.data = (bsbr_attention.q_proj.bias.data + 
                                                  bsbr_attention.k_proj.bias.data) / 2
            bsbr_attention.meta_h_proj.bias.data = (bsbr_attention.k_proj.bias.data + 
                                                  bsbr_attention.v_proj.bias.data) / 2
        
        return bsbr_attention
    
    def convert_gpt2_layer_to_bsbr(self, 
                                  gpt_layer, 
                                  hidden_dim: int,
                                  num_heads: int,
                                  ff_dim: int) -> BSBRLayer:
        """
        Convert a GPT2 layer to a BSBR layer.
        
        Args:
            gpt_layer: The GPT2 layer to convert
            hidden_dim: Hidden dimension size
            num_heads: Number of attention heads
            ff_dim: Feed-forward intermediate dimension
            
        Returns:
            A BSBR layer with weights initialized from the GPT2 layer
        """
        # Ensure ff_dim is not None or zero
        if ff_dim is None or ff_dim == 0:
            ff_dim = 4 * hidden_dim  # Default to 4x hidden_dim as in standard transformers
            
        bsbr_layer = BSBRLayer(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            chunk_size=self.chunk_size,
            ff_dim=ff_dim,
            dropout=gpt_layer.attn.attn_dropout.p,
            compression_factor=self.compression_factor
        )
        
        # Convert attention layer
        bsbr_layer.attention = self.convert_gpt2_attention_to_bsbr(
            gpt_layer.attn, hidden_dim, num_heads
        )
        
        # Copy layer norms
        bsbr_layer.layer_norm1.weight.data = gpt_layer.ln_1.weight.data
        bsbr_layer.layer_norm1.bias.data = gpt_layer.ln_1.bias.data
        bsbr_layer.layer_norm2.weight.data = gpt_layer.ln_2.weight.data
        bsbr_layer.layer_norm2.bias.data = gpt_layer.ln_2.bias.data
        
        # Copy feed-forward network
        # GPT-2 MLP has: c_fc -> gelu -> c_proj
        # BSBR FF has: Linear -> GELU -> Dropout -> Linear -> Dropout
        
        # First linear layer (hidden_dim -> ff_dim)
        if hasattr(gpt_layer.mlp, 'c_fc'):
            c_fc_weight = gpt_layer.mlp.c_fc.weight
            c_fc_bias = gpt_layer.mlp.c_fc.bias if hasattr(gpt_layer.mlp.c_fc, 'bias') else None
            
            # Check c_fc weight shape and transpose if needed
            # nn.Linear expects weight of shape (out_features, in_features)
            # For first layer: out_features = ff_dim, in_features = hidden_dim
            if c_fc_weight.shape[0] == hidden_dim and c_fc_weight.shape[1] == ff_dim:
                # Weight is transposed compared to nn.Linear expectation, need to transpose
                c_fc_weight_fixed = c_fc_weight.t()
            elif c_fc_weight.shape[0] == ff_dim and c_fc_weight.shape[1] == hidden_dim:
                # Weight already has the correct shape for nn.Linear
                c_fc_weight_fixed = c_fc_weight
            else:
                # Try to reshape or give a reasonable error
                # This is a heuristic approach
                if c_fc_weight.numel() == ff_dim * hidden_dim:
                    c_fc_weight_fixed = c_fc_weight.reshape(ff_dim, hidden_dim)
                else:
                    # Initialize with a random weight if shapes don't match
                    c_fc_weight_fixed = torch.randn(ff_dim, hidden_dim) * 0.02
                    print(f"Warning: Had to initialize feedforward weights randomly due to shape mismatch")
            
            bsbr_layer.ff[0].weight.data = c_fc_weight_fixed
            if c_fc_bias is not None:
                if c_fc_bias.shape[0] == ff_dim:
                    bsbr_layer.ff[0].bias.data = c_fc_bias
                else:
                    # Initialize with zeros if shape doesn't match
                    bsbr_layer.ff[0].bias.data = torch.zeros(ff_dim)
        
        # Second linear layer (ff_dim -> hidden_dim)
        if hasattr(gpt_layer.mlp, 'c_proj'):
            c_proj_weight = gpt_layer.mlp.c_proj.weight
            c_proj_bias = gpt_layer.mlp.c_proj.bias if hasattr(gpt_layer.mlp.c_proj, 'bias') else None
            
            # Check c_proj weight shape and transpose if needed
            # nn.Linear expects weight of shape (out_features, in_features)
            # For second layer: out_features = hidden_dim, in_features = ff_dim
            if c_proj_weight.shape[0] == ff_dim and c_proj_weight.shape[1] == hidden_dim:
                # Weight is transposed compared to nn.Linear expectation, need to transpose
                c_proj_weight_fixed = c_proj_weight.t()
            elif c_proj_weight.shape[0] == hidden_dim and c_proj_weight.shape[1] == ff_dim:
                # Weight already has the correct shape for nn.Linear
                c_proj_weight_fixed = c_proj_weight
            else:
                # Try to reshape or give a reasonable error
                # This is a heuristic approach
                if c_proj_weight.numel() == hidden_dim * ff_dim:
                    c_proj_weight_fixed = c_proj_weight.reshape(hidden_dim, ff_dim)
                else:
                    # Initialize with a random weight if shapes don't match
                    c_proj_weight_fixed = torch.randn(hidden_dim, ff_dim) * 0.02
                    print(f"Warning: Had to initialize feedforward weights randomly due to shape mismatch")
            
            bsbr_layer.ff[3].weight.data = c_proj_weight_fixed
            if c_proj_bias is not None:
                if c_proj_bias.shape[0] == hidden_dim:
                    bsbr_layer.ff[3].bias.data = c_proj_bias
                else:
                    # Initialize with zeros if shape doesn't match
                    bsbr_layer.ff[3].bias.data = torch.zeros(hidden_dim)
        
        return bsbr_layer
    
    def convert_gpt2_model_to_bsbr(self, gpt_model: GPT2Model) -> BSBRModel:
        """
        Convert a complete GPT2 model to a BSBR model.
        
        Args:
            gpt_model: The GPT2 model to convert
            
        Returns:
            A BSBR model with weights initialized from the GPT2 model
        """
        config = gpt_model.config
        
        # Extract configuration with fallbacks
        hidden_size = getattr(config, 'hidden_size', getattr(config, 'n_embd', None))
        if hidden_size is None:
            raise ValueError("Could not determine hidden_size from model config")
            
        num_hidden_layers = getattr(config, 'num_hidden_layers', getattr(config, 'n_layer', None))
        if num_hidden_layers is None:
            raise ValueError("Could not determine num_hidden_layers from model config")
            
        num_attention_heads = getattr(config, 'num_attention_heads', getattr(config, 'n_head', None))
        if num_attention_heads is None:
            raise ValueError("Could not determine num_attention_heads from model config")
            
        vocab_size = getattr(config, 'vocab_size', None)
        if vocab_size is None:
            raise ValueError("Could not determine vocab_size from model config")
            
        # Check for n_inner or infer from hidden_size
        ff_dim = getattr(config, 'n_inner', None)
        if ff_dim is None:
            ff_dim = getattr(config, 'intermediate_size', 4 * hidden_size)
            
        dropout = getattr(config, 'resid_pdrop', getattr(config, 'hidden_dropout_prob', 0.1))
        
        # Create a new BSBR model
        bsbr_model = BSBRModel(
            vocab_size=vocab_size,
            hidden_dim=hidden_size,
            num_layers=num_hidden_layers,
            num_heads=num_attention_heads,
            chunk_size=self.chunk_size,
            ff_dim=ff_dim,
            dropout=dropout,
            compression_factor=self.compression_factor
        )
        
        # Copy embedding weights
        bsbr_model.embedding.weight.data = gpt_model.wte.weight.data
        
        # Copy positional encoding
        # Note: GPT uses learned positional embeddings while BSBR uses sinusoidal
        # We'll initialize with sinusoidal but could adapt to use learned if needed
        
        # Copy transformer layers
        for i, gpt_layer in enumerate(gpt_model.h):
            bsbr_model.layers[i] = self.convert_gpt2_layer_to_bsbr(
                gpt_layer, 
                hidden_size, 
                num_attention_heads,
                ff_dim
            )
        
        # Copy final layer norm
        bsbr_model.layer_norm.weight.data = gpt_model.ln_f.weight.data
        bsbr_model.layer_norm.bias.data = gpt_model.ln_f.bias.data
        
        return bsbr_model
    
    def convert_pretrained_model(self, model_name_or_path: str) -> BSBRModel:
        """
        Convert a pre-trained HuggingFace model to a BSBR model.
        
        Args:
            model_name_or_path: Name or path of the pre-trained model
            
        Returns:
            A BSBR model with weights initialized from the pre-trained model
        """
        # Load the pre-trained model
        original_model = AutoModel.from_pretrained(model_name_or_path)
        
        # Check if it's a GPT2 model
        if isinstance(original_model, GPT2Model):
            return self.convert_gpt2_model_to_bsbr(original_model)
        else:
            raise ValueError(f"Model type {type(original_model)} is not supported yet. Only GPT2 models are currently supported.")


def convert_to_bsbr(
    model_name_or_path: str,
    chunk_size: int = 128,
    compression_factor: Optional[int] = None
) -> BSBRModel:
    """
    Convenience function to convert a pre-trained transformer model to BSBR.
    
    Args:
        model_name_or_path: Name or path of the pre-trained model
        chunk_size: Size of chunks for BSBR processing
        compression_factor: Optional factor to compress state vectors
        
    Returns:
        A BSBR model with weights initialized from the pre-trained model
    """
    converter = TransformerToBSBRConverter(
        chunk_size=chunk_size,
        compression_factor=compression_factor
    )
    return converter.convert_pretrained_model(model_name_or_path) 