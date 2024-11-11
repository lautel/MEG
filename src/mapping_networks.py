import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, List, Callable
from enum import Enum
from argparser import ModelArguments
from pytorch_metric_learning import losses, distances
from losses import NTBXentLoss


class MappingType(Enum):
    MLP="mlp"
    Transformer="transformer"
    Linear="linear"
    def __str__(self):
        return self.value


class LinearMappingNetwork(nn.Module):
    def __init__(self, 
                 input_dim,
                 output_dim,
                 dtype: torch.dtype=torch.float16,
                 bias: bool=True, 
                 activation: Callable[[torch.Tensor], torch.Tensor] = F.tanh,
    ):
        super(LinearMappingNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, output_dim, bias=True, dtype=dtype)

    def forward(self, x: torch.Tensor, lm_embeddings=None, identity_indexes=None) -> torch.Tensor:
        x = self.fc1(x)
        loss = None
        return x, loss


class MultiHeadMLPMappingNetwork(nn.Module):
    def __init__(self, output_dim: int, hidden_dim: int, t: int, model_args: ModelArguments,
        dtype: torch.dtype=torch.float16):
        """
        Initialize a Multi-Head Multi-Layer Perceptron (MLP) with N layers and H heads.
        
        Args:
            input_dim (int): Dimension of the input features.
            hidden_dim (int): Dimension of the hidden layer.
            output_dim (int): Dimension of the output layer.
            num_layers (int): Number of layers in each MLP (N layers).
            num_heads (int): Number of parallel heads (H heads).
        """
        super(MultiHeadMLPMappingNetwork, self).__init__()
        self.dtype = dtype
        
        input_dim = model_args.prefix_dim
        num_layers = model_args.num_layers_map_network
        num_heads = model_args.num_heads_map_network

        # Define Losses
        self.xent_loss_fn = losses.SelfSupervisedLoss(
            losses.NTXentLoss(temperature=t),
            symmetric=False)  # temperature: The MoCo paper uses 0.07, while SimCLR uses 0.5.
        self.bxent_loss_fn = NTBXentLoss(temperature=t)
        self.cont_loss_fn = losses.SelfSupervisedLoss(
            losses.ContrastiveLoss(pos_margin=0.9,neg_margin=0.1,distance=distances.CosineSimilarity()),
            symmetric=False)
        self.back_trans_loss_fn = nn.CosineEmbeddingLoss()
        self.alpha = nn.Parameter(torch.tensor([0.5], dtype=dtype), requires_grad=True)

        # Define mapping G: from graph space to LM space
        self.G = nn.ModuleList([
            self._make_mlp(input_dim, hidden_dim, output_dim, num_layers, dtype) for _ in range(num_heads)
        ])

        # Define mapping F: from LM space to graph space. 
        # Used as regularizer during traning. Not used at inference time.
        self.F = nn.ModuleList([
            self._make_mlp(output_dim, hidden_dim, input_dim, num_layers, dtype) for _ in range(num_heads)
        ])
        
    def _make_mlp(self, input_dim, hidden_dim, output_dim, num_layers, dtype):
        """
        Helper function to create a single MLP with specified number of layers.
        """
        layers = []
        # First layer takes input_dim
        layers.append(nn.Linear(input_dim, hidden_dim, bias=True, dtype=dtype))
        layers.append(nn.ReLU())
        for _ in range(1, num_layers):
            # Hidden layers take hidden_dim as input and output
            layers.append(nn.Linear(hidden_dim, hidden_dim, bias=True, dtype=dtype))
            layers.append(nn.ReLU())
        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim, bias=True, dtype=dtype))
        
        return nn.Sequential(*layers)
    
    def g_func(self, x: torch.Tensor):
        # Pass input x through each head and store the output
        head_outputs = [head(x) for head in self.G]
        # Concatenate outputs from each head along the last dimension
        outputs = torch.cat(head_outputs, dim=-1)
        return outputs
    
    def f_func(self, x: torch.Tensor):
        # Pass input x through each head and store the output
        head_outputs = [head(x) for head in self.F]
        # Concatenate outputs from each head along the last dimension
        outputs = torch.cat(head_outputs, dim=-1)
        return outputs

    def forward(self, input_embeds, target_embeds: torch.Tensor=None, **kwargs):
        """
        Forward pass through all num_heads heads in parallel.
        """
        if input_embeds.dtype != self.dtype:
            input_embeds = input_embeds.to(dtype=self.dtype)

        g_x = self.g_func(input_embeds)

        loss = None
        if target_embeds is not None:
            if len(g_x.shape) == 3:
                g_x=g_x.squeeze(1)  # needed for xent_loss_fn
            if target_embeds.dtype != self.dtype:
                target_embeds = target_embeds.to(dtype=self.dtype)
            if len(target_embeds.shape) == 3:
                target_embeds=target_embeds.squeeze(1)  # needed for xent_loss_fn
            
            xent_loss = self.xent_loss_fn(g_x, target_embeds)
            if len(input_embeds.shape) == 3:
                g_x=g_x.unsqueeze(1)  # back to original shape
            
            f_g_x = self.f_func(g_x)
            if len(f_g_x.shape) == 3:
                f_g_x=f_g_x.squeeze(1)
            if len(input_embeds.shape) == 3:
                input_embeds=input_embeds.squeeze(1)
            y = torch.ones(len(input_embeds), device=input_embeds.device)  # maximize similarity
            back_trans_loss = self.back_trans_loss_fn(f_g_x, input_embeds, y)
            loss = self.alpha * xent_loss + (1 - self.alpha) * back_trans_loss
            loss = loss.squeeze(0)
            return dict(loss=loss, xent_loss=xent_loss, back_trans_loss=back_trans_loss, logits=g_x)
        return dict(logits=g_x, loss=None)



class TransformerMappingNetwork(nn.Module):
    def __init__(
        self,
        output_dim,
        hidden_dim,
        t,
        model_args: ModelArguments,
        dtype: torch.dtype=torch.float16,
        activation: Callable[[torch.Tensor], torch.Tensor] = F.relu,
    ):
        super(TransformerMappingNetwork, self).__init__()
        self.dtype = dtype
        
        # Define Losses
        self.xent_loss_fn = losses.SelfSupervisedLoss(
            losses.NTXentLoss(temperature=t),
            symmetric=False)  # temperature: The MoCo paper uses 0.07, while SimCLR uses 0.5.
        self.bxent_loss_fn = NTBXentLoss(temperature=t)
        self.cont_loss_fn = losses.SelfSupervisedLoss(
            losses.ContrastiveLoss(pos_margin=0.9,neg_margin=0.1,distance=distances.CosineSimilarity()),
            symmetric=False)
        self.back_trans_loss_fn = nn.CosineEmbeddingLoss()
        self.alpha = nn.Parameter(torch.tensor([0.5], dtype=dtype), requires_grad=True)

        # Define mapping G: from graph space to LM space
        down = nn.Linear(model_args.prefix_dim, hidden_dim, bias=True, dtype=dtype)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=model_args.num_heads_map_network,
            dim_feedforward=2 * hidden_dim,
            dropout=0.,
            activation=activation,
            batch_first=True,
            norm_first=True,  # activations are normalized before the self-attention block
            dtype=dtype
        )
        transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=model_args.num_layers_map_network) #, enable_nested_tensor=True)
        # prefix_const = nn.Parameter(
        #     torch.randn(model_args.output_prefix_length, hidden_dim, dtype=dtype))
        up = nn.Linear(hidden_dim, output_dim, bias=True, dtype=dtype)
        self.G = nn.Sequential(
            down, 
            transformer,
            up
        )

        # Define mapping F: from LM space to graph space. 
        # Used as regularizer during traning. Not used at inference time.
        fdown = nn.Linear(output_dim, hidden_dim, bias=True, dtype=dtype)
        fencoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=model_args.num_heads_map_network,
            dim_feedforward=2 * hidden_dim,
            dropout=0.,
            activation=activation,
            batch_first=True,
            norm_first=True,  # activations are normalized before the self-attention block
            dtype=dtype
        )
        ftransformer = nn.TransformerEncoder(
            fencoder_layer, num_layers=model_args.num_layers_map_network) #, enable_nested_tensor=True)
        fup = nn.Linear(hidden_dim, model_args.prefix_dim, bias=True, dtype=dtype)
        self.F = nn.Sequential(
            fdown, 
            ftransformer,
            fup
        )

    def g_func(self, x: torch.Tensor):
        x = self.G(x)
        return x
    
    def f_func(self, x: torch.Tensor):
        x = self.F(x)
        return x
    
    def forward(self, input_embeds: torch.Tensor, target_embeds: torch.Tensor=None, **kwargs):
        if input_embeds.dtype != self.dtype:
            input_embeds = input_embeds.to(dtype=self.dtype)

        g_x = self.g_func(input_embeds)

        loss = None
        if target_embeds is not None:
            if len(g_x.shape) == 3:
                g_x=g_x.squeeze(1)  # needed for xent_loss_fn
            if target_embeds.dtype != self.dtype:
                target_embeds = target_embeds.to(dtype=self.dtype)
            if len(target_embeds.shape) == 3:
                target_embeds=target_embeds.squeeze(1)  # needed for xent_loss_fn
            
            xent_loss = self.xent_loss_fn(g_x, target_embeds)
            if len(input_embeds.shape) == 3:
                g_x=g_x.unsqueeze(1)  # back to original shape
            
            f_g_x = self.f_func(g_x)
            if len(f_g_x.shape) == 3:
                f_g_x=f_g_x.squeeze(1)
            if len(input_embeds.shape) == 3:
                input_embeds=input_embeds.squeeze(1)
            y = torch.ones(len(input_embeds), device=input_embeds.device)  # maximize similarity
            back_trans_loss = self.back_trans_loss_fn(f_g_x, input_embeds, y)
            loss = self.alpha * xent_loss + (1 - self.alpha) * back_trans_loss
            loss = loss.squeeze(0)
            return dict(loss=loss, xent_loss=xent_loss, back_trans_loss=back_trans_loss, logits=g_x)
        return dict(logits=g_x, loss=None)
    
    def forward_mapping_only(self, input_embeds: torch.Tensor, target_embeds: torch.Tensor=None, codes: torch.Tensor=None, **kwargs):
        # Codes are for each entity, check for same codes within the batch
        if input_embeds.dtype != self.dtype:
            input_embeds = input_embeds.to(dtype=self.dtype)
        g_x = self.g_func(input_embeds)
        if len(g_x.shape) == 3:
            g_x=g_x.squeeze(1)

        loss = None
        if target_embeds is not None:
            assert codes is not None 
            if target_embeds.dtype != self.dtype:
                target_embeds = target_embeds.to(dtype=self.dtype)
            if len(target_embeds.shape) == 3:
                target_embeds=target_embeds.squeeze(1)
            
            xent_loss = self.bxent_loss_fn(g_x, codes)
            # print(xent_loss)
            
            f_g_x = self.f_func(g_x)
            if len(f_g_x.shape) == 3:
                f_g_x=f_g_x.squeeze(1)
            if len(input_embeds.shape) == 3:
                input_embeds=input_embeds.squeeze(1)
            y = torch.ones(len(input_embeds), device=input_embeds.device)  # maximize similarity
            back_trans_loss = self.back_trans_loss_fn(f_g_x, input_embeds, y)
            loss = self.alpha * xent_loss + (1 - self.alpha) * back_trans_loss
            loss = loss.squeeze(0)
            return dict(loss=loss, xent_loss=xent_loss, back_trans_loss=back_trans_loss, logits=g_x)
        return dict(logits=g_x)

    def from_pretrained(self, checkpoint_path):
        print("Loading weights of the mapping function (transformer)")
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        if "module" in state_dict:
            model_state_dict = state_dict['module']
            self.load_state_dict(model_state_dict)
        else:
            self.load_state_dict(state_dict)


if __name__ == "__main__":
    from collections import namedtuple
    from utils import summary_parameters

    hidden_dim=256
    output_dim=4096
    t=1.0
    ModelInfo = namedtuple("model_args", ["prefix_dim",  "num_layers_map_network", "num_heads_map_network"])
    model_args = ModelInfo(256, 2, 1)

    mlp = MultiHeadMLPMappingNetwork(
        output_dim,
        hidden_dim,
        t,
        model_args
    )

    summary_parameters(mlp)
