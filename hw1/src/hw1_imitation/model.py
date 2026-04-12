"""Model definitions for Push-T imitation policies."""

from __future__ import annotations

import abc
from typing import Literal, TypeAlias
from collections import OrderedDict

import torch
from torch import nn


class BasePolicy(nn.Module, metaclass=abc.ABCMeta):
    """Base class for action chunking policies."""

    def __init__(self, state_dim: int, action_dim: int, chunk_size: int) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size

    @abc.abstractmethod
    def compute_loss(
        self, state: torch.Tensor, action_chunk: torch.Tensor
    ) -> torch.Tensor:
        """Compute training loss for a batch."""

    @abc.abstractmethod
    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,  # only applicable for flow policy
    ) -> torch.Tensor:
        """Generate a chunk of actions with shape (batch, chunk_size, action_dim)."""


class MSEPolicy(BasePolicy):
    """Predicts action chunks with an MSE loss."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
        use_bias: bool = False,
    ) -> None:

        """
        Shape of the MLP: 

        state_dim -> hidden -> act -> hidden -> act -> out
        """
        super().__init__(state_dim, action_dim, chunk_size)
        modules = []

        hidden_dims = list(hidden_dims)
        in_dims = [state_dim] + hidden_dims
        out_dims = hidden_dims + [action_dim * chunk_size]
        n_layers = len(out_dims)

        for i, (d_in, d_out) in enumerate(zip(in_dims, out_dims)):
            modules.append((f"linear-{i}",  nn.Linear(d_in, d_out, bias=use_bias)))
            if i < n_layers - 1:
                modules.append((f"relu-{i}", nn.ReLU()))

        self.model = nn.Sequential(OrderedDict(modules))

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        B, _ = state.shape
        pred_action_chunk = self.model(state).reshape(B, self.chunk_size, self.action_dim)
        return pred_action_chunk

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        """
        state: [B, |S|]
        action_chunk: [B, chunk_size, |A|]
        """
        pred_action_chunk = self.sample_actions(state)
        loss = torch.mean(torch.norm(action_chunk - pred_action_chunk, p=2)**2)
        return pred_action_chunk, loss

    def forward(
        self, 
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ):
        """
        state: [B, |S|]
        """
        return self.compute_loss(state, action_chunk)



class FlowMatchingPolicy(BasePolicy):
    """Predicts action chunks with a flow matching loss."""

    ### TODO: IMPLEMENT FlowMatchingPolicy HERE ###
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        raise NotImplementedError


PolicyType: TypeAlias = Literal["mse", "flow"]


def build_policy(
    policy_type: PolicyType,
    *,
    state_dim: int,
    action_dim: int,
    chunk_size: int,
    use_bias: bool,
    hidden_dims: tuple[int, ...] = (128, 128),
) -> BasePolicy:
    if policy_type == "mse":
        return MSEPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
            use_bias=use_bias,
        )
    if policy_type == "flow":
        return FlowMatchingPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
            use_bias=use_bias,
        )
    raise ValueError(f"Unknown policy type: {policy_type}")
