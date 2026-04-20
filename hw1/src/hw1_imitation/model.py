"""Model definitions for Push-T imitation policies."""

from __future__ import annotations

import abc
from typing import Literal, TypeAlias
from collections import OrderedDict

import torch
from torch import nn


class BasePolicy(nn.Module, metaclass=abc.ABCMeta):
    """Base class for action chunking policies."""

    def __init__(self, state_dim: int, action_dim: int, chunk_size: int, use_bias: bool) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.use_bias = use_bias

    def build_mlp(self, in_dims, out_dims):
        modules = []
        n_layers = len(out_dims)

        for i, (d_in, d_out) in enumerate(zip(in_dims, out_dims)):
            modules.append((f"linear-{i}",  nn.Linear(d_in, d_out, bias=self.use_bias)))
            if i < n_layers - 1:
                modules.append((f"relu-{i}", nn.ReLU()))

        return nn.Sequential(OrderedDict(modules))

    def forward(
        self, 
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ):
        """
        state: [B, |S|]
        """
        return self.compute_loss(state, action_chunk)

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
        super().__init__(state_dim, action_dim, chunk_size, use_bias)
        hidden_dims = list(hidden_dims)
        in_dims = [state_dim] + hidden_dims
        out_dims = hidden_dims + [action_dim * chunk_size]
        self.model = self.build_mlp(in_dims, out_dims)

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



class FlowMatchingPolicy(BasePolicy):
    """Predicts action chunks with a flow matching loss."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
        use_bias: bool = False,
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size, use_bias)
        # also conditions on A_tau and tau 
        hidden_dims = list(hidden_dims)
        action_chunk_size = action_dim * chunk_size
        # input is o_t, A_t,tau, tau 
        in_dims = [state_dim + action_chunk_size + 1] + hidden_dims
        out_dims = hidden_dims + [action_chunk_size]
        self.model = self.build_mlp(in_dims, out_dims)

    def get_random_noise(self, batch_size):
        """
        Returns random noise of shape [batch_size, chunk_size * |A|]
        """
        return torch.rand(batch_size, self.chunk_size * self.action_dim)

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        """
        state: [B, |S|]
        action_chunk: [B, chunk_size, |A|]
        """
        B, _ = state.shape
        # TODO: possible i did this reshape wrong
        # [B, chunk_size, |A|] => [B, chunk_size * |A|]
        action_chunk = action_chunk.reshape(B, -1)
        # create random noise for a_0
        a_0 = self.get_random_noise(B)
        # target velocity 
        v_target = action_chunk - a_0 
        # tau: [B, 1]
        # sample diff timesteps uniformly [0, 1) for each elem in the batch
        # TODO: idk if this is right or if we want to fix the timestep
        tau = torch.rand(B, device=state.device).unsqueeze(-1)
        # linearly interpolate to get a_tau 
        a_tau = (1 - tau) * a_0 + tau * action_chunk
        x_in = torch.cat([state, a_tau, tau], dim=1)
        v_pred = self.model(x_in)
        # TODO: kinda redundant way to define the loss
        loss = torch.mean(torch.norm(v_pred - v_target, p=2)**2)
        return loss

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        """
        state: [B, |S|]
        number of flow steps

        TODO: clean up the notation
        """
        # initialize a_t to a_0 
        B, _ = state.shape
        a_t = self.get_random_noise(B)
        tau_steps = torch.linspace(0, 1, steps=num_steps, device=state.device)
        d_tau = 1.0 / num_steps
        for i in range(num_steps):
            tau_t = torch.tensor([tau_steps[i]]*B, device=state.device).unsqueeze(-1)
            x_in = torch.cat([state, a_t, tau_t], dim=1)
            a_t = a_t + self.model(x_in) * d_tau
        return a_t.reshape(B, self.chunk_size, self.action_dim)


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
