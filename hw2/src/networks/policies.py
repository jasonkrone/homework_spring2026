import itertools
from torch import nn
from torch.nn import functional as F
import torch.distributions as D
from torch import optim

import numpy as np
import torch

from infrastructure import pytorch_util as ptu


class MLPPolicy(nn.Module):
    """Base MLP policy, which can take an observation and output a distribution over actions.

    This class should implement the `forward` and `get_action` methods. The `update` method should be written in the
    subclasses, since the policy update rule differs for different algorithms.
    """

    def __init__(
        self,
        ac_dim: int,
        ob_dim: int,
        discrete: bool,
        n_layers: int,
        layer_size: int,
        learning_rate: float,
    ):
        super().__init__()

        if discrete:
            self.logits_net = ptu.build_mlp(
                input_size=ob_dim,
                output_size=ac_dim,
                n_layers=n_layers,
                size=layer_size,
            ).to(ptu.device)
            parameters = self.logits_net.parameters()
        else:
            self.mean_net = ptu.build_mlp(
                input_size=ob_dim,
                output_size=ac_dim,
                n_layers=n_layers,
                size=layer_size,
            ).to(ptu.device)
            self.logstd = nn.Parameter(
                torch.zeros(ac_dim, dtype=torch.float32, device=ptu.device)
            )
            parameters = itertools.chain([self.logstd], self.mean_net.parameters())
            

        self.optimizer = optim.Adam(
            parameters,
            learning_rate,
        )

        self.discrete = discrete

    @torch.no_grad()
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """Takes a single observation (as a numpy array) and returns a single action (as a numpy array)."""
        obs = torch.tensor(obs, device=ptu.device, dtype=torch.float)
        fw = self.forward(obs)

        if self.discrete:
            categorical = D.Categorical(logits=fw)
            action = ptu.to_numpy(categorical.sample())
        else:
            mu, stdev = fw
            action = D.MultiVariateNormal(mu, scale_tril=torch.diag(stdev)).sample()
        return action

    def forward(self, obs: torch.FloatTensor):
        """
        This function defines the forward pass of the network.  You can return anything you want, but you should be
        able to differentiate through it. For example, you can return a torch.FloatTensor. You can also return more
        flexible objects, such as a `torch.distributions.Distribution` object. It's up to you!
        """
        # TODO: jpk probably better to do distributions here

        if self.discrete:
            out = self.logits_net(obs)
        else:
            mu = self.mean_net(obs)
            stdev = torch.exp(self.logstd)
            out = (mu, stdev)
        return out

    def update(self, obs: np.ndarray, actions: np.ndarray, *args, **kwargs) -> dict:
        """
        Performs one iteration of gradient descent on the provided batch of data. You don't need to implement this
        method in the base class, but you do need to implement it in the subclass.
        """
        raise NotImplementedError


class MLPPolicyPG(MLPPolicy):
    """Policy subclass for the policy gradient algorithm."""

    def update(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        advantages: np.ndarray,
    ) -> dict:
        """Implements the policy gradient actor update."""
        """
        obs: [T, d_obs] 
        """
        obs = ptu.from_numpy(obs)
        actions = ptu.from_numpy(actions).long()
        advantages = ptu.from_numpy(advantages)

        self.optimizer.zero_grad()

        if self.discrete:
            logits = self.forward(obs)
            loss_fn = nn.modules.loss.CrossEntropyLoss(reduction="none")
            loss = loss_fn(logits, actions) 
        else:
            mu, stdev = self.forward(obs)
            loss = -1.0 * D.MultiVariateNormal(mu, scale_tril=torch.diag(stdev)).log_prob(actions)

        loss = loss * advantages
        loss = loss.mean()

        # i feel like this reduction is wrong b/c it's reducing by 1 / (N*H) vs. just 1/N 
        loss.backward()
        self.optimizer.step()

        return {
            "Actor Loss": loss.item(),
        }
