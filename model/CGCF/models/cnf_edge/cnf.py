import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn import Parameter
from torchdiffeq import odeint_adjoint
from torchdiffeq import odeint as odeint_normal


class CNF(nn.Module):
    def __init__(self, odefunc, T=1.0, train_T=False, regularization_fns=None,
                 solver='dopri5', atol=1e-5, rtol=1e-5, use_adjoint=True):
        super().__init__()
        self.train_T = train_T
        self.T = T
        if train_T:
            self.register_parameter("sqrt_end_time", nn.Parameter(torch.sqrt(torch.tensor(T))))

        if regularization_fns is not None and len(regularization_fns) > 0:
            raise NotImplementedError("Regularization not supported")
        self.use_adjoint = use_adjoint
        self.odefunc = odefunc
        self.solver = solver
        self.atol = atol
        self.rtol = rtol
        self.test_solver = solver
        self.test_atol = atol
        self.test_rtol = rtol
        self.solver_options = {}

    def forward(self, x, node_attr, edge_attr, edge_index, logpx=None, integration_times=None, reverse=False):
        if logpx is None:
            _logpx = torch.zeros(*x.shape[:-1], 1).to(x)
        else:
            _logpx = logpx

        states = (x, _logpx, node_attr, edge_attr)
        # atol = [self.atol] * 3
        # rtol = [self.rtol] * 3
        atol = [self.atol] * len(states)
        rtol = [self.rtol] * len(states)

        if integration_times is None:
            if self.train_T:
                integration_times = torch.stack(
                    [torch.tensor(0.0).to(x), self.sqrt_end_time * self.sqrt_end_time]
                ).to(x)
            else:
                integration_times = torch.tensor([0., self.T], requires_grad=False).to(x)

        if reverse:
            integration_times = _flip(integration_times, 0)

        # print(integration_times)

        # Refresh the odefunc statistics,
        # and prepare the graph.
        self.odefunc.before_odeint(edge_index=edge_index)
        odeint = odeint_adjoint if self.use_adjoint else odeint_normal
        if self.training:
            state_t = odeint(
                self.odefunc,
                states,
                integration_times.to(x),
                atol=atol,
                rtol=rtol,
                method=self.solver,
                options=self.solver_options,
            )
        else:
            state_t = odeint(
                self.odefunc,
                states,
                integration_times.to(x),
                atol=self.test_atol,
                rtol=self.test_rtol,
                method=self.test_solver,
            )

        if len(integration_times) == 2:
            state_t = tuple(s[1] for s in state_t)

        z_t, logpz_t = state_t[:2]

        if logpx is not None:
            return z_t, logpz_t
        else:
            return z_t

    def num_evals(self):
        return self.odefunc._num_evals.item()


def _flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1, dtype=torch.long, device=x.device)
    return x[tuple(indices)]


class SequentialFlow(nn.Module):
    """A generalized nn.Sequential container for normalizing flows."""

    def __init__(self, layer_list):
        super(SequentialFlow, self).__init__()
        self.chain = nn.ModuleList(layer_list)

    def forward(self, x, node_attr, edge_attr, edge_index, logpx=None, reverse=False, inds=None, integration_times=None):
        if inds is None:
            if reverse:
                inds = range(len(self.chain) - 1, -1, -1)
            else:
                inds = range(len(self.chain))

        if logpx is None:
            for i in inds:
                x = self.chain[i](
                    x, 
                    node_attr=node_attr, 
                    edge_attr=edge_attr, 
                    edge_index=edge_index, 
                    logpx=logpx, 
                    integration_times=integration_times, 
                    reverse=reverse
                )
            return x
        else:
            for i in inds:
                x, logpx = self.chain[i](
                    x, 
                    node_attr=node_attr, 
                    edge_attr=edge_attr, 
                    edge_index=edge_index, 
                    logpx=logpx, 
                    integration_times=integration_times, 
                    reverse=reverse
                )
            return x, logpx


def reduce_tensor(tensor, world_size=None):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    if world_size is None:
        world_size = dist.get_world_size()

    rt /= world_size
    return rt


class MovingBatchNormNd(nn.Module):
    def __init__(self, num_features, eps=1e-4, decay=0.1, bn_lag=0., affine=True, sync=False):
        super(MovingBatchNormNd, self).__init__()
        self.num_features = num_features
        self.sync = sync
        self.affine = affine
        self.eps = eps
        self.decay = decay
        self.bn_lag = bn_lag
        self.register_buffer('step', torch.zeros(1))
        if self.affine:
            self.weight = Parameter(torch.Tensor(num_features))
            self.bias = Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.reset_parameters()

    @property
    def shape(self):
        raise NotImplementedError

    def reset_parameters(self):
        self.running_mean.zero_()
        self.running_var.fill_(1)
        if self.affine:
            self.weight.data.zero_()
            self.bias.data.zero_()

    def forward(self, x, logpx=None, reverse=False):
        if reverse:
            return self._reverse(x, logpx)
        else:
            return self._forward(x, logpx)

    def _forward(self, x, logpx=None):
        num_channels = x.size(-1)
        used_mean = self.running_mean.clone().detach()
        used_var = self.running_var.clone().detach()

        if self.training:
            # compute batch statistics
            x_t = x.transpose(0, 1).reshape(num_channels, -1)
            batch_mean = torch.mean(x_t, dim=1)

            if self.sync:
                batch_ex2 = torch.mean(x_t**2, dim=1)
                batch_mean = reduce_tensor(batch_mean)
                batch_ex2 = reduce_tensor(batch_ex2)
                batch_var = batch_ex2 - batch_mean**2
            else:
                batch_var = torch.var(x_t, dim=1)

            # moving average
            if self.bn_lag > 0:
                used_mean = batch_mean - (1 - self.bn_lag) * (batch_mean - used_mean.detach())
                used_mean /= (1. - self.bn_lag**(self.step[0] + 1))
                used_var = batch_var - (1 - self.bn_lag) * (batch_var - used_var.detach())
                used_var /= (1. - self.bn_lag**(self.step[0] + 1))

            # update running estimates
            self.running_mean -= self.decay * (self.running_mean - batch_mean.data)
            self.running_var -= self.decay * (self.running_var - batch_var.data)
            self.step += 1

        # perform normalization
        used_mean = used_mean.view(*self.shape).expand_as(x)
        used_var = used_var.view(*self.shape).expand_as(x)

        y = (x - used_mean) * torch.exp(-0.5 * torch.log(used_var + self.eps))

        if self.affine:
            weight = self.weight.view(*self.shape).expand_as(x)
            bias = self.bias.view(*self.shape).expand_as(x)
            y = y * torch.exp(weight) + bias

        if logpx is None:
            return y
        else:
            return y, logpx - self._logdetgrad(x, used_var).sum(-1, keepdim=True)

    def _reverse(self, y, logpy=None):
        used_mean = self.running_mean
        used_var = self.running_var

        if self.affine:
            weight = self.weight.view(*self.shape).expand_as(y)
            bias = self.bias.view(*self.shape).expand_as(y)
            y = (y - bias) * torch.exp(-weight)

        used_mean = used_mean.view(*self.shape).expand_as(y)
        used_var = used_var.view(*self.shape).expand_as(y)
        x = y * torch.exp(0.5 * torch.log(used_var + self.eps)) + used_mean

        if logpy is None:
            return x
        else:
            return x, logpy + self._logdetgrad(x, used_var).sum(-1, keepdim=True)

    def _logdetgrad(self, x, used_var):
        logdetgrad = -0.5 * torch.log(used_var + self.eps)
        if self.affine:
            weight = self.weight.view(*self.shape).expand(*x.size())
            logdetgrad += weight
        return logdetgrad

    def __repr__(self):
        return (
            '{name}({num_features}, eps={eps}, decay={decay}, bn_lag={bn_lag},'
            ' affine={affine})'.format(name=self.__class__.__name__, **self.__dict__)
        )


def stable_var(x, mean=None, dim=1):
    if mean is None:
        mean = x.mean(dim, keepdim=True)
    mean = mean.view(-1, 1)
    res = torch.pow(x - mean, 2)
    max_sqr = torch.max(res, dim, keepdim=True)[0]
    var = torch.mean(res / max_sqr, 1, keepdim=True) * max_sqr
    var = var.view(-1)
    # change nan to zero
    var[var != var] = 0
    return var


class MovingBatchNorm1d(MovingBatchNormNd):
    @property
    def shape(self):
        return [1, -1]

    def forward(self, x, node_attr, edge_attr, edge_index, logpx=None, integration_times=None, reverse=False):
        ret = super(MovingBatchNorm1d, self).forward(x, logpx=logpx, reverse=reverse)
        return ret
