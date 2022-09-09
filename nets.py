import torch
import torch.nn.functional as F
from torch import nn, jit
import numpy as np
from distribs import TruncatedNormal, SquashedNormal
from torch.nn.utils import spectral_norm
from utils import weight_init, qr_weight_init


def maybe_sn(m, use_sn):
    return spectral_norm(m) if use_sn else m


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims, spectral_norms):
        super().__init__()
        assert len(hidden_dims) == len(spectral_norms)
        layers = []
        for dim, use_sn in zip(hidden_dims, spectral_norms):
            layers += [
                maybe_sn(nn.Linear(input_dim, dim), use_sn),
                nn.ReLU(inplace=True),
            ]
            input_dim = dim

        layers += [nn.Linear(input_dim, output_dim)]

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class DoubleQCritic(nn.Module):
    def __init__(
        self, obs_type, obs_dim, action_dim, feature_dim, hidden_dims, spectral_norms
    ):
        super().__init__()

        self.obs_type = obs_type

        if obs_type == "pixels":
            # for pixels actions will be added after trunk
            self.trunk = nn.Sequential(
                nn.Linear(obs_dim, feature_dim), nn.LayerNorm(feature_dim), nn.Tanh()
            )
            trunk_dim = feature_dim + action_dim
        else:
            # for states actions come in the beginning
            self.trunk = nn.Sequential(
                nn.Linear(obs_dim + action_dim, hidden_dims[0]),
                nn.LayerNorm(hidden_dims[0]),
                nn.Tanh(),
            )
            trunk_dim = hidden_dims[0]

        self.q1_net = MLP(trunk_dim, 1, hidden_dims, spectral_norms)
        self.q2_net = MLP(trunk_dim, 1, hidden_dims, spectral_norms)

        self.apply(weight_init)

    def forward(self, obs, action):
        inpt = obs if self.obs_type == "pixels" else torch.cat([obs, action], dim=-1)
        h = self.trunk(inpt)
        h = torch.cat([h, action], dim=-1) if self.obs_type == "pixels" else h

        q1 = self.q1_net(h)
        q2 = self.q2_net(h)

        return q1, q2


class DeterministicActor(nn.Module):
    def __init__(self, obs_dim, action_dim, feature_dim, hidden_dims, spectral_norms):
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, feature_dim), nn.LayerNorm(feature_dim), nn.Tanh()
        )

        self.policy_net = MLP(feature_dim, action_dim, hidden_dims, spectral_norms)

        self.apply(weight_init)

    def forward(self, obs, std):
        h = self.trunk(obs)
        mu = self.policy_net(h)
        mu = torch.tanh(mu)
        std = torch.ones_like(mu) * std

        dist = TruncatedNormal(mu, std)
        return dist


class StochasticActor(nn.Module):
    def __init__(
        self,
        obs_dim,
        action_dim,
        feature_dim,
        hidden_dims,
        spectral_norms,
        log_std_bounds,
    ):
        super().__init__()

        self.log_std_bounds = log_std_bounds

        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, feature_dim), nn.LayerNorm(feature_dim), nn.Tanh()
        )

        self.policy_net = MLP(feature_dim, 2 * action_dim, hidden_dims, spectral_norms)

        self.apply(weight_init)

    def forward(self, obs):
        h = self.trunk(obs)
        mu, log_std = self.policy_net(h).chunk(2, dim=-1)

        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std_min, log_std_max = self.log_std_bounds
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1)

        std = log_std.exp()

        dist = SquashedNormal(mu, std)
        return dist


class Encoder(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()

        self.repr_dim = 32 * 35 * 35

        self.convnet = nn.Sequential(
            nn.Conv2d(obs_dim, 32, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
        )

        self.apply(weight_init)

    def forward(self, obs):
        obs = obs / 255.0 - 0.5
        h = self.convnet(obs)
        h = h.view(h.shape[0], -1)
        return h


class DQNEncoder(nn.Module):
    def __init__(self, obs_dim, qr=False):
        super().__init__()
        self.repr_dim = 64 * 7 * 7

        self.convnet = nn.Sequential(
            nn.Conv2d(obs_dim, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
        )

        init_fn = weight_init if not qr else qr_weight_init
        self.apply(init_fn)

    def forward(self, obs):
        obs = obs / 255.0
        h = self.convnet(obs)
        h = h.view(h.shape[0], -1)
        return h


class DQNCritic(nn.Module):
    def __init__(self, repr_dim, num_actions, hidden_dim, feature_dim, trunk_type):
        super().__init__()

        # Create Trunk
        if trunk_type == "id":
            self.trunk = nn.Identity()
            self.feature_dim = repr_dim
        elif trunk_type == "proj":
            self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim))
            self.feature_dim = feature_dim
        elif trunk_type == "proj+ln+tanh":
            self.trunk = nn.Sequential(
                nn.Linear(repr_dim, feature_dim),
                nn.LayerNorm(feature_dim),
                nn.Tanh(),
            )
            self.feature_dim = feature_dim

        # Critic Heads
        self.V = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )
        self.A = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )

        self.apply(weight_init)

    def forward(self, obs):
        h = self.trunk(obs)
        v = self.V(h)
        a = self.A(h)
        q = v + a - a.mean(1, keepdim=True)
        return q


class OTILRep(nn.Module):
    def __init__(
        self,
        obs_dim,
        num_actions,
        hidden_dim,
        feature_dim,
        encoder_params=None,
        critic_params=None,
    ):
        super().__init__()
        self.encoder = DQNEncoder(obs_dim)
        self.critic = DQNCritic(
            self.encoder.repr_dim, num_actions, hidden_dim, feature_dim, trunk_type="id"
        )

        self.load_weights(encoder_params, critic_params)

        # remove last layers
        self.critic.V[2] = nn.Identity()
        self.critic.A[2] = nn.Identity()

        # Feature Dim
        self.feature_dim = feature_dim

    def load_weights(self, encoder_params=None, critic_params=None):
        if encoder_params:
            self.encoder.load_state_dict(encoder_params)
        if critic_params:
            self.critic.load_state_dict(critic_params)

    def forward(self, obs):
        h_enc = self.encoder(obs)
        h_v = self.critic.V(h_enc)
        h_a = self.critic.A(h_enc)
        return h_enc, h_v, h_a


# TODO: after testing switch to this
class NewQRDQNCritic(nn.Module):
    def __init__(
        self,
        repr_dim,
        num_actions,
        num_quantiles,
        feature_dim=512,
        trunk_type="proj",
        dueling_net=False,
        noisy_linear=False,
    ):
        super().__init__()
        self.n_act = num_actions
        self.n_quant = num_quantiles
        self.dueling_net = dueling_net
        self.noisy_linear = noisy_linear
        self.feature_dim = feature_dim

        linear = NoisyLinear if noisy_linear else nn.Linear

        if dueling_net:
            self.V = nn.Sequential(
                linear(repr_dim, feature_dim),
                nn.ReLU(),
                linear(feature_dim, 1),
            )
            self.A = nn.Sequential(
                linear(repr_dim, feature_dim),
                nn.ReLU(),
                linear(feature_dim, num_actions * num_quantiles),
            )
        else:
            self.Q = nn.Sequential(
                linear(repr_dim, feature_dim),
                nn.ReLU(),
                linear(feature_dim, num_actions * num_quantiles),
            )

        self.apply(qr_weight_init)

    def forward(self, obs):
        bs = obs.size(0)

        if not self.dueling_net:
            quantiles = self.Q(obs).view(bs, self.n_quant, self.n_act)
        else:
            advantages = self.A(obs).view(bs, self.n_quant, self.n_act)
            baselines = self.V(obs).view(bs, self.n_quant, 1)
            quantiles = baselines + advantages - advantages.mean(dim=2, keepdim=True)

        return quantiles


class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma=0.5):
        super(NoisyLinear, self).__init__()

        # Learnable parameters.
        self.mu_W = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.sigma_W = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.mu_bias = nn.Parameter(torch.FloatTensor(out_features))
        self.sigma_bias = nn.Parameter(torch.FloatTensor(out_features))

        # Factorized noise parameters.
        self.register_buffer("eps_p", torch.FloatTensor(in_features))
        self.register_buffer("eps_q", torch.FloatTensor(out_features))

        self.in_features = in_features
        self.out_features = out_features
        self.sigma = sigma

        self.reset()
        self.sample()

    def reset(self):
        bound = 1 / np.sqrt(self.in_features)
        self.mu_W.data.uniform_(-bound, bound)
        self.mu_bias.data.uniform_(-bound, bound)
        self.sigma_W.data.fill_(self.sigma / np.sqrt(self.in_features))
        self.sigma_bias.data.fill_(self.sigma / np.sqrt(self.out_features))

    def f(self, x):
        return x.normal_().sign().mul(x.abs().sqrt())

    def sample(self):
        self.eps_p.copy_(self.f(self.eps_p))
        self.eps_q.copy_(self.f(self.eps_q))

    def forward(self, x):
        if self.training:
            weight = self.mu_W + self.sigma_W * self.eps_q.ger(self.eps_p)
            bias = self.mu_bias + self.sigma_bias * self.eps_q.clone()
        else:
            weight = self.mu_W
            bias = self.mu_bias

        return F.linear(x, weight, bias)


class QRDQNCritic(nn.Module):
    def __init__(
        self, repr_dim, num_actions, num_quantiles, feature_dim=512, trunk_type="proj"
    ):
        super().__init__()
        self.n_act = num_actions
        self.n_quant = num_quantiles

        # Create Trunk
        if trunk_type == "id":
            self.trunk = nn.Identity()
            self.feature_dim = repr_dim
        elif trunk_type == "proj":
            self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim))
            self.feature_dim = feature_dim
        elif trunk_type == "proj+ln+tanh":
            self.trunk = nn.Sequential(
                nn.Linear(repr_dim, feature_dim),
                nn.LayerNorm(feature_dim),
                nn.Tanh(),
            )
            self.feature_dim = feature_dim

        self.Q = nn.Linear(feature_dim, num_actions * num_quantiles)

        self.apply(qr_weight_init)

    def forward(self, obs):
        bs = obs.size(0)
        # NOTE: adding non-linearity here like original paper
        h = F.relu(self.trunk(obs))
        quantiles = self.Q(h).view(bs, self.n_quant, self.n_act)

        return quantiles


class Discriminator(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        enc_input_dim=None,
        enc_output_dim=None,
        output_dim=1,
    ):
        super().__init__()

        if enc_input_dim is not None:
            if not enc_output_dim:
                enc_output_dim = input_dim
            self.encoder = Encoder(enc_input_dim)
            self.encoder_trunk = nn.Sequential(
                nn.Linear(self.encoder.repr_dim, enc_output_dim),
                nn.LayerNorm(enc_output_dim),
                nn.Tanh(),
            )
        else:
            self.encoder = nn.Identity()

        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

        self.apply(weight_init)

    def encode(self, obs):
        return self.encoder_trunk(self.encoder(obs))

    def forward(self, obs, encode=False):
        if encode:
            obs = self.encode(obs)
        return self.trunk(obs)


class InverseDynamicsModel(nn.Module):
    def __init__(self, obs_dim, hidden_dim, action_dim, mlp_dense=False, separate_enc=False):
        """Note (s, s') so 8 channels"""
        super().__init__()
        self.mlp_dense = mlp_dense
        self.separate_enc = separate_enc
        repr_dim = 64 * 7 * 7
        
        if not separate_enc:
            self.convnet = nn.Sequential(
                nn.Conv2d(obs_dim, 32, 8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, 4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, 3, stride=1),
                nn.ReLU(),
                nn.Flatten(),
            )
        else:
            self.convnet = nn.Sequential(
                nn.Conv2d(int(obs_dim/2), 32, 8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, 4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, 3, stride=1),
                nn.ReLU(),
                nn.Flatten(),
            )
            self.convnet2 = nn.Sequential(
                nn.Conv2d(int(obs_dim/2), 32, 8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, 4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, 3, stride=1),
                nn.ReLU(),
                nn.Flatten(),
            )

        self.n_layers = 3
        linears = []
        prev_dim = None
        for i in range(3):
            if i == 0:
                if not separate_enc:
                    in_dim = repr_dim
                else:
                    in_dim = repr_dim * 2
                prev_dim = in_dim
            else:
                in_dim = hidden_dim
                if mlp_dense:
                    in_dim += prev_dim
                prev_dim += hidden_dim

            out_dim = action_dim if i == self.n_layers - 1 else hidden_dim

            linears.append(nn.Linear(in_dim, out_dim))

        self.linears = nn.ModuleList(linears)

        self.apply(weight_init)

    def forward(self, obs, next_obs):
        obs = obs / 255.0 - 0.5
        next_obs = next_obs / 255.0 - 0.5
        if not self.separate_enc:
            h = self.convnet(torch.cat([obs, next_obs], dim=1))
        else:
            h1 = self.convnet(obs)
            h2 = self.convnet(next_obs)
            h = torch.cat([h1, h2], dim=1)
        # MLP / DenseNet
        outs = [h]
        for i, layer in enumerate(self.linears):
            out = layer(h)
            outs.append(out)
            if self.mlp_dense:
                h = torch.cat(outs, dim=1)  # residual layer
            else:
                h = out
            if i != self.n_layers - 1:
                h = F.relu(h)
        return h

#class InverseDynamicsModel(nn.Module):
#    def __init__(self, obs_dim, hidden_dim, action_dim, mlp_dense=False, separate_enc=False):
#        """Note (s, s') so 8 channels"""
#        super().__init__()
#        self.mlp_dense = mlp_dense
#        self.separate_enc = separate_enc
#        repr_dim = 64 * 7 * 7
#        
#        if not separate_enc:
#            self.convnet = nn.Sequential(
#                nn.Conv2d(obs_dim, 32, 8, stride=4),
#                nn.ReLU(),
#                nn.Conv2d(32, 64, 4, stride=2),
#                nn.ReLU(),
#                nn.Conv2d(64, 64, 3, stride=1),
#                nn.ReLU(),
#                nn.Flatten(),
#            )
#        else:
#            self.convnet = nn.Sequential(
#                nn.Conv2d(int(obs_dim/2), 32, 8, stride=4),
#                nn.ReLU(),
#                nn.Conv2d(32, 64, 4, stride=2),
#                nn.ReLU(),
#                nn.Conv2d(64, 64, 3, stride=1),
#                nn.ReLU(),
#                nn.Flatten(),
#            )
#            self.convnet2 = nn.Sequential(
#                nn.Conv2d(int(obs_dim/2), 32, 8, stride=4),
#                nn.ReLU(),
#                nn.Conv2d(32, 64, 4, stride=2),
#                nn.ReLU(),
#                nn.Conv2d(64, 64, 3, stride=1),
#                nn.ReLU(),
#                nn.Flatten(),
#            )
#        
#        self.linears = nn.Sequential(
#            nn.Linear(repr_dim, hidden_dim),
#            nn.ReLU(),
#            nn.Linear(hidden_dim, hidden_dim),
#            nn.ReLU(),
#            nn.Linear(hidden_dim, action_dim),
#        )
#
#        self.apply(weight_init)
#
#    def forward(self, obs, next_obs):
#        obs = obs / 255.0 - 0.5
#        next_obs = next_obs / 255.0 - 0.5
#        if not self.separate_enc:
#            h = self.convnet(torch.cat([obs, next_obs], dim=1))
#        else:
#            h1 = self.convnet(obs)
#            h2 = self.convnet(next_obs)
#            h = torch.cat([h1, h2], dim=1)
#        return self.linears(h)

def inverse_test():
    device = torch.device('cuda')
    models = [InverseDynamicsModel(8, 512, 4, False, False).to(device) for _ in range(7)]
    obs = torch.rand(7, 64, 4, 84, 84).to(device)
    next_obs = torch.rand(7, 64,  4, 84, 84).to(device)
    #out = idm(obs, next_obs)

    from functorch import vmap, combine_state_for_ensemble

    fmodel, params, buffers = combine_state_for_ensemble(models)
    [p.requires_grad_() for p in params]

    from torch.utils.benchmark import Timer
    without_vmap = Timer(
        stmt="[model(s, sp) for model, s, sp in zip(models, obs, next_obs)]",
        globals=globals()
    )
    with_vmap = Timer(
        stmt="vmap(fmodel)(params, buffers, obs, next_obs)",
        globals=globals()
    )
    without_vmap_single = Timer(
        stmt="models[0](obs[0], next_obs[0])",
        globals=globals()
    )
    print(f'Predictions without vmap {without_vmap.timeit(100)}')
    print(f'Predictions with vmap {with_vmap.timeit(100)}')
    print(f'Predictions without vmap single {without_vmap_single.timeit(100)}')

def head_ensemble_test():
    device = torch.device('cuda')
    encoder = DQNEncoder(8).to(device)
    heads = [nn.Sequential(
            nn.Linear(encoder.repr_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        ).to(device) for _ in range(7)]
    
    obs = torch.rand(7, 64, 4, 84, 84).to(device)
    next_obs = torch.rand(7, 64,  4, 84, 84).to(device)

    def run():
        x = torch.cat([obs, next_obs], dim=1)
        feat = encoder(x)
        out = [head(feat) for head in heads]
        return torch.stack(out)

    separate_heads = Timer(
        stmt="run()", globals=globals()
            )
    print(f'Separate Heads without vmap {separate_heads.timeit(100)}')



if __name__ == "__main__":
    from torch.utils.benchmark import Timer
    hidden_dim=512
    action_dim=4
    device = torch.device('cuda')
    conv_groups = nn.Sequential(
                nn.Conv2d(8*7, 32*7, 8, stride=4, groups=7),
                nn.ReLU(),
                nn.Conv2d(32*7, 64*7, 4, stride=2, groups=7),
                nn.ReLU(),
                nn.Conv2d(64*7, 64*7, 3, stride=1, groups=7),
                nn.ReLU(),
                nn.Flatten()
            ).to(device)
    flatten_layer = nn.Flatten().to(device)
    convnet = [nn.Sequential(
        nn.Conv2d(8, 32, 8, stride=4),
        nn.ReLU(),
        nn.Conv2d(32, 64, 4, stride=2),
        nn.ReLU(),
        nn.Conv2d(64, 64, 3, stride=1),
        nn.ReLU(),
        nn.Flatten()
    ).to(device) for _ in range(7)]
    obs_group = torch.rand(64, 4*7, 84, 84).to(device)
    next_obs_group = torch.rand(64,  4*7, 84, 84).to(device)
    
    obs = torch.rand(64, 4, 84, 84).to(device)
    next_obs = torch.rand(64,  4, 84, 84).to(device)

    from functorch import vmap, combine_state_for_ensemble

    fmodel, params, buffers = combine_state_for_ensemble(convnet)
    [p.requires_grad_() for p in params]

    def run():
        x = torch.cat([obs_group, next_obs_group], dim=1)
        out = conv_groups(x)
        return out.reshape(64, 7, -1).transpose(0, 1)

    def run_vmap():
        x = torch.cat([obs, next_obs], dim=1)
        x = x.repeat(7, 1, 1, 1, 1)
        return vmap(fmodel)(params, buffers, x)
    separate_heads = Timer(
        stmt="run()", globals=globals()
            )
    separate_heads_vmap = Timer(stmt="run_vmap()", globals=globals())
    print(f'Groups {separate_heads.timeit(1000)}')
    print(f'Vmap {separate_heads_vmap.timeit(1000)}')
    exit()
    
    out = run_vmap()
    print(out.shape)
    exit()
    out, flatten_out = run()
    print(out.shape)
    print(flatten_out.shape)
    h = out[:, 64:128, :, :].view(64, -1)
    h2 = flatten_out[:, 64*7*7:2*64*7*7]
    
    #final = flatten_out.reshape(64, 7, -1).movedim(0, 1)
    final = flatten_out.reshape(64, 7, -1).transpose(0, 1)
    
    print(torch.all(final[1] == h2))
    #out2 = out.reshape(64, 7, -1)
    #out3 = out2.movedim(0, 1)
    #print(torch.all(out3[1] == out2[:, 1]))
    #print(torch.all(out[:, 64*7*7:64*7*7*2]== out2[:, 1]))
    #print(out2.shape)
    #print(64*7*7)
    #print(out.shape)
#    out2 = out.reshape(64, 7, 64, 7, 7)
#    out3 = out.reshape(7, 64, 64, 7, 7)
#    print(torch.all(out2[:, 1, :, :, :] == out[:, 64:128, :, :]))
#    print(torch.all(out3[1, :, :, :, :].movedim(0, 1) == out[:, 64:128, :, :]))
#    print(out.shape)
    exit()

    separate_heads = Timer(
        stmt="run()", globals=globals()
            )
    separate_heads_vmap = Timer(stmt="run_vmap()", globals=globals())
    print(f'Groups {separate_heads.timeit(100)}')
    print(f'Vmap {separate_heads_vmap.timeit(100)}')
    exit()
    encoder = DQNEncoder(8).to(device)
    heads = [nn.Sequential(
            nn.Linear(encoder.repr_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        ).to(device) for _ in range(7)]
    
    obs = torch.rand(64, 4, 84, 84).to(device)
    next_obs = torch.rand(64,  4, 84, 84).to(device)

    from functorch import vmap, combine_state_for_ensemble

    fmodel, params, buffers = combine_state_for_ensemble(heads)
    [p.requires_grad_() for p in params]

    def run():
        x = torch.cat([obs, next_obs], dim=1)
        feat = encoder(x)
        out = [head(feat) for head in heads]
        return torch.stack(out)

    def run_vmap():
        x = torch.cat([obs, next_obs], dim=1)
        feat = encoder(x)
        return vmap(fmodel, in_dims=(0, 0, None))(params, buffers, feat)

    separate_heads = Timer(
        stmt="run()", globals=globals()
            )
    separate_heads_vmap = Timer(stmt="run_vmap()", globals=globals())
    print(f'Separate Heads without vmap {separate_heads.timeit(100)}')
    print(f'Separate Heads with vmap {separate_heads_vmap.timeit(100)}')
