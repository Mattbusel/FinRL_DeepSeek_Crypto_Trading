import torch as th
import torch.nn as nn
from typing import Tuple

TEN = th.Tensor

"""network"""


class NnSeqBnMLP(nn.Module):
    def __init__(self, dims, if_inp_norm=False, if_layer_norm=True, activation=None):
        super(NnSeqBnMLP, self).__init__()

        mlp_list = []
        if if_inp_norm:
            mlp_list.append(nn.BatchNorm1d(dims[0], momentum=0.9))

        mlp_list.append(nn.Linear(dims[0], dims[1]))
        for i in range(1, len(dims) - 1):
            mlp_list.append(nn.GELU())
            mlp_list.append(nn.LayerNorm(dims[i])) if if_layer_norm else None
            mlp_list.append(nn.Linear(dims[i], dims[i + 1]))

        if activation is not None:
            mlp_list.append(activation)

        self.mlp = nn.Sequential(*mlp_list)

        if activation is not None:
            layer_init_with_orthogonal(self.mlp[-2], std=0.1)
        else:
            layer_init_with_orthogonal(self.mlp[-1], std=0.1)

    def forward(self, seq):
        d0, d1, d2 = seq.shape
        inp = seq.reshape(d0 * d1, -1)
        out = self.mlp(inp)
        return out.reshape(d0, d1, -1)

    def reset_parameters(self, std=1.0, bias_const=1e-6):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                th.nn.init.orthogonal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, bias_const)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, std)
                nn.init.constant_(module.bias, 0)


class RnnRegNet(nn.Module):
    def __init__(self, inp_dim, mid_dim, out_dim, num_layers):
        super(RnnRegNet, self).__init__()
        self.rnn1 = nn.LSTM(mid_dim, mid_dim, num_layers=num_layers)
        self.mlp1 = NnSeqBnMLP(
            dims=(mid_dim, mid_dim, mid_dim), if_layer_norm=False, activation=nn.GELU()
        )

        self.rnn2 = nn.GRU(mid_dim, mid_dim, num_layers=num_layers)
        self.mlp2 = NnSeqBnMLP(
            dims=(mid_dim, mid_dim, mid_dim), if_layer_norm=False, activation=nn.GELU()
        )

        self.mlp_inp = NnSeqBnMLP(
            dims=(inp_dim, mid_dim, mid_dim), if_layer_norm=False, activation=nn.GELU()
        )
        self.mlp_out = NnSeqBnMLP(
            dims=(mid_dim * 2, mid_dim * 2, out_dim),
            if_layer_norm=False,
            activation=nn.Tanh(),
        )

    def forward(self, inp, hid=None):
        hid1, hid2 = (None, None) if hid is None else hid
        inp = self.mlp_inp(inp)

        rnn1, hid1 = self.rnn1(inp, hid1)
        tmp1 = self.mlp1(rnn1)

        rnn2, hid2 = self.rnn2(inp, hid2)
        tmp2 = self.mlp2(rnn2)

        tmp = th.concat((tmp1, tmp2), dim=2)
        out = self.mlp_out(tmp)
        return out, (hid1, hid2)

    @staticmethod
    def get_obj_value(criterion, out: TEN, lab: TEN, wup_dim: int) -> TEN:
        obj = criterion(out, lab)[wup_dim:, :, :]
        return obj


def layer_init_with_orthogonal(layer, std=1.0, bias_const=1e-6):
    th.nn.init.orthogonal_(layer.weight, std)
    th.nn.init.constant_(layer.bias, bias_const)


def check_rnn_reg_net():
    seq_len = 3600
    batch_size = 3
    inp_dim = 10
    mid_dim = 16
    out_dim = 8
    num_layers = 2

    net = RnnRegNet(
        inp_dim=inp_dim, mid_dim=mid_dim, out_dim=out_dim, num_layers=num_layers
    )

    inp = th.randn(seq_len, batch_size, inp_dim)
    lab = th.randn(seq_len, batch_size, out_dim)

    out, hid = net.forward(inp)
    print(out.shape)
    loss_func = nn.MSELoss()
    opt = th.optim.Adam(net.parameters(), lr=1e-3)

    for i in range(10):
        out, _ = net.forward(inp)
        loss = loss_func(out, lab)
        print(i, loss)
        opt.zero_grad()
        loss.backward()
        opt.step()


def check_rnn_in_real_trading():
    # Assume that the input sequence is a vector sequence of length seq_length, and the dimension of each vector is input_size
    seq_length = 24
    input_size = 10

    # Define a simple RNN model
    rnn = RnnRegNet(input_size, 20, 1, 1)

    # Randomly generate an input sequence
    input_seq = th.randn(seq_length, 1, input_size)  # Assume batch size is 1

    # Initialize the hidden state of the RNN
    hidden_state = None

    # Use a for loop to output each bit in the sequence dimension
    output_seq = []
    for t in range(seq_length):
        # Get the input for the current time step
        input_t = input_seq[t, :, :].unsqueeze(0)

        # Run the RNN on the current time step
        output_t, hidden_state = rnn(input_t, hidden_state)

        # Save the output of the current time step to the output sequence
        output_seq.append(output_t)

    # Concatenate the output sequences into a tensor
    output_seq = th.cat(output_seq, dim=0)

    output_seq2, _ = rnn(input_seq)

    # Output sequence dimensions
    print(output_seq.shape)
    t = th.abs(output_seq - output_seq2)
    print(t)
    print(t.mean())


if __name__ == "__main__":
    check_rnn_reg_net()
    # check_rnn_in_real_trading()