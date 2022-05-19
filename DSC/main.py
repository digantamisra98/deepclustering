import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from post_clustering import spectral_clustering, acc, nmi
import scipy.io as sio
import math
import pandas as pd


class AutoEncoder(nn.Module):
    def __init__(self, sequence_length, hidden_sizes):
        """
        :param channels: a list containing all channels including the input image channel (1 for gray, 3 for RGB)
        """
        super(AutoEncoder, self).__init__()
        assert isinstance(hidden_sizes, list)
        self.encoder = nn.Sequential()
        self.encoder.add_module("input", nn.Linear(sequence_length, hidden_sizes[0]))
        for i in range(1, len(hidden_sizes)):
            #  Each layer will divide the size of feature map by 2
            self.encoder.add_module(
                "linear%d" % i,
                nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]),
            )
            self.encoder.add_module("relu%d" % i, nn.ReLU(True))

        self.decoder = nn.Sequential()
        hidden_sizes = list(reversed(hidden_sizes))
        for i in range(len(hidden_sizes) - 1):
            # Each layer will double the size of feature map
            self.decoder.add_module(
                "inv-linear%d" % (i + 1),
                nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]),
            )
            self.decoder.add_module("relud%d" % i, nn.ReLU(True))
        self.decoder.add_module("output", nn.Linear(hidden_sizes[-1], sequence_length))

    def forward(self, x):
        h = self.encoder(x)
        y = self.decoder(h)
        return y


class SelfExpression(nn.Module):
    def __init__(self, n):
        super(SelfExpression, self).__init__()
        self.Coefficient = nn.Parameter(
            1.0e-8 * torch.ones(n, n, dtype=torch.float32), requires_grad=True
        )

    def forward(self, x):  # shape=[n, d]
        y = torch.matmul(self.Coefficient, x)
        return y


class DSCNet(nn.Module):
    def __init__(self, num_samples, sequence_length, hidden_sizes):
        super(DSCNet, self).__init__()
        self.n = num_samples
        self.ae = AutoEncoder(sequence_length, hidden_sizes)
        self.self_expression = SelfExpression(self.n)

    def forward(self, x):  # shape=[n, c, w, h]
        z = self.ae.encoder(x)

        # self expression layer, reshape to vectors, multiply Coefficient, then reshape back
        shape = z.shape
        z = z.view(self.n, -1)  # shape=[n, d]
        z_recon = self.self_expression(z)  # shape=[n, d]
        z_recon_reshape = z_recon.view(shape)

        x_recon = self.ae.decoder(z_recon_reshape)  # shape=[n, c, w, h]
        return x_recon, z, z_recon

    def loss_fn(self, x, x_recon, z, z_recon, weight_coef, weight_selfExp):
        loss_ae = F.mse_loss(x_recon, x, reduction="sum")
        loss_coef = torch.sum(torch.pow(self.self_expression.Coefficient, 2))
        loss_selfExp = F.mse_loss(z_recon, z, reduction="sum")
        loss = loss_ae + weight_coef * loss_coef + weight_selfExp * loss_selfExp

        return loss


def train(
    model,  # type: DSCNet
    x,
    y,
    epochs,
    lr=1e-3,
    weight_coef=1.0,
    weight_selfExp=150,
    device="cuda",
    alpha=0.04,
    dim_subspace=12,
    ro=8,
    show=10,
):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32, device=device)
    x = x.to(device)
    if isinstance(y, torch.Tensor):
        y = y.to("cpu").numpy()
    K = len(np.unique(y))
    for epoch in range(epochs):
        x_recon, z, z_recon = model(x)
        loss = model.loss_fn(
            x,
            x_recon,
            z,
            z_recon,
            weight_coef=weight_coef,
            weight_selfExp=weight_selfExp,
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % show == 0 or epoch == epochs - 1:
            C = model.self_expression.Coefficient.detach().to("cpu").numpy()
            y_pred = spectral_clustering(C, K, dim_subspace, alpha, ro)
            print(
                "Epoch %02d: loss=%.4f, acc=%.4f, nmi=%.4f"
                % (epoch, loss.item() / y_pred.shape[0], acc(y, y_pred), nmi(y, y_pred))
            )


if __name__ == "__main__":
    import argparse
    import warnings

    parser = argparse.ArgumentParser(description="DSCNet")
    parser.add_argument(
        "--db",
        default="bitcoin",
        choices=["bitcoin", "wind"],
    )
    parser.add_argument("--show-freq", default=10, type=int)
    parser.add_argument("--ae-weights", default=None)
    parser.add_argument("--save-dir", default="results")
    args = parser.parse_args()
    print(args)
    import os

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    db = args.db

    if db == "bitcoin":
        # load data
        # data = sio.loadmat("datasets/ORL_32x32.mat")
        # x, y = data["fea"].reshape((-1, 1, 32, 32)), data["gnd"]
        # y = np.squeeze(y - 1)  # y in [0, 1, ..., K-1]

        data = pd.read_csv("cryp.csv")
        x = np.array(data["rate"].tolist())
        y = np.array(data["label"].tolist())
        y = np.squeeze(y - 1)  # y in [0, 1, ..., K-1]
        # X_train = X[:3500]
        # y_train = y[:3500]
        # X_test = X[3500:]
        # y_test = y[3500:]
        # train_data= TimeseriesDataset(X_train,y_train)
        # train_present = X_train[3500:]
        # test_data= TimeseriesDataset(X_test,y_test)
        # # train_dataset = TimeseriesDataset(X_lstm, y_lstm, seq_len=4)
        # train_loader = torch.utils.data.DataLoader(train_data, batch_size = args.batch_size, shuffle = False)
        # test_loader = torch.utils.data.DataLoader(test_data, batch_size = args.batch_size, shuffle = False)

        # network and optimization parameters
        num_sample = x.shape[0]
        channels = [1, 3, 3, 5]
        kernels = [3, 3, 3]
        epochs = 700
        weight_coef = 2.0
        weight_selfExp = 0.2

        # post clustering parameters
        alpha = 0.2  # threshold of C
        dim_subspace = 3  # dimension of each subspace
        ro = 1  #

    # dscnet = DSCNet(num_samples=num_sample, channels=channels, kernels=kernels)
    dscnet = DSCNet(
        num_samples=NUM_SAMPLES,
        sequence_length=SEQUENCE_LENGTH,
        hidden_sizes=[3, 3, 5],
    )
    dscnet.to(device)

    train(
        dscnet,
        x,
        y,
        epochs,
        weight_coef=weight_coef,
        weight_selfExp=weight_selfExp,
        alpha=alpha,
        dim_subspace=dim_subspace,
        ro=ro,
        show=args.show_freq,
        device=device,
    )
    torch.save(dscnet.state_dict(), args.save_dir + "/%s-model.ckp" % args.db)
