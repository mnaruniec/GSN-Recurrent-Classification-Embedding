from copy import deepcopy
from datetime import datetime
from functools import partial
from itertools import chain

from torch import nn, optim
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt

from constants import *
from input import get_dataloaders


class ParticleNet(nn.Module):
    def __init__(
            self,
            hidden_fc=DEFAULT_HIDDEN_FC,
            recurrent_layers=DEFAULT_RECURRENT_LAYERS,
            recurrent_features=DEFAULT_RECURRENT_FEATURES,
    ):
        super().__init__()

        self.recurrent = nn.LSTM(
            2, hidden_size=recurrent_features, num_layers=recurrent_layers, batch_first=True,
        )

        features = [recurrent_features] + hidden_fc
        hidden_layers = chain(*[
            [
                nn.Linear(in_features=in_features, out_features=out_features),
                nn.ReLU(),
                nn.BatchNorm1d(out_features),
            ]
            for in_features, out_features in zip(features, hidden_fc)
        ])

        self.linear = nn.Sequential(
            *hidden_layers,
            nn.Linear(in_features=features[-1], out_features=NUM_CLASSES),
        )

    def forward(self, x):
        out, (h, c) = self.recurrent(x)
        x = h[-1]
        x = self.linear(x)
        return x


class ParticleTrainer:
    def __init__(
            self,
            embedding = False,
            optimizer_lambda = partial(optim.Adam, lr=DEFAULT_LR, weight_decay=DEFAULT_WEIGHT_DECAY, amsgrad=True),
            mb_size=DEFAULT_MB_SIZE,
            num_epochs=DEFAULT_NUM_EPOCHS,
            patience=DEFAULT_PATIENCE,
            stat_period=DEFAULT_STAT_PERIOD,
            stat_mbs=DEFAULT_STAT_MBS,
            epoch_train_eval=DEFAULT_EPOCH_TRAIN_EVAL,
            **net_kwargs,
    ):
        self.net_kwargs = net_kwargs

        self.optimizer_lambda = optimizer_lambda

        self.mb_size = mb_size
        self.num_epochs = num_epochs
        self.patience = patience
        self.stat_period = stat_period
        self.stat_mbs = stat_mbs
        self.epoch_train_eval = epoch_train_eval

        self.net = None
        self.criterion = None
        self.optimizer = None

        self.embedding = embedding

        self.train_dl, self.valid_dl, self.test_dl = get_dataloaders()
        self.trunc_train_dl, self.trunc_valid_dl, self.trunc_test_dl = get_dataloaders(truncation_p=0.3)

    def init_net(self):
        self.net = ParticleNet(**self.net_kwargs)
        self.net.to(DEVICE)
        self.net.train()

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = self.optimizer_lambda(self.net.parameters())

    def evaluate_on(self, dataloader: DataLoader, full=False) -> (float, int, float):
        """ Returns (pixel_acc, pixel_count, avg_loss) """
        with torch.no_grad():
            net = self.net
            net.eval()

            correct = 0
            total = 0

            running_loss = 0.
            i = 0

            for data in dataloader:
                i += 1
                images, labels = data

                outputs = net(images)

                _, predicted = torch.max(outputs.data, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                loss = self.criterion(outputs, labels)
                running_loss += loss.item()

                if not full and i >= self.stat_mbs:
                    break

        net.train()
        return correct / total, total, running_loss / i

    def run_evaluation(self, dataloader, ds_name: str = ''):
        acc, total, loss = self.evaluate_on(dataloader, full=True)

        print(f'{ds_name} stats: acc: {(100 * acc):.2f}%, loss: {loss:.4f}')

        return acc, loss

    def train_batch(self, data):
        inputs, labels = data

        self.net.train()
        self.optimizer.zero_grad()

        outputs = self.net(inputs)
        loss = self.criterion(outputs, labels)

        loss.backward()
        self.optimizer.step()

        return loss.item()

    def train(self, reset_net=True, plot_loss=True):
        if reset_net:
            self.init_net()

        train_losses = []
        valid_losses = []
        epoch_losses = []
        epoch_x = 0
        epoch_xs = []

        last_epoch = 0
        best_state_dict = None
        best_epoch = 0
        best_epoch_loss = 10 ** 9

        try:
            for epoch in range(1, self.num_epochs + 1):
                print(f'EPOCH {epoch}')

                train_dl, valid_dl = (self.train_dl, self.valid_dl) if epoch > 1 \
                    else (self.trunc_train_dl, self.trunc_valid_dl)

                train_loss = 0.0
                last_epoch = epoch

                for i, data in enumerate(train_dl, 0):
                    train_loss += self.train_batch(data)

                    if i % self.stat_period == self.stat_period - 1:
                        epoch_x += 1

                        train_loss = train_loss / self.stat_period
                        train_losses.append(train_loss)

                        acc, total, valid_loss = self.evaluate_on(valid_dl)

                        valid_losses.append(valid_loss)

                        print(f'Epoch {epoch}, batch {i + 1}, train loss: {train_loss:.4f}, '
                              f'valid acc: {100 * acc:.2f}%, valid loss: {valid_loss:.4f}')

                        train_loss = 0.0

                _, epoch_loss = self.run_evaluation(valid_dl, 'VALID')

                epoch_losses.append(epoch_loss)
                epoch_xs.append(epoch_x)

                # early stopping & snapshotting
                if epoch_loss < best_epoch_loss:
                    best_epoch_loss = epoch_loss
                    best_epoch = epoch
                    best_state_dict = deepcopy(self.net.state_dict())
                elif len(epoch_losses) > self.patience:
                    if all((l > best_epoch_loss for l in epoch_losses[-(self.patience + 1):])):
                        print(f'No improvement in last {self.patience + 1} epochs, early stopping.')
                        break

                if self.epoch_train_eval:
                    self.run_evaluation(train_dl, 'TRAIN')

        except Exception as e:
            print(f'Exception thrown: {repr(e)}')
        finally:
            if plot_loss:
                plt.plot(
                    range(len(train_losses)), train_losses, 'r',
                    range(len(valid_losses)), valid_losses, 'b',
                    epoch_xs, epoch_losses, 'g',
                )
                plt.show()

            if best_state_dict and best_epoch != last_epoch:
                print(f'Restoring snapshot from epoch {best_epoch} with valid loss: {best_epoch_loss:.4f}')
                self.net.load_state_dict(best_state_dict)

            acc, loss = self.run_evaluation(self.test_dl, 'TEST')

            snapshot_path = SNAPSHOT_PATH\
                            + f'Snap_a{10000 * acc:.0f}_{datetime.now().strftime("%d_%m_%Y_%H_%M")}'
            torch.save(best_state_dict, snapshot_path)
            #self.run_evaluation(self.train_dl, 'TRAIN')

            return acc, loss

    def load_snapshot(self, path: str):
        self.init_net()
        self.net.load_state_dict(torch.load(path))
