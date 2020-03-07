import numpy as np
import torch
import torch.nn as tnn
import torch.nn.functional as F
import torch.optim as topti
from torchtext import data
from torchtext.vocab import GloVe
from imdb_dataloader import IMDB



# Class for creating the neural network.
class Network(tnn.Module):
    def __init__(self):
        super(Network, self).__init__()

        self.conv1 = tnn.Conv1d(in_channels=50, kernel_size=8, padding=5, out_channels=50)
        self.maxpool1 = tnn.MaxPool1d(4)
        self.conv2 = tnn.Conv1d(in_channels=50, kernel_size=8, padding=5, out_channels=50)
        self.maxpool2 = tnn.MaxPool1d(4)
        self.conv3 = tnn.Conv1d(in_channels=50, kernel_size=8, padding=5, out_channels=50)

        self.rnn = tnn.GRU(
            input_size=50,
            hidden_size=100,
            batch_first=True,
            bidirectional=True,
            num_layers=2,
            dropout=0.3
        )
        self.fc1 = tnn.Linear(4*100, 64)
        self.fc2 = tnn.Linear(64, 1)

    def forward(self, input, length):
        """
        DO NOT MODIFY FUNCTION SIGNATURE
        Create the forward pass through the network.
        """

        # x = input.permute(0, 2, 1)
        # x = self.conv1(x)
        # x = torch.relu(x)
        # x = self.maxpool1(x)
        # x = self.conv2(x)
        # x = torch.relu(x)
        # x = self.maxpool2(x)
        # x = self.conv3(x)
        # x = torch.relu(x)
        # M = tnn.MaxPool1d(kernel_size=x.size(2))
        # x = M(x)
        # rnn_out, h_n = self.rnn(x.permute(0, 2, 1))
        pack = tnn.utils.rnn.pack_padded_sequence(input=input, lengths=length, batch_first=True)
        rnn_out, h_n = self.rnn(pack)
        fw1_hn = h_n[0, :, :]
        fw2_hn = h_n[1, :, :]
        bw1_hn = h_n[2, :, :]
        bw2_hn = h_n[3, :, :]
        h_n = torch.cat([fw1_hn, fw2_hn, bw1_hn, bw2_hn], 1)
        x = self.fc1(h_n)
        x = F.relu(x)
        output = self.fc2(x)
        return output.view(input.size(0))


class PreProcessing():
    def pre(x):
        """Called after tokenization"""
        return x

    def post(batch, vocab):
        """Called after numericalization but prior to vectorization"""
        return batch, vocab

    # text_field = data.Field(lower=True, include_lengths=True, batch_first=True, preprocessing=pre, postprocessing=post)
    text_field = data.Field(lower=True, include_lengths=True, batch_first=True)


def lossFunc():
    """
    Define a loss function appropriate for the above networks that will
    add a sigmoid to the output and calculate the binary cross-entropy.
    """
    return tnn.BCEWithLogitsLoss()

def main():
    # Use a GPU if available, as it should be faster.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device: " + str(device))

    # Load the training dataset, and create a data loader to generate a batch.
    textField = PreProcessing.text_field
    labelField = data.Field(sequential=False)

    train, dev = IMDB.splits(textField, labelField, train="train", validation="dev")

    textField.build_vocab(train, dev, vectors=GloVe(name="6B", dim=50))
    labelField.build_vocab(train, dev)

    trainLoader, testLoader = data.BucketIterator.splits((train, dev), shuffle=True, batch_size=64,
                                                         sort_key=lambda x: len(x.text), sort_within_batch=True)

    net = Network().to(device)
    criterion =lossFunc()
    optimiser = topti.Adam(net.parameters(), lr=0.001)  # Minimise the loss using the Adam algorithm.

    for epoch in range(10):
        running_loss = 0

        for i, batch in enumerate(trainLoader):
            # Get a batch and potentially send it to GPU memory.
            inputs, length, labels = textField.vocab.vectors[batch.text[0]].to(device), batch.text[1].to(
                device), batch.label.type(torch.FloatTensor).to(device)

            labels -= 1

            # PyTorch calculates gradients by accumulating contributions to them (useful for
            # RNNs).  Hence we must manually set them to zero before calculating them.
            optimiser.zero_grad()

            # Forward pass through the network.
            output = net(inputs, length)
            loss = criterion(output, labels)

            # Calculate gradients.
            loss.backward()

            # Minimise the loss according to the gradient.
            optimiser.step()

            running_loss += loss.item()

            if i % 32 == 31:
                print("Epoch: %2d, Batch: %4d, Loss: %.3f" % (epoch + 1, i + 1, running_loss / 32))
                running_loss = 0

    num_correct1 = 0
    num_correct2 = 0

    # Save mode
    torch.save(net.state_dict(), "./model.pth")
    print('loading')
    m = torch.load('./model.pth', map_location='cpu')
    torch.save(m, './model.pth')
    print('saving')
    print("Saved model")

    # Evaluate network on the test dataset.  We aren't calculating gradients, so disable autograd to speed up
    # computations and reduce memory usage.
    with torch.no_grad():
        for batch in testLoader:
            # Get a batch and potentially send it to GPU memory.
            inputs, length, labels = textField.vocab.vectors[batch.text[0]].to(device), batch.text[1].to(
                device), batch.label.type(torch.FloatTensor).to(device)

            labels -= 1

            # Get predictions
            outputs = torch.sigmoid(net(inputs, length))
            predicted = torch.round(outputs)

            num_correct1 += torch.sum(labels == predicted).item()
        for batch in trainLoader:
            # Get a batch and potentially send it to GPU memory.
            inputs, length, labels = textField.vocab.vectors[batch.text[0]].to(device), batch.text[1].to(
                device), batch.label.type(torch.FloatTensor).to(device)

            labels -= 1

            # Get predictions
            outputs = torch.sigmoid(net(inputs, length))
            predicted = torch.round(outputs)

            num_correct2 += torch.sum(labels == predicted).item()

    accuracy_dev = 100 * num_correct1 / len(dev)
    accuracy_train = 100 * num_correct2 / len(train)

    print(f"dev Classification accuracy: {accuracy_dev}")
    print(f"train Classification accuracy: {accuracy_train}")

if __name__ == '__main__':
    main()
