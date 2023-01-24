import torch
import numpy as np


def add_label_to_x(x, y):
    B, WH = x.shape
    x_clone = x.clone()
    x_clone[:, :10] *= 0.0  # clear out first 10 pixels

    for x_, y_ in zip(x_clone, y):
        x_[y_] = x_.max()  # because we did mean-std to input

    return x_clone, y


def predict_linear(ff_layer, input, device=torch.device("cpu"), debug=False):
    B, XY = input.shape
    with torch.no_grad():
        goodnesses = []
        for label in range(10):
            h_x, h_y = add_label_to_x(
                input, (torch.ones((B,)) * label).type(torch.LongTensor)
            )

            h_x = h_x.to(device)

            goodness = ff_layer.goodness(h_x)

            if debug:
                print(f"test for {label}, goodness = {goodness}")

            goodnesses.append(goodness)

            # plt.imshow(h_x.cpu().reshape(28, 28))
            # plt.title(f"test for {label}, goodness {goodness}")
            # plt.show()
        return np.array(goodnesses).argmax()


def predict_mlp(ffmlp, input, debug=False):
    B, XY = input.shape
    with torch.no_grad():
        goodnesses = []
        for label in range(10):
            h_x, h_y = add_label_to_x(
                input, (torch.ones((B,)) * label).type(torch.LongTensor)
            )

            layer_goodnesses = []

            # h_x_orig = h_x.detach().clone()

            for layer in ffmlp.layers:
                h_x = layer(h_x)

                layer_goodnesses += [h_x.pow(2).mean(1)]

            if debug:
                print(f"test for {label}, goodness={sum(layer_goodnesses)}")
            goodnesses.append(sum(layer_goodnesses))

            # plt.imshow(h_x_orig.cpu().reshape(28, 28))
            # plt.title(f"test for {label}, network goodness {sum(layer_goodnesses)}")
            # plt.show()
        return np.array(goodnesses).argmax()
