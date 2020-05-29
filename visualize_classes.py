from matplotlib import pyplot as plt

from constants import *
from input import load_file


SAMPLES_PER_CLASS = 30

def get_samples(xs, ys, samples_per_class=SAMPLES_PER_CLASS):
    samples = [[] for _ in range(NUM_CLASSES)]

    for i, (x, y) in enumerate(zip(xs, ys)):
        if len(samples[y]) < SAMPLES_PER_CLASS:
            samples[y].append((i, x.numpy()))

        if all(len(cl) >= SAMPLES_PER_CLASS for cl in samples):
            break

    return samples


def plot_class(samples, class_idx):
    for i, s in samples:
        plt.xlim(-1, 10)
        plt.ylim(-1, 10)

        xs = s[..., 0]
        ys = s[..., 1]
        us = xs[1:] - xs[:-1]
        vs = ys[1:] - ys[:-1]
        xs = xs[:-1]
        ys = ys[:-1]

        plt.title(f"Index {i}, class {class_idx}")
        plt.quiver(xs, ys, us, vs, scale_units='xy', angles='xy', scale=1, color=[(1, 1, 0, 1), (1., 165./255., 0., 1.), (1, 0, 0, 1), (0, 0, 0, 1)])
        plt.savefig(f"./report/img/class_{class_idx}/{i}.png")
        plt.show()
        pass


def main():
    test_x = load_file(TEST_X_PATH, is_labels=False)
    test_y = load_file(TEST_Y_PATH, is_labels=True)

    samples = get_samples(test_x, test_y)

    for i, cl in enumerate(samples):
        plot_class(cl, class_idx=i)


if __name__ == "__main__":
    main()