from typing import Optional

from matplotlib import pyplot as plt


def plot_embeddings(map, matrix, file: Optional[str] = None, annotate=True):
    xs = []
    ys = []
    cs = []

    for i in range(10):
        for j in range(10):
            numeric = map[i, j]
            xs.append(matrix[numeric, 0])
            ys.append(matrix[numeric, 1])
            cs.append((i / 9., j / 9., 0.))
            if annotate:
                plt.annotate((i, j), (matrix[numeric, 0], matrix[numeric, 1]))

    plt.scatter(xs, ys, c=cs)

    if file:
        plt.savefig(file)
    plt.show()
