from os.path import basename

import numpy as np

from constants import *
from network import ParticleTrainer
from utils import plot_embeddings

SNAPSHOT = BEST_EMBEDDED_SNAPSHOT_PATH
SAVE_FILE = True


def plot_ground_truth():
    map = np.array(list(range(100))).reshape([10, 10])
    matrix = np.array([[i, j] for i in range(10) for j in range(10)])

    plot_embeddings(
        map,
        matrix,
        file=FINAL_EMBEDDING_PATH + '/ground_truth.png'
    )


def main():
    trainer = ParticleTrainer(embedding=True)
    trainer.load_snapshot(BEST_EMBEDDED_SNAPSHOT_PATH)

    embedding_map = trainer.embedding_map.numpy()
    embedding_matrix = trainer.net.embedding.weight
    embedding_matrix = embedding_matrix.cpu().detach().numpy()

    plot_ground_truth()

    plot_embeddings(
        embedding_map, embedding_matrix,
        file=FINAL_EMBEDDING_PATH + f'/{basename(SNAPSHOT)}.png' if SAVE_FILE else None,
    )


if __name__ == "__main__":
    main()
