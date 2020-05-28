from os.path import basename

from constants import *
from network import ParticleTrainer
from utils import plot_embeddings

SNAPSHOT = BEST_EMBEDDED_SNAPSHOT_PATH


def main():
    trainer = ParticleTrainer(embedding=True)
    trainer.load_snapshot(BEST_EMBEDDED_SNAPSHOT_PATH)

    embedding_map = trainer.embedding_map.numpy()
    embedding_matrix = trainer.net.embedding.weight
    embedding_matrix = embedding_matrix.cpu().detach().numpy()

    plot_embeddings(embedding_map, embedding_matrix, file=FINAL_EMBEDDING_PATH + f'/{basename(SNAPSHOT)}.png')


if __name__ == "__main__":
    main()
