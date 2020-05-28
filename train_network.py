from  constants import *
from network import ParticleTrainer
from utils import plot_embeddings


USE_EMBEDDING = True
PLOT_EMBEDDING_HISTORY = False


def main():
    trainer = ParticleTrainer(embedding=USE_EMBEDDING, store_embeddings=PLOT_EMBEDDING_HISTORY)
    trainer.train()

    if USE_EMBEDDING and PLOT_EMBEDDING_HISTORY:
        embedding_history = trainer.embedding_history
        embedding_map = trainer.embedding_map.numpy()
        for i, embedding in enumerate(embedding_history):
            plot_embeddings(embedding_map, embedding, annotate=False, file=EMBEDDING_HISTORY_PATH + f'/{i}.png')


if __name__ == "__main__":
    main()
