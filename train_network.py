from network import ParticleTrainer


def main():
    trainer = ParticleTrainer(embedding=False)
    trainer.train()


if __name__ == "__main__":
    main()