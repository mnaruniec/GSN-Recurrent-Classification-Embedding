from constants import *
from input import load_file,get_dataloaders


def main():
    train, valid, test = get_dataloaders(truncation_p=0.2)
    pass


if __name__ == "__main__":
    main()