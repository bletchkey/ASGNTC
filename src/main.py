import sys

from gol_adv_sys.Training import Training

def main():

    train = Training()
    train.run()

    return 0


if __name__ == '__main__':
    RETCODE = main()
    sys.exit(RETCODE)

