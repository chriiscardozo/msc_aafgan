from Models.Experiment import Experiment
import sys

def main():
    experiment_key = "current"
    if len(sys.argv) > 1:
        experiment_key = sys.argv[1]
    e = Experiment('celeba_dcgan.json', experiment_key, dataset='CelebA')
    e.run()
    e.finish()

if __name__ == '__main__':
    main()
