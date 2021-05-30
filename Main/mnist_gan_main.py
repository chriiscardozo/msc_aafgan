from Models.Experiment import Experiment

def main():
    e = Experiment('mnist_gan.json')
    e.run()
    e.finish()

if __name__ == '__main__':
    main()