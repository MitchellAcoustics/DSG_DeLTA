import sys, os, argparse

sys.path.append(os.path.split(os.path.dirname(os.path.realpath(__file__)))[0])
from framework.data_generator import *
from framework.processing import *
from framework.models_pytorch import *


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-lr_init', type=float, required=True)
    parser.add_argument('-batch_size', type=int, required=True)
    parser.add_argument('-epochs', type=int, required=True)
    args = parser.parse_args()

    lr_init = args.lr_init
    batch_size = args.batch_size
    epochs = args.epochs

    basic_name = 'sys_' + str(lr_init).replace('-', '')

    suffix, system_name = define_system_name(basic_name=basic_name,
                                             batch_size=batch_size)
    system_path = os.path.join(os.getcwd(), system_name)

    models_dir = system_path

    Model = TinyCNN

    event_class = len(config.event_labels)
    model = Model(event_class=event_class, batchnormal=True)
    print(model)

    if config.cuda:
        model.cuda()

    generator = DataGenerator(batch_size=batch_size, normalization=True)

    training_testing(generator, model, config.cuda, models_dir, epochs, batch_size, lr_init = config.lr_init)



if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except (ValueError, IOError) as e:
        sys.exit(e)















