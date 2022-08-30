from Model import Model
from Dataset_Generator import Dataset_Generator

def main():
    # Generate minibatch
    #dataset_generator = Dataset_Generator('data')
    #dataset_generator.generate_minibatch(0.2, 'minibatch_data')

    supervised_model = Model()
    supervised_model.load_model('vgg19')
    supervised_model.load_dataset('minibatch_data', 50)
    supervised_model.transfer_learning_init()
    supervised_model.train_model(0.000005, 10)

    supervised_model.plot_accuracy()
    supervised_model.plot_loss()
    #supervised_model.load_model('models/vgg19_full_benchmark.pt', True)

    supervised_model.test_model()

if __name__ == "__main__":
    main()