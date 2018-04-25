import unittest
import logging
from GeneticAlgorithm.chromosome import *
import keras as k
from keras.utils import to_categorical
from keras.backend import image_data_format
from keras.datasets import cifar10


class TestChromosome(unittest.TestCase):

    def mutate_model_test(self):
        self.setup_csvlogger()
        for i in range(1, 400):
            print("Chromosome num: %d", i)
            while True:
                print("creating chromosome")
                chromo = Chromosome((32, 32, 3))
                try:
                    for i in range(random.randrange(10)):
                        print("adding conv_node")
                        chromo.add_random_conv_node()
                    for i in range(random.randrange(5)):
                        print("adding random vertex")
                        chromo.add_random_vertex()
                    for i in range(random.randrange(5)):
                        print("adding dense node")
                        chromo.add_random_dense_node()
                    chromo.build().summary()
                    break
                except CantAddNode:
                    del chromo
                    print("Failed to assemble chromosome")
                    continue
            print("Topology:\n")
            print(str(chromo))
            chromo.evaluate(self.unpack_data(10))
            print("{} {} {}".format(chromo.fitness, chromo.accuracy, chromo.parameters))

    @staticmethod
    def unpack_data(num_labels):
        (train_dataset, train_labels), (test_dataset, test_labels) = cifar10.load_data()

        print('train_dataset shape:', train_dataset.shape)
        print(train_dataset.shape[0], 'train samples')
        print(test_dataset.shape[0], 'test samples')

        img_rows = 32
        img_cols = 32

        if image_data_format == 'channels_first':
            train_dataset = train_dataset.reshape(train_dataset.shape[0], 3, img_rows, img_cols)
            test_dataset = test_dataset.reshape(test_dataset.shape[0], 3, img_rows, img_cols)
        else:
            train_dataset = train_dataset.reshape(train_dataset.shape[0], img_rows, img_cols, 3)
            test_dataset = test_dataset.reshape(test_dataset.shape[0], img_rows, img_cols, 3)

        train_labels = to_categorical(train_labels, num_labels)
        test_labels = to_categorical(test_labels, num_labels)

        train_dataset = train_dataset.astype('float32')
        test_dataset = test_dataset.astype('float32')
        train_dataset /= 255
        test_dataset /= 255

        return {"train_dataset": train_dataset, "train_labels": train_labels,
                "valid_dataset": None, "valid_labels": None,
                "test_dataset": test_dataset, "test_labels": test_labels}

    @staticmethod
    def intermittent_logging(chromosome, generation_num):
        with open('trend_graph.csv', 'a', newline='') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=' ',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
            spamwriter.writerow([generation_num, ',',
                                 chromosome.id, ',',
                                 chromosome.age, ',',
                                 chromosome.accuracy, ',',
                                 chromosome.fitness, ',',
                                 chromosome.parameters, ',',
                                 len(chromosome), ',',
                                 chromosome.num_conv_nodes(), ',',
                                 chromosome.num_dense_nodes(), ',',
                                 ])

    @staticmethod
    def setup_csvlogger():
        with open('trend_graph.csv', 'w', newline='') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=' ',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
            spamwriter.writerow(['generation', ',',
                                 'id', ',',
                                 'Age', ',',
                                 'Fitness', ',',
                                 'Accuracy', ',',
                                 'Parameters', ',',
                                 'Num Layers', ',',
                                 'Num Conv Layers', ',',
                                 'Num Dense Layers', ',',
                                 ])


if __name__ == '__main__':

    unittest.main()
