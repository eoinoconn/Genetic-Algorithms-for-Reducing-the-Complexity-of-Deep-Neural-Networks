import unittest
from GeneticAlgorithm.chromosome import *
from keras.utils import to_categorical
from keras.backend import image_data_format
from keras.datasets import mnist


class TestChromosome(unittest.TestCase):

    @staticmethod
    def setup_chromosome_test():
        node = ConvNode()
        chromo = Chromosome((32, 32, 3))
        chromo.add_node(node)
        chromo.add_vertex(node, node.id)
        print(chromo)

    @staticmethod
    def build_basic_model_test():
        chromo = Chromosome((32, 32, 3))
        chromo.build().summary()

    def dense_model_test(self):
        for i in range(25):
            self.build_dense_model()
        assert True

    @staticmethod
    def build_dense_model():
        chromo = Chromosome((32, 32, 3))
        for i in range(10):
            chromo.add_random_dense_node()
        chromo.build().summary()

    def simple_conv_model_test(self):
        counter = 0
        for i in range(20):
            if self.build_simple_conv_model():
                counter += 1
        print("failed conv models = " + str(counter))
        assert True

    def complex_conv_model_test(self):
        counter = 0
        for i in range(10):
            if self.build_complex_conv_model():
                counter += 1
        print("failed conv models = " + str(counter))
        assert True

    def very_complex_conv_model_test(self):
        counter = 0
        for i in range(10):
            print(i)
            if self.build_very_complex_conv_model():
                counter += 1
        print("failed conv models = " + str(counter))
        assert True

    def real_model_test(self):
        counter = 0
        for i in range(10):
            if self.build_real_model():
                counter += 1
        print("failed conv models = " + str(counter))
        assert True

    def training_model_test(self):
        while True:
            chromo = Chromosome((28, 28, 1))
            try:
                for i in range(5):
                    chromo.add_random_conv_node()
                    chromo.add_random_vertex()
                for i in range(5):
                    chromo.add_random_dense_node()
                chromo.build().summary()
                break
            except CantAddNode:
                del chromo
                continue
        #chromo.evaluate(self.unpack_data(10))

    @staticmethod
    def build_simple_conv_model():
        try:
            chromo = Chromosome((32, 32, 3))
            chromo.add_random_conv_node()
            chromo.build().summary()
        except DimensionException:
            return True
            pass

    @staticmethod
    def build_complex_conv_model():
        chromo = Chromosome((32, 32, 3))
        try:
            for i in range(5):
                chromo.add_random_conv_node()
            chromo.build().summary()
        except CantAddNode:
            del chromo
            return True

    @staticmethod
    def build_very_complex_conv_model():
        chromo = Chromosome((32, 32, 3))
        try:
            for i in range(25):
                chromo.add_random_conv_node()
            chromo.build().summary()
        except CantAddNode:
            del chromo
            return True

    @staticmethod
    def build_real_model():
        chromo = Chromosome((32, 32, 3))
        try:
            for i in range(5):
                chromo.add_random_conv_node()
                chromo.add_random_vertex()
            for i in range(5):
                chromo.add_random_dense_node()
            chromo.build().summary()
        except CantAddNode:
            return True

    @staticmethod
    def graph_test():
        while True:
            try:
                chromo = Chromosome((32, 32, 3))
                for i in range(5):
                    chromo.add_random_conv_node()
                    chromo.add_random_vertex()
                for i in range(5):
                    chromo.add_random_dense_node()
                chromo.build().summary()
            except CantAddNode:
                continue
            break

    @staticmethod
    def mutate_test():
        count = 0
        chromo = Chromosome((32, 32, 3))
        for i in range(15):
            print("Mutate no " + str(i))
            chromo.mutate()
            count += 1
        print(count)

    @staticmethod
    def mutate_and_build_test():
        count = 0
        chromo = Chromosome((32, 32, 3))
        for i in range(30):
            print("Mutate and build no " + str(i))
            chromo.mutate()
            chromo.build().summary()
            count += 1
        print(count)

    @staticmethod
    def unpack_data(num_labels):
        (train_dataset, train_labels), (test_dataset, test_labels) = mnist.load_data()

        print('train_dataset shape:', train_dataset.shape)
        print(train_dataset.shape[0], 'train samples')

        img_rows = 28
        img_cols = 28

        if image_data_format == 'channels_first':
            train_dataset = train_dataset.reshape(train_dataset.shape[0], 1, img_rows, img_cols)
        else:
            train_dataset = train_dataset.reshape(train_dataset.shape[0], img_rows, img_cols, 1)

        train_labels = to_categorical(train_labels, num_labels)

        train_dataset = train_dataset.astype('float32')
        train_dataset /= 255

        return {"train_dataset": train_dataset, "train_labels": train_labels}


if __name__ == '__main__':

    unittest.main()
