from keras.models import Sequential
from keras.layers import Flatten


class Chromosome(object):

    def __init__(self):
        self.genes = []

    def add_gene(self, gene):
        self.genes.append(gene)

    def remove_gene(self, index=None):
        if index == None:
            del self.genes[len(self.genes)-1]
        else:
            self.genes.remove(index)

    def create_model(self):
        model = Sequential()

        for x in range(0, len(self.genes)-1):
            model = self.genes[x].compile(model)

        model = self.genes[len(self.genes)-1].compile(model, output=True)
        model.summary()
        return model

    def __len__(self):
        return len(self.genes)
