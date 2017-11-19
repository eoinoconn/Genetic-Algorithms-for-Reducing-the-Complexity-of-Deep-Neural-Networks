from keras.models import Sequential


class Chromosome(object):

    def __init__(self):
        self.genes = []
        self.model = Sequential()

    def add_gene(self, gene):
        self.genes.append(gene)

    def remove_gene(self, index=-1):
        if index == -1:
            self.genes.remove(len(self.genes))
        else:
            self.genes.remove(index)


    def create_model(self):
        for gene in self.genes:
            self.model = gene.compile(self.model)
        return self.model
