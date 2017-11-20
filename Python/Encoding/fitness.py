import chromosome
import keras


class Fitness:
    def __init__(self, optimal_fitness=False, chromosome=None, train_dataset=None, train_labels=None, valid_dataset=None,
                 valid_labels=None, test_dataset=None, test_labels=None):
        if optimal_fitness:
            self.accuracy = 0.90
        else:

            model = chromosome.create_model()
            model.compile(loss='categorical_crossentropy',
                          optimizer='sgd',
                          metrics=['accuracy'])
            model.fit(train_dataset, train_labels, epochs=5, batch_size=100, validation_data=(valid_dataset, valid_labels))
            loss_and_metrics = model.evaluate(test_dataset, test_labels, batch_size=100)
            self.accuracy = loss_and_metrics[1]

    def __str__(self):
        return "{} Accuracy\n".format(
            self.accuracy
        )

    def __gt__(self, other):
        return self.accuracy > (other.accuracy-0.03)
