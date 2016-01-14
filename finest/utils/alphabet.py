"""
Alphabet maps objects to integer ids. It provides two way mapping from the index to the objects.
"""
import json
import os
import finest.utils.utils as utils


class Alphabet:
    def __init__(self, name, special_instances=(), keep_growing=True):
        self.__name = name

        self.instance2index = {}
        self.instances = []
        self.keep_growing = keep_growing

        # Index 0 is occupied by default, all else following.
        self.default_index = 0
        self.next_index = 1

        for instance in special_instances:
            self.add(instance)

        self.logger = utils.get_logger('Alphabet')

    def add(self, instance):
        if instance not in self.instance2index:
            self.instances.append(instance)
            self.instance2index[instance] = self.next_index
            self.next_index += 1

    def get_index(self, instance):
        try:
            return self.instance2index[instance]
        except KeyError:
            if self.keep_growing:
                index = self.next_index
                self.add(instance)
                return index
            else:
                return self.default_index

    def get_instance(self, index):
        return self.instances[index]

    def size(self):
        return len(self.instances) + 1

    def iteritems(self):
        return self.instance2index.iteritems()

    def stop_grow(self):
        self.keep_growing = False

    def restart_grow(self):
        self.keep_growing = True

    def to_json(self):
        return json.dumps({'instance2index': self.instance2index, 'instances': self.instances})

    def from_json(self, json_obj):
        data = json.load(json_obj)
        self.instances = data['instances']
        self.instance2index = data['instance2index']

    def save(self, output_directory):
        """
        Save both the model architecture and the weights to the given directory.
        :param output_directory: Directory to save model and weights.
        :return:
        """
        try:
            json.dump(self.to_json(), open(os.path.join(output_directory, self.__name), 'w'))
        except Exception as e:
            self.logger.warn("Model structure is not saved: " % repr(e))

    def load(self, input_directory):
        """
        Load model architecture and weights from the give directory. This allow we use old models even the structure
        changes.
        :param input_directory: Directory to save model and weights
        :return:
        """
        self.from_json(open(os.path.join(input_directory, self.__name)).read())
