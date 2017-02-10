"""
Alphabet maps objects to integer ids. It provides two way mapping from the index to the objects.
"""
import json
import os
import finest.utils.utils as utils


class Alphabet:
    default_index = 0

    def __init__(self, name, special_instances=(), keep_growing=True):
        self.__name = name

        self.__instance2index = {}
        self.__instances = []
        self.__keep_growing = keep_growing

        # Index 0 is occupied by default, all else following.
        self.__next_index = 1

        for instance in special_instances:
            self.add(instance)

        self.logger = utils.get_logger('Alphabet')

    def add(self, instance):
        """
        Add an instance to the alphabet, the index will be automatically increased.
        :param instance: An instance that can be put into a dict.
        :return:
        """
        if instance not in self.__instance2index:
            self.__instances.append(instance)
            self.__instance2index[instance] = self.__next_index
            self.__next_index += 1

    def get_index(self, instance):
        """
        Get the index of the instance. If the instance is not there: we will (1) return a default index if auto grow is
        not enabled and (2) add this instance into the alphabet if auto grow is enabled.
        :param instance: The instance you are seeking.
        :return:
        """
        try:
            return self.__instance2index[instance]
        except KeyError:
            if self.__keep_growing:
                index = self.__next_index
                self.add(instance)
                return index
            else:
                return Alphabet.default_index

    def get_instance(self, index):
        if index == 0:
            # First index is occupied by the wildcard element.
            return None
        return self.__instances[index - 1]

    def has_instance(self, instance):
        return instance in self.__instance2index

    def size(self):
        return len(self.__instances) + 1

    def iteritems(self):
        return self.__instance2index.iteritems()

    def enumerate_items(self, start=1):
        if start < 1 or start >= self.size():
            raise IndexError("Enumerate is allowed between [1 : size of the alphabet)")
        return zip(range(start, len(self.__instances) + 1), self.__instances[start - 1:])

    def stop_auto_grow(self):
        self.__keep_growing = False

    def restart_auto_grow(self):
        self.__keep_growing = True

    def get_content(self):
        """
        Get the content that is suitable for JSON dumping.
        :return:
        """
        return {'instance2index': self.__instance2index, 'instances': self.__instances, 'next_index': self.__next_index}

    def from_json(self, data):
        """
        Populate the alphabet with JSON data.
        :param data:
        :return:
        """
        self.__instances = data["instances"]
        self.__instance2index = data["instance2index"]
        self.__next_index = data["next_index"]

    def get_copy(self):
        """
        Make a copy of the alphabet and return it.
        :return: The copy of the alphabet.
        """
        the_copy = Alphabet(self.__name)
        the_copy.__next_index = self.__next_index
        the_copy.__keep_growing = self.__keep_growing
        the_copy.__instances = self.__instances[:]
        the_copy.__instance2index = self.__instance2index.copy()
        return the_copy

    def save(self, output_directory, name=None):
        """
        Save both alhpabet records to the given directory.
        :param output_directory: Directory to save model and weights.
        :param name: The alphabet saving name, optional.
        :return:
        """
        saving_name = name if name else self.__name
        try:
            json.dump(self.get_content(), open(os.path.join(output_directory, saving_name + ".json"), 'w'))
        except Exception as e:
            self.logger.warn("Alphabet is not saved: " % repr(e))

    def load(self, input_directory, name=None):
        """
        Load model architecture and weights from the give directory. This allow we use old models even the structure
        changes.
        :param input_directory: Directory to save model and weights
        :return:
        """
        loading_name = name if name else self.__name
        self.from_json(json.load(open(os.path.join(input_directory, loading_name + ".json"))))
