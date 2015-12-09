class Alphabet:
    def __init__(self, special_instances=(), keep_growing=True):
        self.next_index = 0
        self.instance2index = {}
        self.instances = []
        self.keep_growing = keep_growing
        for instance in special_instances:
            self.add(instance)

    def add(self, instance):
        if instance not in self.instance2index:
            self.instances.append(instance)
            self.instance2index[instance] = self.next_index
            self.next_index += 1

    def get_index(self, instance):
        try:
            return self.instance2index[instance]
        except KeyError:
            index = self.next_index
            self.add(instance)
            return index

    def get_instance(self, index):
        return self.instances[index]

    def size(self):
        return len(self.instances)

    def iteritems(self):
        return self.instance2index.iteritems()
