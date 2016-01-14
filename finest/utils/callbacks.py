from keras.callbacks import Callback


class MonitorLoss(Callback):
    def __init__(self, logger, monitor='val_loss'):
        super(MonitorLoss, self).__init__()
        self.logger = logger
        self.monitor = monitor

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current is None:
            self.logger.info("Cannot record loss, %s is not available" % self.monitor)
        else:
            self.logger.info("Epoch %05d : validation loss is %.4f" % (epoch, current))
