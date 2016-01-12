from keras.callbacks import Callback


class MonitorLoss(Callback):
    def __init__(self, logger, monitor='val_loss'):
        super(MonitorLoss, self).__init__()
        self.logger = logger
        self.monitor = monitor

    def on_epoch_end(self, epoch, logs={}):
        self.logger.info("Epoch %05d : validation loss is %.2f" % (epoch, logs.get(self.monitor)))
