import logging
import sys
import os
import psutil
import json
from keras.models import model_from_json


def get_logger(name, level=logging.INFO, handler=sys.stdout,
               formatter='%(asctime)s - %(name)s - %(levelname)s - %(message)s'):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(formatter)
    stream_handler = logging.StreamHandler(handler)
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def get_memory_usage():
    process = psutil.Process(os.getpid())
    print process.memory_info().rss


def change_single_node_layer_weights(model, layer_name, new_weights, input_dim=None, output_dim=None):
    """
    Take a Keras model, and change the specified layer weight to the new weight, and change the shape of that layer if
    necessary. The layer must be a single node layer (i.e. not a shared layer). Otherwise the shape will be different.
    :param model: The given Keras model.
    :param layer_name:  The name of the layer to be changed.
    :param new_weights: List of weights to assign to the new layer.
    :param input_dim: If the input shape changes, specify it.
    :param output_dim: If the output shape changes, specify it.
    :return: Weight changed new model.
    """
    # Take the old weights, but replace the specific layer weights with our target weight.
    weights = []

    model.get_weights()

    for layer in model.layers:
        if layer.name == layer_name:
            weights += new_weights
        else:
            weights += layer.get_weights()

    # Hacking the input dimension of the model by changing the network config.
    new_config = json.loads(model.to_json())

    for layer_config in new_config["config"]["layers"]:
        if not layer_config["name"] == layer_name:
            continue
        if input_dim is not None:
            layer_config['config']['input_dim'] = input_dim

        if output_dim is not None:
            layer_config['config']['output_dim'] = output_dim

    new_model = model_from_json(json.dumps(new_config))

    # Set new weights to the model.
    new_model.set_weights(weights)

    return new_model
