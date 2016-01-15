#!/usr/bin/python

import json


def from_json(data_json_str):
    return json.loads(data_json_str)


def dump(data, out):
    json.dump(data, open(out, 'w'))


f = "architecture"
t = "architecture.json"

dump(from_json(json.load(open(f))), t)
