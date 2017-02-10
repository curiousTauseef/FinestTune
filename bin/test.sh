#!/usr/bin/env bash
THEANORC=~/.theanorc python experiment.py --model all --oov random --test ../data/POS-penn/wsj/split1/wsj1.test.original
