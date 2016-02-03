#!/usr/bin/env bash
THEANORC=~/.theanorc experiment.py --model vanilla --oov random --test ../data/POS-penn/wsj/split1/wsj1.test.original
