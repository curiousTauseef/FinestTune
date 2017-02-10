#!/usr/bin/env bash
#THEANORC=~/.theanorc experiment.py --model vanilla --oov random --train ../data/POS-penn/wsj/split1/wsj1.train.original\
# --test ../data/POS-penn/wsj/split1/wsj1.test.original --dev ../data/POS-penn/wsj/split1/wsj1.dev.original --overwrite

THEANORC=~/.theanorc python experiment.py --overwrite
