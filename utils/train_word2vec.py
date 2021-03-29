import os
import sys
import logging
import multiprocessing

from gensim.models import Word2Vec

# TODO train Linux manual pages
if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))

    # check and process input arguments
    if len(sys.argv) < 3:
        print("Using: python train_word2vec.py [input_text] [output_word_vector]")
        sys.exit(1)
    input_file, output_file = sys.argv[1:3]


