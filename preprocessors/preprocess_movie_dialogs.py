"""Preprocesses Cornell Movie Dialog data."""
import nltk
import tensorflow as tf

tf.app.flags.DEFINE_string("raw_data", "", "Raw data path")
tf.app.flags.DEFINE_string("out_file", "", "File to write preprocessed data "
                                           "to.")

FLAGS = tf.app.flags.FLAGS


def main(_):
    with open(FLAGS.raw_data, "r") as raw_data, open(FLAGS.out_file, "w") as out:

        for line in raw_data:
            parts = line.split(" +++$+++ ")
            dialog_line = parts[-1]
            print(dialog_line)

            try:
                s = dialog_line.strip().lower()
                preprocessed_line = " ".join(nltk.word_tokenize(s))
                out.write(preprocessed_line + "\n")

            except:
                continue

if __name__ == "__main__":

    try:
        tf.app.run()

    except:
        pass