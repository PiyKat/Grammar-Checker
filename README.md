# Grammar-Checker

The purpose of the code is to correct simple grammatical mistakes using deep learning techniques, more specifically a delayed sequence to sequence model with attention mechanism.

## Dataset

Given a sample of text like this, the next step is to generate input-output pairs to be used during training. 
This is done by:
1. Drawing a sample sentence from the dataset.
2. Setting the input sequence to this sentence after randomly applying certain perturbations.
3. Setting the output sequence to the unperturbed sentence.

where the perturbations applied in step (2) are intended to introduce small grammatical errors which we would like the model to learn to correct.

Thus far, these perturbations are limited to the:
- subtraction of articles (a, an, the)
- replacement of a few common homophones with one of their counterparts (e.g. replacing "their" with "there", "then" with "than")

In this project, each perturbation is applied in 25% of cases where it could potentially be applied. Please feel free to include more gramatical perturbations. 

### Training
Given this augmented dataset, training proceeds in a very similar manner to [TensorFlow's sequence-to-sequence tutorial](https://www.tensorflow.org/tutorials/seq2seq/). 
That is, we train a sequence-to-sequence model using LSTM encoders and decoders with an attention mechanism with SGD optimization.

### Decoding

Although we can use a standard decoder for our task, there is a better approach for solving this task. Given that our grammatical errors only span a fixed subdomainm we can configure our decoder to handle text in such a way that it all tokens
of the sequence should exist in the input sample or a set of corrective tokens. The corrective set is provided in the provided during the training.

This prior is carried out through a modification to the seq2seq model's decoding loop in addition to a post-processing step that resolves out-of-vocabulary (OOV) tokens by a process known as biased decoding:
Biased decoding is used to restrict the decoding such that it only ever chooses tokens from the input sequence or corrective token set, the code applies a binary mask to the model's logits prior to extracting the prediction to be fed into the next time step.

Note that this logic is not used during training, as this would only serve to eliminate potentially useful signal from the model.

**Handling OOV Tokens**

Since the decoding bias described above is applied within the truncated vocabulary used by the model, we will still see the unknown token in its output for any OOV tokens. 


## Code Structure
This project reuses and slightly extends TensorFlow's [`Seq2SeqModel`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/models/rnn/translate/seq2seq_model.py), which itself implements a sequence-to-sequence model with an attention mechanism as described in https://arxiv.org/pdf/1412.7449v3.pdf. 

The primary contributions of this project are:

- `data_reader.py`: an abstract class that defines the interface for classes which are capable of reading a source dataset and producing input-output pairs, where the input is a grammatically incorrect variant of a source sentence and the output is the original sentence.
- `text_corrector_data_readers.py`: contains a few implementations of `DataReader` over the [Cornell Movie-Dialogs Corpus](http://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html).
- `text_corrector_models.py`: contains a version of `Seq2SeqModel` modified such that it implements the logic described in [Biased Decoding](#biased-decoding)
- `correct_text.py`: a collection of helper functions that together allow for the training of a model and the usage of it to decode errant input sequences (at test time).
- `TextCorrector.ipynb`: an IPython notebook which ties together all of the above pieces to proprocess text and train the model.

### Example Usage
This project is compatible with Tensorflow>=1.2 and Python3.5/3.6.

**Preprocess Movie Dialog Data**
```
python preprocessors/preprocess_movie_dialogs.py --raw_data movie_lines.txt \
                                                 --out_file preprocessed_movie_lines.txt
```
This preprocessed file can then be split up however you like to create training, validation, and testing sets.

**Training:**
```
python correct_text.py --train_path /movie_dialog_train.txt \
                       --val_path /movie_dialog_val.txt \
                       --config DefaultMovieDialogConfig \
                       --data_reader_type MovieDialogReader \
                       --model_path /movie_dialog_model
```

**Testing:**
```
python correct_text.py --test_path /movie_dialog_test.txt \
                       --config DefaultMovieDialogConfig \
                       --data_reader_type MovieDialogReader \
                       --model_path /movie_dialog_model \
                       --decode
```

