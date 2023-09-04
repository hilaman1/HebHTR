import numpy as np
import os
from io import StringIO
import sys
from typing import List, Tuple
import tensorflow.compat.v1 as tf
from word_beam_search import WordBeamSearch

tf.disable_v2_behavior()
# disable eager mode
# tf.compat.v1.disable_eager_execution()

'''
Handwritten text recognition model written by Harald Scheidl:
https://github.com/githubharald/SimpleHTR
'''


class DecoderType:
    BestPath = 0
    WordBeamSearch = 1


class Model:
    # model constants
    batchSize = 50
    imgSize = (128, 32) #desired image size
    maxTextLen = 32

    def __init__(self, charList, decoderType=DecoderType.BestPath,
                 mustRestore = False, dump=False):
        self.rnnOut3d = None
        self.gtTexts = None
        self.ctcIn3dTBC = None
        self.wbs_input = None
        self.decoder = None
        self.dump = dump
        self.charList = charList
        self.decoderType = decoderType
        self.mustRestore = mustRestore
        self.snapID = 0

        # Whether to use normalization over a batch or a population
        self.is_train = tf.placeholder(tf.bool, name='is_train')
        # input image batch
        self.inputImgs = tf.placeholder(tf.float32, shape=(
            None, Model.imgSize[0], Model.imgSize[1]))

        # setup CNN, RNN and CTC
        self.setupCNN()
        self.setupRNN()
        self.setupCTC()

        # setup optimizer to train NN
        self.batchesTrained = 0
        self.learningRate = tf.placeholder(tf.float32, shape=[])
        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(self.update_ops):
            self.optimizer = tf.train.RMSPropOptimizer(
                self.learningRate).minimize(self.loss)

        # initialize TF
        (self.sess, self.saver) = self.setupTF()

    def setupCNN(self):
        cnnIn4d = tf.expand_dims(input=self.inputImgs, axis=3)

        # list of parameters for the layers
        kernelVals = [5, 5, 3, 3, 3]
        featureVals = [1, 32, 64, 128, 128, 256]
        strideVals = poolVals = [(2, 2), (2, 2), (1, 2), (1, 2), (1, 2)]
        numLayers = len(strideVals)

        # create layers
        pool = cnnIn4d  # input to first CNN layer
        for i in range(numLayers):
            kernel = tf.Variable(tf.truncated_normal(
                [kernelVals[i], kernelVals[i], featureVals[i],
                 featureVals[i + 1]], stddev=0.1))
            conv = tf.nn.conv2d(input=pool, filters=kernel, padding='SAME',
                                strides=(1, 1, 1, 1))
            conv_norm = tf.compat.v1.layers.batch_normalization(conv,
                                                      training=self.is_train)
            relu = tf.nn.relu(conv_norm)
            pool = tf.nn.max_pool(relu, (1, poolVals[i][0], poolVals[i][1], 1),
                                  (1, strideVals[i][0], strideVals[i][1], 1),
                                  'VALID')

        self.cnnOut4d = pool

    def setupRNN(self):
        rnnIn3d = tf.squeeze(self.cnnOut4d, axis=[2])

        # basic cells which is used to build RNN
        numHidden = 256
        cells = [
            tf.nn.rnn_cell.LSTMCell(num_units=numHidden, state_is_tuple=True)
            for _ in range(2)]  # 2 layers

        # stack basic cells
        stacked = tf.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=True)

        # bidirectional RNN
        # BxTxF -> BxTx2H
        ((fw, bw), _) = tf.nn.bidirectional_dynamic_rnn(cell_fw=stacked,
                                                        cell_bw=stacked,
                                                        inputs=rnnIn3d,
                                                        dtype=rnnIn3d.dtype)

        # BxTxH + BxTxH -> BxTx2H -> BxTx1X2H
        concat = tf.expand_dims(tf.concat([fw, bw], 2), 2)

        # project output to chars (including blank): BxTx1x2H -> BxTx1xC -> BxTxC
        kernel = tf.Variable(
            tf.truncated_normal([1, 1, numHidden * 2, len(self.charList) + 1],
                                stddev=0.1))
        self.rnnOut3d = tf.squeeze(
            tf.nn.atrous_conv2d(value=concat, filters=kernel, rate=1,
                                padding='SAME'), axis=[2])

    def setupCTC(self):
        # BxTxC -> TxBxC
        self.ctcIn3dTBC = tf.transpose(a=self.rnnOut3d, perm=[1, 0, 2])
        # ground truth text as sparse tensor
        self.gtTexts = tf.SparseTensor(
            tf.placeholder(tf.int64, shape=[None, 2]),
            tf.placeholder(tf.int32, [None]), tf.placeholder(tf.int64, [2]))

        # calc loss for batch
        self.seqLen = tf.placeholder(tf.int32, [None])
        self.loss = tf.reduce_mean(
            tf.nn.ctc_loss(labels=self.gtTexts, inputs=self.ctcIn3dTBC,
                           sequence_length=self.seqLen,
                           ctc_merge_repeated=True))

        # calc loss for each element to compute label probability
        self.savedCtcInput = tf.placeholder(tf.float32,
                                            shape=[Model.maxTextLen, None,
                                                   len(self.charList) + 1])
        self.lossPerElement = tf.nn.ctc_loss(labels=self.gtTexts,
                                             inputs=self.savedCtcInput,
                                             sequence_length=self.seqLen,
                                             ctc_merge_repeated=True)

        # decoder: either best path decoding or beam search decoding
        if self.decoderType == DecoderType.BestPath:
            self.decoder = tf.nn.ctc_greedy_decoder(inputs=self.ctcIn3dTBC,
                                                    sequence_length=self.seqLen)
        elif self.decoderType == DecoderType.WordBeamSearch:
            # import compiled word beam search operation (see https://github.com/githubharald/CTCWordBeamSearch)
            # add to current cwd HebHTR folder
            current_dir = os.getcwd()
            globals()[current_dir] = current_dir
            # word_beam_search_module = tf.load_op_library(fr'{current_dir}/TFWordBeamSearch.so')
            # word_beam_search_module = tf.load_op_library('./TFWordBeamSearch.so')
            # word_beam_search.cp310-win_amd64.pyd
            # prepare information about language (dictionary, characters in dataset, characters forming words)
            chars = str().join(self.charList)

            with open('model/wordCharList.txt', "rb") as f:
                byte = f.read(1)
                if byte != "":
                    byte = f.read()
                    myString = byte.decode("utf8")
                    wordChars = myString.splitlines()[0]
            corpus = open('data/corpus.txt', encoding="utf8").read()
            # with open('data/corpus_old.txt', 'r', encoding='utf-8') as file:
            #     corpus = file.read()

            # decode using the "Words" mode of word beam search.
                #as for arg2 of WordBeamSearch:
                    # "Words": only use dictionary, no scoring: O(1)
                    # "NGrams": use dictionary and score beams with LM: O(log(W))
                    # "NGramsForecast": forecast (possible) next words and apply LM to these words: O(W*log(W))
                    # "NGramsForecastAndSample": restrict number of (possible) next words to at most 20 words: O(W)
            self.decoder = WordBeamSearch(50, 'Words', 0.00,
                                          corpus.encode('utf8'), chars.encode('utf8'),
                                          wordChars.encode('utf8'))
            # the input to the decoder must have softmax already applied
            self.wbs_input = tf.nn.softmax(self.ctcIn3dTBC, axis=2)

    def setupTF(self):
        # TF session
        sess = tf.compat.v1.Session()
        # sess.run(tf.compat.v1.global_variables_initializer())
        # sess = tf.Session()  # TF session

        saver = tf.train.Saver(max_to_keep=1)  # saver saves model to file
        modelDir = 'model/'
        latestSnapshot = tf.train.latest_checkpoint(
            modelDir)  # is there a saved model?

        # if model must be restored (for inference), there must be a snapshot
        if self.mustRestore and not latestSnapshot:
            raise Exception('No saved model found in: ' + modelDir)

        # load saved model if available
        if latestSnapshot:
            saver.restore(sess, latestSnapshot)
        else:
            sess.run(tf.global_variables_initializer())

        return (sess, saver)

    def toSparse(self, texts):
        indices = []
        values = []
        shape = [len(texts), 0]  # last entry must be max(labelList[i])

        # go over all texts
        for (batchElement, text) in enumerate(texts):
            # convert to string of label (i.e. class-ids)
            labelStr = [self.charList.index(c) for c in text]
            # sparse tensor must have size of max. label-string
            if len(labelStr) > shape[1]:
                shape[1] = len(labelStr)
            # put each label into sparse tensor
            for (i, label) in enumerate(labelStr):
                indices.append([batchElement, i])
                values.append(label)

        return (indices, values, shape)

    @staticmethod
    def dump_nn_output(rnn_output: np.ndarray) -> None:
        """Dump the output of the NN to CSV file(s)."""
        current_dir = os.getcwd()
        dump_dir = fr'{current_dir}/dump/'
        if not os.path.isdir(dump_dir):
            os.mkdir(dump_dir)

        # iterate over all batch elements and create a CSV file for each one
        max_t, max_b, max_c = rnn_output.shape
        for b in range(max_b):
            csv = ''
            for t in range(max_t):
                for c in range(max_c):
                    csv += str(rnn_output[t, b, c]) + ';'
                csv += '\n'
            fn = dump_dir + 'rnnOutput_' + str(b) + '.csv'
            print('Write dump of NN to file: ' + fn)
            with open(fn, 'w') as f:
                f.write(csv)

    def decoderOutputToText(self, ctcOutput, batchSize):
        """Extract texts from output of CTC decoder."""
        # contains string of labels for each batch element
        encodedLabelStrs = [[] for i in range(batchSize)]

        # word beam search: label strings terminated by blank
        if self.decoderType == DecoderType.WordBeamSearch:
            blank = len(self.charList)
            for b in range(batchSize):
                for label in ctcOutput[b]:
                    if label == blank:
                        break
                    encodedLabelStrs[b].append(label)

        # TF decoders: label strings are contained in sparse tensor
        else:
            # ctc returns tuple, first element is SparseTensor
            decoded = ctcOutput[0][0]

            # go over all indices and save mapping: batch -> values
            idxDict = {b: [] for b in range(batchSize)}
            for (idx, idx2d) in enumerate(decoded.indices):
                label = decoded.values[idx]
                print("current label is ", label )
                batchElement = idx2d[0]  # index according to [b,t]
                encodedLabelStrs[batchElement].append(label)

        # map labels to chars for all batch elements
        return [str().join([self.charList[c] for c in labelStr]) for labelStr in
                encodedLabelStrs]

    def inferBatch(self, batch, calcProbability=True, probabilityOfGT=False):
        # decode, optionally save RNN output
        numBatchElements = len(batch.imgs)

        # put tensors to be evaluated into list
        evalList = []

        if self.decoderType == DecoderType.WordBeamSearch:
            evalList.append(self.wbs_input)
        else:
            evalList.append(self.decoder)

        if self.dump_nn_output or calcProbability:
            evalList.append(self.ctcIn3dTBC)
        # evalRnnOutput = self.dump or calcProbability
        # evalList = [self.decoder] + ([self.ctcIn3dTBC] if evalRnnOutput else [])
        feedDict = {self.inputImgs: batch.imgs,
                    self.seqLen: [Model.maxTextLen] * numBatchElements,
                    self.is_train: False}
        evalRes = self.sess.run(evalList, feedDict)
        #decoded = evalRes[0]

        # TF decoders: decoding already done in TF graph
        if self.decoderType != DecoderType.WordBeamSearch:
            decoded = evalRes[0]
        # word beam search decoder: decoding is done in C++ function compute()
        else:
            decoded = self.decoder.compute(evalRes[0])

        texts = self.decoderOutputToText(decoded, numBatchElements)
        #texts_second_best =
        #texts_third_best =
        print(texts)
        # feed RNN output and recognized text into CTC loss to compute labeling probability
        probs = None
        if calcProbability: #TODO- from the paper- the probability of seeing the beam-labeling at the current timestep is calculated
            sparse = self.toSparse(
                batch.gtTexts) if probabilityOfGT else self.toSparse(texts)
            ctcInput = evalRes[1]
            evalList = self.lossPerElement
            feedDict = {self.savedCtcInput: ctcInput, self.gtTexts: sparse,
                        self.seqLen: [Model.maxTextLen] * numBatchElements,
                        self.is_train: False}
            lossVals = self.sess.run(evalList, feedDict)
            probs = np.exp(-lossVals)
            print(probs)

            # dump the output of the NN to CSV file(s)
            if self.dump:
                self.dump_nn_output(evalRes[1])

            return (texts, probs)

    def save(self) -> None:
        """Save model to file."""
        self.snap_ID += 1
        self.saver.save(self.sess, '../model/snapshot', global_step=self.snap_ID)