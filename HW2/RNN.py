import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense, TimeDistributed, LSTM, Dropout
import numpy as np

# Data paths
LANGUAGE = ['English', ' Japanese', 'Bulgarian']
TRAIN_SET = ['data/ptb.2-21.txt', 'data/jv.train.txt', 'data/btb.train.txt']
TRAIN_LABEL = ['data/ptb.2-21.tgs', 'data/jv.train.tgs', 'data/btb.train.tgs'] 
TEST_SET = ['data/ptb.22.txt', 'data/jv.test.txt', 'data/btb.test.txt']
TEST_LABEL = ['data/ptb.22.tgs', 'data/jv.test.tgs', 'data/btb.test.tgs'] 

for language in range(len(LANGUAGE)):
    print('training', LANGUAGE[language], 'model...')
    # Loading data
    train_data = []
    train_label = []
    with open(TRAIN_SET[language], 'r', encoding='utf-8') as train_sentences:
        for line in train_sentences:
            line = line.strip()
            if line:
                train_data.append(line)
    
    with open(TRAIN_LABEL[language], 'r', encoding='utf-8') as train_labels:
        for line in train_labels:
            line = line.strip()
            if line:
                train_label.append(line)
    #print(train_data[:10], train_label[:10])

    # Tokenization
    print('tokenizing', LANGUAGE[language], 'sentences...')
    word_tokenizer = Tokenizer()
    word_tokenizer.fit_on_texts(train_data)
    X_data = word_tokenizer.texts_to_sequences(train_data)

    tag_tokenizer = Tokenizer()
    tag_tokenizer.fit_on_texts(train_label)
    y_data = tag_tokenizer.texts_to_sequences(train_label)
    
    # pad sequences to the same length
    # Determine the maximum sequence length from both features and labels
    max_seq_length = max(max([len(seq) for seq in X_data]), max([len(seq) for seq in y_data]))

    # Now pad both the features and the labels to the maximum sequence length
    X_data = np.array(pad_sequences(X_data, maxlen=max_seq_length, padding='post'))
    y_data = np.array(pad_sequences(y_data, maxlen=max_seq_length, padding='post'))

    
    #print(X_data[:5], y_data[:5])
    # model definition
    model = Sequential()
    model.add(Embedding(input_dim=len(word_tokenizer.word_index) + 1, output_dim=128))
    model.add(Dropout(0.1))
    model.add(LSTM(units=256, return_sequences=True))
    model.add(TimeDistributed(Dense(len(tag_tokenizer.word_index) + 1, activation='softmax')))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # training model
    print('model training...')
    model.fit(X_data, y_data, batch_size=128, epochs=10)

    # model evaluation
    print('evaluating', LANGUAGE[language], 'model...')
    # Loading data
    test_data = []
    test_label = []
    with open(TEST_SET[language], 'r', encoding='utf-8') as test_sentences:
        for line in test_sentences:
            line = line.strip()
            if line:
                test_data.append(line)
    
    with open(TEST_LABEL[language], 'r', encoding='utf-8') as test_labels:
        for line in test_labels:
            line = line.strip()
            if line:
                test_label.append(line)
    #print(train_data[:10], train_label[:10])

    # Tokenization
    X_test = word_tokenizer.texts_to_sequences(test_data)
    y_test = tag_tokenizer.texts_to_sequences(test_label)
    test_lens = []
    for sample in y_test:
        test_lens.append(len(sample))
    
    # pad sequences to the same length
    # Determine the maximum sequence length from both features and labels
    max_seq_length = max(max([len(seq) for seq in X_test]), max([len(seq) for seq in y_test]))

    # Now pad both the features and the labels to the maximum sequence length
    X_test = np.array(pad_sequences(X_test, maxlen=max_seq_length, padding='post'))
    y_test = np.array(pad_sequences(y_test, maxlen=max_seq_length, padding='post'))

    # model.evaluate(X_test, y_test)
    predictions = model.predict(X_test)
    predict_tags = np.argmax(predictions, axis=-1)
    '''
    print(predict_tags)
    print(y_test)
    print(test_lens)
    '''
    error_by_sentence_count = 0
    total_sentence = len(y_test)
    error_by_word_count = 0
    total_words = sum(test_lens)
    for i in range(len(y_test)):
        if not (predict_tags[i] == y_test[i]).all():
            error_by_sentence_count += 1
    line = 0
    for word_len in test_lens:
        for i in range(word_len):
            if not predict_tags[line][i] == y_test[line][i]:
                error_by_word_count += 1
        line += 1
    print("error rate by word:      ", error_by_word_count / total_words, f" ({error_by_word_count} errors out of {total_words})")
    print("error rate by sentence:  ", error_by_sentence_count / total_sentence, f" ({error_by_sentence_count} errors out of {total_sentence})")

    # debugg purpose
    #break
    ################







