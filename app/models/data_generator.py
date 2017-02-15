import os
import sys
import pickle
import numpy as np
module_home = os.environ['NEURAL_PATH']
sys.path.insert(0, module_home)


class DataGenerator(object):

    def __init__(self, batch_size, directory, data_type, complete=False):
        """
        data_type: "training", "validation", or "test"
        complete: If True returns human readable story-query information
        """
        self.question_dir = os.path.join(directory,'questions',data_type)
        self.filelist = os.listdir(self.question_dir)
        self.batch_size = batch_size
        self.cur_file = 0
        self.cur_file_sample_index = 0
        self.cur_file_content = None
        self.cur_file_indices = []
        self.nb_samples_in_file = 0
        self.complete = complete
        self.idx_to_word = None
        self.init_next_file()

        metadata_dict = {}
        f = open(os.path.join(directory, 'metadata', 'metadata.txt'), 'r')
        for line in f:
            entry = line.split(':')
            metadata_dict[entry[0]] = int(entry[1])
        f.close()

        if complete:
            f = open(os.path.join(directory, 'metadata', 'idx_to_word.pickle'), 'rb')
            self.idx_to_word = pickle.load(f)
            f.close()

        self.nb_samples_epoch = metadata_dict[data_type] 


    def get_nb_samples_epoch(self):
        return self.nb_samples_epoch


    def __iter__(self):
        return self


    def __next__(self):
        return self.next()


    def wordify(self, l):
        length = len(l.flat);
        result = np.empty((length,), dtype='object')
        for i in xrange(length):
            if l.flat[i] == 0:
               break 
            result[i] = self.idx_to_word[l.flat[i]]
        np.reshape(result, l.shape)
        return result

    def get_complete_data(self, X, Xq, y):
        X_words = self.wordify(X)
        Xq_words = self.wordify(Xq)
        y_words = self.wordify(y)
        return ((X_words, Xq_words), y_words)


    def init_next_file(self):
        if self.cur_file >= len(self.filelist):
            np.random.shuffle(self.filelist)
            self.cur_file = 0
        next_fn = self.filelist[self.cur_file]
        self.cur_file += 1
        self.cur_file_content = np.load(os.path.join(self.question_dir, next_fn))
        nb_samples_in_file = self.cur_file_content['X'].shape[0]
        self.cur_file_sample_index = 0
        self.cur_file_indices = np.random.permutation(np.arange(nb_samples_in_file))
        self.nb_samples_in_file = self.cur_file_content['X'].shape[0]


    def get_nb_samples(self, nb_samples):
        indices = \
                self.cur_file_indices[self.cur_file_sample_index:self.cur_file_sample_index + nb_samples]
        X = self.cur_file_content['X'][indices]
        Xq = self.cur_file_content['Xq'][indices]
        y = self.cur_file_content['y'][indices]
        self.cur_file_sample_index += nb_samples
        return X, Xq, y


    def next(self):
        needed_samples = self.batch_size

        Xs = []
        Xqs = []
        ys = []
        while needed_samples > 0:
            nb_samples_from_cur_file = min(needed_samples, self.nb_samples_in_file - self.cur_file_sample_index)
            if nb_samples_from_cur_file > 0:
                X_cur_file, Xq_cur_file, y_cur_file = self.get_nb_samples(nb_samples_from_cur_file)
                Xs.append(X_cur_file)
                ys.append(y_cur_file)
                Xqs.append(Xq_cur_file)
                needed_samples -= nb_samples_from_cur_file
            else:
                self.init_next_file()
        
        X = np.vstack(Xs)
        Xq = np.vstack(Xqs)
        y = np.vstack(ys)

        if self.complete:
            complete_data = self.get_complete_data(X, Xq, y)
            return (([X, Xq], y), complete_data)
        else:
            return [X, Xq], y
