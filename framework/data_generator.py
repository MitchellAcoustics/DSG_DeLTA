import numpy as np
import h5py, os, pickle, torch
import time
from framework.utilities import calculate_scalar, scale
import framework.config as config



class DataGenerator(object):
    def __init__(self, batch_size, seed=42, normalization=False):

        self.batch_size = batch_size
        self.random_state = np.random.RandomState(seed)
        self.validate_random_state = np.random.RandomState(0)

        # Load data
        load_time = time.time()

        file_path = os.path.join(config.root, '5foldcv_training.pickle')
        if not os.path.exists(file_path):
            self.train_audio_ids, self.train_rates, self.train_event_label = self.load_rate_event_id_training()
            self.train_x = self.load_x(self.train_audio_ids)
            data = {}
            data['audio_ids'] = self.train_audio_ids
            data['rates'] = self.train_rates
            data['event_label'] = self.train_event_label
            data['x'] = self.train_x
            with open(file_path, 'wb') as f:
                pickle.dump(data, f)
        else:
            print('using: ', file_path)
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            self.train_audio_ids, self.train_rates, self.train_event_label = \
                data['audio_ids'], data['rates'], data['event_label']
            self.train_x = data['x']

        file_path = os.path.join(config.root, '5foldcv_validation.pickle')
        if not os.path.exists(file_path):
            self.val_audio_ids, self.val_rates, self.val_event_label = self.load_rate_event_id_validation()
            self.val_x = self.load_x(self.val_audio_ids)
            data = {}
            data['audio_ids'] = self.val_audio_ids
            data['rates'] = self.val_rates
            data['event_label'] = self.val_event_label
            data['x'] = self.val_x
            with open(file_path, 'wb') as f:
                pickle.dump(data, f)
        else:
            print('using: ', file_path)
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            self.val_audio_ids, self.val_rates, self.val_event_label = \
                data['audio_ids'], data['rates'], data['event_label']
            self.val_x = data['x']

        print('Loading data time: {:.3f} s'.format(time.time() - load_time))

        print('Split development data to {} training and {} '
              'validation data. '.format(len(self.train_audio_ids),
                                         len(self.val_audio_ids)))

        self.normal = normalization
        if self.normal:
            (self.mean, self.std) = calculate_scalar(self.train_x)

    def load_x(self, audio_ids):
        x_list = []
        for each_id in audio_ids:
            idfile = os.path.join(config.root, 'DeLTA_mp3_boost_8dB_mel64', each_id+'.npy')
            x_list.append(np.load(idfile))
        return np.array(x_list)


    def load_rate_event_id_training(self):
        ############ read all dataset ################################################################
        data_path = os.path.join(config.root, 'training_validation_dataset',
                                 'proportionally_randomly_split')
        sub_set = 'training'
        audio_id_file = os.path.join(data_path, sub_set + '_audio_id.txt')
        rate_file = os.path.join(data_path, sub_set + '_annoyance_rate.txt')
        event_label_file = os.path.join(data_path, sub_set + '_sound_source.txt')

        all_audio_ids = []
        with open(audio_id_file, 'r') as f:
            for line in f.readlines():
                part = line.split('\n')[0]
                if part:
                    all_audio_ids.append(part)
        rates1 = np.loadtxt(rate_file)[:, None]
        event_label1 = np.loadtxt(event_label_file)

        sub_set = 'validation'
        audio_id_file = os.path.join(data_path, sub_set + '_audio_id.txt')
        rate_file = os.path.join(data_path, sub_set + '_annoyance_rate.txt')
        event_label_file = os.path.join(data_path, sub_set + '_sound_source.txt')

        with open(audio_id_file, 'r') as f:
            for line in f.readlines():
                part = line.split('\n')[0]
                if part:
                    all_audio_ids.append(part)
        rates2 = np.loadtxt(rate_file)[:, None]
        event_label2 = np.loadtxt(event_label_file)

        all_rates = np.concatenate((rates1, rates2), axis=0)
        all_event_label = np.concatenate((event_label1, event_label2), axis=0)
        # print(len(audio_ids), rates.shape, event_label.shape)

        ##############################################################################################
        data_path = os.path.join(config.root, 'training_validation_dataset',
                                 'five_fold_split_andrew')
        audio_ids = []

        sub_set = [1, 2, 3, 4, 5]
        for fold in sub_set:
            audio_id_file = os.path.join(data_path, 'fold_' + str(fold) + '_ids.txt')
            with open(audio_id_file, 'r') as f:
                for line in f.readlines():
                    part = line.split('\n')[0]
                    if part:
                        audio_ids.append(part)

        # print(all_audio_ids)
        # print(len(all_audio_ids))
        ids_index = [all_audio_ids.index(each+'.mp3') for each in audio_ids]

        rates = all_rates[ids_index]
        event_label = all_event_label[ids_index]
        return audio_ids, rates, event_label

    def load_rate_event_id_validation(self):
        ############ read all dataset ################################################################
        data_path = os.path.join(config.root, 'training_validation_dataset',
                                 'proportionally_randomly_split')
        sub_set = 'training'
        audio_id_file = os.path.join(data_path, sub_set + '_audio_id.txt')
        rate_file = os.path.join(data_path, sub_set + '_annoyance_rate.txt')
        event_label_file = os.path.join(data_path, sub_set + '_sound_source.txt')

        all_audio_ids = []
        with open(audio_id_file, 'r') as f:
            for line in f.readlines():
                part = line.split('\n')[0]
                if part:
                    all_audio_ids.append(part)
        rates1 = np.loadtxt(rate_file)[:, None]
        event_label1 = np.loadtxt(event_label_file)

        sub_set = 'validation'
        audio_id_file = os.path.join(data_path, sub_set + '_audio_id.txt')
        rate_file = os.path.join(data_path, sub_set + '_annoyance_rate.txt')
        event_label_file = os.path.join(data_path, sub_set + '_sound_source.txt')

        with open(audio_id_file, 'r') as f:
            for line in f.readlines():
                part = line.split('\n')[0]
                if part:
                    all_audio_ids.append(part)
        rates2 = np.loadtxt(rate_file)[:, None]
        event_label2 = np.loadtxt(event_label_file)

        all_rates = np.concatenate((rates1, rates2), axis=0)
        all_event_label = np.concatenate((event_label1, event_label2), axis=0)
        # print(len(audio_ids), rates.shape, event_label.shape)

        ##############################################################################################
        data_path = os.path.join(config.root, 'training_validation_dataset',
                                 'five_fold_split_andrew')
        audio_ids = []

        audio_id_file = os.path.join(data_path, 'valid_ids.txt')
        with open(audio_id_file, 'r') as f:
            for line in f.readlines():
                part = line.split('\n')[0]
                if part:
                    audio_ids.append(part)

        # print(all_audio_ids)
        # print(len(all_audio_ids))
        ids_index = [all_audio_ids.index(each+'.mp3') for each in audio_ids]

        rates = all_rates[ids_index]
        event_label = all_event_label[ids_index]
        return audio_ids, rates, event_label


    def generate_train(self):
        audios_num = len(self.train_audio_ids)
        audio_indexes = [i for i in range(audios_num)]

        self.random_state.shuffle(audio_indexes)

        iteration = 0
        pointer = 0

        while True:
            if pointer >= audios_num:
                pointer = 0
                self.random_state.shuffle(audio_indexes)

            # Get batch indexes
            batch_audio_indexes = audio_indexes[pointer: pointer + self.batch_size]
            pointer += self.batch_size

            iteration += 1

            batch_x = self.train_x[batch_audio_indexes]
            batch_y = self.train_rates[batch_audio_indexes]
            batch_y_event = self.train_event_label[batch_audio_indexes]

            if self.normal:
                batch_x = self.transform(batch_x)

            yield batch_x, batch_y, batch_y_event


    def generate_validate(self, data_type, max_iteration=None):
        audios_num = len(self.val_audio_ids)
        audio_indexes = [i for i in range(audios_num)]

        self.validate_random_state.shuffle(audio_indexes)

        print('Number of {} audios in {}'.format(len(audio_indexes), data_type))

        iteration = 0
        pointer = 0

        while True:
            if iteration == max_iteration:
                break

            # Reset pointer
            if pointer >= audios_num:
                break

            batch_audio_indexes = audio_indexes[pointer: pointer + self.batch_size]
            pointer += self.batch_size

            iteration += 1

            batch_x = self.val_x[batch_audio_indexes]
            batch_y = self.val_rates[batch_audio_indexes]
            batch_y_event = self.val_event_label[batch_audio_indexes]

            if self.normal:
                batch_x = self.transform(batch_x)

            yield batch_x, batch_y, batch_y_event

    def transform(self, x):
        """Transform data.

        Args:
          x: (batch_x, seq_len, freq_bins) | (seq_len, freq_bins)

        Returns:
          Transformed data.
        """

        return scale(x, self.mean, self.std)




