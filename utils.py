
import os
import numpy as np
from scipy.misc import imsave


def generate_data(data, size=1000):
    inputs, targets, reals = [], [], []
    num_labels = data.test_labels.shape[1]
    i = 0
    while i < size and i < len(data.test_data):
        inputs.append(data.test_data[i].astype(np.float32))
        reals.append(np.argmax(data.test_labels[i]))
        other_labels = [x for x in range(
            num_labels) if data.test_labels[i][x] == 0]
        random_target = [0 for _ in range(num_labels)]
        random_target[np.random.choice(other_labels)] = 1
        targets.append(random_target)
        i += 1
    inputs = np.array(inputs)
    targets = np.array(targets)
    return inputs, targets, reals


def l_inf_dist(orig_img, new_img):
    return np.max(np.abs(orig_img.ravel() - new_img.ravel()))


def l_2_dist(orig_img, new_img):
    return np.sqrt(np.sum((orig_img.ravel()-new_img.ravel())**2))


def l_0_dist(orig_img, new_img):
    return np.sum((orig_img.ravel() - new_img.ravel()) != 0)


def save_image(img, path):
    imsave(path, (img+0.5))


class ResultLogger(object):
    def __init__(self, path, flags):
        assert path is not None, 'You must give an output results dir.'
        self.path = path
        if not os.path.exists(path):
            os.mkdir(path)
        if flags is not None:
            config_fh = open(os.path.join(self.path, 'config.txt'), 'w')
            config_fh.write(str(flags))
            config_fh.close()
        self.results_fh = open(os.path.join(self.path, 'results.csv'), 'w')
        self.results_fh.write(
            'index, real, target, queries, l_0, l_2, l_inf, time\n')
        self.time_total = 0.0
        self.queries_total = 0.0
        self.query_list = []
        self.l2_total = 0.0
        self.num_success = 0.0

    def add_result(self, idx, src_img, adv_img, real, target, queries, attack_time, margin_log):
        save_image(src_img,
                   os.path.join(self.path, 'orig_{}.jpg'.format(idx)))
        save_image(adv_img,
                   os.path.join(self.path, 'adv_{}.jpg'.format(idx)))
        np.save(os.path.join(self.path, 'log_{}.npy'.format(idx)), margin_log)
        attack_l2 = l_2_dist(src_img, adv_img)
        self.results_fh.write('{}, {}, {}, {}, {}, {}, {}, {}\n'.format(
            idx, real, target, queries,
            l_0_dist(src_img, adv_img),
            attack_l2,
            l_inf_dist(src_img, adv_img),
            attack_time
        ))

        self.num_success += 1
        self.time_total += attack_time
        self.queries_total += queries
        self.query_list.append(queries)
        self.l2_total += attack_l2
        self.results_fh.flush()

    def close(self, num_attempts):
        self.results_fh.close()
        stats_fh = open(os.path.join(self.path, 'stats.txt'), 'w')
        stats_fh.write('Success rate = {}/{} ({} %%)\n'.format(int(self.num_success),
                                                               num_attempts, 100*self.num_success/num_attempts))
        stats_fh.write('Mean queries count = {}.\n'.format(
            self.queries_total / self.num_success
        ))
        stats_fh.write('median queries count = {}.\n'.format(
            np.median(self.query_list)
        ))
        stats_fh.write('Mean l2 distance = {}.\n'.format(
            self.l2_total / self.num_success
        ))
        stats_fh.write('Mean attack time = {} seconds.\n'.format(
            self.time_total / self.num_success
        ))
        stats_fh.close()
