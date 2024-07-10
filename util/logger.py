import os
import argparse
import matplotlib.pyplot as plt
from typing import List

from util.args import save_args


class Logger:
    """
    Object for managing the log directory.
    """
    def __init__(self, log_dir: str):  # Store log in log_dir

        self._log_dir = log_dir
        self._logs = dict()

        # Ensure the directories exist
        if not os.path.isdir(self.log_dir):
            os.mkdir(self.log_dir)
        if not os.path.isdir(self.metadata_dir):
            os.mkdir(self.metadata_dir)
        if not os.path.isdir(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)

        print(f"Logger log dir: {self._log_dir}", flush=True)

    @property
    def log_dir(self):
        return self._log_dir

    @property
    def checkpoint_dir(self):
        return self._log_dir + '/checkpoints'

    @property
    def metadata_dir(self):
        return self._log_dir + '/metadata'

    def log_message(self, msg: str):
        """
        Write a message to the log file
        :param msg: the message string to be written to the log file
        """
        if not os.path.isfile(self.log_dir + '/log.txt'):
            open(self.log_dir + '/log.txt', 'w').close() #make log file empty if it already exists
        with open(self.log_dir + '/log.txt', 'a') as f:
            f.write(msg+"\n")

    def create_log(self, log_name: str, key_name: str, *value_names):
        """
        Create a csv for logging information
        :param log_name: The name of the log. The log filename will be <log_name>.csv.
        :param key_name: The name of the attribute that is used as key (e.g. epoch number)
        :param value_names: The names of the attributes that are logged
        """
        if log_name in self._logs.keys():
            raise Exception('Log already exists!')
        # Add to existing logs
        self._logs[log_name] = (key_name, value_names)
        # Create log file. Create columns
        with open(self.log_dir + f'/{log_name}.csv', 'w') as f:
            f.write(','.join((key_name,) + value_names) + '\n')

    def log_values(self, log_name, key, *values):
        """
        Log values in an existent log file
        :param log_name: The name of the log file
        :param key: The key attribute for logging these values
        :param values: value attributes that will be stored in the log
        """
        if log_name not in self._logs.keys():
            raise Exception('Log not existent!')
        if len(values) != len(self._logs[log_name][1]):
            raise Exception('Not all required values are logged!')
        # Write a new line with the given values
        with open(self.log_dir + f'/{log_name}.csv', 'a') as f:
            f.write(','.join(str(v) for v in (key,) + values) + '\n')

    def log_args(self, args: argparse.Namespace):
        save_args(args, self._log_dir)


def create_csv_log(logger: Logger, num_classes: int):
    log_cols = [
        'log_epoch_overview',
        'epoch',
        'test_top1_acc',
        'test_top5_acc',
        'almost_sim_nonzeros',
        'local_size_all_classes',
        'almost_nonzeros_pooled',
        'num_nonzero_prototypes',
        'mean_train_acc',
        'mean_train_loss_during_epoch',
    ]
    if num_classes == 2:
        log_cols[log_cols.index('test_top5_acc')] = 'test_f1'
        print(
            "Your dataset only has two classes. "
            "Is the number of samples per class similar? "
            "If the data is imbalanced, we recommend to use the "
            "--weighted_loss flag to account for the imbalance.",
            flush=True,
        )
    logger.create_log(*log_cols)


def plot_learning_rate_curve(
    log_dir: str,
    name: str,
    lr_vec: List[float],
):
    save_dir = os.path.join(log_dir, f'{name}.png')
    plt.clf()
    plt.plot(lr_vec)
    plt.savefig(save_dir)