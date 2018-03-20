from KNN import KNN
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from mpi4py import MPI
import argparse

comm = MPI.COMM_WORLD
rank = comm.Get_rank()


def plot(path, acc):
    """
    Plot accuracy under various random seeds
    :type path: str
    :type acc: numpy.ndarray
    :return: nothing
    """
    seed, k = acc.shape
    print('seed, k = %d, %d' % (seed, k))

    acc_mean = np.mean(acc, axis=0)
    acc_std = np.std(acc, axis=0)
    acc_min = acc_mean - acc_std
    acc_max = acc_mean + acc_std
    k_range = np.arange(1, k + 1, dtype=np.int32)

    plt.style.use('ggplot')
    plt.figure('K Nearest Neighbour')

    plt.plot(k_range, acc_min, lw=1, color='r', alpha=0.22)
    plt.plot(k_range, acc_max, lw=1, color='r', alpha=0.22)
    plt.fill_between(k_range, acc_min, acc_max, color='r', alpha=0.2)

    plt.plot(k_range, acc_mean, marker='o', mec='r', mfc='w', ls='--', ms=5, lw=1, label='Mean Accuracy', color='r', alpha=0.7)
    plt.xticks(k_range, ['%d' % i for i in k_range])

    plt.xlabel('K Value')
    plt.ylabel('Accuracy under Various Seeds')
    if not os.path.exists(path):
        os.mkdir(path)
    plt.savefig(os.path.join(path, 'plot.png'), dpi=128)


def main():
    parser = argparse.ArgumentParser(description='KNN')
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--seed', type=int, default=123456)
    parser.add_argument('--action', type=str, default='test')
    args = parser.parse_args()
    if args.action == 'cv':
        knn = KNN(
            './data/semeion_train.csv',
            './data/semeion_test.csv',
        )
        knn.update_seed(args.seed + rank)
        k_range = args.k
        acc = np.zeros((k_range, ))
        for k in range(k_range):
            acc[k] = knn.test_cv(k + 1, verbose=False)
            print('seed[%d]k[%d]: %.2f%%' % (rank, k, 100 * acc[k]))

        # Gather accuracy in different threads
        all_acc = comm.allgather(acc)
        all_acc = np.vstack(all_acc)

        if rank == 0:
            with open('accuracy.json', 'w') as f:
                json.dump(all_acc.tolist(), f)
            plot('plot', all_acc)
    elif args.action == 'test':
        if rank == 0:
            knn = KNN(
                './data/semeion_train.csv',
                './data/semeion_test.csv',
            )
            knn.update_seed(args.seed + rank)
            k_range = args.k
            acc = np.zeros((k_range, ))
            for k in range(k_range):
                acc[k] = knn.test(k + 1)
            plot('plot', acc.reshape((1, k_range)))


if __name__ == '__main__':
    main()
