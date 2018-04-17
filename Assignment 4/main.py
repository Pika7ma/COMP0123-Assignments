from Generator import Generator
import argparse
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=123456)
    parser.add_argument('--pro', type=int, nargs='+', default=(1, 1, 1))
    parser.add_argument('--verbose', dest='verbose', action='store_true')
    parser.set_defaults(verbose=False)
    parser.add_argument('--plot', dest='plot', action='store_true')
    parser.set_defaults(plot=False)
    args = parser.parse_args()
    g = Generator(args.seed)
    print(args.pro)
    g.gen_mvnorm(args.pro, n=args.n)
    if args.verbose:
        g.print_mean_cov()
        print('---------Result---------')
        g.bayes_predict()
        g.euclid_predict()
    if args.plot:
        g.bayes_plot(os.path.join('./', str.join('', [str(i) for i in args.pro])))
        g.euclid_plot(os.path.join('./', str.join('', [str(i) for i in args.pro])))


if __name__ == '__main__':
    main()
