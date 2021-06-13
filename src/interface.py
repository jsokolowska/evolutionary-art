import argparse
import main
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Pixel-based algorithm for evolutionary art creation and imitation")

    parser.add_argument('--iterations', "-it", type=int, metavar='I', default=150,
                        help='number of iterations for evolutionary algorithm')
    parser.add_argument('--step', '-s', metavar='S', type=int, default=30,
                        help='step size for algorithm, every step iteration an image will be shown or saved')
    parser.add_argument('--show', action='store_true', help='if true show images')
    parser.add_argument('width', type=int, help='result image width')
    parser.add_argument('height', type=int, help='result image height')
    parser.add_argument('--input', type=str, help='image to evolve from')

    parser.add_argument('--functions', '-f', type=int, metavar='F_NUM', choices=range(6),
                        help='aesthetic functions per image chanel')

    parser.add_argument('--weights', '-w', nargs=4, default=[1, 1, 1, 1.3],
                        help='target weights for each chanel and target artwork, by default=[1,1,1,1.3]')

    group = parser.add_argument_group()
    group.add_argument('--target', '-t', type=str,
                       help='path to artwork that will be used as target artwork to imitate')
    group.add_argument('--imitation_mode', '--mode', choices=['clip', 'resize'], metavar='MODE', type=str,
                       help='clip or resize')

    group2 = parser.add_argument_group('Inner algorithm parameters')
    group2.add_argument('-p', type=int, default=10, help='optional parameter p to be used for aesthetic fitness functions')
    group2.add_argument('--max_velocity', type=int, default=10, help='max particle velocity for PSO algorithm')
    group2.add_argument('-c1', type=float, default=2, help='')
    group2.add_argument('-c2', type=float, default=2, help='')
    group2.add_argument('--inertia_min', type=float, default=0.4, help='')
    group2.add_argument('--inertia_max', type=float, default=0.9, help='')

    args = parser.parse_args()
    print(os.getcwd())
    main.init_and_run(args)
