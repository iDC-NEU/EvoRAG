import time
import argparse
# import json
import multiprocessing
import json

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--db',
                        type=str,
                        default=None,
                        help='database name, e.g. test.')
    # parser.add_argument('--reuse', action='store_true', help="reuse database")
    parser.add_argument('--data',
                        type=str,
                        required=True,
                        help='dataset name.')
    parser.add_argument("--iteration", type=str, help="number of iteration (e.g., 3, 5 ) ", default='')
    # parser.add_argument(
    #     '--proc',
    #     type=int,
    #     # required=True,
    #     default=10,
    #     help='processor numbers, e.g. test.')

    args = parser.parse_args()
    # if not args.db:
    if args.data == 'rgb':
        redundancy_rate = 0.293
        triplet

    print(args)

    #with open(f"../logs/triplets/{args.data}.json", "r", encoding="utf-8") as file:
    for i in range(int(args.iteration)):
        with open(f"./logs/{args.data}/{args.db}_unchanged_{i}.json", "r", encoding="utf-8") as file:
            triplets_score = json.load(file)

        loaded_triplets = []
        for key, value in triplets_score.items():
            if value["score"] > 70:
                loaded_triplets.append(value["triplet"])


    # loaded_triplets = [(str(x), str(y), str(z)) for x, y, z in loaded_triplets]
    loaded_triplets = list(set(loaded_triplets))

    for triplet in loaded_triplets:
        assert len(triplet) == 3
        for x in triplet:
            assert len(x) > 0, triplet

    if args.iteration:
        print(f'load {len(loaded_triplets)} triplets from ./logs/triplets/{args.db}_unchanged_{args.iteration}.json.')
    else:
        print(f'load {len(loaded_triplets)} triplets from ./logs/triplets/{args.db}.json.')

    parallel_insert(loaded_triplets, args.db, args.proc, args.reuse)
