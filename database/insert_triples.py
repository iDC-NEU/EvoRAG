import time
import argparse
from database.graph.nebulagraph.nebulagraph import NebulaDB
# import json
import multiprocessing
import json
# import os
# import sys
# from utils.utils import create_dir
# home_dir = os.path.expanduser('~')

# if log:
#     log_dir = os.path.join(home_dir, 'rag-cache/experiment/exp1-motivation/log')
#     create_dir(log_dir)
#     log_file = os.path.join(log_dir, f'insert-{len(triplets)}-triplets-pid{pid}.log')

# create_dir('./log')
# savedStdout = sys.stdout  #保存标准输出流
# file = open(log_filename, 'w')
# sys.stdout = file  #标准输出重定向至文件

# if args.log_path:
#     file.close()
#     sys.stdout = savedStdout  #标准输出重定向至文件

# from tqdm import tqdm


def insert_triple(pid, triplets, graph_db: NebulaDB, verbose=False, log=False):
    start_time = time.time()
    # for i, triplet in tqdm(enumerate(triplets), f"insert triplets in {db_names}"):
    for i, triplet in enumerate(triplets):
        if i and i % 10000 == 0 and verbose:
            print(
                f'processor {pid} insert {i}/{len(triplets)} triplets, cost {time.time() - start_time : .3f}s.'
            )
        if triplet[0] and triplet[1] and triplet[2]:
            graph_db.upsert_triplet(triplet)
    end_time = time.time()
    print(
        f'pid {pid} insert {len(triplets)} triplets cost {end_time - start_time : .3f}s.'
    )


def parallel_insert(triplets, db_name, nproc=5, reuse=False):
    start_time = time.time()
    processes = []
    n_triplets = len(triplets)
    if n_triplets < 100:
        nproc = 1
    step = (n_triplets + nproc - 1) // nproc

    print(n_triplets, db_name, nproc, step)
    print(f'\ninsert {n_triplets} triplets in {db_name}, nproc={nproc}')

    # nebula_db = NebulaDB(db_name)
    nebula_db = NebulaDB(space_name = db_name)
    if not reuse:
        nebula_db.clear()

    # assert(False)

    for i in range(nproc):
        start = i * step
        end = min(start + step, n_triplets)
        print(f'pid {i} take {start}-{end}')
        p = multiprocessing.Process(target=insert_triple,
                                    args=(
                                        i,
                                        triplets[start:end],
                                        nebula_db,
                                        True,
                                    ))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()
    end_time = time.time()

    print(f'insert_triple_parallel cost {end_time - start_time:.3f}')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--db',
                        type=str,
                        default=None,
                        help='database name, e.g. test.')
    parser.add_argument('--reuse', action='store_true', help="reuse database")
    parser.add_argument('--data',
                        type=str,
                        required=True,
                        help='dataset name.')
    parser.add_argument('--algorithm',
                        type=str,
                        default=None)
    parser.add_argument("--iteration", type=int, help="number of iteration (e.g., 3, 5 ) ", default=0)
    parser.add_argument("--threshold", type=int, help="Whether to discard triplets below the minimum threshold", default=0)
    parser.add_argument(
        '--proc',
        type=int,
        # required=True,
        default=10,
        help='processor numbers, e.g. test.')

    args = parser.parse_args()
    if not args.db:
        args.db = args.data

    print(args)

    #with open(f"../logs/triplets/{args.data}.json", "r", encoding="utf-8") as file:
    # if args.iteration:
    #     with open(f"./logs/triplets/{args.db}_{args.iteration}.json", "r", encoding="utf-8") as file:
    #         triplets_score = json.load(file)
    # else:
    #     with open(f"./logs/triplets/{args.db}.json", "r", encoding="utf-8") as file:
    #         triplets_score = json.load(file)
    
    if args.algorithm:
        path = f"./logs/{args.algorithm}/triplets/{args.db}_{args.iteration}.json"
    else:
        path = f"./logs/triplets/{args.db}_{args.iteration}.json"
        
    with open(path, "r", encoding="utf-8") as file:
        triplets_score = json.load(file)

    ###完善代码
    # 初始化 loaded_triplets 列表
    loaded_triplets = []

    # 遍历 triplets_score 中的每个项
    for key, value in triplets_score.items():
        # 获取 triplet 字段中的三个短语
        if value["score"] > args.threshold:
            x, y, z = value["triplet"]
            # 将三元组添加到 loaded_tripletss
            loaded_triplets.append((str(x), str(y), str(z)))


    # loaded_triplets = [(str(x), str(y), str(z)) for x, y, z in loaded_triplets]
    loaded_triplets = list(set(loaded_triplets))

    for triplet in loaded_triplets:
        assert len(triplet) == 3
        for x in triplet:
            assert len(x) > 0, triplet


    print(f'load {len(loaded_triplets)} triplets from {path}.')

    # print(f'load {len(loaded_triplets)} triplets from ./logs/triplets/{args.db}.json.')

    parallel_insert(loaded_triplets, args.db, args.proc, args.reuse)
