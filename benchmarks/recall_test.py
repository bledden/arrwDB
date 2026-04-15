from rust_hnsw import RustFastHNSWIndex
import numpy as np, time, h5py

with h5py.File('/home/ubuntu/.cache/arrwdb_bench/ann/sift-128-euclidean.hdf5', 'r') as f:
    train = np.array(f['train'], dtype=np.float32)
    test = np.array(f['test'], dtype=np.float32)
    gt = np.array(f['neighbors'], dtype=np.int32)

# Use raw vectors with L2 metric — matches ground truth exactly
n = train.shape[0]

for ef_c in [400, 800]:
    idx = RustFastHNSWIndex(dimension=128, m=48, ef_construction=ef_c, ef_search=50, metric='l2')
    ids = [f'v{i}' for i in range(n)]
    t0 = time.time()
    idx.build_bulk(ids, train.ravel())
    bt = time.time() - t0
    print(f'ef_c={ef_c}: build={bt:.0f}s ({n/bt:.0f} vec/s)')

    for ef in [100, 200, 400, 800]:
        recalls = []
        t0 = time.time()
        for i in range(10000):
            results = idx.search(test[i], k=10, ef_override=ef)
            rids = {int(v[1:]) for v, _ in results}
            gt_ids = set(gt[i, :10].tolist())
            recalls.append(len(rids & gt_ids) / 10)
        st = time.time() - t0
        qps = 10000 / st
        print(f'  ef={ef}: recall={np.mean(recalls):.4f} QPS={qps:.0f}')
    print()
