import sys
from collections import deque

def main():
    _readline = sys.stdin.readline
    _write = sys.stdout.write
    _flush = sys.stdout.flush

    first = _readline().split()
    N = int(first[0])
    T = int(first[1])
    NN = N * N
    THRESHOLD = 6

    grid_flat = bytearray(NN)
    empty_list = []
    for r in range(N):
        row = _readline().rstrip('\n')
        base = r * N
        for c in range(N):
            if row[c] == '.':
                grid_flat[base + c] = 1
                empty_list.append(base + c)

    num_empty = len(empty_list)

    adj = [None] * NN
    for idx in empty_list:
        r = idx // N
        c = idx - r * N
        nbrs = []
        if r > 0 and grid_flat[idx - N]:
            nbrs.append(idx - N)
        if r < N - 1 and grid_flat[idx + N]:
            nbrs.append(idx + N)
        if c > 0 and grid_flat[idx - 1]:
            nbrs.append(idx - 1)
        if c < N - 1 and grid_flat[idx + 1]:
            nbrs.append(idx + 1)
        adj[idx] = nbrs

    def bfs(start):
        dist = [-1] * NN
        dist[start] = 0
        q = deque()
        q.append(start)
        _adj = adj
        while q:
            fi = q.popleft()
            d = dist[fi] + 1
            for ni in _adj[fi]:
                if dist[ni] < 0:
                    dist[ni] = d
                    q.append(ni)
        return dist

    NUM_LANDMARKS = 40
    if num_empty <= NUM_LANDMARKS:
        landmarks = list(empty_list)
        landmark_dist = [bfs(lm) for lm in landmarks]
    else:
        landmarks = []
        landmark_dist = []
        min_dist = [10 ** 9] * NN
        first_lm = empty_list[0]
        landmarks.append(first_lm)
        d0 = bfs(first_lm)
        landmark_dist.append(d0)
        for idx in empty_list:
            v = d0[idx]
            if 0 <= v < min_dist[idx]:
                min_dist[idx] = v
        for _ in range(NUM_LANDMARKS - 1):
            best_idx = -1
            best_val = -1
            for idx in empty_list:
                if min_dist[idx] > best_val:
                    best_val = min_dist[idx]
                    best_idx = idx
            if best_idx == -1:
                break
            landmarks.append(best_idx)
            dk = bfs(best_idx)
            landmark_dist.append(dk)
            for idx in empty_list:
                v = dk[idx]
                if 0 <= v < min_dist[idx]:
                    min_dist[idx] = v

    num_landmarks = len(landmarks)

    # Compact distance arrays: lm_dist_compact[k][i] = distance from landmark k to empty_list[i]
    lm_dist_compact = []
    for k in range(num_landmarks):
        dk = landmark_dist[k]
        lm_dist_compact.append([dk[idx] for idx in empty_list])

    # Free full landmark_dist to save memory
    landmark_dist = None

    # Select best first landmark (minimizes max bucket size over all empty cells)
    best_first = 0
    best_first_max_bucket = num_empty + 1
    for k in range(num_landmarks):
        buckets = {}
        max_b = 0
        dk = lm_dist_compact[k]
        for i in range(num_empty):
            dd = dk[i]
            if dd < 0:
                continue
            cnt = buckets.get(dd, 0) + 1
            buckets[dd] = cnt
            if cnt > max_b:
                max_b = cnt
        if max_b < best_first_max_bucket:
            best_first_max_bucket = max_b
            best_first = k

    # Build bucket_map for first landmark: distance -> list of compact indices
    bucket_map = {}
    fl_dist = lm_dist_compact[best_first]
    for i in range(num_empty):
        dd = fl_dist[i]
        if dd < 0:
            continue
        if dd in bucket_map:
            bucket_map[dd].append(i)
        else:
            bucket_map[dd] = [i]

    # Precompute best second landmark and sub-bucket maps for each first-landmark bucket
    # that has more than THRESHOLD candidates
    second_lm_for_bucket = {}
    sub_bucket_maps = {}
    for dist_val, cand_list in bucket_map.items():
        if len(cand_list) <= THRESHOLD:
            continue
        best_k = -1
        best_score = len(cand_list) + 1
        for k in range(num_landmarks):
            if k == best_first:
                continue
            buckets = {}
            max_b = 0
            dk = lm_dist_compact[k]
            for i in cand_list:
                dd = dk[i]
                cnt = buckets.get(dd, 0) + 1
                buckets[dd] = cnt
                if cnt > max_b:
                    max_b = cnt
                    if max_b >= best_score:
                        break
            else:
                if max_b < best_score:
                    best_score = max_b
                    best_k = k
        if best_k != -1:
            second_lm_for_bucket[dist_val] = best_k
            # Build sub-bucket map: (first_dist, second_dist) -> list of compact indices
            dk = lm_dist_compact[best_k]
            sbm = {}
            for i in cand_list:
                dd2 = dk[i]
                if dd2 in sbm:
                    sbm[dd2].append(i)
                else:
                    sbm[dd2] = [i]
            sub_bucket_maps[dist_val] = sbm

    def query_cell(flat_idx):
        r = flat_idx // N
        c = flat_idx - r * N
        _write(str(r + 1))
        _write(' ')
        _write(str(c + 1))
        _write('\n')
        _flush()
        return int(_readline())

    def select_best_landmark(cand_indices, queried):
        best_k = -1
        best_score = len(cand_indices) + 1
        for k in range(num_landmarks):
            if k in queried:
                continue
            buckets = {}
            max_b = 0
            dk = lm_dist_compact[k]
            for i in cand_indices:
                dd = dk[i]
                cnt = buckets.get(dd, 0) + 1
                buckets[dd] = cnt
                if cnt > max_b:
                    max_b = cnt
                    if max_b >= best_score:
                        break
            else:
                if max_b < best_score:
                    best_score = max_b
                    best_k = k
        return best_k

    query_count = 0
    max_queries = T

    while query_count < max_queries:
        resp = query_cell(landmarks[best_first])
        query_count += 1
        if resp == 0:
            continue
        if resp < 0:
            continue

        candidates = bucket_map.get(resp)
        if not candidates:
            continue

        queried = {best_first}

        # Use precomputed second landmark if available
        if resp in second_lm_for_bucket and query_count < max_queries:
            k2 = second_lm_for_bucket[resp]
            resp2 = query_cell(landmarks[k2])
            query_count += 1
            queried.add(k2)
            if resp2 == 0:
                continue
            sbm = sub_bucket_maps[resp]
            candidates = sbm.get(resp2)
            if not candidates:
                continue

        # Continue narrowing with dynamic landmark selection if needed
        while len(candidates) > THRESHOLD and query_count < max_queries:
            k = select_best_landmark(candidates, queried)
            if k == -1:
                break
            resp_k = query_cell(landmarks[k])
            query_count += 1
            queried.add(k)
            if resp_k == 0:
                candidates = None
                break
            dk = lm_dist_compact[k]
            candidates = [ci for ci in candidates if dk[ci] == resp_k]

        if not candidates:
            continue

        # Query remaining candidates directly
        for ci in candidates:
            if query_count >= max_queries:
                break
            resp_c = query_cell(empty_list[ci])
            query_count += 1
            if resp_c == 0:
                break

if __name__ == "__main__":
    main()
