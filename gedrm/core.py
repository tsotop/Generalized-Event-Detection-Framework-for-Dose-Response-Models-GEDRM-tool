# gedrm/core.py
import numpy as np
import numba

# build_sparse_table:
def build_sparse_table(arr: np.ndarray):
    N = arr.size
    K = int(np.floor(np.log2(N))) + 1
    st = np.empty((K, N), dtype=arr.dtype)
    st[0] = arr
    for k in range(1, K):
        span = 1 << (k - 1)
        length = N - (1 << k) + 1
        st[k, :length] = np.minimum(
            st[k-1, :length], st[k-1, span:span+length]
        )
    log2 = np.zeros(N+1, dtype=np.int32)
    for i in range(2, N+1):
        log2[i] = log2[i // 2] + 1
    return st, log2
    
# rmq_query:
@numba.njit(cache=True)
def rmq_query(st, log2, L, R):
    """
    Constant-time minimum query over arr[L:R+1].
    """
    j = log2[R - L + 1]
    left = st[j, L]
    right = st[j, R - (1 << j) + 1]
    return left if left < right else right

# detect_sev_with_len
@numba.njit(cache=True)
def detect_exceedance_events(stressor_values, thresholds, window_sizes, st, log2, baseline):
    """
    For each target i and window j, skip if threshold < baseline.
    Use RMQ to test min(stressor[t:t+w]) >= threshold.
    """
    n_targets, n_dur = thresholds.shape
    N = stressor_values.size
    start_flags = np.zeros((n_targets, N), dtype=numba.boolean)
    match_len   = np.zeros((n_targets, N), dtype=np.int32)
    
    for i in range(n_targets): # Renamed 'n_sev'
        for t in range(N):
            max_len = N - t
            for j in range(n_dur):
                w = window_sizes[j]
                if w > max_len:
                    break
                if baseline >= 0.0 and thresholds[i, j] < baseline:
                    continue
                
                # Renamed 'ssc'
                if rmq_query(st, log2, t, t + w - 1) >= thresholds[i, j]:
                    start_flags[i, t] = True
                    match_len[i, t]   = w
                    break
    return start_flags, match_len