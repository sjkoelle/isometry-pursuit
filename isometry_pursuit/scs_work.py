"""Source-matched sparse-direct arithmetic accounting for SCS 3.2.9.

Only the QDLDL numeric factorization and factor solves are counted exactly here.
Cone, vector, residual, normalization and CVXPY compilation work are excluded.
"""
import ctypes
from pathlib import Path
import subprocess

import numpy as np
from scipy import sparse


def build_audit_library(source_root, output_path):
    """Build the unmodified AMD/QDLDL routines from a supplied SCS source tree."""
    root, output = Path(source_root), Path(output_path)
    amd, qdldl = root / 'linsys/external/amd', root / 'linsys/external/qdldl'
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(['cc', '-O2', '-shared', '-fPIC', '-DDLONG=1',
                    '-I'+str(root/'include'), '-I'+str(amd), '-I'+str(qdldl),
                    *map(str, sorted(amd.glob('*.c'))), str(qdldl/'qdldl.c'),
                    '-o', str(output)], check=True)
    return output


def sparse_direct_work(A, library_path):
    """Reproduce SCS's AMD ordering and QDLDL symbolic fill for P_quadratic=0.

    KKT diagonal values do not affect symbolic fill. Positive/negative unit
    diagonals model its sparsity pattern. The factor has implicit unit diagonal.
    QDLDL source gives sum_j(s_j**2 + 2*s_j) + system_size numeric-factor FLOPs,
    with s_j the strict-lower column counts. A factor solve takes
    4*nnz(L_strict) + system_size FLOPs. Multiply/add/divide each count as one.
    Integer ordering/symbolic operations and memory movement are excluded.
    """
    A = sparse.csc_matrix(A)
    m, n = A.shape
    K = sparse.triu(sparse.bmat([[sparse.eye(n), A.T],
                                [A, -sparse.eye(m)]], format='csc'), format='csc')
    K.sort_indices()
    size = n + m
    lib = ctypes.CDLL(str(Path(library_path).resolve()))
    integer = ctypes.c_longlong
    ip, dp = ctypes.POINTER(integer), ctypes.POINTER(ctypes.c_double)
    lib.amd_order.argtypes = [integer, ip, ip, ip, dp, dp]
    lib.amd_order.restype = integer
    lib.QDLDL_etree.argtypes = [integer, ip, ip, ip, ip, ip]
    lib.QDLDL_etree.restype = integer
    def ptr(a):
        return a.ctypes.data_as(ip)
    ap, ai = K.indptr.astype(np.int64), K.indices.astype(np.int64)
    permutation, info = np.zeros(size, dtype=np.int64), np.zeros(20)
    status = lib.amd_order(size, ptr(ap), ptr(ai), ptr(permutation), None,
                           info.ctypes.data_as(dp))
    if status not in (0, 1):
        raise RuntimeError(f'AMD failed with status {status}')
    symmetric = K + sparse.triu(K, k=1).T
    permuted = sparse.triu(symmetric[permutation, :][:, permutation], format='csc')
    permuted.sort_indices()
    ap, ai = permuted.indptr.astype(np.int64), permuted.indices.astype(np.int64)
    work, counts, tree = (np.zeros(size, dtype=np.int64) for _ in range(3))
    nnz = int(lib.QDLDL_etree(size, ptr(ap), ptr(ai), ptr(work), ptr(counts), ptr(tree)))
    if nnz < 0:
        raise RuntimeError(f'QDLDL symbolic analysis failed with status {nnz}')
    return dict(conic_variables=n, conic_constraints=m, A_nnz=int(A.nnz),
                kkt_dimension=size, factor_strict_lower_nnz=nnz,
                factorization_flops=int(np.sum(counts*counts + 2*counts)) + size,
                factor_solve_flops=4*nnz + size)
