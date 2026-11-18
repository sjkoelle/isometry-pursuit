"""Verify full-grid search losses against hashed frozen inputs without refitting."""
import argparse
from collections import Counter
import hashlib
import io
import json
import math
from pathlib import Path

import numpy as np
from scipy.linalg import svd

from manage import ROOT
from isometry_pursuit.cloud_search_plot import join_full_synthetic_results


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',type=Path,default=ROOT/'results/cloud/full-synthetic-20261116-001')
    args=parser.parse_args()
    solvers,_,rows=join_full_synthetic_results(ROOT,args.bundle)
    searches=json.loads((args.bundle/'search_records.json').read_text())['runs']
    sources={r['data_sha256']:r for r in solvers}
    matrices={};errors=[];checked=0;maximum_relative_error=0.
    for search in searches:
        digest=search['data_sha256']
        if digest not in matrices:
            source=sources[digest]
            path=ROOT/'results/cloud'/source['source_run_id']/source['data_key']
            body=path.read_bytes()
            if hashlib.sha256(body).hexdigest()!=digest:
                errors.append(search['job_id']+': input hash mismatch')
                continue
            matrices[digest]=np.load(io.BytesIO(body),allow_pickle=False)
        matrix=matrices[digest]
        candidates=search['candidate_indices']
        for prefix in ('initial_', ''):
            if prefix=='' and search['status']!='complete':
                continue
            indices=search.get(prefix+'indices')
            loss=search.get(prefix+'loss')
            if indices is None or loss is None:
                if prefix=='':errors.append(search['job_id']+': missing completed subset')
                continue
            if (len(indices)!=matrix.shape[0] or len(set(indices))!=len(indices)
                    or not set(indices).issubset(candidates)):
                errors.append(search['job_id']+': invalid '+prefix+'indices')
                continue
            subset=matrix[:,indices]
            # Avoid the local Accelerate matmul path while independently
            # evaluating the same rcond=0 Moore-Penrose inverse.
            u,s,vh=svd(subset,full_matrices=False,lapack_driver='gesvd')
            inverse=np.divide(1.,s,out=np.zeros_like(s),where=s!=0)
            beta=np.einsum('pk,k,kd->pd',vh.T,inverse,u.T)
            actual=float(np.linalg.norm(beta,axis=1).sum())
            residual=float(np.linalg.norm(np.einsum('dp,pk->dk',subset,beta)-np.eye(matrix.shape[0])))
            relative=abs(actual-loss)/max(1.,abs(loss))
            maximum_relative_error=max(maximum_relative_error,relative)
            if not math.isclose(actual,loss,rel_tol=1e-7,abs_tol=1e-7) or residual>1e-6:
                errors.append(search['job_id']+': '+prefix+'loss or reconstruction mismatch')
            checked+=1
    summary=dict(integrity_passed=not errors,matrices=len(rows),unique_inputs=len(matrices),
                 search_records=len(searches),statuses=dict(Counter(r['status'] for r in searches)),
                 subsets_checked=checked,maximum_relative_loss_error=maximum_relative_error,errors=errors)
    (args.bundle/'search_audit.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))
    if errors:raise SystemExit(1)


if __name__=='__main__':
    main()
