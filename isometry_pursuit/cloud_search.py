"""Search saved candidate dictionaries without repeating basis pursuit."""
import time

import numpy as np

from .algorithm import backward_elimination, forward_backward, greedy
from .experiment import _gl_loss
from .transformation import exponential_transformation


def search_saved_support(matrix, candidates, settings, progress=None):
    """Run one initializer and swap refinement; return original column indices."""
    candidates = np.asarray(candidates, dtype=int)
    d, p = matrix.shape
    if len(set(candidates.tolist())) != len(candidates) or np.any(candidates < 0) or np.any(candidates >= p):
        raise ValueError('Invalid candidate indices')
    kind = settings['initialization']
    if kind not in ('forward', 'backward'):
        raise ValueError('Unknown search initialization')
    result = dict(status='pending', candidate_count=len(candidates), D=d, P=p,
                  settings=settings, loss=None, indices=None)
    if len(candidates) < d:
        return {**result, 'status': 'insufficient_support'}
    work = (d*d*sum(k*(k-1) for k in range(d+1, len(candidates)+1))
            if kind=='backward' else sum((len(candidates)-k+1)*d*k*k for k in range(1,d+1)))
    result['initialization_work_proxy'] = work
    maximum_work = settings.get('max_backward_work')
    if kind=='backward' and maximum_work is not None and work > maximum_work:
        return {**result, 'status': 'work_budget'}
    x = matrix if settings.get('search_on_raw', False) else exponential_transformation(matrix)
    sub = x[:, candidates]
    seed = int(settings['search_seed'])
    result.update(phase='initialization', initialization_status='running')
    if progress is not None:
        progress(result)
    started = time.perf_counter()
    initializer = greedy if kind=='forward' else backward_elimination
    initial = (list(range(d)) if len(candidates)==d else
               initializer(sub, _gl_loss, target_dimension=d, random_state=seed))
    result.update(initialization_seconds=time.perf_counter()-started,
                  initialization_status='complete',
                  initial_indices=candidates[initial].tolist(), initial_loss=float(_gl_loss(sub[:, initial])),
                  phase='swap')
    if progress is not None:
        progress(result)
    started = time.perf_counter()
    selected = (initial if len(candidates)==d else forward_backward(
        sub, _gl_loss, target_dimension=d, selected_indices=initial, random_state=seed))
    indices = candidates[selected].tolist()
    loss = float(_gl_loss(x[:, indices]))
    if len(indices)!=d or not np.isfinite(loss):
        raise ValueError('Search did not produce a finite full-rank subset')
    result.update(status='complete', indices=indices, loss=loss,
                  swap_seconds=time.perf_counter()-started)
    return result
