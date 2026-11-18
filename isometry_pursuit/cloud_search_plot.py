"""Join saved solver and search stages for the manuscript's four pipelines."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


SEARCH_KEYS = dict(forward_full='forward_backward_full', backward_full='backward_swap_full',
                   forward_restricted='forward_backward_restricted', backward_restricted='backward_swap_restricted')


def join_full_synthetic_results(root, bundle):
    """Validate and join the fixed 130-condition, 25-replicate Lambda grid.

    Local results supply identifiers and data-generation metadata only. Their
    solver/search outputs and timings never enter the returned observations.
    """
    root, bundle = Path(root), Path(bundle)
    def read_runs(path):
        saved = json.loads(path.read_text())
        return saved['runs'] if isinstance(saved, dict) else saved
    original = read_runs(root/'results/identity_swap_comparison_25x_D64/identity_experiment_runs.json')
    by_run = {r['run_id']: r for r in original}
    conditions = {}
    for r in original:
        key = (r['D'], r['corrupting_D'], r['num_corrupting_matrices'])
        conditions.setdefault(key, set()).add(r['replicate'])
    if (len(original) != 3250 or len(by_run) != 3250 or len(conditions) != 130
            or any(v != set(range(25)) for v in conditions.values())
            or {r['D'] for r in original} != {2, 4, 8, 16, 32, 64}):
        raise ValueError('Expected the original 130 conditions with 25 fixed replicates each')
    solvers = read_runs(bundle/'records.json')
    searches = read_runs(bundle/'search_records.json')
    aliases = read_runs(bundle/'aliases.json')
    by_solver = {r['job_id']: r for r in solvers}
    by_search = {r['job_id']: r for r in searches}
    links = {(a['optimizer_job_id'], a['search']): a['search_job_id'] for a in aliases}
    if (len(solvers) != 3250 or len(by_solver) != 3250
            or {r['saved_run_id'] for r in solvers} != set(by_run)):
        raise ValueError('Optimizer bundle must contain exactly the 3250 fixed matrices')
    if len(by_search) != len(searches) or len(links) != len(aliases):
        raise ValueError('Duplicate search identifiers or aliases')
    expected_links = {(r['job_id'], key) for r in solvers for key in SEARCH_KEYS}
    if set(links) != expected_links or set(links.values()) != set(by_search):
        raise ValueError('Expected all four pipelines on every matrix; no partial main figures')
    rows, initializers = [], []
    for solver in solvers:
        old = by_run[solver['saved_run_id']]
        settings = solver['settings']
        if settings['solver'] != 'scs' or settings['scale'] != .1:
            raise ValueError('Main figures require only SCS with scale 0.1')
        for key in ('D', 'corrupting_D', 'num_corrupting_matrices', 'replicate'):
            if solver[key] != old[key]:
                raise ValueError(f'Optimizer metadata mismatch: {key}')
        if solver['data_seed'] != old['seed'] or solver['shape'] != [old['D'], old['total_columns']]:
            raise ValueError('Optimizer matrix seed or shape differs from the fixed grid')
        if (not solver.get('data_sha256') or not np.isfinite(solver['basis_pursuit_time_seconds'])
                or solver['basis_pursuit_time_seconds'] <= 0
                or solver['basis_pursuit_iterations'] <= 0):
            raise ValueError('Missing observed optimizer timing, iterations, or matrix hash')
        row = {k: old[k] for k in ('run_id', 'D', 'corrupting_D', 'num_corrupting_matrices', 'total_columns', 'replicate', 'seed')}
        row.update(shared_full_search_key=old['run_id'], data_sha256=solver['data_sha256'],
                   optimizer_settings=settings, solver_status=solver['status'], optimizer_job_id=solver['job_id'])
        for search, prefix in SEARCH_KEYS.items():
            result = by_search[links[(solver['job_id'], search)]]
            if (result['data_sha256'] != solver['data_sha256']
                    or result['data_seed'] != old['seed'] or result['shape'] != solver['shape']):
                raise ValueError('Search and optimizer used different matrices')
            kind = 'forward' if search.startswith('forward') else 'backward'
            candidates = (list(range(old['total_columns'])) if search.endswith('_full')
                          else solver['basis_pursuit_indices'])
            if (result['settings']['initialization'] != kind
                    or result['settings']['search_seed'] != old['seed']
                    or not result['settings'].get('search_on_raw')
                    or set(result['candidate_indices']) != set(candidates)):
                raise ValueError('Search initialization or candidate support mismatch')
            status = result['status']
            if status not in ('complete', 'time_limit', 'insufficient_support'):
                raise ValueError(f'Unresolved search status {status}; do not publish partial main figures')
            if status == 'complete' and (result.get('loss') is None or not np.isfinite(result['loss'])):
                raise ValueError('Completed search has no finite loss')
            row[prefix+'_loss'] = result.get('loss') if status == 'complete' else None
            row[prefix+'_indices'] = result.get('indices')
            row[prefix+'_error'] = None if status == 'complete' else status
            initialization_status=result.get('initialization_status', 'complete' if status=='complete' else None)
            row[prefix+'_initialization_status'] = initialization_status
            if search.endswith('_full'):
                if ('initialization_seconds' not in result or 'initialization_work_proxy' not in result
                        or initialization_status not in ('complete', 'time_limit')):
                    raise ValueError('Missing observed pure-initializer timing')
                initializers.append({**result, 'saved_run_id': old['run_id'],
                                     'initialization_status': initialization_status})
        rows.append(row)
    return solvers, initializers, rows


def join_search_results(root, search_overrides=None):
    root=Path(root)
    saved=json.loads((root/'results/cloud/large-grid-20261116-001/records.json').read_text())
    solvers=saved['runs'] if isinstance(saved,dict) else saved
    searches=json.loads((root/'results/cloud/large-search-20261116-001/records.json').read_text())['runs']
    aliases=json.loads((root/'results/cloud/large-search-setup/aliases.json').read_text())
    by_search={r['job_id']:r for r in searches}
    if search_overrides is not None:
        updated=json.loads(Path(search_overrides).read_text())['runs']
        if {r['job_id'] for r in updated}!={r['job_id'] for r in searches if r['status']=='work_budget'}:
            raise ValueError('Replacement search batch is incomplete')
        for result in updated:
            old=by_search[result['job_id']]
            if old['status']!='work_budget' or result['data_sha256']!=old['data_sha256'] or result['candidate_indices']!=old['candidate_indices']:
                raise ValueError('Unexpected replacement search result')
            by_search[result['job_id']]=result
        searches=list(by_search.values())
    links={(a['optimizer_job_id'],a['search']):a['search_job_id'] for a in aliases}
    expected={a['search_job_id'] for a in aliases}
    if set(by_search)!=expected:
        raise ValueError('Search batch is incomplete; do not publish partial figures')
    original=json.loads((root/'results/identity_swap_comparison_25x_D64/identity_experiment_runs.json').read_text())['runs']
    by_run={r['run_id']:r for r in original}
    merged=[r for r in original if r['D']<32]
    cloud=[]
    for solver in solvers:
        old=by_run[solver['saved_run_id']]
        row={k:old[k] for k in ('run_id','D','corrupting_D','num_corrupting_matrices','total_columns','replicate','seed')}
        row.update(shared_full_search_key=solver['data_sha256'],optimizer_settings=solver['settings'],
                   solver_status=solver['status'],optimizer_job_id=solver['job_id'])
        for search,prefix in SEARCH_KEYS.items():
            result=by_search[links[(solver['job_id'],search)]]
            if result['data_sha256']!=solver['data_sha256']:
                raise ValueError('Search and optimizer used different matrices')
            row[prefix+'_loss']=result.get('loss') if result['status']=='complete' else None
            row[prefix+'_indices']=result.get('indices')
            row[prefix+'_error']=None if result['status']=='complete' else result['status']
        cloud.append(row)
    folder=root/'results/cloud/large-search-20261116-001'
    combined=merged+cloud
    (folder/'figure3_search_runs.json').write_text(json.dumps({'runs':combined},indent=2)+'\n')
    (folder/'paired_cloud_searches.json').write_text(json.dumps({'runs':cloud},indent=2)+'\n')
    return solvers,searches,cloud


def plot_updated_runtime(root,solvers,searches,main=False,ax=None):
    root=Path(root);folder=root/'results/complexity_runtime'
    points=[]
    if not main:
        old=pd.read_csv(folder/'complexity_runtime_summary.csv')
        early=pd.read_csv(folder/'complexity_runtime_identity_early_stopped_summary.csv')
        old=pd.concat([old[(old.D<32)&(old.method!='isometry_pursuit')],early[early.D<32]],ignore_index=True)
        points=[dict(method=r.method,D=int(r.D),work=r.arithmetic_work_proxy,
                     seconds=r.median_runtime_seconds,capped=False,source='local benchmark') for r in old.itertuples()]
    sample_hashes={r['data_sha256'] for r in solvers}
    for r in solvers:
        points.append(dict(method='isometry_pursuit' if r['settings']['solver']=='scs' else 'isometry_pursuit_admm',
                           D=r['D'],work=r['basis_pursuit_iterations']*r['D']**2*r['shape'][1],
                           seconds=r['basis_pursuit_time_seconds'],capped=r['status']!='solved',source='Lambda',
                           data_sha256=r['data_sha256'],saved_run_id=r.get('saved_run_id')))
    for r in searches:
        if ('initialization_seconds' in r and r['candidate_indices']==list(range(r['P']))
                and (not main or r['data_sha256'] in sample_hashes)):
            points.append(dict(method='forward_greedy' if r['settings']['initialization']=='forward' else 'backward_greedy',
                               D=r['D'],work=r['initialization_work_proxy'],seconds=r['initialization_seconds'],
                               capped=r.get('initialization_status')=='time_limit',source='Lambda',
                               data_sha256=r['data_sha256'],saved_run_id=r.get('saved_run_id')))
    colors={'forward_greedy':'#4c78a8','backward_greedy':'#54a24b',
            'isometry_pursuit':'#f58518','isometry_pursuit_admm':'#b279a2'}
    labels={'forward_greedy':'Forward greedy','backward_greedy':'Backward greedy',
            'isometry_pursuit':'Isometry Pursuit (SCS)','isometry_pursuit_admm':'Isometry Pursuit (ADMM)'}
    if main:
        labels['isometry_pursuit']='Isometry Pursuit'
        colors['isometry_pursuit_admm']=colors['isometry_pursuit']
    markers={2:'o',4:'s',8:'^',16:'D',32:'P',64:'h'}
    if main:
        markers={d:marker for d,marker in markers.items() if d in {r['D'] for r in points}}
    standalone=ax is None
    if standalone:
        fig,ax=plt.subplots(figsize=(8.2,5.8));fig.subplots_adjust(bottom=.17)
    else:
        fig=ax.figure
    groups={}
    for r in points:
        if r['work']<=0 or r['seconds']<=0:continue
        groups.setdefault((r['method'],r['D'],r['capped']),[]).append(r)
    for (method,dimension,capped), group in groups.items():
        style=dict(facecolors='none',edgecolors=colors[method]) if capped else dict(color=colors[method])
        ax.scatter([r['work'] for r in group],[r['seconds'] for r in group],
                   marker=markers[dimension],s=34,alpha=.25 if main and not capped else .5,linewidths=.8,**style)
    if not main:
        real=pd.read_csv(folder/'complexity_runtime_real_summary.csv')
        re=pd.read_csv(folder/'complexity_runtime_early_stopped_ip_summary.csv')
        real=pd.concat([real[real.method!='isometry_pursuit'],re],ignore_index=True)
        for r in real.itertuples():
            ax.scatter(r.arithmetic_work_proxy,r.median_runtime_seconds,marker='*',s=130,
                       color=colors[r.method],edgecolors='black',linewidths=.6,zorder=5)
            ax.annotate(r.dataset,(r.arithmetic_work_proxy,r.median_runtime_seconds),xytext=(4,5),
                        textcoords='offset points',fontsize=7)
    cutoff=245.36574106800015
    ax.axhline(cutoff,color=colors['backward_greedy'],linestyle='--',alpha=.8,linewidth=1.25)
    ax.annotate('backward time limit',xy=(.98,cutoff),xycoords=ax.get_yaxis_transform(),
            xytext=(0,7),textcoords='offset points',
            ha='right',va='bottom',fontsize=8,color=colors['backward_greedy'],
            bbox=dict(facecolor='white',edgecolor='none',alpha=.75,pad=1))
    legend_methods=[m for m in colors if not (main and m=='isometry_pursuit_admm')]
    first=ax.legend(handles=[Line2D([],[],color=colors[m],marker='o',linestyle='none',label=labels[m]) for m in legend_methods],
                    loc='upper left',fontsize=8,title='Method',title_fontsize=9)
    ax.add_artist(first)
    ax.legend(handles=[Line2D([],[],color='.35',marker=marker,linestyle='none',label=f'D={d}') for d,marker in markers.items()]
              +([] if main else [Line2D([],[],color='.35',marker='*',linestyle='none',label='Real dataset')]),
              loc='lower right',fontsize=8,title='Dimension',title_fontsize=9)
    ax.set(xscale='log',yscale='log',xlabel='Arithmetic-work proxy',ylabel='Wall-clock time (s)',
           title='Arithmetic work versus runtime')
    ax.grid(True,which='both',alpha=.2)
    if not standalone:
        return points
    fig.text(.5,.025,'Pure initializers; swap refinement omitted.',ha='center',fontsize=9)
    path=root/'figures/complexity_runtime/complexity_vs_runtime_overlay_with_early_stopping'
    if main:
        path=root/'figures/complexity_runtime/complexity_vs_runtime_main'
    fig.savefig(path.with_suffix('.png'),dpi=200);fig.savefig(path.with_suffix('.pdf'));plt.close(fig)
    pd.DataFrame(points).to_csv(folder/('main_runtime_points.csv' if main else 'cloud_search_figure_points.csv'),index=False)

