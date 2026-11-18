"""Update Figure 3a/b from the completed solver and greedy-search stages."""
import argparse
import json
from pathlib import Path

from isometry_pursuit.cloud_search_plot import join_full_synthetic_results,join_search_results,plot_updated_runtime


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('.'))
    parser.add_argument('--search-overrides',type=Path)
    parser.add_argument('--full-grid-bundle',type=Path,help='Validated 3250-matrix Lambda bundle for the main figures')
    parser.add_argument('--audit-only',action='store_true',help='Validate and report without writing results or figures')
    parser.add_argument('--all-settings',action='store_true',help='Generate the supplementary full-grid runtime plot')
    parser.add_argument('--main-solver',choices=['scs','admm'],default='scs')
    parser.add_argument('--main-parameter',type=float,default=.1)
    args=parser.parse_args()
    if args.full_grid_bundle:
        if args.all_settings or args.main_solver != 'scs' or args.main_parameter != .1 or args.search_overrides:
            parser.error('The full main grid requires SCS scale 0.1 and already merged search records')
        solvers,searches,cloud=join_full_synthetic_results(args.root,args.full_grid_bundle)
        print(f'Validated {len(cloud)} fixed matrices: 130 conditions, 25 replicates, all six dimensions.')
        for prefix in ['forward_backward_full','backward_swap_full','forward_backward_restricted','backward_swap_restricted']:
            valid=[r for r in cloud if r[prefix+'_loss'] is not None]
            print(prefix,'complete',len(valid),'of',len(cloud),
                  'optimal loss',sum(abs(r[prefix+'_loss']-r['D'])<1e-6 for r in valid))
        if args.audit_only:
            return
        (args.full_grid_bundle/'figure3_main_search_runs.json').write_text(json.dumps({'runs':cloud},indent=2)+'\n')
        plot_updated_runtime(args.root,solvers,searches,main=True)
        return
    if args.audit_only:
        parser.error('--audit-only requires --full-grid-bundle')
    solvers,searches,cloud=join_search_results(args.root,args.search_overrides)
    if args.all_settings:
        plot_updated_runtime(args.root,solvers,searches)
    else:
        parameter='scale' if args.main_solver=='scs' else 'rho'
        solvers=[r for r in solvers if r['settings']['solver']==args.main_solver and r['settings'][parameter]==args.main_parameter]
        cloud=[r for r in cloud if r['optimizer_settings']['solver']==args.main_solver and r['optimizer_settings'][parameter]==args.main_parameter]
        if len(solvers)!=65 or len(cloud)!=65:
            raise ValueError('Expected one fixed setting on all 65 cloud matrices')
        folder=args.root/'results/cloud/large-search-20261116-001'
        main=cloud
        (folder/'figure3_main_search_runs.json').write_text(json.dumps({'runs':main},indent=2)+'\n')
        plot_updated_runtime(args.root,solvers,searches,main=True)
    print(f'Joined {len(solvers)} solver results and {len(searches)} unique searches.')
    for prefix in ['forward_backward_full','backward_swap_full','forward_backward_restricted','backward_swap_restricted']:
        valid=[r for r in cloud if r[prefix+'_loss'] is not None]
        print(prefix,'complete',len(valid),'of',len(cloud),
              'optimal loss',sum(abs(r[prefix+'_loss']-r['D'])<1e-6 for r in valid))


if __name__=='__main__':
    main()
