"""Reproduce the five grayscale simulation figures from saved numerical results."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch


def plot_saved(data, output):
    DATA, OUT = Path(data), Path(output)
    OUT.mkdir(parents=True, exist_ok=True)
    with np.load(DATA / 'normal_rep000.npz', allow_pickle=False) as z:
        example = {k: z[k] for k in z.files}
    with np.load(DATA / 'block_rep000_1110.npz', allow_pickle=False) as z:
        block = {k: z[k] for k in z.files}
    group = pd.read_csv(DATA / 'group_summary.csv')
    plt.rcParams.update({'font.size':10,'axes.titlesize':11,'axes.labelsize':10,
        'legend.fontsize':9,'xtick.labelsize':9,'ytick.labelsize':9,'font.family':'DejaVu Sans',
        'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':300})
    COLORS={'BALM':'#191919','ALMA':'#696969','TBM':'#A0A0A0'}
    METHOD={'coupled':'BALM','alma':'ALMA','tbm_r':'TBM'}
    def finish(fig, name):
        fig.savefig(OUT / name, dpi=300, facecolor='white')
        fig.savefig(OUT / Path(name).with_suffix('.pdf'), facecolor='white')
        plt.close(fig)
    def edge_matrix(q, n=68):
        a=np.zeros((n,n));i,j=np.triu_indices(n,1);a[i,j]=q;a[j,i]=q
        np.fill_diagonal(a,np.nan)
        return a
    scale=max(np.abs(example['Q_true']).max(),np.abs(example['Q_mean']).max())
    fig,axes=plt.subplots(2,3,figsize=(7.4,4.65),layout='constrained')
    for r,key in enumerate(['Q_true','Q_mean']):
        for m in range(3):
            ax=axes[r,m]
            im=ax.imshow(edge_matrix(example[key][m]),cmap='gray_r',vmin=-scale,vmax=scale,interpolation='nearest')
            if r==0:ax.set_title(f'Template {m+1}')
            ax.set_xticks([0,33,67],['1','34','68']);ax.set_yticks([0,33,67],['1','34','68'])
            if r==1:ax.set_xlabel('Node')
            if m==0:ax.set_ylabel(('Truth' if r==0 else 'BALM')+'\u2009\u2014\u2009Node')
    fig.colorbar(im,ax=axes,shrink=.86,pad=.025,label='Signed strength')
    finish(fig,'true_estimated_templates.png')

    keys=['truth_probability','BALM_class_mean','ALMA_class_mean','TBM_class_mean']
    rowlabels=['Truth\nprobability','BALM\nplug-in probability','ALMA\nreconstruction score','TBM\nprobability estimate']
    order_nodes=block['node_order']
    assert all(np.min(block[k])>=0 and np.max(block[k])<=1 for k in keys)
    fig,axes=plt.subplots(4,3,figsize=(7.4,8.35),layout='constrained')
    for r,key in enumerate(keys):
        for m in range(3):
            ax=axes[r,m]
            a=edge_matrix(block[key][m])[np.ix_(order_nodes,order_nodes)]
            im=ax.imshow(a,cmap='gray_r',vmin=0,vmax=1,interpolation='nearest')
            if r==0:ax.set_title(f'Class {m+1}')
            ax.set_xticks([0,33,67],['1','34','68']);ax.set_yticks([0,33,67],['1','34','68'])
            if r==3:ax.set_xlabel('Node (common order)')
            if m==0:ax.set_ylabel(rowlabels[r],fontsize=10,labelpad=8)
    fig.colorbar(im,ax=axes,orientation='horizontal',fraction=.025,pad=.015,shrink=.8,
                 label='Binary probability / reconstruction score',ticks=[0,.25,.5,.75,1])
    finish(fig,'block_binary_patterns.png')

    W=example['W_true'];E=example['W_mean']
    order=np.lexsort((-W.max(1),W.argmax(1)))
    fig=plt.figure(figsize=(7.4,3.35),layout='constrained')
    gs=fig.add_gridspec(1,4,width_ratios=[1,1,.10,1.65])
    heat=[fig.add_subplot(gs[0,k]) for k in [0,1]]
    for ax,a,title in [(heat[0],W,'True memberships'),(heat[1],E,'BALM posterior means')]:
        im=ax.imshow(a[order],aspect='auto',cmap='gray_r',vmin=0,vmax=1,interpolation='nearest')
        ax.set_xticks([0,1,2],['1','2','3']);ax.set_xlabel('Template');ax.set_title(title,fontsize=10)
        ax.set_yticks([0,49,99],['1','50','100'])
    heat[0].set_ylabel('Subject (common order)')
    fig.colorbar(im,cax=fig.add_subplot(gs[0,2]),label='Weight',ticks=[0,.5,1])
    ax=fig.add_subplot(gs[0,3])
    for m,c in enumerate(COLORS.values()):
        ax.scatter(W[:,m],E[:,m],s=14,alpha=.7,color=c,marker=['o','s','^'][m],label=f'Template {m+1}',edgecolors='#191919',linewidths=.3)
    ax.plot([0,1],[0,1],color='#555555',lw=.9,ls='--')
    ax.set(xlabel='True weight',ylabel='Estimated weight',xlim=(-.025,1.025),ylim=(-.025,1.025),title='All 300 weights')
    ax.set_xticks([0,.5,1]);ax.set_yticks([0,.5,1]);ax.set_aspect('equal')
    ax.legend(frameon=False,fontsize=8,loc='upper left',handlelength=.6,borderpad=.2,labelspacing=.3)
    finish(fig,'true_estimated_memberships.png')

    labels={'True dominant class':W.argmax(1),'BALM':E.argmax(1),'ALMA':example['alma_labels'],'TBM':example['tbm_r_labels']}
    fig,axes=plt.subplots(1,4,figsize=(7.4,2.65),sharey=True)
    for k,(ax,(name,l)) in enumerate(zip(axes,labels.items())):
        v=l[order]
        ax.imshow(v[:,None]==v[None,:],cmap=ListedColormap(['#F0F0F0','#333333']),vmin=0,vmax=1,interpolation='nearest')
        ax.set_title(name,fontsize=10);ax.set_xlabel('Subject')
        ax.set_xticks([0,49,99],['1','50','100'],fontsize=8)
        ax.set_yticks([0,49,99],['1','50','100'])
        if k==0:ax.set_ylabel('Subject')
    fig.legend(handles=[Patch(facecolor='#333333',label='Same class'),Patch(facecolor='#F0F0F0',edgecolor='#CCCCCC',label='Different classes')],
               loc='lower center',ncol=2,frameon=False,bbox_to_anchor=(.5,.015))
    fig.subplots_adjust(left=.10,right=.97,top=.86,bottom=.32,wspace=.15)
    finish(fig,'matched_subject_grouping.png')

    rows=[(['alpha01','normal_fresh','alpha3'],[r'$\alpha=0.1$',r'$\alpha=1$',r'$\alpha=3$'],'Membership concentration'),
          (['normal_fresh','n30','L200','K2'],['Reference',r'$n=30$',r'$L=200$',r'$K=2$'],'Network size, subjects, and rank'),
          (['normal_fresh','weak_signal','high_noise'],['Reference','Weak signal','High noise'],'Signal and noise')]
    fig,axes=plt.subplots(3,2,figsize=(7.4,7.15))
    markers={'BALM':'o','ALMA':'s','TBM':'^'}
    for r,(settings,ticks,title) in enumerate(rows):
        for c,metric in enumerate(['AUC','ARI']):
            ax=axes[r,c]
            for j,(key,label) in enumerate(METHOD.items()):
                d=group[group.method==key].set_index('setting').loc[settings]
                ax.errorbar(np.arange(len(settings))+(j-1)*.15,d[metric+'_mean'],yerr=d[metric+'_sd'],
                    fmt=markers[label],markersize=5,capsize=3,elinewidth=1.1,color=COLORS[label],label=label)
            ax.set_xticks(range(len(settings)),ticks)
            ax.set_xlim(-.45,len(settings)-.55)
            ax.set_ylabel(metric);ax.grid(axis='y',alpha=.22)
            ax.set_title(f'({chr(97+r*2+c)}) '+title,loc='left',fontsize=10)
            if metric=='AUC':
                ax.set_ylim(.45,.85);ax.set_yticks([.5,.6,.7,.8])
            else:
                ax.set_ylim(-.08,1.08);ax.set_yticks([0,.25,.5,.75,1])
            # Ensure no uncertainty interval is cropped by the fixed common scale.
            for key in METHOD:
                d=group[group.method==key].set_index('setting').loc[settings]
                assert (d[metric+'_mean']-d[metric+'_sd']).min()>=ax.get_ylim()[0]
                assert (d[metric+'_mean']+d[metric+'_sd']).max()<=ax.get_ylim()[1]
    handles,labs=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labs,loc='upper center',ncol=3,frameon=False,bbox_to_anchor=(.5,.995))
    fig.subplots_adjust(left=.10,right=.985,bottom=.065,top=.92,hspace=.53,wspace=.28)
    finish(fig,'combined_performance.png')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('results/simulation'))
    parser.add_argument('--output', type=Path, default=Path('outputs/simulation_figures'))
    args = parser.parse_args()
    plot_saved(args.data, args.output)
    print(f'Saved five PNG/PDF figures to {args.output}')


if __name__ == '__main__':
    main()
