"""Analytic hypercube memory projection; no GPU allocation or throughput model."""
import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

MIB = 1 << 20
GIB = 1 << 30
PIXELS = 32 * 512 * 1024
COUNT_BYTES = 8  # One scalar event count per bin, no per-pixel count/mask.


def state_bytes(bins, replicas=2, sum_bytes=4):
    return bins * replicas * (PIXELS * sum_bytes + COUNT_BYTES)


def bin_capacity(device_gib, reserve_gib=8, replicas=2, sum_bytes=4):
    return max(0, int((device_gib-reserve_gib)*GIB) // state_bytes(1, replicas, sum_bytes))


def project(output):
    output.mkdir(parents=True, exist_ok=False)
    capacities = [dict(device_budget_gib=d, reserve_gib=8, replicas=r,
                       max_bins=bin_capacity(d, replicas=r),
                       first_over_budget_bin=bin_capacity(d, replicas=r)+1)
                  for d in (40, 80) for r in (1, 2, 4, 8)]
    examples = []
    for dims in ((30,), (10,10), (20,20), (30,30), (10,10,10), (100,100), (100,100,100)):
        bins = math.prod(dims)
        examples.append(dict(dimensions=list(dims), bins=bins,
                             two_replica_state_gib=state_bytes(bins)/GIB,
                             with_reserve_gib=8+state_bytes(bins)/GIB,
                             cpu_float64_result_gib=bins*PIXELS*8/GIB,
                             ideal_gpus_40=math.ceil(bins/bin_capacity(40)),
                             ideal_gpus_80=math.ceil(bins/bin_capacity(80))))
    # Independent hand-calculated boundary/reference checks, not GPU tests.
    assert PIXELS*2 == 32*MIB and PIXELS*4 == 64*MIB
    assert state_bytes(1000) == 125*GIB + 16000
    assert bin_capacity(40) == 255 and bin_capacity(80) == 575
    for row in capacities:
        budget = (row['device_budget_gib']-row['reserve_gib'])*GIB
        assert state_bytes(row['max_bins'], row['replicas']) <= budget
        assert state_bytes(row['max_bins']+1, row['replicas']) > budget
    report = dict(kind='analytic projection; not measured OOM or throughput',
                  raw_bytes=PIXELS*2, pixels=PIXELS, float32_sum_bytes_per_bin=PIXELS*4,
                  scalar_count_bytes_per_bin=COUNT_BYTES, reserve_gib=8,
                  capacity_unit='GiB, illustrative budgets; query actual device and account all users',
                  replicas='independent complete banks per physical GPU, e.g. BD processes times streams',
                  assumptions=['full dense image per bin', 'float32 sums except dtype comparison',
                               'fixed 8 GiB non-cube reserve for illustration; may need to grow with replicas',
                               'GPU sharding has balanced bin counts and two banks per owner GPU',
                               'no pixel masks/counts, variance, labels, allocator padding or extra output staging in cube term'],
                  checks='hand values and every integer capacity boundary passed',
                  capacities=capacities, examples=examples)
    (output/'projection.json').write_text(json.dumps(report, indent=2)+'\n')
    with (output/'examples.csv').open('w', newline='') as f:
        writer=csv.DictWriter(f, fieldnames=examples[0].keys());writer.writeheader();writer.writerows(examples)

    plt.rcParams.update({'font.size':10, 'axes.titlesize':12, 'axes.labelsize':10,
                         'figure.facecolor':'white', 'axes.spines.top':False, 'axes.spines.right':False})
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), layout='constrained')
    fig.suptitle('JF hypercube memory: bin dimensions multiply; stream replicas multiply again', fontsize=16)
    bins=np.unique(np.geomspace(1,100000,500).astype(np.int64))
    ax=axes[0,0]
    colors=['#0072B2','#D55E00','#009E73','#CC79A7']
    for r,color in zip((1,2,4,8),colors):
        ax.loglog(bins, state_bytes(bins,r)/GIB, label=f'{r} bank'+('s' if r>1 else '')+' / GPU', color=color, lw=2)
    for budget,color in ((32,'#555555'),(72,'#999999')):
        ax.axhline(budget,color=color,ls='--',lw=1)
        ax.text(1.3,budget*1.13,f'{budget} GiB cube allowance',fontsize=9,color=color)
    ax.scatter([255,575],[state_bytes(255)/GIB,state_bytes(575)/GIB],color=colors[1],s=30,zorder=5)
    ax.set(xlabel='Total logical bins = product of user bin dimensions', ylabel='Persistent cube state per GPU (GiB)',
           title='A. Dense replication: float32 sums + scalar counts', xlim=(1,100000))
    ax.grid(True,which='major',alpha=.2);ax.legend(loc='lower right',fontsize=9)

    ax=axes[0,1]
    dims=np.array([5,10,20,30,50,100,200])
    total=8+state_bytes(dims[:,None]*dims[None,:])/GIB
    mesh=ax.imshow(total,origin='lower',norm=LogNorm(vmin=8,vmax=6000),cmap='YlOrRd',aspect='auto')
    ax.set_xticks(range(len(dims)),dims);ax.set_yticks(range(len(dims)),dims)
    for y in range(len(dims)):
        for x in range(len(dims)):
            v=total[y,x]
            ax.text(x,y,f'{v:.1f}' if v<100 else f'{v:,.0f}',ha='center',va='center',fontsize=9,
                    color='white' if v>700 else 'black')
    ax.set(xlabel='Bins along user dimension 1',ylabel='Bins along user dimension 2',
           title='B. Two banks: total GiB including 8 GiB reserve')
    fig.colorbar(mesh,ax=ax,label='Projected total GPU memory (GiB)',shrink=.85)

    ax=axes[1,0]
    for device,color in ((40,colors[0]),(80,colors[1])):
        cap=bin_capacity(device)
        owners=(bins+cap-1)//cap
        ax.loglog(bins,owners,label=f'{device} GiB budget: {cap} bins / GPU',color=color,lw=2)
    ax.axvline(1000,color='#777777',ls=':',lw=1)
    ax.annotate('1,000 bins: 4 or 2 GPUs',xy=(1000,4),xytext=(1600,10),
                fontsize=9,arrowprops={'arrowstyle':'->','color':'#555555'})
    ax.set(xlabel='Total logical bins',ylabel='Minimum GPUs from memory alone',
           title='C. Ideal bin ownership: two banks per GPU',xlim=(1,100000),ylim=(.8,600))
    ax.grid(True,which='major',alpha=.2);ax.legend(loc='upper left',fontsize=9)

    ax=axes[1,1]
    x=np.arange(1,5)
    for n,color in ((10,colors[0]),(20,colors[1]),(50,colors[2])):
        ax.semilogy(x,8+state_bytes(n**x)/GIB,'o-',label=f'{n} bins per dimension',color=color,lw=2)
    for budget in (40,80):
        ax.axhline(budget,color='#777777',ls='--',lw=1)
        ax.text(3.6,budget*1.13,f'{budget} GiB',fontsize=9)
    ax.set_xticks(x)
    ax.set(xlabel='Number of user bin dimensions',ylabel='Projected total GPU memory (GiB)',
           title='D. Dimension growth: two banks + 8 GiB reserve')
    ax.grid(True,which='major',alpha=.2);ax.legend(loc='upper left',fontsize=9)
    fig.supxlabel('Assumptions: 32 MiB uint16 raw → 64 MiB float32 sum/bin; 8 GiB reserve is illustrative.\n'
                  'Sharding requires routing/ownership changes. Counts included; no throughput or actual-device OOM prediction.',fontsize=10)
    for suffix in ('png','pdf','svg'):
        fig.savefig(output/f'hypercube_memory.{suffix}',dpi=160)
    # Match repository whitespace conventions for the text SVG artifact.
    svg = output/'hypercube_memory.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    plt.close(fig)
    print(json.dumps({'capacities':capacities,'examples':examples},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    project(p.parse_args().output)
