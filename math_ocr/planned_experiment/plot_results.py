"""Static scientific plots of measured results, not predicted scores."""
import json,os
from pathlib import Path
os.environ['MPLCONFIGDIR']=str(Path(__file__).resolve().parent/'plot_cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import run as e

r=json.loads((e.ROOT/'results.json').read_text());seeds=[42,43,44]
fig,axs=plt.subplots(2,2,figsize=(10,7),layout='constrained')
def bars(ax,names,key,title):
    values=[[100*r['runs'][f'{name}_{s}']['test'][key] for s in seeds] for name in names]
    ax.bar(names,[np.mean(x) for x in values],yerr=[np.std(x,ddof=1) for x in values],capsize=4,color=['#94a3b8','#38bdf8','#0284c7'][:len(names)])
    ax.set(title=title,ylabel='Percent',ylim=(0,100));ax.grid(axis='y',alpha=.2)
bars(axs[0,0],['C0','C1','C2'],'accuracy','Symbol accuracy: mean +/- SD (3 seeds)')
bars(axs[0,1],['L0','L1'],'exact_match','Full-expression exact match')
axs[1,0].bar(['Components','With merging'],[r['segmentation'][k]['f1'] for k in ['S0','S1']],color=['#94a3b8','#0284c7']);axs[1,0].set(title='Segmentation F1 (IoU >= 0.5)',ylim=(0,1));axs[1,0].grid(axis='y',alpha=.2)
matrix=np.array([[np.mean([100*r['runs'][f'{name}_{s}']['test']['by_type'][style]['exact_match'] for s in seeds]) for style in ['vertical','slash']] for name in ['Fvertical','Fslash']]);ax=axs[1,1];im=ax.imshow(matrix,vmin=0,vmax=100,cmap='Blues');ax.set(xticks=[0,1],xticklabels=['Vertical test','Slash test'],yticks=[0,1],yticklabels=['Vertical train','Slash train'],title='Fraction exact match (%)')
for i in range(2):
    for j in range(2):ax.text(j,i,f'{matrix[i,j]:.1f}',ha='center',va='center',color='black')
fig.colorbar(im,ax=ax,fraction=.046)
fig.suptitle('CROHME glyph composites: 23 classes, 450 test images\nNot notebook photographs; historical source glyphs were reused',fontsize=12)
fig.savefig(e.ROOT/'summary.png',dpi=160);print(e.ROOT/'summary.png')
