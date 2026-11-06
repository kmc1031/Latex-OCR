"""Static scientific plots of this one run; not model comparisons."""
import os,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
os.environ['MPLCONFIGDIR']=str(ROOT/'plot_cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
history=json.loads((ROOT/'training_history.json').read_text(encoding='utf-8'));test=json.loads((ROOT/'mathwriting_test.json').read_text(encoding='utf-8'));diag=json.loads((ROOT/'diagnostics_crohme.json').read_text(encoding='utf-8'));chosen=json.loads((ROOT/'frozen_selection.json').read_text(encoding='utf-8'))
fig,axs=plt.subplots(2,2,figsize=(11,7),layout='constrained')
epochs=[r['epoch'] for r in history]
axs[0,0].plot(epochs,[r['loss'] for r in history]);axs[0,0].set(title='Training loss (sequence + auxiliary after warmup)',xlabel='Epoch',ylabel='Loss');axs[0,0].grid(alpha=.2)
axs[0,1].plot(epochs,[100*r['mathwriting_validation']['exact_match'] for r in history],label='Validation');axs[0,1].axvline(chosen['epoch'],ls='--',color='gray',label='Selected epoch');axs[0,1].axhline(test['exact_match']*100,color='tomato',label='Frozen-model test');axs[0,1].set(title='Real MathWriting full exact match',xlabel='Epoch',ylabel='Percent');axs[0,1].legend();axs[0,1].grid(alpha=.2)
seen=json.loads((ROOT/'seen_training_diagnostic.json').read_text(encoding='utf-8'));axs[1,0].bar(['Seen train\n(n=100)','Validation\n(n=363)','New test\n(n=317)'],[100*seen['exact_match'],100*chosen['selection'][0],100*test['exact_match']],color=['#94a3b8','#38bdf8','#0284c7']);axs[1,0].set(title='Generalization gap (seen train: post-hoc diagnosis)',ylabel='Percent',ylim=(0,100));axs[1,0].grid(axis='y',alpha=.2)
axs[1,1].bar(['Auto','GT regions','GT symbols'],[100*diag[k] for k in ['auto_exact_match','gt_regions_exact_match','oracle_exact_match']],color=['#94a3b8','#38bdf8','#0284c7']);axs[1,1].set(title='Auxiliary CROHME diagnosis (252 formulas)',ylabel='Percent',ylim=(0,100));axs[1,1].grid(axis='y',alpha=.2)
fig.suptitle(f"ONE frozen CNN-LSTM experiment | primary test: {test['correct_count']}/{test['n']} exact\nDigital handwriting; notebook photographs not evaluated",fontsize=12);fig.savefig(ROOT/'summary.png',dpi=160);print(ROOT/'summary.png')
