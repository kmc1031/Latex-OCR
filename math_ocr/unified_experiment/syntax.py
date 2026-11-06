"""Syntactic normalization only: no algebra, lookup, or invalid-brace repair."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from physics_syntax import parse,tokenize

OPS={'eq':('=',0),'add':('+',1),'subtraction':('-',1),'multiply':('',2),'crossproduct':('\\times ',2)}

def serialize(node):
    k=node[0]
    if k=='symbol':return node[1]
    if k=='div':return '\\frac{'+serialize(node[1])+'}{'+serialize(node[2])+'}'
    if k in ['sup','sub']:
        base=serialize(node[1]);base='('+base+')' if node[1][0] in OPS else base
        return base+('^' if k=='sup' else '_')+'{'+serialize(node[2])+'}'
    if k in ['positive','negative']:return ('+' if k=='positive' else '-')+'('+serialize(node[1])+')'
    if k not in OPS:raise ValueError('unsupported structure '+k)
    op,prec=OPS[k];a,b=node[1:];left=serialize(a);right=serialize(b)
    if a[0] in OPS and OPS[a[0]][1]<prec:left='('+left+')'
    if b[0] in OPS and OPS[b[0]][1]<=prec:right='('+right+')'
    if k=='multiply' and a[0]==b[0]=='symbol' and a[1][-1:].isdigit() and b[1][:1].isdigit():right='('+right+')'
    return left+op+right

def normalize(tokens):
    tree=parse(tokens);result=tokenize(serialize(tree))
    if parse(result)!=tree:raise ValueError('normalizer changed syntax tree')
    return result,repr(tree)
