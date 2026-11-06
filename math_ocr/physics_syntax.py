"""A bounded LaTeX surface grammar for scalar/vector introductory physics.

This parses notation, not physical dimensions or algebraic equivalence. No eval,
CAS rewriting, or formula lookup is used to correct OCR predictions.
"""
import re

GREEK = dict(zip(
    'alpha beta gamma delta epsilon varepsilon zeta eta theta vartheta iota kappa lambda mu nu xi pi rho sigma tau upsilon phi varphi chi psi omega Gamma Delta Theta Lambda Xi Pi Sigma Upsilon Phi Psi Omega'.split(),
    'α β γ δ ϵ ε ζ η θ ϑ ι κ λ μ ν ξ π ρ σ τ υ ϕ φ χ ψ ω Γ Δ Θ Λ Ξ Π Σ Υ Φ Ψ Ω'.split()))
GLYPHS = {'\\'+k:v for k,v in GREEK.items()}
GLYPHS.update({'\\nabla':'∇','\\partial':'∂','\\infty':'∞','\\ell':'ℓ','\\hbar':'ℏ',
               '\\int':'∫','\\oint':'∮','\\sum':'∑','\\prod':'∏'})
FUNCTIONS = {'\\'+s for s in ['sin','cos','tan','ln','log','exp']}
WRAPPERS = {'\\vec':'vec','\\hat':'hat','\\bar':'bar','\\dot':'dot','\\ddot':'ddot','\\mathrm':'roman'}
OPERATORS = {'=':('eq',0), '\\approx':('approx',0), '\\neq':('neq',0),
             '\\leq':('leq',0),'\\geq':('geq',0),'\\propto':('propto',0),
             '+':('add',1),'-':('subtraction',1),'\\pm':('pm',1),
             '\\cdot':('dotproduct',2),'\\times':('crossproduct',2),'/':('div',2)}
OP_TEX = {name:token for token,(name,_) in OPERATORS.items()}


def tokenize(text):
    return re.findall(r'\\[a-zA-Z]+|\\.|[^\s]', text)


class Parser:
    def __init__(self,tokens,preserve_groups=False):
        self.tokens=[t for t in tokens if t not in ['\\left','\\right','\\,','\\!','\\;','\\:']]
        self.i=0
        self.preserve_groups=preserve_groups

    def grouped(self):
        if self.i>=len(self.tokens) or self.tokens[self.i]!='{':
            raise ValueError('Expected braces')
        self.i+=1;node=self.expression()
        if self.i>=len(self.tokens) or self.tokens[self.i]!='}':raise ValueError('Unclosed braces')
        self.i+=1
        return node

    def atom(self,allow_scripts=True):
        if self.i>=len(self.tokens):raise ValueError('Missing operand')
        t=self.tokens[self.i];self.i+=1
        if t in ['(','[','{']:
            node=self.expression()
            if self.i>=len(self.tokens) or self.tokens[self.i]!={'(':')','[':']','{':'}'}[t]:raise ValueError('Unclosed group')
            self.i+=1
            if self.preserve_groups and t in ['(','[']:node=('round' if t=='(' else 'square',node)
        elif t=='\\frac':node=('div',self.grouped(),self.grouped())
        elif t=='\\sqrt':node=('sqrt',self.grouped())
        elif t in WRAPPERS:node=(WRAPPERS[t],self.grouped())
        elif t in ['+','-']:node=('positive' if t=='+' else 'negative',self.atom())
        elif t.isdigit() or t=='.':
            value=t
            while self.i<len(self.tokens) and (self.tokens[self.i].isdigit() or self.tokens[self.i]=='.'):
                value+=self.tokens[self.i];self.i+=1
            if not re.fullmatch(r'\d+(?:\.\d+)?',value):raise ValueError('Invalid number')
            node=('symbol',value)
        elif t in GLYPHS or t in FUNCTIONS or (len(t)==1 and t.isascii() and t.isalpha()):node=('symbol',t)
        else:raise ValueError('Unsupported symbol: '+t)
        while allow_scripts and self.i<len(self.tokens) and self.tokens[self.i] in ['^','_']:
            op=self.tokens[self.i];self.i+=1
            # LaTeX permits one unbraced atom as well as the canonical group.
            if self.i<len(self.tokens) and self.tokens[self.i]=='{':child=self.grouped()
            elif self.i<len(self.tokens) and self.tokens[self.i].isdigit():
                child=('symbol',self.tokens[self.i]);self.i+=1
            else:child=self.atom(False)
            node=('sup' if op=='^' else 'sub',node,child)
        return node

    def expression(self,min_prec=0):
        node=self.atom()
        while self.i<len(self.tokens):
            token=self.tokens[self.i]
            if token in ['}',')',']']:break
            if token in OPERATORS:
                kind,prec=OPERATORS[token]
                if prec<min_prec:break
                self.i+=1;right=self.expression(prec+1)
            else:
                if min_prec>2:break
                kind='multiply';right=self.expression(3)
            node=(kind,node,right)
        return node


def parse(tokens,preserve_groups=False):
    parser=Parser(tokens,preserve_groups);tree=parser.expression()
    if parser.i!=len(parser.tokens):raise ValueError('Trailing tokens')
    return tree


def latex(tree):
    kind=tree[0]
    if kind=='symbol':return tree[1]+' ' if tree[1].startswith('\\') else tree[1]
    if kind in ['round','square']:return ('(' if kind=='round' else '[')+latex(tree[1])+(')' if kind=='round' else ']')
    if kind=='div':return '\\frac{'+latex(tree[1])+'}{'+latex(tree[2])+'}'
    if kind=='sqrt':return '\\sqrt{'+latex(tree[1])+'}'
    if kind in WRAPPERS.values():return next(k for k,v in WRAPPERS.items() if v==kind)+'{'+latex(tree[1])+'}'
    if kind in ['sup','sub']:return '{'+latex(tree[1])+'}'+('^' if kind=='sup' else '_')+'{'+latex(tree[2])+'}'
    if kind in ['negative','positive']:return ('-' if kind=='negative' else '+')+'('+latex(tree[1])+')'
    if kind=='multiply':return '{'+latex(tree[1])+'}{'+latex(tree[2])+'}'
    return '('+latex(tree[1])+OP_TEX[kind]+' '+latex(tree[2])+')'


def result(tokens):
    try:
        tree=parse(tokens,preserve_groups=True)
        return tree,latex(tree)
    except (ValueError,RecursionError):
        return None,None


def canonical(tree):
    """Normalize product/fraction grouping for layout comparison only.

    No cancellation, commutation, substitutions, or physical-law correction.
    E.g. G*(m/r) and (G*m)/r share an ordered numerator/denominator list.
    """
    if tree[0] in ['round','square']:return canonical(tree[1])
    if tree[0] in ['multiply','div']:
        numerator=[];denominator=[]
        def collect(node,in_denominator=False):
            if node[0] in ['round','square']:collect(node[1],in_denominator)
            elif node[0]=='multiply':
                collect(node[1],in_denominator);collect(node[2],in_denominator)
            elif node[0]=='div':
                collect(node[1],in_denominator);collect(node[2],not in_denominator)
            else:(denominator if in_denominator else numerator).append(canonical(node))
        collect(tree)
        return ('ratio',tuple(numerator),tuple(denominator))
    return tuple(canonical(child) if isinstance(child,tuple) else child for child in tree)
