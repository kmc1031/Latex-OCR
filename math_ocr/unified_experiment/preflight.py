"""Executable correctness checks. No training or accuracy claims."""
import io,json
import torch
from engine import Model,ROOT,CFG,TOKENS
from syntax import normalize,parse,tokenize

torch.set_num_threads(1);torch.manual_seed(42)
checks={}
m=Model();checks['parameter_count']=sum(p.numel() for p in m.parameters())
x=torch.rand(7,1,32,32);g=torch.rand(7,8);previous=torch.tensor([[1,5,6],[1,7,0]])
z=m(x,g,[3,4],previous);assert list(z.shape)==[2,3,79]
loss=torch.nn.functional.cross_entropy(z.reshape(-1,79),torch.tensor([5,6,2,7,2,0]),ignore_index=0);loss.backward();assert m.cnn[0].weight.grad.abs().sum()>0
checks['sequence_loss_reaches_cnn']=True
m.eval()
with torch.inference_mode():before=m(x,g,[3,4],previous)
buf=io.BytesIO();torch.save(m.state_dict(),buf);buf.seek(0);loaded=Model();loaded.load_state_dict(torch.load(buf,weights_only=True));loaded.eval()
with torch.inference_mode():after=loaded(x,g,[3,4],previous)
assert torch.equal(before,after);checks['checkpoint_reload_exact']=True
for text in ['F=ma','x^{2}+a_{12}','a+b/c','(a+b)/c','a-(b-c)','x+(y+z)','0.50+O','a\\times (b+c)']:
    ts=tokenize(text);norm,identity=normalize(ts);assert repr(parse(norm))==identity
assert normalize(tokenize('(a+b)/c'))==normalize(tokenize('\\frac{a+b}{c}'))
assert normalize(tokenize('0+1'))!=normalize(tokenize('O+l'))
try:normalize(tokenize('x^{2'))
except ValueError:pass
else:raise AssertionError('Invalid braces were accepted')
checks.update({'token_shapes':True,'syntax_roundtrip':True,'fraction_view_equivalence':True,'case_and_digit_distinction':True,'invalid_braces_not_repaired':True,'passed':True,'note':'Correctness/gradient/reload checks only; weights are untrained and were not exported as an OCR model.'})
(ROOT/'preflight.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding='utf-8');print(checks)
