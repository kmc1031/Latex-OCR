import json
from pathlib import Path
import unittest
import numpy as np
from PIL import Image
import torch
from model import FormulaOCR,ModelConfig
from train_physics import initialize

ROOT=Path(__file__).resolve().parent


class FixedLocal(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.old=torch.load(ROOT/'runs/physics_handwriting_v1/best.pt',map_location='cpu',weights_only=False)
        row=json.loads((ROOT/'data_handwriting/val.jsonl').read_text(encoding='utf-8').splitlines()[0])
        cls.image=torch.from_numpy((255-np.array(Image.open(ROOT/'data_handwriting'/row['image']))).astype(np.float32)/255)[None,None]
        cls.widths=torch.tensor([cls.image.shape[-1]])
        lookup={w:i for i,w in enumerate(cls.old['vocab'])}
        cls.tokens=torch.tensor([[1]+[lookup[t] for t in row['tokens']]])

    def setUp(self):
        self.model=initialize(self.old,self.old['vocab'],auxiliary=True,local_features=96).eval()

    def test_zero_initialized_local_branch_preserves_transferred_function(self):
        baseline=FormulaOCR(ModelConfig(**self.old['config'])).eval();baseline.load_state_dict(self.old['model'])
        with torch.inference_mode():
            expected=baseline(self.image,self.widths,self.tokens)
            actual=self.model(self.image,self.widths,self.tokens)
        torch.testing.assert_close(expected,actual,atol=3e-5,rtol=1e-5)

    def test_local_beam_one_matches_free_greedy_without_target_length(self):
        greedy=self.model.generate(self.image,self.widths,40)
        paths,_=self.model.generate_candidates(self.image,self.widths,40,1)
        self.assertTrue(torch.equal(greedy,paths[:,0,:]))

    def test_local_training_has_finite_gradients(self):
        self.model.train()
        outputs,presence,length=self.model.forward_with_aux(self.image,self.widths,self.tokens)
        loss=outputs.square().mean()+presence.square().mean()+length.square().mean()
        loss.backward()
        self.assertTrue(torch.isfinite(self.model.decoder.weight_ih_l0.grad).all())
        self.model.eval();self.model.zero_grad()

    def test_deeper_raw_context_transfer_preserves_logits_and_free_decoding(self):
        # Exercise a genuinely active local branch, rather than only the zeros
        # used when first transferring from the global-context model.
        with torch.no_grad():self.model.decoder.weight_ih_l0[:,-96:].normal_(0,.01)
        package={**self.old,'config':self.model.config_dict(),'model':self.model.state_dict()}
        deeper=initialize(package,package['vocab'],auxiliary=True,local_features=96,
                          raw_context=True,refinement_blocks=2).eval()
        with torch.inference_mode():
            expected=self.model(self.image,self.widths,self.tokens)
            actual=deeper(self.image,self.widths,self.tokens)
        torch.testing.assert_close(expected,actual,atol=5e-5,rtol=1e-5)
        greedy=deeper.generate(self.image,self.widths,40)
        paths,_=deeper.generate_candidates(self.image,self.widths,40,1)
        self.assertTrue(torch.equal(greedy,paths[:,0,:]))


if __name__=='__main__':unittest.main()
