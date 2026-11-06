"""Checkpoint and decoder invariants, separate from accuracy measurements."""
import json
from pathlib import Path
import unittest
import numpy as np
from PIL import Image
import torch
from model import FormulaOCR,ModelConfig
from train_physics import initialize

ROOT=Path(__file__).resolve().parent


class ModelChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        cls.old=torch.load(ROOT/'comparison/best_model.pt',map_location='cpu',weights_only=False)
        cls.vocab=json.loads((ROOT/'data_physics/vocab.json').read_text(encoding='utf-8'))
        cls.model=initialize(cls.old,cls.vocab).eval()
        cls.image=torch.from_numpy((255-np.array(Image.open(ROOT/'comparison/images/slash/test_00002.png'))).astype(np.float32)/255)[None,None]
        cls.widths=torch.tensor([cls.image.shape[-1]])

    def test_transfer_preserves_existing_token_logits(self):
        legacy=FormulaOCR(ModelConfig(**self.old['config'])).eval();legacy.load_state_dict(self.old['model'])
        tokens=['<BOS>','y','_','{','6','}']
        old_ids=torch.tensor([[self.old['vocab'].index(t) for t in tokens]])
        new_ids=torch.tensor([[self.vocab.index(t) for t in tokens]])
        with torch.inference_mode():
            expected=legacy(self.image,self.widths,old_ids)
            actual=self.model(self.image,self.widths,new_ids)[:,:,[self.vocab.index(t) for t in self.old['vocab']]]
        torch.testing.assert_close(expected,actual,atol=2e-5,rtol=1e-5)

    def test_beam_one_matches_greedy(self):
        greedy=self.model.generate(self.image,self.widths,32)
        paths,_=self.model.generate_candidates(self.image,self.widths,32,beam_width=1)
        self.assertTrue(torch.equal(greedy,paths[:,0,:]))

    def test_new_symbols_have_distinct_learnable_embeddings(self):
        for token in ['\\pi','\\theta','\\Delta','0','F','\\vec','\\frac','\\mathrm']:
            self.assertIn(token,self.vocab)
        self.assertEqual(self.model.config.context_bins,8)
        self.assertTrue(self.model.embedding.weight.requires_grad)


if __name__=='__main__':unittest.main()
