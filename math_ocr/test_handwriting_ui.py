"""Check the exported handwriting package through the running local UI API."""
import io
import json
from pathlib import Path
import unittest
from urllib.request import Request,urlopen

from PIL import Image
import torch
from physics_syntax import canonical,parse

ROOT=Path(__file__).resolve().parent
URL='http://127.0.0.1:8765'


def request(path,data=None):
    headers={'Content-Type':'image/png'} if data is not None else {}
    with urlopen(Request(URL+path,data=data,headers=headers),timeout=60) as response:
        return response.read()


class HandwritingUI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.status=json.loads(request('/api/status'))
        package=torch.load(ROOT/'physics/best_model.pt',map_location='cpu',weights_only=False)
        rows=[json.loads(x) for x in (ROOT/'data_handwriting/val.jsonl').read_text(encoding='utf-8').splitlines()]
        cls.expected={f'handwriting_{i}':next(r['tokens'] for r in rows if r['ink_id']==demo['ink_id'])
                      for i,demo in enumerate(package['demo_samples'])}

    def test_default_profile_and_vocabulary(self):
        self.assertEqual(self.status['profile'],'physics')
        self.assertEqual(self.status['preprocessing'],'handwriting-v1')
        for symbol in [r'\pi',r'\omega',r'\vec','F','0',r'\frac']:
            self.assertIn(symbol,self.status['vocabulary'])
        self.assertTrue({'physics','legacy'}<={x['id'] for x in self.status['profiles']})

    def test_all_validation_demonstrations_use_the_real_inference_pipeline(self):
        for sample in self.status['samples']:
            with self.subTest(sample=sample['id']):
                raw=request('/api/sample/'+sample['id']+'?profile=physics')
                prediction=json.loads(request('/api/recognize?profile=physics',raw))
                self.assertTrue(prediction['terminated'])
                self.assertTrue(prediction['syntax_valid'])
                self.assertEqual(canonical(parse(prediction['tokens'])),canonical(parse(self.expected[sample['id']])))
                self.assertEqual(prediction['prepared_size'][1],64)
                self.assertLessEqual(prediction['prepared_size'][0],512)

    def test_handwritten_input_on_larger_canvas_is_accepted(self):
        sample=self.status['samples'][0]['id']
        original=Image.open(io.BytesIO(request('/api/sample/'+sample+'?profile=physics'))).convert('RGB')
        original=original.resize((original.width*3,original.height*3))
        canvas=Image.new('RGB',(original.width+100,original.height+80),'white')
        canvas.paste(original,(50,40));buffer=io.BytesIO();canvas.save(buffer,'PNG')
        result=json.loads(request('/api/recognize?profile=physics',buffer.getvalue()))
        self.assertIn('tokens',result)
        self.assertTrue(result['prepared_image'].startswith('data:image/png;base64,'))
        self.assertEqual(result['prepared_size'][1],64)


if __name__=='__main__':unittest.main(verbosity=2)
