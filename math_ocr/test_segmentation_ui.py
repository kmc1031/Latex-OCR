"""Actual image-only inference in the additional segmentation profile."""
import json,unittest
from urllib.request import Request,urlopen

URL='http://127.0.0.1:8765'
def get(path,data=None):
    headers={'Content-Type':'image/png'} if data is not None else {}
    with urlopen(Request(URL+path,data=data,headers=headers),timeout=60) as response:return response.read()

class SegmentationUI(unittest.TestCase):
    def test_profile_and_real_box_predictions(self):
        status=json.loads(get('/api/status?profile=segments'))
        self.assertEqual(status['preprocessing'],'segmentation-v1');self.assertIn('MLP',status['architecture'])
        sample=status['samples'][0]['id'];image=get('/api/sample/'+sample+'?profile=segments')
        answer=json.loads(get('/api/recognize?profile=segments',image))
        self.assertGreater(len(answer['boxes']),0);self.assertEqual(len(answer['boxes']),len(answer['labels']))
        self.assertTrue(answer['segmentation_overlay'].startswith('data:image/png;base64,'))
        self.assertLessEqual(answer['prepared_size'][0],1024);self.assertLessEqual(answer['prepared_size'][1],256)
        for box in answer['boxes']:
            self.assertGreater(box[2],box[0]);self.assertGreater(box[3],box[1])

if __name__=='__main__':unittest.main()
