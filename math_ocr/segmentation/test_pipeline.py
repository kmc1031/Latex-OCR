import unittest
import numpy as np
import torch
from PIL import Image,ImageDraw
from common import crop_symbol,geometry
from pipeline import assemble,components,preprocess
from physics_syntax import canonical,parse,tokenize
from train_detector import boxes_from_output

class GeometryRelations(torch.nn.Module):
    def forward(self,x):
        g=x[:,-16:];scores=torch.zeros((len(x),7));scores[:,1]=4
        scores[g[:,1]<-.4,2]=8;scores[g[:,1]>.4,3]=8
        return scores

class Checks(unittest.TestCase):
    def expression(self,boxes,labels):return assemble(boxes,labels,np.zeros((len(labels),1)),torch.zeros((len(labels),128)),GeometryRelations(),'cpu')
    def test_superscript_is_generated_from_layout(self):
        actual=self.expression([[0,20,20,60],[23,0,35,20]],['x','2'])
        self.assertEqual(canonical(parse(tokenize(actual))),canonical(parse(tokenize('x^{2}'))))
    def test_fraction_inside_radical_retains_all_children(self):
        actual=self.expression([[0,0,110,100],[25,45,95,48],[40,10,65,35],[40,65,65,90]],['\\sqrt','\\frac','a','b'])
        self.assertEqual(canonical(parse(tokenize(actual))),canonical(parse(tokenize(r'\sqrt{\frac{a}{b}}'))))
    def test_independent_baseline_symbols_are_retained(self):
        self.assertEqual(tokenize(self.expression([[0,20,15,50],[20,25,35,45],[40,20,55,50]],['x','+','y'])),['x','+','y'])
    def test_crop_preserves_aspect_and_preprocessing_rejects_blank(self):
        image=Image.new('L',(100,60),255);ImageDraw.Draw(image).line((20,15,60,40),fill=0,width=2)
        self.assertEqual(crop_symbol(image).shape,(64,64));self.assertGreater(len(components(image)),0)
        with self.assertRaises(ValueError):preprocess(Image.new('L',(100,100),255))
    def test_center_detector_decodes_xy_and_size_in_image_coordinates(self):
        heat=torch.full((1,1,16,16),-20.);wh=torch.ones((1,2,16,16));off=torch.zeros_like(wh)
        heat[0,0,5,4]=20;wh[0,:,5,4]=4
        boxes=boxes_from_output((heat,wh,off),64,64,.9)
        self.assertEqual(len(boxes),1);np.testing.assert_allclose(boxes[0][0],[8,12,24,28])

if __name__=='__main__':unittest.main()
