import unittest
import numpy as np
from PIL import Image,ImageDraw
from handwriting_data import annotations,rasterize,preprocess_handwriting


class HandwritingData(unittest.TestCase):
    def test_ink_coordinates_are_translation_and_scale_invariant(self):
        def ink(offset,scale):
            points=[(0,100),(50,0),(100,100)]
            trace=','.join(f'{x*scale+offset} {y*scale+offset} {i}' for i,(x,y) in enumerate(points))
            root,_=annotations(('<ink xmlns="http://www.w3.org/2003/InkML"><trace>'+trace+'</trace></ink>').encode())
            return np.array(rasterize(root))
        np.testing.assert_array_equal(ink(0,1),ink(2000,10))

    def test_blank_paper_is_not_recognized_as_ink(self):
        for value in [255,230,190]:
            with self.assertRaises(ValueError):preprocess_handwriting(Image.new('L',(300,200),value))

    def test_shaded_paper_crops_to_written_content(self):
        values=np.tile(np.linspace(175,245,400,dtype=np.uint8),(180,1))
        paper=Image.fromarray(values);draw=ImageDraw.Draw(paper)
        draw.line([(140,110),(160,65),(180,110)],fill=25,width=4)
        draw.line([(148,93),(172,93)],fill=25,width=4)
        result=preprocess_handwriting(paper)
        self.assertEqual(result.height,64);self.assertLess(result.width,100)
        self.assertLess(np.asarray(result).min(),80)


if __name__=='__main__':unittest.main()
