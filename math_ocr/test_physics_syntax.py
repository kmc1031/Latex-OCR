import unittest
from physics_syntax import parse,tokenize,latex,canonical,result
from slash_semantics import slash_tokens


class PhysicsSyntax(unittest.TestCase):
    def test_fraction_layouts_preserve_implicit_products(self):
        for text in [r'K=\frac{1}{2}mv^{2}',r'F=G\frac{m_{1}m_{2}}{r^{2}}',r'T=2\pi\sqrt{\frac{l}{g}}']:
            tokens=tokenize(text)
            self.assertEqual(canonical(parse(tokens)),canonical(parse(slash_tokens(tokens))))

    def test_physics_roundtrip(self):
        from prepare_physics import LAWS
        for _,text in LAWS:
            tree=parse(tokenize(text))
            self.assertEqual(tree,parse(tokenize(latex(tree))),text)

    def test_invalid_predictions_are_rejected(self):
        for text in ['',r'\frac{x}',r'\unknown{x}','x=','{x+1','1.2.3']:
            with self.assertRaises(ValueError):parse(tokenize(text))

    def test_display_preserves_function_parentheses_without_changing_grading(self):
        for text,visible in [(r'F(y)=O(\frac{1}{\sqrt{y}})','(y)'),(r'g[\theta]=x','[')]:
            tokens=tokenize(text)
            tree,converted=result(tokens)
            self.assertIn(visible,converted)
            self.assertEqual(canonical(tree),canonical(parse(tokens)))


if __name__=='__main__':unittest.main()
