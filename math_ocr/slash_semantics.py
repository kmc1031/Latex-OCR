"""Restricted mathematical syntax used by the paired notation experiment.

This preserves expression structure; it is not symbolic algebra simplification.
Supports only the generated symbols, +/-/=, division, scripts, and square roots.
"""


def group(tokens, pos, opening="{", closing="}"):
    if pos >= len(tokens) or tokens[pos] != opening:
        raise ValueError("Expected a group")
    depth = 1
    for end in range(pos + 1, len(tokens)):
        if tokens[end] == opening:
            depth += 1
        elif tokens[end] == closing:
            depth -= 1
            if depth == 0:
                return tokens[pos + 1:end], end + 1
    raise ValueError("Unclosed group")


def slash_tokens(tokens):
    out, i = [], 0
    while i < len(tokens):
        if tokens[i] == "\\frac":
            numerator, end = group(tokens, i + 1)
            denominator, i = group(tokens, end)
            out += ["("] + slash_tokens(numerator) + [")", "/", "("] + slash_tokens(denominator) + [")"]
        else:
            out.append(tokens[i])
            i += 1
    return out


class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.i = 0

    def atom(self):
        if self.i >= len(self.tokens):
            raise ValueError("Missing operand")
        token = self.tokens[self.i]
        self.i += 1
        if token in ["(", "{"]:
            end = ")" if token == "(" else "}"
            node = self.expression(0)
            if self.i >= len(self.tokens) or self.tokens[self.i] != end:
                raise ValueError("Missing closing group")
            self.i += 1
        elif token == "\\frac":
            a = self.required_group()
            b = self.required_group()
            node = ("div", a, b)
        elif token == "\\sqrt":
            node = ("sqrt", self.required_group())
        elif token in ["+", "-"]:
            node = ("positive" if token == "+" else "negative", self.atom())
        elif len(token) == 1 and token.isalnum():
            node = ("symbol", token)
        else:
            raise ValueError(f"Unexpected token: {token}")
        while self.i < len(self.tokens) and self.tokens[self.i] in ["^", "_"]:
            script = self.tokens[self.i]
            self.i += 1
            node = ("sup" if script == "^" else "sub", node, self.required_group())
        return node

    def required_group(self):
        if self.i >= len(self.tokens) or self.tokens[self.i] != "{":
            raise ValueError("Expected braces")
        self.i += 1
        node = self.expression(0)
        if self.i >= len(self.tokens) or self.tokens[self.i] != "}":
            raise ValueError("Missing closing brace")
        self.i += 1
        return node

    def expression(self, min_prec=0):
        node = self.atom()
        precedence = {"=": 0, "+": 1, "-": 1, "/": 2}
        names = {"=": "eq", "+": "add", "-": "subtraction", "/": "div"}
        while self.i < len(self.tokens):
            op = self.tokens[self.i]
            if op not in precedence or precedence[op] < min_prec:
                break
            self.i += 1
            right = self.expression(precedence[op] + 1)
            node = (names[op], node, right)
        return node


def parse(tokens):
    parser = Parser(tokens)
    tree = parser.expression()
    if parser.i != len(tokens):
        raise ValueError("Trailing/unparsed tokens")
    return tree


def fractions(tree):
    if not isinstance(tree, tuple):
        return []
    result = [tree] if tree[0] == "div" else []
    for child in tree[1:]:
        if isinstance(child, tuple):
            result.extend(fractions(child))
    return result


def self_check():
    from prepare_data import tokenize
    equivalent = [(r"\frac{x+1}{y-2}", "(x+1)/(y-2)"),
                  (r"\frac{x}{\frac{y}{z}}", "(x)/((y)/(z))"),
                  (r"\sqrt{x+2}+a_{3}", r"\sqrt{x+2}+a_{3}")]
    for a,b in equivalent:
        assert parse(tokenize(a)) == parse(tokenize(b))
    assert parse(tokenize("(x+1)/y")) != parse(tokenize("x+1/y"))
    assert parse(tokenize("(x/y)/z")) != parse(tokenize("x/(y/z)"))
    assert parse(tokenize("x/y")) != parse(tokenize("y/x"))
