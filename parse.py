import re
from typing import Generator, List, Tuple

from funcparserlib.lexer import make_tokenizer, Token, LexerError

def compare_indent(base: str, new: str, pos: Tuple[int, int]) -> int:
    if base.startswith(new):
        return -1 if len(base) > len(new) else 0
    elif new.startswith(base):
        return 1
    raise LexerError(pos, f'mixed tab/space indent')

def tokenize(string: str) -> Generator[Token, None, None]:
    specs = [
        ('COMMENT', (r'#[^\r\n]*',)),
        ('SP', (r'[ \t]+',)),
        ('NL', (r'[\r\n]+',)),
        ('COLON', (r':',)),
        ('LPAREN', (r'\(',)),
        ('RPAREN', (r'\)',)),
        ('LSQUARE', (r'\[',)),
        ('RSQUARE', (r'\]',)),
        ('EQ', (r'==',)),
        ('NE', (r'!=',)),
        ('LE', (r'<=',)),
        ('LT', (r'<',)),
        ('GE', (r'>=',)),
        ('GT', (r'>',)),
        ('ASSIGN', (r'=',)),
        ('PASSIGN', (r'\+=',)),
        ('DOT', (r'\.',)),
        ('COMMA', (r',',)),
        ('FLOW', (r'flow',)),
        ('INTERNAL', (r'internal',)),
        ('ENTRYPOINT', (r'entrypoint',)),
        ('IF', (r'if',)),
        ('ELIF', (r'elif',)),
        ('ELSE', (r'else',)),
        ('DO', (r'do',)),
        ('WHILE', (r'while',)),
        ('FORK', (r'fork',)),
        ('BRANCH', (r'branch\d+',)),
        ('RETURN', (r'return',)),
        ('PASS', (r'pass',)),
        ('NOT', (r'not',)),
        ('AND', (r'and',)),
        ('OR', (r'or',)),
        ('TYPE', (r'int|float|str',)),
        ('BOOL', (r'true|false',)),
        ('IN', (r'in',)),
        ('ID', (r'[A-Za-z](?:[A-Za-z0-9_\-]*[A-Za-z0-9]|[A-Za-z0-9]*)',)),
        ('FLOAT', (r'[+-]?[ \t]*(?:\d+\.\d*|\d*\.\d+)',)), # todo: fix this
        ('INT', (r'[+-]?[ \t]*\d+',)), # todo: hex
        ('STRING', (r'"(?:\\\.|[^"\\])*"|\'(?:\\\.|[^\'\\])*\'',)),
    ]

    t = make_tokenizer(specs)
    pstack: List[Token] = []
    indent = ['']

    if string and string[-1] not in ('\r', '\n'):
        string = string + '\n'

    num_lines = len(re.findall(r'\r\n|\r|\n', string))
    last_token = None
    for x in t(string):
        if x.type == 'COMMENT':
            continue
        elif x.type == 'LPAREN':
            pstack.append(x)
        elif x.type == 'RPAREN':
            if not pstack:
                raise LexerError(x.start, 'no parentheses to close')
            if pstack.pop().type != 'LPAREN':
                raise LexerError(x.start, "expecting ']' but got ')'")
        elif x.type == 'LSQUARE':
            pstack.append(x)
        elif x.type == 'RSQUARE':
            if not pstack:
                raise LexerError(x.start, 'no bracket to close')
            if pstack.pop().type != 'LSQUARE':
                raise LexerError(x.start, "expecting ')' but got ']'")
        elif x.type == 'NL':
            if pstack:
                continue
            x = Token('NL', '', start=x.start, end=x.end)
        elif x.type == 'SP':
            if last_token and last_token.type == 'NL':
                indent_diff = compare_indent(indent[-1], x.name, x.start)
                if indent_diff < 0:
                    found = False
                    while indent:
                        s = indent.pop()
                        if s == x.name:
                            indent.append(s)
                            break
                        yield Token('DEDENT', '', start=x.start, end=x.end)
                    if not indent:
                        raise LexerError(x.end, 'dedent to unknown level')
                    continue
                elif indent_diff > 0:
                    indent.append(x.name)
                    x = Token('INDENT', '', start=x.start, end=x.end)
                else:
                    continue
            else:
                continue

        last_token = x
        yield x

    if pstack:
        raise LexerError((num_lines + 1, 0), 'unclosed parentheses/brackets')

    while indent[-1]:
        indent.pop()
        yield Token('DEDENT', '', start=(num_lines + 1, 0), end=(num_lines + 1, 0))

