'''
Memelang v8.01 | info@memelang.net | (c) HOLTWORK LLC | Patents Pending
This script is optimized for training LLMs

ANALOGY TO RELATIONAL DATABASE
AXIS: 0->Value 1->Column_Name 2->Row_Primary_Key 3->Table_Name
SQL: SELECT ... FROM movies WHERE row_id=* AND actor IN ("Mark Hamill", "Mark") AND movie="Star Wars" AND rating>4
MEMELANG: "Mark Hamill",Mark actor * movies | "Star Wars" movie | >4 rating;
'''

import random, re, json, copy
from typing import List, Dict, Any, Iterator
from dataclasses import dataclass, field

Axis = int
NO_AXIS: Axis = -1

SEP_LIST, SEP_LIMIT, WILD, SIGIL_L2R = ',', ' ', '*', '#'

INLINE_QUERY_MODE = False
if INLINE_QUERY_MODE: SEP_VECT, SEP_QUERY = ' | ', '; '
else: SEP_VECT, SEP_QUERY = '\n', ';\n\n'


TOKEN_KIND_PATTERNS = (
	('COMMENT',		r'//[^\n]*'),
	('STAR',		r'"\*"'), # LITERAL ASTERISK, NOT WILDCARD, FOR TRAINING DISTINCTION
	('QUOTE',		r'"(?:[^"\\]|\\.)*"'), # ALWAYS JSON QUOTE ESCAPE EXOTIC CHARS name="John \"Jack\" Kennedy"
	('SEP_QUERY',	r';'),
	('SEP_VECT',	r'\n|\|'),
	('SEP_LIMIT',	r'[ ]'),
	('SEP_LIST',	r','), # OR LIST
	('INEQL',		r'!=|>=|<=|>|<'),
	('EQL',			r'='),
	('WILD',		r'\*'), # WILDCARD
	('IDENT',		r'[A-Za-z][A-Za-z0-9_]*'), # ALPHANUMERIC IDENTIFIERS ARE UNQUOTED
	('FLOAT',		r'-?\d*\.\d+'),
	('INT',			r'-?\d+'),
	('VAR',			rf'{SIGIL_L2R}[1-9]\d*'),
	('SAME',		r'_'), # VARIABLE: "SAME VALUE"
	('MISMATCH',	r'.'),
)

MASTER_PATTERN = re.compile('|'.join(f'(?P<{kind}>{pat})' for kind, pat in TOKEN_KIND_PATTERNS))

DAT_KINDS = {'IDENT', 'QUOTE', 'STAR', 'INT', 'FLOAT', 'VAR', 'SAME', 'WILD'}
UNITARY_KINDS = {'IDENT', 'QUOTE', 'STAR', 'INT', 'FLOAT', 'EQL'}

class Token:
	kind: str
	lexeme: str|None
	children: List['Token']
	datum: Any
	sep: str = ''
	axis: Axis
	unitary: bool

	def __init__(self, kind: str, lexeme: str|None = None, children: List['Token']|None = None, sep: str=''):
		self.kind = kind
		self.axis = NO_AXIS
		self.lexeme = lexeme
		self.children = children or []
		self.sep = sep

		if (lexeme is None) == (not children): raise ValueError('E_LEX_CHILD')

		if children:
			self.datum = None
			self.unitary = all(child.unitary for child in self.children)
			if kind == 'LIST' and len(self.children)!=1: self.unitary = False # ONLY ONE-ITEM LIST IS UNITARY

		else:
			self.unitary = (kind in UNITARY_KINDS)
			if kind == 'QUOTE': 	self.datum = json.loads(lexeme)
			elif kind == 'FLOAT': 	self.datum = float(lexeme)
			elif kind == 'INT':		self.datum = int(lexeme)
			else: 					self.datum = lexeme
			
	@property
	def is_leaf(self) -> bool: return self.lexeme is not None

	def __iter__(self) -> Iterator['Token']:
		if not self.is_leaf: yield from self.children

	def __str__(self) -> str:
		if self.is_leaf: return '' if self.kind=='EQL' else self.lexeme  # ELIDED
		else: return self.sep.join(map(str, self.children))

TOK_EQUALS = Token('EQL', '=')
TOK_SAME = Token('LIST', None, [Token('SAME', '_')])
TOK_EQUALS_SAME = Token('LIMIT', None, [TOK_EQUALS, TOK_SAME])


class Memelang:
	def __init__(self, src: str):
		self.tokens: List[Token] = []
		self.length = 0
		self.i = 0
		self.src = src

		self.replace(list(self.pass_token()))
		self.replace(list(self.pass_list()))
		self.replace(list(self.pass_limit()))
		self.replace(list(self.pass_vect()))
		self.replace(list(self.pass_matrix()))

	def peek(self) -> Token|None:
		return self.tokens[self.i] if self.i < self.length else None

	def peek_kind(self) -> str|None:
		token = self.peek()
		return token.kind if token else None

	def next(self) -> Token:
		tok = self.peek()
		if tok is None: raise SyntaxError('E_EOF')
		self.i += 1
		return tok

	def replace(self, tokens: List[Token]):
		self.i = 0
		self.tokens = tokens
		self.length = len(tokens)

	def __str__(self) -> str:
		return SEP_QUERY.join(map(str, self.tokens))

	# TOKENS FROM TOKEN_KIND_PATTERNS
	def pass_token(self):

		# Sanitize
		self.src = re.sub(r'[ ]+', ' ', self.src)
		self.src = re.sub(r'\s*\|\s*', '|', self.src)
		self.src = re.sub(r'[ ]*[\r\n]\s*', '\n', self.src)
		self.src = re.sub(r'\s*;[;\s]*', ';', self.src)

		for m in MASTER_PATTERN.finditer(self.src):
			kind = m.lastgroup
			text = m.group()
			if kind == 'COMMENT': continue
			if kind == 'MISMATCH': raise SyntaxError(f'Unexpected char {text!r} at {m.start()}')
			yield Token(kind, text)

	# DAT ::= DAT_KINDS
	# LIST ::= DAT {SEP_LIST DAT}
	# OR semantic list of datums
	def pass_list(self):
		while self.peek():
			if self.peek_kind() in DAT_KINDS:
				# NEVER WRAP LIST IN QUOTES
				child_tokens: List[Token] = [self.next()]

				while self.peek_kind() == 'SEP_LIST':
					# NEVER SPACES AROUND COMMA
					self.next()
					if self.peek_kind() not in DAT_KINDS: raise SyntaxError('E_LIST_KIND')

					# NEVER WILDS IN LIST
					if 'WILD' in (child_tokens[0].kind, self.peek_kind()): raise SyntaxError('E_LIST_WILD')
					child_tokens.append(self.next())

				yield Token('LIST', children=child_tokens, sep=SEP_LIST)

			elif self.peek_kind() == 'SEP_LIST': raise SyntaxError('E_LIST_SEP')

			else: yield self.next()

	# LIMIT ::= [EQL|INEQL] LIST
	# Single axis constraint
	def pass_limit(self):

		while self.peek():
			if self.peek_kind() in {'EQL','INEQL','LIST'}:
				opr = TOK_EQUALS if self.peek_kind() not in {'EQL', 'INEQL'} else self.next()

				# NEVER SPACE AFTER OPERATOR
				if self.peek_kind() == 'SEP_LIMIT': raise SyntaxError('E_OPR_SPACE')

				if self.peek_kind() != 'LIST': raise SyntaxError('E_OPR_LIST')
				dlist = self.next()
				child_length = len(dlist.children)

				# NEVER GREATER/LESSER LIST
				if opr.kind == 'INEQL' and child_length>1: raise SyntaxError('E_CMP_LIST')

				yield Token('LIMIT', children=[opr,dlist])

			elif self.peek_kind() in {'SEP_LIMIT','SEP_VECT','SEP_QUERY'}: yield self.next()
			else: raise SyntaxError('E_TOK')

	# VECT ::= LIMIT {SEP_LIMIT LIMIT}
	# Vector of axis constraints
	def pass_vect(self):
		axis: Axis = 0
		child_tokens: List[Token] = []

		while self.peek():
			if self.peek_kind() == 'LIMIT':
				token = self.next()
				token.axis=axis
				child_tokens.append(token)
				axis+=1
				if self.peek_kind() == 'LIMIT': raise SyntaxError('E_SEP_LIMIT')

			elif self.peek_kind() in {None, 'SEP_VECT', 'SEP_QUERY'}:
				if child_tokens: yield Token('VECT', children=child_tokens, sep=SEP_LIMIT)
				axis = 0
				child_tokens = []
				if self.peek_kind(): yield self.next()

			elif self.peek_kind() == 'SEP_LIMIT': self.next()
			else: raise SyntaxError('E_TOK')

	# MTRX ::= VECT {SEP_VECT VECT}
	# Matrix of axis constraints
	def pass_matrix(self):
		child_tokens: List[Token] = []
		while self.peek():
			if self.peek_kind() == 'VECT':
				token = self.next()
				child_tokens.append(token)
			elif self.peek_kind() in {None, 'SEP_QUERY'}:
				if child_tokens: yield Token('MTRX', children=child_tokens, sep=SEP_VECT)
				child_tokens = []
				if self.peek_kind(): self.next()
			elif self.peek_kind() == 'SEP_VECT': self.next()
			else: raise SyntaxError('E_TOK')