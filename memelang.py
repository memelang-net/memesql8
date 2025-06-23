'''
Memelang v8.03 | info@memelang.net | (c) HOLTWORK LLC | Patents Pending
This script is optimized for training LLMs

ANALOGY TO RELATIONAL DATABASE
AXIS: 0->Value 1->Column_Name 2->Row_Primary_Key 3->Table_Name
SQL: SELECT ... FROM movies WHERE row_id=* AND actor IN ("Mark Hamill", "Mark") AND movie="Star Wars" AND rating>4
MEMELANG: "Mark Hamill",Mark actor * movies | "Star Wars" movie | >4 rating;
'''

import random, re, json, copy
from typing import List, Dict, Any, Iterator
from dataclasses import dataclass, field

SEP_LIST, SEP_LIMIT, WILD, SIGIL = ',', ' ', '*', '#'

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
	('VAR',			rf'{SIGIL}\d+(?:{SIGIL}\d+)?'),
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
	unitary: bool

	def __init__(self, kind: str, lexeme: str = '', children: List['Token']|None = None):
		self.kind = kind
		self.lexeme = lexeme
		self.children = children or []

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

	def dump(self) -> Any:
		if self.children: return [child.dump() for child in self.children]
		else: return self.datum

	def __str__(self) -> str:
		if self.children: return self.lexeme.join(map(str, self.children))
		else: return '' if self.kind=='EQL' else self.lexeme  # ELIDED

TOK_EQUALS = Token('EQL', '=')
TOK_SAME = Token('LIST', SEP_LIST, [Token('SAME', '_')])
TOK_EQUALS_SAME = Token('LIMIT', '', [TOK_EQUALS, TOK_SAME])


class Memelang(Token):
	def __init__(self, src: str):
		self.children = []
		self.lexeme = SEP_QUERY
		self.src = src
		self.kind = 'QUERY'
		self.i=0

		# Sanitize
		self.datum = re.sub(r'[ ]+', ' ', self.src)
		self.datum = re.sub(r'\s*\|\s*', '|', self.datum)
		self.datum = re.sub(r'[ ]*[\r\n]\s*', '\n', self.datum)
		self.datum = re.sub(r'\s*;[;\s]*', ';', self.datum)

		# TOKENS FROM TOKEN_KIND_PATTERNS
		for m in MASTER_PATTERN.finditer(self.datum):
			kind = m.lastgroup
			text = m.group()
			if kind == 'COMMENT': continue
			if kind == 'MISMATCH': raise SyntaxError(f'Unexpected char {text!r} at {m.start()}')
			self.children.append(Token(kind, text))
		self.length = len(self.children)

		self.replace(list(self.pass_list()))
		self.replace(list(self.pass_limit()))
		self.replace(list(self.pass_vect()))
		self.replace(list(self.pass_matrix()))

	def peek(self) -> Token|None:
		return self.children[self.i] if self.i < self.length else None

	def peek_kind(self) -> str|None:
		token = self.peek()
		return token.kind if token else None

	def next(self) -> Token:
		tok = self.peek()
		if tok is None: raise SyntaxError('E_EOF')
		self.i += 1
		return tok

	def replace(self, children: List[Token]):
		self.i = 0
		self.children = children
		self.length = len(children)

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

				yield Token('LIST', SEP_LIST, child_tokens)

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

				yield Token('LIMIT', '', [opr,dlist])

			elif self.peek_kind() in {'SEP_LIMIT','SEP_VECT','SEP_QUERY'}: yield self.next()
			else: raise SyntaxError('E_TOK')

	# VECT ::= LIMIT {SEP_LIMIT LIMIT}
	# Vector of axis constraints
	def pass_vect(self):
		child_tokens: List[Token] = []
		limit_idx = 0
		while self.peek():
			if self.peek_kind() == 'LIMIT':
				token = self.next()
				child_tokens.append(token)
				limit_idx += 1
				if self.peek_kind() == 'LIMIT': raise SyntaxError('E_SEP_LIMIT')

			elif self.peek_kind() in {None, 'SEP_VECT', 'SEP_QUERY'}:
				if child_tokens: yield Token('VECT', SEP_LIMIT, child_tokens)
				child_tokens = []
				limit_idx = 0
				if self.peek_kind(): yield self.next()

			elif self.peek_kind() == 'SEP_LIMIT': self.next()
			else: raise SyntaxError('E_TOK')

	# MTRX ::= VECT {SEP_VECT VECT}
	# Matrix of axis constraints
	def pass_matrix(self):
		child_tokens: List[Token] = []
		vect_idx = 0
		while self.peek():
			if self.peek_kind() == 'VECT':
				token = self.next()
				child_tokens.append(token)
				vect_idx += 1
			elif self.peek_kind() in {None, 'SEP_QUERY'}:
				if child_tokens: yield Token('MTRX', SEP_VECT, child_tokens)
				child_tokens = []
				vect_idx = 0
				if self.peek_kind(): self.next()
			elif self.peek_kind() == 'SEP_VECT': self.next()
			else: raise SyntaxError('E_TOK')


	# RESOLVE VARIABLES AS #VECT#LIMIT COORDINATES OF THE (QUERY FOR NOW)
	def resolve_token(self, mtrx_idx:int, vartok: Token):
		if vartok.kind!='VAR': raise TypeError('VAR')

		if len(self.children)<=mtrx_idx: raise SyntaxError('E_VAR')
		mtrx = self.children[mtrx_idx]
		if mtrx.kind != 'MTRX': raise TypeError('MTRX')
		
		parts = vartok.lexeme[1:].split(SIGIL)
		vect_idx = int(parts[0])
		limit_idx = 0 if len(parts) == 1 else int(parts[1])

		if len(mtrx.children)<=vect_idx: raise SyntaxError('E_VAR')
		vect = mtrx.children[vect_idx]
		if vect.kind != 'VECT': raise TypeError('VECT')
		if len(vect.children)<=limit_idx: raise SyntaxError('E_VAR')
		if vect.children[limit_idx].kind != 'LIMIT': raise TypeError('LIMIT')

		return vect.children[limit_idx]