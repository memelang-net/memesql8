'''
Memelang v8.04 | info@memelang.net | (c) HOLTWORK LLC | Patents Pending
This script is optimized for training LLMs

1. EXAMPLE QUERY
MEMELANG: "Mark Hamill",Mark actor * movies; "Star Wars" movie; >4 rating;;

SQL LIMIT_AXIS ANALOG:  0->Value  1->Column_Name  2->Row_Primary_Key  3->Table_Name
SQL: SELECT ... FROM movies WHERE row_id=* AND actor IN ("Mark Hamill", "Mark") AND movie="Star Wars" AND rating>4

RDF LIMIT_AXIS ANALOG:  0->Object_Value  1->Predicate_Name  2->Subject_URI  3->Graph_Name
SPARQL: SELECT … WHERE { GRAPH <movies> {?s actor ?o . FILTER(?o IN ("Mark Hamill","Mark")) . ?s movie "Star Wars" . ?s rating ?r . FILTER(?r > 4)} }

2. EXAMPLE JOIN QUERY
MEMELANG: "Mark Hamill" actor * movies; * movie; _ _ !; * actor;;
SQL: SELECT co.actor FROM movies AS mh JOIN movies AS co ON co.movie=mh.movie AND co.row_id!=mh.row_id WHERE mh.actor='Mark Hamill';
RDF: SELECT ?coActor WHERE { GRAPH <movies> { ?mhRow ex:actor "Mark Hamill" ; ex:movie ?movie . ?coRow ex:movie ?movie ; ex:actor ?coActor . FILTER ( ?coRow != ?mhRow ) } }
'''

import random, re, json, copy
from typing import List, Dict, Any, Iterator, Union
from dataclasses import dataclass, field

Axis = int # >=0
Datum = Union[str, float, int]

SEP_LIST, SEP_LIMIT, WILD, SIGIL = ',', ' ', '*', '#'

PRETTY_TRAIL = '\n' # ' '
SEP_VECT, SEP_QUERY = ';'+PRETTY_TRAIL, ';;'+PRETTY_TRAIL+PRETTY_TRAIL

TOKEN_KIND_PATTERNS = (
	('COMMENT',		r'//[^\n]*'),
	('QUOTE',		r'"(?:[^"\\]|\\.)*"'), # ALWAYS JSON QUOTE ESCAPE EXOTIC CHARS name="John \"Jack\" Kennedy"
	('SEP_QUERY',	r';;'), # DISJUNCTION
	('SEP_VECT',	r';'), # JUNCTION
	('SEP_LIMIT',	r'[ ]'),
	('SEP_LIST',	r','), # OR LIST
	('INEQL',		r'!=|>=|<=|>|<'),
	('EQL',			r'='),
	('WILD',		r'\*'), # WILDCARD, NEVER QUOTE
	('IDENT',		r'[A-Za-z][A-Za-z0-9_]*'), # ALPHANUMERIC IDENTIFIERS ARE UNQUOTED
	('FLOAT',		r'-?\d*\.\d+'),
	('INT',			r'-?\d+'),
	('VAR',			rf'{SIGIL}\d+(?:{SIGIL}\d+){{0,2}}'),
	('SAME',		r'_'), # VARIABLE: "SAME VALUE"
	('DIFF',		r'!'), # VARIABLE: "DIFFERENT VALUE"
	('MISMATCH',	r'.'),
)

MASTER_PATTERN = re.compile('|'.join(f'(?P<{kind}>{pat})' for kind, pat in TOKEN_KIND_PATTERNS))

DAT_KINDS = {'IDENT', 'QUOTE', 'INT', 'FLOAT', 'VAR', 'SAME', 'DIFF', 'WILD'}
UNITARY_KINDS = {'IDENT', 'QUOTE', 'INT', 'FLOAT', 'EQL'}

class Token:
	kind: str
	lexeme: str|None
	children: List['Token']
	datum: Datum
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
		else: return '' if self.kind=='EQL' else self.lexeme # ELIDED

TOK_EQUALS = Token('EQL', '=')
TOK_SAME = Token('LIST', SEP_LIST, [Token('SAME', '_')])
TOK_LIMIT_SAME = Token('LIMIT', '', [TOK_EQUALS, TOK_SAME])


class Memelang(Token):
	def __init__(self, src: str):
		self.children = []
		self.lexeme = SEP_QUERY
		self.src = src
		self.kind = 'QUERY'
		self.i=0
		self.results: List[List[List[List[Datum]]]] = []

		# Sanitize
		self.datum = re.sub(r'[ ]+', ' ', self.src)
		self.datum = re.sub(r'\s*;;\s*', ';;', self.datum)
		self.datum = re.sub(r'\s*;\s*', ';', self.datum)
		self.datum = re.sub(r'[ ]*[\r\n]\s*', '\n', self.datum)

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
		self.carry_fwd()
		self.build_results()

	def peek(self) -> str|None:
		return self.children[self.i].kind if self.i < self.length else None

	def next(self) -> Token:
		if self.i >= self.length: raise SyntaxError('E_EOF')
		self.i += 1
		return self.children[self.i-1]

	def replace(self, children: List[Token]):
		self.i = 0
		self.children = children
		self.length = len(children)

	# DAT ::= DAT_KINDS
	# LIST ::= DAT {SEP_LIST DAT}
	# OR semantic list of datums
	def pass_list(self):
		while self.peek():
			if self.peek() in DAT_KINDS:
				# NEVER WRAP LIST IN QUOTES
				child_tokens: List[Token] = [self.next()]

				while self.peek() == 'SEP_LIST':
					# NEVER SPACES AROUND COMMA
					self.next()
					if self.peek() not in DAT_KINDS: raise SyntaxError('E_LIST_KIND')

					# NEVER WILDS IN LIST
					if 'WILD' in (child_tokens[0].kind, self.peek()): raise SyntaxError('E_LIST_WILD')
					child_tokens.append(self.next())

				yield Token('LIST', SEP_LIST, child_tokens)

			elif self.peek() == 'SEP_LIST': raise SyntaxError('E_LIST_SEP')

			else: yield self.next()

	# LIMIT ::= [EQL|INEQL] LIST
	# Single axis constraint
	def pass_limit(self):

		while self.peek():
			if self.peek() in {'EQL','INEQL','LIST'}:
				opr = TOK_EQUALS if self.peek() not in {'EQL', 'INEQL'} else self.next()

				# NEVER SPACE AFTER OPERATOR
				if self.peek() == 'SEP_LIMIT': raise SyntaxError('E_OPR_SPACE')

				if self.peek() != 'LIST': raise SyntaxError('E_OPR_LIST')
				dlist = self.next()
				child_length = len(dlist.children)

				# NEVER GREATER/LESSER LIST
				if opr.kind == 'INEQL' and child_length>1: raise SyntaxError('E_CMP_LIST')

				yield Token('LIMIT', '', [opr,dlist])

			elif self.peek() in {'SEP_LIMIT','SEP_VECT','SEP_QUERY'}: yield self.next()
			else: raise SyntaxError('E_TOK')

	# VECT ::= LIMIT {SEP_LIMIT LIMIT}
	# Vector of axis constraints
	def pass_vect(self):
		child_tokens: List[Token] = []
		while self.peek():
			if self.peek() == 'LIMIT':
				token = self.next()
				child_tokens.append(token)
				if self.peek() == 'LIMIT': raise SyntaxError('E_SEP_LIMIT')

			elif self.peek() in {None, 'SEP_VECT', 'SEP_QUERY'}:
				if child_tokens: yield Token('VECT', SEP_LIMIT, child_tokens)
				child_tokens = []
				if self.peek(): yield self.next()

			elif self.peek() == 'SEP_LIMIT': self.next()
			else: raise SyntaxError('E_TOK')

	# MTRX ::= VECT {SEP_VECT VECT}
	# Matrix of axis constraints
	def pass_matrix(self):
		child_tokens: List[Token] = []
		while self.peek():
			if self.peek() == 'VECT':
				token = self.next()
				child_tokens.append(token)
			elif self.peek() in {None, 'SEP_QUERY'}:
				if child_tokens: yield Token('MTRX', SEP_VECT, child_tokens)
				child_tokens = []
				if self.peek(): self.next()
			elif self.peek() == 'SEP_VECT': self.next()
			else: raise SyntaxError('E_TOK')


	# HIGHER AXIS RESULTS CARRY FORWARD UNTIL END OF MATRIX
	def carry_fwd(self) -> None:
		for mtrx_axis, mtrx in enumerate(self.children):
			if mtrx.kind != 'MTRX': raise TypeError('E_TYPE_MTRX')
			max_vect_len = 1
			for vect_axis, vect in enumerate(mtrx.children):
				if vect.kind != 'VECT': raise TypeError('E_TYPE_VECT')
				if not all(limit.kind == 'LIMIT' for limit in vect.children): raise TypeError('E_TYPE_LIMIT')
				vect_len = len(vect.children)
				gap = max_vect_len - vect_len
				if gap>0: vect.children.extend(TOK_LIMIT_SAME for _ in range(gap))
				elif gap<0: max_vect_len = vect_len


	def build_results(self) -> None:
		# self.results[mtrx_axis][vect_axis][limit_axis] = []
		self.results = [[[[] for limit in vect.children] for vect in mtrx.children] for mtrx in self.children]


	# VAR ::= # LIMIT [# VECT [# MTRX]]]
	# VARIABLES ARE AXIS COORDINATES OF PRIOR RESULTS
	def resolve_token(self, vartok: Token, from_limit_axis:Axis, from_vect_axis:Axis, from_mtrx_axis:Axis) -> List[Datum]:
		if vartok.kind!='VAR': raise TypeError('VAR')

		try: parts = [int(p) for p in vartok.lexeme.lstrip(SIGIL).split(SIGIL)]
		except ValueError: raise SyntaxError('E_VAR_PARSE')

		if any(idx < 0 for idx in parts): raise SyntaxError('E_VAR_NEG')
		if len(parts) > 3: raise SyntaxError('E_VAR_LONG') # NOT 4D YET

		limit_axis = parts[0]
		vect_axis = from_vect_axis if len(parts)<2 else parts[1]
		mtrx_axis = from_mtrx_axis if len(parts)<3 else parts[2]

		# ALWAYS VARIABLES REFERENCE *PRIOR* LIMITS
		if (mtrx_axis, vect_axis, limit_axis) >= (from_mtrx_axis, from_vect_axis, from_limit_axis): raise SyntaxError('E_VAR_FWD')

		try: result_limit = self.results[mtrx_axis][vect_axis][limit_axis]
		except IndexError: raise SyntaxError('E_VAR_OOB')

		return result_limit