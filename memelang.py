'''
Memelang v8.07 | info@memelang.net | (c) HOLTWORK LLC | Patents Pending
This script is optimized for training LLMs

1. EXAMPLE QUERY
MEMELANG: "Mark Hamill",Mark actor * movies ; * movie ; >4 rating ;;

SQL LIMIT_AXIS ANALOG:  0->Value  1->Column_Name  2->Row_Primary_Key  3->Table_Name
SQL: SELECT ... FROM movies WHERE row_id=* AND actor IN ("Mark Hamill", "Mark") AND movie=* AND rating>4

RDF LIMIT_AXIS ANALOG:  0->Object_Value  1->Predicate_Name  2->Subject_URI  3->Graph_Name
SPARQL: SELECT … WHERE { GRAPH <movies> {?s actor ?o . FILTER(?o IN ("Mark Hamill","Mark")) . ?s movie ?x . ?s rating ?r . FILTER(?r > 4)} }

2. EXAMPLE JOIN QUERY
MEMELANG: "Mark Hamill" actor * movies ; * movie ; _ _ ! ; * actor ;;
SQL: SELECT co.actor FROM movies AS mh JOIN movies AS co ON co.movie=mh.movie AND co.row_id!=mh.row_id WHERE mh.actor='Mark Hamill';
RDF: SELECT ?coActor WHERE { GRAPH <movies> { ?mhRow ex:actor "Mark Hamill" ; ex:movie ?movie . ?coRow ex:movie ?movie ; ex:actor ?coActor . FILTER ( ?coRow != ?mhRow ) } }
'''

import random, re, json
from typing import List

RAND_INT_MIN = 1 << 20
RAND_INT_MAX = 1 << 53

Axis = int # >=0
Datum = str | float | int

WILD, SAME, DIFF, SIGIL, EMPTY =  '*', '_', '!', '#', '\u2205'

SEP_LIMIT, SEP_DATA, SEP_VCTR, SEP_MTRX = ' ', ',', ';', ';;'
SEP_VCTR_PRETTY, SEP_MTRX_PRETTY = ' ; ', ' ;;\n'

TOKEN_KIND_PATTERNS = (
	('COMMENT',		r'//[^\n]*'),
	('QUOTE',		r'"(?:[^"\\]|\\.)*"'), # ALWAYS JSON QUOTE ESCAPE EXOTIC CHARS name="John \"Jack\" Kennedy"

	('SEP_MTRX',	re.escape(SEP_MTRX)),	# MTRX DISJUNCTION, AXIS=0
	('SEP_VCTR',	re.escape(SEP_VCTR)),	# VCTR CONJUNCTION, AXIS=0
	('SEP_LIMIT',	re.escape(SEP_LIMIT)),	# LIMIT CONJUNCTION, AXIS+=1
	('SEP_DATA',	re.escape(SEP_DATA)),	# DATUM DISJUNCTION, AXIS SAME

	('NOT',			r'!='),
	('GE',			r'>='),
	('LE',			r'<='),
	('GT',			r'>'),
	('LT',			r'<'),
	('EQL',			r'='),

	('WILD',		re.escape(WILD)),	# WILDCARD, NEVER QUOTE
	('SAME',		re.escape(SAME)),	# EQUALS DATA FROM (LIMIT_AXIS, VCTR_AXIS-1)
	('DIFF',		re.escape(DIFF)),	# NOT EQUALS DATA FROM (LIMIT_AXIS, VCTR_AXIS-1)
	('EMPTY',		re.escape(EMPTY)),	# EMPTY SET, ANTI-WILD
	('VAR',			rf'(?:{SIGIL}\d+){{1,3}}'),
	
	('IDENT',		r'[A-Za-z][A-Za-z0-9_]*'), # ALPHANUMERIC IDENTIFIERS ARE UNQUOTED
	('FLOAT',		r'-?\d*\.\d+'),
	('INT',			r'-?\d+'),
	('MISMATCH',	r'.'),
)

MASTER_PATTERN = re.compile('|'.join(f'(?P<{kind}>{pat})' for kind, pat in TOKEN_KIND_PATTERNS))

OPR_KINDS = {'NOT','GE','LE','GT','LT','EQL'}
DATUM_KINDS = {'IDENT', 'QUOTE', 'INT', 'FLOAT', 'VAR', 'SAME', 'DIFF', 'WILD', 'EMPTY'}
DATA_KINDS = {'IDENT', 'QUOTE', 'INT', 'FLOAT', 'VAR', 'SAME', 'EMPTY'} # NEVER DIFF OR WILD IN MULTI-DATA LIST
UNITARY_KINDS = {'IDENT', 'QUOTE', 'INT', 'FLOAT', 'EQL'}

class Kind:
	kind: str
	@property
	def unitary(self) -> bool: return False

class Token(Kind):
	lexeme: str
	datum: Datum
	def __init__(self, kind: str, lexeme: str):
		self.kind = kind
		self.lexeme = lexeme
		if kind == 'QUOTE': 	self.datum = json.loads(lexeme)
		elif kind == 'FLOAT': 	self.datum = float(lexeme)
		elif kind == 'INT':		self.datum = int(lexeme)
		else: 					self.datum = lexeme

	def __str__(self) -> str: return self.lexeme

TOK_EQL = Token('EQL', '=')
TOK_NOT = Token('NOT', '!=')
TOK_SAME = Token('SAME', SAME)
TOK_WILD = Token('WILD', WILD)
TOK_EMPTY = Token('EMPTY', EMPTY)


class Limit(Kind):
	kind: str
	opr: Token
	children: List[Token]
	def __init__(self, opr: Token|None = None, children: List[Token]|None = None):
		if opr is None: opr = Token('EQL', '=')
		if children is None: children = []
		self.kind = 'LIMIT'
		self.opr = opr
		self.children = children

	def append(self, token: Token): self.children.append(token)
	def extend(self, tokens: List[Token]): self.children.extend(tokens)
	@property
	def unitary(self) -> bool: return self.opr.kind in UNITARY_KINDS and all(tok.kind in UNITARY_KINDS for tok in self.children)
	def check(self) -> 'Limit':
		if len(self.children)>1 and self.opr.lexeme not in {'=','!='}: raise SyntaxError('E_DATA_OPR')
		if self.children[0].kind not in DATUM_KINDS: raise SyntaxError('E_DATUM_TYPE')
		if len(self.children)>1 and any(c.kind not in DATA_KINDS for c in self.children): raise SyntaxError('E_DATA_TYPE')
		return self

	def dump(self) -> List: return [self.opr.lexeme, [tok.datum for tok in self.children]]
	def __str__(self) -> str: return ('' if self.opr.kind=='EQL' else self.opr.lexeme) + SEP_DATA.join(map(str, self.children))

TOK_EQL_WILD = Limit(None, [TOK_WILD])
TOK_EQL_SAME = Limit(None, [TOK_SAME])
TOK_EQL_EMPTY = Limit(None, [TOK_EMPTY])

class Branch(Kind):
	sep: str
	children: list
	def __init__(self):
		self.children = []

	def dump(self) -> list: return [child.dump() for child in self.children]
	def append(self, token: Kind): self.children.append(token)
	def extend(self, tokens: List[Kind]): self.children.extend(tokens)
	def check(self) -> 'Branch': return self
	def __str__(self) -> str: return self.sep.join(map(str, self.children))
	
	@property
	def unitary(self) -> bool: return all(child.unitary for child in self.children)


class Vector(Branch):
	kind: str = 'VCTR'
	sep: str = SEP_LIMIT
	children: List[Limit]

class Matrix(Branch):
	kind: str = 'MTRX'
	sep = SEP_VCTR
	children: List[Vector]
	results: List[Vector]

	def check(self) -> 'Matrix':
		self.results = [[TOK_EQL_EMPTY for limit in vctr.children] for vctr in self.children]
		return self

	# HIGHER AXIS RESULTS CARRY FORWARD UNTIL END OF MATRIX
	def cylindrify(self) -> None:
		max_axis_all: Axis = max(len(vctr.children) for vctr in self.children)-1
		max_axis_sofar: Axis = 0
	
		for vctr_axis, vctr in enumerate(self.children):
			if not isinstance(vctr, Vector): raise TypeError('E_TYPE_VCTR')
			if not all(isinstance(limit, Limit) for limit in vctr.children): raise TypeError('E_TYPE_LIMIT')

			cur_axis = len(vctr.children)-1

			pad_same = max_axis_sofar - cur_axis
			if pad_same>0: self.children[vctr_axis].children.extend([TOK_EQL_SAME] * pad_same)

			max_axis_sofar = max(max_axis_sofar, cur_axis)
			pad_wild = max_axis_all - max_axis_sofar
			if pad_wild>0: self.children[vctr_axis].children.extend([TOK_EQL_WILD] * pad_wild)

	# STORE UNITARY DATA IN MEMORY
	def store(self) -> None:
		if not self.unitary: raise SyntaxError('E_UNI_MTRX')
		self.cylindrify()
		for vctr_axis, vctr in enumerate(self.children):
			if not vctr.unitary: raise SyntaxError('E_UNI_VCTR')
			for limit_axis, limit in enumerate(vctr.children):
				if limit is TOK_EQL_WILD: self.results[vctr_axis][limit_axis] = Limit(None, [Token('INT', str(random.randrange(RAND_INT_MIN, RAND_INT_MAX)))])
				elif limit is TOK_EQL_SAME:
					if limit_axis == 0: raise SyntaxError('E_SAME_ZERO')
					self.results[vctr_axis][limit_axis] = self.results[vctr_axis][limit_axis-1]
				else: self.results[vctr_axis][limit_axis] = limit

	def coord(self, limit_axis: Axis = 0, vctr_axis: Axis = 0) -> List[Token]:
		return self.results[vctr_axis][limit_axis].children


class Memelang(Branch):
	kind: str = 'MEME'
	sep: str = SEP_MTRX
	children: List[Matrix]

	def __init__(self, src: str):
		self.children = []
		self.src = src
		self.i=0

		# TOKENS FROM TOKEN_KIND_PATTERNS
		for m in MASTER_PATTERN.finditer(self.src):
			kind = m.lastgroup
			text = m.group()
			if kind == 'COMMENT': continue
			if kind == 'MISMATCH': raise SyntaxError(f'Unexpected char {text!r} at {m.start()}')
			self.children.append(Token(kind, text))
		self.length = len(self.children)

		self.replace(list(self.pass_limit()))
		self.replace(list(self.pass_vctr()))
		self.replace(list(self.pass_mtrx()))
		self.cylindrify()

	def peek(self) -> str|None:
		return self.children[self.i].kind if self.i < self.length else None

	def next(self) -> Kind:
		if self.i >= self.length: raise SyntaxError('E_EOF')
		self.i += 1
		return self.children[self.i-1]

	def replace(self, children: List[Matrix]):
		self.i = 0
		self.children = children
		self.length = len(children)

	# DATA ::= DATUM_KINDS {SEP_DATA DATUM_KINDS}
	# LIMIT ::= [OPR_KINDS] DATA
	# Single axis constraint
	def pass_limit(self) -> Token|Limit:
		while self.peek():
			if self.peek() in OPR_KINDS | DATUM_KINDS:
				opr = TOK_EQL if self.peek() not in OPR_KINDS else self.next()
				limit = Limit(opr)

				while self.peek() == 'SEP_LIMIT': self.next() # AVOID SPACE AFTER OPERATOR
				if self.peek() not in DATUM_KINDS: raise SyntaxError('E_OPR_DATA')

				limit.append(self.next())

				while self.peek() == 'SEP_DATA':
					self.next()
					while self.peek() == 'SEP_LIMIT': self.next() # AVOID SPACE AFTER COMMA
					if self.peek() not in DATA_KINDS: raise SyntaxError('E_DATA_KIND')
					limit.append(self.next())

				yield limit.check()

			elif self.peek() in {'SEP_LIMIT','SEP_VCTR','SEP_MTRX'}: yield self.next()
			else: raise SyntaxError('E_TOK')

	# VCTR ::= LIMIT {SEP_LIMIT LIMIT}
	# Conjunctive vector of axis constraints
	def pass_vctr(self) -> Token|Vector:
		vctr = Vector()
		while self.peek():
			if self.peek() == 'LIMIT':
				vctr.append(self.next())
				if self.peek() == 'LIMIT': raise SyntaxError('E_SEP_LIMIT')

			elif self.peek() in {None, 'SEP_VCTR', 'SEP_MTRX'}:
				if vctr.children: yield vctr.check()
				vctr = Vector()
				if self.peek(): yield self.next()

			elif self.peek() == 'SEP_LIMIT': self.next()
			else: raise SyntaxError('E_TOK')

		if vctr.children: yield vctr.check()

	# MTRX ::= VCTR {SEP_VCTR VCTR}
	# Conjunctive matrix of axis constraints
	def pass_mtrx(self) -> Matrix:
		mtrx = Matrix()
		while self.peek():
			if self.peek() == 'VCTR': mtrx.append(self.next())
			elif self.peek() in {None, 'SEP_MTRX'}:
				if mtrx.children: yield mtrx.check()
				mtrx = Matrix()
				if self.peek(): self.next()
			elif self.peek() == 'SEP_VCTR': self.next()
			else: raise SyntaxError('E_TOK')

		if mtrx.children: yield mtrx.check()

	def cylindrify(self) -> None: for mtrx in self.children: mtrx.cylindrify()
	def store(self): for mtrx in self.children: mtrx.store()

	# VAR ::= SIGIL LIMIT [SIGIL VCTR [SIGIL MTRX]]]
	# VARIABLES ARE AXIS COORDINATES OF PRIOR RESULTS
	def expand(self, query_limit: Limit, from_limit_axis: Axis, from_vctr_axis: Axis, from_mtrx_axis: Axis) -> Limit:
		expansion = Limit()

		for tok in query_limit.children:

			if tok.kind=='DIFF':
				if from_vctr_axis < 1: raise SyntaxError('E_DIFF_OOB')
				return Limit(TOK_NOT, self.children[from_mtrx_axis].coord(from_limit_axis, from_vctr_axis-1))

			elif tok.kind=='SAME':
				if from_vctr_axis < 1: raise SyntaxError('E_SAME_OOB')
				expansion.extend(self.children[from_mtrx_axis].coord(from_limit_axis, from_vctr_axis-1))

			elif tok.kind=='VAR':
				try: parts = [int(p) for p in tok.lexeme.lstrip(SIGIL).split(SIGIL)]
				except ValueError: raise SyntaxError('E_VAR_PARSE')

				if any(idx < 0 for idx in parts): raise SyntaxError('E_VAR_NEG')
				if len(parts) > 3: raise SyntaxError('E_VAR_LONG') # NOT 4D YET

				limit_axis = parts[0]
				vctr_axis = from_vctr_axis if len(parts)<2 else parts[1]
				mtrx_axis = from_mtrx_axis if len(parts)<3 else parts[2]

				# ALWAYS VARIABLES REFERENCE *PRIOR* LIMITS
				if (mtrx_axis, vctr_axis, limit_axis) >= (from_mtrx_axis, from_vctr_axis, from_limit_axis): raise SyntaxError('E_VAR_FWD')

				try: expansion.extend(self.children[mtrx_axis].coord(limit_axis, vctr_axis))
				except IndexError: raise SyntaxError('E_VAR_OOB')

			else: expansion.append(tok)

		return expansion


def intersect(query_limit: Limit, store_limit: Limit) -> Limit:
	store_children = list(store_limit.children)
	if not store_children: store_children=[Token('EMPTY', EMPTY)]

	intersection = Limit()

	query_opr, query_datums = query_limit.dump()

	# INVERT WILD TO NOT EMPTY SET
	if query_datums == [WILD]:
		if query_opr == '=': query_opr = '!='
		elif query_opr == '!=': query_opr = '='
		query_datums = [EMPTY]

	if query_opr == '=': intersection.extend([t for t in store_children if t.datum in query_datums])
	elif query_opr == '!=': intersection.extend([t for t in store_children if t.datum not in query_datums])

	# ALWAYS EMPTY RETURNS ANY NUMERIC
	elif EMPTY in query_datums: intersection.extend([t for t in store_children if t.kind in {'INT','FLOAT'}])

	elif len(query_datums)!=1 or not isinstance(query_datums[0], (int,float)): raise TypeError('E_INTER_NUM')
	
	elif query_opr == '>':  intersection.extend([t for t in store_children if t.kind in {'INT','FLOAT'} and t.datum>query_datums[0]])
	elif query_opr == '>=': intersection.extend([t for t in store_children if t.kind in {'INT','FLOAT'} and t.datum>=query_datums[0]])
	elif query_opr == '<':  intersection.extend([t for t in store_children if t.kind in {'INT','FLOAT'} and t.datum<query_datums[0]])
	elif query_opr == '<=': intersection.extend([t for t in store_children if t.kind in {'INT','FLOAT'} and t.datum<=query_datums[0]])

	return intersection