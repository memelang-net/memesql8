'''
Meme v8.08 | info@memelang.net | (c) HOLTWORK LLC | Patents Pending
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
from typing import List, Iterator, Iterable

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
	('SEP_LIMIT',	r'\s+'),				# LIMIT CONJUNCTION, AXIS+=1
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

	@property
	def unitary(self) -> bool: return self.kind in UNITARY_KINDS

	def __str__(self) -> str: return self.lexeme

TOK_EQL = Token('EQL', '=')
TOK_NOT = Token('NOT', '!=')
TOK_SAME = Token('SAME', SAME)
TOK_WILD = Token('WILD', WILD)
TOK_EMPTY = Token('EMPTY', EMPTY)


class Stream:
	def __init__(self, token: Iterable[Token]):
		self.token: Iterator[Token] = iter(token)
		self.buffer: list[Token] = []
	def peek(self) -> str|None: 
		if not self.buffer:
			val = next(self.token, None)
			if val is None: return None
			self.buffer.append(val)
		return self.buffer[0].kind
	def next(self) -> Token: 
		if not self.buffer:
			val = next(self.token, None)
			if val is None: raise SyntaxError('E_EOF')
			self.buffer.append(val)
		return self.buffer.pop(0)


class Branch(Kind):
	sep: str
	opr: Token
	children: List[Kind]
	def __init__(self, opr: Token|None = None, children: List[Kind]|None = None):
		if opr is None: opr = Token('EQL', '=')
		if children is None: children = []
		self.opr = opr
		self.children = children

	def dump(self) -> List: return [self.opr.lexeme, [tok.datum for tok in self.children]]
	def append(self, token: Kind): self.children.append(token)
	def extend(self, tokens: List[Kind]): self.children.extend(tokens)
	def check(self) -> 'Branch': return self
	def __str__(self) -> str: return ('' if self.opr.kind=='EQL' else self.opr.lexeme) + SEP_DATA.join(map(str, self.children))
	
	@property
	def unitary(self) -> bool: return self.opr.unitary and all(tok.unitary for tok in self.children)


class Limit(Branch):
	kind: str = 'LIMIT'
	sep: str = SEP_DATA
	def check(self) -> 'Limit':
		if len(self.children)>1 and self.opr.lexeme not in {'=','!='}: raise SyntaxError('E_DATA_OPR')
		if self.children[0].kind not in DATUM_KINDS: raise SyntaxError('E_DATUM_TYPE')
		if len(self.children)>1 and any(c.kind not in DATA_KINDS for c in self.children): raise SyntaxError('E_DATA_TYPE')
		return self

TOK_EQL_WILD = Limit(None, [TOK_WILD])
TOK_EQL_SAME = Limit(None, [TOK_SAME])
TOK_EQL_EMPTY = Limit(None, [TOK_EMPTY])


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
		for vctr_axis, vctr in enumerate(self.children):
			if not isinstance(vctr, Vector): raise TypeError('E_TYPE_VCTR')
			if not all(isinstance(limit, Limit) for limit in vctr.children): raise TypeError('E_TYPE_LIMIT')

		self.results = [[TOK_EQL_EMPTY for limit in vctr.children] for vctr in self.children]
		return self

	# HIGHER AXIS RESULTS CARRY FORWARD UNTIL END OF MATRIX
	def cylindrify(self) -> None:
		max_axis: Axis = 0
		for vctr_axis, vctr in enumerate(self.children):
			cur_axis = len(vctr.children)-1
			pad_same = max_axis - cur_axis
			if pad_same>0: self.children[vctr_axis].children.extend([TOK_EQL_SAME] * pad_same)
			max_axis = max(max_axis, cur_axis)
		for vctr_axis, vctr in enumerate(self.children):
			cur_axis = len(vctr.children)-1
			pad_wild = max_axis - cur_axis
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


def lex(src) -> Iterator[Token]:
	for m in MASTER_PATTERN.finditer(src):
		kind = m.lastgroup
		if kind == 'COMMENT': continue
		if kind == 'MISMATCH': raise SyntaxError
		yield Token(kind, m.group())


def parse(src: str) -> Iterator[Matrix]:
	tokens = Stream(lex(src))
	mtrx=Matrix()
	vctr=Vector()
	while (kind:=tokens.peek()):

		# LIMIT ::= [OPR_KINDS] (DATUM_KINDS | DATA_KINDS (SEP_DATA DATA_KINDS)+)
		# Single axis constraint
		if kind in OPR_KINDS|DATUM_KINDS:
			opr=tokens.next() if kind in OPR_KINDS else TOK_EQL
			while tokens.peek()=='SEP_LIMIT': tokens.next()

			limit=Limit(opr)

			# First datum
			if tokens.peek() not in DATUM_KINDS: raise SyntaxError('E_OPR_DATA')
			limit.append(tokens.next())

			# Additional data (COMMA SEPARATED)
			while tokens.peek()=='SEP_DATA':
				tokens.next()
				while tokens.peek()=='SEP_LIMIT': tokens.next()
				if tokens.peek() not in DATA_KINDS: raise SyntaxError('E_DATA_KIND')
				limit.append(tokens.next())
			vctr.append(limit.check())
			continue

		# VCTR  ::= LIMIT {SEP_LIMIT LIMIT}
		# Conjunctive vector of axis constraints
		if kind == 'SEP_VCTR':
			tokens.next()
			if vctr.children:
				mtrx.append(vctr.check())
				vctr = Vector()
			continue

		# MTRX  ::= VCTR {SEP_VCTR VCTR}
		# Conjunctive matrix of axis constraints
		if kind == 'SEP_MTRX':
			tokens.next()
			if vctr.children:
				mtrx.append(vctr.check())
				vctr = Vector()
			if mtrx.children:
				yield mtrx.check()
				mtrx = Matrix()
			continue

		# Consume spaces
		while tokens.peek()=='SEP_LIMIT': tokens.next()

	if vctr.children: mtrx.append(vctr.check())
	if mtrx.children: yield mtrx.check()


class Meme(Branch):
	kind: str = 'MEME'
	sep: str = SEP_MTRX
	children: List[Matrix]

	def __init__(self, src: str):
		self.src = src
		self.children = list(parse(src))

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