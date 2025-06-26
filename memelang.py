'''
Meme v8.09 | info@memelang.net | (c) HOLTWORK LLC | Patents Pending
This script is optimized for training LLMs

1. EXAMPLE QUERY
MEMELANG: "Mark Hamill",Mark actor * movies ; * movie ; >4 rating ;;

SQL LIMIT_AXIS ANALOG:  0->Value  1->Column_Name  2->Row_Primary_Key  3->Table_Name
SQL: SELECT ... FROM movies WHERE row_id=* AND actor IN ("Mark Hamill", "Mark") AND movie=* AND rating>4

RDF LIMIT_AXIS ANALOG:  0->Object_Value  1->Predicate_Name  2->Subject_URI  3->Graph_Name
SPARQL: SELECT … WHERE { GRAPH <movies> {?s actor ?o . FILTER(?o IN ("Mark Hamill","Mark")) . ?s movie ?x . ?s rating ?r . FILTER(?r > 4)} }

2. VARIABLE EXAMPLE ACTOR NAME = MOVIE TITLE
MEMELANG: $x=* actor * movies ; $x movie ;;
SQL: SELECT ... FROM movies WHERE actor=movie

3. EXAMPLE JOIN QUERY
MEMELANG: "Mark Hamill" actor * movies ; * movie ; _ _ ! ; * actor ;;
MEMELANG: "Mark Hamill" actor $row_id=* movies ; * movie ; _ _ !=$row_id ; * actor ;;
SQL: SELECT co.actor FROM movies AS mh JOIN movies AS co ON co.movie=mh.movie AND co.row_id!=mh.row_id WHERE mh.actor='Mark Hamill';
RDF: SELECT ?coActor WHERE { GRAPH <movies> { ?mhRow ex:actor "Mark Hamill" ; ex:movie ?movie . ?coRow ex:movie ?movie ; ex:actor ?coActor . FILTER ( ?coRow != ?mhRow ) } }
'''

import random, re, json
from typing import List, Iterator, Iterable, Dict, Tuple

RAND_INT_MIN = 1 << 20
RAND_INT_MAX = 1 << 53

Axis = int # >=0
Datum = str | float | int

WILD, SAME, DIFF, EMPTY =  '*', '_', '!', '\u2205'

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
	('VAR',			rf'\$[A-Za-z0-9_]+'),
	
	('IDENT',		r'[A-Za-z][A-Za-z0-9_]*'), # ALPHANUMERIC IDENTIFIERS ARE UNQUOTED
	('FLOAT',		r'-?\d*\.\d+'),
	('INT',			r'-?\d+'),
	('MISMATCH',	r'.'),
)

MASTER_PATTERN = re.compile('|'.join(f'(?P<{kind}>{pat})' for kind, pat in TOKEN_KIND_PATTERNS))

OPR_KINDS = {'NOT','GE','LE','GT','LT','EQL'}
SEP_KINDS = {'SEP_MTRX','SEP_VCTR','SEP_LIMIT','SEP_DATA',None}
SUGAR_KINDS = {'DIFF', 'WILD'}
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
	kind: str = 'BRNCH'
	sep: str
	opr: Token
	var: Token
	children: List[Kind]
	def __init__(self, opr: Token|None = None, children: List[Kind]|None = None):
		if opr is None: opr = Token('EQL', '=')
		if children is None: children = []
		self.opr = opr
		self.children = children
		self.var = None

	def reopr(self, opr: Token): self.opr = opr
	def dump(self) -> List: return [self.opr.lexeme, [tok.datum for tok in self.children]]
	def append(self, token: Kind): self.children.append(token)
	def extend(self, tokens: List[Kind]): self.children.extend(tokens)
	def check(self) -> 'Branch': return self
	def __str__(self) -> str: return ('' if self.opr.kind=='EQL' else self.opr.lexeme) + SEP_DATA.join(map(str, self.children))
	
	@property
	def unitary(self) -> bool: return self.opr.unitary and all(tok.unitary for tok in self.children)


class Limit(Branch):
	sep: str = SEP_DATA
	def check(self) -> 'Limit':

		# DATA LIST
		if len(self.children)>1:
			if self.opr.kind not in {'EQL','NOT'}: raise SyntaxError('E_DATA_OPR')
			if any(c.kind not in DATA_KINDS for c in self.children): raise SyntaxError('E_DATA_TYPE')

		# DATUM
		else:
			kind = self.children[0].kind
	
			# DESUGAR EQL DIFF to NOT SAME
			if kind == 'DIFF':
				if self.opr.kind != 'EQL': raise SyntaxError('E_OPR')
				self.opr = TOK_NOT
				self.children = [TOK_SAME]

			# DESUGAR EQL WILD to NOT EMPTY
			elif kind == 'WILD':
				if self.opr.kind != 'EQL': raise SyntaxError('E_OPR')
				self.opr = TOK_NOT
				self.children = [TOK_EMPTY]

			elif kind not in DATA_KINDS: raise SyntaxError('E_DATA_TYPE')

			# VAR MUST BIND TO NON-UNITARY QUERY
			elif self.var and self.unitary: raise SyntaxError('E_VAR_UNI')

		return self

	@property
	def unitary(self) -> bool: return self.opr.unitary and len(self.children)==1 and self.children[0].unitary


TOK_EQL_SAME = Limit(None, [TOK_SAME])
TOK_NOT_EMPTY = Limit(TOK_NOT, [TOK_EMPTY])
TOK_EQL_EMPTY = Limit(None, [TOK_EMPTY])


class Vector(Branch):
	sep: str = SEP_LIMIT
	children: List[Limit]


class Matrix(Branch):
	sep = SEP_VCTR
	children: List[Vector]

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
			if pad_wild>0: self.children[vctr_axis].children.extend([TOK_NOT_EMPTY] * pad_wild)


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
	limit=Limit()
	var = None
	while tokens.peek():

		# LIMIT ::= [[VAR] OPR] DATUM {SEP_DATA DATUM}
		# Single axis constraint

		# [VAR]
		if tokens.peek() == 'VAR':
			var = tokens.next()
			# 1. VAR OPR -> $x=* -> bind
			if tokens.peek() in OPR_KINDS: limit.var = var
			# 2. VAR SEP_KINDS -> $x -> read
			if tokens.peek() in SEP_KINDS:
				limit.append(var)
				if tokens.peek()=='SEP_DATA': tokens.next() # CONSUME COMMA

		# [OPR]
		if tokens.peek() in OPR_KINDS:
			limit.reopr(tokens.next())
			if tokens.peek()=='SEP_LIMIT': raise SyntaxError('E_SPACE')
			if tokens.peek() not in DATA_KINDS|SUGAR_KINDS: raise SyntaxError('E_COMMA')

		# DATUM {SEP_DATA DATUM}
		if tokens.peek() in DATA_KINDS|SUGAR_KINDS:
			limit.append(tokens.next())
			while tokens.peek()=='SEP_DATA':
				tokens.next()
				if tokens.peek()=='SEP_LIMIT': raise SyntaxError('E_SPACE')
				if tokens.peek() not in DATA_KINDS: raise SyntaxError('E_DATA_KIND')
				limit.append(tokens.next())

		# Consume spaces
		if tokens.peek()=='SEP_LIMIT':
			if limit.children: vctr.append(limit.check())
			limit = Limit()
			tokens.next()
			continue

		# VCTR ::= LIMIT {SEP_LIMIT LIMIT}
		# Conjunctive vector of axis constraints
		if tokens.peek() == 'SEP_VCTR':
			if limit.children: vctr.append(limit.check())
			if vctr.children: mtrx.append(vctr.check())
			limit = Limit()
			vctr = Vector()
			tokens.next()
			continue

		# MTRX ::= VCTR {SEP_VCTR VCTR}
		# Conjunctive matrix of axis constraints
		if tokens.peek() == 'SEP_MTRX':
			if limit.children: vctr.append(limit.check())
			if vctr.children: mtrx.append(vctr.check())
			if mtrx.children: yield mtrx.check()
			limit = Limit()
			vctr = Vector()
			mtrx = Matrix()
			tokens.next()
			continue

		raise SyntaxError('E_TOK')

	if limit.children: vctr.append(limit.check())
	if vctr.children: mtrx.append(vctr.check())
	if mtrx.children: yield mtrx.check()


class Meme(Branch):
	sep: str = SEP_MTRX
	children: List[Matrix]
	results: List[List[List[List[Token]]]]
	bindings: Dict[str, Tuple[Axis, Axis, Axis]]

	def __init__(self, src: str):
		self.src = src
		self.bindings = {}
		self.children = list(parse(src))

	def cylindrify(self) -> None:
		for mtrx in self.children: mtrx.cylindrify()
		self.results = [[[[] for limit in vctr.children] for vctr in mtrx.children] for mtrx in self.children]

	def store(self): 
		for mtrx_axis, mtrx in enumerate(self.children):
			if not mtrx.unitary: raise SyntaxError('E_UNI_MTRX')
			for vctr_axis, vctr in enumerate(mtrx.children):
				if not vctr.unitary: raise SyntaxError('E_UNI_VCTR')
				for limit_axis, limit in enumerate(vctr.children):
					if limit is TOK_NOT_EMPTY: self.results[mtrx_axis][vctr_axis][limit_axis] = [Token('INT', str(random.randrange(RAND_INT_MIN, RAND_INT_MAX)))]
					elif limit is TOK_EQL_SAME:
						if limit_axis == 0: raise SyntaxError('E_SAME_ZERO')
						self.results[mtrx_axis][vctr_axis][limit_axis] = self.results[mtrx_axis][vctr_axis][limit_axis-1]
					else: self.results[mtrx_axis][vctr_axis][limit_axis] = limit.children

	def check(self) -> 'Meme':
		for mxtr_axis, mtrx in enumerate(self.children):
			if not isinstance(mtrx, Matrix): raise TypeError('E_TYPE_VCTR')
			for vctr_axis, vctr in enumerate(mtrx.children):
				if not isinstance(vctr, Vector): raise TypeError('E_TYPE_VCTR')
				for limit_axis, limit in enumerate(vctr.children):
					if not isinstance(limit, Limit): raise TypeError('E_TYPE_LIMIT')
					if limit.var: self.bindings[limit.var.lexeme] = (mxtr_axis, vctr_axis, limit_axis)

		self.results = [[[[] for limit in vctr.children] for vctr in mtrx.children] for mtrx in self.children]


	def expand(self, tokens: List[Token], from_limit_axis: Axis, from_vctr_axis: Axis, from_mtrx_axis: Axis) -> List[Token]:
		expansion: List[Token] = []
		for tok in tokens:
			if tok.kind == 'SAME':
				if from_vctr_axis < 1: raise SyntaxError('E_SAME_OOB')
				expansion.extend(self.results[from_mtrx_axis][from_vctr_axis-1][from_limit_axis])
			elif tok.kind == 'VAR':
				axes = self.bindings[tok.lexeme]
				expansion.extend(self.results[axes[0]][axes[1]][axes[2]])
			else: expansion.append(tok)
		return expansion


def intersect(query_limit: Limit, store_limit: Limit) -> Limit:
	store_children = list(store_limit.children)
	if not store_children: store_children=[Token('EMPTY', EMPTY)]

	intersection = Limit()

	query_opr, query_datums = query_limit.dump()

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