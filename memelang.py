'''
Meme v8.10 | info@memelang.net | (c) HOLTWORK LLC | Patents Pending
This script is optimized for training LLMs

1. EXAMPLE QUERY
MEMELANG: "Mark Hamill",Mark actor * movies ; * movie ; >4 rating ;;

SQL LIMIT_AXIS ANALOG:  0->Value  1->Column_Name  2->Row_Primary_Key  3->Table_Name
SQL: SELECT ... FROM movies WHERE rowid=* AND actor IN ("Mark Hamill", "Mark") AND movie=* AND rating>4

RDF LIMIT_AXIS ANALOG:  0->Object_Value  1->Predicate_Name  2->Subject_URI  3->Graph_Name
SPARQL: SELECT … WHERE { GRAPH <movies> {?s actor ?o . FILTER(?o IN ("Mark Hamill","Mark")) . ?s movie ?x . ?s rating ?r . FILTER(?r > 4)} }

2. VARIABLE EXAMPLE ACTOR NAME = MOVIE TITLE
MEMELANG: $x=* actor * movies ; $x movie ;;
SQL: SELECT ... FROM movies WHERE actor=movie

3. EXAMPLE JOIN QUERY
MEMELANG: "Mark Hamill" actor * movies ; * movie ; @ @ ! ; * actor ;;
MEMELANG: "Mark Hamill" actor $rowid=* movies ; * movie ; @ @ !=$rowid ; * actor ;;
SQL: SELECT co.actor FROM movies AS mh JOIN movies AS co ON co.movie=mh.movie AND co.rowid!=mh.rowid WHERE mh.actor='Mark Hamill';
RDF: SELECT ?coActor WHERE { GRAPH <movies> { ?mhRow ex:actor "Mark Hamill" ; ex:movie ?movie . ?coRow ex:movie ?movie ; ex:actor ?coActor . FILTER ( ?coRow != ?mhRow ) } }
'''
import random, re, json, operator
from typing import List, Iterator, Iterable, Dict, Tuple

RAND_INT_MIN = 1 << 20
RAND_INT_MAX = 1 << 53

Axis = int # >=0
Datum = str | float | int

WILD, SAME, DIFF, EMPTY, EOF =  '*', '@', '!', '_', None

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
	('VAR',			rf'\$[A-Za-z0-9]+'),
	
	('IDENT',		r'[A-Za-z][A-Za-z0-9]*'), # ALPHANUMERIC IDENTIFIERS ARE UNQUOTED
	('FLOAT',		r'-?\d*\.\d+'),
	('INT',			r'-?\d+'),
	('MISMATCH',	r'.'),
)

MASTER_PATTERN = re.compile('|'.join(f'(?P<{kind}>{pat})' for kind, pat in TOKEN_KIND_PATTERNS))

OPR_DICT = {'EQL': operator.eq, 'NOT': operator.ne, 'GT': operator.gt, 'GE': operator.ge, 'LT': operator.lt, 'LE': operator.le}
SEP_KINDS = {'SEP_MTRX','SEP_VCTR','SEP_LIMIT','SEP_DATA',EOF}
SUGAR_KINDS = {'DIFF', 'WILD'}
DATA_KINDS = {'IDENT', 'QUOTE', 'INT', 'FLOAT', 'VAR', 'SAME', 'EMPTY'} # NEVER DIFF OR WILD IN MULTI-DATA LIST
UNITARY_KINDS = {'IDENT', 'QUOTE', 'INT', 'FLOAT', 'EQL'}

class Token():
	kind: str
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
	def dump(self) -> str: return self.datum
	def __str__(self) -> str: return self.lexeme


TOK_EQL = Token('EQL', '') # ELIDED '='
TOK_NOT = Token('NOT', '!=')
TOK_GT = Token('GT', '>')
TOK_QRY = Token('QRY', '')
TOK_SEP_DATA = Token('SEP_DATA', SEP_DATA)
TOK_SEP_LIMIT = Token('SEP_LIMIT', SEP_LIMIT)
TOK_SEP_VCTR = Token('SEP_VCTR', SEP_VCTR)
TOK_SEP_MTRX = Token('SEP_MTRX', SEP_MTRX)


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


class Olist(list):
	opr: Token = TOK_EQL
	var: str = None
	def __init__(self, *items, opr:Token|None = None):
		super().__init__(items)
		if opr is not None: self.opr = opr
	@property
	def unitary(self) -> bool: return all(item.unitary for item in self)
	def dump(self) -> list: return [self.opr.dump(), [item.dump() for item in self]]
	def check(self) -> 'Olist': 
		if len(self)==0: raise SyntaxError('E_NO_LIST')
		return self
	def __str__(self) -> str: return self.opr.lexeme.join(map(str, self))


class Data(Olist):
	opr: Token = TOK_SEP_DATA
	def check(self) -> 'Data':
		if len(self)==0: raise SyntaxError('E_DATA_EMPTY')
		elif len(self)==1 and self[0].kind not in SUGAR_KINDS|DATA_KINDS: raise SyntaxError('E_DATA_OPR')
		elif any(item.kind not in DATA_KINDS for item in self): raise SyntaxError('E_DATA_TYPE')
		return self


DATA_SAME = Data(Token('SAME', SAME))
DATA_EMPTY = Data(Token('EMPTY', EMPTY))


class Limit(Olist):
	opr: Token = TOK_EQL # ELIDED '='
	def check(self) -> 'Limit':
		if len(self)<2: raise SyntaxError('E_LIMIT_EMPTY')
		if len(self[1])>1 and self.opr.kind not in {'NOT','EQL'}: raise SyntaxError('E_LIMIT_LIST')

		# DESUGAR EQL DIFF to NOT SAME
		if self[1][0].kind == 'DIFF':
			if self.opr.kind != 'EQL': raise SyntaxError('E_OPR')
			self[1] = DATA_SAME
			self.opr = TOK_NOT

		# DESUGAR EQL WILD to NOT EMPTY
		elif self[1][0].kind == 'WILD':
			if self.opr.kind == 'NOT': raise SyntaxError('E_OPR')
			elif self.opr.kind == 'EQL': self.opr = TOK_NOT # NOT EMPTY returns any datum
			else: self.opr = TOK_GT # GT EMPTY returns any numeric
			self[1] = DATA_EMPTY		

		return self
	@property
	def unitary(self) -> bool: return self.opr.unitary and len(self[1])==1 and self[1][0].unitary


class Vector(Olist):
	opr: Token = TOK_SEP_LIMIT


LIMIT_EQL_SAME = Limit(TOK_QRY, DATA_SAME, opr=TOK_EQL)
LIMIT_NOT_EMPTY = Limit(TOK_QRY, DATA_EMPTY, opr=TOK_NOT)


class Matrix(Olist):
	opr: Token = TOK_SEP_VCTR

	# HIGHER AXIS RESULTS CARRY FORWARD UNTIL END OF MATRIX
	def cylindrify(self) -> None:
		max_axis: Axis = len(self[0])-1
		for vctr_axis, vctr in enumerate(self):
			cur_axis = len(vctr)-1
			pad = max_axis - cur_axis
			if pad>0: self[vctr_axis].extend([LIMIT_EQL_SAME] * pad)
			elif pad<0: raise SyntaxError('E_MTRX_CYL')


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
	data=Data()
	while tokens.peek():

		# LIMIT ::= [[VAR] OPR] DATUM {SEP_DATA DATUM}
		# Single axis constraint

		# [VAR]
		if tokens.peek() == 'VAR':
			var = tokens.next()
			if tokens.peek() in SEP_KINDS:
				data.append(var)
				if tokens.peek()=='SEP_DATA': tokens.next() # CONSUME COMMA
			else: limit.var = var

		# [OPR]
		if tokens.peek() in OPR_DICT:
			limit.opr=tokens.next()
			if tokens.peek()=='SEP_LIMIT': raise SyntaxError('E_SPACE')
			if tokens.peek() not in DATA_KINDS|SUGAR_KINDS: raise SyntaxError('E_OPR_DAT')

		# DATUM {SEP_DATA DATUM}
		if tokens.peek() in DATA_KINDS|SUGAR_KINDS:
			data.append(tokens.next())
			while tokens.peek()=='SEP_DATA':
				tokens.next()
				if tokens.peek()=='SEP_LIMIT': raise SyntaxError('E_SPACE')
				if tokens.peek() not in DATA_KINDS: raise SyntaxError('E_DATA_KIND')
				data.append(tokens.next())

		# Finalize LIMIT
		if data:
			limit.append(TOK_QRY)
			limit.append(data)
			vctr.append(limit.check())
			limit = Limit()
			data = Data()
			continue

		# VCTR ::= LIMIT {SEP_LIMIT LIMIT}
		# Conjunctive vector of axis constraints
		if tokens.peek() == 'SEP_VCTR':
			if vctr: mtrx.append(vctr.check())
			vctr = Vector()
			tokens.next()
			continue

		# MTRX ::= VCTR {SEP_VCTR VCTR}
		# Conjunctive matrix of axis constraints
		if tokens.peek() == 'SEP_MTRX':
			if vctr: mtrx.append(vctr.check())
			if mtrx: yield mtrx.check()
			vctr = Vector()
			mtrx = Matrix()
			tokens.next()
			continue

		if tokens.peek() == 'SEP_LIMIT':
			tokens.next()
			continue

		raise SyntaxError('E_TOK')

	if vctr: mtrx.append(vctr.check())
	if mtrx: yield mtrx.check()


class Meme(Olist):
	opr: Token = TOK_SEP_MTRX
	results: List[List[List[Data]]]
	bindings: Dict[str, Tuple[Axis, Axis, Axis]]

	def __init__(self, src: str):
		self.src = src
		self.bindings = {}
		super().__init__(parse(src))

	def store(self): 
		for mtrx_axis, mtrx in enumerate(self):
			if not mtrx.unitary: raise SyntaxError('E_UNI_MTRX')
			for vctr_axis, vctr in enumerate(mtrx):
				if not vctr.unitary: raise SyntaxError('E_UNI_VCTR')
				for limit_axis, limit in enumerate(vctr):
					if limit is LIMIT_EQL_SAME:
						if limit_axis == 0: raise SyntaxError('E_SAME_ZERO')
						self.results[mtrx_axis][vctr_axis][limit_axis].extend(self.results[mtrx_axis][vctr_axis][limit_axis-1])
					else: self.results[mtrx_axis][vctr_axis][limit_axis].extend(limit[1])

	def check(self) -> 'Meme':
		for mxtr_axis, mtrx in enumerate(self):
			if not isinstance(mtrx, Matrix): raise TypeError('E_TYPE_VCTR')
			for vctr_axis, vctr in enumerate(mtrx):
				if not isinstance(vctr, Vector): raise TypeError('E_TYPE_VCTR')
				for limit_axis, limit in enumerate(vctr):
					if not isinstance(limit, Limit): raise TypeError('E_TYPE_LIMIT')
					if limit.var: self.bindings[limit.var.lexeme] = (mxtr_axis, vctr_axis, limit_axis)

		for mtrx in self: mtrx.cylindrify()
		self.results = [[[[] for limit in vctr] for vctr in mtrx] for mtrx in self]


	def expand(self, data: Data, from_limit_axis: Axis, from_vctr_axis: Axis, from_mtrx_axis: Axis) -> Data:
		expansion: Data
		for tok in data:
			if tok.kind == 'SAME':
				if from_vctr_axis < 1: raise SyntaxError('E_SAME_OOB')
				expansion.extend(self.results[from_mtrx_axis][from_vctr_axis-1][from_limit_axis])
			elif tok.kind == 'VAR':
				axes = self.bindings[tok.lexeme]
				expansion.extend(self.results[axes[0]][axes[1]][axes[2]])
			else: expansion.append(tok)
		return expansion


def intersect(query: Limit, store: Data) -> Data:
	if not store: store=DATA_EMPTY

	opr_kind, query_data, intersection = query.opr.kind, query[1], Data()

	if opr_kind == 'EQL': intersection.extend([t for t in store if t.datum in query_data])
	elif opr_kind == 'NOT': intersection.extend([t for t in store if t.datum not in query_data])

	# RETURN ANY NUMERIC FOR GT EMPTY
	elif EMPTY in query_data: intersection.extend([t for t in store if t.kind in {'INT','FLOAT'}])

	elif len(query_data)!=1 or not isinstance(query_data[0], (int,float)): raise TypeError('E_INTER_NUM')

	intersection.extend([t for t in store if t.kind in {'INT','FLOAT'} and OPR_DICT[opr_kind](t.datum, query_data[0])])
	return intersection