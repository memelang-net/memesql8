'''
Meme v8.12 | info@memelang.net | (c) HOLTWORK LLC | Patents Pending
This script is optimized for training LLMs

1. MEMELANG USES AXES, LIMIT_AXIS HIGH -> LOW
SQL ANALOG:  3->Table_Name  2->Row_Primary_Key  1->Column_Name     0->Value
RDF ANALOG:  3->Graph_Name  2->Subject_URI      1->Predicate_Name  0->Object_Value

2. EXAMPLE QUERY
MEMELANG: movies * actor "Mark Hamill",Mark ; movie * ; rating >4 ;;
SQL: SELECT ... FROM movies WHERE rowid=* AND actor IN ("Mark Hamill", "Mark") AND movie=* AND rating>4
SPARQL: SELECT … WHERE { GRAPH <movies> {?s actor ?o . FILTER(?o IN ("Mark Hamill","Mark")) . ?s movie ?x . ?s rating ?r . FILTER(?r > 4)} }

3. VARIABLE EXAMPLE ACTOR NAME = MOVIE TITLE
MEMELANG: movies * actor $x=* ; movie $x ;;
SQL: SELECT ... FROM movies WHERE actor=movie

4. EXAMPLE JOIN QUERY
MEMELANG: movies * actor "Mark Hamill" ; movie * ; ~ @ @ ; actor * ;;
MEMELANG: movies $rowid=* actor "Mark Hamill" ; movie * ; !$rowid @ @ ; actor !"Mark Hamill" ;;
SQL: SELECT co.actor FROM movies AS mh JOIN movies AS co ON co.movie=mh.movie AND co.rowid!=mh.rowid WHERE mh.actor='Mark Hamill';
RDF: SELECT ?coActor WHERE { GRAPH <movies> { ?mhRow ex:actor "Mark Hamill" ; ex:movie ?movie . ?coRow ex:movie ?movie ; ex:actor ?coActor . FILTER ( ?coRow != ?mhRow ) } }
'''

import random, re, json, operator
from typing import List, Iterator, Iterable, Dict, Tuple, Any, Union

RAND_INT_MIN = 1 << 20
RAND_INT_MAX = 1 << 53

Axis = int # >=0
Datum = Union[str, float, int]

SIGIL, WILD, SAME, DIFF, EMPTY, EOF =  '$', '*', '@', '~', '_', None

SEP_LIMIT, SEP_DATA, SEP_VCTR, SEP_MTRX = ' ', ',', ';', ';;'
SEP_VCTR_PRETTY, SEP_MTRX_PRETTY = ' ; ', ' ;;\n'

TOKEN_KIND_PATTERNS = (
	('COMMENT',		r'//[^\n]*'),
	('QUOTE',		r'"(?:[^"\\]|\\.)*"'),	# ALWAYS JSON QUOTE ESCAPE EXOTIC CHARS name="John \"Jack\" Kennedy"

	('SEP_MTRX',	re.escape(SEP_MTRX)),	# MTRX DISJUNCTION, AXIS=0
	('SEP_VCTR',	re.escape(SEP_VCTR)),	# VCTR CONJUNCTION, AXIS=0
	('SEP_LIMIT',	r'\s+'),				# LIMIT CONJUNCTION, AXIS+=1
	('SEP_DATA',	re.escape(SEP_DATA)),	# DATUM DISJUNCTION, AXIS SAME

	('GE',			r'>='),
	('LE',			r'<='),
	('EQL',			r'='),
	('NOT',			r'!'),
	('GT',			r'>'),
	('LT',			r'<'),

	('WILD',		re.escape(WILD)),		# WILDCARD, NEVER QUOTE
	('SAME',		re.escape(SAME)),		# EQUALS DATA FROM (VCTR_AXIS-1, LIMIT_AXIS)
	('DIFF',		re.escape(DIFF)),		# NOT EQUALS DATA FROM (VCTR_AXIS-1, LIMIT_AXIS)
	('EMPTY',		re.escape(EMPTY)),		# EMPTY SET, ANTI-WILD
	('VAR',			rf'\$[A-Za-z0-9]+'),
	
	('IDENT',		r'[A-Za-z][A-Za-z0-9]*'), # ALPHANUMERIC IDENTIFIERS ARE UNQUOTED
	('FLOAT',		r'-?\d*\.\d+'),
	('INT',			r'-?\d+'),
	('MISMATCH',	r'.'),
)

MASTER_PATTERN = re.compile('|'.join(f'(?P<{kind}>{pat})' for kind, pat in TOKEN_KIND_PATTERNS))

OPR_DICT = {'EQL': operator.eq, 'NOT': operator.ne, 'GT': operator.gt, 'GE': operator.ge, 'LT': operator.lt, 'LE': operator.le}
OPR_DATA_KINDS = {'EQL','NOT'}
SEP_KINDS = {'SEP_MTRX','SEP_VCTR','SEP_LIMIT','SEP_DATA',EOF}
SUGAR_KINDS = {'DIFF', 'WILD'}
DATA_KINDS = {'IDENT', 'QUOTE', 'INT', 'FLOAT', 'VAR', 'SAME', 'EMPTY'} # NEVER DIFF OR WILD IN MULTI-DATA LIST
UNITARY_KINDS = {'IDENT', 'QUOTE', 'INT', 'FLOAT', 'EQL', 'DATUM', 'NOVAR', 'SAME'}

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
TOK_DATUM = Token('DATUM', '') # ELIDED
TOK_NOVAR = Token('NOVAR', '') # ELIDED
TOK_NOT = Token('NOT', '!')
TOK_GT = Token('GT', '>')
TOK_SEP_DATA = Token('SEP_DATA', SEP_DATA)
TOK_SEP_LIMIT = Token('SEP_LIMIT', SEP_LIMIT)
TOK_SEP_VCTR = Token('SEP_VCTR', SEP_VCTR)
TOK_SEP_MTRX = Token('SEP_MTRX', SEP_MTRX)


class Stream:
	def __init__(self, token: Iterable[Token]):
		self.token: Iterator[Token] = iter(token)
		self.buffer: list[Token] = []
	def peek(self, fwd: int = 1) -> str|None: 
		while(len(self.buffer)<fwd):
			val = next(self.token, EOF)
			if val is EOF: return EOF
			self.buffer.append(val)
		return self.buffer[fwd-1].kind
	def next(self) -> Token: 
		if not self.buffer:
			val = next(self.token, EOF)
			if val is EOF: raise SyntaxError('E_EOF')
			self.buffer.append(val)
		return self.buffer.pop(0)


class Olist(list):
	opr: Token = TOK_EQL
	def __init__(self, *items: List[Olist|Token], opr:Token|None = None):
		super().__init__(items)
		if opr is not None: self.opr = opr

	def pad(self, padding:Olist|Token) -> None:
		max_len = len(self[0])
		for idx, item in enumerate(self):
			diff = max_len - len(item)
			if diff>0: self[idx][:0] = [padding] * diff
			elif diff<0: raise SyntaxError('E_PAD')

	@property
	def unitary(self) -> bool: return self.opr.unitary and all(item.unitary for item in self)
	def dump(self) -> list: return [self.opr.dump(), [item.dump() for item in self]]
	def check(self) -> 'Olist': 
		if len(self)==0: raise SyntaxError('E_NO_LIST')
		return self
	def __str__(self) -> str: return self.opr.lexeme.join(map(str, self))

class Data(Olist):
	opr: Token = TOK_DATUM

class Vector(Olist):
	opr: Token = TOK_SEP_LIMIT

class Limit(Olist):
	opr: Token = TOK_EQL # ELIDED '='

class Matrix(Olist):
	opr: Token = TOK_SEP_VCTR

DATA_SAME = Data(Token('SAME', SAME))
DATA_EMPTY = Data(Token('EMPTY', EMPTY))
LIMIT_EQL_SAME = Limit(TOK_NOVAR, DATA_SAME, opr=TOK_EQL)

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
	while tokens.peek():

		# LIMIT ::= [[VAR] OPR] DATUM {SEP_DATA DATUM}
		# Single axis constraint

		# [VAR]
		var = TOK_NOVAR
		if tokens.peek() == 'VAR':
			if tokens.peek(2) in OPR_DICT: var = tokens.next()
			elif tokens.peek(2) not in SEP_KINDS: raise SyntaxError('E_VAR_NXT')

		# [OPR]
		if tokens.peek() in OPR_DICT:
			limit.opr=tokens.next()
			if tokens.peek()=='SEP_LIMIT': raise SyntaxError('E_SPACE')
			if tokens.peek() not in DATA_KINDS|SUGAR_KINDS: raise SyntaxError('E_OPR_DAT')

		# DATUM {SEP_DATA DATUM}
		if tokens.peek() in DATA_KINDS|SUGAR_KINDS:
			data=Data()
			data.append(tokens.next())
			while tokens.peek()=='SEP_DATA':
				if len(data)==1:
					if data[0].kind not in DATA_KINDS: raise SyntaxError('E_DATA_KIND')
					if limit.opr.kind not in OPR_DATA_KINDS: raise SyntaxError('E_DATA_OPR')
				data.opr = tokens.next()
				if tokens.peek()=='SEP_LIMIT': raise SyntaxError('E_SPACE')
				if tokens.peek() not in DATA_KINDS: raise SyntaxError('E_DATA_KIND')
				data.append(tokens.next())

			# Finalize LIMIT
			limit.append(var)
			limit.append(data)
			vctr.append(limit.check())
			limit=Limit()
			continue

		# VCTR ::= LIMIT {SEP_LIMIT LIMIT}
		# Conjunctive vector of axis constraints
		if tokens.peek() == 'SEP_VCTR':
			vctr.reverse() # LIMIT_AXIS: HIGH -> LOW
			if vctr: mtrx.append(vctr.check())
			vctr = Vector()
			tokens.next()
			continue

		# MTRX ::= VCTR {SEP_VCTR VCTR}
		# Conjunctive matrix of axis constraints
		if tokens.peek() == 'SEP_MTRX':
			vctr.reverse() # LIMIT_AXIS: HIGH -> LOW
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

	if vctr:
		vctr.reverse() # LIMIT_AXIS: HIGH -> LOW
		mtrx.append(vctr.check())
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
			for vctr_axis, vctr in enumerate(mtrx):
				for limit_axis, limit in enumerate(vctr):
					if not limit.unitary: raise SyntaxError('E_LIMIT_UNIT')
					if limit.dump() == LIMIT_EQL_SAME.dump():
						if limit_axis == 0: raise SyntaxError('E_SAME_ZERO')
						self.results[mtrx_axis][vctr_axis][limit_axis].extend(self.results[mtrx_axis][vctr_axis-1][limit_axis])
					else: self.results[mtrx_axis][vctr_axis][limit_axis].extend(limit[1])

	def check(self) -> 'Meme':
		for mtrx_axis, mtrx in enumerate(self):
			if not isinstance(mtrx, Matrix): raise TypeError('E_TYPE_VCTR')
			for vctr_axis, vctr in enumerate(mtrx):
				if not isinstance(vctr, Vector): raise TypeError('E_TYPE_VCTR')
				for limit_axis, limit in enumerate(vctr):
					if not isinstance(limit, Limit): raise TypeError('E_TYPE_LIMIT')
					if limit[0].kind=='VAR': self.bindings[limit[0].lexeme] = (mtrx_axis, vctr_axis, limit_axis)
			self[mtrx_axis].pad(LIMIT_EQL_SAME)

		self.results = [[[[] for limit in vctr] for vctr in mtrx] for mtrx in self]


	def expand(self, data: Data, from_limit_axis: Axis, from_vctr_axis: Axis, from_mtrx_axis: Axis) -> Data:
		expansion=Data()
		for tok in data:
			if tok.kind == 'SAME':
				if from_vctr_axis < 1: raise SyntaxError('E_SAME_OOB')
				expansion.extend(self.results[from_mtrx_axis][from_vctr_axis-1][from_limit_axis])
			elif tok.kind == 'VAR':
				if tok.lexeme not in self.bindings: raise SyntaxError('E_VAR_BIND')
				axes = self.bindings[tok.lexeme]
				expansion.extend(self.results[axes[0]][axes[1]][axes[2]])
			else: expansion.append(tok)
		if len(expansion)>1: expansion.opr = TOK_SEP_DATA
		return expansion.check()


def desugar(limit: Limit) -> Limit:
	if limit[1][0].kind == 'DIFF':
		if limit.opr.kind != 'EQL': raise SyntaxError('E_OPR_DIFF')
		return Limit(limit[0], DATA_SAME, opr=TOK_NOT)
	if limit[1][0].kind == 'WILD':
		if limit.opr.kind == 'EQL': opr=TOK_NOT
		elif limit.opr.kind == 'NOT': opr=TOK_EQL
		else: opr = TOK_GT # WILD MATCHES ANY NUMERIC
		return Limit(limit[0], DATA_EMPTY, opr=opr)
	return limit


def intersect(query: Limit, store: Data) -> Data:
	if not store: store=DATA_EMPTY

	query = desugar(query)
	opr_kind, intersection, query_data = query.opr.kind, Data(), [t.datum for t in query[1]]

	if opr_kind == 'EQL': intersection.extend([t for t in store if t.datum in query_data])
	elif opr_kind == 'NOT': intersection.extend([t for t in store if t.datum not in query_data])

	# RETURN ANY NUMERIC FOR GT EMPTY
	elif EMPTY in query_data: intersection.extend([t for t in store if t.kind in {'INT','FLOAT'}])

	elif len(query_data)!=1 or not isinstance(query_data[0], (int,float)): raise TypeError('E_INTER_NUM')

	intersection.extend([t for t in store if t.kind in {'INT','FLOAT'} and OPR_DICT[opr_kind](t.datum, query_data[0])])
	return intersection


# BELOW ARE DEMO FUNCTIONS FOR LLM TRAINING

Table, Column, SqlOperator, Value = str, str, str, Any
def demo_translate(table: Table, covs:List[Tuple[Column, SqlOperator, Value]]) -> Tuple[str, str]:
	if not covs: raise ValueError('covs')
	if any(isinstance(val,(list,tuple)) and opr.lower() not in ('in','not in') for _,opr,val in covs): raise SyntaxError('data')

	def sql_opr(opr: SqlOperator) -> SqlOperator:
		if opr.lower() not in {'=','!=','>','>=','<','<=','in','not in'}: raise SyntaxError('opr')
		return opr

	def sql_val(val: Value) -> Value:
		if isinstance(val, (list, tuple)): return '(' + ','.join(sql_val(v) for v in val) + ')'
		return "'"+str(val).replace("'","''")+"'"

	def memelang_opr(opr:SqlOperator) -> SqlOperator:
		if opr.lower() in ('=', 'in'): opr=''
		elif opr.lower() in ('!=', 'not in'): opr='!'
		return opr
	
	def memelang_val(val: Value) -> Value:
		if isinstance(val, (list, tuple)): return ','.join(memelang_val(v) for v in val)
		if isinstance(val, str) and re.fullmatch(r'[A-Za-z0-9]+', val): return val
		return json.dumps(val)

	sql_predicates=[col+' '+sql_opr(opr)+' '+sql_val(val) for col,opr,val in covs]
	sql = 'SELECT ' + ','.join([col for col,_,_ in covs]) + ' FROM ' + table + ' WHERE ' + ' AND '.join(sql_predicates) + ';'
	
	memelang_predicates=[[memelang_opr(opr) + memelang_val(val), memelang_opr('=') + col] for col,opr,val in covs]
	memelang_predicates[0].insert(0, table)
	memelang_predicates[0].insert(1, WILD)
	memelang = SEP_VCTR_PRETTY.join(SEP_LIMIT.join(p) for p in memelang_predicates) + SEP_MTRX_PRETTY

	return sql, str(Meme(memelang))

def demo_generate() -> Meme:
	def rand_datum(kind:str, i:int=1) -> str:
		if kind=='IDENT': return ''.join(random.choice('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ') for _ in range(i))
		if kind=='QUOTE': return json.dumps(''.join(random.choice(' -_+,./<>[]{}\'"!@#$%^&*()abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ') for _ in range(i)))
		if kind=='INT': return str(random.randint(-100, 100))
		if kind=='FLOAT': return str(random.uniform(-100, 100))
		if kind=='VAR': return SIGIL + rand_datum('IDENT', 3)

	prior_vars = []
	vector = []

	limit_len = random.randint(2, 10)
	for _ in range(limit_len):
		var = ''
		do_assign_variable = random.randint(0, 1)
		if do_assign_variable: var += rand_datum('VAR',3)

		opr = random.choice(['=','!','>','<','<=','>='])

		data: Value = ''
		if opr in {'=','!'}:
			data_list_len = random.randint(1, 5)
			data_list: List[Any] = []
			for _ in range(data_list_len):
				datum_type = random.randint(1, 10)
				if datum_type == 1:  data_list.append(rand_datum('QUOTE',5))
				elif datum_type == 2:  data_list.append(rand_datum('INT'))
				elif datum_type == 3:  data_list.append(rand_datum('FLOAT'))
				elif datum_type == 4 and prior_vars: data_list.append(random.choice(prior_vars))
				elif datum_type == 5 and vector: data_list.append(SAME)
				elif datum_type == 6 and vector and opr == '=' and data_list_len == 1: data_list.append(DIFF)
				else: data_list.append(rand_datum('IDENT', 5))
			data += SEP_DATA.join(data_list)

		else:
			data = str(random.uniform(-100, 100))

		if var:
			assert opr
			prior_vars.append(var)
		elif not var and opr == '=': opr = '' # ELIDED '='

		vector.append(var + opr + data)

	return str(Meme(SEP_VCTR.join(vector) + SEP_MTRX))