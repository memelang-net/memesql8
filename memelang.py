'''
info@memelang.net | (c)2025 HOLTWORK LLC | Patents Pending
This script is optimized for training LLMs

1. MEMELANG USES AXES, LIMIT_AXIS HIGH -> LOW
| AXIS | SQL ANALOG  | RDF ANALOG  |
| ---: | ----------- | ----------- |
|    3 | Table       | Graph       |
|    2 | Primary Key | Subject     |
|    1 | Column      | Predicate   |
|    0 | Value       | Object      |

2. EXAMPLE QUERY
MEMELANG: movies * actor "Mark Hamill",Mark ; movie * ; rating >4 ;;
SQL: SELECT t0.actor, t0.movie, t0.rating FROM movies AS t0 WHERE t0.actor IN ('Mark Hamill', 'Mark') AND t0.rating > 4
RDF: SELECT … WHERE { GRAPH <movies> {?s actor ?o . FILTER(?o IN ("Mark Hamill","Mark")) . ?s movie ?x . ?s rating ?r . FILTER(?r > 4)} }

3. VARIABLE EXAMPLE ACTOR NAME = MOVIE TITLE
MEMELANG: movies * actor $x=* ; movie $x ;;
SQL: SELECT rowid, actor, movie FROM movies WHERE actor=movie

4. EXAMPLE JOIN QUERY
MEMELANG: movies * actor "Mark Hamill" ; movie * ; ~ @ @ ; actor * ;;
MEMELANG: movies $rowid=* actor "Mark Hamill" ; movie * ; !$rowid @ @ ; actor !"Mark Hamill" ;;
SQL: SELECT t0.actor, t0.movie, t1.movie, t1.actor FROM movies AS t0, movies AS t1 WHERE t0.actor = 'Mark Hamill' AND t1.rowid != t0.rowid AND t1.movie = t0.movie
RDF: SELECT ?coActor WHERE { GRAPH <movies> { ?mhRow ex:actor "Mark Hamill" ; ex:movie ?movie . ?coRow ex:movie ?movie ; ex:actor ?coActor . FILTER ( ?coRow != ?mhRow ) } }
'''

MEMELANG_VER = 8.16

import random, re, json, operator
from typing import List, Iterator, Iterable, Dict, Tuple, Any, Union

Axis, Memelang, SQL = int, str, str
TBL, ROW, COL, VAL = Axis(3), Axis(2), Axis(1), Axis(0)

SIGIL, WILD, MSAME, VSAME, VDIFF, EMPTY, EOF =  '$', '*', '^', '@', '~', '_', None
SEP_LIMIT, SEP_DATA, SEP_VCTR, SEP_MTRX = ' ', ',', ';', ';;'
SEP_VCTR_PRETTY, SEP_MTRX_PRETTY = ' ; ', ' ;;\n'

TOKEN_KIND_PATTERNS = (
	('COMMENT',		r'//[^\n]*'),
	('QUOTE',		r'"(?:[^"\\]|\\.)*"'),	# ALWAYS JSON QUOTE ESCAPE EXOTIC CHARS name="John \"Jack\" Kennedy"
	('META',		r"'[^']*'"),
	('IGNORE',		r'-*\|'),
	('SEP_MTRX',	re.escape(SEP_MTRX)),	# MTRX DISJUNCTION, AXIS=0
	('SEP_VCTR',	re.escape(SEP_VCTR)),	# VCTR CONJUNCTION, AXIS=0
	('SEP_LIMIT',	r'\s+'),				# LIMIT CONJUNCTION, AXIS-=1
	('SEP_DATA',	re.escape(SEP_DATA)),	# DATUM DISJUNCTION, AXIS SAME
	('GE',			r'>='),
	('LE',			r'<='),
	('NOT',			r'!=?'),
	('EQL',			r'='),
	('GT',			r'>'),
	('LT',			r'<'),
	('WILD',		re.escape(WILD)),		# WILDCARD, MATCHES WHOLE VALUE, NEVER QUOTE
	('MSAME',		re.escape(MSAME)),		# REFERENCES (MTRX_AXIS-1, VCTR_AXIS=-1, LIMIT_AXIS)
	('VSAME',		re.escape(VSAME)),		# REFERENCES (MTRX_AXIS,   VCTR_AXIS-1,  LIMIT_AXIS)
	('VDIFF',		re.escape(VDIFF)),		# ANTI-REFERENCES (MTRX_AXIS, VCTR_AXIS-1, LIMIT_AXIS)
	('EMPTY',		re.escape(EMPTY)),		# EMPTY SET, ANTI-WILD
	('VAR',			rf'\$[A-Za-z0-9]+'),
	('ALNUM',		r'[A-Za-z][A-Za-z0-9]*'), # ALPHANUMERICS ARE UNQUOTED
	('FLOAT',		r'-?\d*\.\d+'),
	('INT',			r'-?\d+'),
	('MISMATCH',	r'.'),
)

MASTER_PATTERN = re.compile('|'.join(f'(?P<{kind}>{pat})' for kind, pat in TOKEN_KIND_PATTERNS))

OPR_DICT = {'EQL': operator.eq, 'NOT': operator.ne, 'GT': operator.gt, 'GE': operator.ge, 'LT': operator.lt, 'LE': operator.le}
OPR_DATA_KINDS = {'EQL','NOT'}
SEP_KINDS = {'SEP_MTRX','SEP_VCTR','SEP_LIMIT','SEP_DATA',EOF}
SUGAR_KINDS = {'VDIFF', 'WILD'}
DATA_KINDS = {'ALNUM', 'QUOTE', 'INT', 'FLOAT', 'VAR', 'VSAME', 'MSAME', 'EMPTY'} # NEVER VDIFF OR WILD IN MULTI-DATA LIST
UNITARY_KINDS = {'ALNUM', 'QUOTE', 'INT', 'FLOAT', 'EQL', 'DATUM', 'NOVAR', 'VSAME', 'MSAME'}

class Token():
	kind: str
	lexeme: str
	datum: Union[str, float, int]
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
	def __str__(self) -> Memelang: return self.lexeme


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
		self.buffer: List[Token] = []
	def peek(self, fwd: int = 1) -> Union[str, None]:
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
	def __init__(self, *items: Union['Olist', Token], opr:Token|None = None):
		super().__init__(items)
		if opr is not None: self.opr = opr

	def prepend(self, item):
		self.insert(0, item)

	def pad(self, padding:Union['Olist', Token]) -> None:
		max_len = len(self[0])
		for idx, item in enumerate(self):
			diff = max_len - len(item)
			if diff>0: self[idx] += [padding] * diff
			elif diff<0: raise SyntaxError('E_PAD') # FIRST MUST BE LONGEST

	@property
	def unitary(self) -> bool: return self.opr.unitary and all(item.unitary for item in self)
	def dump(self) -> List: return [self.opr.dump(), [item.dump() for item in self]]
	def check(self) -> 'Olist': 
		if len(self)==0: raise SyntaxError('E_NO_LIST')
		return self
	def __str__(self) -> Memelang: return self.opr.lexeme.join(map(str, self))

class Data(Olist):
	opr: Token = TOK_DATUM

DATA_MSAME = Data(Token('MSAME', MSAME))
DATA_VSAME = Data(Token('VSAME', VSAME))
DATA_EMPTY = Data(Token('EMPTY', EMPTY))


class Limit(Olist):
	opr: Token = TOK_EQL # ELIDED '='

	@property
	def k1(self) -> str: return self[1][0].kind

	def check(self) -> Limit:
		if len(self)!=2: raise SyntaxError('E_NO_LIST')
		if self.k1 == 'VDIFF':
			if self.opr.kind != 'EQL': raise SyntaxError('E_OPR_VDIFF')
			self.opr=TOK_NOT
			self[1]=DATA_VSAME
		if self.k1 == 'WILD':
			if self.opr.kind == 'EQL': self.opr=TOK_NOT
			elif self.opr.kind == 'NOT': self.opr=TOK_EQL
			else: self.opr = TOK_GT # WILD MATCHES ANY NUMERIC
			self[1]=DATA_EMPTY
		return self

	@property
	def wild(self) -> bool: return self.opr.kind in {'NOT','GT'} and self.k1 == 'EMPTY'

	@property
	def eql(self) -> bool: return self.opr.kind == 'EQL'


class Vector(Olist):
	opr: Token = TOK_SEP_LIMIT

class Matrix(Olist):
	opr: Token = TOK_SEP_VCTR

LIMIT_EQL_VSAME = Limit(TOK_NOVAR, DATA_VSAME, opr=TOK_EQL)

def lex(src: Memelang) -> Iterator[Token]:
	for m in MASTER_PATTERN.finditer(src):
		kind = m.lastgroup
		if kind in {'COMMENT','META','IGNORE'}: continue
		if kind == 'MISMATCH': raise SyntaxError('E_TOK')
		yield Token(kind, m.group())


def parse(src: Memelang) -> Iterator[Matrix]:
	tokens = Stream(lex(src))
	bound_vars = []
	mtrx=Matrix()
	vctr=Vector()
	limit=Limit()
	while tokens.peek():

		# LIMIT ::= [[VAR] OPR] DATUM {SEP_DATA DATUM}
		# Single axis constraint

		# [VAR]
		var = TOK_NOVAR
		if tokens.peek() == 'VAR':
			if tokens.peek(2) in OPR_DICT: 
				var = tokens.next()
				bound_vars.append(var.lexeme)
			elif tokens.peek(2) not in SEP_KINDS: raise SyntaxError('E_VAR_NXT')

		# [OPR]
		if tokens.peek() in OPR_DICT:
			limit.opr=tokens.next()
			if tokens.peek()=='SEP_LIMIT': raise SyntaxError('E_NEVER_SPACE_AFTER_OPR')
			if tokens.peek() not in DATA_KINDS|SUGAR_KINDS: raise SyntaxError('E_OPR_DAT')

		# DATUM {SEP_DATA DATUM}
		if tokens.peek() in DATA_KINDS|SUGAR_KINDS:
			data=Data()
			data.append(tokens.next())
			while tokens.peek()=='SEP_DATA':
				data.opr = tokens.next()
				if tokens.peek()=='SEP_LIMIT': raise SyntaxError('E_NEVER_SPACE_AFTER_COMMA')
				if tokens.peek() not in DATA_KINDS: raise SyntaxError('E_DATA_KIND')
				data.append(tokens.next())

			# LOGIC CHECKS
			if any(t.kind == 'VAR' and t.lexeme not in bound_vars for t in data): raise SyntaxError('E_VAR_UNDEF')
			if len(mtrx)==0 and any(t.kind == 'VSAME' for t in data): raise SyntaxError('E_VSAME_OOB')
			if len(data)>1 and any(t.kind in SUGAR_KINDS for t in data): raise SyntaxError('E_DATA_KIND')
			if len(data)>1 and limit.opr.kind not in OPR_DATA_KINDS: raise SyntaxError('E_DATA_OPR')

			# FINALIZE LIMIT
			limit.append(var)
			limit.append(data)
			vctr.prepend(limit.check()) # LIMIT_AXIS: HIGH -> LOW
			limit=Limit()
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

	if vctr:
		mtrx.append(vctr.check())
	if mtrx: yield mtrx.check()


class SQLUtil():
	@staticmethod
	def escape(token: Token, bindings: dict) -> SQL:
		if token.kind == 'VSAME':
			if VSAME not in bindings: raise SyntaxError('E_SAME_PREV')
			return bindings[VSAME]
		elif token.kind == 'VAR':
			if token.lexeme not in bindings: raise SyntaxError('E_VAR_BIND')
			return bindings[token.lexeme]
		return "'" + str(token.datum).replace("'", "''") + "'" if isinstance(token.datum, str) else str(token.datum)

	@staticmethod
	def compare(alias_col: str, limit: Limit, bindings: dict) -> SQL:
		if len(limit[1]) > 1:
			if limit.opr.kind == 'EQL': sym = 'IN'
			elif limit.opr.kind == 'NOT': sym = 'NOT IN'
			else: raise SyntaxError()
			return f'{alias_col} {sym} ('+ ', '.join(SQLUtil.escape(v, bindings) for v in limit[1]) + ')'
		sym = {'EQL':'=','NOT':'!=','GT':'>','GE':'>=','LT':'<','LE':'<='}[limit.opr.kind]
		return f'{alias_col} {sym} {SQLUtil.escape(limit[1][0], bindings)}'


class Meme(Olist):
	opr: Token = TOK_SEP_MTRX
	results: List[List[List[Data]]]
	bindings: Dict[str, Tuple[Axis, Axis, Axis]]
	src: Memelang

	def __init__(self, src: Memelang):
		self.src = src
		self.bindings = {}
		super().__init__(*parse(src))

	def store(self): 
		for mtrx_axis, mtrx in enumerate(self):
			for vctr_axis, vctr in enumerate(mtrx):
				for limit_axis, limit in enumerate(vctr):
					if not limit.unitary: raise SyntaxError('E_LIMIT_UNIT')
					if limit.dump() == LIMIT_EQL_VSAME.dump():
						if limit_axis == 0: raise SyntaxError('E_VSAME_ZERO')
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
			self[mtrx_axis].pad(LIMIT_EQL_VSAME)

		self.results = [[[[] for limit in vctr] for vctr in mtrx] for mtrx in self]

		return self

	def expand(self, data: Data, from_limit_axis: Axis, from_vctr_axis: Axis, from_mtrx_axis: Axis) -> Data:
		expansion=Data()
		for tok in data:
			if tok.kind == 'VSAME':
				if from_vctr_axis < 1: raise SyntaxError('E_VSAME_OOB')
				expansion.extend(self.results[from_mtrx_axis][from_vctr_axis-1][from_limit_axis])
			elif tok.kind == 'MSAME':
				if from_mtrx_axis < 1: raise SyntaxError('E_MSAME_OOB')
				expansion.extend(self.results[from_mtrx_axis-1][-1][from_limit_axis])
			elif tok.kind == 'VAR':
				if tok.lexeme not in self.bindings: raise SyntaxError('E_VAR_BIND')
				axes = self.bindings[tok.lexeme]
				expansion.extend(self.results[axes[0]][axes[1]][axes[2]])
			else: expansion.append(tok)
		if len(expansion)>1: expansion.opr = TOK_SEP_DATA
		return expansion.check()


	def to_table(self, primary_col:str = 'id') -> SQL:
		ALIAS: Axis = 4
		alias_idx: int = 0
		statements: List[SQL] = []
		
		for mtrx in self:
			mtrx.pad(LIMIT_EQL_VSAME)
			froms, wheres, selects, sqlbind = [], [], [], {}
			prev = [None, None, None, None, None]

			for vctr in mtrx:
				curr = [None, None, None, None, None]
				same_row = vctr[ROW].eql and (vctr[ROW].k1 == 'VSAME' or (vctr[ROW].k1 in {'INT', 'FLOAT', 'ALNUM'} and vctr[ROW][1][0].datum == prev[ROW]))

				# TABLE and COLUMN
				for axis in (TBL,COL):
					if not vctr[axis].eql: raise SyntaxError(f'E_SQL_SUPPORT_OPR_V{axis}')
					elif vctr[axis].k1 == 'VSAME': curr[axis] = prev[axis]
					elif vctr[axis].k1 in {'ALNUM', 'QUOTE'}: curr[axis] = vctr[axis][1][0].datum 
					else: raise SyntaxError(f'E_SQL_SUPPORT_VAL_V{axis}')

				# TABLE ALIAS
				if not prev[ALIAS] or prev[TBL] != curr[TBL] or not same_row:
					curr[ALIAS] = f't{alias_idx}'
					froms.append(f'{curr[TBL]} AS {curr[ALIAS]}')
					prev[TBL] = curr[TBL]
					alias_idx += 1
				else: curr[ALIAS] = prev[ALIAS]

				# PRIMARY KEY
				if same_row: curr[ROW] = prev[ROW]
				else:
					if prev[ALIAS]: sqlbind[VSAME]=f'{prev[ALIAS]}.{primary_col}'
					wheres.append(SQLUtil.compare(f'{curr[ALIAS]}.{primary_col}', vctr[ROW], sqlbind))
					if vctr[ROW].eql and vctr[ROW].k1 in {'INT', 'FLOAT', 'ALNUM'}: curr[ROW] = vctr[ROW][1][0].datum
					else: curr[ROW] = None

				# VALUE
				if prev[ALIAS]: sqlbind[VSAME]=f'{prev[ALIAS]}.{prev[COL]}'
				if not vctr[VAL].wild: wheres.append(SQLUtil.compare(f'{curr[ALIAS]}.{curr[COL]}', vctr[VAL], sqlbind))

				# BIND VARS
				if vctr[ROW][0].kind == 'VAR': sqlbind[vctr[ROW][0].lexeme]=f'{curr[ALIAS]}.{primary_col}'
				if vctr[VAL][0].kind == 'VAR': sqlbind[vctr[VAL][0].lexeme]=f'{curr[ALIAS]}.{curr[COL]}'

				selects.append(f'{curr[ALIAS]}.{curr[COL]}')
				prev = curr[:]

			statements.append('SELECT '+ ', '.join(list(dict.fromkeys(selects))) + ' FROM ' + ', '.join(froms) + ' WHERE ' + ' AND '.join(wheres))

		return ' UNION '.join(statements)


def intersect(query: Limit, store: Data) -> Data:
	if not store: store=DATA_EMPTY

	opr_kind, intersection, query_data = query.opr.kind, Data(), [t.datum for t in query[1]]

	if opr_kind == 'EQL': intersection.extend([t for t in store if t.datum in query_data])
	elif opr_kind == 'NOT': intersection.extend([t for t in store if t.datum not in query_data])

	# RETURN ANY NUMERIC FOR GT EMPTY
	elif EMPTY in query_data: intersection.extend([t for t in store if t.kind in {'INT','FLOAT'}])

	elif len(query_data)!=1 or not isinstance(query_data[0], (int,float)): raise TypeError('E_INTER_NUM')

	intersection.extend([t for t in store if t.kind in {'INT','FLOAT'} and OPR_DICT[opr_kind](t.datum, query_data[0])])
	return intersection


# GENERATE RANDOM MEMELANG DATA
class Fuzz():
	@staticmethod
	def datum(kind:str, i:int=1) -> Memelang:
		if kind=='ALNUM': return ''.join(random.choice('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ') for _ in range(i))
		if kind=='QUOTE': return json.dumps(''.join(random.choice(' -_+,./<>[]{}\'"!@#$%^&*()abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ') for _ in range(i)))
		if kind=='INT': return str(random.randint(-i, i))
		if kind=='FLOAT': return str(random.uniform(-i, i))
		if kind=='VAR': return SIGIL + Fuzz.datum('ALNUM', i)

	@staticmethod
	def limit(bindings: List[str]|None = None) -> Memelang:
		if not bindings: bindings = []
		var = ''
		do_assign_variable = random.randint(0, 1)
		if do_assign_variable: var += Fuzz.datum('VAR',3)

		opr = random.choice(['=','!','>','<','<=','>='])

		data: str = ''
		if opr in {'=','!'}:
			data_list_len = random.randint(1, 5)
			data_list: List[Any] = []
			for _ in range(data_list_len):
				datum_type = random.randint(1, 10)
				if datum_type == 1:  data_list.append(Fuzz.datum('QUOTE',10))
				elif datum_type == 2:  data_list.append(Fuzz.datum('INT', 100))
				elif datum_type == 3:  data_list.append(Fuzz.datum('FLOAT', 100))
				elif datum_type == 4 and bindings: data_list.append(random.choice(bindings))
				elif datum_type == 5 and VSAME in bindings: data_list.append(VSAME)
				elif datum_type == 6 and VSAME in bindings and opr == '=' and data_list_len == 1: data_list.append(VDIFF)
				else: data_list.append(Fuzz.datum('ALNUM', 5))
			data += SEP_DATA.join(data_list)
		else:
			data = Fuzz.datum('FLOAT', 100)

		if var:
			assert opr
			bindings.append(var)

		return var + opr + data

	@staticmethod
	def vector(limit_len:int = 4) -> Memelang:
		bindings, vector = [], []
		for i in range(limit_len):
			if i>0: bindings.append(VSAME)
			vector.append(Fuzz.limit(bindings))
		return SEP_LIMIT.join(vector) + SEP_VCTR_PRETTY

	@staticmethod
	def mtrx_table(col_len:int = 5) -> Memelang:
		return Fuzz.datum('ALNUM',5) + SEP_LIMIT + WILD + SEP_LIMIT + SEP_VCTR_PRETTY.join(Fuzz.datum('ALNUM',5) + Fuzz.limit() for _ in range(col_len)) + SEP_MTRX_PRETTY


# TRANSLATE SQL TO MEMELANG
class SQL2Memelang():
	@staticmethod
	def output(sql_output: str) -> Memelang:
		lines=[l for l in sql_output.splitlines() if l.startswith('|')]
		if not lines:return ''
		header=[c.strip() for c in lines[0].strip('|').split('|')]
		mtrxs=[]
		for line in lines[1:]:
			cells=[c.strip() for c in line.strip('|').split('|')]
			if len(cells)!=len(header):continue
			id_val=cells[0]
			parts=[f'{header[i]} {cells[i]}' for i in range(1,len(header))]
			mtrxs.append(f'$rowid={id_val} ' + SEP_VCTR_PRETTY.join(parts))
		return SEP_MTRX_PRETTY.join(mtrxs)

	@staticmethod
	def insert(sql_insert: SQL) -> Memelang:
		m = re.search(r'INSERT\s+INTO\s+(\w+)\s*\((.*?)\)\s*VALUES\s*(.*);', sql_insert, re.I | re.S)
		if not m: return ''
		table = m.group(1)
		header = [h.strip() for h in m.group(2).split(',')]
		rows_sql = re.findall(r'\(([^()]*)\)', m.group(3))
		mtrxs = []
		for idx, row in enumerate(rows_sql):
			cells = [c.strip(" '\"") for c in re.findall(r"'[^']*'|[^,]+", row)]
			if len(cells) != len(header): continue
			rowid = cells[0]
			col_tokens = header[1:] if idx == 0 else [MSAME] * (len(header) - 1)
			parts = [f'{col_tokens[i]} {cells[i + 1]}' for i in range(len(col_tokens))]
			mtrxs.append(f'{table} $rowid={rowid} ' + SEP_VCTR_PRETTY.join(parts))
		return SEP_MTRX_PRETTY.join(mtrxs)


if __name__ == '__main__':
	memelangs: List[Memelang] = [
		'movies * actor "Mark Hamill",Mark ; movie * ; rating >4 ;;',
		'movies * actor "Mark Hamill" ; movie * ; ~ @ @ ; actor * ;;',
		Fuzz.mtrx_table()		
	]
	for src in memelangs: print(src, Meme(src).to_table())