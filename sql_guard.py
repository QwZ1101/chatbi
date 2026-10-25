"""Conservative MySQL SELECT subset; reject unsupported constructs before connecting."""
import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError, TokenError

POLICY_VERSION = 'stock-daily-select-v1'
MAX_ROWS = 5000
_ALLOWED = {
    'Select', 'From', 'Table', 'TableAlias', 'Identifier', 'Column', 'Star', 'Literal',
    'Alias', 'Where', 'Group', 'Having', 'Order', 'Ordered', 'Limit', 'Offset',
    'Distinct', 'Paren', 'Null', 'Boolean', 'Placeholder', 'Between', 'In', 'Like',
    'EQ', 'NEQ', 'GT', 'GTE', 'LT', 'LTE', 'And', 'Or', 'Not', 'Is',
    'Add', 'Sub', 'Mul', 'Div', 'Mod', 'Neg',
    'Count', 'Sum', 'Avg', 'Min', 'Max', 'Round', 'Abs', 'Coalesce', 'Nullif',
    'TimeToStr', 'Year', 'Month', 'Day', 'DayOfMonth', 'Date', 'CurrentDate',
    'DateAdd', 'DateSub', 'Var', 'Anonymous',
}


class UnsafeSQL(ValueError):
    pass


def validate_sql(sql, allow_parameters=False):
    if not isinstance(sql, str) or not sql.strip() or len(sql) > 20000:
        raise UnsafeSQL('SQL 必须为非空字符串，且不超过 20000 字符')
    # Conservatively reject comment syntax even inside literals. In particular,
    # MySQL executable comments must not survive parser normalization.
    if any(marker in sql for marker in ('/*', '--', '#')):
        raise UnsafeSQL('不允许 SQL 注释或执行提示')
    try:
        statements = sqlglot.parse(sql, read='mysql')
    except (ParseError, TokenError, RecursionError) as error:
        raise UnsafeSQL('SQL 解析失败') from error
    if len(statements) != 1 or not isinstance(statements[0], exp.Select):
        raise UnsafeSQL('只允许一条普通 SELECT 语句')
    tree = statements[0]
    nodes = list(tree.walk())
    if sum(isinstance(node, exp.Select) for node in nodes) != 1:
        raise UnsafeSQL('当前版本不支持子查询、CTE 或 UNION')
    tables = list(tree.find_all(exp.Table))
    if len(tables) != 1 or tables[0].name.lower() != 'stock_daily' or tables[0].db or tables[0].catalog:
        raise UnsafeSQL('只允许访问当前库的 stock_daily 单表')
    for node in nodes:
        name = type(node).__name__
        if name not in _ALLOWED:
            raise UnsafeSQL('不允许的 SQL 结构：' + name)
        if isinstance(node, exp.Anonymous) and node.name.upper() != 'YEARWEEK':
            raise UnsafeSQL('不允许的函数：' + node.name)
        if isinstance(node, exp.Placeholder) and not allow_parameters:
            raise UnsafeSQL('模型 SQL 不能包含未绑定参数')
        if isinstance(node, exp.Var) and (node.name.upper() not in {'DAY', 'WEEK', 'MONTH', 'QUARTER', 'YEAR'} or not isinstance(node.parent, (exp.DateAdd, exp.DateSub))):
            raise UnsafeSQL('不允许变量或该日期单位')
        if isinstance(node, exp.Column) and (node.db or node.catalog):
            raise UnsafeSQL('不允许跨库列引用')
    for key, maximum in (('limit', MAX_ROWS), ('offset', 50000)):
        clause = tree.args.get(key)
        if clause is not None:
            literal = clause.expression
            if not isinstance(literal, exp.Literal) or literal.is_string or not literal.this.isdigit():
                raise UnsafeSQL(key.upper() + ' 必须为非负整数')
            number = int(literal.this)
            if key == 'offset' and number > maximum:
                raise UnsafeSQL('OFFSET 超出限制')
            if key == 'limit':
                tree.set('limit', exp.Limit(expression=exp.Literal.number(min(number, maximum))))
    if tree.args.get('limit') is None:
        tree = tree.limit(MAX_ROWS)
    # Execute the parsed/re-generated SQL, not the original text.
    return tree.sql(dialect='mysql')
