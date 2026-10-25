"""Single read path for all four SQL-based tools. Fail closed on unsafe grants."""
import os
from pathlib import Path
import re
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL
from sqlalchemy.pool import NullPool
from sql_guard import validate_sql
from trace_logging import emit

load_dotenv(Path(__file__).resolve().parent / '.env.chatbi-readonly', override=False)
DB_CONFIG = {
    'host': os.getenv('MYSQL_HOST', '127.0.0.1'),
    'port': os.getenv('MYSQL_PORT', '3306'),
    'user': os.getenv('MYSQL_USER', 'chatbi_reader'),
    'password': os.getenv('MYSQL_PASSWORD', ''),
    'db': os.getenv('MYSQL_DB', 'chatbi'),
}


def db_url():
    if DB_CONFIG['user'].lower() == 'root' or not DB_CONFIG['password']:
        raise RuntimeError('请配置非 root 的只读数据库账号和密码，见 SECURITY_GUIDE.md')
    return URL.create('mysql+mysqlconnector', username=DB_CONFIG['user'], password=DB_CONFIG['password'],
                      host=DB_CONFIG['host'], port=int(DB_CONFIG['port']), database=DB_CONFIG['db'], query={'charset': 'utf8mb4'})


def validate_grants(grants, database):
    """Accept only USAGE and table SELECT. Roles/extra privileges fail closed."""
    allowed_table = re.compile(r'^GRANT SELECT ON `' + re.escape(database) + r'`\.`stock_daily` TO ', re.I)
    has_select = False
    for grant in grants:
        if 'WITH GRANT OPTION' in grant.upper():
            raise PermissionError('数据库账号不能拥有转授权权限')
        if re.match(r'^GRANT USAGE ON \*\.\* TO ', grant, re.I):
            continue
        if allowed_table.match(grant):
            has_select = True
            continue
        raise PermissionError('数据库账号存在额外权限或角色，请使用仅有 stock_daily SELECT 权限的专用账号')
    if not has_select:
        raise PermissionError('数据库账号没有 stock_daily SELECT 权限')


def read_dataframe(query, params=None):
    sql = validate_sql(str(query), allow_parameters=params is not None)
    engine = create_engine(db_url(), connect_args={'connection_timeout': 5}, poolclass=NullPool)
    try:
        with engine.connect() as conn:
            try:
                conn.exec_driver_sql('SET ROLE NONE')
                if conn.exec_driver_sql('SELECT CURRENT_ROLE()').scalar() != 'NONE':
                    raise PermissionError('数据库会话仍有活动角色，拒绝执行')
                grants = [row[0] for row in conn.exec_driver_sql('SHOW GRANTS')]
                validate_grants(grants, DB_CONFIG['db'])
                conn.rollback()
                conn.exec_driver_sql('SET SESSION MAX_EXECUTION_TIME = 5000')
                conn.exec_driver_sql('START TRANSACTION READ ONLY')
                emit('database_read', policy='table_select_only', read_only_transaction=True, row_cap=5000, max_execution_ms=5000)
                return pd.read_sql(text(sql), conn, params=params)
            finally:
                conn.rollback()
    finally:
        engine.dispose()
