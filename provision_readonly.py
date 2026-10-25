"""Create a NEW local MySQL reader, never alter/revoke existing accounts.

Run with CHATBI_ADMIN_USER / CHATBI_ADMIN_PASSWORD environment variables.
Do not pass credentials on the command line. Saves the NEW reader secret only
to the ignored .env.chatbi-readonly file. No external model calls.
"""
import json
import os
from pathlib import Path
import re
import secrets
import subprocess
import mysql.connector


def provision():
    root = Path(__file__).resolve().parent
    config = root / '.env.chatbi-readonly'
    if config.exists():
        raise RuntimeError('本地只读配置已存在，停止；不会覆盖现有凭据')
    admin = os.environ.get('CHATBI_ADMIN_USER', '')
    password = os.environ.get('CHATBI_ADMIN_PASSWORD', '')
    database = os.environ.get('MYSQL_DB', 'chatbi')
    port = int(os.environ.get('MYSQL_PORT', '3306'))
    if not admin or not password or not re.fullmatch(r'[A-Za-z0-9_]+', database):
        raise RuntimeError('需要管理员环境变量和合法库名')
    conn = mysql.connector.connect(host='127.0.0.1', port=port, user=admin, password=password, database=database, connection_timeout=5)
    reader_password = secrets.token_urlsafe(32)
    try:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM mysql.user WHERE User = 'chatbi_reader'")
        if cur.fetchone()[0]:
            raise RuntimeError('chatbi_reader 已存在，停止；不会修改现有账号')
        cur.execute('SELECT TABLE_TYPE FROM information_schema.tables WHERE TABLE_SCHEMA=%s AND TABLE_NAME=%s', (database, 'stock_daily'))
        if cur.fetchone() != ('BASE TABLE',):
            raise RuntimeError('stock_daily 必须为已有基础表')
        cur.execute('CREATE USER %s@%s IDENTIFIED BY %s', ('chatbi_reader', 'localhost', reader_password))
        cur.execute('GRANT SELECT ON `' + database + '`.`stock_daily` TO %s@%s', ('chatbi_reader', 'localhost'))
        # Exclusive create: never overwrite a credential file created concurrently.
        with config.open('x', encoding='utf-8') as file:
            file.write('MYSQL_HOST=127.0.0.1\nMYSQL_PORT=' + str(port) + '\nMYSQL_DB=' + database + '\nMYSQL_USER=chatbi_reader\nMYSQL_PASSWORD=' + reader_password + '\n')
        if os.name == 'nt':
            identity = subprocess.check_output(['whoami'], text=True).strip()
            subprocess.run(['icacls', str(config), '/inheritance:r', '/grant:r', identity + ':(F)'], capture_output=True, check=True)
        else:
            config.chmod(0o600)
        cur.close()
    finally:
        conn.close()
    print(json.dumps({'created': 'chatbi_reader@localhost', 'grant': 'SELECT on ' + database + '.stock_daily', 'credential_file': config.name}, ensure_ascii=False))


if __name__ == '__main__':
    provision()
