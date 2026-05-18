"""DuckDB metadata manager for data registry."""

import datetime

import duckdb


class MetadataManager:
    def __init__(self, db_path: str):
        self.conn = duckdb.connect(db_path)
        self._init_table()

    def _init_table(self):
        self.conn.execute("CREATE SEQUENCE IF NOT EXISTS data_registry_id_seq")
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS data_registry (
                id          INTEGER PRIMARY KEY DEFAULT nextval('data_registry_id_seq'),
                api_name    VARCHAR NOT NULL,
                trade_date  DATE,
                file_path   VARCHAR NOT NULL,
                row_count   INTEGER NOT NULL,
                pull_time   TIMESTAMP DEFAULT now(),
                UNIQUE(api_name, trade_date)
            )
        """)

    def insert(
        self,
        api_name: str,
        trade_date: str | None,
        file_path: str,
        row_count: int,
    ):
        if trade_date is None:
            self.conn.execute(
                "DELETE FROM data_registry WHERE api_name = ? AND trade_date IS NULL",
                [api_name],
            )
        self.conn.execute(
            "INSERT INTO data_registry "
            "(api_name, trade_date, file_path, row_count) "
            "VALUES (?, ?, ?, ?) "
            "ON CONFLICT (api_name, trade_date) DO UPDATE SET "
            "file_path = EXCLUDED.file_path, "
            "row_count = EXCLUDED.row_count, "
            "pull_time = now()",
            [api_name, trade_date, file_path, row_count],
        )

    def get_last_date(self, api_name: str) -> datetime.date | None:
        result = self.conn.execute(
            "SELECT MAX(trade_date) FROM data_registry WHERE api_name = ?",
            [api_name],
        ).fetchone()
        if result and result[0]:
            return result[0]
        return None

    def delete_since(self, api_name: str, since: str | None):
        if since is None:
            self.conn.execute(
                "DELETE FROM data_registry WHERE api_name = ?",
                [api_name],
            )
        else:
            self.conn.execute(
                "DELETE FROM data_registry WHERE api_name = ? AND trade_date >= ?",
                [api_name, since],
            )

    def summary(self):
        return self.conn.execute(
            "SELECT api_name, COUNT(*), MAX(trade_date) "
            "FROM data_registry GROUP BY api_name ORDER BY api_name"
        ).fetchall()
