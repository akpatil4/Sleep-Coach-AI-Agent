# app/adapters/zetaris.py
# Reads the sleep table from the local Zetaris instance over JDBC.
import os
import subprocess

import jaydebeapi
import jpype
import pandas as pd

DRIVER = "com.zetaris.lightning.jdbc.LightningDriver"
URL = os.getenv("ZETARIS_JDBC_URL", "jdbc:zetaris:lightning@localhost:10000")
JAR = os.path.expanduser(os.getenv(
    "ZETARIS_JDBC_JAR",
    "~/zetaris-platform/jdbc/ndp-jdbc-driver-2.4.3.1-7eff043-driver.jar"))
NUMERIC = ["sleep_quality", "caffeine_cups", "last_caffeine_hour", "screen_min", "mood"]


def _start_jvm():
    """Start Java once per process (JAVA_HOME must be an arm64 Java 17 on Apple Silicon)."""
    if jpype.isJVMStarted():
        return
    java_home = os.environ.get("JAVA_HOME") or \
        subprocess.check_output(["/usr/libexec/java_home"]).decode().strip()
    jvm = os.path.join(java_home, "lib", "server", "libjvm.dylib")
    if not os.path.exists(jvm):
        jvm = jpype.getDefaultJVMPath()
    jpype.startJVM(jvm, classpath=[JAR])


def run_sql(sql):
    """Send ONE SQL statement to Zetaris and return the rows as a DataFrame."""
    for name in ("ZETARIS_USER", "ZETARIS_PASSWORD"):
        if not os.getenv(name):
            raise RuntimeError(f"{name} is not set (put it in .env).")
    _start_jvm()
    conn = jaydebeapi.connect(
        DRIVER, URL, [os.environ["ZETARIS_USER"], os.environ["ZETARIS_PASSWORD"]], JAR)
    try:
        cur = conn.cursor()
        cur.execute(sql)
        cols = [str(d[0]).lower().split(".")[-1] for d in cur.description]
        rows = [[None if v is None else str(v) for v in row] for row in cur.fetchall()]
    finally:
        conn.close()
    return pd.DataFrame(rows, columns=cols)


def fetch_daily_log():
    """The agent's ONLY data read in normal mode: a SELECT through Zetaris."""
    table = os.getenv("ZETARIS_TABLE")
    if not table:
        raise RuntimeError("ZETARIS_TABLE is not set.")
    df = run_sql(f"SELECT * FROM {table}")
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    for col in NUMERIC:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df