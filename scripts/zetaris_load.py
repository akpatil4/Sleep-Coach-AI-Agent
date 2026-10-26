# scripts/zetaris_load.py
# Copies the CSV into the managed file-store folder inside the Zetaris container and
# tries to create the table from several path forms.
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env", override=True)

import jaydebeapi
import jpype

DB = os.getenv("ZETARIS_DB", "SLEEPDB")
TABLE = "daily_log"
CONTAINER = "zetaris-driver-1"
BASE = "/home/zetaris/customer-data"
JAR = os.path.expanduser(
    "~/zetaris-platform/jdbc/ndp-jdbc-driver-2.4.3.1-7eff043-driver.jar")


def sh(*args):
    return subprocess.run(list(args), capture_output=True, text=True)


# 1. Find the organisation folder (the one that is not 'shared') and its file-store.
dirs = [d.rstrip("/") for d in sh("docker", "exec", CONTAINER, "ls", "-1p", BASE).stdout.split()
        if d.endswith("/")]
org = next((d for d in dirs if d != "shared"), None)
print("organisation folder:", org)
if not org:
    sys.exit("Could not find the organisation folder. Is Zetaris running?")

store = f"{BASE}/{org}/file-store"
copy = sh("docker", "cp", "data/daily_log.csv", f"{CONTAINER}:{store}/daily_log.csv")
print("copy into file-store:", "ok" if copy.returncode == 0 else copy.stderr.strip()[:200])
sh("docker", "exec", "-u", "0", CONTAINER, "chmod", "644", f"{store}/daily_log.csv")
print(sh("docker", "exec", CONTAINER, "ls", "-l", store).stdout)

candidates = [
    "daily_log.csv",
    "file-store/daily_log.csv",
    f"{org}/file-store/daily_log.csv",
    f"{store}/daily_log.csv",
    f"file://{store}/daily_log.csv",
    f"{store}/",
    f"customer-data/{org}/file-store/daily_log.csv",
]

# 2. Connect.
java_home = os.environ.get("JAVA_HOME") or \
    subprocess.check_output(["/usr/libexec/java_home"]).decode().strip()
jvm = os.path.join(java_home, "lib", "server", "libjvm.dylib")
if not jpype.isJVMStarted():
    jpype.startJVM(jvm, classpath=[JAR])
conn = jaydebeapi.connect(
    "com.zetaris.lightning.jdbc.LightningDriver",
    os.getenv("ZETARIS_JDBC_URL", "jdbc:zetaris:lightning@localhost:10000"),
    [os.environ["ZETARIS_USER"], os.environ["ZETARIS_PASSWORD"]], JAR)


def short(err):
    m = re.search(r"<<<.*?>>>", err)
    return m.group(0) if m else err[:200]


def run(sql):
    cur = conn.cursor()
    try:
        cur.execute(sql.strip().rstrip(";"))
        return True, (cur.fetchall() if cur.description else None)
    except Exception as exc:
        return False, str(exc)
    finally:
        cur.close()


def step(label, sql):
    ok, res = run(sql)
    print(f"[{'ok' if ok else 'FAILED'}] {label}")
    if not ok:
        print("      ", short(res))
    return ok, res


step("create database ('already exists' is fine)",
     f'CREATE LIGHTNING DATABASE {DB} DESCRIBE BY "{DB}"')

created, used = False, None
for path in candidates:
    created, _ = step(
        f"create table from {path}",
        f'CREATE LIGHTNING FILESTORE TABLE {TABLE} FROM {DB} FORMAT CSV OPTIONS ('
        f'PATH "{path}", header "true", inferSchema "true")')
    if created:
        used = path
        break

if created:
    print("\nTable created using path:", used)
    for name in [f"{DB}.{TABLE}", TABLE]:
        ok, res = step(f"count rows in {name}", f"SELECT COUNT(*) FROM {name}")
        if ok:
            print("       row count:", res)
            ok2, rows = step(f"first rows of {name}", f"SELECT * FROM {name} LIMIT 3")
            if ok2:
                for r in rows:
                    print("       ", r)
            print(f"\nWorking table name: {name}")
            print(f"Put this in .env:  ZETARIS_TABLE={name}")
            break
else:
    print("\nNo path form was accepted. Paste this whole output.")
conn.close()