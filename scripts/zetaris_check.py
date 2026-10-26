# Connects to the local Zetaris over JDBC and runs SELECT 1.
import os
import subprocess
import jpype
import jaydebeapi

JAR = os.path.expanduser(
    "~/zetaris-platform/jdbc/ndp-jdbc-driver-2.4.3.1-7eff043-driver.jar")

# Prefer an explicit JAVA_HOME (the arm64 Java 17), else ask macOS.
java_home = os.environ.get("JAVA_HOME") or \
    subprocess.check_output(["/usr/libexec/java_home"]).decode().strip()
jvm = os.path.join(java_home, "lib", "server", "libjvm.dylib")
print("Using Java at:", java_home)
print("libjvm exists:", os.path.exists(jvm))

if not jpype.isJVMStarted():
    jpype.startJVM(jvm, classpath=[JAR])

conn = jaydebeapi.connect(
    "com.zetaris.lightning.jdbc.LightningDriver",
    "jdbc:zetaris:lightning@localhost:10000",
    [os.environ["ZETARIS_USER"], os.environ["ZETARIS_PASSWORD"]],
    JAR,
)
cur = conn.cursor()
cur.execute("SELECT 1")
print(cur.fetchall())
cur.close()
conn.close()
