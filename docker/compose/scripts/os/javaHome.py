import subprocess, os, re

completedProcess = subprocess.run(
    "java -XshowSettings:properties",
    shell=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    universal_newlines=True,
    check=False
)

output = completedProcess.stderr.strip()
regexMatch = re.search(r'java\.home = (.+)', output)
if regexMatch:
    JAVA_HOME = regexMatch.group(1).strip()
else:
    JAVA_HOME = ""

if JAVA_HOME:
    os.system(f'setx JAVA_HOME "{JAVA_HOME}" /M')
    print(f'JAVA_HOME has been set to: {JAVA_HOME}')
else:
    print("Error: Unable to retrieve or set JAVA_HOME.")