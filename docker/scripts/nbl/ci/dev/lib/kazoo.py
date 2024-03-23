import os, subprocess, sys, argparse, kazoo.exceptions, kazoo.client, socket

def getLocalIPV4():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        local_ipv4 = s.getsockname()[0] # Get the local IPv4 address
        s.close()
        
        return local_ipv4
    except socket.error:
        return None
    

def resolveServiceToIPv4(serviceName):
    try:
        ipv4Address = socket.gethostbyname(serviceName)
        return ipv4Address
    except socket.gaierror as e:
        print(f"Error while resolving {serviceName} to an IPv4 address: {e}")
        return None


class KazooConnector:
    def __init__(self, dnsServiceName):
        self.dnsServiceName = dnsServiceName
        self.host = resolveServiceToIPv4(self.dnsServiceName)
        self.zk = kazoo.client.KazooClient(hosts=self.dnsServiceName)

    def connect(self):
        self.zk.start()
        print(f"Connected to {self.dnsServiceName} kazoo host")

    def disconnect(self):
        self.zk.stop()
        self.zk.close()
        print(f"Disconnected from {self.dnsServiceName} kazoo host")

    def requestServerShutdown(self):
        self.createKazooAtomic("/sdRequest")
        print(f"Requested shutdown of {self.dnsServiceName} kazoo host")

    def createKazooAtomic(self, zNodePath):
        if not self.zk.exists(zNodePath):
            self.zk.create(zNodePath, b"")

    def getKazooAtomic(self, zNodePath):
        if self.zk.exists(zNodePath):
            data, _ = self.zk.get(zNodePath)
            return data.decode()
        else:
            return ""

    def appendKazooAtomic(self, zNodePath, data):
        while True:
            try:
                currentData, stat = self.zk.get(zNodePath)
                newData = currentData.decode() + data
                self.zk.set(zNodePath, newData.encode(), version=stat.version)
                break
            except kazoo.exceptions.BadVersionException:
                pass


def shutdownOs():
    if os.name == 'nt' or os.name == 'java': # For windows and java (in the rare case of running jython)
        return os.system('shutdown /s /f 0')
    elif os.name == 'posix': # For Unix, Linux, Mac
        return os.system('shutdown -h now')
    else:
        print('Unknown operating system') # Docs for os.name listed only the above three cases
        return 1


def healthyCheck(host):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            ipv4 = host
            
            if host == "localhost" or host == "127.0.0.1":
                ipv4 = getLocalIPV4()
        
            s.settimeout(5)
            s.connect((ipv4, 2181))
        
        print(f"Connected to {ipv4} kazoo host")

        # TODO: find lib which does nice shutdown cross platform
        sdProcess = subprocess.run("zkCli.cmd get /sdRequest", stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        shutdown = not sdProcess.stderr.strip()

        if shutdown:
            print("Requested shutdown...")

            if shutdownOs() != 0:
                print(f"Could not shutdown container")

        return True
    except (socket.error, socket.timeout):
        print(f"Excpetion caught while trying to connect to kazoo host: \"{socket.error}\"")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Nabla CI Pipeline nbl.ci.dev.kazoo Framework Module")
    
    parser.add_argument("--host", help="Kazoo Server host", type=str, default="localhost")
    
    args = parser.parse_args()

    if healthyCheck(args.host):
        sys.exit(0)  # healthy
    else:
        sys.exit(1)  # not healthy
