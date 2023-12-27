import kazoo.exceptions, kazoo.client, socket, time, sys

def connectToKazooServer(host):
    zk = kazoo.client.KazooClient(hosts=host)
    zk.start()
    
    return zk

def createKazooAtomic(zNodePath):
    if not zk.exists(zNodePath):
        zk.create(zNodePath, b"")

def getKazooAtomic(zNodePath):
    if zk.exists(zNodePath):
        data, _ = zk.get(zNodePath)
        
        return data.decode()
    else:
        return ""

def kazooAtomicPrepend(zNodePath, data):
    while True:
        currentData, stat = zk.get(zNodePath)
        newData = currentData.decode() + data

        try:
            zk.set(zNodePath, newData.encode(), version=stat.version)
            break
        except kazoo.exceptions.BadVersionException:
            continue

def healthyCheck(host):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(5)
            s.connect((host, 2181))
        return True
    except (socket.error, socket.timeout):
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Nabla CI Pipeline nbl.ci.dev.kazoo Framework Module")
    
    arser.add_argument("--host", help="Kazoo Server host", type=str, required=True)
    
    args = parser.parse_args()

    if healthyCheck(args.host):
        sys.exit(0)  # healthy
    else:
        sys.exit(1)  # not healthy
