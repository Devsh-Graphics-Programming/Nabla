import os, subprocess, sys, argparse


def parseInputArguments():
    parser = argparse.ArgumentParser(description="Nabla CI Pipeline compose Framework script")
    
    parser.add_argument("--ssh", help="SSH key file used for github authentication, required to clone Nabla", type=str, required=True)
    parser.add_argument("--platform", help="Target platform", type=str, default="windows")
    parser.add_argument("--arch", help="Target arch", type=str, default="x86_64")
   
    args = parser.parse_args()
    
    return args
 
  
def main():
    try:
        args = parseInputArguments()
        os.path.dirname(os.path.abspath(__file__))

        os.chdir(os.path.dirname(os.path.abspath(__file__)))

        key = args.ssh
        platform = args.platform
        arch = args.arch
        
        if subprocess.call(["docker", "network", "inspect", "nabla.network"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) != 0:
           subprocess.run(["docker", "network", "create", "--driver", "nat", "--subnet", "172.28.0.0/16", "--gateway", "172.28.5.1", "nabla.network"], check=True) # create nabla.network network if not present
        
        if subprocess.call(["docker", "volume", "inspect", "nabla.repository"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) != 0:
           subprocess.run(["docker", "volume", "create", "nabla.repository"], check=True) # create nabla.repository volume if not present
           
        if subprocess.call(["docker", "volume", "inspect", "nabla.artifactory"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) != 0:
           subprocess.run(["docker", "volume", "create", "nabla.artifactory"], check=True) # create nabla.artifactory volume if not present
        
        if subprocess.call(["docker", "volume", "inspect", "ssh"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) != 0:
           subprocess.run(["docker", "volume", "create", "ssh"], check=True) # create ssh volume if not present
        
        # TODO: Unix/Macos when needed
        #subprocess.call(f"docker rm -f dev.ssh.intermediate", stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        #subprocess.run(f"docker run -d -v ssh:C:\\volume-mount-point --name dev.ssh.intermediate artifactory.devsh.eu/nabla/windows/base:latest", check=True) # create intermediate container
        #subprocess.run(f"docker start dev.ssh.intermediate", check=True) # start intermediate container
        #subprocess.run(f"docker cp {key} dev.ssh.intermediate:C:\\volume-mount-point", check=True) # copy ssh key to ssh volume
        #subprocess.call(f"docker rm -f dev.ssh.intermediate", stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
       
        os.chdir("./compose/ci/stages/dev")
       
        compose = [
            "docker", "compose",
            "-f", f"./compose.{platform}.{arch}.yml",
            "--env-file", "../.env/platform/windows/.env"
        ]
        
        subprocess.run(compose + ["up", "--build"], check=True) # compose up pipeline
        subprocess.run(compose + ["down"], check=True) # compose down pipeline
        
    except subprocess.CalledProcessError as e:
        print(f"Subprocess failed with exit code {e.returncode}")
        sys.exit(e.returncode)
        
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(-1)
      
      
if __name__ == "__main__":
    main()
