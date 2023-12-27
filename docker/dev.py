import os, subprocess, sys, argparse


def parseInputArguments():
    parser = argparse.ArgumentParser(description="Nabla CI Pipeline compose Framework script")
    
    parser.add_argument("--ssh", help="SSH key file used for github authentication, required to clone Nabla", type=str, required=True)
    parser.add_argument("--platform", help="Target platform", type=str, default="windows")
   
    args = parser.parse_args()
    
    return args
 
  
def main():
    try:
        args = parseInputArguments()
        os.path.dirname(os.path.abspath(__file__))

        os.chdir(os.path.dirname(os.path.abspath(__file__)))

        key = args.ssh
        platform = args.platform
        
        subprocess.run(f"docker compose -f ./compose/ci/stages/dev/init/compose.yml build nabla.init.{platform}", check=True) # build base image
        
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
        
        mainCompose = [
            "docker", "compose",
            "-f", "./compose/ci/stages/dev/init/compose.yml",
            "-f", "./compose/ci/stages/dev/kazoo/compose.yml",
            "-f", "./compose/ci/stages/dev/checkout/compose.yml",
            "-f", "./compose/ci/stages/dev/submodules/compose.yml",
            "-f", "./compose/ci/stages/dev/cmake/compose.yml",
            "-f", "./compose/ci/stages/dev/build/compose.yml",
            "--env-file", "./compose/ci/stages/.env/.env"
        ]
        
        postCompose = [
            "docker", "compose",
            "-f", "./compose/ci/stages/dev/cpack/compose.yml",
            "--env-file", "./compose/ci/stages/.env/.env"
        ]
        
        subprocess.run(mainCompose + ["up", "--build"], check=True) # compose up main pipeline
        subprocess.run(postCompose + ["up", "--build"], check=True) # compose up post pipeline
        
        subprocess.run(postCompose + ["down"], check=True) # compose down main pipeline
        subprocess.run(postCompose + ["down"], check=True) # compose down post pipeline
        
    except subprocess.CalledProcessError as e:
        print(f"Subprocess failed with exit code {e.returncode}")
        sys.exit(e.returncode)
        
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(-1)
      
      
if __name__ == "__main__":
    main()
