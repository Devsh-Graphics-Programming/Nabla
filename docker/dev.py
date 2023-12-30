import os, subprocess, sys, argparse


def parseInputArguments():
    parser = argparse.ArgumentParser(description="Nabla CI Pipeline compose Framework script")
    
    parser.add_argument("--platform", help="Target platform", type=str, default="windows")
    parser.add_argument("--arch", help="Target arch", type=str, default="x86_64")
    parser.add_argument('--profiles', nargs='*', default=["dev.dynamic.debug"], help='Target list of profiles to apply')
   
    args = parser.parse_args()
    
    return args


def updateSubmodules(root):
    updateSubmoduleScript = os.path.normpath(os.path.join(root, "cmake/submodules/update.cmake"))
    return subprocess.run(f"cmake -P \"{updateSubmoduleScript}\"", check=True)
 
  
def main():
    try:
        args = parseInputArguments()
        
        root = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../"))
               
        updateSubmodules(root)
         
        os.chdir(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "compose/ci/stages/dev")))
        
        platform = args.platform
        arch = args.arch
        
        if subprocess.call(["docker", "network", "inspect", "nabla.network"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) != 0:
           subprocess.run(["docker", "network", "create", "--driver", "nat", "--subnet", "172.28.0.0/16", "--gateway", "172.28.5.1", "nabla.network"], check=True) # create nabla.network network if not present
        
        envFile = os.path.abspath(f"../.env/platform/{platform}/.env")
        profiles = (lambda profiles: [item for profile in profiles for item in ["--profile", profile]])(args.profiles)

        compose = [
            "docker", "compose",
            "-f", f"./compose.yml",
            "--env-file", envFile
        ] + profiles
        
        subprocess.run(compose + ["build"], check=True)
        subprocess.run(compose + ["config"], check=True)
        subprocess.run(compose + ["create", "--force-recreate"], check=True)
        subprocess.run(compose + ["up"], check=True)
        subprocess.run(compose + ["down"], check=True)
        
    except subprocess.CalledProcessError as e:
        print(f"Subprocess failed with exit code {e.returncode}")
        sys.exit(e.returncode)
        
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(-1)
      
      
if __name__ == "__main__":
    main()
