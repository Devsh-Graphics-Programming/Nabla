import os, subprocess, sys, argparse


def parseInputArguments():
    parser = argparse.ArgumentParser(description="Nabla CI Pipeline nbl.ci.dev.init Framework Module")
 
    args = parser.parse_args()
    
    return args


def init():
    THIS_PROJECT_SSH_DIRECTORY = os.environ.get('THIS_PROJECT_SSH_DIRECTORY', '')

    if not THIS_PROJECT_SSH_DIRECTORY:
        raise ValueError("THIS_PROJECT_SSH_DIRECTORY environment variables doesn't exist!")

    key = os.path.normpath(os.path.join(THIS_PROJECT_SSH_DIRECTORY, "id_rsa"))

    # TODO: Unix/MacOS when needed
    subprocess.run(f"icacls.exe {key} /reset", check=True)
    subprocess.run(f"icacls.exe {key} /GRANT:R ContainerAdministrator:(R)", check=True)
    subprocess.run(f"icacls.exe {key} /inheritance:r", check=True)
   
    try:
        subprocess.run("ssh -o StrictHostKeyChecking=no -T git@github.com", check=True)
    except subprocess.CalledProcessError as e:
        if not (e.returncode == 0 or e.returncode == 1):
            raise ValueError("Could not authenticate with provided rsa key, exiting...")
            

def main():
    try:
        init()
        
    except subprocess.CalledProcessError as e:
        print(f"Subprocess failed with exit code {e.returncode}")
        sys.exit(e.returncode)
        
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(-1)
      
      
if __name__ == "__main__":
    main()
