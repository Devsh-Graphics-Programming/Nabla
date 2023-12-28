import os, subprocess, sys, argparse, shutil


def parseInputArguments():
    parser = argparse.ArgumentParser(description="Nabla CI Pipeline nbl.ci.dev.init Framework Module")
 
    parser.add_argument("--key", help="RSA input key file", type=str, required=True)

    args = parser.parse_args()
    
    return args


def init():
    THIS_PROJECT_SSH_DIRECTORY = os.environ.get('THIS_PROJECT_SSH_DIRECTORY', '')

    if not THIS_PROJECT_SSH_DIRECTORY:
        raise ValueError("THIS_PROJECT_SSH_DIRECTORY environment variables doesn't exist!")

    args = parseInputArguments()
   
    try:
        inputKey = args.key
        targetKey = os.path.normpath(os.path.join(THIS_PROJECT_SSH_DIRECTORY, "id_rsa"))

        # TODO: Unix/MacOS when needed
        subprocess.run(f"icacls.exe {targetKey} /reset", check=False)

        shutil.copy(inputKey, targetKey)
        print(f"Copied \"{inputKey}\" to \"{targetKey}\"")

        subprocess.run(f"icacls.exe {targetKey} /GRANT:R ContainerAdministrator:(R)", check=True)
        subprocess.run(f"icacls.exe {targetKey} /inheritance:r", check=True)

        subprocess.run("ssh -o StrictHostKeyChecking=no -T git@github.com", check=True)
    except subprocess.CalledProcessError as e:
        if not (e.returncode == 0 or e.returncode == 1):
            raise ValueError("Could not authenticate with provided rsa key, exiting...")
    except FileNotFoundError:
        raise ValueError(f"Input key file \"{inputKey}\" not found")
            
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
