import os, subprocess, sys, argparse

def parseInputArguments():
    parser = argparse.ArgumentParser(description="Nabla CI Pipeline nbl.ci.dev.checkout Framework Module")
   
    args = parser.parse_args()
    
    return args


def checkout(targetRevision):
    isGitRepository = os.path.exists(".git") and os.path.isdir(".git")
    
    if not isGitRepository:
        subprocess.run(f"git init", check=True)
        subprocess.run(f"git remote add origin git@github.com:Devsh-Graphics-Programming/Nabla.git", check=True)
        
    subprocess.run(f"git fetch --no-tags --force --progress --depth=1 -- origin {targetRevision}", check=True)
    subprocess.run(f"git checkout {targetRevision}", check=True)
  
  
def main():
    try:
        THIS_PROJECT_NABLA_DIRECTORY = os.environ.get('THIS_PROJECT_NABLA_DIRECTORY', '')

        if not THIS_PROJECT_NABLA_DIRECTORY:
            raise ValueError("THIS_PROJECT_NABLA_DIRECTORY environment variables doesn't exist!")
            
        NABLA_TARGET_REVISION = os.environ.get('NABLA_TARGET_REVISION', '')

        if not NABLA_TARGET_REVISION:
            raise ValueError("NABLA_TARGET_REVISION environment variables doesn't exist!")
            
        os.chdir(THIS_PROJECT_NABLA_DIRECTORY)

        args = parseInputArguments()

        print(f"Target \"{NABLA_TARGET_REVISION}\"")
        
        checkout(NABLA_TARGET_REVISION)
        
    except subprocess.CalledProcessError as e:
        print(f"Subprocess failed with exit code {e.returncode}")
        sys.exit(e.returncode)
        
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(-1)
      
      
if __name__ == "__main__":
    main()