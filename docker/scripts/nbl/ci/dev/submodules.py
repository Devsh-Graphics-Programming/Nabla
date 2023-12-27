import os, subprocess, sys, argparse, glob, debugpy, asyncio, socket

def parseInputArguments():
    parser = argparse.ArgumentParser(description="Nabla CI Pipeline nbl.ci.dev.submodules Framework Module")
    
    args = parser.parse_args()
    
    return args


def updateSubmodules():
    return subprocess.run(f"cmake -S . -B ./build/.submodules -DNBL_EXIT_ON_UPDATE_GIT_SUBMODULE=ON -DNBL_CI_GIT_SUBMODULES_SHALLOW=ON", check=True)


def main():
    try:
        THIS_PROJECT_NABLA_DIRECTORY = os.environ.get('THIS_PROJECT_NABLA_DIRECTORY', '')

        if not THIS_PROJECT_NABLA_DIRECTORY:
            raise ValueError("THIS_PROJECT_NABLA_DIRECTORY environment variables doesn't exist!")

        THIS_PROJECT_ARTIFACTORY_NABLA_DIRECTORY = os.environ.get('THIS_PROJECT_ARTIFACTORY_NABLA_DIRECTORY', '')
    
        updateSubmodules()
        
    except subprocess.CalledProcessError as e:
        print(f"Subprocess failed with exit code {e.returncode}")
        sys.exit(e.returncode)
        
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(-1)
      
      
if __name__ == "__main__":
    main()
