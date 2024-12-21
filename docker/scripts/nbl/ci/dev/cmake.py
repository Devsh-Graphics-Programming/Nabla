import os, subprocess, sys, argparse

def parseInputArguments():
    parser = argparse.ArgumentParser(description="Nabla CI Pipeline nbl.ci.dev.cmake Framework Module")
    
    parser.add_argument("--libType", help="Target library type", type=str, default="dynamic")
    parser.add_argument("--config", help="Target library type", type=str, default="release")

    args = parser.parse_args()
    
    return args

def configure(libType, config):
    subprocess.run(f"cmake . --preset ci-configure-{libType}-msvc-{config}", check=True)

def main():
    try:
        THIS_PROJECT_NABLA_DIRECTORY = os.environ.get('THIS_PROJECT_NABLA_DIRECTORY', '')

        if not THIS_PROJECT_NABLA_DIRECTORY:
            raise ValueError("THIS_PROJECT_NABLA_DIRECTORY environment variables doesn't exist!")
            
        THIS_PROJECT_ARCH = os.environ.get('THIS_PROJECT_ARCH', '')

        if not THIS_PROJECT_ARCH:
            raise ValueError("THIS_PROJECT_ARCH environment variables doesn't exist!")

        os.chdir(THIS_PROJECT_NABLA_DIRECTORY)

        args = parseInputArguments()

        libType = args.libType
        config = args.config
        
        configure(libType, config)

    except subprocess.CalledProcessError as e:
        print(f"Subprocess failed with exit code {e.returncode}")
        sys.exit(e.returncode)
        
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(-1)
      
      
if __name__ == "__main__":
    main()
