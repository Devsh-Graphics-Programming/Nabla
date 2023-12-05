import os, subprocess, sys, argparse

def parseInputArguments():
    parser = argparse.ArgumentParser(description="Argument parser")
    parser.add_argument("--cishallowconfigure", help="Specify target CMake configuration", type=str)
    parser.add_argument("--config", help="Specify target CMake configuration", type=str)
    parser.add_argument("--arch", help="Specify target architecture", type=str)
    parser.add_argument("--libType", help="Specify target library type", type=str)
    
    args = parser.parse_args()
    
    return args
    
def clone(targetRevision):
    subprocess.run(f"git init", check=True)
    subprocess.run(f"git remote add origin git@github.com:Devsh-Graphics-Programming/Nabla.git", check=True)
    subprocess.run(f"git fetch --no-tags --force --progress --depth=1 -- origin {targetRevision}", check=True)
    subprocess.run(f"git checkout {targetRevision}", check=True)
    
def configure(libType):
    subprocess.run(f"cmake . --preset ci-configure-{libType}-msvc", check=True)
    
def build(libType, config):
    subprocess.run(f"cmake --build --preset ci-build-{libType}-msvc --config {config}", check=True)

try:
    THIS_PROJECT_NABLA_DIRECTORY = os.environ.get('THIS_PROJECT_NABLA_DIRECTORY', '')

    if not THIS_PROJECT_NABLA_DIRECTORY:
        raise ValueError("THIS_PROJECT_NABLA_DIRECTORY enviroment variables doesn't exist!")
    
    print(f"THIS_PROJECT_NABLA_DIRECTORY=\"{THIS_PROJECT_NABLA_DIRECTORY}\"")
    
    os.chdir(THIS_PROJECT_NABLA_DIRECTORY)
    
    args = parseInputArguments()
   
    if args.cishallowconfigure:
        targetRevision = args.cishallowconfigure
        
        clone(targetRevision)
        configure("static")  # TODO: maybe execute with only
        configure("dynamic") # update submodule mode and then configure async
        exit(0)

    config = "Release"
    if args.config:
        config = args.config
        
    archValue = "x86_64"
    if args.arch:
        archValue = args.arch
        
    libType = "dynamic"
    if args.libType:
        libType = args.libType

    configure(libType)
    build(libType, config)

except subprocess.CalledProcessError as e:
    print(f"Subprocess failed with exit code {e.returncode}")
    sys.exit(e.returncode)
    
except Exception as e:
    print(f"Unexpected error: {e}")
    sys.exit(-1)