import os, subprocess, sys, argparse

def parseInputArguments():
    parser = argparse.ArgumentParser(description="Nabla CI Framework cross platform build pipeline script")
    parser.add_argument("--init-clone-generate-directories", help="This flag tells the script to clone and configure all build directories", type=bool, default=False)
    parser.add_argument("--build", help="This flag tells the script to build Nabla", type=bool, default=True)
    parser.add_argument("--target-revision", help="Target revision or branch's HEAD to fetch and checkout", type=str, default="docker")
    parser.add_argument("--config", help="Target CMake configuration", type=str, default="Release")
    parser.add_argument("--arch", help="Target architecture", type=str, default="x86_64")
    parser.add_argument("--libType", help="Target library type", type=str, default="dynamic")
    
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
        
    os.chdir(THIS_PROJECT_NABLA_DIRECTORY)
    
    args = parseInputArguments()
    
    targetRevision = args.target_revision
    config = args.config
    arch = args.arch
    libType = args.libType
    
    print(f"Target {targetRevision} revision!")
    
    if args.init_clone_generate_directories:
        clone(targetRevision)
        configure("static")  # TODO: maybe execute with only
        configure("dynamic") # update submodule mode and then configure async
    else:
        configure(libType)
        if args.build:
            build(libType, config)
    
except subprocess.CalledProcessError as e:
    print(f"Subprocess failed with exit code {e.returncode}")
    sys.exit(e.returncode)
    
except Exception as e:
    print(f"Unexpected error: {e}")
    sys.exit(-1)