import os, subprocess, sys, argparse, glob

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
    
def configure(libType, updateSubmodulesOnly = False):
    return subprocess.run(f"cmake . --preset ci-configure-{libType}-msvc", check=True)

def updateSubmodules():
    return subprocess.run(f"cmake -S . -B ./build_submodules_update -DNBL_EXIT_ON_UPDATE_GIT_SUBMODULE=ON -DNBL_CI_GIT_SUBMODULES_SHALLOW=ON", check=True)

def buildNabla(libType, config):
    return subprocess.run(f"cmake --build --preset ci-build-{libType}-msvc --config {config}", check=False)
    
def buildProject(libType, config, buildDirectory):
    return subprocess.run(f"cmake --build \"{buildDirectory}\" --config {config}", check=False)
  
def cpack(libType, config, CPACK_INSTALL_CMAKE_PROJECTS, packageDirectory):
    if not packageDirectory:
        packageDirectory = f"./package/{config}"
        
    return subprocess.run(f"cpack --preset ci-package-{libType}-msvc -C {config} -B \"{packageDirectory}\" -D CPACK_INSTALL_CMAKE_PROJECTS=\"{CPACK_INSTALL_CMAKE_PROJECTS}\"", check=True)
  
def getCPackBundleHash(buildDirectory, target, component = "ALL", relativeDirectory = "/"):
    return f"{buildDirectory};{target};{component};{relativeDirectory};"

try:
    THIS_PROJECT_NABLA_DIRECTORY = os.environ.get('THIS_PROJECT_NABLA_DIRECTORY', '')

    if not THIS_PROJECT_NABLA_DIRECTORY:
        raise ValueError("THIS_PROJECT_NABLA_DIRECTORY enviroment variables doesn't exist!")
        
    THIS_PROJECT_ARTIFACTORY_NABLA_DIRECTORY = os.environ.get('THIS_PROJECT_ARTIFACTORY_NABLA_DIRECTORY', '')
    
    if not THIS_PROJECT_ARTIFACTORY_NABLA_DIRECTORY:
        print("THIS_PROJECT_ARTIFACTORY_NABLA_DIRECTORY not defined, using default <buildPath>/package/<configuration> path for artifacts") 
        
    os.chdir(THIS_PROJECT_NABLA_DIRECTORY)
    
    args = parseInputArguments()
    
    targetRevision = args.target_revision
    config = args.config
    arch = args.arch
    libType = args.libType
    
    print(f"Target {targetRevision} revision!")
    
    if args.init_clone_generate_directories:
        clone(targetRevision)
        updateSubmodules()
        configure("static")  # TODO: maybe execute with only
        configure("dynamic") # update submodule mode and then configure async
    else:
        if args.build:
            topBuildDirectory = os.path.normpath(os.path.join(THIS_PROJECT_NABLA_DIRECTORY, f"build/{libType}"))
            cpackBundleHash = getCPackBundleHash(topBuildDirectory, "Headers")
            
            result = buildNabla(libType, config)
            
            if result.returncode == 0:
                cpackBundleHash += getCPackBundleHash(topBuildDirectory, "Libraries") + getCPackBundleHash(topBuildDirectory, "Runtime")
                
                matchingFiles = glob.glob('examples_tests/*/config.json.template', recursive=True)
                exBuildDirectories = list(set(os.path.normpath(os.path.join(topBuildDirectory, os.path.dirname(file))) for file in matchingFiles))

                for exBuildDirectory in exBuildDirectories:
                    result = buildProject(libType, config, exBuildDirectory)
                    
                    if result.returncode == 0:
                        cpackBundleHash += getCPackBundleHash(exBuildDirectory, "Executables") + getCPackBundleHash(exBuildDirectory, "Media")
                        
            print(cpackBundleHash)            
            cpack(libType, config, cpackBundleHash, f"{THIS_PROJECT_ARTIFACTORY_NABLA_DIRECTORY}/Nabla/artifacts/{config}")
                
except subprocess.CalledProcessError as e:
    print(f"Subprocess failed with exit code {e.returncode}")
    sys.exit(e.returncode)
    
except Exception as e:
    print(f"Unexpected error: {e}")
    sys.exit(-1)