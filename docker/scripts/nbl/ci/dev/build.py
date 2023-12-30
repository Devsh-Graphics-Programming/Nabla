import os, subprocess, sys, argparse
from .lib.kazoo import *

def parseInputArguments():
    parser = argparse.ArgumentParser(description="Nabla CI Pipeline nbl.ci.dev.build Framework Module")
    
    parser.add_argument("--config", help="Target CMake configuration", type=str, default="Release")
    parser.add_argument("--libType", help="Target library type", type=str, default="dynamic")
   
    args = parser.parse_args()
    
    return args

def buildNabla(libType, config):
    return subprocess.run(f"cmake --build --preset ci-build-{libType}-msvc-{config}", check=False)


def buildProject(libType, config, buildDirectory):
    return subprocess.run(f"cmake --build \"{buildDirectory}\" --config {config}", check=False)
  
  
def getCPackBundleHash(buildDirectory, target, component = "ALL", relativeDirectory = "/"):
    return f"{buildDirectory};{target};{component};{relativeDirectory};"
    
  
def main():
    try:
        THIS_PROJECT_NABLA_DIRECTORY = os.environ.get('THIS_PROJECT_NABLA_DIRECTORY', '')

        if not THIS_PROJECT_NABLA_DIRECTORY:
            raise ValueError("THIS_PROJECT_NABLA_DIRECTORY environment variables doesn't exist!")
            
        THIS_PROJECT_PLATFORM = os.environ.get('THIS_PROJECT_PLATFORM', '')

        if not THIS_PROJECT_PLATFORM:
            raise ValueError("THIS_PROJECT_PLATFORM environment variables doesn't exist!")
        
        THIS_PROJECT_ARCH = os.environ.get('THIS_PROJECT_ARCH', '')

        if not THIS_PROJECT_ARCH:
            raise ValueError("THIS_PROJECT_ARCH environment variables doesn't exist!")
        
        THIS_SERVICE_BINARY_PROJECT_PATH = os.environ.get('THIS_SERVICE_BINARY_PROJECT_PATH', '')
    
        os.chdir(THIS_PROJECT_NABLA_DIRECTORY)

        args = parseInputArguments()

        config = args.config
        lowerCaseConfig = config.lower()
        libType = args.libType
        
        topBuildDirectory = os.path.normpath(os.path.join(THIS_PROJECT_NABLA_DIRECTORY, f"build/.docker/{THIS_PROJECT_PLATFORM}/{THIS_PROJECT_ARCH}/{libType}/{lowerCaseConfig}"))
        targetBuildDirectory = os.path.normpath(os.path.join(topBuildDirectory, THIS_SERVICE_BINARY_PROJECT_PATH))
        
        if topBuildDirectory == targetBuildDirectory:
            buildNabla(libType, lowerCaseConfig)
            cpackBundleHash = getCPackBundleHash(topBuildDirectory, "Libraries") + getCPackBundleHash(topBuildDirectory, "Runtime")
        else:
            buildProject(libType, config, targetBuildDirectory)
            cpackBundleHash += getCPackBundleHash(targetBuildDirectory, "Executables") + getCPackBundleHash(targetBuildDirectory, "Media")
                
        kazooConnector = KazooConnector(f"dev.nabla.kazoo.server.{libType}.{lowerCaseConfig}.x86_64.{THIS_PROJECT_PLATFORM}") # DNS record as compose service name
        kazooConnector.connect()

        zNodePath = f"/CPACK_INSTALL_CMAKE_PROJECTS"
        kazooConnector.createKazooAtomic(zNodePath)
        kazooConnector.appendKazooAtomic(zNodePath, cpackBundleHash)
        print(f"Atomic update performed on {zNodePath} zNode path")
        print(f"cpackBundleHash = {cpackBundleHash}")
        
        kazooConnector.disconnect()
                
    except subprocess.CalledProcessError as e:
        print(f"Subprocess failed with exit code {e.returncode}")
        sys.exit(e.returncode)
        
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(-1)
      
      
if __name__ == "__main__":
    main()
