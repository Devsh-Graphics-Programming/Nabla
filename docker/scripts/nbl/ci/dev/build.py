import os, subprocess, sys, argparse, .lib.kazoo

def parseInputArguments():
    parser = argparse.ArgumentParser(description="Nabla CI Pipeline nbl.ci.dev.build Framework Module")
    
    parser.add_argument("--target-project-directory", help="Build Target project's directory path, relative to top build directory", type=str, default="")
    parser.add_argument("--config", help="Target CMake configuration", type=str, default="Release")
    parser.add_argument("--libType", help="Target library type", type=str, default="dynamic")
   
    args = parser.parse_args()
    
    return args

def buildNabla(libType, config):
    return subprocess.run(f"cmake --build --preset ci-build-{libType}-msvc --config {config}", check=False)


def buildProject(libType, config, buildDirectory):
    return subprocess.run(f"cmake --build \"{buildDirectory}\" --config {config}", check=False)
  
  
def getCPackBundleHash(buildDirectory, target, component = "ALL", relativeDirectory = "/"):
    return f"{buildDirectory};{target};{component};{relativeDirectory};"
    
  
def main():
    try:
        THIS_PROJECT_NABLA_DIRECTORY = os.environ.get('THIS_PROJECT_NABLA_DIRECTORY', '')

        if not THIS_PROJECT_NABLA_DIRECTORY:
            raise ValueError("THIS_PROJECT_NABLA_DIRECTORY environment variables doesn't exist!")
            
        os.chdir(THIS_PROJECT_NABLA_DIRECTORY)

        args = parseInputArguments()

        config = args.config
        libType = args.libType
        
        topBuildDirectory = os.path.normpath(os.path.join(THIS_PROJECT_NABLA_DIRECTORY, f"build/{libType}"))
        targetBuildDirectory = os.path.normpath(os.path.join(topBuildDirectory, args.target_project_directory)
        
        if topBuildDirectory == targetBuildDirectory:
            buildNabla(libType, config)
            cpackBundleHash = getCPackBundleHash(topBuildDirectory, "Libraries") + getCPackBundleHash(topBuildDirectory, "Runtime")
        else:
            buildProject(libType, config, targetBuildDirectory)
            cpackBundleHash += getCPackBundleHash(targetBuildDirectory, "Executables") + getCPackBundleHash(targetBuildDirectory, "Media")
            
        zk = connectToKazooServer("nabla.kazoo.server.windows") # DNS as compose service name
        createKazooAtomic(f"/{config}_{libType}_CPACK_INSTALL_CMAKE_PROJECTS", cpackBundleHash)
        zk.stop()
                
    except subprocess.CalledProcessError as e:
        print(f"Subprocess failed with exit code {e.returncode}")
        sys.exit(e.returncode)
        
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(-1)
      
      
if __name__ == "__main__":
    main()
