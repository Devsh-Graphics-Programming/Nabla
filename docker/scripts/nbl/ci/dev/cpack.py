import os, subprocess, sys, argparse, glob, debugpy, asyncio, socket

def parseInputArguments():
    parser = argparse.ArgumentParser(description="Nabla CI Pipeline nbl.ci.dev.cpack Framework Module")
    
    arser.add_argument("--config", help="Target CMake configuration", type=str, default="Release")
    parser.add_argument("--libType", help="Target library type", type=str, default="dynamic")

    args = parser.parse_args()
    
    return args

def cpack(libType, config, CPACK_INSTALL_CMAKE_PROJECTS, packageDirectory):
    if not packageDirectory:
        packageDirectory = f"./package/{config}"
        
    return subprocess.run(f"cpack --preset ci-package-{libType}-msvc -C {config} -B \"{packageDirectory}\" -D CPACK_INSTALL_CMAKE_PROJECTS=\"{CPACK_INSTALL_CMAKE_PROJECTS}\"", check=True)


def main():
    try:
        THIS_PROJECT_NABLA_DIRECTORY = os.environ.get('THIS_PROJECT_NABLA_DIRECTORY', '')

        if not THIS_PROJECT_NABLA_DIRECTORY:
            raise ValueError("THIS_PROJECT_NABLA_DIRECTORY environment variables doesn't exist!")

        THIS_PROJECT_ARTIFACTORY_NABLA_DIRECTORY = os.environ.get('THIS_PROJECT_ARTIFACTORY_NABLA_DIRECTORY', '')

        if not THIS_PROJECT_ARTIFACTORY_NABLA_DIRECTORY:
            print("THIS_PROJECT_ARTIFACTORY_NABLA_DIRECTORY not defined, using default <topBuildDirectory>/package/<configuration> path for artifacts")
            
        os.chdir(THIS_PROJECT_NABLA_DIRECTORY)

        args = parseInputArguments()

        config = args.config
        libType = args.libType
        
        zk = connectToKazooServer("nabla.kazoo.server.windows") # DNS as compose service name
        cpackBundleHash = getKazooAtomic(f"/{config}_{libType}_CPACK_INSTALL_CMAKE_PROJECTS")
        zk.stop()
         
        if cpackBundleHash:
            cpack(libType, config, cpackBundleHash, f"{THIS_PROJECT_ARTIFACTORY_NABLA_DIRECTORY}/Nabla/artifacts/{config}")
        else:
            print("CPACK_INSTALL_CMAKE_PROJECTS is empty, skipping cpack..."
                
    except subprocess.CalledProcessError as e:
        print(f"Subprocess failed with exit code {e.returncode}")
        sys.exit(e.returncode)
        
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(-1)
      
      
if __name__ == "__main__":
    main()
