import os, subprocess, argparse
from .lib.kazoo import *

def parseInputArguments():
    parser = argparse.ArgumentParser(description="Nabla CI Pipeline nbl.ci.dev.cpack Framework Module")
    
    parser.add_argument("--config", help="Target CMake configuration", type=str, default="Release")
    parser.add_argument("--libType", help="Target library type", type=str, default="dynamic")

    args = parser.parse_args()
    
    return args

def cpack(libType, config, CPACK_INSTALL_CMAKE_PROJECTS, packageDirectory):
    if not packageDirectory:
        packageDirectory = f"./package/{config}/{libType}"
        
    return subprocess.run(f"cpack --preset ci-package-{libType}-msvc-{config} -B \"{packageDirectory}\" -D CPACK_INSTALL_CMAKE_PROJECTS=\"{CPACK_INSTALL_CMAKE_PROJECTS}\"", check=True)


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
            
        os.chdir(THIS_PROJECT_NABLA_DIRECTORY)

        args = parseInputArguments()

        config = args.config
        lowerCaseConfig = config.lower()
        libType = args.libType

        kazooConnector = KazooConnector(f"dev.nabla.kazoo.server.{libType}.{lowerCaseConfig}.x86_64.{THIS_PROJECT_PLATFORM}") # DNS record as compose service name
        kazooConnector.connect()

        zNodePath = f"/CPACK_INSTALL_CMAKE_PROJECTS"
        cpackBundleHash = kazooConnector.getKazooAtomic(zNodePath)
        print(f"Atomic read performed on {zNodePath} zNode path")

        kazooConnector.requestServerShutdown()
        kazooConnector.disconnect()
         
        if cpackBundleHash:
            print(f"CPACK_INSTALL_CMAKE_PROJECTS = {cpackBundleHash}")
            cpack(libType, lowerCaseConfig, cpackBundleHash, f"{THIS_PROJECT_NABLA_DIRECTORY}/build/artifacts/{THIS_PROJECT_PLATFORM}/{THIS_PROJECT_ARCH}/{config}/{libType}")
        else:
            print("CPACK_INSTALL_CMAKE_PROJECTS is empty, skipping cpack...")

    except subprocess.CalledProcessError as e:
        print(f"Subprocess failed with exit code {e.returncode}")
        sys.exit(e.returncode)
        
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(-1)
      
      
if __name__ == "__main__":
    main()
