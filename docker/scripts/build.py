import os, subprocess, sys

try:
    THIS_PROJECT_NABLA_DIRECTORY = os.environ.get('THIS_PROJECT_NABLA_DIRECTORY', '')

    if not THIS_PROJECT_NABLA_DIRECTORY:
        raise ValueError("THIS_PROJECT_NABLA_DIRECTORY enviroment variables doesn't exist!")
    
    print(f"THIS_PROJECT_NABLA_DIRECTORY=\"{THIS_PROJECT_NABLA_DIRECTORY}\"")

    # Change the current working directory to THIS_PROJECT_NABLA_DIRECTORY
    os.chdir(THIS_PROJECT_NABLA_DIRECTORY)

    # Configure Nabla as static library
    subprocess.run("cmake -S . -B ./docker/build/static", check=True) # TODO: Make special docker cmake preset

except subprocess.CalledProcessError as e:
    print(f"Subprocess failed with exit code {e.returncode}")
    sys.exit(e.returncode)
    
except Exception as e:
    print(f"Unexpected error: {e}")
    sys.exit(-1)