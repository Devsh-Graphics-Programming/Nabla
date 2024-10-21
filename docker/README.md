# Nabla docker CI build pipelines

## Run Nabla build pipelines locally in containers and produce artifacts

Currently only Windows platform with target *x86_64* architecture is supported.

### Requirements

- [***Docker Desktop***](https://www.docker.com/products/docker-desktop/)
- [***Enabled HyperV if you want to virtualize environment or Windows Pro/Enterprise with minimum 19045 OS build version if you want to run containers with no virtualization as host processes (recommended for much better performance)***](https://docs.docker.com/desktop/install/windows-install/#system-requirements)

### How To

#### Wish to execute the pipelines with isolated virtualized environment or as processes with native performance?

Worth to mention *whenever you launch any **terminal** you execute docker commands with* always grant **administrator privileges** or **[add yourself to docker-users group](https://stackoverflow.com/a/67831886),** it's annoying to do it each time.

##### Hyper-V isolation (slow but production safe)

By default Docker Desktop forces you to use either WSL or Hyper-V so you just need to run Docker Desktop Engine and make sure you are switched to `Windows Containers` mode - if you want to use virtualized environment then you can just skip to next paragraph with volumes.

![Containers for Windows](https://user-images.githubusercontent.com/65064509/152947300-affca592-35a7-4e4c-a7fc-2055ce1ba528.png)

##### Process isolation (native performance, less safe)

If you want to run containers without virtualization then you need to go through some steps and execute [Docker Daemon](https://docs.docker.com/config/daemon/) yourself. Process isolation is possible only with Windows Pro or Enterprise edition. Nabla docker pipelines use [Windows Server Core 2022 LTSC](https://hub.docker.com/_/microsoft-windows-servercore) tag images, you can override a base OS image but then you need to [make sure it will work on your machine](https://learn.microsoft.com/en-us/virtualization/windowscontainers/deploy-containers/version-compatibility?tabs=windows-server-2022%2Cwindows-11). By default Docker Desktop installs the Daemon in `C:\Program Files\Docker\Docker\resources` path and is called `dockerd.exe`. To execute the Daemon with process isolation mode you need to 

```bash
dockerd --exec-opt isolation=process
```

and ***keep it running when working with containers***. If you see `failed to load listeners: open //./pipe/docker_engine_windows: Access is denied` error then Docker Desktop is most probably already running its own instance of the daemon - you need to kill all of its processes, task manager it is. If there are no issues and the Daemon is running correctly execute

```bash
docker ps
```

and check if there are any errors from client side. If there are no issues go to next paragraph with volumes, otherwise most probably your error is related to default `docker_engine_windows` pipe name set by Docker Desktop configuration file - you may need to adjust its name by creating yourown json configuration file with following content

```json
{
  "experimental": true,
  "hosts": [
    "npipe:////./pipe/docker_engine"
  ]
}
```

and passing to the Daemon as argument by

```bash
dockerd --exec-opt isolation=process --config-file <path_to_your_created_json_config>
```

#### Create volumes shared between containers

Nabla doesn't support *https* clones hence you will need to share your SSH key with the containers. To do so you need to create [docker volume](https://docs.docker.com/storage/volumes/) which stores your SSH key. Nabla compose files expects this volume to be named `ssh`. Create the volume by

```bash
docker volume create ssh
```

A volume called `aritfactory.nabla` for Nabla output pipeline artifacts is required too

```bash
docker volume create aritfactory.nabla
```

Locate your SSH key, open your terminal and execute

```bash
type <path_to_your_ssh_key> > <ssh_volume_path>/id_rsa
```

to copy your key to `ssh` volume.

#### Build base image with OS setup

```bash
cd .\docker\platform\windows\compose\ci
```

and build the base image. If you want to build *with default setup* 

```bash
docker compose build base
```

you may also want to change *its base* image (or any other available ARG variable)

```bash
docker compose build --build-arg BASE_IMAGE="mcr.microsoft.com/windows:ltsc2019" base
```

in case your operating system doesn't meet some requirements for instance due to your distribution build number which is too low - you should override `BASE_IMAGE` build argument.

#### Adjust permissions of your SSH key in `ssh` volume

To make sure there are no issues with the key ownership and permissions, grant them to containers by executing following command

```bash
docker run --rm -t -i --mount type=volume,src=ssh,dst=C:/Users/ContainerAdministrator/.ssh --mount type=volume,src=nabla,dst=C:/Users/ContainerAdministrator/Nabla artifactory.devsh.eu/nabla/windows/base icacls.exe C:/Users/ContainerAdministrator/.ssh/id_rsa /reset; icacls.exe C:/Users/ContainerAdministrator/.ssh/id_rsa /GRANT:R 'ContainerAdministrator:(R)'; icacls.exe C:/Users/ContainerAdministrator/.ssh/id_rsa /inheritance:r; ssh -T git@github.com
```

note that this creates intermediate container which terminates itself on finish with mounted `ssh` volume you created before, it grants permissions to the key and also performs ssh request to *github.com* with you key - you should see your nickname with a message about successful authentication.

#### Run an instance container, cache build directories and commit to instance image

Run base container which will clone Nabla, update submodules and configure build directories with static & dynamic library types, you can also specify additional optional `--target-revision <NABLA_TARGET_REVISION>` runtime argument at the end to force the container to clone specified revision or branch - by default `docker` branch is taken and its HEAD commit is cloned.

```bash
docker compose run -i --name dev.nabla.base.x86_64.windows base
```

Once everything is built commit your container to an instance image by executing

```bash
docker commit dev.nabla.base.x86_64.windows artifactory.devsh.eu/nabla/windows/instance:latest
```

This will create `artifactory.devsh.eu/nabla/windows/instance:latest` image with configured build directories cached and ready to being built by all containers which will use this image.

#### Run the Nabla CI build pipeline

```bash
cd .\instances
```

and

```bash
docker compose up
```

It will run all services as containers from the previously created instance image executing independently CI pipeline axes and output artifacts from the services to `artifactory.nabla` volume.

### Remote Debugging Container build pipelines

```
set BUILD_SCRIPT_ARGS="--debug=True" 
```

and `docker compose up` in the *instances* directory. Install [Visual Studio Code](https://code.visualstudio.com/), navigate to `<NBL_ROOT_DIRECTORY>/docker/scripts` directory, open the 

directory *with Code*, hit *Run and Debug* and pick the pipeline service configuration you would like to Debug.

![dcp](https://github.com/Devsh-Graphics-Programming/Nabla-Site-Media/blob/master/docker/readme/dockerPipelineConfigurations.png?raw=true)
