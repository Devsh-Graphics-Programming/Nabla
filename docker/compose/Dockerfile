# escape=`

ARG BASE_IMAGE=mcr.microsoft.com/windows/servercore:ltsc2022

ARG THIS_PROJECT_WORKING_DIRECTORY="C:\docker"
ARG THIS_PROJECT_NABLA_DIRECTORY="C:/Users/ContainerAdministrator/Nabla/bind"
ARG VULKAN_SDK_INSTALL_DIRECTORY="${THIS_PROJECT_WORKING_DIRECTORY}\dependencies\VulkanSDK"
ARG VS_INSTALL_DIRECTORY="${THIS_PROJECT_WORKING_DIRECTORY}\dependencies\VS\BuildTools"
ARG APACHE_ZOOKEEPER_INSTALL_DIRECTORY="${THIS_PROJECT_WORKING_DIRECTORY}\dependencies\ApacheZooKeeper"
ARG JAVA_HOME_SCRIPT="${THIS_PROJECT_WORKING_DIRECTORY}\dependencies\scripts\java"
ARG VS_DEV_CMD_DIRECTORY="${VS_INSTALL_DIRECTORY}\Common7\Tools"

FROM ${BASE_IMAGE}

SHELL ["cmd", "/S", "/C"]

ARG VS_INSTALL_DIRECTORY

RUN `
	# Download the Build Tools bootstrapper.
	`
	curl -SL --output vs_buildtools.exe https://aka.ms/vs/17/release/vs_buildtools.exe `
	`
	# Install Build Tools with the Microsoft.VisualStudio.Workload.VCTools recommended workload and ATL & ATLMFC, excluding some Windows SDKs.
	`
	&& (start /w vs_buildtools.exe --quiet --wait --norestart --nocache `
	--installPath "%VS_INSTALL_DIRECTORY%" `
	--add Microsoft.VisualStudio.Workload.VCTools --includeRecommended `
	--add Microsoft.VisualStudio.Component.VC.ATL `
	--add Microsoft.VisualStudio.Component.VC.ATLMFC `
	--remove Microsoft.VisualStudio.Component.Windows10SDK.10240 `
	--remove Microsoft.VisualStudio.Component.Windows10SDK.10586 `
	--remove Microsoft.VisualStudio.Component.Windows10SDK.14393 `
	--remove Microsoft.VisualStudio.Component.Windows81SDK `
	|| IF "%ERRORLEVEL%"=="3010" EXIT 0) `
 	`
  	# add VS's CMake to the system PATH and cleanup
	`
	&& setx PATH "%PATH%;%VS_INSTALL_DIRECTORY%\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin" /M `
	`
	# Cleanup
	`
	&& del /q vs_buildtools.exe

ARG VULKAN_SDK_INSTALL_DIRECTORY

RUN `
	# Download VulkanSDK
	`
	curl -SL --output VulkanSDK-Installer.exe https://sdk.lunarg.com/sdk/download/1.3.268.0/windows/VulkanSDK-1.3.268.0-Installer.exe `
	`
	# Install VulkanSDK
	`
	&& VulkanSDK-Installer.exe install --root "%VULKAN_SDK_INSTALL_DIRECTORY%" --default-answer --accept-licenses --confirm-command `
	`
	# Cleanup
	`
	&& del /q VulkanSDK-Installer.exe

RUN `
	# Download & install choco packet manager
	`
	powershell Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://chocolatey.org/install.ps1'))	

RUN `
	# Download & install executable Strawberry Perl 5.28.2.1
	`
	choco install -y strawberryperl --version 5.28.2.1

RUN `
	# Download & install Python 3.11.7
	`
	choco install -y python --version 3.11.7

RUN `
	# Donwload debugpy Python module
	`
	python -m pip install --upgrade debugpy

RUN `
	# Download & install git 2.43.0
	`
	choco install -y git --version 2.43.0

ARG APACHE_ZOOKEEPER_INSTALL_DIRECTORY

RUN `
	# Download Apache ZooKeeper
	`
	curl -SL --output zookeeper.zip https://dlcdn.apache.org/zookeeper/zookeeper-3.8.3/apache-zookeeper-3.8.3-bin.tar.gz `
	`
	# Create install directory
	`
	&& mkdir "%APACHE_ZOOKEEPER_INSTALL_DIRECTORY%" `
	`
	# Unpack
	`
    && tar -xf zookeeper.zip -C "%APACHE_ZOOKEEPER_INSTALL_DIRECTORY%" `
	`
	# Cleanup
	`
    && del /q zookeeper.zip `
	`
	&& setx PATH "%PATH%;%APACHE_ZOOKEEPER_INSTALL_DIRECTORY%\apache-zookeeper-3.8.3-bin\bin" /M

RUN `
	# Download kazoo 2.8.0 Python (more recent versions doesn't work well with Windows) module
	`
	python -m pip install kazoo==2.8.0
	
RUN `
	# Download psutil Python module
	`
	python -m pip install psutil

RUN `
	# Download OpenJDK 11 LTS
	`
	choco install -y openjdk11

RUN `
	# Download & install nasm 2.16.1
	`
	choco install -y nasm --version 2.16.1
	
RUN `
	# Download & install nano 7.2.36
	` 
	choco install -y nano --version 7.2.36

ARG THIS_PROJECT_WORKING_DIRECTORY

RUN `
	setx THIS_PROJECT_WORKING_DIRECTORY "%THIS_PROJECT_WORKING_DIRECTORY%" /M

ARG THIS_PROJECT_NABLA_DIRECTORY	
	
RUN `
	setx THIS_PROJECT_NABLA_DIRECTORY "%THIS_PROJECT_NABLA_DIRECTORY%" /M `
	&& setx PATH "%PATH%;%THIS_PROJECT_NABLA_DIRECTORY%/docker/scripts" /M
	
RUN `
	git config --system --add safe.directory *
	
ARG JAVA_HOME_SCRIPT	
	
COPY scripts\os\javaHome.py ${JAVA_HOME_SCRIPT}\javaHome.py

RUN `
	py "%JAVA_HOME_SCRIPT%\javaHome.py"
	
COPY scripts\os\resources\zoo.cfg ${APACHE_ZOOKEEPER_INSTALL_DIRECTORY}\apache-zookeeper-3.8.3-bin\conf\zoo.cfg

ARG VS_DEV_CMD_DIRECTORY

RUN `
	setx PATH "%PATH%;%VS_DEV_CMD_DIRECTORY%" /M
	
RUN `
	reg add "HKLM\SYSTEM\CurrentControlSet\Control\FileSystem" /v "LongPathsEnabled" /t REG_DWORD /d 1 /f

ENTRYPOINT ["powershell.exe", "-NoLogo", "-ExecutionPolicy", "Bypass"]