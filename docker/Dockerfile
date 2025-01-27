# escape=`

ARG BASE_IMAGE=mcr.microsoft.com/windows/servercore:ltsc2022-amd64

FROM ${BASE_IMAGE}

SHELL ["cmd", "/S", "/C"]

USER ContainerAdministrator

ENV THIS_PROJECT_WORKING_DIRECTORY="C:\docker"
ENV THIS_PROJECT_NABLA_DIRECTORY="C:/Users/ContainerAdministrator/Nabla/bind"
ENV VULKAN_SDK_INSTALL_DIRECTORY="${THIS_PROJECT_WORKING_DIRECTORY}\dependencies\VulkanSDK"
ENV VS_INSTALL_DIRECTORY="${THIS_PROJECT_WORKING_DIRECTORY}\dependencies\VS\BuildTools"

RUN `
	# Download the Build Tools (17.11.5 October 8, 2024 version) bootstrapper. https://learn.microsoft.com/en-us/visualstudio/releases/2022/release-history
	`
	curl -SL --output vs_buildtools.exe https://download.visualstudio.microsoft.com/download/pr/69e24482-3b48-44d3-af65-51f866a08313/471c9a89fa8ba27d356748ae0cf25eb1f362184992dc0bb6e9ccf10178c43c27/vs_BuildTools.exe `
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
  	# Add VS's CMake to the system PATH and cleanup
	`
	&& setx PATH "%PATH%;%VS_INSTALL_DIRECTORY%\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin" /M `
	`
	# Cleanup
	`
	&& del /q vs_buildtools.exe

ENV VS_DEV_CMD_DIRECTORY="${VS_INSTALL_DIRECTORY}\Common7\Tools"

RUN `
	# Add VS_DEV_CMD_DIRECTORY to the system PATH
	`
	setx PATH "%PATH%;%VS_DEV_CMD_DIRECTORY%" /M

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
	# Download & install executable Strawberry Perl
	`
	choco install -y strawberryperl --version 5.28.2.1

RUN `
	# Download & install Python
	`
	choco install -y python --version 3.11.6

RUN `
	# Download & install git
	`
	choco install -y git --version 2.43.0

RUN `
	# Download & install nasm
	`
	choco install -y nasm --version 2.16.1

RUN `
	# Download & install ninja
	` 
	choco install -y ninja --version 1.12.1
	
RUN `
	# Enable Long Paths feature
	`
	reg add "HKLM\SYSTEM\CurrentControlSet\Control\FileSystem" /v "LongPathsEnabled" /t REG_DWORD /d 1 /f

RUN `
	# Force git to use HTTPS protocol & trust containers
	`
	git config --system protocol.*.allow always `
    `
    && git config --system url."https://github.com/".insteadOf "git@github.com:" `
	`
	&& git config --system --add safe.directory *

RUN `
	# Post environment setup
	`
	setx VS_INSTALL_DIRECTORY "%VS_INSTALL_DIRECTORY%" /M `
	`
	&& setx PATH "%PATH%;%VS_INSTALL_DIRECTORY%\VC\Auxiliary\Build" /M `
	`
	&& setx NBL_CI_MODE "ON"

WORKDIR ${THIS_PROJECT_NABLA_DIRECTORY}

COPY --from=dcr.devsh.eu/nabla/source/git-cache:latest /gitcache/.git ./.git

ENTRYPOINT ["cmd.exe", "/K"]