# escape=`

ARG BASE_IMAGE=mcr.microsoft.com/windows:20H2-amd64

ARG THIS_PROJECT_WORKING_DIRECTORY="C:\docker"
ARG THIS_PROJECT_NABLA_DIRECTORY="${THIS_PROJECT_WORKING_DIRECTORY}\project\Nabla"
ARG VULKAN_SDK_INSTALL_DIRECTORY="${THIS_PROJECT_WORKING_DIRECTORY}\dependencies\VulkanSDK"
ARG PERL_INSTALL_DIRECTORY="${THIS_PROJECT_WORKING_DIRECTORY}\dependencies\StrawberryPerl"
ARG VS_INSTALL_DIRECTORY="${THIS_PROJECT_WORKING_DIRECTORY}\dependencies\VS\BuildTools"
ARG PYTHON_INSTALL_DIRECTORY="${THIS_PROJECT_WORKING_DIRECTORY}\dependencies\Python"
ARG GIT_INSTALL_DIRECTORY="${THIS_PROJECT_WORKING_DIRECTORY}\dependencies\git"
ARG NASM_INSTALL_DIRECTORY="${THIS_PROJECT_WORKING_DIRECTORY}\dependencies\nasm"
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

ARG PERL_INSTALL_DIRECTORY

RUN `
	# Download executable Strawberry Perl 5.28.2 distribution 
	`
	curl -SL --output strawberry-perl.msi https://strawberryperl.com/download/5.28.2.1/strawberry-perl-5.28.2.1-64bit.msi `
	`
	&& msiexec.exe /i strawberry-perl.msi /quiet /norestart TARGETDIR="%PERL_INSTALL_DIRECTORY%" `
    `
	# Cleanup
	`
	&& del /q strawberry-perl.msi

ARG PYTHON_INSTALL_DIRECTORY

RUN `
	# Download Python 3.10.9
	`
	curl -SL --output python-installer.exe https://www.python.org/ftp/python/3.10.9/python-3.10.9-amd64.exe `
	`
	# Install Python
	`
	&& start /w python-installer.exe /quiet TargetDir="%PYTHON_INSTALL_DIRECTORY%" Include_launcher=0 AddToPath=1 InstallAllUsers=1 PrependPath=1 `
	`
	# Cleanup
	`
	&& del /q python-installer.exe

ARG GIT_INSTALL_DIRECTORY

RUN `
	# Download Git
	`
	curl -SL --output git.zip https://github.com/git-for-windows/git/releases/download/v2.41.0.windows.3/MinGit-2.41.0.3-64-bit.zip `
	`
	# Create git install directory
	`
	&& mkdir "%GIT_INSTALL_DIRECTORY%" `
	`
	# Unzip Git.
	`
	&& tar -xf git.zip -C "%GIT_INSTALL_DIRECTORY%" `
	`
	# Add Git to the system PATH
	`
	&& setx PATH "%PATH%;%GIT_INSTALL_DIRECTORY%\cmd" /M `
	`
	# Cleanup
	`
	&& del /q git.zip

ARG NASM_INSTALL_DIRECTORY

RUN `
	# Download NASM
	`
	curl -SL --output nasm.zip https://www.nasm.us/pub/nasm/releasebuilds/2.16.02rc5/win64/nasm-2.16.02rc5-win64.zip `
	`
	# Create nasm install directory
	`
	&& mkdir "%NASM_INSTALL_DIRECTORY%" `
	`
	# Unzip NASM
	`
	&& tar -xf nasm.zip -C "%NASM_INSTALL_DIRECTORY%" `
	`
	# Add NASM to the system PATH
	`
	&& setx PATH "%PATH%;%NASM_INSTALL_DIRECTORY%" /M `
	`
	# Cleanup
	`
	&& del /q nasm.zip

ARG THIS_PROJECT_WORKING_DIRECTORY

RUN `
	setx THIS_PROJECT_WORKING_DIRECTORY "%THIS_PROJECT_WORKING_DIRECTORY%" /M

ARG THIS_PROJECT_NABLA_DIRECTORY

RUN `
	setx THIS_PROJECT_NABLA_DIRECTORY "%THIS_PROJECT_NABLA_DIRECTORY%" /M
	
RUN `
	git config --system --add safe.directory *

ARG VS_DEV_CMD_DIRECTORY

RUN `
	setx PATH "%PATH%;%VS_DEV_CMD_DIRECTORY%" /M

ENTRYPOINT ["VsDevCmd.bat", "&&", "powershell.exe", "-NoLogo", "-ExecutionPolicy", "Bypass"]