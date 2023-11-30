# escape=`

ARG BASE_IMAGE=artifactory.devsh.eu/nabla/windows/base:latest

ARG NABLA_TARGET_REVISION="master"

FROM ${BASE_IMAGE}

SHELL ["cmd", "/S", "/C"]

RUN `
	# Create Nabla project directory
	`
	mkdir "%THIS_PROJECT_WORKING_DIRECTORY%\project\nabla" `
	`
	# Create NABLA_PROJECT_DIRECTORY environment variable
	`
	&& setx NABLA_PROJECT_DIRECTORY "%THIS_PROJECT_WORKING_DIRECTORY%\project\nabla" /M

ARG NABLA_TARGET_REVISION	

RUN `
	# Git init Nabla project
	git init "%NABLA_PROJECT_DIRECTORY%" `
	`
	# Add remote
	`
	&& git -C "%NABLA_PROJECT_DIRECTORY%" remote add origin https://github.com/Devsh-Graphics-Programming/Nabla.git `
	`
	# Shallow clone Nabla
	`
	&& git -C "%NABLA_PROJECT_DIRECTORY%" fetch --no-tags --force --progress --depth=1 -- origin %NABLA_TARGET_REVISION% `
	`
	# Checkout NABLA_TARGET_REVISION
	`
	&& git -C "%NABLA_PROJECT_DIRECTORY%" checkout %NABLA_TARGET_REVISION%
	
RUN `
	# Configure Nabla as static library
	`
	cmake -S "%NABLA_PROJECT_DIRECTORY%" --preset ci-configure-static-msvc
	
RUN `
	# Configure Nabla as dynamic library
	`
	cmake -S "%NABLA_PROJECT_DIRECTORY%" --preset ci-configure-dynamic-msvc
	
ENTRYPOINT ["VsDevCmd.bat", "&&", "powershell.exe", "-NoLogo", "-ExecutionPolicy", "Bypass"]