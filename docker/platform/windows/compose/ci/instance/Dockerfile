# escape=`

# This image is used for CI production runs to cache configured
# build directory ,also a shallow clone is performed 
# each time targeting NABLA_TARGET_REVISION.
#
# For a development & debug local runs the base image 
# is used which mounts Nabla from your host machine 
# to into container services as using bind mounts
# https://docs.docker.com/storage/bind-mounts/#configure-bind-propagation

ARG BASE_IMAGE=artifactory.devsh.eu/nabla/windows/base:latest

ARG NABLA_TARGET_REVISION="master"

FROM ${BASE_IMAGE}

SHELL ["cmd", "/S", "/C"]

ARG NABLA_TARGET_REVISION

RUN `
	setx NABLA_TARGET_REVISION "${NABLA_TARGET_REVISION}" /M

RUN `
	# Create Nabla project directory
	`
	mkdir "%THIS_PROJECT_NABLA_DIRECTORY%" `
	`
	# Git init Nabla project
	`
	git init "%THIS_PROJECT_NABLA_DIRECTORY%" `
	`
	# Add remote
	`
	&& git -C "%THIS_PROJECT_NABLA_DIRECTORY%" remote add origin https://github.com/Devsh-Graphics-Programming/Nabla.git `
	`
	# Shallow clone Nabla
	`
	&& git -C "%THIS_PROJECT_NABLA_DIRECTORY%" fetch --no-tags --force --progress --depth=1 -- origin %NABLA_TARGET_REVISION% `
	`
	# Checkout NABLA_TARGET_REVISION
	`
	&& git -C "%THIS_PROJECT_NABLA_DIRECTORY%" checkout %NABLA_TARGET_REVISION%
	
RUN `
	# Configure Nabla as static library
	`
	cmake -S "%THIS_PROJECT_NABLA_DIRECTORY%" --preset ci-configure-static-msvc
	
RUN `
	# Configure Nabla as dynamic library
	`
	cmake -S "%THIS_PROJECT_NABLA_DIRECTORY%" --preset ci-configure-dynamic-msvc
	
ENTRYPOINT ["VsDevCmd.bat", "&&", "powershell.exe", "-NoLogo", "-ExecutionPolicy", "Bypass"]