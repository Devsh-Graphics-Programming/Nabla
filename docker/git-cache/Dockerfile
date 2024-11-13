# escape=`

ARG BASE_IMAGE=mcr.microsoft.com/windows/servercore:ltsc2022-amd64

FROM ${BASE_IMAGE}

SHELL ["cmd", "/S", "/C"]

RUN ` 
    # Install Chocolatey
    `
    powershell -Command "Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://chocolatey.org/install.ps1'))"

RUN `
    # Install Git
    `
    choco install -y git --version 2.43.0

RUN `
	# Install CMake
	`
	curl -SL --output cmake.zip https://github.com/Kitware/CMake/releases/download/v3.31.0/cmake-3.31.0-windows-x86_64.zip `
	`
	&& mkdir "./cmake" `
	`
	&& tar -xf cmake.zip -C "./cmake" `
	`
	&& del /q cmake.zip

WORKDIR C:\gitcache

RUN `
    # Post environment setup
    `
    git config --system protocol.*.allow always `
    `
    && git config --system url."https://github.com/".insteadOf "git@github.com:" `
    `
    && git config --system --add safe.directory * `
    `
    && setx THIS_PROJECT_GIT_CACHE "C:\gitcache" `
    `
    && git init `
    `
    && git remote add origin https://github.com/Devsh-Graphics-Programming/Nabla.git 

ENTRYPOINT ["cmd.exe", "/K"]