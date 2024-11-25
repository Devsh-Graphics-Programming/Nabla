@echo off
REM Set cache image reference
set IMAGE_NAME=dcr.devsh.eu/nabla/source/git-cache:latest

REM Start the git-cache-updater container in detached mode and capture the container ID
set CONTAINER_ID=
for /f "delims=" %%i in ('docker-compose run --remove-orphans -d git-cache-updater') do set CONTAINER_ID=%%i

REM Check if the container started successfully
if "%CONTAINER_ID%"=="" (
    echo Failed to start the git-cache-updater container.
    exit /b 1
)

echo Started container with ID %CONTAINER_ID%

REM Fetch master commits
docker exec -i -t %CONTAINER_ID% git fetch origin master
if %errorlevel% neq 0 (
    echo "Error: git fetch failed"
    docker stop %CONTAINER_ID%
    exit /b %errorlevel%
)

REM Checkout master. TODO: since it happens at runtime I could loop over /remotes' CURRENT branches and track all history
docker exec -i -t %CONTAINER_ID% git checkout master -f
if %errorlevel% neq 0 (
    echo "Error: git checkout failed"
    docker stop %CONTAINER_ID%
    exit /b %errorlevel%
)

REM Update & checkout submodules with CMake
docker exec -i -t %CONTAINER_ID% "C:\cmake\cmake-3.31.0-windows-x86_64\bin\cmake" -P cmake\submodules\update.cmake
if %errorlevel% neq 0 (
    echo "Error: CMake submodule update failed"
    docker stop %CONTAINER_ID%
    exit /b %errorlevel%
)

docker exec -i -t %CONTAINER_ID% cmd /C "for /d %%i in (*) do if /i not %%i==.git rmdir /s /q %%i"

if %errorlevel% neq 0 (
    echo "Error: failed to clean up files"
    docker stop %CONTAINER_ID%
    exit /b %errorlevel%
)

docker exec -i -t %CONTAINER_ID% cmd /C "for %%i in (*) do if /i not %%i==.git del /q %%i"

if %errorlevel% neq 0 (
    echo "Error: failed to clean up files"
    docker stop %CONTAINER_ID%
    exit /b %errorlevel%
)

REM Stop the container before committing
docker stop %CONTAINER_ID%
if %errorlevel% neq 0 (
    echo "Error: failed to stop container"
    exit /b %errorlevel%
)

REM Commit the updated container as a new image
docker commit %CONTAINER_ID% %IMAGE_NAME%
if %errorlevel% neq 0 (
    echo "Error: failed to commit the container"
    exit /b %errorlevel%
)

echo Git cache updated and committed as %IMAGE_NAME%.

REM Remove the update container
docker rm %CONTAINER_ID%
if %errorlevel% neq 0 (
    echo "Error: failed to remove the update container"
    exit /b %errorlevel%
)

echo Removed %CONTAINER_ID% update container.