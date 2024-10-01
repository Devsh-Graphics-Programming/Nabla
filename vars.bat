call "C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Auxiliary/Build/vcvars64.bat"
call wsl -e fish
@REM cmake.exe -DNBL_DYNAMIC_MSVC_RUNTIME=OFF -DNBL_STATIC_BUILD=ON -DNBL_CI_MODE=OFF -DNBL_UPDATE_GIT_SUBMODULE=ON -DNBL_RUN_TESTS=OFF -DNBL_CPACK_CI=OFF -SD:/Nabla -BD:/Nabla/build/static -G "Ninja Multi-Config"