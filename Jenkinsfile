stage('CMake')
{
	  agent.execute("cmake -DNBL_COMPILE_WITH_CUDA:BOOL=ON -DNBL_BUILD_OPTIX:BOOL=ON -DNBL_BUILD_MITSUBA_LOADER:BOOL=ON -DNBL_BUILD_RADEON_RAYS:BOOL=ON -DNBL_RUN_TESTS:BOOL=ON -S ./ -B ./build -T v143", true)
	  agent.execute("git -C ./3rdparty/gli reset --hard") // due to gli build system bug
	  agent.execute("cmake -DNBL_COMPILE_WITH_CUDA:BOOL=ON -DNBL_BUILD_OPTIX:BOOL=ON -DNBL_BUILD_MITSUBA_LOADER:BOOL=ON -DNBL_BUILD_RADEON_RAYS:BOOL=ON -DNBL_RUN_TESTS:BOOL=ON -S ./ -B ./build -T v143")
}

stage('Compile Nabla')
{
	  agent.execute("cmake --build ./build --target Nabla --config Release -j12 -v")
}
