def execute(agent, config)
{
	stage("CMake")
	{
		  agent.execute("cmake -DNBL_UPDATE_GIT_SUBMODULE=OFF -DNBL_COMPILE_WITH_CUDA:BOOL=OFF -DNBL_BUILD_OPTIX:BOOL=OFF -DNBL_BUILD_MITSUBA_LOADER:BOOL=OFF -DNBL_BUILD_RADEON_RAYS:BOOL=OFF -DNBL_RUN_TESTS:BOOL=ON -S ./ -B ./build -T v143")
	}

	stage("Compile Nabla with ${config} configuration")
	{
		agent.execute("cmake --build ./build --target Nabla --config ${config} -j12 -v")
	}	
}

return this
