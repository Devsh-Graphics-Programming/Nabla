def cmake(agent, buildType, platform)
{
	// use Multi-Configuration generators only regarding the platform in future
	
	stage("Configure Nabla project with CMake as ${buildType} build")
	{
		if(buildType == "Static")
		{
			agent.execute("cmake -DNBL_UPDATE_GIT_SUBMODULE=OFF -DNBL_COMPILE_WITH_CUDA:BOOL=OFF -DNBL_BUILD_OPTIX:BOOL=OFF -DNBL_BUILD_MITSUBA_LOADER:BOOL=OFF -DNBL_BUILD_RADEON_RAYS:BOOL=OFF -DNBL_RUN_TESTS:BOOL=ON -S ./ -B ./build_static -T v143")
		}
		else if(buildType == "Dynamic")
		{
			agent.execute("cmake -DNBL_STATIC_BUILD=OFF -DNBL_DYNAMIC_MSVC_RUNTIME=ON -DNBL_UPDATE_GIT_SUBMODULE=OFF -DNBL_COMPILE_WITH_CUDA:BOOL=OFF -DNBL_BUILD_OPTIX:BOOL=OFF -DNBL_BUILD_MITSUBA_LOADER:BOOL=OFF -DNBL_BUILD_RADEON_RAYS:BOOL=OFF -DNBL_RUN_TESTS:BOOL=ON -S ./ -B ./build_dynamic -T v143")
		}
		else 
		{
			error "Intenral error!"
		}
	}
}

def build(agent, config, buildType, platform)
{
	stage("Build ${buildType} Nabla with ${config} configuration")
	{
		def buildDirectory = buildType == "Static" ? "build_static" : (buildType == "Dynamic" ? "build_dynamic" : "")
		agent.execute("cmake --build ./${buildDirectory} --target Nabla --config ${config} -j12 -v")
	}	
}

return this
