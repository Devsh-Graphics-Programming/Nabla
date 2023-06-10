class Builder
{
	public static enum BUILD_TYPE 
	{
		STATIC, DYNAMIC
	}
	
	public Builder(_agent)
	{
		agent = _agent
	}
	
	public def cmake(BUILD_TYPE buildType, platform)
	{
		// use Multi-Configuration generators only regarding the platform in future
		def commonFlags = "-DNBL_UPDATE_GIT_SUBMODULE=OFF -DNBL_COMPILE_WITH_CUDA:BOOL=OFF -DNBL_BUILD_OPTIX:BOOL=OFF -DNBL_BUILD_MITSUBA_LOADER:BOOL=OFF -DNBL_BUILD_RADEON_RAYS:BOOL=OFF -DNBL_RUN_TESTS:BOOL=ON"
		def extraFlags = ""
		def toolchain = "v143"
		def buildDirectory = getNameOfBuildDirectory(buildType)
		
		switch (buildType)
		{
			case BUILD_TYPE.STATIC:
				break
			case BUILD_TYPE.DYNAMIC:
				extraFlags = "-DNBL_STATIC_BUILD=OFF -DNBL_DYNAMIC_MSVC_RUNTIME=ON"
				break	
		}
		
		agent.execute("cmake ${commonFlags} ${extraFlags} -S ./ -B ./${buildDirectory} -T ${toolchain}")
	}
	
	public def build(config, BUILD_TYPE buildType, platform)
	{
		def buildDirectory = getNameOfBuildDirectory(buildType)
		agent.execute("cmake --build ./${buildDirectory} --target Nabla --config ${config} -j12 -v")
	}
	
	private def getNameOfBuildDirectory(BUILD_TYPE buildType)
	{
		switch (buildType)
		{
			case BUILD_TYPE.STATIC:
				return "build_static"
			case BUILD_TYPE.DYNAMIC:
				return "build_dynamic"
		}
	}
	
	private def agent
}

def create(_agent)
{
	return new Builder(_agent)
}

return this
