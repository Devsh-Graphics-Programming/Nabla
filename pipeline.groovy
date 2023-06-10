class Builder
{
	public static enum BUILD_TYPE 
	{
		STATIC, DYNAMIC
	}

	public static enum CONFIGURATION 
	{
		RELEASE, RELWITHDEBINFO, DEBUG
	}

	public static enum ARCH
	{
		X86_64	
	}

	public static enum PLATFORM
	{
		WINDOWS	
	}
	
	public def getBuildTypes() // due to groovy sandbox
	{
		return [BUILD_TYPE.STATIC, BUILD_TYPE.DYNAMIC]	
	}
	
	public Builder(_agent, PLATFORM _platform)
	{
		agent = _agent
		platform = _platform
		
		matrixAxes = [
			ARCH: [ARCH.X86_64], // Hardcoded since we only target one arch
			PLATFORM: [_platform], // Platform is determined by an upstream host
			BUILD_TYPE: [BUILD_TYPE.STATIC, BUILD_TYPE.DYNAMIC],
			CONFIGURATION: [CONFIGURATION.RELEASE, CONFIGURATION.RELWITHDEBINFO, CONFIGURATION.DEBUG]
		    ]
		
		axes = getMatrixAxes(matrixAxes)
	}
	
	public def cmake(BUILD_TYPE buildType)
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
	
	public def build(CONFIGURATION config, BUILD_TYPE buildType)
	{
		def buildDirectory = getNameOfBuildDirectory(buildType)
		def nameOfConfig = getNameOfConfig(config)
		agent.execute("cmake --build ./${buildDirectory} --target Nabla --config ${nameOfConfig} -j12 -v")
	}
	
	private def getAxes() 
	{
	    	return axes
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
	
	private def getNameOfConfig(CONFIGURATION config)
	{
		switch (config)
		{
			case CONFIGURATION.RELEASE:
				return "Release"
			case CONFIGURATION.RELWITHDEBINFO:
				return "RelWithDebInfo"
			case CONFIGURATION.DEBUG:
				return "Debug"
		}
	}
	
	@NonCPS
	private def getMatrixAxes(Map matrix_axes) 
	{
	    	List axes = []
	    	matrix_axes.each { axis, values ->
			List axisList = []
			values.each { value ->
		    		axisList << [(axis): value]
			}
		axes << axisList
	    }
	    // calculate cartesian product
	    axes.combinations()*.sum()
	}
	
	private def agent
	private PLATFORM platform
	private def matrixAxes
	private def axes
}

def create(_agent, _platform)
{
	if(_platform == "Windows")
		return new Builder(_agent, Builder.PLATFORM.WINDOWS)
	else
		throw new Exception("Could not create Builder due to unknown platform!")
}

return this
