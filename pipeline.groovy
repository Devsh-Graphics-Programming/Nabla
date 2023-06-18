import org.DevshGraphicsProgramming.Agent
import org.DevshGraphicsProgramming.IBuilder

class CNablaBuilder extends IBuilder
{
	@InheritConstructors
	public CNablaBuilder(Agent _agent)
	{
		
	}
	
	@Override
	public boolean prepare(Map axisMapping)
	{
		impl_cmake(axisMapping.get("BUILD_TYPE"))
		return true
	}
	
	@Override
  	public boolean build(Map axisMapping)
	{
		impl_build(axisMapping.get("CONFIGURATION"), axisMapping.get("BUILD_TYPE"))
		return true
	}
	
	@Override
  	public boolean test(Map axisMapping)
	{
		return true
	}
	
	@Override
	public boolean deploy(Map axisMapping)
	{
		return true
	}
	
	private def impl_cmake(BUILD_TYPE buildType)
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
	
	private def impl_build(CONFIGURATION config, BUILD_TYPE buildType)
	{
		def buildDirectory = getNameOfBuildDirectory(buildType)
		def nameOfConfig = getNameOfConfig(config)
		agent.execute("cmake --build ./${buildDirectory} --target Nabla --config ${nameOfConfig} -j12 -v")
	}
}

def create(Agent _agent)
{
	return new CNablaBuilder(_agent)
}

return this
