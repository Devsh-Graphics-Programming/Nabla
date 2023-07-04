import org.DevshGraphicsProgramming.Agent
import org.DevshGraphicsProgramming.BuilderInfo
import org.DevshGraphicsProgramming.IBuilder

class CCPack extends IBuilder
{
	public CCPack(Agent _agent, BuilderInfo _info)
	{
		super(_agent, _info)
	}
	
	@Override
	public boolean prepare(Map axisMapping)
	{		
		return true
	}
	
	@Override
  	public boolean build(Map axisMapping)
	{
		return true
	}
	
	@Override
  	public boolean test(Map axisMapping)
	{
		return true
	}
	
	@Override
	public boolean install(Map axisMapping)
	{
		return true
	}
	
	public boolean install(final Map axisMapping, final String CPACK_INSTALL_CMAKE_PROJECTS) // TODO: I start thinking we should create metadata interface and pass an extra metadata Object nulled by default for each interface abstract method IBuilder declares 
	{
		final def preset = getPreset(axisMapping)	
		final def nameOfConfig = getNameOfConfig(axisMapping.get("CONFIGURATION"))
		
		agent.execute("cpack --preset ${preset} -C ${nameOfConfig} -D CPACK_INSTALL_CMAKE_PROJECTS=\"${CPACK_INSTALL_CMAKE_PROJECTS}\"")
	
		return true
	}
	
	private def getPreset(final Map _axisMapping) // currently we only maintain Windows as host with MSVC target
	{
		def preset
		
		switch (_axisMapping.get("BUILD_TYPE"))
		{
			case IBuilder.BUILD_TYPE.STATIC:
				preset = "ci-package-static-msvc"
				break
			case IBuilder.BUILD_TYPE.DYNAMIC:
				preset = "ci-package-dynamic-msvc"
				break
		}
		
		return preset
	}
}

def create(Agent _agent, BuilderInfo _info)
{
	return new CCPack(_agent, _info)
}

return this