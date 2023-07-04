import org.DevshGraphicsProgramming.Agent
import org.DevshGraphicsProgramming.BuilderInfo
import org.DevshGraphicsProgramming.IBuilder

class CNablaBuilder extends IBuilder
{
	public static enum PRESET_TYPE 
	{
		CONFIGURE, BUILD
	}

	public CNablaBuilder(Agent _agent, BuilderInfo _info)
	{
		super(_agent, _info)
	}
	
	@Override
	public boolean prepare(Map axisMapping)
	{
		final def preset = getPreset(PRESET_TYPE.CONFIGURE, axisMapping)
			
		agent.execute("cmake . --preset ${preset}")
		
		return true
	}
	
	@Override
  	public boolean build(Map axisMapping)
	{
		final def preset = getPreset(PRESET_TYPE.BUILD, axisMapping)	
		final def nameOfConfig = getNameOfConfig(axisMapping.get("CONFIGURATION"))
		
		agent.execute("cmake --build --preset ${preset} --config ${nameOfConfig} -j12 -v")
		
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
	
	private def getPreset(final PRESET_TYPE presetType, final Map _axisMapping) // currently we only maintain Windows as host with MSVC target
	{
		def mode, preset
		
		switch (presetType)
		{
			case PRESET_TYPE.CONFIGURE:
				mode = "configure"
				break
			case PRESET_TYPE.BUILD:
				mode = "build"
				break
		}
	
		switch (_axisMapping.get("BUILD_TYPE"))
		{
			case IBuilder.BUILD_TYPE.STATIC:
				preset = "ci-${mode}-static-msvc"
				break
			case IBuilder.BUILD_TYPE.DYNAMIC:
				preset = "ci-${mode}-dynamic-msvc"
				break
		}
		
		return preset
	}
}

def create(Agent _agent, BuilderInfo _info)
{
	return new CNablaBuilder(_agent, _info)
}

return this