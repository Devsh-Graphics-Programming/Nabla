#ifndef _NBL_BUILTIN_HLSL_SUBGROUP_SHUFFLE_PORTABILITY_INCLUDED_
#define _NBL_BUILTIN_HLSL_SUBGROUP_SHUFFLE_PORTABILITY_INCLUDED_

#include <nbl/builtin/hlsl/subgroup/basic_portability.hlsl>

namespace nbl
{
namespace hlsl
{
namespace subgroup
{
	template<typename T>
	T Shuffle(T value, uint index)
	{
	#ifdef NBL_GL_KHR_shader_subgroup_shuffle
		return native::Shuffle<T>(value, index); // TODO (PentaKon): Maybe change to functor, depending on portability implementation
	#else
		return portability::Shuffle<T>(value, index);
	#endif
	}
	
	template<typename T>
	T ShuffleUp(T value, uint delta)
	{
	  return Shuffle(value, ID() - delta);
	}
	
	template<typename T>
	T ShuffleDown(T value, uint delta)
	{
	  return Shuffle(value, ID() + delta);
	}
	
namespace native 
{
	template<typename T>
	T Shuffle(T value, uint index)
	{
		return WaveReadLaneFirst(value, index);
	}
}	
namespace portability
{
	template<typename T>
	T Shuffle(T value, uint index)
	{
		// TODO (PentaKon): Implement
		return value; // placeholder
	}
}	
}
}
}

#endif