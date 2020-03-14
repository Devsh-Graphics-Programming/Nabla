#ifndef __IRR_EXT_OPTIX_PROGRAM_GROUP_H_INCLUDED__
#define __IRR_EXT_OPTIX_PROGRAM_GROUP_H_INCLUDED__

#include "irr/core/core.h"

#include "optix.h"

namespace irr
{
namespace ext
{
namespace OptiX
{

class IContext;


class IProgramGroup final : public core::IReferenceCounted
{
	public:
		inline OptixProgramGroup getOptiXHandle() {return programGroup;}

	protected:
		friend class OptiX::IContext;

		IProgramGroup(const OptixProgramGroup& _programGroup) : programGroup(_programGroup) {}
		~IProgramGroup()
		{
			if (programGroup)
				optixProgramGroupDestroy(programGroup);
		}

		OptixProgramGroup programGroup;
};


}
}
}

#endif