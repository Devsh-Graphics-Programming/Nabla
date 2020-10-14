#ifndef __IRR_EXT_OPTIX_MODULE_H_INCLUDED__
#define __IRR_EXT_OPTIX_MODULE_H_INCLUDED__

#include "irr/core/core.h"

#include "optix.h"

namespace irr
{
namespace ext
{
namespace OptiX
{

class IContext;


class IModule final : public core::IReferenceCounted
{
	public:
		inline OptixModule getOptiXHandle() {return module;}

	protected:
		friend class OptiX::IContext;

		IModule(const OptixModule& _module) : module(_module) {}
		~IModule()
		{
			if (module)
				optixModuleDestroy(module);
		}

		OptixModule module;
};


}
}
}

#endif