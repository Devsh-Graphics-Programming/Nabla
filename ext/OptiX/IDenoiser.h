#ifndef __IRR_EXT_OPTIX_DENOISER_H_INCLUDED__
#define __IRR_EXT_OPTIX_DENOISER_H_INCLUDED__

#include "optix.h"

namespace irr
{
namespace ext
{
namespace OptiX
{

class IContext;

class IDenoiser final : public core::IReferenceCounted
{
	public:
		inline OptixDenoiser getOptiXHandle() {return denoiser;}

		inline OptixResult computeMemoryResources(OptixDenoiserSizes* returnSizes, const uint32_t* maxresolution)
		{
			return optixDenoiserComputeMemoryResources(denoiser,maxresolution[0],maxresolution[1],returnSizes);
		}

	protected:
		friend class OptiX::IContext;

		IDenoiser(OptixDenoiser _denoiser) : denoiser(_denoiser) {}
		~IDenoiser()
		{
			if (denoiser)
				optixDenoiserDestroy(denoiser);
		}

		OptixDenoiser denoiser;
};

}
}
}

#endif