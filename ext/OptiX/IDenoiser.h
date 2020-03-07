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