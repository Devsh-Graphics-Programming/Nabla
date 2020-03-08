#ifndef __IRR_EXT_OPTIX_DENOISER_H_INCLUDED__
#define __IRR_EXT_OPTIX_DENOISER_H_INCLUDED__

#include "../src/irr/video/CCUDAHandler.h"

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

		inline OptixResult setup(	CUstream stream, uint32_t* outputDims,
									const cuda::CCUDAHandler::GraphicsAPIObjLink<video::IGPUBuffer>& stateBuffer, uint32_t stateSizeInBytes,
									const cuda::CCUDAHandler::GraphicsAPIObjLink<video::IGPUBuffer>& scratchBuffer, uint32_t scratchSizeInBytes,
									uint32_t stateBufferOffset=0u, uint32_t scratchBufferOffset=0u)
		{
			alreadySetup = optixDenoiserSetup(	denoiser,stream,outputDims[0],outputDims[1],
												stateBuffer.asBuffer.pointer+stateBufferOffset,stateSizeInBytes,
												scratchBuffer.asBuffer.pointer+scratchBufferOffset,scratchSizeInBytes);
			return alreadySetup;
		}
/*
		inline OptixResult invoke(CUstream stream)
		{
			if (alreadySetup!=OPTIX_SUCCESS)
				return alreadySetup;
			return optixDenoiserInvoke(denoiser,stream,params,nullptr,0,inputLayers,numInputLayers,offx,offy,outputLayer,scratch,scratchBytes);
		}
*/
	protected:
		friend class OptiX::IContext;

		IDenoiser(OptixDenoiser _denoiser) : denoiser(_denoiser), alreadySetup(OPTIX_ERROR_DENOISER_NOT_INITIALIZED) {}
		~IDenoiser()
		{
			if (denoiser)
				optixDenoiserDestroy(denoiser);
		}

		OptixDenoiser denoiser;
		OptixResult alreadySetup;
};

}
}
}

#endif