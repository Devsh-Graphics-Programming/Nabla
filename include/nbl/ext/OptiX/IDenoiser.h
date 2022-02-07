// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_EXT_OPTIX_DENOISER_H_INCLUDED__
#define __NBL_EXT_OPTIX_DENOISER_H_INCLUDED__

#include "../../../../src/nbl/video/CCUDAHandler.h"

#include <optix.h>
#include <optix_denoiser_tiling.h>

namespace nbl
{
namespace ext
{
namespace OptiX
{
class IContext;

class IDenoiser final : public core::IReferenceCounted
{
public:
    inline OptixDenoiser getOptiXHandle() { return denoiser; }

    inline OptixResult computeMemoryResources(OptixDenoiserSizes* returnSizes, const uint32_t* maxresolution)
    {
        return optixDenoiserComputeMemoryResources(denoiser, maxresolution[0], maxresolution[1], returnSizes);
    }

    inline OptixResult getLastSetupResult() const { return alreadySetup; }

    inline OptixResult setup(CUstream stream, const uint32_t* outputDims,
        const cuda::CCUDAHandler::GraphicsAPIObjLink<video::IGPUBuffer>& stateBuffer, uint32_t stateSizeInBytes,
        const cuda::CCUDAHandler::GraphicsAPIObjLink<video::IGPUBuffer>& scratchBuffer, uint32_t scratchSizeInBytes,
        uint32_t stateBufferOffset = 0u, uint32_t scratchBufferOffset = 0u)
    {
        alreadySetup = optixDenoiserSetup(denoiser, stream, outputDims[0], outputDims[1],
            stateBuffer.asBuffer.pointer + stateBufferOffset, stateSizeInBytes,
            scratchBuffer.asBuffer.pointer + scratchBufferOffset, scratchSizeInBytes);
        return alreadySetup;
    }

    inline OptixResult computeIntensity(CUstream stream, const OptixImage2D* inputImage,
        const cuda::CCUDAHandler::GraphicsAPIObjLink<video::IGPUBuffer>& intensityBuffer,
        const cuda::CCUDAHandler::GraphicsAPIObjLink<video::IGPUBuffer>& scratchBuffer,
        size_t scratchSizeInBytes,
        uint32_t intensityBufferOffset = 0u, uint32_t scratchBufferOffset = 0u)
    {
        return optixDenoiserComputeIntensity(denoiser, stream, inputImage,
            intensityBuffer.asBuffer.pointer + intensityBufferOffset,
            scratchBuffer.asBuffer.pointer + scratchBufferOffset, scratchSizeInBytes);
    }

    inline OptixResult invoke(CUstream stream, const OptixDenoiserParams* params,
        const OptixImage2D* inputLayersBegin, const OptixImage2D* inputLayersEnd,
        const OptixImage2D* outputLayer,
        const cuda::CCUDAHandler::GraphicsAPIObjLink<video::IGPUBuffer>& scratchBuffer,
        size_t scratchSizeInBytes,
        uint32_t inputOffsetX = 0u, uint32_t inputOffsetY = 0u,
        const cuda::CCUDAHandler::GraphicsAPIObjLink<video::IGPUBuffer>& denoiserData = {},
        size_t denoiserDataSize = 0ull,
        uint32_t scratchBufferOffset = 0u, uint32_t denoiserDataOffset = 0u)
    {
        if(alreadySetup != OPTIX_SUCCESS)
            return alreadySetup;
        return optixDenoiserInvoke(denoiser, stream, params,
            denoiserData.asBuffer.pointer + denoiserDataOffset, denoiserDataSize,
            inputLayersBegin, inputLayersEnd - inputLayersBegin,
            inputOffsetX, inputOffsetY, outputLayer,
            scratchBuffer.asBuffer.pointer + scratchBufferOffset, scratchSizeInBytes);
    }

    inline OptixResult tileAndInvoke(
        CUstream stream,
        const OptixDenoiserParams* params,
        const OptixImage2D* inputLayers,
        unsigned int numInputLayers,
        const OptixImage2D* outputLayer,
        const cuda::CCUDAHandler::GraphicsAPIObjLink<video::IGPUBuffer>& scratch,
        size_t scratchSizeInBytes,
        unsigned int overlapWindowSizeInPixels,
        unsigned int tileWidth,
        unsigned int tileHeight,
        const cuda::CCUDAHandler::GraphicsAPIObjLink<video::IGPUBuffer>& denoiserState = {},
        size_t denoiserStateSizeInBytes = 0ull)
    {
        if(alreadySetup != OPTIX_SUCCESS)
            return alreadySetup;
        return optixUtilDenoiserInvokeTiled(
            denoiser,
            stream,
            params,
            denoiserState.asBuffer.pointer,
            denoiserStateSizeInBytes,
            inputLayers,
            numInputLayers,
            outputLayer,
            scratch.asBuffer.pointer,
            scratchSizeInBytes,
            overlapWindowSizeInPixels,
            tileWidth,
            tileHeight);
    }

protected:
    friend class OptiX::IContext;

    IDenoiser(OptixDenoiser _denoiser)
        : denoiser(_denoiser), alreadySetup(OPTIX_ERROR_DENOISER_NOT_INITIALIZED) {}
    ~IDenoiser()
    {
        if(denoiser)
            optixDenoiserDestroy(denoiser);
    }

    OptixDenoiser denoiser;
    OptixResult alreadySetup;
};

}
}
}

#endif