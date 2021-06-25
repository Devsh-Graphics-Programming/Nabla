// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#define _NBL_STATIC_LIB_
#include <nabla.h>

using namespace nbl;
using namespace nbl::asset;
using namespace nbl::core;
using namespace nbl::video;

/*
	Discrete convolution for getting input image after SAT calculations

	- support [-1.5,1.5]
	- (weight = -1) in [-1.5,-0.5]
	- (weight = 1) in [-0.5,0.5]
	- (weight = 0) in [0.5,1.5] and in range over the support
*/

using CDiscreteConvolutionRatioForSupport = std::ratio<3, 2>; //!< 1.5
class CDiscreteConvolutionFilterKernel : public CFloatingPointSeparableImageFilterKernelBase<CDiscreteConvolutionFilterKernel>
{
	using Base = CFloatingPointSeparableImageFilterKernelBase<CDiscreteConvolutionFilterKernel>;

public:
	CDiscreteConvolutionFilterKernel() : Base(1.5f, 0.5f) {}

	inline float weight(float x, int32_t channel) const
	{
		if (x >= -1.5f && x <= -0.5f)
			return -1.0f;
		else if (x >= -0.5f && x <= 0.5f)
			return 1.0f;
		else
			return 0.0f;
	}
};

using DiscreteConvolutionBlitFilter = asset::CBlitImageFilter<false, true, DefaultSwizzle, CWhiteNoiseDither, CDiscreteConvolutionFilterKernel, CDiscreteConvolutionFilterKernel, CBoxImageFilterKernel>;

int main()
{
	nbl::SIrrlichtCreationParameters params;
	params.Bits = 24; //may have to set to 32bit for some platforms
	params.ZBufferBits = 24; //we'd like 32bit here
	params.DriverType = video::EDT_OPENGL; //! Only Well functioning driver, software renderer left for sake of 2D image drawing
	params.WindowSize = core::dimension2d<uint32_t>(1280, 720);
	params.Fullscreen = false;
	params.Vsync = false;
	params.Doublebuffer = true;
	params.Stencilbuffer = false; //! This will not even be a choice soon
	params.AuxGLContexts = 16;
	auto device = createDeviceEx(params);

	if (!device)
		return 1;

	video::IVideoDriver* driver = device->getVideoDriver();
    IAssetManager* assetManager = device->getAssetManager();

	// Create inImage from buffer
	const std::array<uint32_t, 2> inImageDims = { 3u, 2u };

	IImage::SCreationParams inImageParams = {};
	inImageParams.flags = static_cast<IImage::E_CREATE_FLAGS>(0u);
	inImageParams.type = IImage::ET_2D;
	inImageParams.format = asset::EF_R32_SFLOAT;
	inImageParams.extent = { inImageDims[0], inImageDims[1], 1 };
	inImageParams.arrayLayers = 1u;
	inImageParams.mipLevels = 1u;
	inImageParams.samples = asset::ICPUImage::ESCF_1_BIT;

	auto inImageRegions = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<asset::IImage::SBufferCopy>>(1ull);
	auto& region = (*inImageRegions)[0];
	region.bufferImageHeight = 0u;
	region.bufferOffset = 0ull;
	region.bufferRowLength = inImageDims[0];
	region.imageExtent = { inImageDims[0], inImageDims[1], 1u };
	region.imageOffset = { 0u, 0u, 0u };
	region.imageSubresource.baseArrayLayer = 0u;
	region.imageSubresource.layerCount = 1u;
	region.imageSubresource.mipLevel = 0u;

	auto inImageBuffer = core::make_smart_refctd_ptr<asset::ICPUBuffer>(inImageDims[0] * inImageDims[1] * asset::getTexelOrBlockBytesize(inImageParams.format));

	float* inPixel = (float*)inImageBuffer->getPointer();
	float k = 1.f;
	for (uint32_t y = 0u; y < inImageDims[1]; ++y)
	{
		for (uint32_t x = 0u; x < inImageDims[0]; ++x)
		{
			inPixel[y * inImageDims[0] + x] = k++;
		}
	}

	core::smart_refctd_ptr<ICPUImage> inImage = ICPUImage::create(std::move(inImageParams));
	inImage->setBufferAndRegions(core::smart_refctd_ptr(inImageBuffer), inImageRegions);

	// Create out image
	auto outImage = core::move_and_static_cast<ICPUImage>(inImage->clone());
	memset(outImage->getBuffer()->getPointer(), 0, outImage->getBuffer()->getSize());

	{
		DiscreteConvolutionBlitFilter blitImageFilter;
		DiscreteConvolutionBlitFilter::state_type state = {};

		core::vectorSIMDu32 extentLayerCount = core::vectorSIMDu32(0, 0, 0, inImage->getCreationParameters().arrayLayers) + inImage->getMipSize();

		state.inOffsetBaseLayer = core::vectorSIMDu32();
		state.inExtentLayerCount = extentLayerCount;
		state.inImage = inImage.get();

		state.outOffsetBaseLayer = core::vectorSIMDu32();
		state.outExtentLayerCount = extentLayerCount;
		state.outImage = outImage.get();

		state.swizzle = {};

		state.ditherState = _NBL_NEW(std::remove_pointer<decltype(state.ditherState)>::type);
		state.scratchMemoryByteSize = blitImageFilter.getRequiredScratchByteSize(&state);
		state.scratchMemory = reinterpret_cast<uint8_t*>(_NBL_ALIGNED_MALLOC(state.scratchMemoryByteSize, 32));

		if (!blitImageFilter.execute(std::execution::par_unseq, &state))
			os::Printer::log("Something went wrong while performing discrete convolution operation!", ELL_WARNING);

		_NBL_DELETE(state.ditherState);
		_NBL_ALIGNED_FREE(state.scratchMemory);
	}

	// Print the output buffer
	{
		float* outPixel = (float*)outImage->getBuffer()->getPointer();
		for (uint32_t y = 0; y < inImageDims[1]; ++y)
		{
			for (uint32_t x = 0; x < inImageDims[0]; ++x)
			{
				std::cout << *outPixel++ << "\t";
			}
			std::cout << std::endl;
		}
	}

	return 0;
}