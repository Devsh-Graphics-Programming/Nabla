// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#define _NBL_STATIC_LIB_
#include <nabla.h>

using namespace nbl;
using namespace nbl::asset;
using namespace nbl::core;
using namespace nbl::video;

using ScaledBoxKernel = asset::CScaledImageFilterKernel<CBoxImageFilterKernel>;
using BlitFilter = asset::CBlitImageFilter<false, false, asset::VoidSwizzle, asset::IdentityDither, ScaledBoxKernel, ScaledBoxKernel, ScaledBoxKernel>;

core::smart_refctd_ptr<ICPUImage> createCPUImage(const std::array<uint32_t, 2>& dims)
{
	IImage::SCreationParams imageParams = {};
	imageParams.flags = static_cast<IImage::E_CREATE_FLAGS>(0u);
	imageParams.type = IImage::ET_2D;
	imageParams.format = asset::EF_R32_SFLOAT;
	imageParams.extent = { dims[0], dims[1], 1 };
	imageParams.mipLevels = 1u;
	imageParams.arrayLayers = 1u;
	imageParams.samples = asset::ICPUImage::ESCF_1_BIT;

	auto imageRegions = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<asset::IImage::SBufferCopy>>(1ull);
	auto& region = (*imageRegions)[0];
	region.bufferImageHeight = 0u;
	region.bufferOffset = 0ull;
	region.bufferRowLength = dims[0];
	region.imageExtent = { dims[0], dims[1], 1u };
	region.imageOffset = { 0u, 0u, 0u };
	region.imageSubresource.baseArrayLayer = 0u;
	region.imageSubresource.layerCount = 1u;
	region.imageSubresource.mipLevel = 0;

	size_t bufferSize = asset::getTexelOrBlockBytesize(imageParams.format) * region.imageExtent.width * region.imageExtent.height;
	auto imageBuffer = core::make_smart_refctd_ptr<asset::ICPUBuffer>(bufferSize);
	core::smart_refctd_ptr<ICPUImage> image = ICPUImage::create(std::move(imageParams));
	image->setBufferAndRegions(core::smart_refctd_ptr(imageBuffer), imageRegions);

	return image;
}

void blit(core::smart_refctd_ptr<ICPUImage> inImage, core::smart_refctd_ptr<ICPUImage> outImage, bool useLUT)
{
	const core::vectorSIMDf scaleX(3.f, 1.f, 1.f, 1.f);
	const core::vectorSIMDf scaleY(1.f, 3.f, 1.f, 1.f);
	const core::vectorSIMDf scaleZ(1.f, 1.f, 1.f, 1.f);

	auto kernelX = ScaledBoxKernel(scaleX, CBoxImageFilterKernel()); // [-3/2, 3/2]
	auto kernelY = ScaledBoxKernel(scaleY, CBoxImageFilterKernel()); // [-3/2, 3/2]
	auto kernelZ = ScaledBoxKernel(scaleZ, CBoxImageFilterKernel()); // [-1/2, 1/2]

	BlitFilter::state_type blitFilterState(std::move(kernelX), std::move(kernelY), std::move(kernelZ));
	blitFilterState.inOffsetBaseLayer = core::vectorSIMDu32();
	blitFilterState.inExtentLayerCount = core::vectorSIMDu32(0u, 0u, 0u, inImage->getCreationParameters().arrayLayers) + inImage->getMipSize();
	blitFilterState.inImage = inImage.get();

	blitFilterState.outOffsetBaseLayer = core::vectorSIMDu32();
	blitFilterState.outExtentLayerCount = core::vectorSIMDu32(0u, 0u, 0u, outImage->getCreationParameters().arrayLayers) + outImage->getMipSize();
	blitFilterState.outImage = outImage.get();

	blitFilterState.axisWraps[0] = ISampler::ETC_CLAMP_TO_EDGE;
	blitFilterState.axisWraps[1] = ISampler::ETC_CLAMP_TO_EDGE;
	blitFilterState.axisWraps[2] = ISampler::ETC_CLAMP_TO_EDGE;
	blitFilterState.borderColor = ISampler::E_TEXTURE_BORDER_COLOR::ETBC_FLOAT_OPAQUE_WHITE;

	blitFilterState.enableLUTUsage = useLUT;

	blitFilterState.scratchMemoryByteSize = BlitFilter::getRequiredScratchByteSize(&blitFilterState);
	blitFilterState.scratchMemory = reinterpret_cast<uint8_t*>(_NBL_ALIGNED_MALLOC(blitFilterState.scratchMemoryByteSize, 32));

	if (!BlitFilter::execute(&blitFilterState))
		os::Printer::log("Blit filter just shit the bed", ELL_WARNING);

	_NBL_ALIGNED_FREE(blitFilterState.scratchMemory);
}

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

	const std::array<uint32_t, 2> inImageDims = { 800u, 5u };
	const std::array<uint32_t, 2> outImageDims = { 16u, 8u };

	core::smart_refctd_ptr<ICPUImage> inImage = createCPUImage(inImageDims);

	std::random_device rd;
	std::mt19937 mt(rd());
	const float anyRandomLowerBound = 0.f;
	const float anyRandomUpperBound = 1.5f;
	std::uniform_real_distribution<float> dist(anyRandomLowerBound, anyRandomUpperBound);

	float k = 1.f;
	float* inImagePixel = (float*)inImage->getBuffer()->getPointer();
	for (uint32_t y = 0; y < inImageDims[1]; ++y)
	{
		for (uint32_t x = 0; x < inImageDims[0]; ++x)
		{
			*inImagePixel++ = k++;// dist(mt);
		}
	}
	core::smart_refctd_ptr<ICPUImage> outImage_withoutLUT = createCPUImage(outImageDims);
	core::smart_refctd_ptr<ICPUImage> outImage_withLUT = createCPUImage(outImageDims);

	using ScaledBoxKernel = asset::CScaledImageFilterKernel<CBoxImageFilterKernel>;
	using BlitFilter = asset::CBlitImageFilter<false, false, asset::VoidSwizzle, asset::IdentityDither, ScaledBoxKernel, ScaledBoxKernel, ScaledBoxKernel>;

	blit(inImage, outImage_withoutLUT, false);
	blit(inImage, outImage_withLUT, true);

	// Test
	printf("Result: ");
	float* outPixel_withoutLUT = (float*)outImage_withoutLUT->getBuffer()->getPointer();
	float* outPixel_withLUT = (float*)outImage_withLUT->getBuffer()->getPointer();
	for (uint32_t y = 0; y < outImageDims[1]; ++y)
	{
		for (uint32_t x = 0; x < outImageDims[0]; ++x)
		{
			if (outPixel_withoutLUT[y * outImageDims[0] + x] != outPixel_withLUT[y * outImageDims[0] + x])
			{
				printf("Failed at (%u, %u)\n", x, y);
				printf("Without LUT: %f\n", outPixel_withoutLUT[y * outImageDims[0] + x]);
				printf("With LUT: %f\n", outPixel_withLUT[y * outImageDims[0] + x]);
				__debugbreak();
			}
		}
	}
	printf("Passed\n");

	return 0;
}