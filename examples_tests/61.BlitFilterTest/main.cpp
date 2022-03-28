// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#define _NBL_STATIC_LIB_
#include <nabla.h>

#include "../common/CommonAPI.h"
#include "nbl/ext/ScreenShot/ScreenShot.h"

using namespace nbl;
using namespace nbl::asset;
using namespace nbl::core;
using namespace nbl::video;

#define FATAL_LOG(x, ...) {logger->log(##x, system::ILogger::ELL_ERROR, __VA_ARGS__); exit(-1);}

using ScaledBoxKernel = asset::CScaledImageFilterKernel<CBoxImageFilterKernel>;
using BlitFilter = asset::CBlitImageFilter<asset::VoidSwizzle, asset::IdentityDither, void, false, ScaledBoxKernel, ScaledBoxKernel, ScaledBoxKernel>;

core::smart_refctd_ptr<ICPUImage> createCPUImage(const core::vectorSIMDu32& dims, const asset::IImage::E_TYPE imageType, const asset::E_FORMAT format)
{
	IImage::SCreationParams imageParams = {};
	imageParams.flags = static_cast<IImage::E_CREATE_FLAGS>(0u);
	imageParams.type = imageType;
	imageParams.format = format;
	imageParams.extent = { dims[0], dims[1], dims[2] };
	imageParams.mipLevels = 1u;
	imageParams.arrayLayers = 1u;
	imageParams.samples = asset::ICPUImage::ESCF_1_BIT;

	auto imageRegions = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<asset::IImage::SBufferCopy>>(1ull);
	auto& region = (*imageRegions)[0];
	region.bufferImageHeight = 0u;
	region.bufferOffset = 0ull;
	region.bufferRowLength = dims[0];
	region.imageExtent = { dims[0], dims[1], dims[2] };
	region.imageOffset = { 0u, 0u, 0u };
	region.imageSubresource.baseArrayLayer = 0u;
	region.imageSubresource.layerCount = 1u;
	region.imageSubresource.mipLevel = 0;

	size_t bufferSize = asset::getTexelOrBlockBytesize(imageParams.format) * static_cast<size_t>(region.imageExtent.width) * region.imageExtent.height * region.imageExtent.depth;
	auto imageBuffer = core::make_smart_refctd_ptr<asset::ICPUBuffer>(bufferSize);
	core::smart_refctd_ptr<ICPUImage> image = ICPUImage::create(std::move(imageParams));
	image->setBufferAndRegions(core::smart_refctd_ptr(imageBuffer), imageRegions);

	return image;
}

static inline asset::IImageView<asset::ICPUImage>::E_TYPE getImageViewTypeFromImageType_CPU(const asset::IImage::E_TYPE type)
{
	switch (type)
	{
	case asset::IImage::ET_1D:
		return asset::ICPUImageView::ET_1D;
	case asset::IImage::ET_2D:
		return asset::ICPUImageView::ET_2D;
	case asset::IImage::ET_3D:
		return asset::ICPUImageView::ET_3D;
	default:
		__debugbreak();
		return static_cast<asset::IImageView<asset::ICPUImage>::E_TYPE>(0u);
	}
}

static inline video::IGPUImageView::E_TYPE getImageViewTypeFromImageType_GPU(const video::IGPUImage::E_TYPE type)
{
	switch (type)
	{
	case video::IGPUImage::ET_1D:
		return video::IGPUImageView::ET_1D;
	case video::IGPUImage::ET_2D:
		return video::IGPUImageView::ET_2D;
	case video::IGPUImage::ET_3D:
		return video::IGPUImageView::ET_3D;
	default:
		__debugbreak();
		return static_cast<video::IGPUImageView::E_TYPE>(0u);
	}
}

float computeAlphaCoverage(const double referenceAlpha, asset::ICPUImage* image)
{
	const uint32_t mipLevel = 0u;

	uint32_t alphaTestPassCount = 0u;
	const auto& extent = image->getCreationParameters().extent;

	for (uint32_t z = 0u; z < extent.depth; ++z)
	{
		for (uint32_t y = 0u; y < extent.height; ++y)
		{
			for (uint32_t x = 0u; x < extent.width; ++x)
			{
				const core::vectorSIMDu32 texCoord(x, y, z);
				core::vectorSIMDu32 dummy;
				const void* encodedPixel = image->getTexelBlockData(mipLevel, texCoord, dummy);

				double decodedPixel[4];
				asset::decodePixelsRuntime(image->getCreationParameters().format, &encodedPixel, decodedPixel, dummy.x, dummy.y);

				if (decodedPixel[3] > referenceAlpha)
					++alphaTestPassCount;
			}
		}
	}

	const float alphaCoverage = float(alphaTestPassCount) / float(extent.width * extent.height * extent.depth);
	return alphaCoverage;
};

class CBlitFilter
{
private:
	struct alignas(16) vec3_aligned
	{
		float x, y, z;
	};

	struct alignas(16) uvec3_aligned
	{
		uint32_t x, y, z;
	};

public:
	static constexpr uint32_t NBL_GLSL_DEFAULT_WORKGROUP_DIM = 16u;
	static constexpr uint32_t NBL_GLSL_DEFAULT_WORKGROUP_SIZE = 256u;
	static constexpr uint32_t NBL_GLSL_DEFAULT_BIN_COUNT = 256u;

	struct alpha_test_push_constants_t
	{
		float referenceAlpha;
	};

	struct blit_push_constants_t
	{
		uvec3_aligned inDim;
		uvec3_aligned outDim;
		vec3_aligned negativeSupport;
		vec3_aligned positiveSupport;
		uvec3_aligned windowDim;
		uvec3_aligned phaseCount;
		uint32_t windowsPerWG;
		uint32_t axisCount;
	};

	struct normalization_push_constants_t
	{
		uvec3_aligned outDim;
		uint32_t inPixelCount;
		float referenceAlpha;
	};

	struct dispatch_info_t
	{
		uint32_t wgCount[3];
	};

	CBlitFilter(video::ILogicalDevice* logicalDevice, const uint32_t smemSize = 16*1024u) : device(logicalDevice), sharedMemorySize(smemSize)
	{
		sampler = nullptr;
		{
			video::IGPUSampler::SParams params = {};
			params.TextureWrapU = asset::ISampler::ETC_CLAMP_TO_EDGE;
			params.TextureWrapV = asset::ISampler::ETC_CLAMP_TO_EDGE;
			params.TextureWrapW = asset::ISampler::ETC_CLAMP_TO_EDGE;
			params.BorderColor = asset::ISampler::ETBC_FLOAT_OPAQUE_BLACK;
			params.MinFilter = asset::ISampler::ETF_LINEAR;
			params.MaxFilter = asset::ISampler::ETF_LINEAR;
			params.MipmapMode = asset::ISampler::ESMM_NEAREST;
			params.AnisotropicFilter = 0u;
			params.CompareEnable = 0u;
			params.CompareFunc = asset::ISampler::ECO_ALWAYS;
			sampler = logicalDevice->createGPUSampler(std::move(params));
		}

		constexpr uint32_t BLIT_DESCRIPTOR_COUNT = 4u;
		asset::E_DESCRIPTOR_TYPE types[BLIT_DESCRIPTOR_COUNT] = { asset::EDT_COMBINED_IMAGE_SAMPLER, asset::EDT_STORAGE_IMAGE, asset::EDT_UNIFORM_BUFFER, asset::EDT_STORAGE_BUFFER }; // input image, output image, cached weights, alpha histogram
		blitDSLayout = getDSLayout(BLIT_DESCRIPTOR_COUNT, types, logicalDevice, sampler);

		asset::SPushConstantRange pcRange = {};
		{
			pcRange.stageFlags = asset::IShader::ESS_COMPUTE;
			pcRange.offset = 0u;
			pcRange.size = sizeof(blit_push_constants_t);
		}

		blitPipelineLayout = logicalDevice->createGPUPipelineLayout(&pcRange, &pcRange + 1ull, core::smart_refctd_ptr(blitDSLayout));
	}

	inline core::smart_refctd_ptr<video::IGPUDescriptorSetLayout> getDefaultBlitDSLayout() const { return blitDSLayout; }
	inline core::smart_refctd_ptr<video::IGPUPipelineLayout> getDefaultBlitPipelineLayout() const { return blitPipelineLayout; }

	inline core::smart_refctd_ptr<video::IGPUDescriptorSetLayout> getDefaultAlphaTestDSLayout()
	{
		if (!alphaTestDSLayout)
		{
			constexpr uint32_t DESCRIPTOR_COUNT = 2u;
			asset::E_DESCRIPTOR_TYPE types[DESCRIPTOR_COUNT] = { asset::EDT_COMBINED_IMAGE_SAMPLER, asset::EDT_STORAGE_BUFFER }; // input image, alpha test atomic counter
			alphaTestDSLayout = getDSLayout(DESCRIPTOR_COUNT, types, device, sampler);
		}

		return alphaTestDSLayout;
	}

	inline core::smart_refctd_ptr<video::IGPUPipelineLayout> getDefaultAlphaTestPipelineLayout()
	{
		if (!alphaTestPipelineLayout)
		{
			asset::SPushConstantRange pcRange = {};
			{
				pcRange.stageFlags = asset::IShader::ESS_COMPUTE;
				pcRange.offset = 0u;
				pcRange.size = sizeof(alpha_test_push_constants_t);
			}

			alphaTestPipelineLayout = device->createGPUPipelineLayout(&pcRange, &pcRange + 1ull, core::smart_refctd_ptr(alphaTestDSLayout));
		}

		return alphaTestPipelineLayout;
	}

	core::smart_refctd_ptr<video::IGPUSpecializedShader> getDefaultAlphaTestSpecializedShader(const asset::IImage::E_TYPE imageType)
	{
		if (!alphaTestSpecShader[imageType])
		{
			auto system = device->getPhysicalDevice()->getSystem();
			system::future<core::smart_refctd_ptr<system::IFile>> future;
			const char* shaderpath = "";
			switch (imageType)
			{
			case asset::IImage::ET_1D:
				shaderpath = "../alpha_test_1d.comp";
				break;
			case asset::IImage::ET_2D:
				shaderpath = "../alpha_test_2d.comp";
				break;
			case asset::IImage::ET_3D:
				shaderpath = "../alpha_test_3d.comp";
				break;
			default:
				_NBL_DEBUG_BREAK_IF(true);
				return nullptr;
			}
			const bool status = system->createFile(future, shaderpath, static_cast<system::IFile::E_CREATE_FLAGS>(system::IFile::ECF_READ | system::IFile::ECF_MAPPABLE));
			if (!status)
				return nullptr;

			auto glslFile = future.get();
			auto buffer = core::make_smart_refctd_ptr<asset::ICPUBuffer>(glslFile->getSize());
			memcpy(buffer->getPointer(), glslFile->getMappedPointer(), glslFile->getSize());

			auto cpuShader = core::make_smart_refctd_ptr<asset::ICPUShader>(std::move(buffer), asset::IShader::buffer_contains_glsl_t{}, asset::IShader::ESS_COMPUTE, "????");

			// Todo(achal): This needs to be extended to 3D images
			auto cpuShaderOverriden = asset::IGLSLCompiler::createOverridenCopy(cpuShader.get(),
				"#define _NBL_GLSL_WORKGROUP_SIZE_X_ %d\n"
				"#define _NBL_GLSL_WORKGROUP_SIZE_Y_ %d\n"
				"#define _NBL_GLSL_WORKGROUP_SIZE_Z_ %d\n",
				NBL_GLSL_DEFAULT_WORKGROUP_DIM, NBL_GLSL_DEFAULT_WORKGROUP_DIM, NBL_GLSL_DEFAULT_WORKGROUP_DIM);

			cpuShaderOverriden->setFilePathHint(shaderpath);

			auto gpuUnspecShader = device->createGPUShader(std::move(cpuShaderOverriden));

			alphaTestSpecShader[imageType] = device->createGPUSpecializedShader(gpuUnspecShader.get(), { nullptr, nullptr, "main" });
		}

		return alphaTestSpecShader[imageType];
	}

	inline core::smart_refctd_ptr<video::IGPUComputePipeline> getDefaultAlphaTestPipeline(const asset::IImage::E_TYPE imageType)
	{
		if (!alphaTestPipeline[imageType])
			alphaTestPipeline[imageType] = device->createGPUComputePipeline(nullptr, core::smart_refctd_ptr(alphaTestPipelineLayout), core::smart_refctd_ptr(alphaTestSpecShader[imageType]));

		return alphaTestPipeline[imageType];
	}

	static void updateAlphaTestDescriptorSet(video::ILogicalDevice* logicalDevice, video::IGPUDescriptorSet* ds, core::smart_refctd_ptr<video::IGPUImageView> inImageView, core::smart_refctd_ptr<video::IGPUBuffer> alphaTestCounterBuffer)
	{
		constexpr uint32_t MAX_DESCRIPTOR_COUNT = 5u;

		const auto& bindings = ds->getLayout()->getBindings();
		const uint32_t descriptorCount = static_cast<uint32_t>(bindings.size());
		assert(descriptorCount < MAX_DESCRIPTOR_COUNT);

		video::IGPUDescriptorSet::SWriteDescriptorSet writes[MAX_DESCRIPTOR_COUNT] = {};
		video::IGPUDescriptorSet::SDescriptorInfo infos[MAX_DESCRIPTOR_COUNT] = {};

		for (uint32_t i = 0u; i < descriptorCount; ++i)
		{
			writes[i].dstSet = ds;
			writes[i].binding = i;
			writes[i].arrayElement = 0u;
			writes[i].count = 1u;
			writes[i].info = &infos[i];
			writes[i].descriptorType = bindings.begin()[i].type;
		}

		// input image
		infos[0].desc = inImageView;
		infos[0].image.imageLayout = asset::EIL_GENERAL;
		infos[0].image.sampler = nullptr;

		// alpha test counter buffer (this will be assimilated into the scratch buffer soon which will hold the space both for the histogram and this atomic counter)
		infos[1].desc = alphaTestCounterBuffer;
		infos[1].buffer.offset = 0u;
		infos[1].buffer.size = alphaTestCounterBuffer->getCachedCreationParams().declaredSize;

		logicalDevice->updateDescriptorSets(descriptorCount, writes, 0u, nullptr);
	}

	inline void buildAlphaTestParameters(const float referenceAlpha, const core::vectorSIMDu32& inImageExtent, alpha_test_push_constants_t& outPC, dispatch_info_t& outDispatchInfo)
	{
		outPC.referenceAlpha = referenceAlpha;

		const core::vectorSIMDu32 workgroupSize(NBL_GLSL_DEFAULT_WORKGROUP_DIM, NBL_GLSL_DEFAULT_WORKGROUP_DIM, NBL_GLSL_DEFAULT_WORKGROUP_DIM, 1u);
		const core::vectorSIMDu32 workgroupCount = (inImageExtent + workgroupSize - core::vectorSIMDu32(1u, 1u, 1u, 1u)) / workgroupSize;
		outDispatchInfo.wgCount[0] = workgroupCount.x;
		outDispatchInfo.wgCount[1] = workgroupCount.y;
		outDispatchInfo.wgCount[2] = workgroupCount.z;
	}

	inline core::smart_refctd_ptr<video::IGPUDescriptorSetLayout> getDefaultNormalizationDSLayout()
	{
		if (!normalizationDSLayout)
		{
			constexpr uint32_t DESCRIPTOR_COUNT = 3u;
			asset::E_DESCRIPTOR_TYPE types[DESCRIPTOR_COUNT] = { asset::EDT_STORAGE_IMAGE, asset::EDT_STORAGE_BUFFER, asset::EDT_STORAGE_BUFFER }; // image to normalize, alpha histogram, alpha test atomic counter
			normalizationDSLayout = getDSLayout(DESCRIPTOR_COUNT, types, device, sampler);
		}

		return normalizationDSLayout;
	}

	inline core::smart_refctd_ptr<video::IGPUPipelineLayout> getDefaultNormalizationPipelineLayout()
	{
		if (!normalizationPipelineLayout)
		{
			asset::SPushConstantRange pcRange = {};
			{
				pcRange.stageFlags = asset::IShader::ESS_COMPUTE;
				pcRange.offset = 0u;
				pcRange.size = sizeof(normalization_push_constants_t);
			}

			normalizationPipelineLayout = device->createGPUPipelineLayout(&pcRange, &pcRange + 1ull, core::smart_refctd_ptr(normalizationDSLayout));
		}

		return normalizationPipelineLayout;
	}

	core::smart_refctd_ptr<video::IGPUSpecializedShader> getDefaultNormalizationSpecializedShader(const asset::IImage::E_TYPE imageType)
	{
		if (!normalizationSpecShader[imageType])
		{
			auto system = device->getPhysicalDevice()->getSystem();
			system::future<core::smart_refctd_ptr<system::IFile>> future;
			const char* shaderpath = "";
			switch (imageType)
			{
			case asset::IImage::ET_1D:
				shaderpath = "../normalization_1d.comp";
				break;
			case asset::IImage::ET_2D:
				shaderpath = "../normalization_2d.comp";
				break;
			case asset::IImage::ET_3D:
				shaderpath = "../normalization_3d.comp";
				break;
			default:
				_NBL_DEBUG_BREAK_IF(true);
				return nullptr;
			}
			const bool status = system->createFile(future, shaderpath, static_cast<system::IFile::E_CREATE_FLAGS>(system::IFile::ECF_READ | system::IFile::ECF_MAPPABLE));
			if (!status)
				return nullptr;

			auto glslFile = future.get();
			auto buffer = core::make_smart_refctd_ptr<asset::ICPUBuffer>(glslFile->getSize());
			memcpy(buffer->getPointer(), glslFile->getMappedPointer(), glslFile->getSize());

			auto cpuShader = core::make_smart_refctd_ptr<asset::ICPUShader>(std::move(buffer), asset::IShader::buffer_contains_glsl_t{}, asset::IShader::ESS_COMPUTE, "????");

			auto cpuShaderOverriden = asset::IGLSLCompiler::createOverridenCopy(cpuShader.get(),
				"#define _NBL_GLSL_WORKGROUP_SIZE_X_ %d\n"
				"#define _NBL_GLSL_WORKGROUP_SIZE_Y_ %d\n"
				"#define _NBL_GLSL_WORKGROUP_SIZE_Z_ %d\n"
				"#define _NBL_GLSL_BIN_COUNT_ %d\n",
				// Todo(achal): Not make DEFAULT_WG_SIZE_Z = 1
				NBL_GLSL_DEFAULT_WORKGROUP_DIM, NBL_GLSL_DEFAULT_WORKGROUP_DIM, 1u, NBL_GLSL_DEFAULT_BIN_COUNT);

			cpuShaderOverriden->setFilePathHint(shaderpath);

			auto gpuUnspecShader = device->createGPUShader(std::move(cpuShaderOverriden));

			normalizationSpecShader[imageType] = device->createGPUSpecializedShader(gpuUnspecShader.get(), { nullptr, nullptr, "main" });
		}

		return normalizationSpecShader[imageType];
	}

	inline core::smart_refctd_ptr<video::IGPUComputePipeline> getDefaultNormalizationPipeline(const asset::IImage::E_TYPE imageType)
	{
		if (!normalizationPipeline[imageType])
			normalizationPipeline[imageType] = device->createGPUComputePipeline(nullptr, core::smart_refctd_ptr(normalizationPipelineLayout), core::smart_refctd_ptr(normalizationSpecShader[imageType]));

		return normalizationPipeline[imageType];
	}

	static void updateNormalizationDescriptorSet(video::ILogicalDevice* logicalDevice, video::IGPUDescriptorSet* ds, core::smart_refctd_ptr<video::IGPUImageView> outImageView, core::smart_refctd_ptr<video::IGPUBuffer> alphaHistogramBuffer, core::smart_refctd_ptr<video::IGPUBuffer> alphaTestCounterBuffer)
	{
		constexpr uint32_t MAX_DESCRIPTOR_COUNT = 5u;

		const auto& bindings = ds->getLayout()->getBindings();
		const uint32_t descriptorCount = static_cast<uint32_t>(bindings.size());
		assert(descriptorCount < MAX_DESCRIPTOR_COUNT);

		video::IGPUDescriptorSet::SWriteDescriptorSet writes[MAX_DESCRIPTOR_COUNT] = {};
		video::IGPUDescriptorSet::SDescriptorInfo infos[MAX_DESCRIPTOR_COUNT] = {};

		for (uint32_t i = 0u; i < descriptorCount; ++i)
		{
			writes[i].dstSet = ds;
			writes[i].binding = i;
			writes[i].arrayElement = 0u;
			writes[i].count = 1u;
			writes[i].info = &infos[i];
			writes[i].descriptorType = bindings.begin()[i].type;
		}

		// input image
		infos[0].desc = outImageView;
		infos[0].image.imageLayout = asset::EIL_GENERAL;
		infos[0].image.sampler = nullptr;

		// alpha histogram buffer
		infos[1].desc = alphaHistogramBuffer;
		infos[1].buffer.offset = 0u;
		infos[1].buffer.size = alphaHistogramBuffer->getCachedCreationParams().declaredSize;

		// alpha test counter buffer
		infos[2].desc = alphaTestCounterBuffer;
		infos[2].buffer.offset = 0u;
		infos[2].buffer.size = alphaTestCounterBuffer->getCachedCreationParams().declaredSize;

		logicalDevice->updateDescriptorSets(descriptorCount, writes, 0u, nullptr);
	}

	void buildNormalizationParameters(const core::vectorSIMDu32& inImageExtent, const core::vectorSIMDu32& outImageExtent, const float referenceAlpha, normalization_push_constants_t& outPC, dispatch_info_t& outDispatchInfo)
	{
		outPC.outDim = { outImageExtent.x, outImageExtent.y, outImageExtent.z };
		outPC.inPixelCount = inImageExtent.x * inImageExtent.y * inImageExtent.z;
		outPC.referenceAlpha = static_cast<float>(referenceAlpha);

		const core::vectorSIMDu32 workgroupSize(NBL_GLSL_DEFAULT_WORKGROUP_DIM, NBL_GLSL_DEFAULT_WORKGROUP_DIM, NBL_GLSL_DEFAULT_WORKGROUP_DIM, 1u);
		const core::vectorSIMDu32 workgroupCount = (outImageExtent + workgroupSize - core::vectorSIMDu32(1u, 1u, 1u, 1u)) / workgroupSize;

		outDispatchInfo.wgCount[0] = workgroupCount.x;
		outDispatchInfo.wgCount[1] = workgroupCount.y;
		outDispatchInfo.wgCount[2] = workgroupCount.z;
	}

	core::smart_refctd_ptr<video::IGPUSpecializedShader> createBlitSpecializedShader(const asset::E_FORMAT inFormat, const asset::E_FORMAT outFormat, const asset::IImage::E_TYPE inImageType,
		const core::vectorSIMDu32& inExtent, const core::vectorSIMDu32& outExtent)
	{
		const uint32_t inChannelCount = asset::getFormatChannelCount(inFormat);
		const uint32_t outChannelCount = asset::getFormatChannelCount(outFormat);
		assert(outChannelCount <= inChannelCount);

		// fail if the window cannot be preloaded into shared memory
		core::vectorSIMDf scale = static_cast<core::vectorSIMDf>(inExtent).preciseDivision(static_cast<core::vectorSIMDf>(outExtent));

		const core::vectorSIMDu32 windowDim = static_cast<core::vectorSIMDu32>(core::ceil(scale));
		// It is important not to use asset::getTexelOrBlockBytesize here.
		// Even though input pixels are stored in the shared memory we use outChannelCount here because only outChannelCount channels of
		// the input image actually need blitting and hence only those channels are stored in shared memory.
		const size_t windowSize = static_cast<size_t>(windowDim.x) * windowDim.y * windowDim.z * outChannelCount * sizeof(float);
		if (windowSize > sharedMemorySize)
			return nullptr;

		auto system = device->getPhysicalDevice()->getSystem();
		system::future<core::smart_refctd_ptr<system::IFile>> future;
		const char* shaderpath = "../default_compute_blit.comp";
		const bool status = system->createFile(future, shaderpath, static_cast<system::IFile::E_CREATE_FLAGS>(system::IFile::ECF_READ | system::IFile::ECF_MAPPABLE));
		if (!status)
			return nullptr;

		auto glslFile = future.get();
		auto buffer = core::make_smart_refctd_ptr<asset::ICPUBuffer>(glslFile->getSize());
		memcpy(buffer->getPointer(), glslFile->getMappedPointer(), glslFile->getSize());

		auto cpuShader = core::make_smart_refctd_ptr<asset::ICPUShader>(std::move(buffer), asset::IShader::buffer_contains_glsl_t{}, asset::IShader::ESS_COMPUTE, "????");

		// inFormat should support SAMPLED_BIT format feature

		if (!isOutImageFormatAllowed(outFormat))
			return nullptr;

		const asset::E_FORMAT outImageViewFormat = getOutImageViewFormat(outFormat);

		const char* outFormatGLSLString = getGLSLFormatStringFromFormat(outImageViewFormat);

		const uint32_t smemFloatCount = sharedMemorySize / (sizeof(float) * outChannelCount);

		auto cpuShaderOverriden = asset::IGLSLCompiler::createOverridenCopy(cpuShader.get(),
			"#define _NBL_GLSL_WORKGROUP_SIZE_ %d\n"
			"#define _NBL_GLSL_BLIT_OUT_CHANNEL_COUNT_ %d\n"
			"#define _NBL_GLSL_BLIT_DIM_COUNT_ %d\n"
			"#define _NBL_GLSL_BLIT_OUT_IMAGE_FORMAT_ %s\n"
			"#define _NBL_GLSL_BLIT_SMEM_FLOAT_COUNT_ %d\n",
			NBL_GLSL_DEFAULT_WORKGROUP_SIZE,
			outChannelCount,
			static_cast<uint32_t>(inImageType + 1u),
			outFormatGLSLString,
			smemFloatCount);

		cpuShaderOverriden->setFilePathHint(shaderpath);

		auto gpuUnspecShader = device->createGPUShader(std::move(cpuShaderOverriden));

		auto specShader = device->createGPUSpecializedShader(gpuUnspecShader.get(), { nullptr, nullptr, "main" });

		return specShader;
	}

	inline core::smart_refctd_ptr<video::IGPUComputePipeline> createBlitPipeline(core::smart_refctd_ptr<video::IGPUSpecializedShader>&& specShader)
	{
		auto blitPipeline = device->createGPUComputePipeline(nullptr, core::smart_refctd_ptr(blitPipelineLayout), std::move(specShader));
		return blitPipeline;
	}

	static void updateBlitDescriptorSet(video::ILogicalDevice* logicalDevice, video::IGPUDescriptorSet* ds, core::smart_refctd_ptr<video::IGPUImageView> inImageView, core::smart_refctd_ptr<video::IGPUImageView> outImageView, core::smart_refctd_ptr<video::IGPUBuffer> phaseSupportLUT, core::smart_refctd_ptr<video::IGPUBuffer> alphaHistogramBuffer = nullptr)
	{
		constexpr uint32_t MAX_DESCRIPTOR_COUNT = 5u;

		const auto& bindings = ds->getLayout()->getBindings();
		const uint32_t bindingCount = static_cast<uint32_t>(bindings.size());
		const uint32_t descriptorCount = alphaHistogramBuffer ? bindingCount : bindingCount - 1u;
		assert(descriptorCount < MAX_DESCRIPTOR_COUNT);

		video::IGPUDescriptorSet::SWriteDescriptorSet writes[MAX_DESCRIPTOR_COUNT] = {};
		video::IGPUDescriptorSet::SDescriptorInfo infos[MAX_DESCRIPTOR_COUNT] = {};

		for (uint32_t i = 0u; i < descriptorCount; ++i)
		{
			writes[i].dstSet = ds;
			writes[i].binding = i;
			writes[i].arrayElement = 0u;
			writes[i].count = 1u;
			writes[i].info = &infos[i];
			writes[i].descriptorType = bindings.begin()[i].type;
		}

		// input image
		infos[0].desc = inImageView;
		infos[0].image.imageLayout = asset::EIL_GENERAL; // Todo(achal): Make it not GENERAL, this is a sampled image
		infos[0].image.sampler = nullptr;

		// output image
		infos[1].desc = outImageView;
		infos[1].image.imageLayout = asset::EIL_GENERAL;
		infos[1].image.sampler = nullptr;

		// phase support LUT (cached weights)
		infos[2].desc = phaseSupportLUT;
		infos[2].buffer.offset = 0ull;
		infos[2].buffer.size = phaseSupportLUT->getCachedCreationParams().declaredSize;

		if (alphaHistogramBuffer)
		{
			infos[3].desc = alphaHistogramBuffer;
			infos[3].buffer.offset = 0ull;
			infos[3].buffer.size = alphaHistogramBuffer->getCachedCreationParams().declaredSize;
		}

		logicalDevice->updateDescriptorSets(descriptorCount, writes, 0u, nullptr);
	}

	inline void buildBlitParameters(const core::vectorSIMDu32& inImageExtent, const core::vectorSIMDu32& outImageExtent, const asset::IImage::E_TYPE inImageType, const asset::E_FORMAT inImageFormat, blit_push_constants_t& outPC, dispatch_info_t& outDispatchInfo)
	{
		outPC.inDim.x = inImageExtent.x; outPC.inDim.y = inImageExtent.y; outPC.inDim.z = inImageExtent.z;
		outPC.outDim.x = outImageExtent.x; outPC.outDim.y = outImageExtent.y; outPC.outDim.z = outImageExtent.z;

		core::vectorSIMDf scale = static_cast<core::vectorSIMDf>(inImageExtent).preciseDivision(static_cast<core::vectorSIMDf>(outImageExtent));
		
		const core::vectorSIMDf negativeSupport = core::vectorSIMDf(-0.5f, -0.5f, -0.5f) * scale;
		const core::vectorSIMDf positiveSupport = core::vectorSIMDf(0.5f, 0.5f, 0.5f) * scale;

		outPC.negativeSupport.x = negativeSupport.x; outPC.negativeSupport.y = negativeSupport.y; outPC.negativeSupport.z = negativeSupport.z;
		outPC.positiveSupport.x = positiveSupport.x; outPC.positiveSupport.y = positiveSupport.y; outPC.positiveSupport.z = positiveSupport.z;

		const core::vectorSIMDu32 windowDim = static_cast<core::vectorSIMDu32>(core::ceil(scale));
		outPC.windowDim.x = windowDim.x; outPC.windowDim.y = windowDim.y; outPC.windowDim.z = windowDim.z;

		const core::vectorSIMDu32 phaseCount = BlitFilter::getPhaseCount(inImageExtent, outImageExtent, inImageType);
		outPC.phaseCount.x = phaseCount.x; outPC.phaseCount.y = phaseCount.y; outPC.phaseCount.z = phaseCount.z;

		const uint32_t windowPixelCount = outPC.windowDim.x * outPC.windowDim.y * outPC.windowDim.z;
		const uint32_t smemPerWindow = windowPixelCount * (asset::getFormatChannelCount(inImageFormat) * sizeof(float));
		outPC.windowsPerWG = sharedMemorySize / smemPerWindow;

		outPC.axisCount = static_cast<uint32_t>(inImageType) + 1u;

		const uint32_t totalWindowCount = outPC.outDim.x * outPC.outDim.y * outPC.outDim.z;
		const uint32_t wgCount = (totalWindowCount + outPC.windowsPerWG - 1) / outPC.windowsPerWG;

		outDispatchInfo.wgCount[0] = wgCount;
		outDispatchInfo.wgCount[1] = 1u;
		outDispatchInfo.wgCount[2] = 1u;
	}

	template <typename push_constants_t>
	inline void dispatchHelper(video::IGPUCommandBuffer* cmdbuf, const video::IGPUPipelineLayout* pipelineLayout, const push_constants_t& pushConstants, const dispatch_info_t& dispatchInfo)
	{
		cmdbuf->pushConstants(pipelineLayout, asset::IShader::ESS_COMPUTE, 0u, sizeof(push_constants_t), &pushConstants);
		cmdbuf->dispatch(dispatchInfo.wgCount[0], dispatchInfo.wgCount[1], dispatchInfo.wgCount[2]);
	}

	inline void blit(
		video::IGPUCommandBuffer* cmdbuf,
		const BlitFilter::CState::E_ALPHA_SEMANTIC alphaSemantic,
		video::IGPUDescriptorSet* alphaTestDS,
		video::IGPUDescriptorSet* blitDS,
		video::IGPUComputePipeline* blitPipeline,
		video::IGPUDescriptorSet* normalizationDS,
		const core::vectorSIMDu32& inImageExtent,
		const asset::IImage::E_TYPE inImageType,
		const asset::E_FORMAT inImageFormat,
		core::smart_refctd_ptr<video::IGPUImage> outImage,
		const float referenceAlpha = 0.f,
		core::smart_refctd_ptr<video::IGPUBuffer> alphaTestCounterBuffer = nullptr)
	{
		if (alphaSemantic == BlitFilter::CState::EAS_REFERENCE_OR_COVERAGE)
		{
			cmdbuf->bindDescriptorSets(asset::EPBP_COMPUTE, alphaTestPipeline[inImageType]->getLayout(), 0u, 1u, &alphaTestDS);
			cmdbuf->bindComputePipeline(alphaTestPipeline[inImageType].get());
			CBlitFilter::alpha_test_push_constants_t alphaTestPC;
			CBlitFilter::dispatch_info_t alphaTestDispatchInfo;
			buildAlphaTestParameters(referenceAlpha, inImageExtent, alphaTestPC, alphaTestDispatchInfo);
			dispatchHelper<CBlitFilter::alpha_test_push_constants_t>(cmdbuf, alphaTestPipeline[inImageType]->getLayout(), alphaTestPC, alphaTestDispatchInfo);
		}

		cmdbuf->bindDescriptorSets(asset::EPBP_COMPUTE, blitPipeline->getLayout(), 0u, 1u, &blitDS);
		cmdbuf->bindComputePipeline(blitPipeline);
		CBlitFilter::blit_push_constants_t blitPC;
		CBlitFilter::dispatch_info_t blitDispatchInfo;
		const core::vectorSIMDu32 outImageExtent(outImage->getCreationParameters().extent.width, outImage->getCreationParameters().extent.height, outImage->getCreationParameters().extent.depth, 1u);
		buildBlitParameters(inImageExtent, outImageExtent, inImageType, inImageFormat, blitPC, blitDispatchInfo);
		dispatchHelper<CBlitFilter::blit_push_constants_t>(cmdbuf, blitPipeline->getLayout(), blitPC, blitDispatchInfo);

		// After this dispatch ends and finishes writing to outImage, normalize outImage
		if (alphaSemantic == BlitFilter::CState::EAS_REFERENCE_OR_COVERAGE)
		{
			// Memory dependency to ensure the alpha test pass has finished writing to alphaTestCounterBuffer
			video::IGPUCommandBuffer::SBufferMemoryBarrier alphaTestBarrier = {};
			alphaTestBarrier.barrier.srcAccessMask = asset::EAF_SHADER_WRITE_BIT;
			alphaTestBarrier.barrier.dstAccessMask = asset::EAF_SHADER_READ_BIT;
			alphaTestBarrier.srcQueueFamilyIndex = ~0u;
			alphaTestBarrier.dstQueueFamilyIndex = ~0u;
			alphaTestBarrier.buffer = alphaTestCounterBuffer;
			alphaTestBarrier.size = alphaTestCounterBuffer->getCachedCreationParams().declaredSize;

			// Memory dependency to ensure that the previous compute pass has finished writing to the output image
			video::IGPUCommandBuffer::SImageMemoryBarrier readyForNorm = {};
			readyForNorm.barrier.srcAccessMask = asset::EAF_SHADER_WRITE_BIT;
			readyForNorm.barrier.dstAccessMask = static_cast<asset::E_ACCESS_FLAGS>(asset::EAF_SHADER_READ_BIT | asset::EAF_SHADER_WRITE_BIT);
			readyForNorm.oldLayout = asset::EIL_GENERAL;
			readyForNorm.newLayout = asset::EIL_GENERAL;
			readyForNorm.srcQueueFamilyIndex = ~0u;
			readyForNorm.dstQueueFamilyIndex = ~0u;
			readyForNorm.image = outImage;
			readyForNorm.subresourceRange.aspectMask = asset::IImage::EAF_COLOR_BIT;
			readyForNorm.subresourceRange.levelCount = 1u;
			readyForNorm.subresourceRange.layerCount = 1u;
			cmdbuf->pipelineBarrier(asset::EPSF_COMPUTE_SHADER_BIT, asset::EPSF_COMPUTE_SHADER_BIT, asset::EDF_NONE, 0u, nullptr, 1u, &alphaTestBarrier, 1u, &readyForNorm);

			cmdbuf->bindDescriptorSets(asset::EPBP_COMPUTE, normalizationPipeline[inImageType]->getLayout(), 0u, 1u, &normalizationDS);
			cmdbuf->bindComputePipeline(normalizationPipeline[inImageType].get());
			CBlitFilter::normalization_push_constants_t normPC = {};
			CBlitFilter::dispatch_info_t normDispatchInfo = {};
			buildNormalizationParameters(inImageExtent, outImageExtent, referenceAlpha, normPC, normDispatchInfo);
			dispatchHelper<CBlitFilter::normalization_push_constants_t>(cmdbuf, normalizationPipeline[inImageType]->getLayout(), normPC, normDispatchInfo);
		}
	}

	//! WARNING: This function blocks and stalls the GPU!
	void blit(
		video::IGPUQueue* computeQueue,
		const BlitFilter::CState::E_ALPHA_SEMANTIC alphaSemantic,
		video::IGPUDescriptorSet* alphaTestDS,
		video::IGPUDescriptorSet* blitDS,
		video::IGPUComputePipeline* blitPipeline,
		video::IGPUDescriptorSet* normalizationDS,
		const core::vectorSIMDu32& inImageExtent,
		const asset::IImage::E_TYPE inImageType,
		const asset::E_FORMAT inImageFormat,
		core::smart_refctd_ptr<video::IGPUImage> outImage,
		const float referenceAlpha = 0.f,
		core::smart_refctd_ptr<video::IGPUBuffer> alphaTestCounterBuffer = nullptr)
	{
		auto cmdpool = device->createCommandPool(computeQueue->getFamilyIndex(), video::IGPUCommandPool::ECF_NONE);
		core::smart_refctd_ptr<video::IGPUCommandBuffer> cmdbuf;
		device->createCommandBuffers(cmdpool.get(), video::IGPUCommandBuffer::EL_PRIMARY, 1u, &cmdbuf);

		auto fence = device->createFence(video::IGPUFence::ECF_UNSIGNALED);

		cmdbuf->begin(video::IGPUCommandBuffer::EU_ONE_TIME_SUBMIT_BIT);
		blit(cmdbuf.get(), alphaSemantic, alphaTestDS, blitDS, blitPipeline, normalizationDS, inImageExtent, inImageType, inImageFormat, outImage, referenceAlpha, alphaTestCounterBuffer);
		cmdbuf->end();

		video::IGPUQueue::SSubmitInfo submitInfo = {};
		submitInfo.commandBufferCount = 1u;
		submitInfo.commandBuffers = &cmdbuf.get();
		computeQueue->submit(1u, &submitInfo, fence.get());

		device->blockForFences(1u, &fence.get());
	}

	inline asset::E_FORMAT getOutImageViewFormat(const asset::E_FORMAT format)
	{
		// required for all GL, GLES and Vulkan
		const bool isRequiredFormat =
			(format == asset::EF_R32G32B32A32_SFLOAT) ||
			(format == asset::EF_R16G16B16A16_SFLOAT) ||
			(format == asset::EF_R32_SFLOAT) ||
			(format == asset::EF_R8G8B8A8_UNORM) ||
			(format == asset::EF_R8G8B8A8_SNORM);

		if (isRequiredFormat)
		{
			return format;
		}
		else
		{
			__debugbreak();
			return asset::EF_UNKNOWN;
		}
	}

	static inline const char* getGLSLFormatStringFromFormat(const asset::E_FORMAT format)
	{
		const char* result;
		switch (format)
		{
		case asset::EF_R32G32B32A32_SFLOAT:
			result = "rgba32f";
			break;
		case asset::EF_R16G16B16A16_SFLOAT:
			result = "rgba16f";
			break;
		case asset::EF_R32G32_SFLOAT:
			result = "rg32f";
			break;
		case asset::EF_R16G16_SFLOAT:
			result = "rg16f";
			break;
		case asset::EF_B10G11R11_UFLOAT_PACK32:
			result = "r11f_g11f_b10f";
			break;
		case asset::EF_R32_SFLOAT:
			result = "r32f";
			break;
		case asset::EF_R16_SFLOAT:
			result = "r16f";
			break;
		case asset::EF_R16G16B16A16_UNORM:
			result = "rgba16";
			break;
		case asset::EF_A2B10G10R10_UNORM_PACK32:
			result = "rgb10_a2";
			break;
		case asset::EF_R8G8B8A8_UNORM:
			result = "rgba8";
			break;
		case asset::EF_R16G16_UNORM:
			result = "rg16";
			break;
		case asset::EF_R8G8_UNORM:
			result = "rg8";
			break;
		case asset::EF_R16_UNORM:
			result = "r16";
			break;
		case asset::EF_R8_UNORM:
			result = "r8";
			break;
		case asset::EF_R16G16B16A16_SNORM:
			result = "rgba16_snorm";
			break;
		case asset::EF_R8G8B8A8_SNORM:
			result = "rgba8_snorm";
			break;
		case asset::EF_R16G16_SNORM:
			result = "rg16_snorm";
			break;
		case asset::EF_R8G8_SNORM:
			result = "rg8_snorm";
			break;
		case asset::EF_R16_SNORM:
			result = "r16_snorm";
			break;
		default:
			__debugbreak();
		}

		return result;
	}

private:
	core::smart_refctd_ptr<video::IGPUDescriptorSetLayout> alphaTestDSLayout = nullptr;
	core::smart_refctd_ptr<video::IGPUPipelineLayout> alphaTestPipelineLayout = nullptr;
	core::smart_refctd_ptr<video::IGPUSpecializedShader> alphaTestSpecShader[3] = { nullptr };
	core::smart_refctd_ptr<video::IGPUComputePipeline> alphaTestPipeline[3] = { nullptr };

	core::smart_refctd_ptr<video::IGPUDescriptorSetLayout> blitDSLayout = nullptr;
	core::smart_refctd_ptr<video::IGPUPipelineLayout> blitPipelineLayout = nullptr;

	core::smart_refctd_ptr<video::IGPUDescriptorSetLayout> normalizationDSLayout = nullptr;
	core::smart_refctd_ptr<video::IGPUPipelineLayout> normalizationPipelineLayout = nullptr;
	core::smart_refctd_ptr<video::IGPUSpecializedShader> normalizationSpecShader[3] = { nullptr };
	core::smart_refctd_ptr<video::IGPUComputePipeline> normalizationPipeline[3] = { nullptr };

	const uint32_t sharedMemorySize;
	video::ILogicalDevice* device;

	core::smart_refctd_ptr<video::IGPUSampler> sampler = nullptr;

	core::smart_refctd_ptr<video::IGPUDescriptorSetLayout> getDSLayout(const uint32_t descriptorCount, const asset::E_DESCRIPTOR_TYPE* descriptorTypes, video::ILogicalDevice* logicalDevice, core::smart_refctd_ptr<video::IGPUSampler> sampler) const
	{
		constexpr uint32_t MAX_DESCRIPTOR_COUNT = 100u;
		assert(descriptorCount < MAX_DESCRIPTOR_COUNT);

		video::IGPUDescriptorSetLayout::SBinding bindings[MAX_DESCRIPTOR_COUNT] = {};

		for (uint32_t i = 0u; i < descriptorCount; ++i)
		{
			bindings[i].binding = i;
			bindings[i].count = 1u;
			bindings[i].stageFlags = asset::IShader::ESS_COMPUTE;
			bindings[i].type = descriptorTypes[i];

			if (bindings[i].type == asset::EDT_COMBINED_IMAGE_SAMPLER)
				bindings[i].samplers = &sampler;
		}

		auto dsLayout = logicalDevice->createGPUDescriptorSetLayout(bindings, bindings + descriptorCount);
		return dsLayout;
	}

	static inline bool isOutImageFormatAllowed(const asset::E_FORMAT format)
	{
		// Floating point (and normalized integer) formats in the list: https://www.khronos.org/opengl/wiki/Image_Load_Store#Images_in_the_context
		switch (format)
		{
		case EF_R32G32B32A32_SFLOAT:
		case EF_R16G16B16A16_SFLOAT:
		case EF_R32G32_SFLOAT:
		case EF_R16G16_SFLOAT:
		case EF_B10G11R11_UFLOAT_PACK32:
		case EF_R32_SFLOAT:
		case EF_R16_SFLOAT:
		case EF_R16G16B16A16_UNORM:
		case EF_A2B10G10R10_UNORM_PACK32:
		case EF_R8G8B8A8_UNORM:
		case EF_R16G16_UNORM:
		case EF_R8G8_UNORM:
		case EF_R16_UNORM:
		case EF_R8_UNORM:
		case EF_R16G16B16A16_SNORM:
		case EF_R8G8B8A8_SNORM:
		case EF_R16G16_SNORM:
		case EF_R8G8_SNORM:
		case EF_R16_SNORM:
			return true;
		default:
			return false;
		}
	}
};

class BlitFilterTestApp : public ApplicationBase
{
	constexpr static uint32_t SC_IMG_COUNT = 3u;
	constexpr static uint64_t MAX_TIMEOUT = 99999999999999ull;
	constexpr static size_t MAX_SMEM_SIZE = 16ull * 1024ull; // it is probably a good idea to expose VkPhysicalDeviceLimits::maxComputeSharedMemorySize

public:
	void onAppInitialized_impl() override
	{
		CommonAPI::InitOutput initOutput;
		CommonAPI::InitWithNoExt(initOutput, video::EAT_VULKAN, "BlitFilterTest");

		system = std::move(initOutput.system);
		window = std::move(initOutput.window);
		windowCb = std::move(initOutput.windowCb);
		apiConnection = std::move(initOutput.apiConnection);
		surface = std::move(initOutput.surface);
		physicalDevice = std::move(initOutput.physicalDevice);
		logicalDevice = std::move(initOutput.logicalDevice);
		utilities = std::move(initOutput.utilities);
		queues = std::move(initOutput.queues);
		swapchain = std::move(initOutput.swapchain);
		commandPools = std::move(initOutput.commandPools);
		assetManager = std::move(initOutput.assetManager);
		cpu2gpuParams = std::move(initOutput.cpu2gpuParams);
		logger = std::move(initOutput.logger);
		inputSystem = std::move(initOutput.inputSystem);

		{
			logger->log("Test #1");

			const core::vectorSIMDu32 inImageDim(800u, 1u, 1u);
			const asset::IImage::E_TYPE imageType = asset::IImage::ET_1D;
			const asset::E_FORMAT imageFormat = asset::EF_R32G32B32A32_SFLOAT;
			const core::vectorSIMDu32 outImageDim(59u, 1u, 1u);
			const BlitFilter::CState::E_ALPHA_SEMANTIC alphaSemantic = BlitFilter::CState::EAS_NONE_OR_PREMULTIPLIED;
			blitTest(inImageDim, imageType, imageFormat, outImageDim, alphaSemantic);
		}

		{
			logger->log("Test #2");

			const char* pathToInputImage = "../../media/colorexr.exr";
			core::smart_refctd_ptr<asset::ICPUImage> inImage = loadImage(pathToInputImage);
			if (!inImage)
				FATAL_LOG("Failed to load the image at path %s\n", pathToInputImage);

			const auto& inExtent = inImage->getCreationParameters().extent;
			const core::vectorSIMDu32 outImageDim(inExtent.width / 3u, inExtent.height / 7u, inExtent.depth);
			const BlitFilter::CState::E_ALPHA_SEMANTIC alphaSemantic = BlitFilter::CState::EAS_NONE_OR_PREMULTIPLIED;
			blitTest(inImage, outImageDim, alphaSemantic);
		}

#if 0
		{
			printf("Test #3\n");

			const char* pathToInputImage = "alpha_test_input.exr";
			core::smart_refctd_ptr<asset::ICPUImage> inImage = loadImage(pathToInputImage);
			if (!inImage)
				FATAL_LOG("Failed to load the image at path %s\n", pathToInputImage);

			const auto& inExtent = inImage->getCreationParameters().extent;
			const core::vectorSIMDu32 outImageDim(inExtent.width / 3u, inExtent.height / 7u, inExtent.depth);
			const BlitFilter::CState::E_ALPHA_SEMANTIC alphaSemantic = BlitFilter::CState::EAS_REFERENCE_OR_COVERAGE;
			blitTest(inImage, outImageDim, alphaSemantic);
		}
#endif

		{
			logger->log("Test #4");
			const core::vectorSIMDu32 inImageDim(257u, 129u, 63u);
			const asset::IImage::E_TYPE imageType = asset::IImage::ET_3D;
			const asset::E_FORMAT imageFormat = asset::EF_R32_SFLOAT;
			const core::vectorSIMDu32 outImageDim(256u, 128u, 64u);
			const BlitFilter::CState::E_ALPHA_SEMANTIC alphaSemantic = BlitFilter::CState::EAS_NONE_OR_PREMULTIPLIED;
			blitTest(inImageDim, imageType, imageFormat, outImageDim, alphaSemantic);
		}
	}

	void onAppTerminated_impl() override
	{
		logicalDevice->waitIdle();
	}

	void workLoopBody() override
	{

	}

	bool keepRunning() override
	{
		return false;
	}

private:
	void blitTest(core::smart_refctd_ptr<asset::ICPUImage> inImage, const core::vectorSIMDu32& outImageDim, const BlitFilter::CState::E_ALPHA_SEMANTIC alphaSemantic, const float referenceAlpha = 0.f)
	{
		const core::vectorSIMDf scaleX(1.f, 1.f, 1.f, 1.f);
		const core::vectorSIMDf scaleY(1.f, 1.f, 1.f, 1.f);
		const core::vectorSIMDf scaleZ(1.f, 1.f, 1.f, 1.f);

		auto kernelX = ScaledBoxKernel(scaleX, asset::CBoxImageFilterKernel()); // (-1/2, 1/2)
		auto kernelY = ScaledBoxKernel(scaleY, asset::CBoxImageFilterKernel()); // (-1/2, 1/2)
		auto kernelZ = ScaledBoxKernel(scaleZ, asset::CBoxImageFilterKernel()); // (-1/2, 1/2)

		BlitFilter::state_type blitFilterState(std::move(kernelX), std::move(kernelY), std::move(kernelZ));

		blitFilterState.inOffsetBaseLayer = core::vectorSIMDu32();
		blitFilterState.inExtentLayerCount = core::vectorSIMDu32(0u, 0u, 0u, inImage->getCreationParameters().arrayLayers) + inImage->getMipSize();
		blitFilterState.inImage = inImage.get();

		blitFilterState.outOffsetBaseLayer = core::vectorSIMDu32();
		const uint32_t outImageLayerCount = 1u;
		blitFilterState.outExtentLayerCount = core::vectorSIMDu32(outImageDim[0], outImageDim[1], outImageDim[2], 1u);

		blitFilterState.axisWraps[0] = asset::ISampler::ETC_CLAMP_TO_EDGE;
		blitFilterState.axisWraps[1] = asset::ISampler::ETC_CLAMP_TO_EDGE;
		blitFilterState.axisWraps[2] = asset::ISampler::ETC_CLAMP_TO_EDGE;
		blitFilterState.borderColor = asset::ISampler::E_TEXTURE_BORDER_COLOR::ETBC_FLOAT_OPAQUE_WHITE;

		blitFilterState.enableLUTUsage = true;

		blitFilterState.alphaSemantic = alphaSemantic;
		blitFilterState.alphaChannel = 3u;
		blitFilterState.alphaRefValue = referenceAlpha;

		blitFilterState.scratchMemoryByteSize = BlitFilter::getRequiredScratchByteSize(&blitFilterState);
		blitFilterState.scratchMemory = reinterpret_cast<uint8_t*>(_NBL_ALIGNED_MALLOC(blitFilterState.scratchMemoryByteSize, 32));

		blitFilterState.computePhaseSupportLUT(&blitFilterState);

		// CPU
		core::smart_refctd_ptr<asset::ICPUImage> cpuOutput = nullptr;
		{

			cpuOutput = createCPUImage(outImageDim, inImage->getCreationParameters().type, inImage->getCreationParameters().format);

			blitFilterState.outImage = cpuOutput.get();

			logger->log("CPU begin..");
			if (!BlitFilter::execute(core::execution::par_unseq, &blitFilterState))
				logger->log("Failed to blit", system::ILogger::ELL_ERROR);
			logger->log("CPU end..");

			const char* writePath = "cpu_out.exr";
			{
				// create an image view to write the image to disk
				core::smart_refctd_ptr<asset::ICPUImageView> outImageView = nullptr;
				{
					ICPUImageView::SCreationParams viewParams;
					viewParams.flags = static_cast<ICPUImageView::E_CREATE_FLAGS>(0u);
					viewParams.image = cpuOutput;
					viewParams.format = viewParams.image->getCreationParameters().format;
					viewParams.viewType = getImageViewTypeFromImageType_CPU(viewParams.image->getCreationParameters().type);
					viewParams.subresourceRange.baseArrayLayer = 0u;
					viewParams.subresourceRange.layerCount = cpuOutput->getCreationParameters().arrayLayers;
					viewParams.subresourceRange.baseMipLevel = 0u;
					viewParams.subresourceRange.levelCount = cpuOutput->getCreationParameters().mipLevels;

					outImageView = ICPUImageView::create(std::move(viewParams));
				}

				asset::IAssetWriter::SAssetWriteParams wparams(outImageView.get());
				wparams.logger = logger.get();
				if (!assetManager->writeAsset(writePath, wparams))
					FATAL_LOG("Failed to write cpu image at path %s\n", writePath);
			}
		}

		core::smart_refctd_ptr<asset::ICPUImage> gpuOutput = nullptr;
		{
			const auto& inImageType = inImage->getCreationParameters().type;
			const core::vectorSIMDu32 inImageDim(inImage->getCreationParameters().extent.width, inImage->getCreationParameters().extent.height, inImage->getCreationParameters().extent.depth);

			auto outImage = createCPUImage(outImageDim, inImage->getCreationParameters().type, inImage->getCreationParameters().format);

			inImage->addImageUsageFlags(asset::ICPUImage::EUF_SAMPLED_BIT);
			outImage->addImageUsageFlags(asset::ICPUImage::EUF_STORAGE_BIT);

			core::smart_refctd_ptr<video::IGPUImage> inImageGPU = nullptr;
			core::smart_refctd_ptr<video::IGPUImage> outImageGPU = nullptr;
			core::smart_refctd_ptr<video::IGPUImageView> inImageView = nullptr;
			core::smart_refctd_ptr<video::IGPUImageView> outImageView = nullptr;

			if (!getGPUImagesAndTheirViews(inImage, outImage, &inImageGPU, &outImageGPU, &inImageView, &outImageView))
				FATAL_LOG("Failed to convert CPU images to GPU images\n");

			CBlitFilter blitFilter(logicalDevice.get());

			core::smart_refctd_ptr<video::IGPUBuffer> alphaTestCounterBuffer = nullptr;
			core::smart_refctd_ptr<video::IGPUBuffer> alphaHistogramBuffer = nullptr;
			core::smart_refctd_ptr<video::IGPUDescriptorSet> alphaTestDS = nullptr;
			core::smart_refctd_ptr<video::IGPUComputePipeline> alphaTestPipeline = nullptr;
			core::smart_refctd_ptr<video::IGPUDescriptorSet> normDS = nullptr;
			core::smart_refctd_ptr<video::IGPUComputePipeline> normPipeline = nullptr;
			if (alphaSemantic == BlitFilter::CState::EAS_REFERENCE_OR_COVERAGE)
			{
				// create alphaTestCounterBuffer
				{
					const size_t neededSize = sizeof(uint32_t);

					video::IGPUBuffer::SCreationParams creationParams = {};
					creationParams.usage = static_cast<video::IGPUBuffer::E_USAGE_FLAGS>(video::IGPUBuffer::EUF_TRANSFER_DST_BIT | video::IGPUBuffer::EUF_STORAGE_BUFFER_BIT);

					alphaTestCounterBuffer = logicalDevice->createDeviceLocalGPUBufferOnDedMem(creationParams, neededSize);

					asset::SBufferRange<video::IGPUBuffer> bufferRange = {};
					bufferRange.offset = 0ull;
					bufferRange.size = alphaTestCounterBuffer->getCachedCreationParams().declaredSize;
					bufferRange.buffer = alphaTestCounterBuffer;

					const uint32_t fillValue = 0u;
					utilities->updateBufferRangeViaStagingBuffer(queues[CommonAPI::InitOutput::EQT_COMPUTE], bufferRange, &fillValue);
				}

				// create alphaHistogramBuffer
				{
					const size_t neededSize = CBlitFilter::NBL_GLSL_DEFAULT_BIN_COUNT * sizeof(uint32_t);

					video::IGPUBuffer::SCreationParams creationParams = {};
					creationParams.usage = static_cast<video::IGPUBuffer::E_USAGE_FLAGS>(video::IGPUBuffer::EUF_TRANSFER_DST_BIT | video::IGPUBuffer::EUF_STORAGE_BUFFER_BIT);

					alphaHistogramBuffer = logicalDevice->createDeviceLocalGPUBufferOnDedMem(creationParams, neededSize);

					asset::SBufferRange<video::IGPUBuffer> bufferRange = {};
					bufferRange.offset = 0ull;
					bufferRange.size = alphaHistogramBuffer->getCachedCreationParams().declaredSize;
					bufferRange.buffer = alphaHistogramBuffer;

					core::vector<uint32_t> fillValues(CBlitFilter::NBL_GLSL_DEFAULT_BIN_COUNT, 0u);
					utilities->updateBufferRangeViaStagingBuffer(queues[CommonAPI::InitOutput::EQT_COMPUTE], bufferRange, fillValues.data());
				}

				const auto& alphaTestDSLayout = blitFilter.getDefaultAlphaTestDSLayout();
				const auto& alphaTestCompShader = blitFilter.getDefaultAlphaTestSpecializedShader(inImageType);
				const auto& alphaTestPipelineLayout = blitFilter.getDefaultAlphaTestPipelineLayout();
				alphaTestPipeline = blitFilter.getDefaultAlphaTestPipeline(inImageType);

				const uint32_t alphaTestDSCount = 1u;
				auto alphaTestDescriptorPool = logicalDevice->createDescriptorPoolForDSLayouts(video::IDescriptorPool::ECF_NONE, &alphaTestDSLayout.get(), &alphaTestDSLayout.get() + 1ull, &alphaTestDSCount);

				alphaTestDS = logicalDevice->createGPUDescriptorSet(alphaTestDescriptorPool.get(), core::smart_refctd_ptr(alphaTestDSLayout));
				CBlitFilter::updateAlphaTestDescriptorSet(logicalDevice.get(), alphaTestDS.get(), inImageView, alphaTestCounterBuffer);

				const auto& normDSLayout = blitFilter.getDefaultNormalizationDSLayout();
				const auto& normCompShader = blitFilter.getDefaultNormalizationSpecializedShader(inImageType);
				const auto& normPipelineLayout = blitFilter.getDefaultNormalizationPipelineLayout();
				normPipeline = blitFilter.getDefaultNormalizationPipeline(inImageType);

				const uint32_t normDSCount = 1u;
				auto normDescriptorPool = logicalDevice->createDescriptorPoolForDSLayouts(video::IDescriptorPool::ECF_NONE, &normDSLayout.get(), &normDSLayout.get() + 1ull, &normDSCount);

				normDS = logicalDevice->createGPUDescriptorSet(normDescriptorPool.get(), core::smart_refctd_ptr(normDSLayout));
				CBlitFilter::updateNormalizationDescriptorSet(logicalDevice.get(), normDS.get(), outImageView, alphaHistogramBuffer, alphaTestCounterBuffer);
			}

			const core::vectorSIMDu32 inExtent(inImage->getCreationParameters().extent.width, inImage->getCreationParameters().extent.height, inImage->getCreationParameters().extent.depth);
			const core::vectorSIMDu32 outExtent(outImageDim[0], outImageDim[1], outImageDim[2]);
			core::vectorSIMDf scale = static_cast<core::vectorSIMDf>(inExtent).preciseDivision(static_cast<core::vectorSIMDf>(outExtent));

			// Cannot use kernelX/Y/Z.getWindowSize() here because they haven't been scaled (yet) for upscaling/downscaling
			const core::vectorSIMDu32 windowDim = static_cast<core::vectorSIMDu32>(core::ceil(scale));

			const core::vectorSIMDu32 phaseCount = BlitFilter::getPhaseCount(inExtent, outExtent, inImage->getCreationParameters().type);
			core::smart_refctd_ptr<video::IGPUBuffer> phaseSupportLUT = nullptr;
			{
				BlitFilter::value_type* lut = reinterpret_cast<BlitFilter::value_type*>(blitFilterState.scratchMemory + BlitFilter::getPhaseSupportLUTByteOffset(&blitFilterState));

				const size_t lutSize = (static_cast<size_t>(phaseCount.x) * windowDim.x + static_cast<size_t>(phaseCount.y) * windowDim.y + static_cast<size_t>(phaseCount.z) * windowDim.z) * sizeof(float) * 4ull;

				// lut has the LUT in doubles, I want it in floats
				// Todo(achal): Probably need to pack them as half floats? But they are NOT different for each channel??
				// If we're under std140 layout, wouldn't it be better just make a static array of vec4 inside the uniform block
				// since a static array of floats of the same length would take up the same amount of space?
				core::vector<float> lutInFloats(lutSize / sizeof(float));
				for (uint32_t i = 0u; i < lutInFloats.size() / 4; ++i)
				{
					// losing precision here
					lutInFloats[4 * i + 0] = static_cast<float>(lut[i]);
					lutInFloats[4 * i + 1] = static_cast<float>(lut[i]);
					lutInFloats[4 * i + 2] = static_cast<float>(lut[i]);
					lutInFloats[4 * i + 3] = static_cast<float>(lut[i]);
				}

				video::IGPUBuffer::SCreationParams uboCreationParams = {};
				uboCreationParams.usage = static_cast<video::IGPUBuffer::E_USAGE_FLAGS>(video::IGPUBuffer::EUF_UNIFORM_BUFFER_BIT | video::IGPUBuffer::EUF_TRANSFER_DST_BIT);
				phaseSupportLUT = logicalDevice->createDeviceLocalGPUBufferOnDedMem(uboCreationParams, lutSize);

				// fill it up with data
				asset::SBufferRange<video::IGPUBuffer> bufferRange = {};
				bufferRange.offset = 0ull;
				bufferRange.size = lutSize;
				bufferRange.buffer = phaseSupportLUT;
				utilities->updateBufferRangeViaStagingBuffer(queues[CommonAPI::InitOutput::EQT_COMPUTE], bufferRange, lutInFloats.data());
			}

			const auto& inImageFormat = inImage->getCreationParameters().format;

			const auto& blitDSLayout = blitFilter.getDefaultBlitDSLayout();
			const auto& blitShader = blitFilter.createBlitSpecializedShader(inImageFormat, outImageGPU->getCreationParameters().format, inImageType, inImageDim, outImageDim);
			const auto& blitPipelineLayout = blitFilter.getDefaultBlitPipelineLayout();
			const auto& blitPipeline = blitFilter.createBlitPipeline(core::smart_refctd_ptr(blitShader));

			const uint32_t blitDSCount = 1u;
			auto blitDescriptorPool = logicalDevice->createDescriptorPoolForDSLayouts(video::IDescriptorPool::ECF_NONE, &blitDSLayout.get(), &blitDSLayout.get() + 1ull, &blitDSCount);

			auto blitDS = logicalDevice->createGPUDescriptorSet(blitDescriptorPool.get(), core::smart_refctd_ptr(blitDSLayout));
			CBlitFilter::updateBlitDescriptorSet(logicalDevice.get(), blitDS.get(), inImageView, outImageView, phaseSupportLUT);

			logger->log("GPU begin..");
			blitFilter.blit(queues[CommonAPI::InitOutput::EQT_COMPUTE], alphaSemantic, alphaTestDS.get(), blitDS.get(), blitPipeline.get(), normDS.get(), inExtent, inImageType, inImageFormat, outImageGPU, referenceAlpha, alphaTestCounterBuffer);
			logger->log("GPU end..");

			auto outCPUImageView = ext::ScreenShot::createScreenShot(
				logicalDevice.get(),
				queues[CommonAPI::InitOutput::EQT_COMPUTE],
				nullptr,
				outImageView.get(),
				static_cast<asset::E_ACCESS_FLAGS>(0u),
				asset::EIL_GENERAL);

			gpuOutput = outCPUImageView->getCreationParameters().image;

			const char* writePath = "gpu_out.exr";
			asset::IAssetWriter::SAssetWriteParams writeParams(outCPUImageView.get());
			if (!assetManager->writeAsset(writePath, writeParams))
				FATAL_LOG("Failed to write image at path %s\n", writePath);
		}

		// compute RMSE
		{
			const auto& extent0 = cpuOutput->getCreationParameters().extent;
			const auto& extent1 = gpuOutput->getCreationParameters().extent;
			const auto& format0 = cpuOutput->getCreationParameters().format;
			const auto& format1 = gpuOutput->getCreationParameters().format;
			assert(extent0.width == extent1.width && extent0.height == extent1.height && extent0.depth == extent1.depth);
			assert(format0 == format1);

			const uint32_t mipLevel = 0u;
			const uint32_t channelCount = asset::getFormatChannelCount(format0);

			double sqErr = 0.0;
			for (uint32_t z = 0u; z < extent0.depth; ++z)
			{
				for (uint32_t y = 0u; y < extent0.height; ++y)
				{
					for (uint32_t x = 0u; x < extent0.width; ++x)
					{
						const core::vectorSIMDu32 texCoord(x, y, z);

						auto decodePixel = [](asset::ICPUImage* image, const core::vectorSIMDu32& texCoord, double* decodedPixel)
						{
							core::vectorSIMDu32 dummy;
							const void* encodedPixel = image->getTexelBlockData(mipLevel, texCoord, dummy);
							asset::decodePixelsRuntime(image->getCreationParameters().format, &encodedPixel, decodedPixel, dummy.x, dummy.y);
						};

						double decodedPixel0[4];
						decodePixel(cpuOutput.get(), texCoord, decodedPixel0);

						double decodedPixel1[4];
						decodePixel(gpuOutput.get(), texCoord, decodedPixel1);

						for (uint32_t c = 0u; c < channelCount; ++c)
							sqErr += (decodedPixel0[c] - decodedPixel1[c]) * (decodedPixel0[c] - decodedPixel1[c]);
					}
				}
			}

			const double RMSE =  core::sqrt(sqErr / (static_cast<uint64_t>(extent0.width) * extent0.height * extent0.depth));
			logger->log("RMSE: %f\n", system::ILogger::ELL_DEBUG, RMSE);
		}

		_NBL_ALIGNED_FREE(blitFilterState.scratchMemory);
	}

	// Only for 1D and 3D 32 bit floating-point images because ext::ScreenShot doesn't support them yet
	void blitTest(const core::vectorSIMDu32& inImageDim, const asset::IImage::E_TYPE imageType, const asset::E_FORMAT imageFormat, const core::vectorSIMDu32& outImageDim, const BlitFilter::CState::E_ALPHA_SEMANTIC alphaSemantic)
	{
		const uint32_t channelCount = asset::getFormatChannelCount(imageFormat);

		core::smart_refctd_ptr<asset::ICPUImage> inImage = createCPUImage(inImageDim, imageType, imageFormat);

		std::uniform_real_distribution<float> dist(0.f, 20.f);
		std::mt19937 prng;

		float* inImagePixel = reinterpret_cast<float*>(inImage->getBuffer()->getPointer());
		for (uint32_t k = 0u; k < inImageDim[2]; ++k)
		{
			for (uint32_t j = 0u; j < inImageDim[1]; ++j)
			{
				for (uint32_t i = 0; i < inImageDim[0]; ++i)
				{
					const float valueToPut = dist(prng); // replicate the same value in all channels for now
					const uint32_t pixelIndex = k * (inImageDim[1] * inImageDim[0]) + j * inImageDim[0] + i;
					for (uint32_t c = 0u; c < channelCount; ++c)
						inImagePixel[channelCount * pixelIndex + c] = valueToPut;
				}
			}
		}

		const core::vectorSIMDf scaleX(1.f, 1.f, 1.f, 1.f);
		const core::vectorSIMDf scaleY(1.f, 1.f, 1.f, 1.f);
		const core::vectorSIMDf scaleZ(1.f, 1.f, 1.f, 1.f);

		auto kernelX = ScaledBoxKernel(scaleX, asset::CBoxImageFilterKernel()); // (-1/2, 1/2)
		auto kernelY = ScaledBoxKernel(scaleY, asset::CBoxImageFilterKernel()); // (-1/2, 1/2)
		auto kernelZ = ScaledBoxKernel(scaleZ, asset::CBoxImageFilterKernel()); // (-1/2, 1/2)

		BlitFilter::state_type blitFilterState(std::move(kernelX), std::move(kernelY), std::move(kernelZ));

		blitFilterState.inOffsetBaseLayer = core::vectorSIMDu32();
		blitFilterState.inExtentLayerCount = core::vectorSIMDu32(0u, 0u, 0u, inImage->getCreationParameters().arrayLayers) + inImage->getMipSize();
		blitFilterState.inImage = inImage.get();

		blitFilterState.outOffsetBaseLayer = core::vectorSIMDu32();
		const uint32_t outImageLayerCount = 1u;
		blitFilterState.outExtentLayerCount = core::vectorSIMDu32(outImageDim[0], outImageDim[1], outImageDim[2], 1u);

		blitFilterState.axisWraps[0] = asset::ISampler::ETC_CLAMP_TO_EDGE;
		blitFilterState.axisWraps[1] = asset::ISampler::ETC_CLAMP_TO_EDGE;
		blitFilterState.axisWraps[2] = asset::ISampler::ETC_CLAMP_TO_EDGE;
		blitFilterState.borderColor = asset::ISampler::E_TEXTURE_BORDER_COLOR::ETBC_FLOAT_OPAQUE_WHITE;

		blitFilterState.enableLUTUsage = true;

		blitFilterState.alphaSemantic = alphaSemantic;

		blitFilterState.scratchMemoryByteSize = BlitFilter::getRequiredScratchByteSize(&blitFilterState);
		blitFilterState.scratchMemory = reinterpret_cast<uint8_t*>(_NBL_ALIGNED_MALLOC(blitFilterState.scratchMemoryByteSize, 32));

		blitFilterState.computePhaseSupportLUT(&blitFilterState);

		// CPU
		core::vector<float> cpuOutput(static_cast<uint64_t>(outImageDim[0]) * outImageDim[1] * outImageDim[2] * channelCount);
		{

			auto outImage = createCPUImage(outImageDim, inImage->getCreationParameters().type, inImage->getCreationParameters().format);

			blitFilterState.outImage = outImage.get();

			logger->log("CPU begin..");
			if (!BlitFilter::execute(core::execution::par_unseq, &blitFilterState))
				logger->log("Failed to blit\n", system::ILogger::ELL_ERROR);

			logger->log("CPU end..");

			memcpy(cpuOutput.data(), outImage->getBuffer()->getPointer(), cpuOutput.size() * sizeof(float));
		}

		// GPU
		core::vector<float> gpuOutput(static_cast<uint64_t>(outImageDim[0]) * outImageDim[1] * outImageDim[2] * channelCount);
		{
			auto outImage = createCPUImage(outImageDim, inImage->getCreationParameters().type, inImage->getCreationParameters().format);

			inImage->addImageUsageFlags(asset::ICPUImage::EUF_SAMPLED_BIT);
			outImage->addImageUsageFlags(asset::ICPUImage::EUF_STORAGE_BIT);

			core::smart_refctd_ptr<video::IGPUImage> inImageGPU = nullptr;
			core::smart_refctd_ptr<video::IGPUImage> outImageGPU = nullptr;
			core::smart_refctd_ptr<video::IGPUImageView> inImageView = nullptr;
			core::smart_refctd_ptr<video::IGPUImageView> outImageView = nullptr;

			if (!getGPUImagesAndTheirViews(inImage, outImage, &inImageGPU, &outImageGPU, &inImageView, &outImageView))
				FATAL_LOG("Failed to convert CPU images to GPU images\n");

			CBlitFilter blitFilter(logicalDevice.get());

			const core::vectorSIMDu32 inExtent(inImage->getCreationParameters().extent.width, inImage->getCreationParameters().extent.height, inImage->getCreationParameters().extent.depth);
			const core::vectorSIMDu32 outExtent(outImageDim[0], outImageDim[1], outImageDim[2]);
			core::vectorSIMDf scale = static_cast<core::vectorSIMDf>(inExtent).preciseDivision(static_cast<core::vectorSIMDf>(outExtent));

			// Also, cannot use kernelX/Y/Z.getWindowSize() here because they haven't been scaled (yet) for upscaling/downscaling
			const core::vectorSIMDu32 windowDim = static_cast<core::vectorSIMDu32>(core::ceil(scale * core::vectorSIMDf(scaleX.x, scaleY.y, scaleZ.z)));

			const core::vectorSIMDu32 phaseCount = BlitFilter::getPhaseCount(inExtent, outExtent, inImage->getCreationParameters().type);
			core::smart_refctd_ptr<video::IGPUBuffer> phaseSupportLUT = nullptr;
			{
				BlitFilter::value_type* lut = reinterpret_cast<BlitFilter::value_type*>(blitFilterState.scratchMemory + BlitFilter::getPhaseSupportLUTByteOffset(&blitFilterState));

				const size_t lutSize = (static_cast<size_t>(phaseCount.x) * windowDim.x + static_cast<size_t>(phaseCount.y) * windowDim.y + static_cast<size_t>(phaseCount.z) * windowDim.z) * sizeof(float) * 4ull;

				// lut has the LUT in doubles, I want it in floats
				// Todo(achal): Probably need to pack them as half floats? But they are NOT different for each channel??
				// If we're under std140 layout, wouldn't it be better just make a static array of vec4 inside the uniform block
				// since a static array of floats of the same length would take up the same amount of space?
				core::vector<float> lutInFloats(lutSize / sizeof(float));
				for (uint32_t i = 0u; i < lutInFloats.size() / 4; ++i)
				{
					// losing precision here
					lutInFloats[4 * i + 0] = static_cast<float>(lut[i]);
					lutInFloats[4 * i + 1] = static_cast<float>(lut[i]);
					lutInFloats[4 * i + 2] = static_cast<float>(lut[i]);
					lutInFloats[4 * i + 3] = static_cast<float>(lut[i]);
				}

				video::IGPUBuffer::SCreationParams uboCreationParams = {};
				uboCreationParams.usage = static_cast<video::IGPUBuffer::E_USAGE_FLAGS>(video::IGPUBuffer::EUF_UNIFORM_BUFFER_BIT | video::IGPUBuffer::EUF_TRANSFER_DST_BIT);
				phaseSupportLUT = logicalDevice->createDeviceLocalGPUBufferOnDedMem(uboCreationParams, lutSize);

				// fill it up with data
				asset::SBufferRange<video::IGPUBuffer> bufferRange = {};
				bufferRange.offset = 0ull;
				bufferRange.size = lutSize;
				bufferRange.buffer = phaseSupportLUT;
				utilities->updateBufferRangeViaStagingBuffer(queues[CommonAPI::InitOutput::EQT_COMPUTE], bufferRange, lutInFloats.data());
			}

			const auto& blitDSLayout = blitFilter.getDefaultBlitDSLayout();
			const auto& blitPipelineLayout = blitFilter.getDefaultBlitPipelineLayout();
			const auto& blitShader = blitFilter.createBlitSpecializedShader(inImageGPU->getCreationParameters().format, outImageGPU->getCreationParameters().format, inImageGPU->getCreationParameters().type, inImageDim, outImageDim);
			const auto& blitPipeline = blitFilter.createBlitPipeline(core::smart_refctd_ptr(blitShader));

			const uint32_t blitDSCount = 1u;
			auto blitDescriptorPool = logicalDevice->createDescriptorPoolForDSLayouts(video::IDescriptorPool::ECF_NONE, &blitDSLayout.get(), &blitDSLayout.get() + 1ull, &blitDSCount);

			auto blitDS = logicalDevice->createGPUDescriptorSet(blitDescriptorPool.get(), core::smart_refctd_ptr(blitDSLayout));
			CBlitFilter::updateBlitDescriptorSet(logicalDevice.get(), blitDS.get(), inImageView, outImageView, phaseSupportLUT);

			logger->log("GPU begin..");
			blitFilter.blit(queues[CommonAPI::InitOutput::EQT_COMPUTE], alphaSemantic, nullptr, blitDS.get(), blitPipeline.get(), nullptr, inImageDim, imageType, imageFormat, outImageGPU);
			logger->log("GPU end..\n");

			// download results to check
			{
				core::smart_refctd_ptr<video::IGPUBuffer> downloadBuffer = nullptr;
				const size_t downloadSize = gpuOutput.size() * sizeof(float);

				video::IGPUBuffer::SCreationParams creationParams = {};
				creationParams.usage = video::IGPUBuffer::EUF_TRANSFER_DST_BIT;
				downloadBuffer = logicalDevice->createCPUSideGPUVisibleGPUBufferOnDedMem(creationParams, downloadSize);

				core::smart_refctd_ptr<video::IGPUCommandBuffer> cmdbuf = nullptr;
				logicalDevice->createCommandBuffers(commandPools[CommonAPI::InitOutput::EQT_COMPUTE].get(), video::IGPUCommandBuffer::EL_PRIMARY, 1u, &cmdbuf);
				auto fence = logicalDevice->createFence(video::IGPUFence::ECF_UNSIGNALED);

				cmdbuf->begin(video::IGPUCommandBuffer::EU_ONE_TIME_SUBMIT_BIT);

				asset::ICPUImage::SBufferCopy downloadRegion = {};
				downloadRegion.imageSubresource.aspectMask = video::IGPUImage::EAF_COLOR_BIT;
				downloadRegion.imageSubresource.layerCount = 1u;
				downloadRegion.imageExtent = outImageGPU->getCreationParameters().extent;

				cmdbuf->copyImageToBuffer(outImageGPU.get(), asset::EIL_GENERAL, downloadBuffer.get(), 1u, &downloadRegion);

				cmdbuf->end();

				video::IGPUQueue::SSubmitInfo submitInfo = {};
				submitInfo.commandBufferCount = 1u;
				submitInfo.commandBuffers = &cmdbuf.get();
				queues[CommonAPI::InitOutput::EQT_COMPUTE]->submit(1u, &submitInfo, fence.get());

				logicalDevice->blockForFences(1u, &fence.get());

				video::IDriverMemoryAllocation::MappedMemoryRange memoryRange = {};
				memoryRange.memory = downloadBuffer->getBoundMemory();
				memoryRange.length = downloadBuffer->getMemoryReqs().vulkanReqs.size;
				float* mappedGPUData = reinterpret_cast<float*>(logicalDevice->mapMemory(memoryRange));

				memcpy(gpuOutput.data(), mappedGPUData, gpuOutput.size() * sizeof(float));
				logicalDevice->unmapMemory(downloadBuffer->getBoundMemory());
			}
		}

		assert(gpuOutput.size() == cpuOutput.size());

		double sqErr = 0.0;
		for (uint64_t i = 0ull; i < gpuOutput.size(); ++i)
			sqErr += (gpuOutput[i] - cpuOutput[i]) * (gpuOutput[i] - cpuOutput[i]);

		const double RMSE = core::sqrt(sqErr / gpuOutput.size());
		logger->log("RMSE: %f\n", system::ILogger::ELL_DEBUG, RMSE);

		_NBL_ALIGNED_FREE(blitFilterState.scratchMemory);
	}

	float computeAlphaCoverage(const double referenceAlpha, asset::ICPUImage* image)
	{
		const uint32_t mipLevel = 0u;

		uint32_t alphaTestPassCount = 0u;
		const auto& extent = image->getCreationParameters().extent;

		for (uint32_t z = 0u; z < extent.depth; ++z)
		{
			for (uint32_t y = 0u; y < extent.height; ++y)
			{
				for (uint32_t x = 0u; x < extent.width; ++x)
				{
					const core::vectorSIMDu32 texCoord(x, y, z);
					core::vectorSIMDu32 dummy;
					const void* encodedPixel = image->getTexelBlockData(mipLevel, texCoord, dummy);

					double decodedPixel[4];
					asset::decodePixelsRuntime(image->getCreationParameters().format, &encodedPixel, decodedPixel, dummy.x, dummy.y);

					if (decodedPixel[3] > referenceAlpha)
						++alphaTestPassCount;
				}
			}
		}

		const float alphaCoverage = float(alphaTestPassCount) / float(extent.width * extent.height * extent.depth);
		return alphaCoverage;
	};

	bool getGPUImagesAndTheirViews(
		core::smart_refctd_ptr<asset::ICPUImage> inCPU,
		core::smart_refctd_ptr<asset::ICPUImage> outCPU,
		core::smart_refctd_ptr<video::IGPUImage>* inGPU,
		core::smart_refctd_ptr<video::IGPUImage>* outGPU,
		core::smart_refctd_ptr<video::IGPUImageView>* inGPUView,
		core::smart_refctd_ptr<video::IGPUImageView>* outGPUView)
	{
		core::smart_refctd_ptr<asset::ICPUImage> tmp[2] = { inCPU, outCPU };
		cpu2gpuParams.beginCommandBuffers();
		auto gpuArray = cpu2gpu.getGPUObjectsFromAssets(tmp, tmp + 2ull, cpu2gpuParams);
		cpu2gpuParams.waitForCreationToComplete();
		if (!gpuArray || gpuArray->size() < 2ull || (!(*gpuArray)[0]))
			return false;

		*inGPU = gpuArray->begin()[0];
		*outGPU = gpuArray->begin()[1];

		// do layout transition to GENERAL
		// (I think it might be a good idea to allow the user to change asset::ICPUImage's initialLayout and have the asset converter
		// do the layout transition for them)
		core::smart_refctd_ptr<video::IGPUCommandBuffer> cmdbuf = nullptr;
		logicalDevice->createCommandBuffers(commandPools[CommonAPI::InitOutput::EQT_COMPUTE].get(), video::IGPUCommandBuffer::EL_PRIMARY, 1u, &cmdbuf);

		auto fence = logicalDevice->createFence(video::IGPUFence::ECF_UNSIGNALED);
		video::IGPUCommandBuffer::SImageMemoryBarrier barriers[2] = {};

		barriers[0].oldLayout = asset::EIL_UNDEFINED;
		barriers[0].newLayout = asset::EIL_GENERAL;
		barriers[0].srcQueueFamilyIndex = ~0u;
		barriers[0].dstQueueFamilyIndex = ~0u;
		barriers[0].image = *inGPU;
		barriers[0].subresourceRange.aspectMask = video::IGPUImage::EAF_COLOR_BIT;
		barriers[0].subresourceRange.levelCount = 1u;
		barriers[0].subresourceRange.layerCount = 1u;

		barriers[1] = barriers[0];
		barriers[1].image = *outGPU;

		cmdbuf->begin(video::IGPUCommandBuffer::EU_ONE_TIME_SUBMIT_BIT);
		cmdbuf->pipelineBarrier(asset::EPSF_TOP_OF_PIPE_BIT, asset::EPSF_BOTTOM_OF_PIPE_BIT, asset::EDF_NONE, 0u, nullptr, 0u, nullptr, 2u, barriers);
		cmdbuf->end();

		video::IGPUQueue::SSubmitInfo submitInfo = {};
		submitInfo.commandBufferCount = 1u;
		submitInfo.commandBuffers = &cmdbuf.get();
		queues[CommonAPI::InitOutput::EQT_COMPUTE]->submit(1u, &submitInfo, fence.get());
		logicalDevice->blockForFences(1u, &fence.get());

		// create views for images
		{
			video::IGPUImageView::SCreationParams inCreationParams = {};
			inCreationParams.image = *inGPU;
			inCreationParams.viewType = getImageViewTypeFromImageType_GPU((*inGPU)->getCreationParameters().type);
			inCreationParams.format = (*inGPU)->getCreationParameters().format;
			inCreationParams.subresourceRange.aspectMask = video::IGPUImage::EAF_COLOR_BIT;
			inCreationParams.subresourceRange.layerCount = 1u;
			inCreationParams.subresourceRange.levelCount = 1u;

			video::IGPUImageView::SCreationParams outCreationParams = inCreationParams;
			outCreationParams.image = *outGPU;
			outCreationParams.format = (*outGPU)->getCreationParameters().format;

			*inGPUView = logicalDevice->createGPUImageView(std::move(inCreationParams));
			*outGPUView = logicalDevice->createGPUImageView(std::move(outCreationParams));
		}

		return true;
	};

	core::smart_refctd_ptr<asset::ICPUImage> loadImage(const char* path)
	{
		core::smart_refctd_ptr<asset::ICPUImage> inImage = nullptr;
		{
			constexpr auto cachingFlags = static_cast<nbl::asset::IAssetLoader::E_CACHING_FLAGS>(nbl::asset::IAssetLoader::ECF_DONT_CACHE_REFERENCES & nbl::asset::IAssetLoader::ECF_DONT_CACHE_TOP_LEVEL);

			asset::IAssetLoader::SAssetLoadParams loadParams(0ull, nullptr, cachingFlags);
			auto cpuImageBundle = assetManager->getAsset(path, loadParams);
			auto cpuImageContents = cpuImageBundle.getContents();
			if (cpuImageContents.empty() || cpuImageContents.begin() == cpuImageContents.end())
				return nullptr;

			auto asset = *cpuImageContents.begin();
			if (asset->getAssetType() == asset::IAsset::ET_IMAGE_VIEW)
				__debugbreak(); // it would be weird if the loaded image is already an image view

			inImage = core::smart_refctd_ptr_static_cast<asset::ICPUImage>(asset);
		}
		return inImage;
	};

	core::smart_refctd_ptr<nbl::ui::IWindowManager> windowManager;
	core::smart_refctd_ptr<nbl::ui::IWindow> window;
	core::smart_refctd_ptr<CommonAPI::CommonAPIEventCallback> windowCb;
	core::smart_refctd_ptr<nbl::video::IAPIConnection> apiConnection;
	core::smart_refctd_ptr<nbl::video::ISurface> surface;
	core::smart_refctd_ptr<nbl::video::IUtilities> utilities;
	core::smart_refctd_ptr<nbl::video::ILogicalDevice> logicalDevice;
	video::IPhysicalDevice* physicalDevice;
	std::array<video::IGPUQueue*, CommonAPI::InitOutput::MaxQueuesCount> queues;
	core::smart_refctd_ptr<nbl::video::ISwapchain> swapchain;
	core::smart_refctd_ptr<video::IGPURenderpass> renderpass = nullptr;
	std::array<nbl::core::smart_refctd_ptr<video::IGPUFramebuffer>, CommonAPI::InitOutput::MaxSwapChainImageCount> fbos;
	std::array<nbl::core::smart_refctd_ptr<nbl::video::IGPUCommandPool>, CommonAPI::InitOutput::MaxQueuesCount> commandPools;
	core::smart_refctd_ptr<nbl::system::ISystem> system;
	core::smart_refctd_ptr<nbl::asset::IAssetManager> assetManager;
	video::IGPUObjectFromAssetConverter::SParams cpu2gpuParams;
	core::smart_refctd_ptr<nbl::system::ILogger> logger;
	core::smart_refctd_ptr<CommonAPI::InputSystem> inputSystem;
	video::IGPUObjectFromAssetConverter cpu2gpu;

public:
	void setWindow(core::smart_refctd_ptr<nbl::ui::IWindow>&& wnd) override
	{
		window = std::move(wnd);
	}
	void setSystem(core::smart_refctd_ptr<nbl::system::ISystem>&& s) override
	{
		system = std::move(s);
	}
	nbl::ui::IWindow* getWindow() override
	{
		return window.get();
	}
	video::IAPIConnection* getAPIConnection() override
	{
		return apiConnection.get();
	}
	video::ILogicalDevice* getLogicalDevice()  override
	{
		return logicalDevice.get();
	}
	video::IGPURenderpass* getRenderpass() override
	{
		return renderpass.get();
	}
	void setSurface(core::smart_refctd_ptr<video::ISurface>&& s) override
	{
		surface = std::move(s);
	}
	void setFBOs(std::vector<core::smart_refctd_ptr<video::IGPUFramebuffer>>& f) override
	{
		for (int i = 0; i < f.size(); i++)
		{
			fbos[i] = core::smart_refctd_ptr(f[i]);
		}
	}
	void setSwapchain(core::smart_refctd_ptr<video::ISwapchain>&& s) override
	{
		swapchain = std::move(s);
	}
	uint32_t getSwapchainImageCount() override
	{
		return SC_IMG_COUNT;
	}
	virtual nbl::asset::E_FORMAT getDepthFormat() override
	{
		return nbl::asset::EF_D32_SFLOAT;
	}

	APP_CONSTRUCTOR(BlitFilterTestApp);
};

NBL_COMMON_API_MAIN(BlitFilterTestApp)

extern "C" {  _declspec(dllexport) DWORD NvOptimusEnablement = 0x00000001; }