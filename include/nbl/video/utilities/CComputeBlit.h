#ifndef _NBL_VIDEO_C_COMPUTE_BLIT_H_INCLUDED_

#include "nbl/asset/filters/CBlitImageFilter.h"

namespace nbl::video
{
class CComputeBlit
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
	// This default is only for the blitting step (not alpha test or normalization steps) which always uses a 1D workgroup.
	// For the default values of alpha test and normalization steps, see getDefaultWorkgroupDims.
	static constexpr uint32_t DefaultBlitWorkgroupSize = 256u;
	static constexpr uint32_t DefaultAlphaBinCount = 256u;

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

	CComputeBlit(video::ILogicalDevice* logicalDevice, const uint32_t smemSize = 16 * 1024u) : device(logicalDevice), sharedMemorySize(smemSize)
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

	static inline void getDefaultAlphaTestCounterBufferRange(asset::SBufferRange<video::IGPUBuffer>* outBufferRange, core::smart_refctd_ptr<video::IGPUBuffer> scratchBuffer, video::ILogicalDevice* logicalDevice)
	{
		if (scratchBuffer)
		{
			outBufferRange->offset = 0ull;
			outBufferRange->size = core::alignUp(sizeof(uint32_t), logicalDevice->getPhysicalDevice()->getLimits().SSBOAlignment);
			outBufferRange->buffer = scratchBuffer;
		}
	}

	static inline void getDefaultAlphaHistogramBufferRange(asset::SBufferRange<video::IGPUBuffer>* outBufferRange, core::smart_refctd_ptr<video::IGPUBuffer> scratchBuffer, video::ILogicalDevice* logicalDevice)
	{
		if (scratchBuffer)
		{
			outBufferRange->offset = core::alignUp(sizeof(uint32_t), logicalDevice->getPhysicalDevice()->getLimits().SSBOAlignment);
			outBufferRange->size = sizeof(uint32_t) * DefaultAlphaBinCount;
			outBufferRange->buffer = scratchBuffer;
		}
	}

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

	core::smart_refctd_ptr<video::IGPUSpecializedShader> createAlphaTestSpecializedShader(const asset::IImage::E_TYPE inImageType)
	{
		const char* shaderpath = "../default_compute_alpha_test.comp";
		auto cpuShader = loadShaderFromFile(shaderpath);

		if (!cpuShader)
			return nullptr;

		const auto workgroupDims = getDefaultWorkgroupDims(inImageType);

		auto cpuShaderOverriden = asset::IGLSLCompiler::createOverridenCopy(cpuShader.get(),
			"#define _NBL_GLSL_WORKGROUP_SIZE_X_ %d\n"
			"#define _NBL_GLSL_WORKGROUP_SIZE_Y_ %d\n"
			"#define _NBL_GLSL_WORKGROUP_SIZE_Z_ %d\n"
			"#define _NBL_GLSL_BLIT_ALPHA_TEST_DIM_COUNT_ %d\n",
			inImageType >= asset::IImage::ET_1D ? workgroupDims.x : 1u,
			inImageType >= asset::IImage::ET_2D ? workgroupDims.y : 1u,
			inImageType >= asset::IImage::ET_3D ? workgroupDims.z : 1u,
			static_cast<uint32_t>(inImageType) + 1u);

		cpuShaderOverriden->setFilePathHint(shaderpath);

		auto gpuUnspecShader = device->createGPUShader(std::move(cpuShaderOverriden));

		return device->createGPUSpecializedShader(gpuUnspecShader.get(), { nullptr, nullptr, "main" });
	}

	inline core::smart_refctd_ptr<video::IGPUComputePipeline> createAlphaTestPipeline(core::smart_refctd_ptr<video::IGPUSpecializedShader>&& specShader)
	{
		auto alphaTestPipeline = device->createGPUComputePipeline(nullptr, core::smart_refctd_ptr(alphaTestPipelineLayout), std::move(specShader));
		return alphaTestPipeline;
	}

	static void updateAlphaTestDescriptorSet(video::ILogicalDevice* logicalDevice, video::IGPUDescriptorSet* ds, core::smart_refctd_ptr<video::IGPUImageView> inImageView, const asset::SBufferRange<video::IGPUBuffer>& alphaTestCounter)
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

		// alpha test atomic counter
		infos[1].desc = alphaTestCounter.buffer;
		infos[1].buffer.offset = alphaTestCounter.offset;
		infos[1].buffer.size = alphaTestCounter.size;

		logicalDevice->updateDescriptorSets(descriptorCount, writes, 0u, nullptr);
	}

	inline void buildAlphaTestParameters(const float referenceAlpha, const asset::IImage::E_TYPE inImageType, const core::vectorSIMDu32& inImageExtent, alpha_test_push_constants_t& outPC, dispatch_info_t& outDispatchInfo)
	{
		outPC.referenceAlpha = referenceAlpha;

		const core::vectorSIMDu32 workgroupDims = getDefaultWorkgroupDims(inImageType);
		const core::vectorSIMDu32 workgroupCount = (inImageExtent + workgroupDims - core::vectorSIMDu32(1u, 1u, 1u, 1u)) / workgroupDims;
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

	core::smart_refctd_ptr<video::IGPUSpecializedShader> createNormalizationSpecializedShader(const asset::IImage::E_TYPE inImageType, const asset::E_FORMAT outFormat)
	{
		const char* shaderpath = "../default_compute_normalization.comp";
		auto cpuShader = loadShaderFromFile(shaderpath);
		if (!cpuShader)
			return nullptr;

		const asset::E_FORMAT outImageViewFormat = getOutImageViewFormat(outFormat);
		if (outImageViewFormat == asset::EF_UNKNOWN)
			return nullptr;

		const char* outImageFormatGLSLString = getGLSLFormatStringFromFormat(outFormat);
		const char* outImageViewFormatGLSLString = getGLSLFormatStringFromFormat(outImageViewFormat);

		char formatInclude[1024] = "";
		if (outFormat != outImageViewFormat)
			snprintf(formatInclude, sizeof(formatInclude), "#include <../blit/formats/%s.glsl>\n", outImageFormatGLSLString);

		const char* overrideFormat =
			"#define _NBL_GLSL_WORKGROUP_SIZE_X_ %d\n"
			"#define _NBL_GLSL_WORKGROUP_SIZE_Y_ %d\n"
			"#define _NBL_GLSL_WORKGROUP_SIZE_Z_ %d\n"
			"#define _NBL_GLSL_BLIT_NORMALIZATION_BIN_COUNT_ %d\n"
			"#define _NBL_GLSL_BLIT_NORMALIZATION_DIM_COUNT_ %d\n"
			"#define _NBL_GLSL_BLIT_NORMALIZATION_INOUT_IMAGE_FORMAT_ %s\n"
			"%s\n" // _NBL_GLSL_BLIT_NORMALIZATION_SOFTWARE_CODEC_
			"%s\n"; // format include

		core::vectorSIMDu32 workgroupDim = getDefaultWorkgroupDims(inImageType);

		auto cpuShaderOverriden = asset::IGLSLCompiler::createOverridenCopy(cpuShader.get(),
			overrideFormat,
			inImageType >= asset::IImage::ET_1D ? workgroupDim.x : 1u,
			inImageType >= asset::IImage::ET_2D ? workgroupDim.y : 1u,
			inImageType >= asset::IImage::ET_3D ? workgroupDim.z : 1u,
			DefaultAlphaBinCount,
			static_cast<uint32_t>(inImageType + 1u),
			outImageViewFormatGLSLString,
			outFormat != outImageViewFormat ? "#define _NBL_GLSL_BLIT_NORMALIZATION_SOFTWARE_CODEC_\n" : "",
			reinterpret_cast<const char*>(formatInclude));

		cpuShaderOverriden->setFilePathHint(shaderpath);

		auto gpuUnspecShader = device->createGPUShader(std::move(cpuShaderOverriden));

		return device->createGPUSpecializedShader(gpuUnspecShader.get(), { nullptr, nullptr, "main" });
	}

	inline core::smart_refctd_ptr<video::IGPUComputePipeline> createNormalizationPipeline(core::smart_refctd_ptr<video::IGPUSpecializedShader>&& specShader)
	{
		auto normalizationPipeline = device->createGPUComputePipeline(nullptr, core::smart_refctd_ptr(normalizationPipelineLayout), core::smart_refctd_ptr(specShader));
		return normalizationPipeline;
	}

	static void updateNormalizationDescriptorSet(video::ILogicalDevice* logicalDevice, video::IGPUDescriptorSet* ds, core::smart_refctd_ptr<video::IGPUImageView> outImageView, asset::SBufferRange<video::IGPUBuffer> alphaHistogram, asset::SBufferRange<video::IGPUBuffer> alphaTestCounter)
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
		infos[1].desc = alphaHistogram.buffer;
		infos[1].buffer.offset = alphaHistogram.offset;
		infos[1].buffer.size = alphaHistogram.size;

		// alpha test counter buffer
		infos[2].desc = alphaTestCounter.buffer;
		infos[2].buffer.offset = alphaTestCounter.offset;
		infos[2].buffer.size = alphaTestCounter.size;

		logicalDevice->updateDescriptorSets(descriptorCount, writes, 0u, nullptr);
	}

	static inline core::vectorSIMDu32 getDefaultWorkgroupDims(const asset::IImage::E_TYPE inImageType)
	{
		switch (inImageType)
		{
		case asset::IImage::ET_1D:
			return core::vectorSIMDu32(256, 1, 1, 1);
		case asset::IImage::ET_2D:
			return core::vectorSIMDu32(16, 16, 1, 1);
		case asset::IImage::ET_3D:
			return core::vectorSIMDu32(8, 8, 4, 1);
		default:
			return core::vectorSIMDu32(1, 1, 1, 1);
		}
	}

	void buildNormalizationParameters(const asset::IImage::E_TYPE inImageType, const core::vectorSIMDu32& inImageExtent, const core::vectorSIMDu32& outImageExtent, const float referenceAlpha, normalization_push_constants_t& outPC, dispatch_info_t& outDispatchInfo)
	{
		outPC.outDim = { outImageExtent.x, outImageExtent.y, outImageExtent.z };
		outPC.inPixelCount = inImageExtent.x * inImageExtent.y * inImageExtent.z;
		outPC.referenceAlpha = static_cast<float>(referenceAlpha);

		const core::vectorSIMDu32 workgroupDims = getDefaultWorkgroupDims(inImageType);
		assert(workgroupDims.x * workgroupDims.y * workgroupDims.z <= DefaultAlphaBinCount);
		const core::vectorSIMDu32 workgroupCount = (outImageExtent + workgroupDims - core::vectorSIMDu32(1u, 1u, 1u, 1u)) / workgroupDims;

		outDispatchInfo.wgCount[0] = workgroupCount.x;
		outDispatchInfo.wgCount[1] = workgroupCount.y;
		outDispatchInfo.wgCount[2] = workgroupCount.z;
	}

	core::smart_refctd_ptr<video::IGPUSpecializedShader> createBlitSpecializedShader(const asset::E_FORMAT inFormat, const asset::E_FORMAT outFormat, const asset::IImage::E_TYPE inImageType,
		const core::vectorSIMDu32& inExtent, const core::vectorSIMDu32& outExtent, const asset::CBlitUtilities::E_ALPHA_SEMANTIC alphaSemantic)
	{
		const char* shaderpath = "../default_compute_blit.comp";
		auto cpuShader = loadShaderFromFile(shaderpath);
		if (!cpuShader)
			return nullptr;

		const uint32_t inChannelCount = asset::getFormatChannelCount(inFormat);
		const uint32_t outChannelCount = asset::getFormatChannelCount(outFormat);
		assert(outChannelCount <= inChannelCount);

		core::vectorSIMDf scale = static_cast<core::vectorSIMDf>(inExtent).preciseDivision(static_cast<core::vectorSIMDf>(outExtent));

		const core::vectorSIMDu32 windowDim = static_cast<core::vectorSIMDu32>(core::ceil(scale));
		const uint32_t windowPixelCount = windowDim.x * windowDim.y * windowDim.z;
		// It is important not to use asset::getTexelOrBlockBytesize here.
		// Even though input pixels are stored in the shared memory we use outChannelCount here because only outChannelCount channels of
		// the input image actually need blitting and hence only those channels are stored in shared memory.
		const size_t windowSize = static_cast<size_t>(windowPixelCount) * outChannelCount * sizeof(float);
		// Fail if the window cannot be preloaded into shared memory
		if (windowSize > sharedMemorySize)
			return nullptr;

		// Fail if any dimension of the window is bigger than workgroup size, forcing us to reuse inovcation to process just a single dimension of that window
		if ((windowDim.x > DefaultBlitWorkgroupSize) || (windowDim.y > DefaultBlitWorkgroupSize) || (windowDim.z > DefaultBlitWorkgroupSize))
			return nullptr;

		// Fail if I would need to reuse invocations just to process a single window
		if (windowPixelCount > DefaultBlitWorkgroupSize)
			return nullptr;

		// inFormat should support SAMPLED_BIT format feature

		const asset::E_FORMAT outImageViewFormat = getOutImageViewFormat(outFormat);
		if (outImageViewFormat == asset::EF_UNKNOWN)
			return nullptr;

		const char* outImageFormatGLSLString = getGLSLFormatStringFromFormat(outFormat);
		const char* outImageViewFormatGLSLString = getGLSLFormatStringFromFormat(outImageViewFormat);

		char formatInclude[1024] = "";
		if (outFormat != outImageViewFormat)
			snprintf(formatInclude, sizeof(formatInclude), "#include <../blit/formats/%s.glsl>\n", outImageFormatGLSLString);

		const uint32_t smemFloatCount = sharedMemorySize / (sizeof(float) * outChannelCount);

		const char* overrideFormat =
			"#define _NBL_GLSL_WORKGROUP_SIZE_ %d\n"
			"#define _NBL_GLSL_BLIT_OUT_CHANNEL_COUNT_ %d\n"
			"#define _NBL_GLSL_BLIT_DIM_COUNT_ %d\n"
			"#define _NBL_GLSL_BLIT_OUT_IMAGE_FORMAT_ %s\n"
			"#define _NBL_GLSL_BLIT_SMEM_FLOAT_COUNT_ %d\n"
			"%s" // _NBL_GLSL_BLIT_COVERAGE_SEMANTIC_
			"%s" // _NBL_GLSL_BLIT_SOFTWARE_CODEC_
			"%s"; // format include

		auto cpuShaderOverriden = asset::IGLSLCompiler::createOverridenCopy(cpuShader.get(),
			overrideFormat,
			DefaultBlitWorkgroupSize,
			outChannelCount,
			static_cast<uint32_t>(inImageType + 1u),
			outImageViewFormatGLSLString,
			smemFloatCount,
			alphaSemantic == asset::CBlitUtilities::EAS_REFERENCE_OR_COVERAGE ? "#define _NBL_GLSL_BLIT_COVERAGE_SEMANTIC_\n" : "",
			outFormat != outImageViewFormat ? "#define _NBL_GLSL_BLIT_SOFTWARE_CODEC_\n" : "",
			reinterpret_cast<const char*>(formatInclude));

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

	static void updateBlitDescriptorSet(video::ILogicalDevice* logicalDevice, video::IGPUDescriptorSet* ds, core::smart_refctd_ptr<video::IGPUImageView> inImageView, core::smart_refctd_ptr<video::IGPUImageView> outImageView, core::smart_refctd_ptr<video::IGPUBuffer> phaseSupportLUT, asset::SBufferRange<video::IGPUBuffer> alphaHistogram = {})
	{
		constexpr uint32_t MAX_DESCRIPTOR_COUNT = 5u;

		const auto& bindings = ds->getLayout()->getBindings();
		const uint32_t bindingCount = static_cast<uint32_t>(bindings.size());
		const uint32_t descriptorCount = alphaHistogram.buffer ? bindingCount : bindingCount - 1u;
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

		if (alphaHistogram.buffer)
		{
			infos[3].desc = alphaHistogram.buffer;
			infos[3].buffer.offset = alphaHistogram.offset;
			infos[3].buffer.size = alphaHistogram.size;
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

		const core::vectorSIMDu32 phaseCount = asset::CBlitUtilities::getPhaseCount(inImageExtent, outImageExtent, inImageType);
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
		const asset::CBlitUtilities::E_ALPHA_SEMANTIC alphaSemantic,
		video::IGPUDescriptorSet* alphaTestDS,
		video::IGPUComputePipeline* alphaTestPipeline,
		video::IGPUDescriptorSet* blitDS,
		video::IGPUComputePipeline* blitPipeline,
		video::IGPUDescriptorSet* normalizationDS,
		video::IGPUComputePipeline* normalizationPipeline,
		const core::vectorSIMDu32& inImageExtent,
		const asset::IImage::E_TYPE inImageType,
		const asset::E_FORMAT inImageFormat,
		core::smart_refctd_ptr<video::IGPUImage> outImage,
		const float referenceAlpha = 0.f,
		const asset::SBufferRange<video::IGPUBuffer>* alphaTestCounter = nullptr)
	{
		if (alphaSemantic == asset::CBlitUtilities::EAS_REFERENCE_OR_COVERAGE)
		{
			cmdbuf->bindDescriptorSets(asset::EPBP_COMPUTE, alphaTestPipeline->getLayout(), 0u, 1u, &alphaTestDS);
			cmdbuf->bindComputePipeline(alphaTestPipeline);
			CComputeBlit::alpha_test_push_constants_t alphaTestPC;
			CComputeBlit::dispatch_info_t alphaTestDispatchInfo;
			buildAlphaTestParameters(referenceAlpha, inImageType, inImageExtent, alphaTestPC, alphaTestDispatchInfo);
			dispatchHelper<CComputeBlit::alpha_test_push_constants_t>(cmdbuf, alphaTestPipeline->getLayout(), alphaTestPC, alphaTestDispatchInfo);
		}

		cmdbuf->bindDescriptorSets(asset::EPBP_COMPUTE, blitPipeline->getLayout(), 0u, 1u, &blitDS);
		cmdbuf->bindComputePipeline(blitPipeline);
		CComputeBlit::blit_push_constants_t blitPC;
		CComputeBlit::dispatch_info_t blitDispatchInfo;
		const core::vectorSIMDu32 outImageExtent(outImage->getCreationParameters().extent.width, outImage->getCreationParameters().extent.height, outImage->getCreationParameters().extent.depth, 1u);
		buildBlitParameters(inImageExtent, outImageExtent, inImageType, inImageFormat, blitPC, blitDispatchInfo);
		dispatchHelper<CComputeBlit::blit_push_constants_t>(cmdbuf, blitPipeline->getLayout(), blitPC, blitDispatchInfo);

		// After this dispatch ends and finishes writing to outImage, normalize outImage
		if (alphaSemantic == asset::CBlitUtilities::EAS_REFERENCE_OR_COVERAGE)
		{
			assert(alphaTestCounter);

			// Memory dependency to ensure the alpha test pass has finished writing to alphaTestCounterBuffer
			video::IGPUCommandBuffer::SBufferMemoryBarrier alphaTestBarrier = {};
			alphaTestBarrier.barrier.srcAccessMask = asset::EAF_SHADER_WRITE_BIT;
			alphaTestBarrier.barrier.dstAccessMask = asset::EAF_SHADER_READ_BIT;
			alphaTestBarrier.srcQueueFamilyIndex = ~0u;
			alphaTestBarrier.dstQueueFamilyIndex = ~0u;
			alphaTestBarrier.buffer = alphaTestCounter->buffer;
			alphaTestBarrier.size = alphaTestCounter->size;
			alphaTestBarrier.offset = alphaTestCounter->offset;

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

			cmdbuf->bindDescriptorSets(asset::EPBP_COMPUTE, normalizationPipeline->getLayout(), 0u, 1u, &normalizationDS);
			cmdbuf->bindComputePipeline(normalizationPipeline);
			CComputeBlit::normalization_push_constants_t normPC = {};
			CComputeBlit::dispatch_info_t normDispatchInfo = {};
			buildNormalizationParameters(inImageType, inImageExtent, outImageExtent, referenceAlpha, normPC, normDispatchInfo);
			dispatchHelper<CComputeBlit::normalization_push_constants_t>(cmdbuf, normalizationPipeline->getLayout(), normPC, normDispatchInfo);
		}
	}

	//! WARNING: This function blocks and stalls the GPU!
	void blit(
		video::IGPUQueue* computeQueue,
		const asset::CBlitUtilities::E_ALPHA_SEMANTIC alphaSemantic,
		video::IGPUDescriptorSet* alphaTestDS,
		video::IGPUComputePipeline* alphaTestPipeline,
		video::IGPUDescriptorSet* blitDS,
		video::IGPUComputePipeline* blitPipeline,
		video::IGPUDescriptorSet* normalizationDS,
		video::IGPUComputePipeline* normalizationPipeline,
		const core::vectorSIMDu32& inImageExtent,
		const asset::IImage::E_TYPE inImageType,
		const asset::E_FORMAT inImageFormat,
		core::smart_refctd_ptr<video::IGPUImage> outImage,
		const float referenceAlpha = 0.f,
		const asset::SBufferRange<video::IGPUBuffer>* alphaTestCounter = nullptr)
	{
		auto cmdpool = device->createCommandPool(computeQueue->getFamilyIndex(), video::IGPUCommandPool::ECF_NONE);
		core::smart_refctd_ptr<video::IGPUCommandBuffer> cmdbuf;
		device->createCommandBuffers(cmdpool.get(), video::IGPUCommandBuffer::EL_PRIMARY, 1u, &cmdbuf);

		auto fence = device->createFence(video::IGPUFence::ECF_UNSIGNALED);

		cmdbuf->begin(video::IGPUCommandBuffer::EU_ONE_TIME_SUBMIT_BIT);
		blit(cmdbuf.get(), alphaSemantic, alphaTestDS, alphaTestPipeline, blitDS, blitPipeline, normalizationDS, normalizationPipeline, inImageExtent, inImageType, inImageFormat, outImage, referenceAlpha, alphaTestCounter);
		cmdbuf->end();

		video::IGPUQueue::SSubmitInfo submitInfo = {};
		submitInfo.commandBufferCount = 1u;
		submitInfo.commandBuffers = &cmdbuf.get();
		computeQueue->submit(1u, &submitInfo, fence.get());

		device->blockForFences(1u, &fence.get());
	}

	inline asset::E_FORMAT getOutImageViewFormat(const asset::E_FORMAT format)
	{
		const auto& formatUsages = device->getPhysicalDevice()->getImageFormatUsagesOptimal(format);

		if (formatUsages.storageImage)
		{
			return format;
		}
		else
		{
			const asset::E_FORMAT compatFormat = getCompatClassFormat(format);

			const auto& compatClassFormatUsages = device->getPhysicalDevice()->getImageFormatUsagesOptimal(compatFormat);
			if (!compatClassFormatUsages.storageImage)
				return asset::EF_UNKNOWN;
			else
				return compatFormat;
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
		case asset::EF_R8_UINT:
			result = "r8ui";
			break;
		case asset::EF_R16_UINT:
			result = "r16ui";
			break;
		case asset::EF_R32_UINT:
			result = "r32ui";
			break;
		case asset::EF_R32G32_UINT:
			result = "rg32ui";
			break;
		case asset::EF_R32G32B32A32_UINT:
			result = "rgba32ui";
			break;
		default:
			__debugbreak();
		}

		return result;
	}

private:
	core::smart_refctd_ptr<video::IGPUDescriptorSetLayout> alphaTestDSLayout = nullptr;
	core::smart_refctd_ptr<video::IGPUPipelineLayout> alphaTestPipelineLayout = nullptr;

	core::smart_refctd_ptr<video::IGPUDescriptorSetLayout> blitDSLayout = nullptr;
	core::smart_refctd_ptr<video::IGPUPipelineLayout> blitPipelineLayout = nullptr;

	core::smart_refctd_ptr<video::IGPUDescriptorSetLayout> normalizationDSLayout = nullptr;
	core::smart_refctd_ptr<video::IGPUPipelineLayout> normalizationPipelineLayout = nullptr;

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

	static inline asset::E_FORMAT getCompatClassFormat(const asset::E_FORMAT format)
	{
		const asset::E_FORMAT_CLASS formatClass = asset::getFormatClass(format);
		switch (formatClass)
		{
		case asset::EFC_8_BIT:
			return asset::EF_R8_UINT;
		case asset::EFC_16_BIT:
			return asset::EF_R16_UINT;
		case asset::EFC_32_BIT:
			return asset::EF_R32_UINT;
		case asset::EFC_64_BIT:
			return asset::EF_R32G32_UINT;
		case asset::EFC_128_BIT:
			return asset::EF_R32G32B32A32_UINT;
		default:
			return asset::EF_UNKNOWN;
		}
	}

	core::smart_refctd_ptr<asset::ICPUShader> loadShaderFromFile(const char* filepath)
	{
		auto sys = device->getPhysicalDevice()->getSystem();
		system::ISystem::future_t<core::smart_refctd_ptr<system::IFile>> future;
		sys->createFile(future, filepath, static_cast<system::IFile::E_CREATE_FLAGS>(system::IFile::ECF_READ | system::IFile::ECF_MAPPABLE));

		const auto glslFile = future.get();
		if (!glslFile)
			return nullptr;

		auto buffer = core::make_smart_refctd_ptr<asset::ICPUBuffer>(glslFile->getSize());
		memcpy(buffer->getPointer(), reinterpret_cast<const system::IFile*>(glslFile.get())->getMappedPointer(), glslFile->getSize());

		auto cpuShader = core::make_smart_refctd_ptr<asset::ICPUShader>(std::move(buffer), asset::IShader::buffer_contains_glsl_t{}, asset::IShader::ESS_COMPUTE, "????");

		return cpuShader;
	}
};
}

#define _NBL_VIDEO_C_COMPUTE_BLIT_H_INCLUDED_
#endif
