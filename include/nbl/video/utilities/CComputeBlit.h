#ifndef _NBL_VIDEO_C_COMPUTE_BLIT_H_INCLUDED_

#include "nbl/asset/filters/CBlitImageFilter.h"

namespace nbl::video
{
class CComputeBlit : public core::IReferenceCounted
{
private:
	struct vec3 { float x, y, z; };
	struct uvec3 { uint32_t x, y, z; };

public:
	// This default is only for the blitting step (not alpha test or normalization steps) which always uses a 1D workgroup.
	// For the default values of alpha test and normalization steps, see getDefaultWorkgroupDims.
	static constexpr uint32_t DefaultBlitWorkgroupSize = 256u;
	static constexpr uint32_t DefaultAlphaBinCount = 256u;

#include "nbl/builtin/glsl/blit/parameters.glsl"

	struct dispatch_info_t
	{
		uint32_t wgCount[3];
	};

	CComputeBlit(core::smart_refctd_ptr<video::ILogicalDevice>&& logicalDevice, const uint32_t smemSize = 16 * 1024u) : device(std::move(logicalDevice)), sharedMemorySize(smemSize)
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
			sampler = device->createGPUSampler(std::move(params));
		}

		video::IGPUBuffer::SCreationParams creationParams = {};
		creationParams.usage = video::IGPUBuffer::EUF_STORAGE_BUFFER_BIT;
		m_dummySSBO = device->createDeviceLocalGPUBufferOnDedMem(creationParams, 1ull);
	}

	void getDefaultDescriptorSetLayouts(core::smart_refctd_ptr<video::IGPUDescriptorSetLayout>* outBlitDSLayout,
		core::smart_refctd_ptr<video::IGPUDescriptorSetLayout>* outKernelWeightsDSLayout, const asset::IBlitUtilities::E_ALPHA_SEMANTIC alphaSemantic)
	{
		if (outBlitDSLayout)
		{
			constexpr auto MAX_BLIT_DESCRIPTOR_COUNT = 3;
			asset::E_DESCRIPTOR_TYPE types[MAX_BLIT_DESCRIPTOR_COUNT] = { asset::EDT_COMBINED_IMAGE_SAMPLER, asset::EDT_STORAGE_IMAGE, asset::EDT_STORAGE_BUFFER }; // input image, output image, alpha statistics

			const auto blitType = (alphaSemantic == asset::IBlitUtilities::EAS_REFERENCE_OR_COVERAGE) ? EBT_COVERAGE_ADJUSTMENT : EBT_REGULAR;
			const auto actualDescriptorCount = (blitType == EBT_COVERAGE_ADJUSTMENT) ? 3 : 2;

			if (!m_blitDSLayout[blitType])
				m_blitDSLayout[blitType] = getDSLayout(actualDescriptorCount, types, device.get(), sampler);

			*outBlitDSLayout = m_blitDSLayout[blitType];
		}

		if (outKernelWeightsDSLayout)
		{
			if (!m_kernelWeightsDSLayout)
			{
				constexpr uint32_t DESCRIPTOR_COUNT = 1u;
				asset::E_DESCRIPTOR_TYPE types[DESCRIPTOR_COUNT] = { asset::EDT_UNIFORM_BUFFER };
				m_kernelWeightsDSLayout = getDSLayout(DESCRIPTOR_COUNT, types, device.get(), sampler);
			}

			*outKernelWeightsDSLayout = m_kernelWeightsDSLayout;
		}
	}

	void getDefaultPipelineLayouts(core::smart_refctd_ptr<video::IGPUPipelineLayout>* outBlitPipelineLayout,
		core::smart_refctd_ptr<video::IGPUPipelineLayout>* outCoverageAdjustmentPipelineLayout, const asset::IBlitUtilities::E_ALPHA_SEMANTIC alphaSemantic)
	{
		asset::SPushConstantRange pcRange = {};
		{
			pcRange.stageFlags = asset::IShader::ESS_COMPUTE;
			pcRange.offset = 0u;
			pcRange.size = sizeof(nbl_glsl_blit_parameters_t);
		}

		core::smart_refctd_ptr<video::IGPUDescriptorSetLayout> blitDSLayout;
		core::smart_refctd_ptr<video::IGPUDescriptorSetLayout> kernelWeightsDSLayout;
		getDefaultDescriptorSetLayouts(&blitDSLayout, &kernelWeightsDSLayout, alphaSemantic);

		const auto blitType = (alphaSemantic == asset::IBlitUtilities::EAS_REFERENCE_OR_COVERAGE) ? EBT_COVERAGE_ADJUSTMENT : EBT_REGULAR;

		if (outBlitPipelineLayout)
		{
			if (!m_blitPipelineLayout[blitType])
				m_blitPipelineLayout[blitType] = device->createGPUPipelineLayout(&pcRange, &pcRange + 1ull, core::smart_refctd_ptr(blitDSLayout), std::move(kernelWeightsDSLayout));

			*outBlitPipelineLayout = m_blitPipelineLayout[blitType];
		}

		if (outCoverageAdjustmentPipelineLayout && (blitType == EBT_COVERAGE_ADJUSTMENT))
		{
			if (!m_coverageAdjustmentPipelineLayout)
				m_coverageAdjustmentPipelineLayout = device->createGPUPipelineLayout(&pcRange, &pcRange + 1ull, core::smart_refctd_ptr(blitDSLayout));

			*outCoverageAdjustmentPipelineLayout = m_coverageAdjustmentPipelineLayout;
		}
	}

	core::smart_refctd_ptr<video::IGPUSpecializedShader> createAlphaTestSpecializedShader(const asset::IImage::E_TYPE inImageType);

	inline void buildParameters(nbl_glsl_blit_parameters_t& outPC, const core::vectorSIMDu32& inImageExtent, const core::vectorSIMDu32& outImageExtent, const asset::IImage::E_TYPE inImageType,
		const asset::E_FORMAT inImageFormat, const uint32_t layersToBlit = 1, const float referenceAlpha = 0.f)
	{
		outPC.outDim = { outImageExtent.x, outImageExtent.y, outImageExtent.z };
		outPC.referenceAlpha = referenceAlpha;

		core::vectorSIMDf scale = static_cast<core::vectorSIMDf>(inImageExtent).preciseDivision(static_cast<core::vectorSIMDf>(outImageExtent));
		outPC.fScale = {scale.x, scale.y, scale.z};

		outPC.inPixelCount = inImageExtent.x * inImageExtent.y * inImageExtent.z;

		const core::vectorSIMDf negativeSupport = core::vectorSIMDf(-0.5f, -0.5f, -0.5f) * scale;
		outPC.negativeSupport.x = negativeSupport.x; outPC.negativeSupport.y = negativeSupport.y; outPC.negativeSupport.z = negativeSupport.z;

		outPC.outPixelCount = outPC.outDim.x * outPC.outDim.y * outPC.outDim.z;

		const core::vectorSIMDu32 windowDim = getWindowDim(inImageExtent, outImageExtent);
		outPC.windowDim.x = windowDim.x; outPC.windowDim.y = windowDim.y; outPC.windowDim.z = windowDim.z;

		const core::vectorSIMDu32 phaseCount = asset::IBlitUtilities::getPhaseCount(inImageExtent, outImageExtent, inImageType);
		outPC.phaseCount.x = phaseCount.x; outPC.phaseCount.y = phaseCount.y; outPC.phaseCount.z = phaseCount.z;

		const uint32_t windowPixelCount = outPC.windowDim.x * outPC.windowDim.y * outPC.windowDim.z;
		const uint32_t smemPerWindow = windowPixelCount * (asset::getFormatChannelCount(inImageFormat) * sizeof(float));
		outPC.windowsPerWG = sharedMemorySize / smemPerWindow;
	}

	static inline void buildAlphaTestDispatchInfo(dispatch_info_t& outDispatchInfo, const core::vectorSIMDu32& inImageExtent, const asset::IImage::E_TYPE inImageType, const uint32_t layersToBlit = 1)
	{
		const core::vectorSIMDu32 workgroupDims = getDefaultWorkgroupDims(inImageType, layersToBlit);
		const core::vectorSIMDu32 workgroupCount = (inImageExtent + workgroupDims - core::vectorSIMDu32(1u, 1u, 1u, 1u)) / workgroupDims;
		outDispatchInfo.wgCount[0] = workgroupCount.x;
		outDispatchInfo.wgCount[1] = workgroupCount.y;
		outDispatchInfo.wgCount[2] = workgroupCount.z;
	}

	inline void buildBlitDispatchInfo(dispatch_info_t& outDispatchInfo, const core::vectorSIMDu32& inImageExtent, const core::vectorSIMDu32& outImageExtent, const asset::E_FORMAT inImageFormat, const uint32_t layersToBlit)
	{
		const auto windowDim = getWindowDim(inImageExtent, outImageExtent);
		const uint32_t windowPixelCount = windowDim.x * windowDim.y * windowDim.z;
		const uint32_t smemPerWindow = windowPixelCount * (asset::getFormatChannelCount(inImageFormat) * sizeof(float));
		auto windowsPerWG = sharedMemorySize / smemPerWindow;

		const uint32_t totalWindowCount = outImageExtent.x * outImageExtent.y * outImageExtent.z;
		const uint32_t wgCount = (totalWindowCount + windowsPerWG - 1) / windowsPerWG;

		outDispatchInfo.wgCount[0] = wgCount;
		outDispatchInfo.wgCount[1] = layersToBlit;
		outDispatchInfo.wgCount[2] = 1u;
	}

	static inline void buildNormalizationDispatchInfo(dispatch_info_t& outDispatchInfo, const core::vectorSIMDu32& outImageExtent, const asset::IImage::E_TYPE inImageType, const uint32_t alphaBinCount = DefaultAlphaBinCount, const uint32_t layersToBlit = 1)
	{
		const core::vectorSIMDu32 workgroupDims = getDefaultWorkgroupDims(inImageType, layersToBlit);
		assert(workgroupDims.x * workgroupDims.y * workgroupDims.z <= alphaBinCount);
		const core::vectorSIMDu32 workgroupCount = (outImageExtent + workgroupDims - core::vectorSIMDu32(1u, 1u, 1u, 1u)) / workgroupDims;

		outDispatchInfo.wgCount[0] = workgroupCount.x;
		outDispatchInfo.wgCount[1] = workgroupCount.y;
		outDispatchInfo.wgCount[2] = workgroupCount.z;
	}

	// outImageFormat dictates encoding.
	// outImageViewFormat dictates the GLSL storage image format qualifier.
	core::smart_refctd_ptr<video::IGPUSpecializedShader> createNormalizationSpecializedShader(const asset::IImage::E_TYPE inImageType, const asset::E_FORMAT outImageFormat, const asset::E_FORMAT outImageViewFormat);

	static inline core::vectorSIMDu32 getDefaultWorkgroupDims(const asset::IImage::E_TYPE inImageType, const uint32_t layersToBlit = 1)
	{
		switch (inImageType)
		{
		case asset::IImage::ET_1D:
			return core::vectorSIMDu32(256, layersToBlit, 1, 1);
		case asset::IImage::ET_2D:
			return core::vectorSIMDu32(16, 16, layersToBlit, 1);
		case asset::IImage::ET_3D:
			assert(layersToBlit == 1);
			return core::vectorSIMDu32(8, 8, 4, 1);
		default:
			return core::vectorSIMDu32(1, 1, 1, 1);
		}
	}

	// Scratch buffer allocated for this purpose should be cleared to 0s by the user.
	inline size_t getCoverageAdjustmentScratchSize(const asset::IBlitUtilities::E_ALPHA_SEMANTIC alphaSemantic, const uint32_t alphaBinCount, const uint32_t layersToBlit)
	{
		size_t scratchSize = 0ull; 

		if (alphaSemantic == asset::IBlitUtilities::EAS_REFERENCE_OR_COVERAGE)
		{
			scratchSize += sizeof(uint32_t); // for passed pixel atomic counter
			scratchSize += alphaBinCount * sizeof(uint32_t); // for alpha histogram
		}

		return scratchSize*layersToBlit;
	}

	// Todo(achal): Remove inExtent and outExtent params, they are only used for validation which should be done in the static create method/or the blit method, but not here
	core::smart_refctd_ptr<video::IGPUSpecializedShader> createBlitSpecializedShader(const asset::E_FORMAT inImageFormat, const asset::E_FORMAT outImageFormat, const asset::E_FORMAT outImageViewFormat,
		const asset::IImage::E_TYPE inImageType, const core::vectorSIMDu32& inExtent, const core::vectorSIMDu32& outExtent, const asset::IBlitUtilities::E_ALPHA_SEMANTIC alphaSemantic);

	void updateDescriptorSets(video::IGPUDescriptorSet* blitDS, video::IGPUDescriptorSet* kernelWeightsDS, core::smart_refctd_ptr<video::IGPUImageView> inImageView, core::smart_refctd_ptr<video::IGPUImageView> outImageView, core::smart_refctd_ptr<video::IGPUBuffer> coverageAdjustmentScratchBuffer, core::smart_refctd_ptr<video::IGPUBuffer> kernelWeightsUBO)
	{
		constexpr auto MAX_DESCRIPTOR_COUNT = 3;

		auto updateDS = [this, coverageAdjustmentScratchBuffer](video::IGPUDescriptorSet* ds, video::IGPUDescriptorSet::SDescriptorInfo* infos)
		{
			const auto& bindings = ds->getLayout()->getBindings();
			if (bindings.size() == 3)
				assert(coverageAdjustmentScratchBuffer);

			video::IGPUDescriptorSet::SWriteDescriptorSet writes[MAX_DESCRIPTOR_COUNT] = {};

			for (auto i = 0; i < bindings.size(); ++i)
			{
				writes[i].dstSet = ds;
				writes[i].binding = i;
				writes[i].arrayElement = 0u;
				writes[i].count = 1u;
				writes[i].info = &infos[i];
				writes[i].descriptorType = bindings.begin()[i].type;
			}

			device->updateDescriptorSets(bindings.size(), writes, 0u, nullptr);
		};

		if (blitDS)
		{
			assert(inImageView);
			assert(outImageView);

			video::IGPUDescriptorSet::SDescriptorInfo infos[MAX_DESCRIPTOR_COUNT] = {};
			// input image
			infos[0].desc = inImageView;
			infos[0].image.imageLayout = asset::EIL_GENERAL; // Todo(achal): Make it not GENERAL, this is a sampled image
			infos[0].image.sampler = nullptr;

			// output image
			infos[1].desc = outImageView;
			infos[1].image.imageLayout = asset::EIL_GENERAL;
			infos[1].image.sampler = nullptr;

			// coverage adjustment scratch
			infos[2].desc = coverageAdjustmentScratchBuffer ? coverageAdjustmentScratchBuffer : m_dummySSBO;
			infos[2].buffer.offset = 0;
			infos[2].buffer.size = coverageAdjustmentScratchBuffer ? coverageAdjustmentScratchBuffer->getCachedCreationParams().declaredSize : m_dummySSBO->getCachedCreationParams().declaredSize;

			updateDS(blitDS, infos);
		}

		if (kernelWeightsDS)
		{
			// scaled kernel phased LUT (cached weights)
			video::IGPUDescriptorSet::SDescriptorInfo info = {};
			info.desc = kernelWeightsUBO;
			info.buffer.offset = 0ull;
			info.buffer.size = kernelWeightsUBO->getCachedCreationParams().declaredSize;

			updateDS(kernelWeightsDS, &info);
		}
	}

	inline void blit(
		video::IGPUCommandBuffer* cmdbuf,
		const asset::IBlitUtilities::E_ALPHA_SEMANTIC alphaSemantic,
		video::IGPUDescriptorSet* alphaTestDS,
		video::IGPUComputePipeline* alphaTestPipeline,
		video::IGPUDescriptorSet* blitDS,
		video::IGPUDescriptorSet* blitWeightsDS,
		video::IGPUComputePipeline* blitPipeline,
		video::IGPUDescriptorSet* normalizationDS,
		video::IGPUComputePipeline* normalizationPipeline,
		const core::vectorSIMDu32& inImageExtent,
		const asset::IImage::E_TYPE inImageType,
		const asset::E_FORMAT inImageFormat,
		core::smart_refctd_ptr<video::IGPUImage> normalizationInImage,
		const uint32_t layersToBlit = 1,
		core::smart_refctd_ptr<video::IGPUBuffer> coverageAdjustmentScratchBuffer = nullptr,
		const float referenceAlpha = 0.f,
		const uint32_t alphaBinCount = DefaultAlphaBinCount)
	{
		const core::vectorSIMDu32 outImageExtent(normalizationInImage->getCreationParameters().extent.width, normalizationInImage->getCreationParameters().extent.height, normalizationInImage->getCreationParameters().extent.depth, 1u);

		nbl_glsl_blit_parameters_t pushConstants;
		buildParameters(pushConstants, inImageExtent, outImageExtent, inImageType, inImageFormat, layersToBlit, referenceAlpha);

		if (alphaSemantic == asset::IBlitUtilities::EAS_REFERENCE_OR_COVERAGE)
		{
			dispatch_info_t dispatchInfo;
			buildAlphaTestDispatchInfo(dispatchInfo, inImageExtent, inImageType, layersToBlit);

			cmdbuf->bindDescriptorSets(asset::EPBP_COMPUTE, alphaTestPipeline->getLayout(), 0u, 1u, &alphaTestDS);
			cmdbuf->bindComputePipeline(alphaTestPipeline);
			dispatchHelper(cmdbuf, alphaTestPipeline->getLayout(), pushConstants, dispatchInfo);
		}

		{
			dispatch_info_t dispatchInfo;
			buildBlitDispatchInfo(dispatchInfo, inImageExtent, outImageExtent, inImageFormat, layersToBlit);

			video::IGPUDescriptorSet* ds_raw[] = { blitDS, blitWeightsDS };
			cmdbuf->bindDescriptorSets(asset::EPBP_COMPUTE, blitPipeline->getLayout(), 0, 2, ds_raw);
			cmdbuf->bindComputePipeline(blitPipeline);
			dispatchHelper(cmdbuf, blitPipeline->getLayout(), pushConstants, dispatchInfo);
		}

		// After this dispatch ends and finishes writing to outImage, normalize outImage
		if (alphaSemantic == asset::IBlitUtilities::EAS_REFERENCE_OR_COVERAGE)
		{
			dispatch_info_t dispatchInfo;
			buildNormalizationDispatchInfo(dispatchInfo, outImageExtent, inImageType, alphaBinCount, layersToBlit);

			assert(coverageAdjustmentScratchBuffer);

			// Memory dependency to ensure the alpha test pass has finished writing to alphaTestCounterBuffer
			video::IGPUCommandBuffer::SBufferMemoryBarrier alphaTestBarrier = {};
			alphaTestBarrier.barrier.srcAccessMask = asset::EAF_SHADER_WRITE_BIT;
			alphaTestBarrier.barrier.dstAccessMask = asset::EAF_SHADER_READ_BIT;
			alphaTestBarrier.srcQueueFamilyIndex = ~0u;
			alphaTestBarrier.dstQueueFamilyIndex = ~0u;
			alphaTestBarrier.buffer = coverageAdjustmentScratchBuffer;
			alphaTestBarrier.size = coverageAdjustmentScratchBuffer->getCachedCreationParams().declaredSize;
			alphaTestBarrier.offset = 0;

			// Memory dependency to ensure that the previous compute pass has finished writing to the output image
			video::IGPUCommandBuffer::SImageMemoryBarrier readyForNorm = {};
			readyForNorm.barrier.srcAccessMask = asset::EAF_SHADER_WRITE_BIT;
			readyForNorm.barrier.dstAccessMask = static_cast<asset::E_ACCESS_FLAGS>(asset::EAF_SHADER_READ_BIT | asset::EAF_SHADER_WRITE_BIT);
			readyForNorm.oldLayout = asset::EIL_GENERAL;
			readyForNorm.newLayout = asset::EIL_GENERAL;
			readyForNorm.srcQueueFamilyIndex = ~0u;
			readyForNorm.dstQueueFamilyIndex = ~0u;
			readyForNorm.image = normalizationInImage;
			readyForNorm.subresourceRange.aspectMask = asset::IImage::EAF_COLOR_BIT;
			readyForNorm.subresourceRange.levelCount = 1u;
			readyForNorm.subresourceRange.layerCount = 1u;
			cmdbuf->pipelineBarrier(asset::EPSF_COMPUTE_SHADER_BIT, asset::EPSF_COMPUTE_SHADER_BIT, asset::EDF_NONE, 0u, nullptr, 1u, &alphaTestBarrier, 1u, &readyForNorm);

			cmdbuf->bindDescriptorSets(asset::EPBP_COMPUTE, normalizationPipeline->getLayout(), 0u, 1u, &normalizationDS);
			cmdbuf->bindComputePipeline(normalizationPipeline);
			dispatchHelper(cmdbuf, normalizationPipeline->getLayout(), pushConstants, dispatchInfo);
		}
	}

	//! WARNING: This function blocks and stalls the GPU!
	void blit(
		video::IGPUQueue* computeQueue,
		const asset::IBlitUtilities::E_ALPHA_SEMANTIC alphaSemantic,
		video::IGPUDescriptorSet* alphaTestDS,
		video::IGPUComputePipeline* alphaTestPipeline,
		video::IGPUDescriptorSet* blitDS,
		video::IGPUDescriptorSet* blitKernelWeightsDS,
		video::IGPUComputePipeline* blitPipeline,
		video::IGPUDescriptorSet* normalizationDS,
		video::IGPUComputePipeline* normalizationPipeline,
		const core::vectorSIMDu32& inImageExtent,
		const asset::IImage::E_TYPE inImageType,
		const asset::E_FORMAT inImageFormat,
		core::smart_refctd_ptr<video::IGPUImage> normalizationInImage,
		const uint32_t layersToBlit = 1,
		core::smart_refctd_ptr<video::IGPUBuffer> coverageAdjustmentScratchBuffer = nullptr,
		const float referenceAlpha = 0.f,
		const uint32_t alphaBinCount = DefaultAlphaBinCount)
	{
		auto cmdpool = device->createCommandPool(computeQueue->getFamilyIndex(), video::IGPUCommandPool::ECF_NONE);
		core::smart_refctd_ptr<video::IGPUCommandBuffer> cmdbuf;
		device->createCommandBuffers(cmdpool.get(), video::IGPUCommandBuffer::EL_PRIMARY, 1u, &cmdbuf);

		auto fence = device->createFence(video::IGPUFence::ECF_UNSIGNALED);

		cmdbuf->begin(video::IGPUCommandBuffer::EU_ONE_TIME_SUBMIT_BIT);
		blit(
			cmdbuf.get(), alphaSemantic,
			alphaTestDS, alphaTestPipeline,
			blitDS, blitKernelWeightsDS, blitPipeline,
			normalizationDS, normalizationPipeline,
			inImageExtent, inImageType, inImageFormat, normalizationInImage, layersToBlit,
			coverageAdjustmentScratchBuffer, referenceAlpha, alphaBinCount);
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

	static inline asset::E_FORMAT getIntermediateFormat(const asset::E_FORMAT format)
	{
		using namespace nbl::asset;

		switch (format)
		{
		case EF_R32G32B32A32_SFLOAT:
		case EF_R16G16B16A16_SFLOAT:
		case EF_R16G16B16A16_UNORM:
		case EF_R16G16B16A16_SNORM:
			return EF_R32G32B32A32_SFLOAT;
		
		case EF_R32G32_SFLOAT:
		case EF_R16G16_SFLOAT:
		case EF_R16G16_UNORM:
		case EF_R16G16_SNORM:
			return EF_R32G32_SFLOAT;

		case EF_B10G11R11_UFLOAT_PACK32:
			return EF_R16G16B16A16_SFLOAT;

		case EF_R32_SFLOAT:
		case EF_R16_SFLOAT:
		case EF_R16_UNORM:
		case EF_R16_SNORM:
			return EF_R32_SFLOAT;

		case EF_A2B10G10R10_UNORM_PACK32:
		case EF_R8G8B8A8_UNORM:
			return EF_R16G16B16A16_UNORM;

		case EF_R8G8_UNORM:
			return EF_R16G16_UNORM;

		case EF_R8_UNORM:
			return EF_R16_UNORM;

		case EF_R8G8B8A8_SNORM:
			return EF_R16G16B16A16_SNORM;

		case EF_R8G8_SNORM:
			return EF_R16G16_SNORM;

		default:
			return EF_UNKNOWN;
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
	enum E_BLIT_TYPE : uint8_t
	{
		EBT_REGULAR = 0,
		EBT_COVERAGE_ADJUSTMENT,
		EBT_COUNT
	};

	core::smart_refctd_ptr<video::IGPUDescriptorSetLayout> m_blitDSLayout[EBT_COUNT];
	core::smart_refctd_ptr<video::IGPUDescriptorSetLayout> m_kernelWeightsDSLayout;

	core::smart_refctd_ptr<video::IGPUPipelineLayout> m_blitPipelineLayout[EBT_COUNT];
	core::smart_refctd_ptr<video::IGPUPipelineLayout> m_coverageAdjustmentPipelineLayout;


	core::smart_refctd_ptr<video::IGPUBuffer> m_dummySSBO = nullptr;

	const uint32_t sharedMemorySize;
	core::smart_refctd_ptr<video::ILogicalDevice> device;

	core::smart_refctd_ptr<video::IGPUSampler> sampler = nullptr;

	// Todo(achal): Isn't this just getRealWindowSize?
	static inline core::vectorSIMDu32 getWindowDim(const core::vectorSIMDu32& inImageExtent, const core::vectorSIMDu32& outImageExtent)
	{
		core::vectorSIMDf scale = static_cast<core::vectorSIMDf>(inImageExtent).preciseDivision(static_cast<core::vectorSIMDf>(outImageExtent));
		return static_cast<core::vectorSIMDu32>(core::ceil(scale));
	}

	static inline void dispatchHelper(video::IGPUCommandBuffer* cmdbuf, const video::IGPUPipelineLayout* pipelineLayout, const nbl_glsl_blit_parameters_t& pushConstants, const dispatch_info_t& dispatchInfo)
	{
		cmdbuf->pushConstants(pipelineLayout, asset::IShader::ESS_COMPUTE, 0u, sizeof(nbl_glsl_blit_parameters_t), &pushConstants);
		cmdbuf->dispatch(dispatchInfo.wgCount[0], dispatchInfo.wgCount[1], dispatchInfo.wgCount[2]);
	}

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

	core::smart_refctd_ptr<asset::ICPUShader> getCPUShaderFromGLSL(const system::IFile* glsl);
};
}

#define _NBL_VIDEO_C_COMPUTE_BLIT_H_INCLUDED_
#endif
