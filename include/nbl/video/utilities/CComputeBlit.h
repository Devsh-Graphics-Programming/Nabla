#ifndef _NBL_VIDEO_C_COMPUTE_BLIT_H_INCLUDED_

#include "nbl/asset/filters/CBlitUtilities.h"

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

#include "nbl/builtin/glsl/blit/parameters.glsl"

	struct dispatch_info_t
	{
		uint32_t wgCount[3];
	};

	//! Set smemSize param to ~0u to use all the shared memory available.
	static core::smart_refctd_ptr<CComputeBlit> create(core::smart_refctd_ptr<video::ILogicalDevice>&& logicalDevice, const uint32_t smemSize = ~0u)
	{
		auto result = core::smart_refctd_ptr<CComputeBlit>(new CComputeBlit(std::move(logicalDevice)), core::dont_grab);

		result->setAvailableSharedMemory(smemSize);

		{
			constexpr auto BlitDescriptorCount = 3;
			const asset::E_DESCRIPTOR_TYPE types[BlitDescriptorCount] = { asset::EDT_COMBINED_IMAGE_SAMPLER, asset::EDT_STORAGE_IMAGE, asset::EDT_STORAGE_BUFFER }; // input image, output image, alpha statistics

			for (auto i = 0; i < static_cast<uint8_t>(EBT_COUNT); ++i)
			{
				result->m_blitDSLayout[i] = result->createDSLayout(i == static_cast<uint8_t>(EBT_COVERAGE_ADJUSTMENT) ? 3 : 2, types, result->m_device.get());
				if (!result->m_blitDSLayout[i])
					return nullptr;
			}
		}

		{
			constexpr auto KernelWeightsDescriptorCount = 1;
			asset::E_DESCRIPTOR_TYPE types[KernelWeightsDescriptorCount] = { asset::EDT_UNIFORM_TEXEL_BUFFER };
			result->m_kernelWeightsDSLayout = result->createDSLayout(KernelWeightsDescriptorCount, types, result->m_device.get());

			if (!result->m_kernelWeightsDSLayout)
				return nullptr;
		}

		asset::SPushConstantRange pcRange = {};
		{
			pcRange.stageFlags = asset::IShader::ESS_COMPUTE;
			pcRange.offset = 0u;
			pcRange.size = sizeof(nbl_glsl_blit_parameters_t);
		}

		for (auto i = 0; i < static_cast<uint8_t>(EBT_COUNT); ++i)
		{
			result->m_blitPipelineLayout[i] = result->m_device->createPipelineLayout(&pcRange, &pcRange + 1ull, core::smart_refctd_ptr(result->m_blitDSLayout[i]), core::smart_refctd_ptr(result->m_kernelWeightsDSLayout));
			if (!result->m_blitPipelineLayout[i])
				return nullptr;
		}

		result->m_coverageAdjustmentPipelineLayout = result->m_device->createPipelineLayout(&pcRange, &pcRange + 1ull, core::smart_refctd_ptr(result->m_blitDSLayout[EBT_COVERAGE_ADJUSTMENT]));
		if (!result->m_coverageAdjustmentPipelineLayout)
			return nullptr;

		return result;
	}

	inline void setAvailableSharedMemory(const uint32_t smemSize)
	{
		if (smemSize == ~0u)
			m_availableSharedMemory = m_device->getPhysicalDevice()->getProperties().limits.maxComputeSharedMemorySize;
		else
			m_availableSharedMemory = core::min(core::roundUp(smemSize, static_cast<uint32_t>(sizeof(float) * 64)), m_device->getPhysicalDevice()->getLimits().maxComputeSharedMemorySize);
	}

	inline core::smart_refctd_ptr<video::IGPUDescriptorSetLayout> getDefaultBlitDescriptorSetLayout(const asset::IBlitUtilities::E_ALPHA_SEMANTIC alphaSemantic) const
	{
		if (alphaSemantic == asset::IBlitUtilities::EAS_REFERENCE_OR_COVERAGE)
			return m_blitDSLayout[EBT_COVERAGE_ADJUSTMENT];
		else
			return m_blitDSLayout[EBT_REGULAR];
	}

	inline core::smart_refctd_ptr<video::IGPUDescriptorSetLayout> getDefaultKernelWeightsDescriptorSetLayout() const
	{
		return m_kernelWeightsDSLayout;
	}

	inline core::smart_refctd_ptr<video::IGPUPipelineLayout> getDefaultBlitPipelineLayout(const asset::IBlitUtilities::E_ALPHA_SEMANTIC alphaSemantic) const
	{
		if (alphaSemantic == asset::IBlitUtilities::EAS_REFERENCE_OR_COVERAGE)
			return m_blitPipelineLayout[EBT_COVERAGE_ADJUSTMENT];
		else
			return m_blitPipelineLayout[EBT_REGULAR];
	}

	inline core::smart_refctd_ptr<video::IGPUPipelineLayout> getDefaultCoverageAdjustmentPipelineLayout() const
	{
		return m_coverageAdjustmentPipelineLayout;
	}

	// @param `alphaBinCount` is only required to size the histogram present in the default nbl_glsl_blit_AlphaStatistics_t in default_compute_common.comp
	core::smart_refctd_ptr<video::IGPUSpecializedShader> createAlphaTestSpecializedShader(const asset::IImage::E_TYPE inImageType, const uint32_t alphaBinCount = asset::IBlitUtilities::DefaultAlphaBinCount);

	core::smart_refctd_ptr<video::IGPUComputePipeline> getAlphaTestPipeline(const uint32_t alphaBinCount, const asset::IImage::E_TYPE imageType)
	{
		const auto workgroupDims = getDefaultWorkgroupDims(imageType);
		const auto paddedAlphaBinCount = getPaddedAlphaBinCount(workgroupDims, alphaBinCount);

		assert(paddedAlphaBinCount >= asset::IBlitUtilities::MinAlphaBinCount);
		const auto pipelineIndex = (paddedAlphaBinCount / asset::IBlitUtilities::MinAlphaBinCount) - 1;

		if (m_alphaTestPipelines[pipelineIndex][imageType])
			return m_alphaTestPipelines[pipelineIndex][imageType];

		auto specShader = createAlphaTestSpecializedShader(imageType, paddedAlphaBinCount);
		m_alphaTestPipelines[pipelineIndex][imageType] = m_device->createComputePipeline(nullptr, core::smart_refctd_ptr(m_blitPipelineLayout[EBT_COVERAGE_ADJUSTMENT]), std::move(specShader));

		return m_alphaTestPipelines[pipelineIndex][imageType];
	}

	// @param `outFormat` dictates encoding.
	core::smart_refctd_ptr<video::IGPUSpecializedShader> createNormalizationSpecializedShader(const asset::IImage::E_TYPE inImageType, const asset::E_FORMAT outFormat,
		const uint32_t alphaBinCount = asset::IBlitUtilities::DefaultAlphaBinCount);

	core::smart_refctd_ptr<video::IGPUComputePipeline> getNormalizationPipeline(const asset::IImage::E_TYPE imageType, const asset::E_FORMAT outFormat,
		const uint32_t alphaBinCount = asset::IBlitUtilities::DefaultAlphaBinCount)
	{
		const auto workgroupDims = getDefaultWorkgroupDims(imageType);
		const uint32_t paddedAlphaBinCount = getPaddedAlphaBinCount(workgroupDims, alphaBinCount);
		const SNormalizationCacheKey key = { imageType, paddedAlphaBinCount, outFormat };

		if (m_normalizationPipelines.find(key) == m_normalizationPipelines.end())
		{
			auto specShader = createNormalizationSpecializedShader(imageType, outFormat, paddedAlphaBinCount);
			m_normalizationPipelines[key] = m_device->createComputePipeline(nullptr, core::smart_refctd_ptr(m_blitPipelineLayout[EBT_COVERAGE_ADJUSTMENT]), std::move(specShader));
		}

		return m_normalizationPipelines[key];
	}

	template <typename KernelX, typename KernelY, typename KernelZ>
	core::smart_refctd_ptr<video::IGPUSpecializedShader> createBlitSpecializedShader(
		const asset::E_FORMAT outFormat,
		const asset::IImage::E_TYPE imageType,
		const core::vectorSIMDu32& inExtent,
		const core::vectorSIMDu32& outExtent,
		const asset::IBlitUtilities::E_ALPHA_SEMANTIC alphaSemantic,
		const KernelX& kernelX, const KernelY& kernelY, const KernelZ& kernelZ,
		const uint32_t workgroupSize = DefaultBlitWorkgroupSize,
		const uint32_t alphaBinCount = asset::IBlitUtilities::DefaultAlphaBinCount)
	{
		const auto workgroupDims = getDefaultWorkgroupDims(imageType);
		const auto paddedAlphaBinCount = getPaddedAlphaBinCount(workgroupDims, alphaBinCount);

		using blit_utils_t = asset::CBlitUtilities<KernelX, KernelY, KernelZ>;

		const auto scaledKernelX = blit_utils_t::constructScaledKernel(kernelX, inExtent, outExtent);
		const auto scaledKernelY = blit_utils_t::constructScaledKernel(kernelY, inExtent, outExtent);
		const auto scaledKernelZ = blit_utils_t::constructScaledKernel(kernelZ, inExtent, outExtent);

		std::ostringstream shaderSourceStream;
		shaderSourceStream
			<< "#version 460 core\n"
			<< "#define _NBL_GLSL_WORKGROUP_SIZE_X_ " << workgroupSize << "\n"
			<< "#define _NBL_GLSL_WORKGROUP_SIZE_Y_ " << 1 << "\n"
			<< "#define _NBL_GLSL_WORKGROUP_SIZE_Z_ " << 1 << "\n"
			<< "#define _NBL_GLSL_BLIT_DIM_COUNT_ " << static_cast<uint32_t>(imageType) + 1 << "\n"
			<< "#define _NBL_GLSL_BLIT_ALPHA_BIN_COUNT_ " << paddedAlphaBinCount << "\n";

		const auto castedFormat = getOutImageViewFormat(outFormat);
		const uint32_t outChannelCount = asset::getFormatChannelCount(outFormat);
		
		const char* glslFormatQualifier = asset::IGLSLCompiler::getStorageImageFormatQualifier(castedFormat);

		shaderSourceStream
			<< "#define _NBL_GLSL_BLIT_OUT_CHANNEL_COUNT_ " << outChannelCount << "\n"
			<< "#define _NBL_GLSL_BLIT_OUT_IMAGE_FORMAT_ " << glslFormatQualifier << "\n";

		const core::vectorSIMDf minSupport(-scaledKernelX.negative_support[0], -scaledKernelY.negative_support[1], -scaledKernelZ.negative_support[2]);
		const core::vectorSIMDf maxSupport(scaledKernelX.positive_support[0], scaledKernelY.positive_support[1], scaledKernelZ.positive_support[2]);

		const uint32_t smemFloatCount = m_availableSharedMemory/(sizeof(float)*outChannelCount);
		shaderSourceStream << "#define _NBL_GLSL_BLIT_SMEM_FLOAT_COUNT_ " << smemFloatCount << "\n";

		if (alphaSemantic == asset::IBlitUtilities::EAS_REFERENCE_OR_COVERAGE)
			shaderSourceStream << "#define _NBL_GLSL_BLIT_COVERAGE_SEMANTIC_\n";
		if (outFormat != castedFormat)
			shaderSourceStream << "#define _NBL_GLSL_BLIT_SOFTWARE_ENCODE_FORMAT_ " << outFormat << "\n";
		shaderSourceStream << "#include <nbl/builtin/glsl/blit/default_compute_blit.comp>\n";

		auto cpuShader = core::make_smart_refctd_ptr<asset::ICPUShader>(shaderSourceStream.str().c_str(), asset::IShader::ESS_COMPUTE, "CComputeBlit::createBlitSpecializedShader");
		auto gpuUnspecShader = m_device->createShader(std::move(cpuShader));
		auto specShader = m_device->createSpecializedShader(gpuUnspecShader.get(), { nullptr, nullptr, "main" });

		return specShader;
	}

	template <typename KernelX, typename KernelY, typename KernelZ>
	core::smart_refctd_ptr<video::IGPUComputePipeline> getBlitPipeline(
		const asset::E_FORMAT outFormat,
		const asset::IImage::E_TYPE imageType,
		const core::vectorSIMDu32& inExtent,
		const core::vectorSIMDu32& outExtent,
		const asset::IBlitUtilities::E_ALPHA_SEMANTIC alphaSemantic,
		const KernelX& kernelX, const KernelY& kernelY, const KernelZ& kernelZ,
		const uint32_t workgroupSize = DefaultBlitWorkgroupSize,
		const uint32_t alphaBinCount = asset::IBlitUtilities::DefaultAlphaBinCount)
	{
		const auto paddedAlphaBinCount = getPaddedAlphaBinCount(core::vectorSIMDu32(workgroupSize, 1, 1, 1), alphaBinCount);

		const SBlitCacheKey key =
		{
			.wgSize = workgroupSize,
			.imageType = imageType,
			.alphaBinCount = paddedAlphaBinCount,
			.outFormat = outFormat,
			.smemSize = m_availableSharedMemory,
			.coverageAdjustment = (alphaSemantic == asset::IBlitUtilities::EAS_REFERENCE_OR_COVERAGE)
		};

		if (m_blitPipelines.find(key) == m_blitPipelines.end())
		{
			const auto blitType = (alphaSemantic == asset::IBlitUtilities::EAS_REFERENCE_OR_COVERAGE) ? EBT_COVERAGE_ADJUSTMENT : EBT_REGULAR;

			auto specShader = createBlitSpecializedShader(
				outFormat,
				imageType,
				inExtent,
				outExtent,
				alphaSemantic,
				kernelX, kernelY, kernelZ,
				workgroupSize,
				paddedAlphaBinCount);

			m_blitPipelines[key] = m_device->createComputePipeline(nullptr, core::smart_refctd_ptr(m_blitPipelineLayout[blitType]), std::move(specShader));
		}

		return m_blitPipelines[key];
	}

	//! Returns the number of output texels produced by one workgroup, deciding factor is `m_availableSharedMemory`.
	//! @param outImageFormat is the format of output (of the blit step) image.
	//! If a normalization step is involved then this will be the same as the format of normalization step's input image --which may differ from the
	//! final output format, because we blit to a higher precision format for normalization.
	template <typename KernelX, typename KernelY, typename KernelZ>
	bool getOutputTexelsPerWorkGroup(
		core::vectorSIMDu32& outputTexelsPerWG,
		const core::vectorSIMDu32& inExtent,
		const core::vectorSIMDu32& outExtent,
		const asset::E_FORMAT outImageFormat,
		const asset::IImage::E_TYPE imageType,
		const KernelX& kernelX, const KernelY& kernelY, const KernelZ& kernelZ)
	{
		using blit_utils_t = asset::CBlitUtilities<KernelX, KernelY, KernelZ>;

		const auto scaledKernelX = blit_utils_t::constructScaledKernel(kernelX, inExtent, outExtent);
		const auto scaledKernelY = blit_utils_t::constructScaledKernel(kernelY, inExtent, outExtent);
		const auto scaledKernelZ = blit_utils_t::constructScaledKernel(kernelZ, inExtent, outExtent);

		core::vectorSIMDf scale = static_cast<core::vectorSIMDf>(inExtent).preciseDivision(static_cast<core::vectorSIMDf>(outExtent));

		core::vectorSIMDf minSupport(-scaledKernelX.negative_support[0], -scaledKernelY.negative_support[1], -scaledKernelZ.negative_support[2]);
		core::vectorSIMDf maxSupport(scaledKernelX.positive_support[0], scaledKernelY.positive_support[1], scaledKernelZ.positive_support[2]);

		outputTexelsPerWG = core::vectorSIMDu32(1, 1, 1, 1);
		size_t requiredSmem = getRequiredSharedMemorySize(outputTexelsPerWG, outExtent, imageType, minSupport, maxSupport, scale, asset::getFormatChannelCount(outImageFormat));
		bool failed = true;
		asset::IImage::E_TYPE minDimAxes[3] = { asset::IImage::ET_1D, asset::IImage::ET_2D, asset::IImage::ET_3D };
		while (requiredSmem < m_availableSharedMemory)
		{
			failed = false;

			std::sort(minDimAxes, minDimAxes + imageType+1, [&outputTexelsPerWG](const asset::IImage::E_TYPE a, const asset::IImage::E_TYPE b) -> bool { return outputTexelsPerWG[a] < outputTexelsPerWG[b]; });

			int i = 0;
			for (; i < imageType + 1; ++i)
			{
				const auto axis = minDimAxes[i];

				core::vectorSIMDu32 delta(0, 0, 0, 0);
				delta[axis] = 1;

				if (outputTexelsPerWG[axis] < outExtent[axis])
				{
					// Note: we use outImageFormat's channel count as opposed to its image view's format because, even in the event that they are different, we blit
					// as if we will be writing to a storage image of out
					requiredSmem = getRequiredSharedMemorySize(outputTexelsPerWG + delta, outExtent, imageType, minSupport, maxSupport, scale, asset::getFormatChannelCount(outImageFormat));

					if (requiredSmem <= m_availableSharedMemory)
					{
						outputTexelsPerWG += delta;
						break;
					}
				}
			}
			if (i == imageType + 1) // If we cannot find any axis to increment outputTexelsPerWG along, then break
				break;
		}

		return !failed;
	}

	template <typename KernelX, typename KernelY, typename KernelZ>
	inline void buildParameters(
		nbl_glsl_blit_parameters_t& outPC,
		const core::vectorSIMDu32& inImageExtent,
		const core::vectorSIMDu32& outImageExtent,
		const asset::IImage::E_TYPE imageType,
		const asset::E_FORMAT inImageFormat,
		const KernelX& kernelX, const KernelY& kernelY, const KernelZ& kernelZ,
		const uint32_t layersToBlit = 1, const float referenceAlpha = 0.f)
	{
		using blit_utils_t = asset::CBlitUtilities<KernelX, KernelY, KernelZ>;

		core::vectorSIMDu32 inDim(inImageExtent.x, inImageExtent.y, inImageExtent.z);
		core::vectorSIMDu32 outDim(outImageExtent.x, outImageExtent.y, outImageExtent.z);

		if (imageType < asset::IImage::ET_3D)
		{
			inDim.z = layersToBlit;
			outDim.z = layersToBlit;
		}

		constexpr auto MaxImageDim = 1 << 16;
		const auto maxImageDims = core::vectorSIMDu32(MaxImageDim, MaxImageDim, MaxImageDim, MaxImageDim);
		assert((inDim < maxImageDims) && (outDim < maxImageDims));

		outPC.dims.x = (outDim.x << 16) | inDim.x;
		outPC.dims.y = (outDim.y << 16) | inDim.y;
		outPC.dims.z = (outDim.z << 16) | inDim.z;

		const auto scaledKernelX = blit_utils_t::constructScaledKernel(kernelX, inImageExtent, outImageExtent);
		const auto scaledKernelY = blit_utils_t::constructScaledKernel(kernelY, inImageExtent, outImageExtent);
		const auto scaledKernelZ = blit_utils_t::constructScaledKernel(kernelZ, inImageExtent, outImageExtent);

		core::vectorSIMDf scale = static_cast<core::vectorSIMDf>(inImageExtent).preciseDivision(static_cast<core::vectorSIMDf>(outImageExtent));

		const core::vectorSIMDf minSupport(-scaledKernelX.negative_support[0], -scaledKernelY.negative_support[1], -scaledKernelZ.negative_support[2]);
		const core::vectorSIMDf maxSupport(scaledKernelX.positive_support[0], scaledKernelY.positive_support[1], scaledKernelZ.positive_support[2]);

		core::vectorSIMDu32 outputTexelsPerWG;
		getOutputTexelsPerWorkGroup(outputTexelsPerWG, inImageExtent, outImageExtent, inImageFormat, imageType, kernelX, kernelY, kernelZ);
		const auto preloadRegion = getPreloadRegion(outputTexelsPerWG, imageType, minSupport, maxSupport, scale);

		outPC.secondScratchOffset = core::max(preloadRegion.x * preloadRegion.y * preloadRegion.z, outputTexelsPerWG.x*outputTexelsPerWG.y*preloadRegion.z);
		outPC.iterationRegionXPrefixProducts = {outputTexelsPerWG.x, outputTexelsPerWG.x*preloadRegion.y, outputTexelsPerWG.x*preloadRegion.y*preloadRegion.z};
		outPC.referenceAlpha = referenceAlpha;
		outPC.fScale = {scale.x, scale.y, scale.z};
		outPC.inPixelCount = inImageExtent.x * inImageExtent.y * inImageExtent.z;
		outPC.negativeSupport.x = minSupport.x; outPC.negativeSupport.y = minSupport.y; outPC.negativeSupport.z = minSupport.z;
		outPC.outPixelCount = outImageExtent.x*outImageExtent.y*outImageExtent.z;

		const core::vectorSIMDi32 windowDim = core::max(blit_utils_t::getRealWindowSize(imageType, scaledKernelX, scaledKernelY, scaledKernelZ), core::vectorSIMDi32(1, 1, 1, 1));
		assert(windowDim < maxImageDims);

		const core::vectorSIMDu32 phaseCount = asset::IBlitUtilities::getPhaseCount(inImageExtent, outImageExtent, imageType);
		assert(phaseCount < maxImageDims);

		outPC.windowDimPhaseCount.x = (phaseCount.x << 16) | windowDim.x;
		outPC.windowDimPhaseCount.y = (phaseCount.y << 16) | windowDim.y;
		outPC.windowDimPhaseCount.z = (phaseCount.z << 16) | windowDim.z;

		outPC.kernelWeightsOffsetY = phaseCount.x*windowDim.x;
		outPC.iterationRegionYPrefixProducts = {outputTexelsPerWG.y, outputTexelsPerWG.y*outputTexelsPerWG.x, outputTexelsPerWG.y*outputTexelsPerWG.x*preloadRegion.z};
		outPC.kernelWeightsOffsetZ = outPC.kernelWeightsOffsetY + phaseCount.y*windowDim.y;
		outPC.iterationRegionZPrefixProducts = {outputTexelsPerWG.y, outputTexelsPerWG.y*outputTexelsPerWG.x, outputTexelsPerWG.y*outputTexelsPerWG.x*outputTexelsPerWG.z};
		outPC.outputTexelsPerWGZ = outputTexelsPerWG.z;
		outPC.preloadRegion = { preloadRegion.x, preloadRegion.y, preloadRegion.z };
	}

	static inline void buildAlphaTestDispatchInfo(dispatch_info_t& outDispatchInfo, const core::vectorSIMDu32& inImageExtent, const asset::IImage::E_TYPE imageType, const uint32_t layersToBlit = 1)
	{
		const core::vectorSIMDu32 workgroupDims = getDefaultWorkgroupDims(imageType);
		const core::vectorSIMDu32 workgroupCount = (inImageExtent + workgroupDims - core::vectorSIMDu32(1u, 1u, 1u, 1u)) / workgroupDims;

		outDispatchInfo.wgCount[0] = workgroupCount.x;
		outDispatchInfo.wgCount[1] = workgroupCount.y;
		if (imageType < asset::IImage::ET_3D)
			outDispatchInfo.wgCount[2] = layersToBlit;
		else
			outDispatchInfo.wgCount[2] = workgroupCount[2];
	}

	template <typename KernelX, typename KernelY, typename KernelZ>
	inline void buildBlitDispatchInfo(
		dispatch_info_t& dispatchInfo,
		const core::vectorSIMDu32& inImageExtent,
		const core::vectorSIMDu32& outImageExtent,
		const asset::E_FORMAT inImageFormat,
		const asset::IImage::E_TYPE imageType,
		const KernelX& kernelX, const KernelY& kernelY, const KernelZ& kernelZ,
		const uint32_t workgroupSize = DefaultBlitWorkgroupSize,
		const uint32_t layersToBlit = 1)
	{
		core::vectorSIMDu32 outputTexelsPerWG;
		getOutputTexelsPerWorkGroup(outputTexelsPerWG, inImageExtent, outImageExtent, inImageFormat, imageType, kernelX, kernelY, kernelZ);
		const auto wgCount = (outImageExtent + outputTexelsPerWG - core::vectorSIMDu32(1, 1, 1)) / core::vectorSIMDu32(outputTexelsPerWG.x, outputTexelsPerWG.y, outputTexelsPerWG.z, 1);

		dispatchInfo.wgCount[0] = wgCount[0];
		dispatchInfo.wgCount[1] = wgCount[1];
		if (imageType < asset::IImage::ET_3D)
			dispatchInfo.wgCount[2] = layersToBlit;
		else
			dispatchInfo.wgCount[2] = wgCount[2];
	}

	static inline void buildNormalizationDispatchInfo(dispatch_info_t& outDispatchInfo, const core::vectorSIMDu32& outImageExtent, const asset::IImage::E_TYPE imageType, const uint32_t layersToBlit = 1)
	{
		const core::vectorSIMDu32 workgroupDims = getDefaultWorkgroupDims(imageType);
		const core::vectorSIMDu32 workgroupCount = (outImageExtent + workgroupDims - core::vectorSIMDu32(1u, 1u, 1u, 1u)) / workgroupDims;

		outDispatchInfo.wgCount[0] = workgroupCount.x;
		outDispatchInfo.wgCount[1] = workgroupCount.y;
		if (imageType < asset::IImage::ET_3D)
			outDispatchInfo.wgCount[2] = layersToBlit;
		else
			outDispatchInfo.wgCount[2] = workgroupCount[2];
	}

	static inline core::vectorSIMDu32 getDefaultWorkgroupDims(const asset::IImage::E_TYPE imageType)
	{
		switch (imageType)
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

	static inline size_t getCoverageAdjustmentScratchSize(const asset::IBlitUtilities::E_ALPHA_SEMANTIC alphaSemantic, const asset::IImage::E_TYPE imageType, const uint32_t alphaBinCount, const uint32_t layersToBlit)
	{
		if (alphaSemantic != asset::IBlitUtilities::EAS_REFERENCE_OR_COVERAGE)
			return 0;

		const auto workgroupDims = getDefaultWorkgroupDims(imageType);
		const auto paddedAlphaBinCount = getPaddedAlphaBinCount(workgroupDims, alphaBinCount);
		const auto requiredSize = (sizeof(uint32_t) + paddedAlphaBinCount * sizeof(uint32_t)) * layersToBlit;
		return requiredSize;
	}

	bool updateDescriptorSet(
		video::IGPUDescriptorSet* blitDS,
		video::IGPUDescriptorSet* kernelWeightsDS,
		core::smart_refctd_ptr<video::IGPUImageView> inImageView,
		core::smart_refctd_ptr<video::IGPUImageView> outImageView,
		core::smart_refctd_ptr<video::IGPUBuffer> coverageAdjustmentScratchBuffer,
		core::smart_refctd_ptr<video::IGPUBufferView> kernelWeightsUTB,
		const asset::ISampler::E_TEXTURE_CLAMP wrapU = asset::ISampler::ETC_CLAMP_TO_EDGE,
		const asset::ISampler::E_TEXTURE_CLAMP wrapV = asset::ISampler::ETC_CLAMP_TO_EDGE,
		const asset::ISampler::E_TEXTURE_CLAMP wrapW = asset::ISampler::ETC_CLAMP_TO_EDGE,
		const asset::ISampler::E_TEXTURE_BORDER_COLOR borderColor = asset::ISampler::ETBC_FLOAT_OPAQUE_BLACK)
	{
		constexpr auto MAX_DESCRIPTOR_COUNT = 3;

		auto updateDS = [this, coverageAdjustmentScratchBuffer](video::IGPUDescriptorSet* ds, video::IGPUDescriptorSet::SDescriptorInfo* infos) -> bool
		{
			const auto& bindings = ds->getLayout()->getBindings();
			if ((bindings.size() == 3) && !coverageAdjustmentScratchBuffer)
				return false;

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

			m_device->updateDescriptorSets(bindings.size(), writes, 0u, nullptr);

			return true;
		};

		if (blitDS)
		{
			if (!inImageView || !outImageView)
				return false;

			video::IGPUDescriptorSet::SDescriptorInfo infos[MAX_DESCRIPTOR_COUNT] = {};

			if (!samplers[wrapU][wrapV][wrapW][borderColor])
			{
				video::IGPUSampler::SParams params = {};
				params.TextureWrapU = wrapU;
				params.TextureWrapV = wrapV;
				params.TextureWrapW = wrapW;
				params.BorderColor = borderColor;
				params.MinFilter = asset::ISampler::ETF_NEAREST;
				params.MaxFilter = asset::ISampler::ETF_NEAREST;
				params.MipmapMode = asset::ISampler::ESMM_NEAREST;
				params.AnisotropicFilter = 0u;
				params.CompareEnable = 0u;
				params.CompareFunc = asset::ISampler::ECO_ALWAYS;

				samplers[wrapU][wrapV][wrapW][borderColor] = m_device->createSampler(params);
				if (!samplers[wrapU][wrapV][wrapW][borderColor])
					return false;
			}
			
			infos[0].desc = inImageView;
			infos[0].image.imageLayout = asset::EIL_SHADER_READ_ONLY_OPTIMAL;
			infos[0].image.sampler = samplers[wrapU][wrapV][wrapW][borderColor];

			infos[1].desc = outImageView;
			infos[1].image.imageLayout = asset::EIL_GENERAL;
			infos[1].image.sampler = nullptr;

			if (coverageAdjustmentScratchBuffer)
			{
				infos[2].desc = coverageAdjustmentScratchBuffer;
				infos[2].buffer.offset = 0;
				infos[2].buffer.size = coverageAdjustmentScratchBuffer->getSize();
			}

			if (!updateDS(blitDS, infos))
				return false;
		}

		if (kernelWeightsDS)
		{
			video::IGPUDescriptorSet::SDescriptorInfo info = {};
			info.desc = kernelWeightsUTB;
			info.buffer.offset = 0ull;
			info.buffer.size = kernelWeightsUTB->getUnderlyingBuffer()->getSize();

			if (!updateDS(kernelWeightsDS, &info))
				return false;
		}

		return true;
	}

	//! User is responsible for the memory barriers between previous writes and the first
	//! dispatch on the input image, and future reads of output image and the last dispatch.
	template <typename KernelX, typename KernelY, typename KernelZ>
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
		const KernelX& kernelX,
		const KernelY& kernelY,
		const KernelZ& kernelZ,
		const uint32_t layersToBlit = 1,
		core::smart_refctd_ptr<video::IGPUBuffer> coverageAdjustmentScratchBuffer = nullptr,
		const float referenceAlpha = 0.f,
		const uint32_t alphaBinCount = asset::IBlitUtilities::DefaultAlphaBinCount,
		const uint32_t workgroupSize = DefaultBlitWorkgroupSize)
	{
		const core::vectorSIMDu32 outImageExtent(normalizationInImage->getCreationParameters().extent.width, normalizationInImage->getCreationParameters().extent.height, normalizationInImage->getCreationParameters().extent.depth, 1u);

		nbl_glsl_blit_parameters_t pushConstants;
		buildParameters(pushConstants, inImageExtent, outImageExtent, inImageType, inImageFormat, kernelX, kernelY, kernelZ, layersToBlit, referenceAlpha);

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
			buildBlitDispatchInfo(dispatchInfo, inImageExtent, outImageExtent, inImageFormat, inImageType, kernelX, kernelY, kernelZ, workgroupSize, layersToBlit);

			video::IGPUDescriptorSet* ds_raw[] = { blitDS, blitWeightsDS };
			cmdbuf->bindDescriptorSets(asset::EPBP_COMPUTE, blitPipeline->getLayout(), 0, 2, ds_raw);
			cmdbuf->bindComputePipeline(blitPipeline);
			dispatchHelper(cmdbuf, blitPipeline->getLayout(), pushConstants, dispatchInfo);
		}

		// After this dispatch ends and finishes writing to outImage, normalize outImage
		if (alphaSemantic == asset::IBlitUtilities::EAS_REFERENCE_OR_COVERAGE)
		{
			dispatch_info_t dispatchInfo;
			buildNormalizationDispatchInfo(dispatchInfo, outImageExtent, inImageType, layersToBlit);

			assert(coverageAdjustmentScratchBuffer);

			// Memory dependency to ensure the alpha test pass has finished writing to alphaTestCounterBuffer
			video::IGPUCommandBuffer::SBufferMemoryBarrier alphaTestBarrier = {};
			alphaTestBarrier.barrier.srcAccessMask = asset::EAF_SHADER_WRITE_BIT;
			alphaTestBarrier.barrier.dstAccessMask = asset::EAF_SHADER_READ_BIT;
			alphaTestBarrier.srcQueueFamilyIndex = ~0u;
			alphaTestBarrier.dstQueueFamilyIndex = ~0u;
			alphaTestBarrier.buffer = coverageAdjustmentScratchBuffer;
			alphaTestBarrier.size = coverageAdjustmentScratchBuffer->getSize();
			alphaTestBarrier.offset = 0;

			// Memory dependency to ensure that the previous compute pass has finished writing to the output image,
			// also transitions the layout of said image: GENERAL -> SHADER_READ_ONLY_OPTIMAL
			video::IGPUCommandBuffer::SImageMemoryBarrier readyForNorm = {};
			readyForNorm.barrier.srcAccessMask = asset::EAF_SHADER_WRITE_BIT;
			readyForNorm.barrier.dstAccessMask = static_cast<asset::E_ACCESS_FLAGS>(asset::EAF_SHADER_READ_BIT);
			readyForNorm.oldLayout = asset::EIL_GENERAL;
			readyForNorm.newLayout = asset::EIL_SHADER_READ_ONLY_OPTIMAL;
			readyForNorm.srcQueueFamilyIndex = ~0u;
			readyForNorm.dstQueueFamilyIndex = ~0u;
			readyForNorm.image = normalizationInImage;
			readyForNorm.subresourceRange.aspectMask = asset::IImage::EAF_COLOR_BIT;
			readyForNorm.subresourceRange.levelCount = 1u;
			readyForNorm.subresourceRange.layerCount = normalizationInImage->getCreationParameters().arrayLayers;
			cmdbuf->pipelineBarrier(asset::EPSF_COMPUTE_SHADER_BIT, asset::EPSF_COMPUTE_SHADER_BIT, asset::EDF_NONE, 0u, nullptr, 1u, &alphaTestBarrier, 1u, &readyForNorm);

			cmdbuf->bindDescriptorSets(asset::EPBP_COMPUTE, normalizationPipeline->getLayout(), 0u, 1u, &normalizationDS);
			cmdbuf->bindComputePipeline(normalizationPipeline);
			dispatchHelper(cmdbuf, normalizationPipeline->getLayout(), pushConstants, dispatchInfo);
		}
	}

	//! WARNING: This function blocks and stalls the GPU!
	template <typename KernelX, typename KernelY, typename KernelZ, typename... Args>
	inline void blit(video::IGPUQueue* computeQueue, Args&&... args)
	{
		auto cmdpool = m_device->createCommandPool(computeQueue->getFamilyIndex(), video::IGPUCommandPool::ECF_NONE);
		core::smart_refctd_ptr<video::IGPUCommandBuffer> cmdbuf;
		m_device->createCommandBuffers(cmdpool.get(), video::IGPUCommandBuffer::EL_PRIMARY, 1u, &cmdbuf);

		auto fence = m_device->createFence(video::IGPUFence::ECF_UNSIGNALED);

		cmdbuf->begin(video::IGPUCommandBuffer::EU_ONE_TIME_SUBMIT_BIT);
		blit(cmdbuf.get(), std::forward<Args>(args)...);
		cmdbuf->end();

		video::IGPUQueue::SSubmitInfo submitInfo = {};
		submitInfo.commandBufferCount = 1u;
		submitInfo.commandBuffers = &cmdbuf.get();
		computeQueue->submit(1u, &submitInfo, fence.get());

		m_device->blockForFences(1u, &fence.get());
	}

	//! Returns the original format if supports STORAGE_IMAGE otherwise returns a format in its compat class which supports STORAGE_IMAGE.
	inline asset::E_FORMAT getOutImageViewFormat(const asset::E_FORMAT format)
	{
		const auto& formatUsages = m_device->getPhysicalDevice()->getImageFormatUsagesOptimal(format);

		if (formatUsages.storageImage)
		{
			return format;
		}
		else
		{
			const asset::E_FORMAT compatFormat = getCompatClassFormat(format);

			const auto& compatClassFormatUsages = m_device->getPhysicalDevice()->getImageFormatUsagesOptimal(compatFormat);
			if (!compatClassFormatUsages.storageImage)
				return asset::EF_UNKNOWN;
			else
				return compatFormat;
		}
	}

	static inline asset::E_FORMAT getCoverageAdjustmentIntermediateFormat(const asset::E_FORMAT format)
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

	core::smart_refctd_ptr<video::IGPUComputePipeline> m_alphaTestPipelines[asset::IBlitUtilities::MaxAlphaBinCount / asset::IBlitUtilities::MinAlphaBinCount][asset::IImage::ET_COUNT] = { nullptr };

	struct SNormalizationCacheKey
	{
		asset::IImage::E_TYPE imageType;
		uint32_t alphaBinCount;
		asset::E_FORMAT outFormat;

		inline bool operator==(const SNormalizationCacheKey& other) const
		{
			return (imageType == other.imageType) && (alphaBinCount == other.alphaBinCount) && (outFormat == other.outFormat);
		}
	};
	struct SNormalizationCacheHash
	{
		inline size_t operator() (const SNormalizationCacheKey& key) const
		{
			return
				std::hash<decltype(key.imageType)>{}(key.imageType) ^
				std::hash<decltype(key.alphaBinCount)>{}(key.alphaBinCount) ^
				std::hash<decltype(key.outFormat)>{}(key.outFormat);
		}
	};
	core::unordered_map<SNormalizationCacheKey, core::smart_refctd_ptr<video::IGPUComputePipeline>, SNormalizationCacheHash> m_normalizationPipelines;

	struct SBlitCacheKey
	{
		uint32_t wgSize;
		asset::IImage::E_TYPE imageType;
		uint32_t alphaBinCount;
		asset::E_FORMAT outFormat;
		uint32_t smemSize;
		bool coverageAdjustment;

		inline bool operator==(const SBlitCacheKey& other) const
		{
			return (wgSize == other.wgSize) && (imageType == other.imageType) && (alphaBinCount == other.alphaBinCount) && (outFormat == other.outFormat)
				&& (smemSize == other.smemSize) && (coverageAdjustment == other.coverageAdjustment);
		}
	};
	struct SBlitCacheHash
	{
		inline size_t operator()(const SBlitCacheKey& key) const
		{
			return
				std::hash<decltype(key.wgSize)>{}(key.wgSize) ^
				std::hash<decltype(key.imageType)>{}(key.imageType) ^
				std::hash<decltype(key.alphaBinCount)>{}(key.alphaBinCount) ^
				std::hash<decltype(key.outFormat)>{}(key.outFormat) ^
				std::hash<decltype(key.smemSize)>{}(key.smemSize) ^
				std::hash<decltype(key.coverageAdjustment)>{}(key.coverageAdjustment);
		}
	};
	core::unordered_map<SBlitCacheKey, core::smart_refctd_ptr<video::IGPUComputePipeline>, SBlitCacheHash> m_blitPipelines;

	uint32_t m_availableSharedMemory;
	core::smart_refctd_ptr<video::ILogicalDevice> m_device;

	core::smart_refctd_ptr<video::IGPUSampler> samplers[video::IGPUSampler::ETC_COUNT][video::IGPUSampler::ETC_COUNT][video::IGPUSampler::ETC_COUNT][video::IGPUSampler::ETBC_COUNT] = { nullptr };

	CComputeBlit(core::smart_refctd_ptr<video::ILogicalDevice>&& logicalDevice) : m_device(std::move(logicalDevice)) {}

	static inline void dispatchHelper(video::IGPUCommandBuffer* cmdbuf, const video::IGPUPipelineLayout* pipelineLayout, const nbl_glsl_blit_parameters_t& pushConstants, const dispatch_info_t& dispatchInfo)
	{
		cmdbuf->pushConstants(pipelineLayout, asset::IShader::ESS_COMPUTE, 0u, sizeof(nbl_glsl_blit_parameters_t), &pushConstants);
		cmdbuf->dispatch(dispatchInfo.wgCount[0], dispatchInfo.wgCount[1], dispatchInfo.wgCount[2]);
	}

	core::smart_refctd_ptr<video::IGPUDescriptorSetLayout> createDSLayout(const uint32_t descriptorCount, const asset::E_DESCRIPTOR_TYPE* descriptorTypes, video::ILogicalDevice* logicalDevice) const
	{
		constexpr uint32_t MAX_DESCRIPTOR_COUNT = 5;
		assert(descriptorCount < MAX_DESCRIPTOR_COUNT);

		video::IGPUDescriptorSetLayout::SBinding bindings[MAX_DESCRIPTOR_COUNT] = {};

		for (uint32_t i = 0u; i < descriptorCount; ++i)
		{
			bindings[i].binding = i;
			bindings[i].count = 1u;
			bindings[i].stageFlags = asset::IShader::ESS_COMPUTE;
			bindings[i].type = descriptorTypes[i];
		}

		auto dsLayout = logicalDevice->createDescriptorSetLayout(bindings, bindings + descriptorCount);
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

	//! This calculates the inclusive upper bound on the preload region i.e. it will be reachable for some cases. For the rest it will be bigger
	//! by a pixel in each dimension.
	//! Used to size the shared memory.
	inline core::vectorSIMDu32 getPreloadRegion(
		const core::vectorSIMDu32& outputTexelsPerWG,
		const asset::IImage::E_TYPE imageType,
		const core::vectorSIMDf& scaledMinSupport,
		const core::vectorSIMDf& scaledMaxSupport,
		const core::vectorSIMDf& scale)
	{
		core::vectorSIMDu32 preloadRegion = core::vectorSIMDu32(core::floor((scaledMaxSupport - scaledMinSupport) + core::vectorSIMDf(outputTexelsPerWG - core::vectorSIMDu32(1, 1, 1, 1)) * scale)) + core::vectorSIMDu32(1, 1, 1, 1);

		// Set the unused dimensions to 1 to avoid weird behaviours with scaled kernels
		for (auto axis = imageType + 1; axis < 3u; ++axis)
			preloadRegion[axis] = 1;

		return preloadRegion;
	}

	//! Query shared memory size for a given `outputTexelsPerWG`.
	size_t getRequiredSharedMemorySize(
		const core::vectorSIMDu32& outputTexelsPerWG,
		const core::vectorSIMDu32& outExtent,
		const asset::IImage::E_TYPE imageType,
		const core::vectorSIMDf& scaledMinSupport,
		const core::vectorSIMDf& scaledMaxSupport,
		const core::vectorSIMDf& scale,
		const uint32_t channelCount)
	{
		const auto preloadRegion = getPreloadRegion(outputTexelsPerWG, imageType, scaledMinSupport, scaledMaxSupport, scale);

		const size_t requiredSmem = (core::max(preloadRegion.x * preloadRegion.y * preloadRegion.z, outputTexelsPerWG.x * outputTexelsPerWG.y * preloadRegion.z) + outputTexelsPerWG.x * preloadRegion.y * preloadRegion.z)*channelCount*sizeof(float);
		return requiredSmem;
	};

	static inline uint32_t getPaddedAlphaBinCount(const core::vectorSIMDu32& workgroupDims, const uint32_t oldAlphaBinCount)
	{
		// For the normalization shader, it should be that:
		//	alphaBinCount = k*workGroupSize, k is integer, k >= 1, 
		assert(workgroupDims.x != 0 && workgroupDims.y != 0 && workgroupDims.z != 0);
		const auto wgSize = workgroupDims.x * workgroupDims.y * workgroupDims.z;
		const auto paddedAlphaBinCount = core::roundUp(oldAlphaBinCount, wgSize);
		return paddedAlphaBinCount;
	}
};
}

#define _NBL_VIDEO_C_COMPUTE_BLIT_H_INCLUDED_
#endif
