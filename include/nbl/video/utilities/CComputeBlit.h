#ifndef _NBL_VIDEO_C_COMPUTE_BLIT_H_INCLUDED_
#define _NBL_VIDEO_C_COMPUTE_BLIT_H_INCLUDED_

#include "nbl/asset/filters/CBlitUtilities.h"
#include "nbl/builtin/hlsl/format.hlsl"
#include "nbl/builtin/hlsl/blit/parameters.hlsl"

namespace nbl::video
{

class NBL_API2 CComputeBlit : public core::IReferenceCounted
{
	public:
		// Coverage adjustment needs alpha to be stored in HDR with high precision
		static inline asset::E_FORMAT getCoverageAdjustmentIntermediateFormat(const asset::E_FORMAT format)
		{
			using namespace nbl::asset;

			if (getFormatChannelCount(format)<4 || isIntegerFormat(format))
				return EF_UNKNOWN;
			
			const float precision = asset::getFormatPrecision(format,3,0.f);
			if (isFloatingPointFormat(format))
			{
				if (precision<std::numeric_limits<hlsl::float16_t>::min())
					return EF_R32_SFLOAT;
				return EF_R16_SFLOAT;
			}
			else
			{
				const bool sign = isSignedFormat(format);
				// there's no 24 or 32 bit normalized formats
				if (precision*((sign ? (0x1u<<16):(0x1u<<15))-1)<1.f)
					return EF_R32_SFLOAT;

				if (precision<1.f/255.f)
					return sign ? EF_R8_SNORM:EF_R8_UNORM;
				else
					return sign ? EF_R16_SNORM:EF_R16_UNORM;
			}
		}

		// ctor
		CComputeBlit(
			core::smart_refctd_ptr<ILogicalDevice>&& logicalDevice,
			core::smart_refctd_ptr<asset::IShaderCompiler::CCache>&& cache=nullptr,
			core::smart_refctd_ptr<system::ILogger>&& logger=nullptr
		);

		// if you set the balues too small, we'll correct them ourselves anyway
		struct STask
		{
			uint32_t workgroupSizeLog2 : 4 = 0;
			// the TRUE output format, not the storage view format you might manually encode into
			hlsl::format::TexelBlockFormat outputFormat : 8 = hlsl::format::TexelBlockFormat::TBF_UNKNOWN;
			uint32_t sharedMemoryPerInvocation : 6 = 0;
			uint32_t unused : 14 = 0;
		};
		
		//! Returns the original format if supports STORAGE_IMAGE otherwise returns a format in its compat class which supports STORAGE_IMAGE.
		inline asset::E_FORMAT getOutputViewFormat(const asset::E_FORMAT format)
		{
			const auto& usages = m_device->getPhysicalDevice()->getImageFormatUsagesOptimalTiling();
			const auto& formatUsages = usages[format];

			if (formatUsages.storageImage)
			{
				return format;
			}
			else
			{
				const auto formatNewEnum = static_cast<hlsl::format::TexelBlockFormat>(format);
				const auto compatFormatNewEnum = hlsl::format::getTraits(formatNewEnum).ClassTraits.RawAccessViewFormat;
				const auto compatFormat = static_cast<asset::E_FORMAT>(compatFormatNewEnum);
				assert(compatFormat!=asset::EF_UNKNOWN); // if you hit this, then time to implement missing traits and switch-cases
				const auto& compatClassFormatUsages = usages[compatFormat];
				if (!compatClassFormatUsages.storageImage)
					return asset::EF_UNKNOWN;
				else
					return compatFormat;
			}
		}

#if 0
		// @param `alphaBinCount` is only required to size the histogram present in the default nbl_glsl_blit_AlphaStatistics_t in default_compute_common.comp
		core::smart_refctd_ptr<video::IGPUShader> createAlphaTestSpecializedShader(const asset::IImage::E_TYPE inImageType, const uint32_t alphaBinCount = asset::IBlitUtilities::DefaultAlphaBinCount);

		core::smart_refctd_ptr<video::IGPUComputePipeline> getAlphaTestPipeline(const uint32_t alphaBinCount, const asset::IImage::E_TYPE imageType)
		{
			const auto workgroupDims = getDefaultWorkgroupDims(imageType);
			const auto paddedAlphaBinCount = getPaddedAlphaBinCount(workgroupDims, alphaBinCount);

			assert(paddedAlphaBinCount >= asset::IBlitUtilities::MinAlphaBinCount);
			const auto pipelineIndex = (paddedAlphaBinCount / asset::IBlitUtilities::MinAlphaBinCount) - 1;

			if (m_alphaTestPipelines[pipelineIndex][imageType])
				return m_alphaTestPipelines[pipelineIndex][imageType];

			auto specShader = createAlphaTestSpecializedShader(imageType, paddedAlphaBinCount);
			IGPUComputePipeline::SCreationParams creationParams;
			creationParams.shader.shader = specShader.get();
			creationParams.shader.entryPoint = "main";
			creationParams.layout = m_blitPipelineLayout[EBT_COVERAGE_ADJUSTMENT].get();
			assert(m_device->createComputePipelines(nullptr, { &creationParams, &creationParams + 1 }, &m_alphaTestPipelines[pipelineIndex][imageType]));

			return m_alphaTestPipelines[pipelineIndex][imageType];
		}

		// @param `outFormat` dictates encoding.
		core::smart_refctd_ptr<video::IGPUShader> createNormalizationSpecializedShader(const asset::IImage::E_TYPE inImageType, const asset::E_FORMAT outFormat,
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
				IGPUComputePipeline::SCreationParams creationParams;
				creationParams.shader.shader = specShader.get();
				creationParams.shader.entryPoint = "main";
				creationParams.layout = m_blitPipelineLayout[EBT_COVERAGE_ADJUSTMENT].get();
				assert(m_device->createComputePipelines(nullptr, { &creationParams, &creationParams + 1 }, &m_normalizationPipelines[key]));
			}

			return m_normalizationPipelines[key];
		}

		template <typename BlitUtilities>
		core::smart_refctd_ptr<video::IGPUShader> createBlitSpecializedShader(
			const asset::E_FORMAT									outFormat,
			const asset::IImage::E_TYPE								imageType,
			const core::vectorSIMDu32& inExtent,
			const core::vectorSIMDu32& outExtent,
			const asset::IBlitUtilities::E_ALPHA_SEMANTIC			alphaSemantic,
			const typename BlitUtilities::convolution_kernels_t&	kernels,
			const uint32_t											workgroupSize = 0,
			const uint32_t											alphaBinCount = asset::IBlitUtilities::DefaultAlphaBinCount)
		{
			if (workgroupSize==0)
				workgroupSize = m_device->getPhysicalDevice()->getLimits().maxWorkgroupSize;

			const auto workgroupDims = getDefaultWorkgroupDims(imageType);
			const auto paddedAlphaBinCount = getPaddedAlphaBinCount(workgroupDims, alphaBinCount);

			const uint32_t outChannelCount = asset::getFormatChannelCount(outFormat);
			const uint32_t smemFloatCount = m_availableSharedMemory / (sizeof(float) * outChannelCount);
			const uint32_t blitDimCount = static_cast<uint32_t>(imageType) + 1;

			const auto castedFormat = getOutImageViewFormat(outFormat);
			assert(outFormat == castedFormat);
			const char* formatQualifier = asset::CHLSLCompiler::getStorageImageFormatQualifier(castedFormat);

			std::ostringstream shaderSourceStream;
			shaderSourceStream
				<< "#include \"nbl/builtin/hlsl/blit/common.hlsl\"\n"
				   "#include \"nbl/builtin/hlsl/blit/parameters.hlsl\"\n"
				   "#include \"nbl/builtin/hlsl/blit/compute_blit.hlsl\"\n";

			shaderSourceStream
				<< "typedef nbl::hlsl::blit::consteval_parameters_t<" << workgroupSize << ", 1, 1, " << smemFloatCount << ", "
				<< outChannelCount << ", " << blitDimCount << ", " << paddedAlphaBinCount << "> ceval_params_t;\n";

			shaderSourceStream
				<< "[[vk::combinedImageSampler]] [[vk::binding(0, 0)]]\n"
				   "nbl::hlsl::blit::impl::dim_to_image_properties<ceval_params_t::BlitDimCount>::combined_sampler_t inCS;\n"
				   "[[vk::combinedImageSampler]] [[vk::binding(0, 0)]]\n"
			       "SamplerState inSamp;\n"

				   "[[vk::image_format(\""<< formatQualifier << "\")]]\n"
				   "[[vk::binding(1, 0)]]\n"
				   "nbl::hlsl::blit::impl::dim_to_image_properties<ceval_params_t::BlitDimCount>::image_t outImg;\n"

				   "[[vk::binding(0, 1)]] Buffer<float32_t4> kernelWeights;\n"
			       "[[vk::push_constant]] nbl::hlsl::blit::parameters_t params;"
				   "groupshared float32_t sMem[" << m_availableSharedMemory / sizeof(float) << "];\n";
				
			if (alphaSemantic == asset::IBlitUtilities::EAS_REFERENCE_OR_COVERAGE)
			{
				shaderSourceStream
					<< "[[vk::binding(2 , 0)]] RWStructuredBuffer<uint32_t> statsBuff;\n"
					   "struct HistogramAccessor { void atomicAdd(uint32_t wgID, uint32_t bucket, uint32_t v) { InterlockedAdd(statsBuff[wgID * (ceval_params_t::AlphaBinCount + 1) + bucket], v); } };\n";
			}
			else
			{
				shaderSourceStream << "struct HistogramAccessor { void atomicAdd(uint32_t wgID, uint32_t bucket, uint32_t v) { } };\n";
			}

			shaderSourceStream
				<< "struct KernelWeightsAccessor { float32_t4 get(float32_t idx) { return kernelWeights[idx]; } };\n"
				   "struct SharedAccessor { float32_t get(float32_t idx) { return sMem[idx]; } void set(float32_t idx, float32_t val) { sMem[idx] = val; } };\n"
				   "struct InCSAccessor { float32_t4 get(float32_t3 c, uint32_t l) { return inCS.SampleLevel(inSamp, nbl::hlsl::blit::impl::dim_to_image_properties<ceval_params_t::BlitDimCount>::getIndexCoord<float32_t>(c, l), 0); } };\n"
				   "struct OutImgAccessor { void set(int32_t3 c, uint32_t l, float32_t4 v) { outImg[nbl::hlsl::blit::impl::dim_to_image_properties<ceval_params_t::BlitDimCount>::getIndexCoord<int32_t>(c, l)] = v; } };\n"

				   "[numthreads(ceval_params_t::WorkGroupSize, 1, 1)]\n"
				   "void main(uint32_t3 workGroupID : SV_GroupID, uint32_t localInvocationIndex : SV_GroupIndex)\n"
				   "{\n"
				   "	nbl::hlsl::blit::compute_blit_t<ceval_params_t> blit = nbl::hlsl::blit::compute_blit_t<ceval_params_t>::create(params);\n"
				   "    InCSAccessor inCSA; OutImgAccessor outImgA; KernelWeightsAccessor kwA; HistogramAccessor hA; SharedAccessor sA;\n"
				   "	blit.execute(inCSA, outImgA, kwA, hA, sA, workGroupID, localInvocationIndex);\n"
				   "}\n";

			auto cpuShader = core::make_smart_refctd_ptr<asset::ICPUShader>(shaderSourceStream.str().c_str(), IGPUShader::E_SHADER_STAGE::ESS_COMPUTE, IGPUShader::E_SHADER_STAGE::E_CONTENT_TYPE::ECT_HLSL, "CComputeBlit::createBlitSpecializedShader");
			auto gpuShader = m_device->createShader(std::move(cpuShader.get()));

			return gpuShader;
		}

		template <typename BlitUtilities>
		core::smart_refctd_ptr<video::IGPUComputePipeline> getBlitPipeline(
			const asset::E_FORMAT									outFormat,
			const asset::IImage::E_TYPE								imageType,
			const core::vectorSIMDu32& inExtent,
			const core::vectorSIMDu32& outExtent,
			const asset::IBlitUtilities::E_ALPHA_SEMANTIC			alphaSemantic,
			const typename BlitUtilities::convolution_kernels_t& kernels,
			const uint32_t											workgroupSize = 256,
			const uint32_t											alphaBinCount = asset::IBlitUtilities::DefaultAlphaBinCount)
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

				auto specShader = createBlitSpecializedShader<BlitUtilities>(
					outFormat,
					imageType,
					inExtent,
					outExtent,
					alphaSemantic,
					kernels,
					workgroupSize,
					paddedAlphaBinCount);

				IGPUComputePipeline::SCreationParams creationParams;
				creationParams.shader.shader = specShader.get();
				creationParams.shader.entryPoint = "main";
				creationParams.layout = m_blitPipelineLayout[blitType].get();
				m_device->createComputePipelines(nullptr, { &creationParams, &creationParams + 1 }, &m_blitPipelines[key]);
			}

			return m_blitPipelines[key];
		}

		//! Returns the number of output texels produced by one workgroup, deciding factor is `m_availableSharedMemory`.
		//! @param outImageFormat is the format of output (of the blit step) image.
		//! If a normalization step is involved then this will be the same as the format of normalization step's input image --which may differ from the
		//! final output format, because we blit to a higher precision format for normalization.
		template <typename BlitUtilities>
		bool getOutputTexelsPerWorkGroup(
			core::vectorSIMDu32& outputTexelsPerWG,
			const core::vectorSIMDu32& inExtent,
			const core::vectorSIMDu32& outExtent,
			const asset::E_FORMAT									outImageFormat,
			const asset::IImage::E_TYPE								imageType,
			const typename BlitUtilities::convolution_kernels_t& kernels)
		{
			core::vectorSIMDf scale = static_cast<core::vectorSIMDf>(inExtent).preciseDivision(static_cast<core::vectorSIMDf>(outExtent));

			const core::vectorSIMDf minSupport(std::get<0>(kernels).getMinSupport(), std::get<1>(kernels).getMinSupport(), std::get<2>(kernels).getMinSupport());
			const core::vectorSIMDf maxSupport(std::get<0>(kernels).getMaxSupport(), std::get<1>(kernels).getMaxSupport(), std::get<2>(kernels).getMaxSupport());

			outputTexelsPerWG = core::vectorSIMDu32(1, 1, 1, 1);
			size_t requiredSmem = getRequiredSharedMemorySize(outputTexelsPerWG, outExtent, imageType, minSupport, maxSupport, scale, asset::getFormatChannelCount(outImageFormat));
			bool failed = true;
			asset::IImage::E_TYPE minDimAxes[3] = { asset::IImage::ET_1D, asset::IImage::ET_2D, asset::IImage::ET_3D };
			while (requiredSmem < m_availableSharedMemory)
			{
				failed = false;

				std::sort(minDimAxes, minDimAxes + imageType + 1, [&outputTexelsPerWG](const asset::IImage::E_TYPE a, const asset::IImage::E_TYPE b) -> bool { return outputTexelsPerWG[a] < outputTexelsPerWG[b]; });

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

		template <typename BlitUtilities>
		inline void buildParameters(
			nbl::hlsl::blit::parameters_t& outPC,
			const core::vectorSIMDu32& inImageExtent,
			const core::vectorSIMDu32& outImageExtent,
			const asset::IImage::E_TYPE								imageType,
			const asset::E_FORMAT									inImageFormat,
			const typename BlitUtilities::convolution_kernels_t& kernels,
			const uint32_t											layersToBlit = 1,
			const float												referenceAlpha = 0.f)
		{
			nbl::hlsl::uint16_t3 inDim(inImageExtent.x, inImageExtent.y, inImageExtent.z);
			nbl::hlsl::uint16_t3 outDim(outImageExtent.x, outImageExtent.y, outImageExtent.z);

			if (imageType < asset::IImage::ET_3D)
			{
				inDim.z = layersToBlit;
				outDim.z = layersToBlit;
			}

			constexpr auto MaxImageDim = 1 << 16;
			const auto maxImageDims = core::vectorSIMDu32(MaxImageDim, MaxImageDim, MaxImageDim, MaxImageDim);
			assert((inDim.x < maxImageDims.x) && (inDim.y < maxImageDims.y) && (inDim.z < maxImageDims.z));
			assert((outDim.x < maxImageDims.x) && (outDim.y < maxImageDims.y) && (outDim.z < maxImageDims.z));

			outPC.inputDims = inDim;
			outPC.outputDims = outDim;

			core::vectorSIMDf scale = static_cast<core::vectorSIMDf>(inImageExtent).preciseDivision(static_cast<core::vectorSIMDf>(outImageExtent));

			const core::vectorSIMDf minSupport(std::get<0>(kernels).getMinSupport(), std::get<1>(kernels).getMinSupport(), std::get<2>(kernels).getMinSupport());
			const core::vectorSIMDf maxSupport(std::get<0>(kernels).getMaxSupport(), std::get<1>(kernels).getMaxSupport(), std::get<2>(kernels).getMaxSupport());

			core::vectorSIMDu32 outputTexelsPerWG;
			getOutputTexelsPerWorkGroup<BlitUtilities>(outputTexelsPerWG, inImageExtent, outImageExtent, inImageFormat, imageType, kernels);
			const auto preloadRegion = getPreloadRegion(outputTexelsPerWG, imageType, minSupport, maxSupport, scale);

			outPC.secondScratchOffset = core::max(preloadRegion.x * preloadRegion.y * preloadRegion.z, outputTexelsPerWG.x * outputTexelsPerWG.y * preloadRegion.z);
			outPC.iterationRegionXPrefixProducts = { outputTexelsPerWG.x, outputTexelsPerWG.x * preloadRegion.y, outputTexelsPerWG.x * preloadRegion.y * preloadRegion.z };
			outPC.referenceAlpha = referenceAlpha;
			outPC.fScale = { scale.x, scale.y, scale.z };
			outPC.inPixelCount = inImageExtent.x * inImageExtent.y * inImageExtent.z;
			outPC.negativeSupport.x = minSupport.x; outPC.negativeSupport.y = minSupport.y; outPC.negativeSupport.z = minSupport.z;
			outPC.outPixelCount = outImageExtent.x * outImageExtent.y * outImageExtent.z;

			const core::vectorSIMDi32 windowDim = core::max(BlitUtilities::getWindowSize(imageType, kernels), core::vectorSIMDi32(1, 1, 1, 1));
			assert((windowDim.x < maxImageDims.x) && (windowDim.y < maxImageDims.y) && (windowDim.z < maxImageDims.z));

			const core::vectorSIMDu32 phaseCount = asset::IBlitUtilities::getPhaseCount(inImageExtent, outImageExtent, imageType);
			assert((phaseCount.x < maxImageDims.x) && (phaseCount.y < maxImageDims.y) && (phaseCount.z < maxImageDims.z));
			
			outPC.windowDims.x = windowDim.x;
			outPC.windowDims.y = windowDim.y;
			outPC.windowDims.z = windowDim.z;

			outPC.phaseCount.x = phaseCount.x;
			outPC.phaseCount.y = phaseCount.y;
			outPC.phaseCount.z = phaseCount.z;

			outPC.kernelWeightsOffsetY = phaseCount.x * windowDim.x;
			outPC.iterationRegionYPrefixProducts = { outputTexelsPerWG.y, outputTexelsPerWG.y * outputTexelsPerWG.x, outputTexelsPerWG.y * outputTexelsPerWG.x * preloadRegion.z };
			outPC.kernelWeightsOffsetZ = outPC.kernelWeightsOffsetY + phaseCount.y * windowDim.y;
			outPC.iterationRegionZPrefixProducts = { outputTexelsPerWG.y, outputTexelsPerWG.y * outputTexelsPerWG.x, outputTexelsPerWG.y * outputTexelsPerWG.x * outputTexelsPerWG.z };
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

		template <typename BlitUtilities>
		inline void buildBlitDispatchInfo(
			dispatch_info_t& dispatchInfo,
			const core::vectorSIMDu32& inImageExtent,
			const core::vectorSIMDu32& outImageExtent,
			const asset::E_FORMAT									inImageFormat,
			const asset::IImage::E_TYPE								imageType,
			const typename BlitUtilities::convolution_kernels_t& kernels,
			const uint32_t											workgroupSize = 256,
			const uint32_t											layersToBlit = 1)
		{
			core::vectorSIMDu32 outputTexelsPerWG;
			getOutputTexelsPerWorkGroup<BlitUtilities>(outputTexelsPerWG, inImageExtent, outImageExtent, inImageFormat, imageType, kernels);
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
			core::smart_refctd_ptr<video::IGPUImageView>			inImageView,
			core::smart_refctd_ptr<video::IGPUImageView>			outImageView,
			core::smart_refctd_ptr<video::IGPUBuffer>				coverageAdjustmentScratchBuffer,
			core::smart_refctd_ptr<video::IGPUBufferView>			kernelWeightsUTB,
			const asset::ISampler::E_TEXTURE_CLAMP					wrapU = asset::ISampler::ETC_CLAMP_TO_EDGE,
			const asset::ISampler::E_TEXTURE_CLAMP					wrapV = asset::ISampler::ETC_CLAMP_TO_EDGE,
			const asset::ISampler::E_TEXTURE_CLAMP					wrapW = asset::ISampler::ETC_CLAMP_TO_EDGE,
			const asset::ISampler::E_TEXTURE_BORDER_COLOR			borderColor = asset::ISampler::ETBC_FLOAT_OPAQUE_BLACK)
		{
			constexpr auto MAX_DESCRIPTOR_COUNT = 3;

			auto updateDS = [this, coverageAdjustmentScratchBuffer](video::IGPUDescriptorSet* ds, video::IGPUDescriptorSet::SDescriptorInfo* infos) -> bool
			{
				const auto bindingCount = ds->getLayout()->getTotalBindingCount();
				if ((bindingCount == 3) && !coverageAdjustmentScratchBuffer)
					return false;

				video::IGPUDescriptorSet::SWriteDescriptorSet writes[MAX_DESCRIPTOR_COUNT] = {};

				uint32_t infoIdx = 0;
				uint32_t writeCount = 0;
				for (uint32_t t = 0; t < static_cast<uint32_t>(asset::IDescriptor::E_TYPE::ET_COUNT); ++t)
				{
					const auto type = static_cast<asset::IDescriptor::E_TYPE>(t);
					const auto& redirect = ds->getLayout()->getDescriptorRedirect(type);
					const auto declaredBindingCount = redirect.getBindingCount();

					for (uint32_t i = 0; i < declaredBindingCount; ++i)
					{
						auto& write = writes[writeCount++];
						write.dstSet = ds;
						write.binding = redirect.getBinding(IGPUDescriptorSetLayout::CBindingRedirect::storage_range_index_t{ i }).data;
						write.arrayElement = 0u;
						write.count = redirect.getCount(IGPUDescriptorSetLayout::CBindingRedirect::storage_range_index_t{ i });
						write.info = &infos[infoIdx];

						infoIdx += write.count;
					}
				}
				assert(writeCount == bindingCount);
				m_device->updateDescriptorSets(writeCount, writes, 0u, nullptr);

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
				infos[0].info.image.imageLayout = asset::IImage::LAYOUT::READ_ONLY_OPTIMAL;
				infos[0].info.combinedImageSampler.sampler = samplers[wrapU][wrapV][wrapW][borderColor];

				infos[1].desc = outImageView;
				infos[1].info.image.imageLayout = asset::IImage::LAYOUT::GENERAL;
				infos[1].info.combinedImageSampler.sampler = nullptr;

				if (coverageAdjustmentScratchBuffer)
				{
					infos[2].desc = coverageAdjustmentScratchBuffer;
					infos[2].info.buffer.offset = 0;
					infos[2].info.buffer.size = coverageAdjustmentScratchBuffer->getSize();
				}

				if (!updateDS(blitDS, infos))
					return false;
			}

			if (kernelWeightsDS)
			{
				video::IGPUDescriptorSet::SDescriptorInfo info = {};
				info.desc = kernelWeightsUTB;
				info.info.buffer.offset = 0ull;
				info.info.buffer.size = kernelWeightsUTB->getUnderlyingBuffer()->getSize();

				if (!updateDS(kernelWeightsDS, &info))
					return false;
			}

			return true;
		}

		//! User is responsible for the memory barriers between previous writes and the first
		//! dispatch on the input image, and future reads of output image and the last dispatch.
		template <typename BlitUtilities>
		inline void blit(
			video::IGPUCommandBuffer* cmdbuf,
			const asset::IBlitUtilities::E_ALPHA_SEMANTIC			alphaSemantic,
			video::IGPUDescriptorSet* alphaTestDS,
			video::IGPUComputePipeline* alphaTestPipeline,
			video::IGPUDescriptorSet* blitDS,
			video::IGPUDescriptorSet* blitWeightsDS,
			video::IGPUComputePipeline* blitPipeline,
			video::IGPUDescriptorSet* normalizationDS,
			video::IGPUComputePipeline* normalizationPipeline,
			const core::vectorSIMDu32& inImageExtent,
			const asset::IImage::E_TYPE								inImageType,
			const asset::E_FORMAT									inImageFormat,
			core::smart_refctd_ptr<video::IGPUImage>				normalizationInImage,
			const typename BlitUtilities::convolution_kernels_t& kernels,
			const uint32_t											layersToBlit = 1,
			core::smart_refctd_ptr<video::IGPUBuffer>				coverageAdjustmentScratchBuffer = nullptr,
			const float												referenceAlpha = 0.f,
			const uint32_t											alphaBinCount = asset::IBlitUtilities::DefaultAlphaBinCount,
			const uint32_t											workgroupSize = 256)
		{
			const core::vectorSIMDu32 outImageExtent(normalizationInImage->getCreationParameters().extent.width, normalizationInImage->getCreationParameters().extent.height, normalizationInImage->getCreationParameters().extent.depth, 1u);

			nbl::hlsl::blit::parameters_t pushConstants;
			buildParameters<BlitUtilities>(pushConstants, inImageExtent, outImageExtent, inImageType, inImageFormat, kernels, layersToBlit, referenceAlpha);

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
				buildBlitDispatchInfo<BlitUtilities>(dispatchInfo, inImageExtent, outImageExtent, inImageFormat, inImageType, kernels, workgroupSize, layersToBlit);

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
				IGPUCommandBuffer::SPipelineBarrierDependencyInfo depInfo;
				// Memory dependency to ensure the alpha test pass has finished writing to alphaTestCounterBuffer
				video::IGPUCommandBuffer::SPipelineBarrierDependencyInfo::buffer_barrier_t alphaTestBarrier = {};
				alphaTestBarrier.barrier.dep.srcStageMask = asset::PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT;
				alphaTestBarrier.barrier.dep.srcAccessMask = asset::ACCESS_FLAGS::SHADER_WRITE_BITS;
				alphaTestBarrier.barrier.dep.dstStageMask = asset::PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT;
				alphaTestBarrier.barrier.dep.dstAccessMask = asset::ACCESS_FLAGS::SHADER_READ_BITS;
				alphaTestBarrier.range.buffer = coverageAdjustmentScratchBuffer;
				alphaTestBarrier.range.size = coverageAdjustmentScratchBuffer->getSize();
				alphaTestBarrier.range.offset = 0;

				// Memory dependency to ensure that the previous compute pass has finished writing to the output image,
				// also transitions the layout of said image: GENERAL -> SHADER_READ_ONLY_OPTIMAL
				video::IGPUCommandBuffer::SPipelineBarrierDependencyInfo::image_barrier_t readyForNorm = {};
				readyForNorm.barrier.dep.srcStageMask = asset::PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT;
				readyForNorm.barrier.dep.srcAccessMask = asset::ACCESS_FLAGS::SHADER_WRITE_BITS;
				readyForNorm.barrier.dep.dstStageMask = asset::PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT;
				readyForNorm.barrier.dep.dstAccessMask = asset::ACCESS_FLAGS::SHADER_READ_BITS;
				readyForNorm.oldLayout = video::IGPUImage::LAYOUT::GENERAL;
				readyForNorm.newLayout = video::IGPUImage::LAYOUT::READ_ONLY_OPTIMAL;
				readyForNorm.image = normalizationInImage.get();
				readyForNorm.subresourceRange.aspectMask = asset::IImage::EAF_COLOR_BIT;
				readyForNorm.subresourceRange.levelCount = 1u;
				readyForNorm.subresourceRange.layerCount = normalizationInImage->getCreationParameters().arrayLayers;

				depInfo.bufBarriers = { &alphaTestBarrier, &alphaTestBarrier + 1 };
				depInfo.imgBarriers = { &readyForNorm, &readyForNorm + 1 };

				cmdbuf->pipelineBarrier(asset::E_DEPENDENCY_FLAGS::EDF_NONE, depInfo);

				cmdbuf->bindDescriptorSets(asset::EPBP_COMPUTE, normalizationPipeline->getLayout(), 0u, 1u, &normalizationDS);
				cmdbuf->bindComputePipeline(normalizationPipeline);
				dispatchHelper(cmdbuf, normalizationPipeline->getLayout(), pushConstants, dispatchInfo);
			}
		}
#endif


	private:
		enum E_BLIT_TYPE : uint8_t
		{
			EBT_REGULAR = 0,
			EBT_COVERAGE_ADJUSTMENT,
			EBT_COUNT
		};

		void createAndCachePipelines(CAssetConverter* converter, core::smart_refctd_ptr<IGPUComputePipeline>* pipelines, const std::span<const STask> tasks);

		core::smart_refctd_ptr<ILogicalDevice> m_device;
		system::logger_opt_smart_ptr m_logger;
		core::smart_refctd_ptr<asset::IShaderCompiler::CCache> m_shaderCache;

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

			const size_t requiredSmem = (core::max(preloadRegion.x * preloadRegion.y * preloadRegion.z, outputTexelsPerWG.x * outputTexelsPerWG.y * preloadRegion.z) + outputTexelsPerWG.x * preloadRegion.y * preloadRegion.z) * channelCount * sizeof(float);
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
#endif
