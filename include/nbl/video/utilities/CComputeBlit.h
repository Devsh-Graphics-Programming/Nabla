#ifndef _NBL_VIDEO_C_COMPUTE_BLIT_H_INCLUDED_
#define _NBL_VIDEO_C_COMPUTE_BLIT_H_INCLUDED_

#include "nbl/asset/filters/CBlitUtilities.h"
#include "nbl/builtin/hlsl/format.hlsl"
#include "nbl/builtin/hlsl/blit/parameters.hlsl"

namespace nbl::video
{

class CComputeBlit : public core::IReferenceCounted
{
	public:
		constexpr static inline asset::SPushConstantRange DefaultPushConstantRange = {
			.stageFlags = IGPUShader::E_SHADER_STAGE::ESS_COMPUTE,
			.offset = 0ull,
			.size = sizeof(hlsl::blit::parameters2_t)
		};
		constexpr static inline std::span<const asset::SPushConstantRange> DefaultPushConstantRanges = {&DefaultPushConstantRange,1};

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
		NBL_API2 CComputeBlit(
			core::smart_refctd_ptr<ILogicalDevice>&& logicalDevice,
			core::smart_refctd_ptr<asset::IShaderCompiler::CCache>&& cache=nullptr,
			core::smart_refctd_ptr<system::ILogger>&& logger=nullptr
		);

		// create your pipelines
		struct SPipelines
		{
			core::smart_refctd_ptr<IGPUComputePipeline> blit;
			core::smart_refctd_ptr<IGPUComputePipeline> coverage;
			uint16_t workgroupSize;
		};
		struct SPipelinesCreateInfo
		{
			// required
			CAssetConverter* converter;
			// in theory we _could_ accept either pipeline layout type (or just the base) and make the CPU one back from the GPU
			const asset::ICPUPipelineLayout* layout;
			// must be Uniform Texel Buffer descriptor type
			hlsl::SBindingInfo kernelWeights;
			// must be Sampled Image descriptor type
			hlsl::SBindingInfo inputs;
			// must be Sampler descriptor type
			hlsl::SBindingInfo samplers;
			// must be Storage Image descriptor type
			hlsl::SBindingInfo outputs;
			//! If you set the balues too small, we'll correct them ourselves anyway, default values of 0 means we guess and provide our defaults
			// needs to be at least as big as the maximum subgroup size 
			uint16_t workgroupSizeLog2 : 4 = 0;
			// in bytes, needs to be at least enough to store two full input pixels per invocation
			uint16_t sharedMemoryPerInvocation : 6 = 0;
		};
		NBL_API2 SPipelines createAndCachePipelines(const SPipelinesCreateInfo& info);

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

		// Use the return values of `getOutputViewFormat` and `getCoverageAdjustmentIntermediateFormat` for this
		static inline uint32_t getAlphaBinCount(const uint16_t workgroupSize, const asset::E_FORMAT intermediateAlpha, const uint32_t layersToBlit)
		{
			uint16_t baseBucketCount;
			using format_t = nbl::asset::E_FORMAT;
			switch (intermediateAlpha)
			{
				case format_t::EF_R8_UNORM: [[fallthrough]];
				case format_t::EF_R8_SNORM:
					baseBucketCount = 256;
					break;
				case format_t::EF_R16_SFLOAT:
					baseBucketCount = 512;
					break;
				case format_t::EF_R16_UNORM: [[fallthrough]];
				case format_t::EF_R16_SNORM: [[fallthrough]];
					baseBucketCount = 1024;
					break;
				case format_t::EF_R32_SFLOAT:
					baseBucketCount = 2048;
					break;
				default:
					return 0;
			}
			// the absolute minimum needed to store a single pixel of a worst case format (precise, all 4 channels)
			constexpr auto singlePixelStorage = 4*sizeof(hlsl::float32_t);
			constexpr auto ratio = singlePixelStorage/sizeof(uint16_t);
			// atomicAdd gets performed on MSB or LSB of a single DWORD
			const auto paddedAlphaBinCount = core::min(core::roundUp<uint16_t>(baseBucketCount,workgroupSize*2),workgroupSize*ratio);
			return paddedAlphaBinCount*layersToBlit;
		}
		
		static inline uint32_t getNormalizationByteSize(const uint16_t workgroupSize, const asset::E_FORMAT intermediateAlpha, const uint32_t layersToBlit)
		{
			return getAlphaBinCount(workgroupSize,intermediateAlpha,layersToBlit)*sizeof(uint16_t)+sizeof(uint32_t)+sizeof(uint32_t);
		}
#if 0

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

		//! User is responsible for the memory barriers between previous writes and the first
		//! dispatch on the input image, and future reads of output image and the last dispatch.
		template <typename BlitUtilities>
		inline void blit(
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
// bind omitted
				dispatchHelper(cmdbuf, alphaTestPipeline->getLayout(), pushConstants, dispatchInfo);
			}

			{
				dispatch_info_t dispatchInfo;
				buildBlitDispatchInfo<BlitUtilities>(dispatchInfo, inImageExtent, outImageExtent, inImageFormat, inImageType, kernels, workgroupSize, layersToBlit);
// bind omitted
				dispatchHelper(cmdbuf, blitPipeline->getLayout(), pushConstants, dispatchInfo);
			}

			// After this dispatch ends and finishes writing to outImage, normalize outImage
			if (alphaSemantic == asset::IBlitUtilities::EAS_REFERENCE_OR_COVERAGE)
			{
				dispatch_info_t dispatchInfo;
				buildNormalizationDispatchInfo(dispatchInfo, outImageExtent, inImageType, layersToBlit);

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
		inline size_t getRequiredSharedMemorySize(
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
};

}
#endif
