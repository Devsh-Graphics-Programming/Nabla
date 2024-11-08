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
			.size = sizeof(hlsl::blit::Parameters)
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
			uint16_t sharedMemorySize;
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
		static inline uint32_t getAlphaBinCount(const SPipelines& pipelines, const asset::E_FORMAT intermediateAlpha, const uint32_t layersToBlit)
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
			// atomicAdd gets performed on MSB or LSB of a single DWORD
			const auto paddedAlphaBinCount = core::min<uint16_t>(
				core::roundUp<uint16_t>(baseBucketCount,pipelines.workgroupSize*2),
				(pipelines.sharedMemorySize-sizeof(uint32_t)*2)/sizeof(uint16_t)
			);
			return paddedAlphaBinCount*layersToBlit;
		}
		
		static inline uint32_t getNormalizationByteSize(const SPipelines& pipelines, const asset::E_FORMAT intermediateAlpha, const uint32_t layersToBlit)
		{
			return getAlphaBinCount(pipelines,intermediateAlpha,layersToBlit)*sizeof(uint16_t)+sizeof(uint32_t)+sizeof(uint32_t);
		}
		

		//! Returns the number of output texels produced by one workgroup, deciding factor is `pipelines.sharedmemorySize`.
		//! @param outFormat is the format of output (of the blit step) image.
		template <typename BlitUtilities>
		static inline hlsl::blit::SPerWorkgroup computePerWorkGroup(
			const uint16_t sharedMemorySize, const typename BlitUtilities::convolution_kernels_t& kernels, const IGPUImage::E_TYPE type,
			const bool halfPrecision, const hlsl::uint16_t3 inExtent, const hlsl::uint16_t3 outExtent
		)
		{
			const hlsl::float32_t3 minSupport(std::get<0>(kernels).getMinSupport(), std::get<1>(kernels).getMinSupport(), std::get<2>(kernels).getMinSupport());
			const hlsl::float32_t3 maxSupport(std::get<0>(kernels).getMaxSupport(), std::get<1>(kernels).getMaxSupport(), std::get<2>(kernels).getMaxSupport());
			return computePerWorkGroup(sharedMemorySize,minSupport,maxSupport,type,halfPrecision);
		}
		static hlsl::blit::SPerWorkgroup computePerWorkGroup(
			const uint16_t sharedMemorySize, const hlsl::float32_t3 minSupportInOutput, const hlsl::float32_t3 maxSupportInOutput, const IGPUImage::E_TYPE type,
			const bool halfPrecision, const hlsl::uint16_t3 inExtent, const hlsl::uint16_t3 outExtent
		);

#if 0
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
};

}
#endif
