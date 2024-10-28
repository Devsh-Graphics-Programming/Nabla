#include "nbl/video/utilities/CComputeBlit.h"

using namespace nbl::core;
using namespace nbl::system;
using namespace nbl::asset;
using namespace nbl::video;


CComputeBlit::CComputeBlit(smart_refctd_ptr<ILogicalDevice>&& logicalDevice, smart_refctd_ptr<IShaderCompiler::CCache>&& cache, smart_refctd_ptr<ILogger>&& logger) : m_device(std::move(logicalDevice)), m_logger(nullptr)
{
	if (logger)
		m_logger = std::move(logger);
	else if (auto debugCb=m_device->getPhysicalDevice()->getDebugCallback(); debugCb->getLogger())
		m_logger = smart_refctd_ptr<system::ILogger>(debugCb->getLogger());
	
	if (cache)
		m_shaderCache = std::move(cache);
	else
		m_shaderCache = make_smart_refctd_ptr<IShaderCompiler::CCache>();
}

void CComputeBlit::createAndCachePipelines(CAssetConverter* converter, smart_refctd_ptr<IGPUComputePipeline>* pipelines, const std::span<const STask> tasks)
{
	core::vector<smart_refctd_ptr<ICPUComputePipeline>> cpuPplns;
	cpuPplns.reserve(tasks.size());

	const auto& limits = m_device->getPhysicalDevice()->getLimits();
	for (auto task : tasks)
	{
		// adjust task default values
		{
			if (task.workgroupSizeLog2<limits.maxSubgroupSize)
				task.workgroupSizeLog2 = core::roundDownToPoT(limits.maxComputeWorkGroupInvocations);
			bool useFloat16 = false;
			uint16_t channels = 4;
			using namespace hlsl::format;
			if (task.outputFormat!=TexelBlockFormat::TBF_UNKNOWN)
			{
				channels = getTraits(task.outputFormat).Channels;
				const auto precisionAt1 = getFormatPrecision(static_cast<E_FORMAT>(task.outputFormat),3,1.f);
				const auto precisionAt0 = getFormatPrecision(static_cast<E_FORMAT>(task.outputFormat),3,0.f);
				if (limits.workgroupMemoryExplicitLayout16BitAccess && limits.shaderFloat16 && precisionAt1>=std::exp2f(-11.f) && precisionAt0>=std::numeric_limits<hlsl::float16_t>::min())
					useFloat16 = true;
			}
			// the absolute minimum needed to store a single pixel
			const auto singlePixelStorage = channels*(useFloat16 ? sizeof(hlsl::float16_t):sizeof(hlsl::float32_t));
			// also slightly more memory is needed
			task.sharedMemoryPerInvocation = core::max(singlePixelStorage*2,task.sharedMemoryPerInvocation);
		}
		// create blit pipeline
		cpuPplns.emplace_back(nullptr);
		// create optional coverage normalization pipeline
		cpuPplns.emplace_back(nullptr);
	}

	CAssetConverter::SInputs inputs = {};
	inputs.readCache = converter;
	inputs.logger = m_logger.getRaw();
	std::get<CAssetConverter::SInputs::asset_span_t<ICPUComputePipeline>>(inputs.assets) = {&cpuPplns.data()->get(),cpuPplns.size()};
	inputs.readShaderCache = m_shaderCache.get();
	inputs.writeShaderCache = m_shaderCache.get();
	// no pipeline cache, because we only make the same pipeline once, ever
	auto reserveResults = converter->reserve(inputs);
	assert(reserveResults.getRequiredQueueFlags().value==IQueue::FAMILY_FLAGS::NONE);
	// copy over the results
	{
		auto rIt = reserveResults.getGPUObjects<ICPUComputePipeline>().data();
		// TODO: redo
		for (size_t i=0; i<tasks.size(); i++)
			*(pipelines++) =  (rIt++)->value;
	}

	// this just inserts the pipelines into the cache
	{
		CAssetConverter::SConvertParams params = {};
		auto convertResults = reserveResults.convert(params);
		assert(!convertResults.blocking());
	}
}

#if 0
core::smart_refctd_ptr<video::IGPUShader> CComputeBlit::createAlphaTestSpecializedShader(const asset::IImage::E_TYPE imageType, const uint32_t alphaBinCount)
{
	const auto workgroupDims = getDefaultWorkgroupDims(imageType);
	const auto paddedAlphaBinCount = getPaddedAlphaBinCount(workgroupDims, alphaBinCount);
	const uint32_t blitDimCount = static_cast<uint32_t>(imageType) + 1;

	std::ostringstream shaderSourceStream;

	shaderSourceStream
		<< "#include \"nbl/builtin/hlsl/blit/common.hlsl\"\n"
		   "#include \"nbl/builtin/hlsl/blit/parameters.hlsl\"\n"
		   "#include \"nbl/builtin/hlsl/blit/alpha_test.hlsl\"\n"

		   "typedef nbl::hlsl::blit::consteval_parameters_t<" << workgroupDims.x << ", " << workgroupDims.y << ", " << workgroupDims.z << ", "
		   "0, 0, " << blitDimCount << ", " << paddedAlphaBinCount << "> ceval_params_t;\n"

		   "[[vk::binding(0, 0)]]\n"
		   "nbl::hlsl::blit::impl::dim_to_image_properties<ceval_params_t::BlitDimCount>::combined_sampler_t inCS;\n"

		   "[[vk::binding(2 , 0)]] RWStructuredBuffer<uint32_t> statsBuff;\n"
	       "[[vk::push_constant]] nbl::hlsl::blit::parameters_t params;"

		   "struct PassedPixelsAccessor { void atomicAdd(uint32_t wgID, uint32_t v) { InterlockedAdd(statsBuff[wgID * (ceval_params_t::AlphaBinCount + 1) + ceval_params_t::AlphaBinCount], v); } };\n"
		   "struct InCSAccessor { float32_t4 get(int32_t3 c, uint32_t l) { return inCS[nbl::hlsl::blit::impl::dim_to_image_properties<ceval_params_t::BlitDimCount>::getIndexCoord<int32_t>(c, l)]; } };\n"

		   "[numthreads(ceval_params_t::WorkGroupSizeX, ceval_params_t::WorkGroupSizeY, ceval_params_t::WorkGroupSizeZ)]"
		   "void main(uint32_t3 globalInvocationID : SV_DispatchThreadID, uint32_t3 workGroupID: SV_GroupID)\n"
		   "{\n"
		   "    InCSAccessor inCSA;PassedPixelsAccessor ppA;\n"
		   "	nbl::hlsl::blit::alpha_test(ppA, inCSA, params.inputDims, params.referenceAlpha, globalInvocationID, workGroupID);\n"
		   "}\n";

	auto cpuShader = core::make_smart_refctd_ptr<asset::ICPUShader>(shaderSourceStream.str().c_str(), IGPUShader::E_SHADER_STAGE::ESS_COMPUTE, IGPUShader::E_CONTENT_TYPE::ECT_HLSL, "CComputeBlitGLSLGLSL::createAlphaTestSpecializedShader");
}

core::smart_refctd_ptr<video::IGPUShader> CComputeBlit::createNormalizationSpecializedShader(const asset::IImage::E_TYPE imageType, const uint32_t alphaBinCount)
{
	const auto workgroupDims = getDefaultWorkgroupDims(imageType);
	const auto paddedAlphaBinCount = getPaddedAlphaBinCount(workgroupDims, alphaBinCount);
	const uint32_t blitDimCount = static_cast<uint32_t>(imageType) + 1;

	std::ostringstream shaderSourceStream;

	shaderSourceStream
		<< "#include \"nbl/builtin/hlsl/blit/common.hlsl\"\n"
		   "#include \"nbl/builtin/hlsl/blit/parameters.hlsl\"\n"
		   "#include \"nbl/builtin/hlsl/blit/normalization.hlsl\"\n"

		   "typedef nbl::hlsl::blit::consteval_parameters_t<" << workgroupDims.x << ", " << workgroupDims.y << ", " << workgroupDims.z << ", "
		   "0, 0, " << blitDimCount << ", " << paddedAlphaBinCount << "> ceval_params_t;\n"

		   "[[vk::binding(0, 0)]]\n"
		   "nbl::hlsl::blit::impl::dim_to_image_properties<ceval_params_t::BlitDimCount>::combined_sampler_t inCS;\n"

		   "[[vk::image_format(\"unknown\")]]\n"
		   "[[vk::binding(1, 0)]]\n"
		   "nbl::hlsl::blit::impl::dim_to_image_properties<ceval_params_t::BlitDimCount>::image_t outImg;\n"

		   "[[vk::binding(2 , 0)]] RWStructuredBuffer<uint32_t> statsBuff;\n"
		   "[[vk::push_constant]] nbl::hlsl::blit::parameters_t params;"
		   "groupshared float32_t sMem[ceval_params_t::WorkGroupSize + 1];\n"

           "struct PassedPixelsAccessor { uint32_t get(uint32_t wgID) { return statsBuff[wgID * (ceval_params_t::AlphaBinCount + 1) + ceval_params_t::AlphaBinCount]; } };\n"
		   "struct HistogramAccessor { uint32_t get(uint32_t wgID, uint32_t bucket) { return statsBuff[wgID * (ceval_params_t::AlphaBinCount + 1) + bucket]; } };\n"
		   "struct InCSAccessor { float32_t4 get(int32_t3 c, uint32_t l) { return inCS[nbl::hlsl::blit::impl::dim_to_image_properties<ceval_params_t::BlitDimCount>::getIndexCoord<int32_t>(c, l)]; } };\n"
		   "struct OutImgAccessor { void set(int32_t3 c, uint32_t l, float32_t4 v) { outImg[nbl::hlsl::blit::impl::dim_to_image_properties<ceval_params_t::BlitDimCount>::getIndexCoord<uint32_t>(c, l)] = v; } };\n"
		   "struct SharedAccessor { struct { float32_t get(float32_t idx) { return sMem[idx]; } void set(float32_t idx, float32_t val) { sMem[idx] = val; } } main; };\n"

		   "[numthreads(ceval_params_t::WorkGroupSizeX, ceval_params_t::WorkGroupSizeY, ceval_params_t::WorkGroupSizeZ)]"
		   "void main(uint32_t3 workGroupID : SV_GroupID, uint32_t3 globalInvocationID : SV_DispatchThreadID, uint32_t localInvocationIndex : SV_GroupIndex)"
		   "{\n"
		   "	nbl::hlsl::blit::normalization_t<ceval_params_t> blit = nbl::hlsl::blit::normalization_t<ceval_params_t>::create(params);\n"
           "    InCSAccessor inCSA; OutImgAccessor outImgA; HistogramAccessor hA; PassedPixelsAccessor ppA; SharedAccessor sA;\n"
		   "	blit.execute(inCSA, outImgA, hA, ppA, sA, workGroupID, globalInvocationID, localInvocationIndex);\n"
		   "}\n";

	auto cpuShader = core::make_smart_refctd_ptr<asset::ICPUShader>(shaderSourceStream.str().c_str(), IGPUShader::E_SHADER_STAGE::ESS_COMPUTE, IGPUShader::E_CONTENT_TYPE::ECT_HLSL, "CComputeBlitGLSL::createNormalizationSpecializedShader");
}
#endif