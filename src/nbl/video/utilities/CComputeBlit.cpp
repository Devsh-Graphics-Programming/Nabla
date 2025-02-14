#include "nbl/video/utilities/CComputeBlit.h"
#include "nbl/builtin/hlsl/binding_info.hlsl"
#include "nbl/builtin/hlsl/tgmath.hlsl"

using namespace nbl::core;
using namespace nbl::hlsl;
using namespace nbl::hlsl::blit;
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

auto CComputeBlit::createAndCachePipelines(const SPipelinesCreateInfo& info) -> SPipelines
{
	SPipelines retval;

	std::array<smart_refctd_ptr<ICPUComputePipeline>,2> cpuPplns;

	const auto& limits = m_device->getPhysicalDevice()->getLimits();
	retval.workgroupSize = 0x1u<<info.workgroupSizeLog2;
	if (retval.workgroupSize <limits.maxSubgroupSize)
		retval.workgroupSize = core::roundDownToPoT(limits.maxComputeWorkGroupInvocations);
	// the absolute minimum needed to store a single pixel of a worst case format (precise, all 4 channels)
	constexpr size_t singlePixelStorage = sizeof(float32_t);
	// also slightly more memory is needed to even have a skirt of any size, and we need at least 2 buffers to ping-pong, and this value be better PoT
	const auto sharedMemoryPerInvocation = hlsl::max(singlePixelStorage*4,static_cast<size_t>(info.sharedMemoryPerInvocation));
	retval.sharedMemorySize = sharedMemoryPerInvocation*retval.workgroupSize;

	const auto* layout = info.layout;

	// 
	const auto common = [&]()->std::string
	{
		std::ostringstream tmp;
		tmp << R"===(
#include "nbl/builtin/hlsl/binding_info.hlsl"


using namespace nbl::hlsl;


struct ConstevalParameters
{
	NBL_CONSTEXPR_STATIC_INLINE uint32_t WorkGroupSize = )===" << retval.workgroupSize << R"===(;
	using kernel_weight_binding_t = )===" << layout->getBindingInfoForHLSL({.binding=info.kernelWeights,.requiredStages=IShader::E_SHADER_STAGE::ESS_COMPUTE}) << R"===(;
	using input_sampler_binding_t = )===" << layout->getBindingInfoForHLSL({.binding=info.samplers,.requiredStages=IShader::E_SHADER_STAGE::ESS_COMPUTE}) << R"===(;
	using input_image_binding_t = )===" << layout->getBindingInfoForHLSL({.binding=info.inputs,.requiredStages=IShader::E_SHADER_STAGE::ESS_COMPUTE}) << R"===(;
	using output_binding_t = )===" << layout->getBindingInfoForHLSL({.binding=info.outputs,.requiredStages=IShader::E_SHADER_STAGE::ESS_COMPUTE}) << R"===(;
	NBL_CONSTEXPR_STATIC_INLINE uint32_t SharedMemoryDWORDs = )===" << retval.sharedMemorySize/sizeof(uint32_t) << R"===(;
};
)===";
		return tmp.str();
	}();
	auto createPipeline = [&limits,layout,&common](const char* mainPath)->smart_refctd_ptr<ICPUComputePipeline>
	{
		auto shader = make_smart_refctd_ptr<ICPUShader>(
			(common+"\n#include \""+mainPath+"\"\n").c_str(),
			IShader::E_SHADER_STAGE::ESS_COMPUTE,
			IShader::E_CONTENT_TYPE::ECT_HLSL,
			mainPath
		);
		// make sure there's a hash so asset converter doesn't fail
		{
			auto source = const_cast<ICPUBuffer*>(shader->getContent());
			source->setContentHash(source->computeContentHash());
		}

		ICPUComputePipeline::SCreationParams params = {};
		params.layout = layout;
		params.shader.entryPoint = "main";
		params.shader.shader = shader.get();
		params.shader.requiredSubgroupSize = static_cast<IShader::SSpecInfoBase::SUBGROUP_SIZE>(findMSB(limits.maxSubgroupSize));
		// needed for the prefix and reductions to work
		params.shader.requireFullSubgroups = true;
		return ICPUComputePipeline::create(params);
	};
	// create blit pipeline
	cpuPplns[0] = createPipeline("nbl/builtin/hlsl/blit/default_blit.comp.hlsl");
	cpuPplns[1] = createPipeline("nbl/builtin/hlsl/blit/default_normalize.comp.hlsl");

	CAssetConverter::SInputs inputs = {};
	inputs.readCache = info.converter;
	inputs.logger = m_logger.getRaw();
	std::get<CAssetConverter::SInputs::asset_span_t<ICPUComputePipeline>>(inputs.assets) = {&cpuPplns.data()->get(),cpuPplns.size()};
	inputs.readShaderCache = m_shaderCache.get();
	inputs.writeShaderCache = m_shaderCache.get();
	// no pipeline cache, because we only make the same pipeline once, ever
	auto reserveResults = info.converter->reserve(inputs);
	assert(reserveResults.getRequiredQueueFlags().value==IQueue::FAMILY_FLAGS::NONE);

	// copy over the results
	{
		auto rIt = reserveResults.getGPUObjects<ICPUComputePipeline>().data();
		retval.blit = (rIt++)->value;
		retval.coverage = (rIt++)->value;
	}

	// this just inserts the pipelines into the cache
	{
		CAssetConverter::SConvertParams params = {};
		auto convertResults = reserveResults.convert(params);
		assert(!convertResults.blocking());
	}
	return retval;
}

SPerWorkgroup CComputeBlit::computePerWorkGroup(
	const uint16_t sharedMemorySize, const float32_t3 minSupportInInput, const float32_t3 maxSupportInInput, const IGPUImage::E_TYPE type,
	const uint16_t3 inExtent, const uint16_t3 outExtent, const bool halfPrecision
)
{
	SPerWorkgroup retval;
	memset(&retval,0,sizeof(retval));

	const auto Dims = static_cast<uint8_t>(type)+1;
	const auto scale = float32_t3(inExtent)/float32_t3(outExtent);
	const auto supportWidthInInput = maxSupportInInput-minSupportInInput;

	IGPUImage::E_TYPE minDimAxes[3] = { IGPUImage::ET_1D, IGPUImage::ET_2D, IGPUImage::ET_3D };
	using namespace nbl::hlsl;
	for (uint16_t3 output(1,1,1); true;)
	{
		// now try and grow our support
		const auto combinedSupportInInput = supportWidthInInput+float32_t3(output-uint16_t3(1,1,1))*scale;
		// note that its not ceil on purpose
		uint32_t3 preload = uint32_t3(hlsl::floor(combinedSupportInInput))+uint32_t3(1,1,1);
		// Set the unused dimensions to 1 to avoid weird behaviours with scaled kernels
		for (auto a=Dims; a<3; a++)
			preload[a] = 1;
		// TODO: the blits should probably be implemented with separate preload per channel
		{
			// think in terms of inputs (for now we have a fixed, unoptimized ordering of XYZ)
			const uint16_t firstPass = preload.x*preload.y*preload.z;
			const uint16_t secondPass = output.x*preload.y*preload.z;
			const uint16_t thirdPass = output.x*output.y*preload.z;
			//
			uint32_t otherPreloadOffset = firstPass;
			// third pass aliases first pass input storage
			if (Dims==3 && otherPreloadOffset<thirdPass)
				otherPreloadOffset = thirdPass;
			//
			const auto totalPixels = otherPreloadOffset+(Dims>1 ? secondPass:0u);
			const auto requiredSharedMemory = totalPixels*sizeof(float);//(halfPrecision ? sizeof(float16_t):sizeof(float32_t)); TODO: impl the blit in 16bits
			// too much
			if (requiredSharedMemory>size_t(sharedMemorySize))
				break;
			// still fits, update return value
			retval = SPerWorkgroup::create(scale,Dims,output,preload,otherPreloadOffset);
		}
		
		// we want to fix the dimension that's the smallest, so that we increase the volume of the support by a smallest increment and stay close to a cube shape
		{
			std::sort(minDimAxes,minDimAxes+Dims,[preload](const IGPUImage::E_TYPE a, const IGPUImage::E_TYPE b)->bool
				{
					return preload[a]<preload[b];
				}
			);
			// grow along smallest axis, but skip if already grown to output size
			auto a = 0;
			for (; a<Dims; a++)
			if (output[a]<outExtent[a])
			{
				output[*minDimAxes]++;
				break;
			}
			// can't grow anymore
			if (a==Dims)
				break;
		}
	}
	return retval;
}

#if 0

template <typename BlitUtilities>
core::smart_refctd_ptr<video::IGPUShader> createBlitSpecializedShader(
	const asset::IImage::E_TYPE								imageType,
	const hlsl::uint32_t4& inExtent,
	const hlsl::uint32_t4& outExtent,
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

.......
				
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
}

core::smart_refctd_ptr<video::IGPUShader> CComputeBlit::createAlphaTestSpecializedShader(const asset::IImage::E_TYPE imageType, const uint32_t alphaBinCount)
{
	const auto workgroupDims = getDefaultWorkgroupDims(imageType);
	const auto paddedAlphaBinCount = getPaddedAlphaBinCount(workgroupDims, alphaBinCount);
	const uint32_t blitDimCount = static_cast<uint32_t>(imageType) + 1;

........

		   "struct PassedPixelsAccessor { void atomicAdd(uint32_t wgID, uint32_t v) { InterlockedAdd(statsBuff[wgID * (ceval_params_t::AlphaBinCount + 1) + ceval_params_t::AlphaBinCount], v); } };\n"
		   "struct InCSAccessor { float32_t4 get(int32_t3 c, uint32_t l) { return inCS[nbl::hlsl::blit::impl::dim_to_image_properties<ceval_params_t::BlitDimCount>::getIndexCoord<int32_t>(c, l)]; } };\n"

		   "[numthreads(ceval_params_t::WorkGroupSizeX, ceval_params_t::WorkGroupSizeY, ceval_params_t::WorkGroupSizeZ)]"
		   "void main(uint32_t3 globalInvocationID : SV_DispatchThreadID, uint32_t3 workGroupID: SV_GroupID)\n"
		   "{\n"
		   "    InCSAccessor inCSA;PassedPixelsAccessor ppA;\n"
		   "	nbl::hlsl::blit::alpha_test(ppA, inCSA, params.inputDims, params.referenceAlpha, globalInvocationID, workGroupID);\n"
		   "}\n";
}

// @param `outFormat` dictates encoding.
core::smart_refctd_ptr<video::IGPUShader> CComputeBlit::createNormalizationSpecializedShader(const asset::IImage::E_TYPE imageType, const uint32_t alphaBinCount)
{
	const auto workgroupDims = getDefaultWorkgroupDims(imageType);
	const auto paddedAlphaBinCount = getPaddedAlphaBinCount(workgroupDims, alphaBinCount);
	const uint32_t blitDimCount = static_cast<uint32_t>(imageType) + 1;

....

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
}
#endif