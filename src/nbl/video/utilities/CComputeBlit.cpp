#include "nbl/video/utilities/CComputeBlit.h"

using namespace nbl;
using namespace video;


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

	auto cpuShader = core::make_smart_refctd_ptr<asset::ICPUShader>(shaderSourceStream.str().c_str(), asset::IShader::ESS_COMPUTE, asset::IShader::E_CONTENT_TYPE::ECT_HLSL, "CComputeBlitGLSLGLSL::createAlphaTestSpecializedShader");

	return  m_device->createShader(std::move(cpuShader.get()));
}

core::smart_refctd_ptr<video::IGPUShader> CComputeBlit::createNormalizationSpecializedShader(const asset::IImage::E_TYPE imageType, const asset::E_FORMAT outFormat,
	const uint32_t alphaBinCount)
{
	const auto workgroupDims = getDefaultWorkgroupDims(imageType);
	const auto paddedAlphaBinCount = getPaddedAlphaBinCount(workgroupDims, alphaBinCount);
	const uint32_t blitDimCount = static_cast<uint32_t>(imageType) + 1;

	const auto castedFormat = getOutImageViewFormat(outFormat);
	assert(outFormat == castedFormat);
	const char* formatQualifier = asset::CHLSLCompiler::getStorageImageFormatQualifier(castedFormat);

	std::ostringstream shaderSourceStream;

	shaderSourceStream
		<< "#include \"nbl/builtin/hlsl/blit/common.hlsl\"\n"
		   "#include \"nbl/builtin/hlsl/blit/parameters.hlsl\"\n"
		   "#include \"nbl/builtin/hlsl/blit/normalization.hlsl\"\n"

		   "typedef nbl::hlsl::blit::consteval_parameters_t<" << workgroupDims.x << ", " << workgroupDims.y << ", " << workgroupDims.z << ", "
		   "0, 0, " << blitDimCount << ", " << paddedAlphaBinCount << "> ceval_params_t;\n"

		   "[[vk::binding(0, 0)]]\n"
		   "nbl::hlsl::blit::impl::dim_to_image_properties<ceval_params_t::BlitDimCount>::combined_sampler_t inCS;\n"

		   "[[vk::image_format(\"" << formatQualifier << "\")]]\n"
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

	auto cpuShader = core::make_smart_refctd_ptr<asset::ICPUShader>(shaderSourceStream.str().c_str(), asset::IShader::ESS_COMPUTE, asset::IShader::E_CONTENT_TYPE::ECT_HLSL, "CComputeBlitGLSL::createNormalizationSpecializedShader");

	return m_device->createShader(std::move(cpuShader.get()));
}
