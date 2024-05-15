#include "nbl/video/utilities/CComputeBlit.h"

using namespace nbl;
using namespace video;

#if 0 // TODO: port
core::smart_refctd_ptr<video::IGPUSpecializedShader> CComputeBlit::createAlphaTestSpecializedShader(const asset::IImage::E_TYPE imageType, const uint32_t alphaBinCount)
{
	const auto workgroupDims = getDefaultWorkgroupDims(imageType);
	const auto paddedAlphaBinCount = getPaddedAlphaBinCount(workgroupDims, alphaBinCount);

	std::ostringstream shaderSourceStream;
	shaderSourceStream
		<< "#version 460 core\n"
		<< "#define _NBL_GLSL_WORKGROUP_SIZE_X_ " << ((imageType >= asset::IImage::ET_1D) ? workgroupDims.x : 1) << "\n"
		<< "#define _NBL_GLSL_WORKGROUP_SIZE_Y_ " << ((imageType >= asset::IImage::ET_2D) ? workgroupDims.y : 1) << "\n"
		<< "#define _NBL_GLSL_WORKGROUP_SIZE_Z_ " << ((imageType >= asset::IImage::ET_3D) ? workgroupDims.z : 1) << "\n"
		<< "#define _NBL_GLSL_BLIT_DIM_COUNT_ " << static_cast<uint32_t>(imageType) + 1 << "\n"
		<< "#define _NBL_GLSL_BLIT_ALPHA_BIN_COUNT_ " << paddedAlphaBinCount << "\n"
		<< "#include <nbl/builtin/glsl/blit/default_compute_alpha_test.comp>\n";

	auto cpuShader = core::make_smart_refctd_ptr<asset::ICPUShader>(shaderSourceStream.str().c_str(), asset::IShader::ESS_COMPUTE, asset::IShader::E_CONTENT_TYPE::ECT_GLSL, "CComputeBlit::createAlphaTestSpecializedShader");
	auto gpuUnspecShader = m_device->createShader(std::move(cpuShader));

	return m_device->createSpecializedShader(gpuUnspecShader.get(), { nullptr, nullptr, "main" });
}

core::smart_refctd_ptr<video::IGPUSpecializedShader> CComputeBlit::createNormalizationSpecializedShader(const asset::IImage::E_TYPE imageType, const asset::E_FORMAT outFormat,
	const uint32_t alphaBinCount)
{
	const auto workgroupDims = getDefaultWorkgroupDims(imageType);
	const auto paddedAlphaBinCount = getPaddedAlphaBinCount(workgroupDims, alphaBinCount);

	const auto castedFormat = getOutImageViewFormat(outFormat);
	const char* glslFormatQualifier = asset::CGLSLCompiler::getStorageImageFormatQualifier(castedFormat);

	std::ostringstream shaderSourceStream;
	shaderSourceStream
		<< "#version 460 core\n"
		<< "#define _NBL_GLSL_WORKGROUP_SIZE_X_ " << ((imageType >= asset::IImage::ET_1D) ? workgroupDims.x : 1) << "\n"
		<< "#define _NBL_GLSL_WORKGROUP_SIZE_Y_ " << ((imageType >= asset::IImage::ET_2D) ? workgroupDims.y : 1) << "\n"
		<< "#define _NBL_GLSL_WORKGROUP_SIZE_Z_ " << ((imageType >= asset::IImage::ET_3D) ? workgroupDims.z : 1) << "\n"
		<< "#define _NBL_GLSL_BLIT_DIM_COUNT_ " << static_cast<uint32_t>(imageType) + 1 << "\n"
		<< "#define _NBL_GLSL_BLIT_ALPHA_BIN_COUNT_ " << paddedAlphaBinCount << "\n"
		<< "#define _NBL_GLSL_BLIT_NORMALIZATION_OUT_IMAGE_FORMAT_ " << glslFormatQualifier << "\n";
	if (outFormat != castedFormat)
		shaderSourceStream << "#define _NBL_GLSL_BLIT_SOFTWARE_ENCODE_FORMAT_ " << outFormat << "\n";
	shaderSourceStream << "#include <nbl/builtin/glsl/blit/default_compute_normalization.comp>\n";

	auto cpuShader = core::make_smart_refctd_ptr<asset::ICPUShader>(shaderSourceStream.str().c_str(), asset::IShader::ESS_COMPUTE, asset::IShader::E_CONTENT_TYPE::ECT_GLSL, "CComputeBlit::createNormalizationSpecializedShader");
	auto gpuUnspecShader = m_device->createShader(std::move(cpuShader));

	return m_device->createSpecializedShader(gpuUnspecShader.get(), { nullptr, nullptr, "main" });
}
#endif