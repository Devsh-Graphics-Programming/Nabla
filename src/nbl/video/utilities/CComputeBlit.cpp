#include "nbl/video/utilities/CComputeBlit.h"

using namespace nbl;
using namespace video;

core::smart_refctd_ptr<video::IGPUSpecializedShader> CComputeBlit::createAlphaTestSpecializedShader(const asset::IImage::E_TYPE inImageType, const core::vectorSIMDu32& workgroupDims,
	const uint32_t alphaBinCount)
{
	std::ostringstream shaderSourceStream;
	shaderSourceStream
		<< "#version 460 core\n"
		<< "#define _NBL_GLSL_WORKGROUP_SIZE_X_ " << ((inImageType >= asset::IImage::ET_1D) ? workgroupDims.x : 1) << "\n"
		<< "#define _NBL_GLSL_WORKGROUP_SIZE_Y_ " << ((inImageType >= asset::IImage::ET_2D) ? workgroupDims.y : 1) << "\n"
		<< "#define _NBL_GLSL_WORKGROUP_SIZE_Z_ " << ((inImageType >= asset::IImage::ET_3D) ? workgroupDims.z : 1) << "\n"
		<< "#define _NBL_GLSL_BLIT_DIM_COUNT_ " << static_cast<uint32_t>(inImageType) + 1 << "\n"
		<< "#define _NBL_GLSL_BLIT_ALPHA_BIN_COUNT_ " << alphaBinCount << "\n"
		<< "#include <nbl/builtin/glsl/blit/default_compute_alpha_test.comp>\n";

	auto cpuShader = core::make_smart_refctd_ptr<asset::ICPUShader>(shaderSourceStream.str().c_str(), asset::IShader::ESS_COMPUTE, "CComputeBlit::createAlphaTestSpecializedShader");
	auto gpuUnspecShader = m_device->createShader(std::move(cpuShader));

	return m_device->createSpecializedShader(gpuUnspecShader.get(), { nullptr, nullptr, "main" });
}

core::smart_refctd_ptr<video::IGPUSpecializedShader> CComputeBlit::createNormalizationSpecializedShader(const asset::IImage::E_TYPE inImageType, const asset::E_FORMAT outImageFormat,
	const asset::E_FORMAT outImageViewFormat, const core::vectorSIMDu32& workgroupDims, const uint32_t alphaBinCount)
{
	const char* outImageFormatGLSLString = asset::IGLSLCompiler::getStorageImageFormatQualifier(outImageFormat);
	const char* glslFormatQualifier = asset::IGLSLCompiler::getStorageImageFormatQualifier(outImageViewFormat);

	std::ostringstream shaderSourceStream;
	shaderSourceStream
		<< "#version 460 core\n"
		<< "#define _NBL_GLSL_WORKGROUP_SIZE_X_ " << ((inImageType >= asset::IImage::ET_1D) ? workgroupDims.x : 1) << "\n"
		<< "#define _NBL_GLSL_WORKGROUP_SIZE_Y_ " << ((inImageType >= asset::IImage::ET_2D) ? workgroupDims.y : 1) << "\n"
		<< "#define _NBL_GLSL_WORKGROUP_SIZE_Z_ " << ((inImageType >= asset::IImage::ET_3D) ? workgroupDims.z : 1) << "\n"
		<< "#define _NBL_GLSL_BLIT_DIM_COUNT_ " << static_cast<uint32_t>(inImageType) + 1 << "\n"
		<< "#define _NBL_GLSL_BLIT_ALPHA_BIN_COUNT_ " << alphaBinCount << "\n"
		<< "#define _NBL_GLSL_BLIT_NORMALIZATION_OUT_IMAGE_FORMAT_ " << glslFormatQualifier << "\n"
		<< ((outImageFormat != outImageViewFormat) ? "#define _NBL_GLSL_BLIT_NORMALIZATION_SOFTWARE_CODEC_" : "") << "\n";
	if (outImageFormat != outImageViewFormat)
		shaderSourceStream << "#define _NBL_GLSL_BLIT_SOFTWARE_ENCODE_FORMAT_ " << outImageFormat << "\n";
	shaderSourceStream << "#include <nbl/builtin/glsl/blit/default_compute_normalization.comp>\n";

	auto cpuShader = core::make_smart_refctd_ptr<asset::ICPUShader>(shaderSourceStream.str().c_str(), asset::IShader::ESS_COMPUTE, "CComputeBlit::createNormalizationSpecializedShader");
	auto gpuUnspecShader = m_device->createShader(std::move(cpuShader));

	return m_device->createSpecializedShader(gpuUnspecShader.get(), { nullptr, nullptr, "main" });
}