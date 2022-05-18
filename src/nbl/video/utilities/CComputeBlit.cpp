#include "nbl/video/utilities/CComputeBlit.h"

using namespace nbl;
using namespace video;

core::smart_refctd_ptr<asset::ICPUShader> CComputeBlit::getCPUShaderFromGLSL(const system::IFile* glsl)
{
	auto buffer = core::make_smart_refctd_ptr<asset::ICPUBuffer>(glsl->getSize());
	memcpy(buffer->getPointer(), glsl->getMappedPointer(), glsl->getSize());
	auto cpuShader = core::make_smart_refctd_ptr<asset::ICPUShader>(std::move(buffer), asset::IShader::buffer_contains_glsl_t{}, asset::IShader::ESS_COMPUTE, "");

	return cpuShader;
}

core::smart_refctd_ptr<video::IGPUSpecializedShader> CComputeBlit::createAlphaTestSpecializedShader(const asset::IImage::E_TYPE inImageType)
{
	auto system = device->getPhysicalDevice()->getSystem();
	core::smart_refctd_ptr<const system::IFile> glsl = system->loadBuiltinData<NBL_CORE_UNIQUE_STRING_LITERAL_TYPE("nbl/builtin/glsl/blit/default_compute_alpha_test.comp")>();
	auto cpuShader = getCPUShaderFromGLSL(glsl.get());
	if (!cpuShader)
		return nullptr;

	const auto workgroupDims = getDefaultWorkgroupDims(inImageType);

	auto cpuShaderOverriden = asset::IGLSLCompiler::createOverridenCopy(cpuShader.get(),
		"#define _NBL_GLSL_WORKGROUP_SIZE_X_ %d\n"
		"#define _NBL_GLSL_WORKGROUP_SIZE_Y_ %d\n"
		"#define _NBL_GLSL_WORKGROUP_SIZE_Z_ %d\n"
		"#define _NBL_GLSL_BLIT_ALPHA_TEST_DIM_COUNT_ %d\n",
		inImageType >= asset::IImage::ET_1D ? workgroupDims.x : 1u,
		inImageType >= asset::IImage::ET_2D ? workgroupDims.y : 1u,
		inImageType >= asset::IImage::ET_3D ? workgroupDims.z : 1u,
		static_cast<uint32_t>(inImageType) + 1u);

	auto gpuUnspecShader = device->createGPUShader(std::move(cpuShaderOverriden));

	return device->createGPUSpecializedShader(gpuUnspecShader.get(), { nullptr, nullptr, "main" });
}

core::smart_refctd_ptr<video::IGPUSpecializedShader> CComputeBlit::createNormalizationSpecializedShader(const asset::IImage::E_TYPE inImageType, const asset::E_FORMAT outImageFormat, const asset::E_FORMAT outImageViewFormat)
{
	auto system = device->getPhysicalDevice()->getSystem();
	core::smart_refctd_ptr<const system::IFile> glsl = system->loadBuiltinData<NBL_CORE_UNIQUE_STRING_LITERAL_TYPE("nbl/builtin/glsl/blit/default_compute_normalization.comp")>();
	auto cpuShader = getCPUShaderFromGLSL(glsl.get());
	if (!cpuShader)
		return nullptr;

	const char* outImageFormatGLSLString = getGLSLFormatStringFromFormat(outImageFormat); // = _NBL_GLSL_BLIT_FORMAT_R8G8B8A8_SRGB_
	const char* glslFormatQualifier = getGLSLFormatStringFromFormat(outImageViewFormat);

	// Todo(achal): Remove
	char formatInclude[1024] = "";
	if (outImageFormat != outImageViewFormat)
		snprintf(formatInclude, sizeof(formatInclude), "#include <nbl/builtin/glsl/blit/blit/formats/%s.glsl>\n", outImageFormatGLSLString);

	const char* overrideFormat =
		"#define _NBL_GLSL_WORKGROUP_SIZE_X_ %d\n"
		"#define _NBL_GLSL_WORKGROUP_SIZE_Y_ %d\n"
		"#define _NBL_GLSL_WORKGROUP_SIZE_Z_ %d\n"
		"#define _NBL_GLSL_BLIT_NORMALIZATION_BIN_COUNT_ %d\n"
		"#define _NBL_GLSL_BLIT_NORMALIZATION_DIM_COUNT_ %d\n"
		"#define _NBL_GLSL_BLIT_NORMALIZATION_OUT_IMAGE_FORMAT_ %s\n"
		"%s\n" // _NBL_GLSL_BLIT_NORMALIZATION_SOFTWARE_CODEC_
		"%s\n"; // Todo(achal): Remove format include in favour of the symbol gotten from outImageFormatGLSLString

	core::vectorSIMDu32 workgroupDim = getDefaultWorkgroupDims(inImageType);

	auto cpuShaderOverriden = asset::IGLSLCompiler::createOverridenCopy(cpuShader.get(),
		overrideFormat,
		inImageType >= asset::IImage::ET_1D ? workgroupDim.x : 1u,
		inImageType >= asset::IImage::ET_2D ? workgroupDim.y : 1u,
		inImageType >= asset::IImage::ET_3D ? workgroupDim.z : 1u,
		DefaultAlphaBinCount,
		static_cast<uint32_t>(inImageType + 1u),
		glslFormatQualifier,
		// Todo(achal): Replace this by the symbol from outImageFormatGLSLString, if outImageFormat == outImageViewFormat don't put anything so default encode path will run
		outImageFormat != outImageViewFormat ? "#define _NBL_GLSL_BLIT_NORMALIZATION_SOFTWARE_CODEC_\n" : "",
		// Todo(achal): Remove
		reinterpret_cast<const char*>(formatInclude));

	auto gpuUnspecShader = device->createGPUShader(std::move(cpuShaderOverriden));

	return device->createGPUSpecializedShader(gpuUnspecShader.get(), { nullptr, nullptr, "main" });
}

core::smart_refctd_ptr<video::IGPUSpecializedShader> CComputeBlit::createBlitSpecializedShader(const asset::E_FORMAT inImageFormat, const asset::E_FORMAT outImageFormat, const asset::E_FORMAT outImageViewFormat,
	const asset::IImage::E_TYPE inImageType, const core::vectorSIMDu32& inExtent, const core::vectorSIMDu32& outExtent, const asset::IBlitUtilities::E_ALPHA_SEMANTIC alphaSemantic)
{
	auto system = device->getPhysicalDevice()->getSystem();
	core::smart_refctd_ptr<const system::IFile> glsl = system->loadBuiltinData<NBL_CORE_UNIQUE_STRING_LITERAL_TYPE("nbl/builtin/glsl/blit/default_compute_blit.comp")>();
	auto cpuShader = getCPUShaderFromGLSL(glsl.get());
	if (!cpuShader)
		return nullptr;

	// Todo(achal): All this belongs in validation
	const uint32_t inChannelCount = asset::getFormatChannelCount(inImageFormat);
	const uint32_t outChannelCount = asset::getFormatChannelCount(outImageFormat);
	assert(outChannelCount <= inChannelCount);

	core::vectorSIMDf scale = static_cast<core::vectorSIMDf>(inExtent).preciseDivision(static_cast<core::vectorSIMDf>(outExtent));

	const core::vectorSIMDu32 windowDim = static_cast<core::vectorSIMDu32>(core::ceil(scale));
	const uint32_t windowPixelCount = windowDim.x * windowDim.y * windowDim.z;
	// It is important not to use asset::getTexelOrBlockBytesize here.
	// Even though input pixels are stored in the shared memory we use outChannelCount here because only outChannelCount channels of
	// the input image actually need blitting and hence only those channels are stored in shared memory.
	const size_t windowSize = static_cast<size_t>(windowPixelCount) * outChannelCount * sizeof(float);
	// Fail if the window cannot be preloaded into shared memory
	if (windowSize > sharedMemorySize)
		return nullptr;

	// Fail if any dimension of the window is bigger than workgroup size, forcing us to reuse inovcation to process just a single dimension of that window
	if ((windowDim.x > DefaultBlitWorkgroupSize) || (windowDim.y > DefaultBlitWorkgroupSize) || (windowDim.z > DefaultBlitWorkgroupSize))
		return nullptr;

	// Fail if I would need to reuse invocations just to process a single window
	if (windowPixelCount > DefaultBlitWorkgroupSize)
		return nullptr;

	// inFormat should support SAMPLED_BIT format feature

	const char* outImageFormatGLSLString = getGLSLFormatStringFromFormat(outImageFormat);
	const char* glslFormatQualifier = getGLSLFormatStringFromFormat(outImageViewFormat);

	char formatInclude[1024] = "";
	if (outImageFormat != outImageViewFormat)
		snprintf(formatInclude, sizeof(formatInclude), "#include <nbl/builtin/glsl/blit/blit/formats/%s.glsl>\n", outImageFormatGLSLString);

	const uint32_t smemFloatCount = sharedMemorySize / (sizeof(float) * outChannelCount);

	const char* overrideFormat =
		"#define _NBL_GLSL_WORKGROUP_SIZE_ %d\n"
		"#define _NBL_GLSL_BLIT_OUT_CHANNEL_COUNT_ %d\n"
		"#define _NBL_GLSL_BLIT_DIM_COUNT_ %d\n"
		"#define _NBL_GLSL_BLIT_OUT_IMAGE_FORMAT_ %s\n"
		"#define _NBL_GLSL_BLIT_SMEM_FLOAT_COUNT_ %d\n"
		"%s" // _NBL_GLSL_BLIT_COVERAGE_SEMANTIC_
		"%s" // _NBL_GLSL_BLIT_SOFTWARE_CODEC_
		"%s"; // format include

	auto cpuShaderOverriden = asset::IGLSLCompiler::createOverridenCopy(cpuShader.get(),
		overrideFormat,
		DefaultBlitWorkgroupSize,
		outChannelCount,
		static_cast<uint32_t>(inImageType + 1u),
		glslFormatQualifier,
		smemFloatCount,
		alphaSemantic == asset::IBlitUtilities::EAS_REFERENCE_OR_COVERAGE ? "#define _NBL_GLSL_BLIT_COVERAGE_SEMANTIC_\n" : "",
		outImageFormat != outImageViewFormat ? "#define _NBL_GLSL_BLIT_SOFTWARE_CODEC_\n" : "",
		reinterpret_cast<const char*>(formatInclude));

	auto gpuUnspecShader = device->createGPUShader(std::move(cpuShaderOverriden));

	auto specShader = device->createGPUSpecializedShader(gpuUnspecShader.get(), { nullptr, nullptr, "main" });

	return specShader;
}