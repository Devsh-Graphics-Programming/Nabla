#include "nbl/video/utilities/CComputeBlit.h"

using namespace nbl;
using namespace video;

core::smart_refctd_ptr<video::IGPUSpecializedShader> CComputeBlit::createAlphaTestSpecializedShader(const asset::IImage::E_TYPE inImageType, const core::vectorSIMDu32& workgroupDims,
	const uint32_t alphaBinCount)
{
	char shaderStart[] = "#version 460 core\n#include \"nbl/builtin/glsl/blit/default_compute_alpha_test.comp\"\n";
	auto shader = core::make_smart_refctd_ptr<asset::CCustomAllocatorCPUBuffer<core::null_allocator<uint8_t>>>(strlen(shaderStart), shaderStart, core::adopt_memory);

	auto cpuShader = core::make_smart_refctd_ptr<asset::ICPUShader>(std::move(shader), asset::ICPUShader::buffer_contains_glsl, asset::IShader::ESS_COMPUTE, "CComputeBlit::createAlphaTestSpecializedShader");

	auto cpuShaderOverriden = asset::IGLSLCompiler::createOverridenCopy(cpuShader.get(),
		"#define _NBL_GLSL_WORKGROUP_SIZE_X_ %d\n"
		"#define _NBL_GLSL_WORKGROUP_SIZE_Y_ %d\n"
		"#define _NBL_GLSL_WORKGROUP_SIZE_Z_ %d\n"
		"#define _NBL_GLSL_BLIT_DIM_COUNT_ %d\n"
		"#define _NBL_GLSL_BLIT_ALPHA_BIN_COUNT_ %d\n",
		inImageType >= asset::IImage::ET_1D ? workgroupDims.x : 1u,
		inImageType >= asset::IImage::ET_2D ? workgroupDims.y : 1u,
		inImageType >= asset::IImage::ET_3D ? workgroupDims.z : 1u,
		static_cast<uint32_t>(inImageType) + 1u,
		alphaBinCount);

	auto gpuUnspecShader = device->createGPUShader(std::move(cpuShaderOverriden));

	return device->createGPUSpecializedShader(gpuUnspecShader.get(), { nullptr, nullptr, "main" });
}

core::smart_refctd_ptr<video::IGPUSpecializedShader> CComputeBlit::createNormalizationSpecializedShader(const asset::IImage::E_TYPE inImageType, const asset::E_FORMAT outImageFormat,
	const asset::E_FORMAT outImageViewFormat, const core::vectorSIMDu32& workgroupDims, const uint32_t alphaBinCount)
{
	char shaderStart[] = "#version 460 core\n#include \"nbl/builtin/glsl/blit/default_compute_normalization.comp\"\n";
	auto shader = core::make_smart_refctd_ptr<asset::CCustomAllocatorCPUBuffer<core::null_allocator<uint8_t>>>(strlen(shaderStart), shaderStart, core::adopt_memory);

	auto cpuShader = core::make_smart_refctd_ptr<asset::ICPUShader>(std::move(shader), asset::ICPUShader::buffer_contains_glsl, asset::IShader::ESS_COMPUTE, "CComputeBlit::createNormalizationSpecializedShader");

	const char* outImageFormatGLSLString = getGLSLFormatStringFromFormat(outImageFormat);
	const char* glslFormatQualifier = getGLSLFormatStringFromFormat(outImageViewFormat);

	// Todo(achal): Remove
	char formatInclude[1024] = "";
	if (outImageFormat != outImageViewFormat)
		snprintf(formatInclude, sizeof(formatInclude), "#include <nbl/builtin/glsl/blit/blit/formats/%s.glsl>\n", outImageFormatGLSLString);

	auto cpuShaderOverriden = asset::IGLSLCompiler::createOverridenCopy(cpuShader.get(),
		"#define _NBL_GLSL_WORKGROUP_SIZE_X_ %d\n"
		"#define _NBL_GLSL_WORKGROUP_SIZE_Y_ %d\n"
		"#define _NBL_GLSL_WORKGROUP_SIZE_Z_ %d\n"
		"#define _NBL_GLSL_BLIT_DIM_COUNT_ %d\n"
		"#define _NBL_GLSL_BLIT_ALPHA_BIN_COUNT_ %d\n"
		"#define _NBL_GLSL_BLIT_NORMALIZATION_OUT_IMAGE_FORMAT_ %s\n"
		"%s\n" // _NBL_GLSL_BLIT_NORMALIZATION_SOFTWARE_CODEC_
		"%s\n", // Todo(achal): Remove format include in favour of the symbol gotten from outImageFormatGLSLString
		inImageType >= asset::IImage::ET_1D ? workgroupDims.x : 1u,
		inImageType >= asset::IImage::ET_2D ? workgroupDims.y : 1u,
		inImageType >= asset::IImage::ET_3D ? workgroupDims.z : 1u,
		static_cast<uint32_t>(inImageType) + 1u,
		alphaBinCount,
		glslFormatQualifier,
		outImageFormat != outImageViewFormat ? "#define _NBL_GLSL_BLIT_NORMALIZATION_SOFTWARE_CODEC_\n" : "",
		reinterpret_cast<const char*>(formatInclude));

	auto gpuUnspecShader = device->createGPUShader(std::move(cpuShaderOverriden));

	return device->createGPUSpecializedShader(gpuUnspecShader.get(), { nullptr, nullptr, "main" });
}

core::smart_refctd_ptr<video::IGPUSpecializedShader> CComputeBlit::createBlitSpecializedShader(const asset::E_FORMAT inImageFormat, const asset::E_FORMAT outImageFormat,
	const asset::E_FORMAT outImageViewFormat, const asset::IImage::E_TYPE inImageType, const core::vectorSIMDu32& inExtent, const core::vectorSIMDu32& outExtent,
	const asset::IBlitUtilities::E_ALPHA_SEMANTIC alphaSemantic, const uint32_t workgroupSize, const uint32_t alphaBinCount)
{
	char shaderStart[] = "#version 460 core\n#include \"nbl/builtin/glsl/blit/default_compute_blit.comp\"\n";
	auto shader = core::make_smart_refctd_ptr<asset::CCustomAllocatorCPUBuffer<core::null_allocator<uint8_t>>>(strlen(shaderStart), shaderStart, core::adopt_memory);

	auto cpuShader = core::make_smart_refctd_ptr<asset::ICPUShader>(std::move(shader), asset::ICPUShader::buffer_contains_glsl, asset::IShader::ESS_COMPUTE, "CComputeBlit::createBlitSpecializedShader");
	
	const core::vectorSIMDu32 windowDim = getWindowDim(inExtent, outExtent);

	// Fail if any dimension of the window is bigger than workgroup size, forcing us to reuse inovcation to process just a single dimension of that window
	if ((windowDim.x > workgroupSize) || (windowDim.y > workgroupSize) || (windowDim.z > workgroupSize))
		return nullptr;

	const uint32_t windowPixelCount = windowDim.x * windowDim.y * windowDim.z;
	// Fail if I would need to reuse invocations just to process a single window
	if (windowPixelCount > workgroupSize)
		return nullptr;

	const uint32_t outChannelCount = asset::getFormatChannelCount(outImageFormat);
	// It is important not to use asset::getTexelOrBlockBytesize here.
	// Even though input pixels are stored in the shared memory we use outChannelCount here because only outChannelCount channels of
	// the input image actually need blitting and hence only those channels are stored in shared memory.
	const size_t windowSize = static_cast<size_t>(windowPixelCount) * outChannelCount * sizeof(float);
	// Fail if the window cannot be preloaded into shared memory
	// Todo(achal): Take smem size as a param of this function?
	if (windowSize > sharedMemorySize)
		return nullptr;

	// Todo(achal): All this belongs in validation
	{
		const uint32_t inChannelCount = asset::getFormatChannelCount(inImageFormat);
		assert(outChannelCount <= inChannelCount);

		// inFormat should support SAMPLED_BIT format feature
	}

	const char* outImageFormatGLSLString = getGLSLFormatStringFromFormat(outImageFormat);
	const char* glslFormatQualifier = getGLSLFormatStringFromFormat(outImageViewFormat);

	char formatInclude[1024] = "";
	if (outImageFormat != outImageViewFormat)
		snprintf(formatInclude, sizeof(formatInclude), "#include <nbl/builtin/glsl/blit/blit/formats/%s.glsl>\n", outImageFormatGLSLString);

	// Todo(achal): Get it from outside
	const uint32_t wgSize = DefaultBlitWorkgroupSize;

	const uint32_t smemFloatCount = sharedMemorySize / (sizeof(float) * outChannelCount);

	const char* overrideFormat =
		"#define _NBL_GLSL_WORKGROUP_SIZE_X_ %d\n"
		"#define _NBL_GLSL_WORKGROUP_SIZE_Y_ %d\n"
		"#define _NBL_GLSL_WORKGROUP_SIZE_Z_ %d\n"
		"#define _NBL_GLSL_BLIT_DIM_COUNT_ %d\n"
		"#define _NBL_GLSL_BLIT_ALPHA_BIN_COUNT_ %d\n"

		"#define _NBL_GLSL_BLIT_OUT_CHANNEL_COUNT_ %d\n"
		"#define _NBL_GLSL_BLIT_OUT_IMAGE_FORMAT_ %s\n"
		"#define _NBL_GLSL_BLIT_SMEM_FLOAT_COUNT_ %d\n"
		"%s" // _NBL_GLSL_BLIT_COVERAGE_SEMANTIC_
		"%s" // _NBL_GLSL_BLIT_SOFTWARE_CODEC_
		"%s"; // format include

	auto cpuShaderOverriden = asset::IGLSLCompiler::createOverridenCopy(cpuShader.get(),
		overrideFormat,
		DefaultBlitWorkgroupSize,
		1, 1,
		static_cast<uint32_t>(inImageType + 1u),
		alphaBinCount,
		outChannelCount,
		glslFormatQualifier,
		smemFloatCount,
		alphaSemantic == asset::IBlitUtilities::EAS_REFERENCE_OR_COVERAGE ? "#define _NBL_GLSL_BLIT_COVERAGE_SEMANTIC_\n" : "",
		outImageFormat != outImageViewFormat ? "#define _NBL_GLSL_BLIT_SOFTWARE_CODEC_\n" : "",
		reinterpret_cast<const char*>(formatInclude));

	auto gpuUnspecShader = device->createGPUShader(std::move(cpuShaderOverriden));

	auto specShader = device->createGPUSpecializedShader(gpuUnspecShader.get(), { nullptr, nullptr, "main" });

	return specShader;
}