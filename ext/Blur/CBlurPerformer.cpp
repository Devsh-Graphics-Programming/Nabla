#include "../../ext/Blur/CBlurPerformer.h"

#include "../../source/Irrlicht/COpenGLBuffer.h"
#include "../../source/Irrlicht/COpenGLDriver.h"
#include "../../source/Irrlicht/COpenGLExtensionHandler.h"
#include "../../source/Irrlicht/COpenGL2DTexture.h"

using namespace irr;
using namespace ext;
using namespace Blur;

namespace // traslation-unit-local things
{
constexpr const char* CS_CONVERSIONS = R"XDDD(
uvec2 encodeRgb(vec3 _rgb)
{
    uvec2 ret;
    ret.x = packHalf2x16(_rgb.rg);
    ret.y = packHalf2x16(vec2(_rgb.b, 0.f));

    return ret;
}
vec3 decodeRgb(uvec2 _rgb)
{
    vec3 ret;
    ret.rg = unpackHalf2x16(_rgb.x);
    ret.b = unpackHalf2x16(_rgb.y).x;

    return ret;
}
)XDDD";
constexpr const char* CS_DOWNSAMPLE_SRC = R"XDDD(
#version 430 core
layout(local_size_x = 16, local_size_y = 16) in;

layout(std430, binding = 0) restrict writeonly buffer b {
	uvec2 ssbo[];
};

layout(binding = 0, location = 0) uniform sampler2D in_tex;

%s

void main()
{
    const uvec2 IDX = gl_GlobalInvocationID.xy; // each index corresponds to one pixel in downsampled texture
	
    const ivec2 OUT_SIZE = ivec2(512, 512);
	
    vec2 coords = vec2(1.f) / vec2(OUT_SIZE);
    vec4 avg = (
        texture(in_tex, coords * vec2(IDX)) +
        texture(in_tex, coords * (vec2(IDX) + vec2(1.f, 0.f))) +
        texture(in_tex, coords * (vec2(IDX) + vec2(0.f, 1.f))) +
        texture(in_tex, coords * (vec2(IDX) + vec2(1.f, 1.f)))
    ) / 4.f;

    const uint HBUF_IDX = IDX.y * OUT_SIZE.x + IDX.x;

    ssbo[HBUF_IDX] = encodeRgb(avg.xyz);
}
)XDDD";
constexpr const char* CS_BLUR_SRC = R"XDDD(
#version 430 core
layout(local_size_x = 512) in;

layout(std430, binding = 0) restrict buffer b {
	uvec2 ssbo[];
};

#define FINAL_PASS %d
#define RADIUS 4

#if FINAL_PASS
layout(location = 1, binding = 0, rgba16f) uniform writeonly image2D out_img;
#endif

layout(location = 2) uniform uint inOffset;
layout(location = 3) uniform uint outOffset;
layout(location = 4) uniform uvec2 inMlt;
layout(location = 5) uniform uvec2 outMlt;

shared float smem_r[512];
shared float smem_g[512];
shared float smem_b[512];

%s

void storeShared(uint _idx, uvec2 _val)
{
    const vec3 rgb = decodeRgb(_val);
    smem_r[_idx] = rgb.r;
    smem_g[_idx] = rgb.g;
    smem_b[_idx] = rgb.b;
}
vec3 loadShared(uint _idx)
{
    return vec3(smem_r[_idx], smem_g[_idx], smem_b[_idx]);
}

void main()
{
	uvec2 IDX = gl_GlobalInvocationID.xy;
    if (inMlt == uvec2(512, 1))
        IDX = IDX.yx;

    const uint L_IDX = gl_LocalInvocationIndex;
    const uint READ_IDX = uint(dot(IDX, inMlt)) + inOffset;
        
    storeShared(L_IDX, ssbo[READ_IDX]);

    memoryBarrierShared();
    barrier();

	vec3 res = vec3(0.f);
	for (int i = -RADIUS; i <= RADIUS; ++i)
		res += loadShared(clamp(int(L_IDX) + i, 0, 512-1));
	res /= (float(2*RADIUS) + 1.f);

#if FINAL_PASS
    imageStore(out_img, ivec2(IDX), vec4(res, 1.f));
#else
    ssbo[uint(dot(IDX, outMlt)) + outOffset] = encodeRgb(res);
#endif
}
)XDDD";

inline unsigned createComputeShader(const char* _src)
{
    unsigned program = video::COpenGLExtensionHandler::extGlCreateProgram();
	unsigned cs = video::COpenGLExtensionHandler::extGlCreateShader(GL_COMPUTE_SHADER);

	video::COpenGLExtensionHandler::extGlShaderSource(cs, 1, const_cast<const char**>(&_src), NULL);
	video::COpenGLExtensionHandler::extGlCompileShader(cs);

	// check for compilation errors
    GLint success;
    GLchar infoLog[0x200];
    video::COpenGLExtensionHandler::extGlGetShaderiv(cs, GL_COMPILE_STATUS, &success);
    if (!success)
    {
        video::COpenGLExtensionHandler::extGlGetShaderInfoLog(cs, sizeof(infoLog), nullptr, infoLog);
        os::Printer::log("CS COMPILATION ERROR:\n", infoLog, ELL_ERROR);
        video::COpenGLExtensionHandler::extGlDeleteShader(cs);
        video::COpenGLExtensionHandler::extGlDeleteProgram(program);
        return 0;
	}

	video::COpenGLExtensionHandler::extGlAttachShader(program, cs);
	video::COpenGLExtensionHandler::extGlLinkProgram(program);

	//check linking errors
	success = 0;
    video::COpenGLExtensionHandler::extGlGetProgramiv(program, GL_LINK_STATUS, &success);
    if (success == GL_FALSE)
    {
        video::COpenGLExtensionHandler::extGlGetProgramInfoLog(program, sizeof(infoLog), nullptr, infoLog);
        os::Printer::log("CS LINK ERROR:\n", infoLog, ELL_ERROR);
        video::COpenGLExtensionHandler::extGlDeleteShader(cs);
        video::COpenGLExtensionHandler::extGlDeleteProgram(program);
        return 0;
    }

	return program;
}
}

uint32_t CBlurPerformer::s_texturesEverCreatedCount{};
core::vector2d<size_t> CBlurPerformer::s_outTexSize(512u, 512); // if changing here, remember to also change in dsample shader source

CBlurPerformer* CBlurPerformer::instantiate(video::IVideoDriver* _driver)
{
    unsigned ds{}, gblur{}, fblur{};

    const size_t bufSize = std::max(strlen(CS_BLUR_SRC), strlen(CS_DOWNSAMPLE_SRC)) + strlen(CS_CONVERSIONS) + 100u;
    char* src = (char*)malloc(bufSize);

    auto doCleaning = [&, ds, gblur, fblur, src]() {
        for (unsigned s : { ds, gblur, fblur })
            video::COpenGLExtensionHandler::extGlDeleteProgram(s);
        free(src);
        return nullptr;
    };

    if (!genDsampleCs(src, bufSize))
        return doCleaning();
    ds = createComputeShader(src);
    
    if (!genBlurPassCs(src, bufSize, 0))
        return doCleaning();
    gblur = createComputeShader(src);
    
    if (!genBlurPassCs(src, bufSize, 1))
        return doCleaning();
    fblur = createComputeShader(src);

    for (unsigned s : {ds, gblur, fblur})
    {
        if (!s)
            return doCleaning();
    }
    free(src);

    return new CBlurPerformer(_driver, ds, gblur, fblur);
}

video::ITexture* CBlurPerformer::createBlurredTexture(video::ITexture* _inputTex) const
{
    video::COpenGLExtensionHandler::extGlBindBuffersBase(GL_SHADER_STORAGE_BUFFER, E_SSBO_BINDING, 1, &static_cast<video::COpenGLBuffer*>(m_ssbo)->getOpenGLName());

    {
    video::STextureSamplingParams params;
    params.UseMipmaps = 0;
    params.MaxFilter = params.MinFilter = video::ETFT_LINEAR_NO_MIP;
    params.TextureWrapU = params.TextureWrapV = video::ETC_CLAMP_TO_EDGE;
    const_cast<video::COpenGLDriver::SAuxContext*>(reinterpret_cast<video::COpenGLDriver*>(m_driver)->getThreadContext())->setActiveTexture(0, _inputTex, params);
    }

    const uint32_t size[]{ s_outTexSize.X, s_outTexSize.Y};
    video::ITexture* outputTex = m_driver->addTexture(video::ITexture::ETT_2D, size, 1, ("blur_out" + std::to_string(s_texturesEverCreatedCount++)).c_str(), video::ECF_A16B16G16R16F);

    video::COpenGLExtensionHandler::extGlBindImageTexture(0, static_cast<const video::COpenGL2DTexture*>(outputTex)->getOpenGLName(),
        0, GL_FALSE, 0, GL_WRITE_ONLY, video::COpenGLTexture::getOpenGLFormatAndParametersFromColorFormat(static_cast<const video::COpenGL2DTexture*>(outputTex)->getColorFormat()));



    video::COpenGLExtensionHandler::extGlUseProgram(m_dsampleCs);

    video::COpenGLExtensionHandler::extGlDispatchCompute(32u, 32u, 1u);
    video::COpenGLExtensionHandler::extGlMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    unsigned inOffset{}, outOffset{};
    const core::vector2d<unsigned> HMLT(1u, s_outTexSize.X), VMLT(s_outTexSize.Y, 1u);
    const core::vector2d<unsigned> imultipliers[5]{ HMLT, HMLT, HMLT, VMLT, VMLT };
    const core::vector2d<unsigned> omultipliers[5]{ HMLT, HMLT, VMLT, VMLT, VMLT };

    inOffset = 0u;
    outOffset = s_outTexSize.X * s_outTexSize.Y;

    video::COpenGLExtensionHandler::extGlUseProgram(m_blurGeneralCs);
    for (size_t i = 0u; i < 5u; ++i)
    {
        video::COpenGLExtensionHandler::extGlProgramUniform2uiv(m_blurGeneralCs, E_IN_MLT_LOC, 1, &imultipliers[i].X);
        video::COpenGLExtensionHandler::extGlProgramUniform2uiv(m_blurGeneralCs, E_OUT_MLT_LOC, 1, &omultipliers[i].X);
        video::COpenGLExtensionHandler::extGlProgramUniform1uiv(m_blurGeneralCs, E_IN_OFFSET_LOC, 1, &inOffset);
        video::COpenGLExtensionHandler::extGlProgramUniform1uiv(m_blurGeneralCs, E_OUT_OFFSET_LOC, 1, &outOffset);

        video::COpenGLExtensionHandler::extGlDispatchCompute(1u, (i >= 3u ? s_outTexSize.X : s_outTexSize.Y), 1u);
        video::COpenGLExtensionHandler::extGlMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

        std::swap(inOffset, outOffset);
    }

    video::COpenGLExtensionHandler::extGlUseProgram(m_blurFinalCs);

    video::COpenGLExtensionHandler::extGlProgramUniform2uiv(m_blurFinalCs, E_IN_MLT_LOC, 1, &VMLT.X);
    video::COpenGLExtensionHandler::extGlProgramUniform1uiv(m_blurFinalCs, E_IN_OFFSET_LOC, 1, &inOffset);
    video::COpenGLExtensionHandler::extGlProgramUniform1uiv(m_blurFinalCs, E_OUT_OFFSET_LOC, 1, &outOffset);

    video::COpenGLExtensionHandler::extGlDispatchCompute(1u, s_outTexSize.X, 1u);
    video::COpenGLExtensionHandler::extGlMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT);

    return outputTex;
}

CBlurPerformer::~CBlurPerformer()
{
    m_ssbo->drop();
    video::COpenGLExtensionHandler::extGlDeleteProgram(m_dsampleCs);
    video::COpenGLExtensionHandler::extGlDeleteProgram(m_blurGeneralCs);
    video::COpenGLExtensionHandler::extGlDeleteProgram(m_blurFinalCs);
}

bool CBlurPerformer::genDsampleCs(char* _out, size_t _outSize)
{
    return snprintf(_out, _outSize, CS_DOWNSAMPLE_SRC, CS_CONVERSIONS) > 0;
}
bool CBlurPerformer::genBlurPassCs(char* _out, size_t _outSize, int _finalPass)
{
    return snprintf(_out, _outSize, CS_BLUR_SRC, _finalPass, CS_CONVERSIONS) > 0;
}
