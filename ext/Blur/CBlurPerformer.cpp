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
constexpr const char* CS_DOWNSAMPLE_SRC = R"XDDD(
    #version 430 core
    layout(local_size_x = 16, local_size_y = 16) in; // 16*16==256

    layout(std430, binding = 0) restrict writeonly buffer b0 {
	    vec3 ssbo0[];
    };

    layout(binding = 0, location = 0) uniform sampler2D in_tex;

    void main()
    {
        const uvec2 IDX = gl_GlobalInvocationID.xy; // each index corresponds to one pixel in downsampled texture
	
	    const ivec2 OUT_SIZE = ivec2(512, 512);
	
	    vec2 coords = vec2(1.f) / vec2(OUT_SIZE);
	    vec4 avg = (
		    texture(in_tex, min(coords * vec2(IDX), vec2(1, 1))) +
		    texture(in_tex, min(coords * (vec2(IDX) + vec2(1.f, 0.f)), vec2(1, 1))) +
		    texture(in_tex, min(coords * (vec2(IDX) + vec2(0.f, 1.f)), vec2(1, 1))) +
		    texture(in_tex, min(coords * (vec2(IDX) + vec2(1.f, 1.f)), vec2(1, 1)))
	    ) / 4.f;

	    const uint HBUF_IDX = IDX.y * OUT_SIZE.x + IDX.x;
	
	    ssbo0[HBUF_IDX] = avg.xyz;
    }
)XDDD";
constexpr const char* CS_BLUR_SRC = R"XDDD(
    #version 430 core
    layout(local_size_x = 16, local_size_y = 16) in; // 16*16==256

    layout(std430, binding = 0) restrict buffer b0 {
	    vec3 ssbo0[];
    };
    layout(std430, binding = 1) restrict buffer b1 {
	    vec3 ssbo1[];
    };

    #define IN_BUF %s
    #define OUT_BUF %s
    #define IN_IDX %s
    #define OUT_IDX %s
    #define FINAL_PASS %d

    #if FINAL_PASS
    layout(location = 1, binding = 0, rgba8) uniform writeonly image2D out_img;
    #endif

    void main()
    {
	    const uvec2 IDX = gl_GlobalInvocationID.xy; // each index corresponds to one pixel in downsampled texture
	
	    const ivec2 OUT_SIZE = ivec2(512, 512);

	    const uint HBUF_IDX = IDX.y * OUT_SIZE.x + IDX.x;
	
	    const int VBUF_IDX = int(IDX.x) * OUT_SIZE.y + int(IDX.y);

	    vec3 res = vec3(0.f);
	    for (int i = -4; i < 5; ++i)
		    res += IN_BUF[IN_IDX + i];
	    res/=9.f;
    #if FINAL_PASS
        imageStore(out_img, ivec2(IDX), vec4(res, 1.f));
    #else
        OUT_BUF[OUT_IDX] = res;
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

CBlurPerformer* CBlurPerformer::instantiate(video::IVideoDriver* _driver)
{
    unsigned ds{}, hb1{}, hb2{}, hb3{}, vb1{}, vb2{}, vb3{};

    ds = createComputeShader(CS_DOWNSAMPLE_SRC);
    
    const size_t bufSize = strlen(CS_BLUR_SRC) + 100u;
    char* src = (char*)malloc(bufSize);

    auto deleteShadersF = [&, ds, hb1, hb2, hb3, vb1, vb2, vb3, src]() {
        for (auto s : { ds, hb1, hb2, hb3, vb1, vb2, vb3 })
            video::COpenGLExtensionHandler::extGlDeleteProgram(s);
        free(src);
        return nullptr;
    };

    if (!genBlurPassCs(src, bufSize, "ssbo0", "ssbo1", "HBUF_IDX", "HBUF_IDX", 0))
        return nullptr;
    hb1 = createComputeShader(src);
    
    if (!genBlurPassCs(src, bufSize, "ssbo1", "ssbo0", "HBUF_IDX", "HBUF_IDX", 0))
        return deleteShadersF();
    hb2 = createComputeShader(src);

    if (!genBlurPassCs(src, bufSize, "ssbo0", "ssbo1", "HBUF_IDX", "VBUF_IDX", 0))
        return deleteShadersF();
    hb3 = createComputeShader(src);

    if (!genBlurPassCs(src, bufSize, "ssbo1", "ssbo0", "VBUF_IDX", "VBUF_IDX", 0))
        return deleteShadersF();
    vb1 = createComputeShader(src);

    if (!genBlurPassCs(src, bufSize, "ssbo0", "ssbo1", "VBUF_IDX", "VBUF_IDX", 0))
        return deleteShadersF();
    vb2 = createComputeShader(src);

    if (!genBlurPassCs(src, bufSize, "ssbo1", "ssbo0", "VBUF_IDX", "whatever", 1))
        return deleteShadersF();
    vb3 = createComputeShader(src);

    for (unsigned s : {ds, hb1, hb2, hb3, vb1, vb2, vb3})
    {
        if (!s)
            return deleteShadersF();
    }
    free(src);

    return new CBlurPerformer(_driver, ds, hb1, hb2, hb3, vb1, vb2, vb3);
}

video::ITexture* CBlurPerformer::createBlurredTexture(const video::ITexture* _inputTex) const
{
    video::COpenGLExtensionHandler::extGlBindBuffersBase(GL_SHADER_STORAGE_BUFFER, E_SSBO0_BINDING, 1, &static_cast<video::COpenGLBuffer*>(m_ssbo0)->getOpenGLName());
    video::COpenGLExtensionHandler::extGlBindBuffersBase(GL_SHADER_STORAGE_BUFFER, E_SSBO1_BINDING, 1, &static_cast<video::COpenGLBuffer*>(m_ssbo1)->getOpenGLName());

    const GLenum target = GL_TEXTURE_2D;
    video::COpenGLExtensionHandler::extGlBindTextures(0, 1, &static_cast<const video::COpenGL2DTexture*>(_inputTex)->getOpenGLName(), &target);

    const uint32_t size[]{ 512u, 512u };
    video::ITexture* outputTex = m_driver->addTexture(video::ITexture::ETT_2D, size, 1, ("blur_out" + std::to_string(s_texturesEverCreatedCount++)).c_str(), video::ECF_A8R8G8B8);

    video::COpenGLExtensionHandler::extGlBindImageTexture(0, static_cast<const video::COpenGL2DTexture*>(outputTex)->getOpenGLName(),
        0, GL_FALSE, 0, GL_WRITE_ONLY, video::COpenGLTexture::getOpenGLFormatAndParametersFromColorFormat(static_cast<const video::COpenGL2DTexture*>(outputTex)->getColorFormat()));

    video::COpenGLExtensionHandler::extGlUseProgram(m_dsampleCs);

    video::COpenGLExtensionHandler::extGlDispatchCompute(32u, 32u, 1u);
    video::COpenGLExtensionHandler::extGlMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    for (size_t i = 0u; i < 5u; ++i)
    {
        video::COpenGLExtensionHandler::extGlUseProgram(m_blurCs[i]);

        video::COpenGLExtensionHandler::extGlDispatchCompute(32u, 32u, 1u);
        video::COpenGLExtensionHandler::extGlMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    }
    video::COpenGLExtensionHandler::extGlUseProgram(m_blurCs[5]);
    video::COpenGLExtensionHandler::extGlDispatchCompute(32u, 32u, 1u);
    video::COpenGLExtensionHandler::extGlMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT);

    return outputTex;
}

CBlurPerformer::~CBlurPerformer()
{
    m_ssbo0->drop();
    m_ssbo1->drop();
    video::COpenGLExtensionHandler::extGlDeleteProgram(m_dsampleCs);
    for (size_t i = 0u; i < 6u; ++i)
        video::COpenGLExtensionHandler::extGlDeleteProgram(m_blurCs[i]);
}

bool CBlurPerformer::genBlurPassCs(char * _out, size_t _outSize, const char * _inBufName, const char * _outBufName, const char * _inIdxName, const char * _outIdxName, int _finalPass)
{
    return snprintf(_out, _outSize, CS_BLUR_SRC, _inBufName, _outBufName, _inIdxName, _outIdxName, _finalPass) > 0;
}
