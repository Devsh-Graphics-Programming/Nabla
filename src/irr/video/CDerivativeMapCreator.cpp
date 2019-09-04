#include <irrlicht.h>
#include "../source/Irrlicht/COpenGL2DTexture.h"
#include "../source/Irrlicht/COpenGLDriver.h"
#include "irr/video/CDerivativeMapCreator.h"

using namespace irr;
using namespace video;

uint32_t CDerivativeMapCreator::createComputeShader(const char* _src) const
{
    video::COpenGLDriver* gldriver = static_cast<video::COpenGLDriver*>(m_driver);

    uint32_t program = gldriver->extGlCreateProgram();
    uint32_t cs = gldriver->extGlCreateShader(GL_COMPUTE_SHADER);

    gldriver->extGlShaderSource(cs, 1, const_cast<const char**>(&_src), NULL);
    gldriver->extGlCompileShader(cs);

    // check for compilation errors
    GLint success;
    GLchar infoLog[0x200];
    gldriver->extGlGetShaderiv(cs, GL_COMPILE_STATUS, &success);
    if (!success)
    {
        gldriver->extGlGetShaderInfoLog(cs, sizeof(infoLog), nullptr, infoLog);
        os::Printer::log("CS COMPILATION ERROR:\n", infoLog, ELL_ERROR);
        gldriver->extGlDeleteShader(cs);
        gldriver->extGlDeleteProgram(program);
        return 0;
    }

    gldriver->extGlAttachShader(program, cs);
    gldriver->extGlLinkProgram(program);

    //check linking errors
    success = 0;
    gldriver->extGlGetProgramiv(program, GL_LINK_STATUS, &success);
    if (success == GL_FALSE)
    {
        gldriver->extGlGetProgramInfoLog(program, sizeof(infoLog), nullptr, infoLog);
        os::Printer::log("CS LINK ERROR:\n", infoLog, ELL_ERROR);
        gldriver->extGlDeleteShader(cs);
        gldriver->extGlDeleteProgram(program);
        return 0;
    }

    return program;
}

namespace
{
    constexpr const char* DERIV_MAP_FROM_BUMP_MAP_CS_SRC = R"(
#version 450 core

layout (local_size_x = 16, local_size_y = 16) in;

layout (binding = 7) uniform sampler2D bumpMapSampler;
layout (binding = 0, rg8_snorm) uniform image2D derivativeMapImage;

layout (location = 0) uniform float uHeightScaleFactor;

shared float smem[324];//18*18

int getAddr(in ivec2 threadID)
{
	return 18*(threadID.y+1) + threadID.x+1;
}

void main()
{
	const ivec2 G_IDX = ivec2(gl_GlobalInvocationID.xy);
	
	const ivec2 bumpMapSz = textureSize(bumpMapSampler, 0);

	const ivec2 LC_IDX = ivec2(gl_LocalInvocationID.xy);
	smem[getAddr(LC_IDX)] = texelFetch(bumpMapSampler, G_IDX, 0).x * uHeightScaleFactor;
	if (LC_IDX.x == 0 || LC_IDX.x == 15) // TODO how not to use if-statements??
	{
		ivec2 offset = ivec2(mix(ivec2(1, 0), ivec2(-1, 0), LC_IDX.x == 0));
		smem[getAddr(LC_IDX+offset)] = texelFetch(bumpMapSampler, G_IDX+offset, 0).x * uHeightScaleFactor;
	}
	if (LC_IDX.y == 0 || LC_IDX.y == 15)
	{
		ivec2 offset = ivec2(mix(ivec2(0, 1), ivec2(0, -1), LC_IDX.y == 0));
		smem[getAddr(LC_IDX+offset)] = texelFetch(bumpMapSampler, G_IDX+offset, 0).x * uHeightScaleFactor;
	}
	
	barrier();
	memoryBarrierShared();
	
	vec2 d = vec2(
		smem[getAddr(LC_IDX+ivec2(1, 0))] - smem[getAddr(LC_IDX-ivec2(1, 0))],
		smem[getAddr(LC_IDX+ivec2(0, 1))] - smem[getAddr(LC_IDX-ivec2(0, 1))]
	) * 0.5 * vec2(bumpMapSz);
	
	if (all(lessThan(G_IDX, bumpMapSz)))
		imageStore(derivativeMapImage, G_IDX, vec4(d, 0.0, 0.0));
}
)";
}

CDerivativeMapCreator::~CDerivativeMapCreator()
{
    video::COpenGLDriver* gldriver = static_cast<video::COpenGLDriver*>(m_driver);
    gldriver->extGlDeleteSamplers(1, &m_bumpMapSampler);
    gldriver->extGlDeleteProgram(m_deriv_map_gen_cs);
}

CDerivativeMapCreator::CDerivativeMapCreator(video::IVideoDriver* _driver) : m_driver(_driver)
{
    m_deriv_map_gen_cs = this->createComputeShader(DERIV_MAP_FROM_BUMP_MAP_CS_SRC);

    video::COpenGLDriver* gldriver = static_cast<video::COpenGLDriver*>(m_driver);
    gldriver->extGlGenSamplers(1, &m_bumpMapSampler);
    gldriver->extGlSamplerParameteri(m_bumpMapSampler, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    gldriver->extGlSamplerParameteri(m_bumpMapSampler, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
}

core::smart_refctd_ptr<video::IVirtualTexture> CDerivativeMapCreator::createDerivMapFromBumpMap(video::IVirtualTexture* _bumpMap, float _heightFactor, bool _texWrapRepeat) const
{
    const uint32_t* derivMap_sz = _bumpMap->getSize();
	auto derivMap = m_driver->createGPUTexture(video::ITexture::ETT_2D, derivMap_sz, 1u+uint32_t(std::floor(std::log2(float(core::max_(derivMap_sz[0], derivMap_sz[1]))))), asset::EF_R8G8_SNORM);

    video::COpenGLDriver* gldriver = static_cast<video::COpenGLDriver*>(m_driver);

    gldriver->extGlSamplerParameteri(m_bumpMapSampler, GL_TEXTURE_WRAP_S, _texWrapRepeat ? GL_REPEAT : GL_CLAMP_TO_EDGE);
    gldriver->extGlSamplerParameteri(m_bumpMapSampler, GL_TEXTURE_WRAP_T, _texWrapRepeat ? GL_REPEAT : GL_CLAMP_TO_EDGE);

    const GLenum textype = GL_TEXTURE_2D;
    GLint prevSampler = 0;
    GLint prevTexture = 0;
    //retrieve currently bound resources
    glGetIntegerv(GL_SAMPLER_BINDING, &prevSampler);
    glGetIntegerv(GL_TEXTURE_BINDING_2D, &prevTexture);

    //bind texture and sampling params
    gldriver->extGlBindSamplers(7, 1, &m_bumpMapSampler);
    gldriver->extGlBindTextures(7, 1, &static_cast<video::COpenGL2DTexture*>(_bumpMap)->getOpenGLName(), &textype);

    GLint previousProgram;
    glGetIntegerv(GL_CURRENT_PROGRAM, &previousProgram);

    gldriver->extGlBindImageTexture(0, static_cast<const video::COpenGL2DTexture*>(derivMap.get())->getOpenGLName(),
        0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RG8_SNORM);

    gldriver->extGlUseProgram(m_deriv_map_gen_cs);
    gldriver->extGlProgramUniform1fv(m_deriv_map_gen_cs, 0, 1u, &_heightFactor);
    gldriver->extGlDispatchCompute((derivMap_sz[0]+15u)/16u, (derivMap_sz[1]+15u)/16u, 1u);
    gldriver->extGlMemoryBarrier(
        GL_TEXTURE_FETCH_BARRIER_BIT |
        GL_SHADER_IMAGE_ACCESS_BARRIER_BIT |
        GL_PIXEL_BUFFER_BARRIER_BIT |
        GL_TEXTURE_UPDATE_BARRIER_BIT |
        GL_FRAMEBUFFER_BARRIER_BIT
    );

    // bring back previously bound texture and sampler
    gldriver->extGlBindTextures(7, 1, reinterpret_cast<GLuint*>(&prevTexture), &textype);
    gldriver->extGlBindSamplers(7, 1, reinterpret_cast<GLuint*>(&prevSampler));

    gldriver->extGlBindImageTexture(0u, 0u, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R8); //unbind image
    gldriver->extGlUseProgram(previousProgram); //rebind previously bound program

    derivMap->regenerateMipMapLevels();

    return derivMap;
}