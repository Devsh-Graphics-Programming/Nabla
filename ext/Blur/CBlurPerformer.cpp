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

#define SIZEX %u
#define SIZEY %u

layout(std430, binding = 0) restrict writeonly buffer b {
    uvec2 ssbo[];
};

layout(binding = 0) uniform sampler2D in_tex;

%s // here goes CS_CONVERSIONS

void main()
{
    const uvec2 IDX = gl_GlobalInvocationID.xy; // each index corresponds to one pixel in downsampled texture

    const ivec2 OUT_SIZE = ivec2(SIZEX, SIZEY);

    vec2 coords = vec2(IDX) / vec2(OUT_SIZE);
    vec4 avg = (
        texture(in_tex, coords) +
        textureOffset(in_tex, coords, ivec2(1, 0)) +
        textureOffset(in_tex, coords, ivec2(0, 1)) +
        textureOffset(in_tex, coords, ivec2(1, 1))
    ) / 4.f;

    const uint HBUF_IDX = IDX.y * OUT_SIZE.x + IDX.x;

    if (IDX.x < SIZEX && IDX.y < SIZEY)
        ssbo[HBUF_IDX] = encodeRgb(avg.xyz);
}
)XDDD";
constexpr const char* CS_BLUR_SRC = R"XDDD(
#version 430 core
#define PS_SIZE %u // ACTUAL_SIZE padded to PoT
#define ACTUAL_SIZE %u
layout(local_size_x = ACTUAL_SIZE) in;

#define NUM_BANKS 32
#define LOG_NUM_BANKS 5
#define CONFLICT_FREE_OFFSET(n) ((n) >> LOG_NUM_BANKS)

layout(std430, binding = 0) restrict buffer Samples {
    uvec2 samples[];
};

#define LC_IDX gl_LocalInvocationIndex
#define FINAL_PASS %d

#if FINAL_PASS
layout(binding = 0, rgba16f) uniform writeonly image2D out_img;
#endif


layout(std140, binding = 0) uniform Controls
{
    uint iterNum;
    uint radius;
    uint inOffset;
    uint outOffset;
    uvec2 outMlt;
};

#define SMEM_SIZE (2*PS_SIZE)
shared float smem_r[SMEM_SIZE];
shared float smem_g[SMEM_SIZE];
shared float smem_b[SMEM_SIZE];

%s // here goes CS_CONVERSIONS

void storeShared(uint _idx, vec3 _val)
{
    smem_r[_idx] = _val.r;
    smem_g[_idx] = _val.g;
    smem_b[_idx] = _val.b;
}
vec3 loadShared(uint _idx)
{
    return vec3(smem_r[_idx], smem_g[_idx], smem_b[_idx]);
}

void upsweep(int d, inout uint offset)
{
    memoryBarrierShared();
    barrier();

    if (LC_IDX < d)
    {
        uint ai = offset*(2*LC_IDX+1)-1;
        uint bi = offset*(2*LC_IDX+2)-1;
        ai += CONFLICT_FREE_OFFSET(ai);
        bi += CONFLICT_FREE_OFFSET(bi);

        storeShared(bi, loadShared(bi) + loadShared(ai));
    }
    offset *= 2;
}
void downsweep(int d, inout uint offset)
{
    offset /= 2;
    memoryBarrierShared();
    barrier();

    if (LC_IDX < d)
    {
        uint ai = offset*(2*LC_IDX+1)-1;
        uint bi = offset*(2*LC_IDX+2)-1;
        ai += CONFLICT_FREE_OFFSET(ai);
        bi += CONFLICT_FREE_OFFSET(bi);

        vec3 tmp = loadShared(ai);
        storeShared(ai, loadShared(bi));
        storeShared(bi, loadShared(bi) + tmp);
    }
}
void exPsumInSmem()
{
    const uint IN_START_IDX = gl_WorkGroupID.y*ACTUAL_SIZE;

    const uint ai = LC_IDX;
    const uint bi = LC_IDX + PS_SIZE/2;
    const uint bankOffsetA = CONFLICT_FREE_OFFSET(ai);
    const uint bankOffsetB = CONFLICT_FREE_OFFSET(bi);

    storeShared(ai + bankOffsetA,
	    (ai < ACTUAL_SIZE) ? decodeRgb(samples[IN_START_IDX + inOffset + ai]) : vec3(0)
    );
    storeShared(bi + bankOffsetB,
	    (bi < ACTUAL_SIZE) ? decodeRgb(samples[IN_START_IDX + inOffset + bi]) : vec3(0)
    );

    memoryBarrierShared();
    barrier();

    uint offset = 1;

    %s // here goes unrolled upsweep loop

    if (LC_IDX == 0) { storeShared(PS_SIZE-1 + CONFLICT_FREE_OFFSET(PS_SIZE-1), vec3(0)); }
    memoryBarrierShared();
    barrier();

    %s // here goes unrolled downsweep loop
}

uint getAddr(uint _addr)
{
    return _addr + CONFLICT_FREE_OFFSET(_addr);
}

void main()
{
    uvec2 IDX = gl_GlobalInvocationID.xy;
    if (iterNum > 2)
        IDX = IDX.yx;

    exPsumInSmem();
    memoryBarrierShared();
    barrier();

    // all index constants below (except LAST_IDX) are enlarged by 1 becaue of **exclusive** prefix sum
    const int FIRST_IDX = 1;
    const uint LAST_IDX = ACTUAL_SIZE-1;
    const int L_IDX = int(LC_IDX - radius);
    const uint R_IDX = LC_IDX + radius + 1;
    const uint R_EDGE_IDX = min(R_IDX, LAST_IDX);

    vec3 res =
        loadShared(getAddr(R_EDGE_IDX))
        + (R_IDX - R_EDGE_IDX)*(loadShared(getAddr(LAST_IDX)) - loadShared(getAddr(LAST_IDX-1))) // handle right overflow
        - ((L_IDX < FIRST_IDX) ? ((L_IDX - FIRST_IDX + 1) * loadShared(getAddr(FIRST_IDX))) : loadShared(getAddr(L_IDX))); // also handle left overflow
    res /= (float(2*radius) + 1.f);
#if FINAL_PASS
    imageStore(out_img, ivec2(IDX), vec4(res, 1.f));
#else
    samples[uint(dot(IDX, outMlt)) + outOffset] = encodeRgb(res);
#endif
}
)XDDD";

inline uint32_t createComputeShader(const char* _src)
{
    uint32_t program = video::COpenGLExtensionHandler::extGlCreateProgram();
	uint32_t cs = video::COpenGLExtensionHandler::extGlCreateShader(GL_COMPUTE_SHADER);

	video::COpenGLExtensionHandler::extGlShaderSource(cs, 1, const_cast<const char**>(&_src), NULL);
	video::COpenGLExtensionHandler::extGlCompileShader(cs);

	// check for compilation errors
    GLint success;
    GLchar infoLog[0x400];
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
} // anon ns end

uint32_t CBlurPerformer::s_texturesEverCreatedCount{};

CBlurPerformer* CBlurPerformer::instantiate(video::IVideoDriver* _driver, uint32_t _radius, core::vector2d<uint32_t> _outSize, video::IGPUBuffer* uboBuffer, const size_t& uboDataStaticOffset)
{
    uint32_t ds{},gblurx{}, gblury{}, fblur{};

    const size_t bufSize = std::max(strlen(CS_BLUR_SRC), strlen(CS_DOWNSAMPLE_SRC)) + strlen(CS_CONVERSIONS) + 1000u;
    char* src = (char*)malloc(bufSize);

    auto doCleaning = [&, ds, gblurx, gblury, fblur, src]() {
        for (uint32_t s : { ds, gblurx, gblury, fblur })
            video::COpenGLExtensionHandler::extGlDeleteProgram(s);
        free(src);
        return nullptr;
    };

    if (!genDsampleCs(src, bufSize, _outSize))
        return doCleaning();
    ds = createComputeShader(src);

    if (!genBlurPassCs(src, bufSize, _outSize.X, 0))
        return doCleaning();
    gblurx = createComputeShader(src);
    if (_outSize.X != _outSize.Y)
    {
        if (!genBlurPassCs(src, bufSize, _outSize.Y, 0))
            return doCleaning();
        gblury = createComputeShader(src);
    }
    else gblury = gblurx;

    if (!genBlurPassCs(src, bufSize, _outSize.Y, 1))
        return doCleaning();
    fblur = createComputeShader(src);

    for (uint32_t s : {ds, gblurx, gblury, fblur})
    {
        if (!s)
            return doCleaning();
    }
    free(src);

    return new CBlurPerformer(_driver, ds, gblurx, gblury, fblur, _radius, _outSize, uboBuffer, uboDataStaticOffset);
}

video::ITexture* CBlurPerformer::createOutputTexture(video::ITexture* _inputTex) const
{
    video::ITexture* outputTex = m_driver->addTexture(video::ITexture::ETT_2D, &m_outSize.X, 1, ("__IRR_blur_out" + std::to_string(s_texturesEverCreatedCount++)).c_str(), video::ECF_A16B16G16R16F);

    blurTexture(_inputTex, outputTex);

    return outputTex;
}

void CBlurPerformer::blurTexture(video::ITexture* _inputTex, video::ITexture* _outputTex) const
{
    {
    const uint32_t* sz = _outputTex->getSize();
    assert(sz[0] >= m_outSize.X && sz[1] >= m_outSize.Y &&
        _outputTex->getColorFormat() == video::ECF_A16B16G16R16F &&
        _outputTex->getTextureType() == video::ITexture::ETT_2D
    );
    }

    GLint prevProgram{};
    glGetIntegerv(GL_CURRENT_PROGRAM, &prevProgram);

    bindSSBuffers();

    //texture stuff
    {
        video::STextureSamplingParams params;
        params.UseMipmaps = 0;
        params.MaxFilter = params.MinFilter = video::ETFT_LINEAR_NO_MIP;
        params.TextureWrapU = params.TextureWrapV = video::ETC_CLAMP_TO_EDGE;
        const_cast<video::COpenGLDriver::SAuxContext*>(reinterpret_cast<video::COpenGLDriver*>(m_driver)->getThreadContext())->setActiveTexture(0, _inputTex, params);
    }

    auto prevImgBinding = getCurrentImageBinding(0);
    video::COpenGLExtensionHandler::extGlBindImageTexture(0, static_cast<const video::COpenGL2DTexture*>(_outputTex)->getOpenGLName(),
        0, GL_FALSE, 0, GL_WRITE_ONLY, video::COpenGLTexture::getOpenGLFormatAndParametersFromColorFormat(static_cast<const video::COpenGL2DTexture*>(_outputTex)->getColorFormat()));


    video::COpenGLExtensionHandler::extGlUseProgram(m_dsampleCs);

    video::COpenGLExtensionHandler::extGlDispatchCompute((m_outSize.X+15)/16, (m_outSize.Y+15)/16, 1u);
    video::COpenGLExtensionHandler::extGlMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    uint32_t i = 0u;
    for (; i < 5u; ++i)
    {
        bindUbo(E_UBO_BINDING, i);

        video::COpenGLExtensionHandler::extGlUseProgram(m_blurGeneralCs[i >= 3u]);

        video::COpenGLExtensionHandler::extGlDispatchCompute(1u, (i >= 3u ? m_outSize.X : m_outSize.Y), 1u);
        video::COpenGLExtensionHandler::extGlMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    }

    bindUbo(E_UBO_BINDING, i);

    video::COpenGLExtensionHandler::extGlUseProgram(m_blurFinalCs);

    video::COpenGLExtensionHandler::extGlDispatchCompute(1u, m_outSize.X, 1u);
    video::COpenGLExtensionHandler::extGlMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT);

    video::COpenGLExtensionHandler::extGlUseProgram(prevProgram);
    bindImage(0, prevImgBinding);
}

CBlurPerformer::~CBlurPerformer()
{
    m_samplesSsbo->drop();
    m_ubo->drop();
    video::COpenGLExtensionHandler::extGlDeleteProgram(m_dsampleCs);
    video::COpenGLExtensionHandler::extGlDeleteProgram(m_blurGeneralCs[0]);
    video::COpenGLExtensionHandler::extGlDeleteProgram(m_blurGeneralCs[1]);
    video::COpenGLExtensionHandler::extGlDeleteProgram(m_blurFinalCs);
}

bool CBlurPerformer::genDsampleCs(char* _out, size_t _bufSize, const core::vector2d<uint32_t>& _outTexSize)
{
    return snprintf(_out, _bufSize, CS_DOWNSAMPLE_SRC, _outTexSize.X, _outTexSize.Y, CS_CONVERSIONS) > 0;
}
bool CBlurPerformer::genBlurPassCs(char* _out, size_t _bufSize, uint32_t _outTexSize, int _finalPass)
{
    const uint32_t pot = padToPoT(_outTexSize);
    const char up_fmt[] = "upsweep(%u, offset);\n";
    const char down_fmt[] = "downsweep(%u, offset);\n";
    char buf[sizeof(down_fmt) + 10];

    std::string upsweep; // unrolled upsweep loop
    for (uint32_t d = pot/2u; d > 0; d /= 2u)
    {
        snprintf(buf, sizeof(buf), up_fmt, d);
        upsweep += buf;
    }
    std::string downsweep; // unrolled downsweep loop
    for (uint32_t d = 1u; d < pot; d *= 2u)
    {
        snprintf(buf, sizeof(buf), down_fmt, d);
        downsweep += buf;
    }

    return snprintf(_out, _bufSize, CS_BLUR_SRC, pot, _outTexSize, _finalPass, CS_CONVERSIONS, upsweep.c_str(), downsweep.c_str()) > 0;
}

void CBlurPerformer::bindSSBuffers() const
{
    auto auxCtx = const_cast<video::COpenGLDriver::SAuxContext*>(static_cast<video::COpenGLDriver*>(m_driver)->getThreadContext());

    const video::COpenGLBuffer* bufs[]{ static_cast<const video::COpenGLBuffer*>(m_samplesSsbo) };
    ptrdiff_t offsets[]{ 0, 0 };
    ptrdiff_t sizes[]{ m_samplesSsbo->getSize() };

    auxCtx->setActiveSSBO(E_SAMPLES_SSBO_BINDING, 1u, bufs, offsets, sizes);
}

auto irr::ext::Blur::CBlurPerformer::getCurrentImageBinding(uint32_t _imgUnit) -> ImageBindingData
{
    using gl = video::COpenGLExtensionHandler;

    ImageBindingData data;
    gl::extGlGetIntegeri_v(GL_IMAGE_BINDING_NAME, _imgUnit, (GLint*)&data.name);
    gl::extGlGetIntegeri_v(GL_IMAGE_BINDING_LEVEL, _imgUnit, &data.level);
    gl::extGlGetBooleani_v(GL_IMAGE_BINDING_LAYERED, _imgUnit, &data.layered);
    gl::extGlGetIntegeri_v(GL_IMAGE_BINDING_LAYER, _imgUnit, &data.layer);
    gl::extGlGetIntegeri_v(GL_IMAGE_BINDING_ACCESS, _imgUnit, (GLint*)&data.access);
    gl::extGlGetIntegeri_v(GL_IMAGE_BINDING_FORMAT, _imgUnit, (GLint*)&data.format);

    return data;
}

void irr::ext::Blur::CBlurPerformer::bindImage(uint32_t _imgUnit, const ImageBindingData& _data)
{
    video::COpenGLExtensionHandler::extGlBindImageTexture(_imgUnit, _data.name, _data.level, _data.layered, _data.layer, _data.access, _data.format);
}

void CBlurPerformer::writeUBOData()
{
    video::IDriverMemoryBacked::SDriverMemoryRequirements reqs;
    reqs.vulkanReqs.alignment = 4;
    reqs.vulkanReqs.memoryTypeBits = 0xffffffffu;
    reqs.memoryHeapLocation = video::IDriverMemoryAllocation::ESMT_DEVICE_LOCAL;
    reqs.mappingCapability = video::IDriverMemoryAllocation::EMCF_CAN_MAP_FOR_WRITE|video::IDriverMemoryAllocation::EMCF_COHERENT;
    reqs.prefersDedicatedAllocation = true;
    reqs.requiresDedicatedAllocation = true;
    reqs.vulkanReqs.size = getRequiredUBOSize(m_driver);
    video::IGPUBuffer* stagingBuf = m_driver->createGPUBufferOnDedMem(reqs);

    uint8_t* mappedPtr = reinterpret_cast<uint8_t*>(stagingBuf->getBoundMemory()->mapMemoryRange(video::IDriverMemoryAllocation::EMCAF_WRITE,{0,reqs.vulkanReqs.size}));

    //! all of the above to move out of writeUBOData function

    const core::vector2d<uint32_t> HMLT(1u, m_outSize.X), VMLT(m_outSize.Y, 1u);
    const core::vector2d<uint32_t> multipliers[6]{ HMLT, HMLT, VMLT, VMLT, VMLT, HMLT };
    uint32_t inOffset = 0u;
    uint32_t outOffset = m_outSize.X * m_outSize.Y;
    for (uint32_t i = 0u; i < 6u; ++i)
    {
        BlurPassUBO* destPtr = reinterpret_cast<BlurPassUBO*>(mappedPtr+i*m_paddedUBOSize);

        destPtr->iterNum = i;
        destPtr->inOffset = inOffset;
        destPtr->outOffset = outOffset;
        destPtr->outMlt[0] = multipliers[i].X;
        destPtr->outMlt[1] = multipliers[i].Y;
        destPtr->radius = m_radius;

        std::swap(inOffset, outOffset);
    }

    //! all of the below to move out of writeUBOData function
    stagingBuf->getBoundMemory()->unmapMemory();

    m_driver->copyBuffer(stagingBuf,m_ubo,0,m_uboStaticOffset,reqs.vulkanReqs.size);

    stagingBuf->drop();
}

void CBlurPerformer::bindUbo(uint32_t _bnd, uint32_t _part) const
{
    auto auxCtx = const_cast<video::COpenGLDriver::SAuxContext*>(static_cast<video::COpenGLDriver*>(m_driver)->getThreadContext());

    const video::COpenGLBuffer* buf{ static_cast<const video::COpenGLBuffer*>(m_ubo) };
    ptrdiff_t offset = _part * m_paddedUBOSize+m_uboStaticOffset;
    ptrdiff_t size = m_paddedUBOSize;

    auxCtx->setActiveUBO(_bnd, 1u, &buf, &offset, &size);
}
