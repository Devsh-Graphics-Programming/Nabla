#include "../../ext/Blur/CBlurPerformer.h"

#include "../../source/Irrlicht/COpenGLBuffer.h"
#include "../../source/Irrlicht/COpenGLDriver.h"
#include "../../source/Irrlicht/COpenGLExtensionHandler.h"
#include "../../source/Irrlicht/COpenGL2DTexture.h"
#include "../../source/Irrlicht/CWriteFile.h"

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
#define WG_SIZE %u // min(ACTUAL_SIZE, CBlurPerformer::s_MAX_WORK_GROUP_SIZE)
layout(local_size_x = WG_SIZE) in;

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
    float radius;
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
    _idx = clamp(_idx, 0u, uint(SMEM_SIZE));
    smem_r[_idx] = _val.r;
    smem_g[_idx] = _val.g;
    smem_b[_idx] = _val.b;
}
vec3 loadShared(uint _idx)
{
    _idx = clamp(_idx, 0u, uint(SMEM_SIZE));
    return vec3(smem_r[_idx], smem_g[_idx], smem_b[_idx]);
}

void upsweep(int d, uint offset, uint _tid)
{
    memoryBarrierShared();
    barrier();

    if (_tid < d)
    {
        uint ai = offset*(2*_tid+1)-1;
        uint bi = offset*(2*_tid+2)-1;
        ai += CONFLICT_FREE_OFFSET(ai);
        bi += CONFLICT_FREE_OFFSET(bi);

        storeShared(bi, loadShared(bi) + loadShared(ai));
    }
}
void downsweep(int d, uint offset, uint _tid)
{
    memoryBarrierShared();
    barrier();

    if (_tid < d)
    {
        uint ai = offset*(2*_tid+1)-1;
        uint bi = offset*(2*_tid+2)-1;
        ai += CONFLICT_FREE_OFFSET(ai);
        bi += CONFLICT_FREE_OFFSET(bi);

        vec3 tmp = loadShared(ai);
        storeShared(ai, loadShared(bi));
        storeShared(bi, loadShared(bi) + tmp);
    }
}
void prepSmemForPresum(uint _inStartIdx, uint _tid)
{
    if (_tid >= ACTUAL_SIZE)
        return;

    const uint ai = _tid;
    const uint bi = _tid + PS_SIZE/2;
    const uint bankOffsetA = CONFLICT_FREE_OFFSET(ai);
    const uint bankOffsetB = CONFLICT_FREE_OFFSET(bi);

    storeShared(ai + bankOffsetA,
	    (ai < ACTUAL_SIZE) ? decodeRgb(samples[_inStartIdx + inOffset + ai]) : vec3(0)
    );
    storeShared(bi + bankOffsetB,
	    (bi < ACTUAL_SIZE) ? decodeRgb(samples[_inStartIdx + inOffset + bi]) : vec3(0)
    );   
}
void exPsumInSmem()
{
    const uint IN_START_IDX = gl_WorkGroupID.y*ACTUAL_SIZE;

    %s // here goes prepSmemForPresum unrolled loop

    memoryBarrierShared();
    barrier();

    uint offset = 1;

    int up_itr = 0;
    %s // here goes unrolled upsweep loop

    if (LC_IDX == 0) { storeShared(PS_SIZE-1 + CONFLICT_FREE_OFFSET(PS_SIZE-1), vec3(0)); }
    memoryBarrierShared();
    barrier();

    int down_itr = 0;
    %s // here goes unrolled downsweep loop
}

uint getAddr(uint _addr)
{
    return _addr + CONFLICT_FREE_OFFSET(_addr);
}

// WARNING: calculates resulting address (this function do NOT expect to get address from getAddr())
vec3 loadSharedInterp(float _idx)
{
    float f = fract(_idx);
    if (f < 0.1f) // I think 0.1 is a reasonable value that wouldn't make any visual difference and we don't have to make second memory fetch
        return loadShared(getAddr(uint(_idx)));
    return mix(loadShared(getAddr(uint(_idx))), loadShared(getAddr(uint(_idx)+1u)), f);
}

void blurAndOutput(float RADIUS, uint _tid)
{
    if (_tid >= ACTUAL_SIZE)
        return;

    uvec2 IDX = uvec2(_tid, gl_WorkGroupID.y);
    if (iterNum > 2)
        IDX = IDX.yx;

    // all index constants below (except LAST_IDX) are enlarged by 1 becaue of **exclusive** prefix sum
    const float FIRST_IDX_F = 1.f;
    const uint FIRST_IDX = 1;
    const uint LAST_IDX = ACTUAL_SIZE-1;
    const float L_IDX = float(_tid) - RADIUS;
    const float R_IDX = float(_tid) + RADIUS + 1.f;
    const float R_EDGE_IDX = min(R_IDX, float(LAST_IDX));

    vec3 res =
        loadSharedInterp(R_EDGE_IDX)
        + (R_IDX - R_EDGE_IDX)*(loadShared(getAddr(LAST_IDX)) - loadShared(getAddr(LAST_IDX-1))) // handle right overflow
        - ((L_IDX < FIRST_IDX_F) ? ((L_IDX - FIRST_IDX_F + 1.f) * loadShared(getAddr(FIRST_IDX))) : loadSharedInterp(L_IDX)); // also handle left overflow
    res /= (2.f*RADIUS + 1.f);
#if FINAL_PASS
    imageStore(out_img, ivec2(IDX), vec4(res, 1.f));
#else
    samples[uint(dot(IDX, outMlt)) + outOffset] = encodeRgb(res);
#endif    
}

void main()
{
    uvec2 IDX = gl_GlobalInvocationID.xy;
    if (iterNum > 2)
        IDX = IDX.yx;

    exPsumInSmem();
    memoryBarrierShared();
    barrier();

    const float RADIUS = float(ACTUAL_SIZE) * radius;

    %s // here goes blurAndOutput unrolled loop
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
    GLchar infoLog[0x4000];
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

CBlurPerformer* CBlurPerformer::instantiate(video::IVideoDriver* _driver, float _radius, core::vector2d<uint32_t> _dsFactor, video::IGPUBuffer* uboBuffer, const size_t& uboDataStaticOffset)
{
    return new CBlurPerformer(_driver, _radius, _dsFactor, uboBuffer, uboDataStaticOffset);
}

video::ITexture* CBlurPerformer::createOutputTexture(video::ITexture* _inputTex)
{
    prepareForBlur(_inputTex->getSize());

    video::ITexture* outputTex = m_driver->addTexture(video::ITexture::ETT_2D, &m_outSize.X, 1, ("__IRR_blur_out" + std::to_string(s_texturesEverCreatedCount++)).c_str(), video::ECF_A16B16G16R16F);
    blurTexture(_inputTex, outputTex);

    return outputTex;
}

//#define PROFILE_BLUR_PERFORMER
void CBlurPerformer::blurTexture(video::ITexture* _inputTex, video::ITexture* _outputTex)
{
    GLint prevProgram{};
    glGetIntegerv(GL_CURRENT_PROGRAM, &prevProgram);

#ifdef PROFILE_BLUR_PERFORMER
    video::IQueryObject* timeQuery = m_driver->createElapsedTimeQuery();
    m_driver->beginQuery(timeQuery);
#endif // PROFILE_BLUR_PERFORMER

    {
    const uint32_t* sz = _outputTex->getSize();
    assert(sz[0] >= m_outSize.X && sz[1] >= m_outSize.Y &&
        _outputTex->getColorFormat() == video::ECF_A16B16G16R16F &&
        _outputTex->getTextureType() == video::ITexture::ETT_2D
    );
    }
    prepareForBlur(_inputTex->getSize()); // recalculate output size and recreate shaders if there's a need
    assert(m_dsampleCs);

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

#ifdef PROFILE_BLUR_PERFORMER
    m_driver->endQuery(timeQuery);
    uint32_t timeTaken = 0;
    timeQuery->getQueryResult(&timeTaken);
    os::Printer::log("irr::ext::BlurPerformer GPU time taken:", std::to_string(timeTaken).c_str(), ELL_ERROR);
#endif // PROFILE_BLUR_PERFORMER
}

void CBlurPerformer::prepareForBlur(const uint32_t* _inputSize)
{
    const core::vector2d<uint32_t> outSz = core::vector2d<uint32_t>(_inputSize[0], _inputSize[1]) / m_dsFactor;
    if (outSz == m_outSize && m_dsampleCs /*if dsample CS is absent, we can be sure that none of the shaders are there*/)
        return;

    if (m_dsampleCs)
        deleteShaders(); // or maybe we could cache already compiled shaders (with output size as key) in case they get to be usable again?

    m_outSize = outSz;
    assert(m_outSize.X > s_MAX_OUTPUT_SIZE_XY || m_outSize.Y > s_MAX_OUTPUT_SIZE_XY);
    std::tie(m_dsampleCs, m_blurGeneralCs[0], m_blurGeneralCs[1], m_blurFinalCs) = makeShaders(m_outSize);

    // when output size changes, we also have to reupload new UBO contents
    writeUBOData();

    if (m_samplesSsbo)
        m_samplesSsbo->drop();

    video::IDriverMemoryBacked::SDriverMemoryRequirements reqs;
    reqs.vulkanReqs.alignment = 4;
    reqs.vulkanReqs.memoryTypeBits = 0xffffffffu;
    reqs.memoryHeapLocation = video::IDriverMemoryAllocation::ESMT_DEVICE_LOCAL;
    reqs.mappingCapability = video::IDriverMemoryAllocation::EMCF_CANNOT_MAP;
    reqs.prefersDedicatedAllocation = true;
    reqs.requiresDedicatedAllocation = true;
    reqs.vulkanReqs.size = 2 * 2 * m_outSize.X * m_outSize.Y * sizeof(uint32_t);

    m_samplesSsbo = m_driver->createGPUBufferOnDedMem(reqs);
}

void CBlurPerformer::deleteShaders() const
{
    video::COpenGLExtensionHandler::extGlDeleteProgram(m_dsampleCs);
    video::COpenGLExtensionHandler::extGlDeleteProgram(m_blurGeneralCs[0]);
    if (m_blurGeneralCs[0] != m_blurGeneralCs[1]) // if output size is square, then those are the same shader
        video::COpenGLExtensionHandler::extGlDeleteProgram(m_blurGeneralCs[1]);
    video::COpenGLExtensionHandler::extGlDeleteProgram(m_blurFinalCs);
}

CBlurPerformer::~CBlurPerformer()
{
    if (m_samplesSsbo)
        m_samplesSsbo->drop();
    if (m_ubo)
        m_ubo->drop();
    deleteShaders();
}

bool CBlurPerformer::genDsampleCs(char* _out, size_t _bufSize, const core::vector2d<uint32_t>& _outTexSize)
{
    return snprintf(_out, _bufSize, CS_DOWNSAMPLE_SRC, _outTexSize.X, _outTexSize.Y, CS_CONVERSIONS) > 0;
}
bool CBlurPerformer::genBlurPassCs(char* _out, size_t _bufSize, uint32_t _outTexSize, int _finalPass)
{
    const uint32_t N = (_outTexSize + s_MAX_WORK_GROUP_SIZE-1u) / s_MAX_WORK_GROUP_SIZE;

    const char prepSmemForPresum_fmt[] = "for (uint i = 0; i < %u; ++i) prepSmemForPresum(IN_START_IDX, LC_IDX + i*WG_SIZE);\n";

    const char blurAndOutput_fmt[] = "for (uint i = 0; i < %u; ++i) blurAndOutput(RADIUS, LC_IDX + i*WG_SIZE);\n";

    char buf[1u<<12];

    snprintf(buf, sizeof(buf), prepSmemForPresum_fmt, N);
    std::string prepSmemForPresum = buf;

    snprintf(buf, sizeof(buf), blurAndOutput_fmt, N);
    std::string blurAndOutput = buf;

    std::string upsweep_fmt =
        "for (uint i = 0; i < %u; ++i) upsweep(%u, offset, LC_IDX + i*WG_SIZE);\n"
        "offset *= 2;\n";
    std::string downsweep_fmt =
        "offset /= 2;\n"
        "for (uint i = 0; i < %u; ++i) downsweep(%u, offset, LC_IDX + i*WG_SIZE);\n";

    const uint32_t pot = padToPoT(_outTexSize);

    std::string upsweep; // unrolled upsweep loop
    for (uint32_t up_itr = pot/2u; up_itr > 0; up_itr /= 2u)
    {
        snprintf(buf, sizeof(buf), upsweep_fmt.c_str(), N, up_itr);
        upsweep += buf;
    }
    std::string downsweep; // unrolled downsweep loop
    for (uint32_t down_itr = 1u; down_itr < pot; down_itr *= 2u)
    {
        snprintf(buf, sizeof(buf), downsweep_fmt.c_str(), N, down_itr);
        downsweep += buf;
    }

    return snprintf(_out, _bufSize, CS_BLUR_SRC, pot, _outTexSize, std::min(_outTexSize, s_MAX_WORK_GROUP_SIZE), _finalPass, CS_CONVERSIONS, prepSmemForPresum.c_str(), upsweep.c_str(), downsweep.c_str(), blurAndOutput.c_str()) > 0;
}

void CBlurPerformer::bindSSBuffers() const
{
    auto auxCtx = const_cast<video::COpenGLDriver::SAuxContext*>(static_cast<video::COpenGLDriver*>(m_driver)->getThreadContext());

    const video::COpenGLBuffer* bufs[]{ static_cast<const video::COpenGLBuffer*>(m_samplesSsbo) };
    ptrdiff_t offsets[]{ 0, 0 };
    ptrdiff_t sizes[]{ m_samplesSsbo->getSize() };

    auxCtx->setActiveSSBO(E_SAMPLES_SSBO_BINDING, 1u, bufs, offsets, sizes);
}

auto CBlurPerformer::getCurrentImageBinding(uint32_t _imgUnit) -> ImageBindingData
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

auto CBlurPerformer::makeShaders(const core::vector2d<uint32_t>& _outSize) -> tuple4xu32
{
    uint32_t ds{}, gblurx{}, gblury{}, fblur{};

    const size_t bufSize = std::max(strlen(CS_BLUR_SRC), strlen(CS_DOWNSAMPLE_SRC)) + strlen(CS_CONVERSIONS) + 10000u;
    char* src = (char*)malloc(bufSize);

    auto doCleaning = [&, ds, gblurx, gblury, fblur, src]() {
        for (uint32_t s : { ds, gblurx, gblury, fblur })
            video::COpenGLExtensionHandler::extGlDeleteProgram(s);
        free(src);
        return tuple4xu32(0u, 0u, 0u, 0u);
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

    return std::make_tuple(ds, gblurx, gblury, fblur);
}

void CBlurPerformer::bindUbo(uint32_t _bnd, uint32_t _part) const
{
    auto auxCtx = const_cast<video::COpenGLDriver::SAuxContext*>(static_cast<video::COpenGLDriver*>(m_driver)->getThreadContext());

    const video::COpenGLBuffer* buf{ static_cast<const video::COpenGLBuffer*>(m_ubo) };
    ptrdiff_t offset = _part * m_paddedUBOSize+m_uboStaticOffset;
    ptrdiff_t size = m_paddedUBOSize;

    auxCtx->setActiveUBO(_bnd, 1u, &buf, &offset, &size);
}
