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
constexpr const char* CS_BLUR_SRC = R"XDDD(
#version 430 core
#define PS_SIZE %u // ACTUAL_SIZE padded to PoT
#define ACTUAL_SIZE %u
#define ACTUAL_SIZE_X %u
#define ACTUAL_SIZE_Y %u
#define SSBO_OUT_OFFSET (ACTUAL_SIZE_X*ACTUAL_SIZE_Y) // offset where first pass outputs
#define WG_SIZE %u // min(ACTUAL_SIZE, CBlurPerformer::s_MAX_WORK_GROUP_SIZE)
#define SUB_THR_CNT %u // number of threads simulated by single thread
#define NUM_PASSES %u // number of passes per axis (i.e. in single shader)
layout(local_size_x = WG_SIZE) in;

#define NUM_BANKS 32
#define LOG_NUM_BANKS 5
#define CONFLICT_FREE_OFFSET(n) ((n) >> LOG_NUM_BANKS)

#define BARRIER \
    memoryBarrierShared();\
    barrier()

layout(std430, binding = 0) restrict buffer Samples {
    uvec2 samples[];
};

#define LC_IDX gl_LocalInvocationIndex
#define FINAL_PASS %d

#if FINAL_PASS
layout(binding = 0, rgba16f) uniform writeonly image2D out_img;
#else
layout(binding = 0) uniform sampler2D in_tex;
#endif


layout(std140, binding = 0) uniform Controls
{
    float u_radius;
};

#define SMEM_SIZE (2*PS_SIZE)
shared float smem_r[SMEM_SIZE];
shared float smem_g[SMEM_SIZE];
shared float smem_b[SMEM_SIZE];

%s // here goes CS_CONVERSIONS

uint getAddr(uint _addr)
{
    return _addr + CONFLICT_FREE_OFFSET(_addr);
}
void storeShared(uint _idx, vec3 _val)
{
    _idx = clamp(_idx, 0u, uint(SMEM_SIZE-1));
    smem_r[_idx] = _val.r;
    smem_g[_idx] = _val.g;
    smem_b[_idx] = _val.b;
}
vec3 loadShared(uint _idx)
{
    _idx = clamp(_idx, 0u, uint(SMEM_SIZE-1));
    return vec3(smem_r[_idx], smem_g[_idx], smem_b[_idx]);
}

#if FINAL_PASS==0
void loadFromTexture(uint _tid)
{
    if (_tid >= ACTUAL_SIZE)
        return;

    vec2 coords = vec2(float(_tid), float(gl_WorkGroupID.y)) / vec2(ACTUAL_SIZE_X, ACTUAL_SIZE_Y);
    vec4 avg = (
        texture(in_tex, coords) +
        textureOffset(in_tex, coords, ivec2(1, 0)) +
        textureOffset(in_tex, coords, ivec2(0, 1)) +
        textureOffset(in_tex, coords, ivec2(1, 1))
    ) / 4.f;

    storeShared(getAddr(_tid), avg.rgb);
}
#endif

void upsweep(int d, uint offset, uint _tid)
{
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
void loadFromGmem(uint _inStartIdx, uint _tid)
{
    if (_tid >= ACTUAL_SIZE)
        return;

    const uint ai = _tid;
    const uint bi = _tid + PS_SIZE/2;
    const uint bankOffsetA = CONFLICT_FREE_OFFSET(ai);
    const uint bankOffsetB = CONFLICT_FREE_OFFSET(bi);

    storeShared(ai + bankOffsetA,
	    (ai < ACTUAL_SIZE) ? decodeRgb(samples[_inStartIdx + SSBO_OUT_OFFSET + ai]) : vec3(0)
    );
    storeShared(bi + bankOffsetB,
	    (bi < ACTUAL_SIZE) ? decodeRgb(samples[_inStartIdx + SSBO_OUT_OFFSET + bi]) : vec3(0)
    );   
}
void exPsumInSmem()
{
    uint offset = 1;

    %s // here goes unrolled upsweep loop

    if (LC_IDX == 0) { storeShared(PS_SIZE-1 + CONFLICT_FREE_OFFSET(PS_SIZE-1), vec3(0)); }
    memoryBarrierShared();
    barrier();

    %s // here goes unrolled downsweep loop
}

// WARNING: calculates resulting address (this function do NOT expect to get address from getAddr())
vec3 loadSharedInterp(float _idx)
{
    uint floored = uint(_idx);
    return mix(loadShared(getAddr(floored)), loadShared(getAddr(floored+1u)), _idx-float(floored));
}

vec3 blur(float RADIUS, uint _tid)
{
    if (_tid >= ACTUAL_SIZE)
        return vec3(0.f);

    // all index constants below (except LAST_IDX) are enlarged by 1 becaue of **exclusive** prefix sum
    const float FIRST_IDX_F = 1.f;
    const uint FIRST_IDX = 1;
    const uint LAST_IDX = ACTUAL_SIZE-1;
    const float L_IDX = float(_tid) - RADIUS;
    const float R_IDX = float(_tid) + RADIUS + 1.f;
    const float R_EDGE_IDX = min(R_IDX, float(LAST_IDX));

    vec3 res;
    if (R_IDX < float(LAST_IDX))
        res = loadSharedInterp(R_IDX);
    else
        res = (R_IDX+ 1.f - float(LAST_IDX))*loadShared(getAddr(LAST_IDX)) - (R_IDX - float(LAST_IDX))*loadShared(getAddr(LAST_IDX-1));
    if (L_IDX < FIRST_IDX_F)
        res -= (L_IDX - FIRST_IDX_F + 1.f) * loadShared(getAddr(FIRST_IDX));
    else
        res -= loadSharedInterp(L_IDX);

    return res / (2.f*RADIUS + 1.f);
}

void outputFunc(uint _tid)
{
    if (_tid >= ACTUAL_SIZE)
        return;

    uvec2 IDX = 
#if FINAL_PASS
    uvec2(gl_WorkGroupID.y, _tid);
#else
    uvec2(_tid, gl_WorkGroupID.y);
#endif

    vec3 res = loadShared(getAddr(_tid));
#if FINAL_PASS
    imageStore(out_img, ivec2(IDX), vec4(res, 1.f));
#else
    samples[IDX.x*ACTUAL_SIZE_Y + IDX.y + SSBO_OUT_OFFSET] = encodeRgb(res);
#endif
}

void main()
{
#if FINAL_PASS
    const uint IN_START_IDX = gl_WorkGroupID.y*ACTUAL_SIZE;
    %s // here goes loadFromGmem calls
#else
    %s // here goes loadFromTexture calls
#endif

    const float RADIUS = float(ACTUAL_SIZE) * u_radius;
    for (uint pass = 0u; pass < NUM_PASSES; ++pass)
    {
        exPsumInSmem();

        memoryBarrierShared();
        barrier();

        vec3 blurred[SUB_THR_CNT];
        %s // here goes blur() calls

        barrier();

        %s // here goes (local array)->(SMEM) output loop
    
        memoryBarrierShared();
        barrier();
    }
    %s // here goes output loop
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

uint32_t CBlurPerformer::s_MAX_WORK_GROUP_SIZE = 0u;
constexpr uint32_t CBlurPerformer::s_ABSOLUTELY_MAX_WORK_GROUP_SIZE;
constexpr uint32_t CBlurPerformer::s_MAX_OUTPUT_SIZE_XY;
constexpr uint32_t CBlurPerformer::s_MIN_DS_FACTOR;
constexpr uint32_t CBlurPerformer::s_MAX_DS_FACTOR;

CBlurPerformer* CBlurPerformer::instantiate(video::IVideoDriver* _driver, float _radius, core::vector2d<uint32_t> _dsFactor, uint32_t _passesPerAxis, video::IGPUBuffer* uboBuffer, const size_t& uboDataStaticOffset)
{
    if (s_MAX_WORK_GROUP_SIZE == 0u)
        s_MAX_WORK_GROUP_SIZE = std::min(s_ABSOLUTELY_MAX_WORK_GROUP_SIZE, _driver->getMaxComputeWorkGroupSize(0u));

    return new CBlurPerformer(_driver, _radius, _dsFactor, _passesPerAxis, uboBuffer, uboDataStaticOffset);
}

video::ITexture* CBlurPerformer::createOutputTexture(video::ITexture* _inputTex, const std::string& _name)
{
    prepareForBlur(_inputTex->getSize(), false);

    return m_driver->addTexture(video::ITexture::ETT_2D, &m_outSize.X, 1, _name.c_str(), video::ECF_A16B16G16R16F);
}

#define PROFILE_BLUR_PERFORMER
void CBlurPerformer::blurTexture(video::ITexture* _inputTex, video::ITexture* _outputTex)
{
    GLint prevProgram{};
    glGetIntegerv(GL_CURRENT_PROGRAM, &prevProgram);

#ifdef PROFILE_BLUR_PERFORMER
    uint32_t timeTaken = 0;
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
    prepareForBlur(_inputTex->getSize(), true);
    assert(m_blurGeneralCs[0]);

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

    bindUBO(E_UBO_BINDING);

    constexpr uint32_t iterationsToDo = 2u;
    for (uint32_t i = 0u; i < iterationsToDo; ++i)
    {
        video::COpenGLExtensionHandler::extGlUseProgram(m_blurGeneralCs[i]);

        video::COpenGLExtensionHandler::extGlDispatchCompute(1u, (i ? m_outSize.X : m_outSize.Y), 1u);
        video::COpenGLExtensionHandler::extGlMemoryBarrier(i ? GL_TEXTURE_FETCH_BARRIER_BIT : GL_SHADER_STORAGE_BARRIER_BIT);
    }

    video::COpenGLExtensionHandler::extGlUseProgram(prevProgram);
    bindImage(0, prevImgBinding);

#ifdef PROFILE_BLUR_PERFORMER
    m_driver->endQuery(timeQuery);
    timeQuery->getQueryResult(&timeTaken);
    os::Printer::log("irr::ext::BlurPerformer GPU time taken:", std::to_string(timeTaken).c_str(), ELL_ERROR);
    timeQuery->drop();
#endif // PROFILE_BLUR_PERFORMER
}

void CBlurPerformer::prepareForBlur(const uint32_t* _inputSize, bool _createGpuStuff)
{
    core::vector2d<uint32_t> outSz = core::vector2d<uint32_t>(_inputSize[0], _inputSize[1]) / m_dsFactor;
    if ((_createGpuStuff && m_blurGeneralCs[0]) || (!_createGpuStuff && m_outSize != core::vector2d<uint32_t>(0u, 0u))) // already initialized
    {
        assert(outSz == m_outSize); // once initialized and output size is established, other output sizes are not allowed so that we don't have to recompile shaders
        return;
    }

    m_outSize = outSz;
    assert(m_outSize.X <= s_MAX_OUTPUT_SIZE_XY || m_outSize.Y <= s_MAX_OUTPUT_SIZE_XY);

    if (!_createGpuStuff)
        return;

    std::tie(m_blurGeneralCs[0], m_blurGeneralCs[1]) = makeShaders(m_outSize, m_passesPerAxisNum);

    if (!m_isCustomUbo)
        updateUBO();

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
    video::COpenGLExtensionHandler::extGlDeleteProgram(m_blurGeneralCs[0]);
    if (m_blurGeneralCs[0] != m_blurGeneralCs[1]) // if output size is square, then those are the same shader
        video::COpenGLExtensionHandler::extGlDeleteProgram(m_blurGeneralCs[1]);
}

CBlurPerformer::~CBlurPerformer()
{
    if (m_samplesSsbo)
        m_samplesSsbo->drop();
    if (m_ubo)
        m_ubo->drop();
    deleteShaders();
}

bool CBlurPerformer::genBlurPassCs(char* _out, size_t _bufSize, uint32_t _axisSize, const core::vector2d<uint32_t>& _outTexSize, uint32_t _passes, int _finalPass)
{
    const uint32_t SIM_THREADS_NUM = (_axisSize + s_MAX_WORK_GROUP_SIZE-1u) / s_MAX_WORK_GROUP_SIZE;

    const char loadFromGmem_fmt[] = "for (uint i = 0; i < %u; ++i) loadFromGmem(IN_START_IDX, LC_IDX + i*WG_SIZE);\n";

    const char loadFromTexture_fmt[] = "for (uint i = 0; i < %u; ++i) loadFromTexture(LC_IDX + i*WG_SIZE);\n";

    const char blurAndOutput_fmt[] = "blurred[%u] = blur(RADIUS, LC_IDX + %u*WG_SIZE);\n";

    const char fromLocalToSmem_fmt[] = "if (LC_IDX + %u*WG_SIZE < ACTUAL_SIZE) storeShared(getAddr(LC_IDX + %u*WG_SIZE), blurred[%u]);\n";

    const char outputFunc_fmt[] = "for (uint i = 0; i < %u; ++i) outputFunc(LC_IDX + i*WG_SIZE);\n";

    char buf[1u<<12];

    snprintf(buf, sizeof(buf), loadFromGmem_fmt, SIM_THREADS_NUM);
    std::string loadFromGmem = buf;

    snprintf(buf, sizeof(buf), loadFromTexture_fmt, SIM_THREADS_NUM);
    std::string loadFromTexture = buf;

    std::string blurAndOutput;
    for (uint32_t n = 0u; n < SIM_THREADS_NUM; ++n)
    {
        snprintf(buf, sizeof(buf), blurAndOutput_fmt, n, n);
        blurAndOutput += buf;
    }

    std::string fromLocalToSmem;
    for (uint32_t n = 0u; n < SIM_THREADS_NUM; ++n)
    {
        snprintf(buf, sizeof(buf), fromLocalToSmem_fmt, n, n, n);
        fromLocalToSmem += buf;
    }

    snprintf(buf, sizeof(buf), outputFunc_fmt, SIM_THREADS_NUM);
    std::string outputFunc = buf;

    std::string upsweep_fmt =
        "BARRIER;\n"
        "for (uint i = 0; i < %u; ++i) upsweep(%u, offset, LC_IDX + i*WG_SIZE);\n"
        "offset *= 2;\n";
    std::string downsweep_fmt =
        "BARRIER;\n"
        "offset /= 2;\n"
        "for (uint i = 0; i < %u; ++i) downsweep(%u, offset, LC_IDX + i*WG_SIZE);\n";

    const uint32_t pot = padToPoT(_axisSize);

    std::string upsweep; // unrolled upsweep loop
    for (uint32_t up_itr = pot/2u; up_itr > 0; up_itr /= 2u)
    {
        snprintf(buf, sizeof(buf), upsweep_fmt.c_str(), SIM_THREADS_NUM, up_itr);
        upsweep += buf;
    }
    std::string downsweep; // unrolled downsweep loop
    for (uint32_t down_itr = 1u; down_itr < pot; down_itr *= 2u)
    {
        snprintf(buf, sizeof(buf), downsweep_fmt.c_str(), SIM_THREADS_NUM, down_itr);
        downsweep += buf;
    }

    return snprintf(_out, _bufSize, CS_BLUR_SRC,
        pot,
        _axisSize,
        _outTexSize.X,
        _outTexSize.Y,
        std::min(_axisSize, s_MAX_WORK_GROUP_SIZE),
        SIM_THREADS_NUM,
        _passes,
        _finalPass,
        CS_CONVERSIONS,
        upsweep.c_str(),
        downsweep.c_str(),
        loadFromGmem.c_str(),
        loadFromTexture.c_str(),
        blurAndOutput.c_str(),
        fromLocalToSmem.c_str(),
        outputFunc.c_str()
    ) > 0;
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

void CBlurPerformer::writeUBOData(void* _dst) const
{
    BlurPassUBO* dst = reinterpret_cast<BlurPassUBO*>(_dst);
    dst->radius = m_radius;
}

void CBlurPerformer::updateUBO(const void* _contents)
{
    video::IDriverMemoryBacked::SDriverMemoryRequirements reqs;
    reqs.vulkanReqs.alignment = 4;
    reqs.vulkanReqs.memoryTypeBits = 0xffffffffu;
    reqs.memoryHeapLocation = video::IDriverMemoryAllocation::ESMT_DEVICE_LOCAL;
    reqs.mappingCapability = video::IDriverMemoryAllocation::EMCF_CANNOT_MAP;
    reqs.prefersDedicatedAllocation = true;
    reqs.requiresDedicatedAllocation = true;
    reqs.vulkanReqs.size = getRequiredUBOSize(m_driver);
    video::IGPUBuffer* stagingBuf = m_driver->createGPUBufferOnDedMem(reqs, true);

    stagingBuf->updateSubRange({0u, reqs.vulkanReqs.size}, _contents);

    m_driver->copyBuffer(stagingBuf, m_ubo, 0u, m_uboOffset, reqs.vulkanReqs.size);
    stagingBuf->drop();
}

void CBlurPerformer::updateUBO()
{
    uint8_t mem[1u<<12];
    assert(sizeof(mem) >= getRequiredUBOSize(m_driver));

    writeUBOData(mem);
    updateUBO(mem);
}

auto CBlurPerformer::makeShaders(const core::vector2d<uint32_t>& _outSize, uint32_t _passesPerAxis) -> tuple2xu32
{
    uint32_t blurx{}, blury{};

    const size_t bufSize = strlen(CS_BLUR_SRC) + strlen(CS_CONVERSIONS) + 10000u;
    char* src = (char*)malloc(bufSize);

    auto doCleaning = [&, blurx, blury, src]() {
        for (uint32_t s : { blurx, blury })
            video::COpenGLExtensionHandler::extGlDeleteProgram(s);
        free(src);
        return tuple2xu32(0u, 0u);
    };

    if (!genBlurPassCs(src, bufSize, _outSize.X, _outSize, _passesPerAxis, 0))
        return doCleaning();
    blurx = createComputeShader(src);

    if (!genBlurPassCs(src, bufSize, _outSize.Y, _outSize, _passesPerAxis, 1))
        return doCleaning();
    blury = createComputeShader(src);

    for (uint32_t s : {blurx, blury})
    {
        if (!s)
            return doCleaning();
    }
    free(src);

    return std::make_tuple(blurx, blury);
}

void CBlurPerformer::bindUBO(uint32_t _bnd) const
{
    auto auxCtx = const_cast<video::COpenGLDriver::SAuxContext*>(static_cast<video::COpenGLDriver*>(m_driver)->getThreadContext());

    const video::COpenGLBuffer* buf{ static_cast<const video::COpenGLBuffer*>(m_ubo) };
    ptrdiff_t offset = m_uboOffset;
    ptrdiff_t size = getUBOSizePerShaderPass();

    auxCtx->setActiveUBO(_bnd, 1u, &buf, &offset, &size);
}
