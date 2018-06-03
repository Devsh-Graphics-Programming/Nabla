#define _IRR_STATIC_LIB_
#include <irrlicht.h>
#include "../source/Irrlicht/COpenGLExtensionHandler.h"
#include "../source/Irrlicht/COpenGLBuffer.h"

#include "createComputeShader.h"

using namespace irr;
using namespace core;


//!Same As Last Example
class MyEventReceiver : public IEventReceiver
{
public:

    MyEventReceiver()
    {
    }

    bool OnEvent(const SEvent& event)
    {
        if (event.EventType == irr::EET_KEY_INPUT_EVENT && !event.KeyInput.PressedDown)
        {
            switch (event.KeyInput.Key)
            {
            case irr::KEY_KEY_Q: // switch wire frame mode
                exit(0);
                return true;
            default:
                break;
            }
        }

        return false;
    }

private:
};

class SimpleCallBack : public video::IShaderConstantSetCallBack
{
    int32_t mvpUniformLocation;
    int32_t cameraDirUniformLocation;
    video::E_SHADER_CONSTANT_TYPE mvpUniformType;
    video::E_SHADER_CONSTANT_TYPE cameraDirUniformType;

public:
    SimpleCallBack() : cameraDirUniformLocation(-1), cameraDirUniformType(video::ESCT_FLOAT_VEC3) {}

    virtual void PostLink(video::IMaterialRendererServices* services, const video::E_MATERIAL_TYPE& materialType, const core::array<video::SConstantLocationNamePair>& constants)
    {
        for (size_t i = 0; i<constants.size(); i++)
        {
            if (constants[i].name == "MVP")
            {
                mvpUniformLocation = constants[i].location;
                mvpUniformType = constants[i].type;
            }
            else if (constants[i].name == "cameraPos")
            {
                cameraDirUniformLocation = constants[i].location;
                cameraDirUniformType = constants[i].type;
            }
        }
    }

    virtual void OnSetConstants(video::IMaterialRendererServices* services, int32_t userData)
    {
        core::vectorSIMDf modelSpaceCamPos;
        modelSpaceCamPos.set(services->getVideoDriver()->getTransform(video::E4X3TS_WORLD_VIEW_INVERSE).getTranslation());
        services->setShaderConstant(&modelSpaceCamPos, cameraDirUniformLocation, cameraDirUniformType, 1);
        services->setShaderConstant(services->getVideoDriver()->getTransform(video::EPTS_PROJ_VIEW_WORLD).pointer(), mvpUniformLocation, mvpUniformType, 1);
    }

    virtual void OnUnsetMaterial() {}
};

class ISorter : public IReferenceCounted
{
protected:
    explicit ISorter(video::IVideoDriver* _vd) : m_driver{_vd}, m_idxBuf{nullptr} {}
    virtual ~ISorter() = default;

public:
    virtual void init(scene::ICPUMeshBuffer* _mb) = 0;
    virtual void run(const core::vector3df& _camPos) = 0;
    virtual void setIndexBuffer(const video::IGPUBuffer* _idxBuf)
    {
        if (_idxBuf)
        {
            const video::IGPUBuffer* prev = m_idxBuf;
            m_idxBuf = _idxBuf;
            printf("IDXISZE: %u\n", m_idxBuf->getSize());
            m_idxBuf->grab();
            if (prev)
                prev->drop();
        }
        else printf("NO IDX BUF!!!\n");
    }

protected:
    video::IVideoDriver* m_driver;
    const video::IGPUBuffer* m_idxBuf;
};

class RadixSorter : public ISorter
{
    constexpr static const char* CS_GEN_KEYS_SRC =
R"XDDD(
#version 430 core
layout(local_size_x = 256) in;

#define G_IDX gl_GlobalInvocationID.x

layout(std430, binding = 7) buffer Pos { vec4 pos[]; };
layout(std430, binding = 0) buffer Key { float key[]; };
layout(std430, binding = 1) buffer Indices { uint indices[]; };

layout(location = 1) uniform vec3 camPos;

void main()
{
    vec3 v;
    
    if (2*G_IDX < indices.length())
    {
        v = camPos - pos[indices[2*G_IDX]].xyz;
        key[2*G_IDX] = dot(v, v);
    }
    if (2*G_IDX+1 < indices.length())
    {
        v = camPos - pos[indices[2*G_IDX+1]].xyz;
        key[2*G_IDX+1] = dot(v, v);
    }
}
)XDDD";

    enum
    {
        E_IN_KEYS_BND = 0,
        E_IN_VALS_BND = 1,
        E_OUT_KEYS_BND = 2,
        E_OUT_VALS_BND = 3,
        E_SUMS_BND = 6,
        E_HISTOGRAM_BND = 5,
        E_PRESUM_BND = 4,
        E_POSITIONS_BND = 7,

        E_SHIFT_LOC = 0,
        E_CAM_POS_LOC = 1
    };
    using gl = video::COpenGLExtensionHandler;

    static constexpr size_t ELEMENTS_PER_WG = 512u;

protected:
    ~RadixSorter()
    {
        gl::extGlDeleteProgram(m_genKeysCs);
        gl::extGlDeleteProgram(m_histogramCs);
        gl::extGlDeleteProgram(m_presumCs);
        gl::extGlDeleteProgram(m_permuteCs);
        if (m_posBuf)
            m_posBuf->drop();
        if (m_keyBuf1)
            m_keyBuf1->drop();
        if (m_keyBuf2)
            m_keyBuf2->drop();
        if (m_idxBuf2)
            m_idxBuf2->drop();
        if (m_idxBuf)
            m_idxBuf->drop();
        if (m_psumBuf)
            m_psumBuf->drop();
        if (m_sumsBuf)
            m_sumsBuf->drop();
        if (m_histogramBuf)
            m_histogramBuf->drop();
    }

public:
    RadixSorter(video::IVideoDriver* _vd) :
        ISorter(_vd),
        m_genKeysCs{}, m_histogramCs{}, m_presumCs{}, m_permuteCs{},
        m_posBuf{nullptr}, m_keyBuf1{nullptr}, m_keyBuf2{nullptr}, m_idxBuf2{ nullptr }, m_histogramBuf{nullptr},
        m_wgCnt{}
    {
    }

    void init(scene::ICPUMeshBuffer* _mb) override
    {
        m_histogramBuf = m_driver->createGPUBuffer(64 * 16 * sizeof(GLuint), nullptr);

        scene::IMeshDataFormatDesc<core::ICPUBuffer>* desc = _mb->getMeshDataAndFormat();
        std::vector<core::vectorSIMDf> pos;
        vectorSIMDf v;
        size_t ix = 0u;
        while (_mb->getAttribute(v, _mb->getPositionAttributeIx(), ix++))
            pos.push_back(v);

        const size_t idxCount = pos.size();
        core::ICPUBuffer* idxBuf = new core::ICPUBuffer(4 * idxCount);
        uint32_t* indices = (uint32_t*)idxBuf->getPointer();
        for (uint32_t i = 0u; i < idxCount; ++i)
            indices[i] = i;


        desc->mapIndexBuffer(idxBuf);
        idxBuf->drop();
        _mb->setIndexCount(idxCount);
        _mb->setIndexBufferOffset(0u);
        _mb->setPrimitiveType(scene::EPT_POINTS);
        _mb->setIndexType(video::EIT_32BIT);
        printf("pos.size() == %u\n", pos.size());

        m_wgCnt = (idxCount + ELEMENTS_PER_WG - 1u) / ELEMENTS_PER_WG;
        printf("m_wgCnt == %u\n", m_wgCnt);

        m_posBuf = m_driver->createGPUBuffer(pos.size() * sizeof(v), pos.data());
        m_keyBuf1 = m_driver->createGPUBuffer(idxCount * sizeof(GLuint), nullptr);
        m_keyBuf2 = m_driver->createGPUBuffer(idxCount * sizeof(GLuint), nullptr);
        m_sumsBuf = m_driver->createGPUBuffer(m_wgCnt * 2 * sizeof(GLuint), nullptr);
        m_histogramBuf = m_driver->createGPUBuffer(2 * sizeof(GLuint), nullptr);
        m_psumBuf = m_driver->createGPUBuffer(idxCount * 2 * sizeof(GLuint), nullptr);

        m_genKeysCs = createComputeShader(CS_GEN_KEYS_SRC);
        m_histogramCs = createComputeShaderFromFile("../shaders/histogram.comp");
        m_presumCs = createComputeShaderFromFile("../shaders/xpresum.comp");
        m_permuteCs = createComputeShaderFromFile("../shaders/permute.comp");
    }

    void run(const core::vector3df& _camPos) override
    {
        GLint prevProgram;
        glGetIntegerv(GL_CURRENT_PROGRAM, &prevProgram);

        gl::extGlBindBuffersBase(GL_SHADER_STORAGE_BUFFER, E_HISTOGRAM_BND, 1, &static_cast<const video::COpenGLBuffer*>(m_histogramBuf)->getOpenGLName());
        gl::extGlBindBuffersBase(GL_SHADER_STORAGE_BUFFER, E_IN_KEYS_BND, 1, &static_cast<const video::COpenGLBuffer*>(m_keyBuf1)->getOpenGLName());
        gl::extGlBindBuffersBase(GL_SHADER_STORAGE_BUFFER, E_OUT_KEYS_BND, 1, &static_cast<const video::COpenGLBuffer*>(m_keyBuf2)->getOpenGLName());
        gl::extGlBindBuffersBase(GL_SHADER_STORAGE_BUFFER, E_IN_VALS_BND, 1, &static_cast<const video::COpenGLBuffer*>(m_idxBuf)->getOpenGLName());
        gl::extGlBindBuffersBase(GL_SHADER_STORAGE_BUFFER, E_OUT_VALS_BND, 1, &static_cast<const video::COpenGLBuffer*>(m_idxBuf2)->getOpenGLName());
        gl::extGlBindBuffersBase(GL_SHADER_STORAGE_BUFFER, E_POSITIONS_BND, 1, &static_cast<const video::COpenGLBuffer*>(m_posBuf)->getOpenGLName());
        gl::extGlBindBuffersBase(GL_SHADER_STORAGE_BUFFER, E_PRESUM_BND, 1, &static_cast<const video::COpenGLBuffer*>(m_psumBuf)->getOpenGLName());
        gl::extGlBindBuffersBase(GL_SHADER_STORAGE_BUFFER, E_SUMS_BND, 1, &static_cast<const video::COpenGLBuffer*>(m_sumsBuf)->getOpenGLName());

        gl::extGlUseProgram(m_genKeysCs);
        gl::extGlProgramUniform3fv(m_genKeysCs, E_CAM_POS_LOC, 1u, &_camPos.X);

        gl::extGlDispatchCompute(m_wgCnt, 1u, 1u);
        gl::extGlMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

        const GLuint histogram[] {0u, 0u};
        uint8_t sumsIn[1u<<13]; // enough for up to 1024 work groups, i.e. 524288 indices
        uint8_t sumsOut[1u<<13];
        for (GLuint nbit = 0u; nbit < 31u; ++nbit)
        {
            gl::extGlBindBuffersBase(GL_SHADER_STORAGE_BUFFER, E_IN_KEYS_BND, 1, &static_cast<const video::COpenGLBuffer*>(m_keyBuf1)->getOpenGLName());
            gl::extGlBindBuffersBase(GL_SHADER_STORAGE_BUFFER, E_OUT_KEYS_BND, 1, &static_cast<const video::COpenGLBuffer*>(m_keyBuf2)->getOpenGLName());
            gl::extGlBindBuffersBase(GL_SHADER_STORAGE_BUFFER, E_IN_VALS_BND, 1, &static_cast<const video::COpenGLBuffer*>(m_idxBuf)->getOpenGLName());
            gl::extGlBindBuffersBase(GL_SHADER_STORAGE_BUFFER, E_OUT_VALS_BND, 1, &static_cast<const video::COpenGLBuffer*>(m_idxBuf2)->getOpenGLName());

            // zero histogram
            gl::extGlNamedBufferSubData(static_cast<const video::COpenGLBuffer*>(m_histogramBuf)->getOpenGLName(), 0, sizeof(histogram), histogram);

            gl::extGlUseProgram(m_histogramCs);
            gl::extGlProgramUniform1uiv(m_histogramCs, E_SHIFT_LOC, 1, &nbit);
            gl::extGlDispatchCompute(m_wgCnt, 1u, 1u);
            gl::extGlMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

            gl::extGlUseProgram(m_presumCs);
            gl::extGlProgramUniform1uiv(m_presumCs, E_SHIFT_LOC, 1, &nbit);
            gl::extGlDispatchCompute(m_wgCnt, 1u, 1u);
            gl::extGlMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
            // some other barrier needed here?
            gl::extGlGetNamedBufferSubData(static_cast<const video::COpenGLBuffer*>(m_sumsBuf)->getOpenGLName(), 0, 2*m_wgCnt*sizeof(GLuint), sumsIn);
            xpsum(sumsIn, sumsOut, m_wgCnt);
            gl::extGlNamedBufferSubData(static_cast<const video::COpenGLBuffer*>(m_sumsBuf)->getOpenGLName(), 0, 2*m_wgCnt*sizeof(GLuint), sumsOut);

            gl::extGlUseProgram(m_permuteCs);
            gl::extGlProgramUniform1uiv(m_permuteCs, E_SHIFT_LOC, 1, &nbit);
            gl::extGlDispatchCompute(m_wgCnt, 1u, 1u);
            gl::extGlMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

            std::swap(m_keyBuf1, m_keyBuf2);
            std::swap(m_idxBuf, m_idxBuf2);
        }
        std::swap(m_keyBuf1, m_keyBuf2);
        std::swap(m_idxBuf, m_idxBuf2);
        // copy result to actual index buffer
        gl::extGlCopyNamedBufferSubData(static_cast<const video::COpenGLBuffer*>(m_idxBuf2)->getOpenGLName(), static_cast<const video::COpenGLBuffer*>(m_idxBuf)->getOpenGLName(), 0, 0, m_idxBuf->getSize());

        gl::extGlMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);

        // rebind previous program
        gl::extGlUseProgram(prevProgram);
        // zero ssbo bindings
        gl::extGlBindBuffersBase(GL_SHADER_STORAGE_BUFFER, 0, 8, nullptr);
    }

    void setIndexBuffer(const video::IGPUBuffer* _idxBuf) override
    {
        ISorter::setIndexBuffer(_idxBuf);
        if (m_idxBuf)
        {
            if (m_idxBuf2)
                m_idxBuf2->drop();
            m_idxBuf2 = m_driver->createGPUBuffer(m_idxBuf->getSize(), nullptr);
        }
    }

    void xpsum(const void* _in, void* _out, size_t _cnt)
    {
        using uvec2 = core::vector2d<GLuint>;

        const uvec2* in = (const uvec2*)_in;
        uvec2* out = (uvec2*)_out;

        out[0] = uvec2(0, 0);
        out[1] = in[0];
        for (size_t i = 2u; i < _cnt; ++i)
            out[i] = out[i-1] + in[i-1];
    }

private:
    GLuint m_genKeysCs, m_histogramCs, m_presumCs, m_permuteCs;
    const video::IGPUBuffer *m_posBuf, *m_keyBuf1, *m_keyBuf2, *m_idxBuf2, *m_histogramBuf, *m_psumBuf, *m_sumsBuf;
    GLuint m_wgCnt;
};

class ProgressiveSorter : public ISorter
{
    enum 
    {
        E_OFF_LOC = 0,
        E_CAM_POS_LOC = 1,
        E_SIZE_LOC = 2,

        E_INDICES_BIDING = 0,
        E_POS_BINDING = 1
    };

    static constexpr const char* CS_SRC = R"XDDD(
    #version 430 core
    layout(local_size_x = 256) in;

    layout(std430, binding = 0) restrict buffer b0 {
	    uint indices[];
    };
    layout(std430, binding = 1) restrict readonly buffer b1 {
	    vec4 pos[];
    };

    layout(location = 0) uniform uint off;
    layout(location = 1) uniform vec3 camPos;
    layout(location = 2) uniform uint size;

    void swap(inout uint idx0, inout uint idx1)
    {
	    uint tmp = idx0;
	    idx0 = idx1;
	    idx1 = tmp;
    }

    void main()
    {
	    const uint IDX = 2*gl_GlobalInvocationID.x + off;
	
	    if (IDX + 1 < size)
	    {
		    uint idx0 = indices[IDX], idx1 = indices[IDX+1];
		    const vec3 v0 = camPos - pos[idx0].xyz, v1 = camPos - pos[idx1].xyz;
		    const float key0 = dot(v0, v0), key1 = dot(v1, v1);
		
		    if (key0 < key1)
			    swap(idx0, idx1);
			
		    indices[IDX] = idx0;
		    indices[IDX+1] = idx1;
	    }
    }
    )XDDD";

protected:
    ~ProgressiveSorter()
    {
        if (m_posBuf)
            m_posBuf->drop();
        if (m_idxBuf)
            m_idxBuf->drop();
    }

public:
    ProgressiveSorter(video::IVideoDriver* _vd) :
        ISorter(_vd),
        m_cs{createComputeShader(CS_SRC)},
        m_posBuf{nullptr},
        m_startIdx{0u},
        m_wgCount{}
    {}
    void init(scene::ICPUMeshBuffer* _mb) override
    {
        scene::IMeshDataFormatDesc<core::ICPUBuffer>* desc = _mb->getMeshDataAndFormat();
        std::vector<core::vectorSIMDf> pos;
        vectorSIMDf v;
        size_t ix{};
        while (_mb->getAttribute(v, _mb->getPositionAttributeIx(), ix++))
            pos.push_back(v);

        m_posBuf = m_driver->createGPUBuffer(pos.size() * sizeof(v), pos.data());

        core::ICPUBuffer* idxBuf = new core::ICPUBuffer(4 * pos.size());
        uint32_t* indices = (uint32_t*)idxBuf->getPointer();
        for (uint32_t i = 0u; i < pos.size(); ++i)
            indices[i] = i;

        desc->mapIndexBuffer(idxBuf);
        idxBuf->drop();
        _mb->setIndexCount(pos.size());
        _mb->setIndexBufferOffset(0u);
        _mb->setPrimitiveType(scene::EPT_POINTS);
        _mb->setIndexType(video::EIT_32BIT);
        printf("pos.size() == %u\n", pos.size());

        m_wgCount = (GLuint)std::ceil(double(pos.size()) / 256.);
        printf("wgCount == %u\n", m_wgCount);
    }

    void run(const core::vector3df& _camPos) override
    {
        GLint prevProgram;
        glGetIntegerv(GL_CURRENT_PROGRAM, &prevProgram);

        video::COpenGLExtensionHandler::extGlBindBuffersBase(GL_SHADER_STORAGE_BUFFER, E_POS_BINDING, 1, &static_cast<video::COpenGLBuffer*>(m_posBuf)->getOpenGLName());
        video::COpenGLExtensionHandler::extGlBindBuffersBase(GL_SHADER_STORAGE_BUFFER, E_INDICES_BIDING, 1, &static_cast<const video::COpenGLBuffer*>(m_idxBuf)->getOpenGLName());

        video::COpenGLExtensionHandler::extGlUseProgram(m_cs);

        video::COpenGLExtensionHandler::extGlProgramUniform1uiv(m_cs, E_OFF_LOC, 1u, &m_startIdx);
        video::COpenGLExtensionHandler::extGlProgramUniform3fv(m_cs, E_CAM_POS_LOC, 1u, &_camPos.X);
        const GLuint size = m_posBuf->getSize() / sizeof(core::vectorSIMDf);
        video::COpenGLExtensionHandler::extGlProgramUniform1uiv(m_cs, E_SIZE_LOC, 1u, &size);

        video::COpenGLExtensionHandler::extGlDispatchCompute(m_wgCount, 1u, 1u);

        video::COpenGLExtensionHandler::extGlUseProgram(prevProgram);
        video::COpenGLExtensionHandler::extGlBindBuffersBase(GL_SHADER_STORAGE_BUFFER, E_INDICES_BIDING, 2, nullptr);

        video::COpenGLExtensionHandler::extGlMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);

        m_startIdx = !m_startIdx;
    }

private:
    GLuint m_cs;
    video::IGPUBuffer* m_posBuf;
    mutable GLuint m_startIdx;
    GLuint m_wgCount;
};


class BitonicSorter : public ISorter
{
    enum
    {
        E_J_LOC = 0,
        E_K_LOC = 1,
        E_CAM_POS_LOC = 2,

        E_INDICES_BIDING = 0,
        E_POS_BINDING = 1
    };

    static constexpr const char* CS_SRC = R"XDDD(
    #version 430 core

    layout(local_size_x = 256) in;

    layout(std430, binding = 0) restrict buffer b0 {
	    uint indices[];
    };
    layout(std430, binding = 1) restrict readonly buffer b1 {
	    vec4 pos[];
    };

    layout(location = 0) uniform uint j;
    layout(location = 1) uniform uint k;
    layout(location = 2) uniform vec3 camPos;

    void swap(inout uint idx0, inout uint idx1)
    {
	    uint tmp = idx0;
	    idx0 = idx1;
	    idx1 = tmp;
    }

    void main() {
	    const uint i = gl_GlobalInvocationID.x;
	    const uint ixorj = i ^ j;
	
	    uint idx0 = indices[ixorj], idx1 = indices[i];
	    const vec3 v0 = camPos - pos[idx0].xyz, v1 = camPos - pos[idx1].xyz;
	    const float key0 = dot(v0, v0), key1 = dot(v1, v1);
	
	    if (ixorj > i) {
		    if ((i & k) == 0) { /*sort asc*/
			    if (key0 < key1) {
				    swap(idx0, idx1);
			    }
		    }
		    else { /*sort desc*/
			    if (key1 < key0) {
				    swap(idx0, idx1);
			    }
		    }
		    indices[ixorj] = idx0;
		    indices[i] = idx1;
	    }
    }
    )XDDD";

protected:
    ~BitonicSorter()
    {
        if (m_posBuf)
            m_posBuf->drop();
        if (m_idxBuf)
            m_idxBuf->drop();
    }

public:
    BitonicSorter(video::IVideoDriver* _vd) :
        ISorter(_vd),
        m_cs{ createComputeShader(CS_SRC) },
        m_posBuf{ nullptr },
        m_wgCount{}
    {}
    void init(scene::ICPUMeshBuffer* _mb) override
    {
        scene::IMeshDataFormatDesc<core::ICPUBuffer>* desc = _mb->getMeshDataAndFormat();
        std::vector<core::vectorSIMDf> pos;
        vectorSIMDf v;
        size_t ix{};
        while (_mb->getAttribute(v, _mb->getPositionAttributeIx(), ix++))
            pos.push_back(v);

        m_posBuf = m_driver->createGPUBuffer(pos.size() * sizeof(v), pos.data());

        const size_t idxCount = 1u << ((size_t)std::ceil(std::log2((double)pos.size())));
        core::ICPUBuffer* idxBuf = new core::ICPUBuffer(4 * idxCount);
        uint32_t* indices = (uint32_t*)idxBuf->getPointer();
        memset(indices, 0, (idxCount - pos.size()) * 4);
        for (uint32_t i = idxCount - pos.size(); i < idxCount; ++i)
            indices[i] = i - (idxCount - pos.size());

        desc->mapIndexBuffer(idxBuf);
        idxBuf->drop();
        _mb->setIndexCount(idxCount);
        _mb->setIndexBufferOffset(0u);
        _mb->setPrimitiveType(scene::EPT_POINTS);
        _mb->setIndexType(video::EIT_32BIT);
        printf("pos.size() == %u\n", pos.size());

        m_wgCount = idxCount / 256u;
        printf("wgCount == %u\n", m_wgCount);
    }

    void run(const core::vector3df& _camPos) override
    {
        GLint prevProgram;
        glGetIntegerv(GL_CURRENT_PROGRAM, &prevProgram);

        video::COpenGLExtensionHandler::extGlBindBuffersBase(GL_SHADER_STORAGE_BUFFER, E_POS_BINDING, 1, &static_cast<video::COpenGLBuffer*>(m_posBuf)->getOpenGLName());
        video::COpenGLExtensionHandler::extGlBindBuffersBase(GL_SHADER_STORAGE_BUFFER, E_INDICES_BIDING, 1, &static_cast<const video::COpenGLBuffer*>(m_idxBuf)->getOpenGLName());

        video::COpenGLExtensionHandler::extGlUseProgram(m_cs);

        video::COpenGLExtensionHandler::extGlProgramUniform3fv(m_cs, E_CAM_POS_LOC, 1u, &_camPos.X);

        for (GLuint k = 2; k <= m_wgCount*256u; k <<= 1)
        {
            for (GLuint j = k >> 1; j > 0; j >>= 1)
            {
                video::COpenGLExtensionHandler::extGlProgramUniform1uiv(m_cs, E_J_LOC, 1, &j);
                video::COpenGLExtensionHandler::extGlProgramUniform1uiv(m_cs, E_K_LOC, 1, &k);
                video::COpenGLExtensionHandler::pGlDispatchCompute(m_wgCount, 1, 1);
                video::COpenGLExtensionHandler::pGlMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
            }
        }

        video::COpenGLExtensionHandler::extGlUseProgram(prevProgram);
        video::COpenGLExtensionHandler::extGlBindBuffersBase(GL_SHADER_STORAGE_BUFFER, E_INDICES_BIDING, 2, nullptr);

        video::COpenGLExtensionHandler::extGlMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
    }

private:
    GLuint m_cs;
    video::IGPUBuffer* m_posBuf;
    GLuint m_wgCount;
};

int main()
{
    // create device with full flexibility over creation parameters
    // you can add more parameters if desired, check irr::SIrrlichtCreationParameters
    irr::SIrrlichtCreationParameters params;
    params.Bits = 24; //may have to set to 32bit for some platforms
    params.ZBufferBits = 24; //we'd like 32bit here
    params.DriverType = video::EDT_OPENGL; //! Only Well functioning driver, software renderer left for sake of 2D image drawing
    params.WindowSize = dimension2d<uint32_t>(1280, 720);
    params.Fullscreen = false;
    params.Vsync = true; //! If supported by target platform
    params.Doublebuffer = true;
    params.Stencilbuffer = false; //! This will not even be a choice soon
    IrrlichtDevice* device = createDeviceEx(params);

    if (device == 0)
        return 1; // could not create selected driver.

    video::IVideoDriver* driver = device->getVideoDriver();

    SimpleCallBack* cb = new SimpleCallBack();
    video::E_MATERIAL_TYPE newMaterialType = (video::E_MATERIAL_TYPE)driver->getGPUProgrammingServices()->addHighLevelShaderMaterialFromFiles("../shaders/mesh.vert",
        "", "", "", //! No Geometry or Tessellation Shaders
        "../shaders/mesh.frag",
        3, video::EMT_SOLID, //! 3 vertices per primitive (this is tessellation shader relevant only
        cb, //! Our Shader Callback
        0); //! No custom user data
    cb->drop();


    scene::ISceneManager* smgr = device->getSceneManager();
    driver->setTextureCreationFlag(video::ETCF_ALWAYS_32_BIT, true);
    scene::ICameraSceneNode* camera =
        smgr->addCameraSceneNodeFPS(0, 100.0f, 0.01f);
    camera->setPosition(core::vector3df(-4, 0, 0));
    camera->setTarget(core::vector3df(0, 0, 0));
    camera->setNearValue(0.01f);
    camera->setFarValue(100.0f);
    smgr->setActiveCamera(camera);
    device->getCursorControl()->setVisible(false);
    MyEventReceiver receiver;
    device->setEventReceiver(&receiver);

    scene::ICPUMesh* cpumesh = smgr->getMesh("../../media/cow.obj");
    ISorter* sorter =
        new RadixSorter(driver);
        //new ProgressiveSorter(driver);
        //new BitonicSorter(driver);
    sorter->init(cpumesh->getMeshBuffer(0));
    scene::IGPUMesh* gpumesh = driver->createGPUMeshFromCPU(dynamic_cast<scene::SCPUMesh*>(cpumesh));
    sorter->setIndexBuffer(gpumesh->getMeshBuffer(0)->getMeshDataAndFormat()->getIndexBuffer());
    printf("IDX_TYPE %d\n", gpumesh->getMeshBuffer(0)->getIndexType());
    smgr->addMeshSceneNode(gpumesh, 0, -1, core::vector3df(), core::vector3df(), core::vector3df(4.f))->setMaterialType(newMaterialType);
    gpumesh->drop();

    uint64_t lastFPSTime = 0;

    while (device->run())
    {
        driver->beginScene(true, true, video::SColor(255, 0, 0, 255));

        //! This animates (moves) the camera and sets the transforms
        //! Also draws the meshbuffer
        smgr->drawAll();
        sorter->run(camera->getPosition());

        driver->endScene();

        // display frames per second in window title
        uint64_t time = device->getTimer()->getRealTime();
        if (time - lastFPSTime > 1000)
        {
            std::wostringstream sstr;
            sstr << L"Builtin Nodes Demo - Irrlicht Engine FPS:" << driver->getFPS() << " PrimitvesDrawn:" << driver->getPrimitiveCountDrawn();

            device->setWindowCaption(sstr.str().c_str());
            lastFPSTime = time;
        }
    }

    //create a screenshot
    video::IImage* screenshot = driver->createImage(video::ECF_A8R8G8B8, params.WindowSize);
    glReadPixels(0, 0, params.WindowSize.Width, params.WindowSize.Height, GL_BGRA, GL_UNSIGNED_INT_8_8_8_8_REV, screenshot->getData());
    {
        // images are horizontally flipped, so we have to fix that here.
        uint8_t* pixels = (uint8_t*)screenshot->getData();

        const int32_t pitch = screenshot->getPitch();
        uint8_t* p2 = pixels + (params.WindowSize.Height - 1) * pitch;
        uint8_t* tmpBuffer = new uint8_t[pitch];
        for (uint32_t i = 0; i < params.WindowSize.Height; i += 2)
        {
            memcpy(tmpBuffer, pixels, pitch);
            memcpy(pixels, p2, pitch);
            memcpy(p2, tmpBuffer, pitch);
            pixels += pitch;
            p2 -= pitch;
        }
        delete[] tmpBuffer;
    }
    driver->writeImageToFile(screenshot, "./screenshot.png");
    screenshot->drop();
    device->sleep(3000);

    sorter->drop();
    device->drop();

    return 0;
}