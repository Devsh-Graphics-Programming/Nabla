// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#define _NBL_STATIC_LIB_
#include <nabla.h>

#include "../common/QToQuitEventReceiver.h"


using namespace nbl;
using namespace core;

class ISorter : public IReferenceCounted
{
	protected:
		explicit ISorter(video::IVideoDriver* _vd) : m_driver{_vd}, m_idxBuf(), m_16to32Cs{0u} {}
		virtual ~ISorter() = default;

	public:
		virtual void init(asset::ICPUMeshBuffer* _mb) = 0;
		virtual void run(const core::vector3df& _camPos) = 0;

		//! Takes index buffer from passed meshbuffer and converts it (creates new buffer and overrides old one in meshbuffer) to 32bit indices if they're 16bit.
		//! Assumues that index count is always power of two.
		virtual void setIndexBuffer(video::IGPUMeshBuffer* _mb)
		{
			if (_mb)
			{
				video::IGPUBuffer* idxBuf = nullptr;
				if (_mb->getIndexType() == asset::EIT_16BIT)
				{
					if (!m_16to32Cs)
						m_16to32Cs = createComputeShaderFromFile("../shaders/16to32.comp");
					auto idxBuf16 = _mb->getMeshDataAndFormat()->getIndexBuffer();
					idxBuf = m_driver->createDeviceLocalGPUBufferOnDedMem(4*_mb->getIndexCount());

					auto auxCtx = const_cast<video::COpenGLDriver::SAuxContext*>(static_cast<video::COpenGLDriver*>(m_driver)->getThreadContext());
					const video::COpenGLBuffer* bufs[2]{ static_cast<const video::COpenGLBuffer*>(idxBuf16), static_cast<const video::COpenGLBuffer*>(idxBuf) };
					const ptrdiff_t off[2]{ _mb->getIndexBufferOffset(), 0 }, sz[2]{ 2*_mb->getIndexCount(), 4*_mb->getIndexCount()};
					auxCtx->setActiveSSBO(0, 2, bufs, off, sz);

					GLint prevProgram = 0;
					glGetIntegerv(GL_CURRENT_PROGRAM, &prevProgram);
                
					using gl = video::COpenGLExtensionHandler;
					gl::extGlUseProgram(m_16to32Cs);
					gl::extGlDispatchCompute(_mb->getIndexCount()/2/256, 1, 1);
					gl::extGlMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

					gl::extGlUseProgram(prevProgram);

					_mb->getMeshDataAndFormat()->setIndexBuffer(core::smart_refctd_ptr<video::IGPUBuffer>(idxBuf));
					_mb->setIndexBufferOffset(0u);
					_mb->setIndexType(asset::EIT_32BIT);
				}
				else
					idxBuf = const_cast<video::IGPUBuffer*>(_mb->getMeshDataAndFormat()->getIndexBuffer());

				m_idxBuf = core::smart_refctd_ptr<video::IGPUBuffer>(idxBuf);
				printf("IDXISZE: %u\n", m_idxBuf->getSize());
			}
			else printf("NO IDX BUF!!!\n");
		}

	protected:
		video::IVideoDriver* m_driver;
		core::smart_refctd_ptr<video::IGPUBuffer> m_idxBuf;

	private:
		GLuint m_16to32Cs;
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

layout(push_constant) uniform Control
{
	vec4 camPos;
} ctrl;

void main()
{
    vec3 v;
    
    if (2*G_IDX < indices.length())
    {
        v = ctrl.camPos.xyz - pos[indices[2*G_IDX]].xyz;
        key[2*G_IDX] = dot(v, v);
    }
    if (2*G_IDX+1 < indices.length())
    {
        v = ctrl.camPos.xyz - pos[indices[2*G_IDX+1]].xyz;
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
        E_PRESUM_BND = 4,
        E_HISTOGRAM_BND = 5,
        E_SUMS_BND = 6,
        E_POSITIONS_BND = 7
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
    }

public:
    RadixSorter(video::IVideoDriver* _vd) :
        ISorter(_vd),
        m_genKeysCs{}, m_histogramCs{}, m_presumCs{}, m_permuteCs{},
        m_posBuf(), m_keyBuf1(), m_keyBuf2(), m_idxBuf2(), m_histogramBuf(),
        m_wgCnt{}
    {
    }

    void init(asset::ICPUMeshBuffer* _mb) override
    {
        m_histogramBuf = core::smart_refctd_ptr<video::IGPUBuffer>(m_driver->createDeviceLocalGPUBufferOnDedMem(64 * 16 * sizeof(GLuint)),core::dont_grab);

        asset::IMeshDataFormatDesc<asset::ICPUBuffer>* desc = _mb->getMeshDataAndFormat();
        std::vector<core::vectorSIMDf> pos;
        vectorSIMDf v;
        size_t ix = 0u;
        while (_mb->getAttribute(v, _mb->getPositionAttributeIx(), ix++))
            pos.push_back(v);

        const size_t idxCount = pos.size();
		{
			auto idxBuf = core::make_smart_refctd_ptr<asset::ICPUBuffer>(sizeof(uint32_t)*idxCount);
			uint32_t* indices = (uint32_t*)idxBuf->getPointer();
			for (uint32_t i = 0u; i < idxCount; ++i)
				indices[i] = i;
			desc->setIndexBuffer(std::move(idxBuf));
		}

        _mb->setIndexCount(idxCount);
        _mb->setIndexBufferOffset(0u);
        _mb->setPrimitiveType(asset::EPT_POINTS);
        _mb->setIndexType(asset::EIT_32BIT);
        printf("pos.size() == %u\n", pos.size());

        m_wgCnt = (idxCount + ELEMENTS_PER_WG - 1u) / ELEMENTS_PER_WG;
        printf("m_wgCnt == %u\n", m_wgCnt);

        auto upStrBuf = m_driver->getDefaultUpStreamingBuffer();
        uint32_t offset = video::StreamingTransientDataBufferMT<>::invalid_address;
        uint32_t size = pos.size() * sizeof(v);
        m_posBuf = core::smart_refctd_ptr<video::IGPUBuffer>(m_driver->createDeviceLocalGPUBufferOnDedMem(size),dont_grab);
        while (offset == video::StreamingTransientDataBufferMT<>::invalid_address)
        {
            uint32_t alignment = 4u;
            const void* data = pos.data();
            upStrBuf->multi_place(std::chrono::seconds(1u), 1u, &data, &offset, &size, &alignment);
        }
        if (upStrBuf->needsManualFlushOrInvalidate())
            m_driver->flushMappedMemoryRanges({ {upStrBuf->getBuffer()->getBoundMemory(),offset,size} });
        m_driver->copyBuffer(upStrBuf->getBuffer(), m_posBuf.get(), offset, 0, size);
        upStrBuf->multi_free(1u, &offset, &size, m_driver->placeFence());

        m_keyBuf1 = core::smart_refctd_ptr<video::IGPUBuffer>(m_driver->createDeviceLocalGPUBufferOnDedMem(idxCount * sizeof(GLuint)), dont_grab);
        m_keyBuf2 = core::smart_refctd_ptr<video::IGPUBuffer>(m_driver->createDeviceLocalGPUBufferOnDedMem(idxCount * sizeof(GLuint)), dont_grab);
        m_sumsBuf = core::smart_refctd_ptr<video::IGPUBuffer>(m_driver->createDeviceLocalGPUBufferOnDedMem(m_wgCnt * 2 * sizeof(GLuint)), dont_grab);
        m_histogramBuf = core::smart_refctd_ptr<video::IGPUBuffer>(m_driver->createDeviceLocalGPUBufferOnDedMem(2 * sizeof(GLuint)), dont_grab);
        m_psumBuf = core::smart_refctd_ptr<video::IGPUBuffer>(m_driver->createDeviceLocalGPUBufferOnDedMem(idxCount * 2 * sizeof(GLuint)), dont_grab);
        m_ubo = core::smart_refctd_ptr<video::IGPUBuffer>(m_driver->createDeviceLocalGPUBufferOnDedMem(s_uboSize), dont_grab);

        m_genKeysCs = createComputeShader(CS_GEN_KEYS_SRC);
        m_histogramCs = createComputeShaderFromFile("../shaders/histogram.comp");
        m_presumCs = createComputeShaderFromFile("../shaders/xpresum.comp");
        m_permuteCs = createComputeShaderFromFile("../shaders/permute.comp");
    }

    void run(const core::vector3df& _camPos) override
    {
        GLint prevProgram;
        glGetIntegerv(GL_CURRENT_PROGRAM, &prevProgram);

        {//bind ubo
            auto auxCtx = const_cast<video::COpenGLDriver::SAuxContext*>(static_cast<video::COpenGLDriver*>(m_driver)->getThreadContext());
            auto glbuf = static_cast<const video::COpenGLBuffer*>(m_ubo.get());
            const ptrdiff_t off = 0, sz = 16;
            auxCtx->setActiveUBO(0, 1, &glbuf, &off, &sz);
        }
        bindSSBuffers();
        updateUbo(0, 12, _camPos, 0u);

        gl::extGlUseProgram(m_genKeysCs);

        gl::extGlDispatchCompute(m_wgCnt, 1u, 1u);
        gl::extGlMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

        {//bind ubo
            auto auxCtx = const_cast<video::COpenGLDriver::SAuxContext*>(static_cast<video::COpenGLDriver*>(m_driver)->getThreadContext());
            auto glbuf = static_cast<const video::COpenGLBuffer*>(m_ubo.get());
            const ptrdiff_t off = 16, sz = m_ubo->getSize()-16;
            auxCtx->setActiveUBO(0, 1, &glbuf, &off, &sz);
        }

        const GLuint histogram[] {0u, 0u};
        uint8_t sumsIn[1u<<13]; // enough for up to 1024 work groups, i.e. 524288 indices
        uint8_t sumsOut[1u<<13];
        for (GLuint nbit = 0u; nbit < 31u; ++nbit)
        {
            {
				auto auxCtx = const_cast<video::COpenGLDriver::SAuxContext*>(static_cast<video::COpenGLDriver*>(m_driver)->getThreadContext());
				const video::COpenGLBuffer* bufs[4] = {	static_cast<const video::COpenGLBuffer*>(m_keyBuf1.get()),
														static_cast<const video::COpenGLBuffer*>(m_idxBuf.get()),
														static_cast<const video::COpenGLBuffer*>(m_keyBuf2.get()),
														static_cast<const video::COpenGLBuffer*>(m_idxBuf2.get()) };
				const ptrdiff_t off[4]{ 0, 0, 0, 0 };
				const ptrdiff_t s[4]{ m_keyBuf1->getSize(), m_idxBuf->getSize(), m_keyBuf2->getSize(), m_idxBuf2->getSize() };
				auxCtx->setActiveSSBO(E_IN_KEYS_BND, 4u, bufs, off, s);
            }
            updateUbo(16, 4, _camPos, nbit);

            // zero histogram
            gl::extGlNamedBufferSubData(static_cast<const video::COpenGLBuffer*>(m_histogramBuf.get())->getOpenGLName(), 0, sizeof(histogram), histogram);

            gl::extGlUseProgram(m_histogramCs);
            gl::extGlDispatchCompute(m_wgCnt, 1u, 1u);
            gl::extGlMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

            gl::extGlUseProgram(m_presumCs);
            gl::extGlDispatchCompute(m_wgCnt, 1u, 1u);
            gl::extGlMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
            // some other barrier needed here?
            gl::extGlGetNamedBufferSubData(static_cast<const video::COpenGLBuffer*>(m_sumsBuf.get())->getOpenGLName(), 0, 2*m_wgCnt*sizeof(GLuint), sumsIn);
            xpsum(sumsIn, sumsOut, m_wgCnt);
            gl::extGlNamedBufferSubData(static_cast<const video::COpenGLBuffer*>(m_sumsBuf.get())->getOpenGLName(), 0, 2*m_wgCnt*sizeof(GLuint), sumsOut);

            gl::extGlUseProgram(m_permuteCs);
            gl::extGlDispatchCompute(m_wgCnt, 1u, 1u);
            gl::extGlMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

            std::swap(m_keyBuf1, m_keyBuf2);
            std::swap(m_idxBuf, m_idxBuf2);
        }
        std::swap(m_keyBuf1, m_keyBuf2);
        std::swap(m_idxBuf, m_idxBuf2);
        // copy result to actual index buffer
        gl::extGlCopyNamedBufferSubData(static_cast<const video::COpenGLBuffer*>(m_idxBuf2.get())->getOpenGLName(), static_cast<const video::COpenGLBuffer*>(m_idxBuf.get())->getOpenGLName(), 0, 0, m_idxBuf->getSize());

        gl::extGlMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);

        // rebind previous program
        gl::extGlUseProgram(prevProgram);
    }

    void setIndexBuffer(video::IGPUMeshBuffer* _mb) override
    {
        ISorter::setIndexBuffer(_mb);
        if (m_idxBuf)
            m_idxBuf2 = core::smart_refctd_ptr<video::IGPUBuffer>(m_driver->createDeviceLocalGPUBufferOnDedMem(m_idxBuf->getSize()));
    }

private:
    void updateUbo(ptrdiff_t _offset, ptrdiff_t _size, const core::vector3df& _camPos, GLuint _nbit)
    {
        uint32_t m[5];
        memcpy(m, &_camPos.X, 12);
        m[4] = _nbit;

        auto upStrBuf = m_driver->getDefaultUpStreamingBuffer();
        uint32_t offset = video::StreamingTransientDataBufferMT<>::invalid_address;
        uint32_t size = _size;
        m_posBuf = core::smart_refctd_ptr<video::IGPUBuffer>(m_driver->createDeviceLocalGPUBufferOnDedMem(size),core::dont_grab);
        while (offset == video::StreamingTransientDataBufferMT<>::invalid_address)
        {
            uint32_t alignment = 4u;
            const void* data = reinterpret_cast<uint8_t*>(m) + _offset;
            upStrBuf->multi_place(std::chrono::seconds(1u), 1u, &data, &offset, &size, &alignment);
        }
        if (upStrBuf->needsManualFlushOrInvalidate())
            m_driver->flushMappedMemoryRanges({ {upStrBuf->getBuffer()->getBoundMemory(),offset,size} });
        m_driver->copyBuffer(upStrBuf->getBuffer(), m_posBuf.get(), offset, _offset, size);
        upStrBuf->multi_free(1u, &offset, &size, m_driver->placeFence());
    }

    void bindSSBuffers() const
    {
        auto auxCtx = const_cast<video::COpenGLDriver::SAuxContext*>(static_cast<video::COpenGLDriver*>(m_driver)->getThreadContext());

        const video::COpenGLBuffer* bufs[8]{
            static_cast<const video::COpenGLBuffer*>(m_keyBuf1.get()),
            static_cast<const video::COpenGLBuffer*>(m_idxBuf.get()),
            static_cast<const video::COpenGLBuffer*>(m_keyBuf2.get()),
            static_cast<const video::COpenGLBuffer*>(m_idxBuf2.get()),
            static_cast<const video::COpenGLBuffer*>(m_psumBuf.get()),
            static_cast<const video::COpenGLBuffer*>(m_histogramBuf.get()),
            static_cast<const video::COpenGLBuffer*>(m_sumsBuf.get()),
            static_cast<const video::COpenGLBuffer*>(m_posBuf.get())
        };
        ptrdiff_t offsets[8]{ 0, 0, 0, 0, 0, 0, 0, 0 };
        ptrdiff_t sizes[8];
        for (size_t i = 0u; i < 8u; ++i)
            sizes[i] = bufs[i]->getSize();

        auxCtx->setActiveSSBO(0u, 2u, bufs, offsets, sizes);
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
    core::smart_refctd_ptr<video::IGPUBuffer> m_posBuf, m_keyBuf1, m_keyBuf2, m_idxBuf2, m_histogramBuf, m_psumBuf, m_sumsBuf;
    core::smart_refctd_ptr<video::IGPUBuffer> m_ubo;
    GLuint m_wgCnt;

    constexpr static size_t s_uboSize = 20u;
};

class ProgressiveSorter : public ISorter
{
protected:
    ~ProgressiveSorter() {}

public:
    ProgressiveSorter(video::IVideoDriver* _vd) :
        ISorter(_vd),
        m_cs{createComputeShaderFromFile("../shaders/prog.comp")},
        m_posBuf(),
        m_ubo(),
        m_startIdx{0u},
        m_wgCount{}
    {}

	void init(asset::ICPUMeshBuffer* _mb) override
	{
		asset::IMeshDataFormatDesc<asset::ICPUBuffer>* desc = _mb->getMeshDataAndFormat();
		std::vector<core::vectorSIMDf> pos;
		vectorSIMDf v;
		size_t ix{};
		while (_mb->getAttribute(v, _mb->getPositionAttributeIx(), ix++))
			pos.push_back(v);

		auto upStrBuf = m_driver->getDefaultUpStreamingBuffer();
		uint32_t offset = video::StreamingTransientDataBufferMT<>::invalid_address;
		uint32_t size = pos.size() * sizeof(v);
		m_posBuf = core::smart_refctd_ptr<video::IGPUBuffer>(m_driver->createDeviceLocalGPUBufferOnDedMem(size), core::dont_grab);
		while (offset == video::StreamingTransientDataBufferMT<>::invalid_address)
		{
			uint32_t alignment = 4u;
			const void* data = pos.data();
			upStrBuf->multi_place(std::chrono::seconds(1u), 1u, &data, &offset, &size, &alignment);
		}
		if (upStrBuf->needsManualFlushOrInvalidate())
			m_driver->flushMappedMemoryRanges({ {upStrBuf->getBuffer()->getBoundMemory(),offset,size} });
		m_driver->copyBuffer(upStrBuf->getBuffer(), m_posBuf.get(), offset, 0, size);
		upStrBuf->multi_free(1u, &offset, &size, m_driver->placeFence());

		m_ubo = core::smart_refctd_ptr<video::IGPUBuffer>(m_driver->createDeviceLocalGPUBufferOnDedMem(s_uboSize), core::dont_grab);

		{
			auto idxBuf = core::make_smart_refctd_ptr<asset::ICPUBuffer>(sizeof(uint32_t)*pos.size());
			uint32_t* indices = (uint32_t*)idxBuf->getPointer();
			for (uint32_t i = 0u; i < pos.size(); ++i)
				indices[i] = i;

			desc->setIndexBuffer(std::move(idxBuf));
		}
        _mb->setIndexCount(pos.size());
        _mb->setIndexBufferOffset(0u);
        _mb->setPrimitiveType(asset::EPT_POINTS);
        _mb->setIndexType(asset::EIT_32BIT);
        printf("pos.size() == %u\n", pos.size());

        m_wgCount = (GLuint)std::ceil(double(pos.size()) / 256.);
        printf("wgCount == %u\n", m_wgCount);
    }

    void run(const core::vector3df& _camPos) override
    {
        const uint32_t offset[2]{ 0u, 512u };
        {//bind ubo
            auto auxCtx = const_cast<video::COpenGLDriver::SAuxContext*>(static_cast<video::COpenGLDriver*>(m_driver)->getThreadContext());
            auto glbuf = static_cast<const video::COpenGLBuffer*>(m_ubo.get());
            const ptrdiff_t off = 0, sz = m_ubo->getSize();
            auxCtx->setActiveUBO(0, 1, &glbuf, &off, &sz);
        }
        bindSSBuffers();

        updateUbo(0, s_uboSize, _camPos, offset[m_startIdx], m_posBuf->getSize()/16u);

        GLint prevProgram;
        glGetIntegerv(GL_CURRENT_PROGRAM, &prevProgram);

        using gl = video::COpenGLExtensionHandler;
        gl::extGlUseProgram(m_cs);
        gl::extGlDispatchCompute((m_posBuf->getSize()/16 - offset[m_startIdx] + 1023u) / 1024, 1u, 1u);
        gl::extGlUseProgram(prevProgram);

        gl::extGlMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);

        m_startIdx = !m_startIdx;
    }

private:
    void updateUbo(ptrdiff_t _offset, ptrdiff_t _size, const core::vector3df& _camPos, GLuint _off, GLuint _sz)
    {
        uint32_t m[6];
        memcpy(m, &_camPos.X, 12);
        m[4] = _off;
        m[5] = _sz;

        auto upStrBuf = m_driver->getDefaultUpStreamingBuffer();
        uint32_t offset = video::StreamingTransientDataBufferMT<>::invalid_address;
        uint32_t size = _size;
        m_posBuf = core::smart_refctd_ptr<video::IGPUBuffer>(m_driver->createDeviceLocalGPUBufferOnDedMem(size),core::dont_grab);
        while (offset == video::StreamingTransientDataBufferMT<>::invalid_address)
        {
            uint32_t alignment = 4u;
            const void* data = reinterpret_cast<uint8_t*>(m)+_offset;
            upStrBuf->multi_place(std::chrono::seconds(1u), 1u, &data, &offset, &size, &alignment);
        }
        if (upStrBuf->needsManualFlushOrInvalidate())
            m_driver->flushMappedMemoryRanges({ {upStrBuf->getBuffer()->getBoundMemory(),offset,size} });
        m_driver->copyBuffer(upStrBuf->getBuffer(), m_posBuf.get(), offset, _offset, size);
        upStrBuf->multi_free(1u, &offset, &size, m_driver->placeFence());
    }

    void bindSSBuffers() const
    {
        auto auxCtx = const_cast<video::COpenGLDriver::SAuxContext*>(static_cast<video::COpenGLDriver*>(m_driver)->getThreadContext());

        const video::COpenGLBuffer* bufs[2]{
            static_cast<const video::COpenGLBuffer*>(m_idxBuf.get()),
            static_cast<const video::COpenGLBuffer*>(m_posBuf.get())
        };
        ptrdiff_t offsets[2]{ 0, 0 };
        ptrdiff_t sizes[2]{ m_idxBuf->getSize(), m_posBuf->getSize() };

        auxCtx->setActiveSSBO(0u, 2u, bufs, offsets, sizes);
    }

private:
    GLuint m_cs;
    core::smart_refctd_ptr<video::IGPUBuffer> m_posBuf;
    core::smart_refctd_ptr<video::IGPUBuffer> m_ubo;
    core::smart_refctd_ptr<video::IGPUBuffer> m_mappedBuf;
    mutable GLuint m_startIdx;
    GLuint m_wgCount;

    //video::IDriverFence* m_fences[4];
    constexpr static size_t s_uboSize = 24u;
};


class BitonicSorter : public ISorter
{
protected:
    ~BitonicSorter()
    {
    }

public:
    BitonicSorter(video::IVideoDriver* _vd) :
        ISorter(_vd),
        m_sMergeCs{ createComputeShaderFromFile("../shaders/s_merge.comp") },
        m_gMergeCs{ createComputeShaderFromFile("../shaders/g_merge.comp") },
        m_sSortCs{ 0u },
        m_posBuf{ nullptr },
        m_ubo{ nullptr },
        m_wgCount{}
    {}
    void init(asset::ICPUMeshBuffer* _mb) override
    {
        asset::IMeshDataFormatDesc<asset::ICPUBuffer>* desc = _mb->getMeshDataAndFormat();
        std::vector<core::vectorSIMDf> pos;
        vectorSIMDf v;
        size_t ix{};
        while (_mb->getAttribute(v, _mb->getPositionAttributeIx(), ix++))
            pos.push_back(v);

        auto upStrBuf = m_driver->getDefaultUpStreamingBuffer();
        uint32_t offset = video::StreamingTransientDataBufferMT<>::invalid_address;
        uint32_t size = pos.size() * sizeof(v);
        m_posBuf = core::smart_refctd_ptr<video::IGPUBuffer>(m_driver->createDeviceLocalGPUBufferOnDedMem(size),core::dont_grab);
        while (offset == video::StreamingTransientDataBufferMT<>::invalid_address)
        {
            uint32_t alignment = 4u;
            const void* data = pos.data();
            upStrBuf->multi_place(std::chrono::seconds(1u), 1u, &data, &offset, &size, &alignment);
        }
        if (upStrBuf->needsManualFlushOrInvalidate())
            m_driver->flushMappedMemoryRanges({ {upStrBuf->getBuffer()->getBoundMemory(),offset,size} });
        m_driver->copyBuffer(upStrBuf->getBuffer(), m_posBuf.get(), offset, 0, size);
        upStrBuf->multi_free(1u, &offset, &size, m_driver->placeFence());

        m_ubo = core::smart_refctd_ptr<video::IGPUBuffer>(m_driver->createDeviceLocalGPUBufferOnDedMem(s_uboSize),core::dont_grab);

        const size_t idxCount = 1u << ((size_t)std::ceil(std::log2((double)pos.size())));
		{
			auto idxBuf = core::make_smart_refctd_ptr<asset::ICPUBuffer>(sizeof(uint32_t)*idxCount);
			uint32_t* indices = (uint32_t*)idxBuf->getPointer();
			memset(indices, 0, (idxCount - pos.size()) * 4);
			for (uint32_t i = idxCount - pos.size(); i < idxCount; ++i)
				indices[i] = i - (idxCount - pos.size());

			desc->setIndexBuffer(std::move(idxBuf));
		}
        _mb->setIndexCount(idxCount);
        _mb->setIndexBufferOffset(0u);
        _mb->setPrimitiveType(asset::EPT_POINTS);
        _mb->setIndexType(asset::EIT_32BIT);
        printf("pos.size() == %u\n", pos.size());

        m_wgCount = idxCount / 256u;

        if (m_wgCount == 2u) // element count == 512
            m_sSortCs = makeSortCs(512u);
        else if (m_wgCount > 2u)
            m_sSortCs = makeSortCs(1024u);

        assert(m_sSortCs);

        printf("wgCount == %u\n", m_wgCount);
    }

    void run(const core::vector3df& _camPos) override
    {
        {//bind ubo
        auto auxCtx = const_cast<video::COpenGLDriver::SAuxContext*>(static_cast<video::COpenGLDriver*>(m_driver)->getThreadContext());
        auto glbuf = static_cast<const video::COpenGLBuffer*>(m_ubo.get());
        const ptrdiff_t off = 0, sz = m_ubo->getSize();
        auxCtx->setActiveUBO(0, 1, &glbuf, &off, &sz);
        }
        bindSSBuffers();

        GLint prevProgram;
        glGetIntegerv(GL_CURRENT_PROGRAM, &prevProgram);

        const size_t valCnt = m_wgCount * 256u;
        const size_t sSortWgSize = m_wgCount == 2u ? 512u : 1024u;

        using gl = video::COpenGLExtensionHandler;
        gl::extGlUseProgram(m_sSortCs);
        gl::pGlDispatchCompute(valCnt/sSortWgSize, 1, 1);
        gl::pGlMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

        bool firstItr = true;
        for (GLuint sz = 2u*sSortWgSize; sz <= valCnt; sz <<= 1)
        {
            for (GLuint str = sz>>1; str > 0u; str >>= 1)
            {
                updateUbo(firstItr ? 0 : 16, firstItr ? s_uboSize : 8, _camPos, sz, str);
                firstItr = false;
                if (str > 1024)
                {
                    gl::extGlUseProgram(m_gMergeCs);
                    gl::pGlDispatchCompute(m_wgCount, 1, 1);
                    gl::pGlMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
                }
                else
                {
                    assert(str == 1024);
                    gl::extGlUseProgram(m_sMergeCs);
                    gl::pGlDispatchCompute(valCnt/2048, 1, 1);
                    gl::pGlMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
                    break;
                }
            }
        }

        gl::extGlMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
        gl::extGlUseProgram(prevProgram);
    }

private:
    void updateUbo(ptrdiff_t _offset, ptrdiff_t _size, const core::vector3df& _camPos, GLuint _sz, GLuint _str)
    {
        uint32_t m[6];
        memcpy(m, &_camPos.X, 12);
        m[4] = _sz;
        m[5] = _str;

        auto upStrBuf = m_driver->getDefaultUpStreamingBuffer();
        uint32_t offset = video::StreamingTransientDataBufferMT<>::invalid_address;
        uint32_t size = _size;
        m_posBuf = core::smart_refctd_ptr<video::IGPUBuffer>(m_driver->createDeviceLocalGPUBufferOnDedMem(size),core::dont_grab);
        while (offset == video::StreamingTransientDataBufferMT<>::invalid_address)
        {
            uint32_t alignment = 4u;
            const void* data = reinterpret_cast<uint8_t*>(m) + _offset;
            upStrBuf->multi_place(std::chrono::seconds(1u), 1u, &data, &offset, &size, &alignment);
        }
        if (upStrBuf->needsManualFlushOrInvalidate())
            m_driver->flushMappedMemoryRanges({ {upStrBuf->getBuffer()->getBoundMemory(),offset,size} });
        m_driver->copyBuffer(upStrBuf->getBuffer(), m_posBuf.get(), offset, _offset, size);
        upStrBuf->multi_free(1u, &offset, &size, m_driver->placeFence());
    }

    GLuint makeSortCs(size_t _wgSize)
    {
        void* source = nullptr;
        const size_t sz = loadFileContentsAsStr("../shaders/s_sort.comp", source);
        void* mem = malloc(sz+100);
        snprintf((char*)mem, sz + 100, (const char*)source, _wgSize);

        const GLuint s = createComputeShader((const char*)mem);
        free(source);
        free(mem);

        return s;
    }

    void bindSSBuffers() const
    {
        auto auxCtx = const_cast<video::COpenGLDriver::SAuxContext*>(static_cast<video::COpenGLDriver*>(m_driver)->getThreadContext());

        const video::COpenGLBuffer* bufs[2]{
            static_cast<const video::COpenGLBuffer*>(m_idxBuf.get()),
            static_cast<const video::COpenGLBuffer*>(m_posBuf.get())
        };
        ptrdiff_t offsets[2]{ 0, 0 };
        ptrdiff_t sizes[2]{ m_idxBuf->getSize(), m_posBuf->getSize() };

        auxCtx->setActiveSSBO(0u, 2u, bufs, offsets, sizes);
    }

private:
    GLuint m_sMergeCs, m_gMergeCs, m_sSortCs;
    core::smart_refctd_ptr<video::IGPUBuffer> m_posBuf;
    core::smart_refctd_ptr<video::IGPUBuffer> m_ubo;
    GLuint m_wgCount;

    constexpr static size_t s_uboSize = 24u;
};

int main()
{
    // create device with full flexibility over creation parameters
    // you can add more parameters if desired, check nbl::SIrrlichtCreationParameters
    nbl::SIrrlichtCreationParameters params;
    params.Bits = 24; //may have to set to 32bit for some platforms
    params.ZBufferBits = 24; //we'd like 32bit here
    params.DriverType = video::EDT_OPENGL; //! Only Well functioning driver, software renderer left for sake of 2D image drawing
    params.WindowSize = dimension2d<uint32_t>(1280, 720);
    params.Fullscreen = false;
    params.Vsync = true; //! If supported by target platform
    params.Doublebuffer = true;
    params.Stencilbuffer = false; //! This will not even be a choice soon
    auto device = createDeviceEx(params);

    if (!device)
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
    scene::ICameraSceneNode* camera =
        smgr->addCameraSceneNodeFPS(0, 100.0f, 0.01f);
    camera->setPosition(core::vector3df(-4, 0, 0));
    camera->setTarget(core::vector3df(0, 0, 0));
    camera->setNearValue(0.01f);
    camera->setFarValue(100.0f);
    smgr->setActiveCamera(camera);


    device->getCursorControl()->setVisible(false);
    QToQuitEventReceiver receiver;
    device->setEventReceiver(&receiver);


    asset::IAssetLoader::SAssetLoadParams lparams;
    auto cpumesh = core::smart_refctd_ptr_static_cast<asset::ICPUMesh>(*device->getAssetManager()->getAsset("../../media/cow.obj", lparams).getContents().first);
    ISorter* sorter =
        //new RadixSorter(driver);
        new ProgressiveSorter(driver);
        //new BitonicSorter(driver);
    sorter->init(cpumesh->getMeshBuffer(0));
    auto gpumesh = driver->getGPUObjectsFromAssets(&cpumesh.get(), (&cpumesh.get())+1)->front();
    sorter->setIndexBuffer(gpumesh->getMeshBuffer(0));
	gpumesh->getMeshBuffer(0)->getMaterial().MaterialType = newMaterialType;
    printf("IDX_TYPE %d\n", gpumesh->getMeshBuffer(0)->getIndexType());
	smgr->addMeshSceneNode(std::move(gpumesh), 0, -1, core::vector3df(), core::vector3df(), core::vector3df(4.f));
 

    uint64_t lastFPSTime = 0;

    while (device->run() && receiver.keepOpen())
    {
        driver->beginScene(true, true, video::SColor(255, 255, 255, 255));

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
	{
		core::rect<uint32_t> sourceRect(0, 0, params.WindowSize.Width, params.WindowSize.Height);
		ext::ScreenShot::dirtyCPUStallingScreenshot(device, "screenshot.png", sourceRect, asset::EF_R8G8B8_SRGB);
	}

    return 0;
}