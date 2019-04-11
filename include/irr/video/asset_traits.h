#ifndef __IRR_ASSET_TRAITS_H_INCLUDED__
#define __IRR_ASSET_TRAITS_H_INCLUDED__

#include "IGPUBuffer.h"
#include "ITexture.h"
#include "irr/video/IGPUMeshBuffer.h"
#include "irr/asset/ICPUMeshBuffer.h"
#include "irr/asset/ICPUMesh.h"
#include "irr/video/IGPUMesh.h"

namespace irr { namespace video
{

template<typename BuffT>
class SOffsetBufferPair : public core::IReferenceCounted
{
protected:
    virtual ~SOffsetBufferPair()
    {
        if (m_buffer)
            m_buffer->drop();
    }

public:
    SOffsetBufferPair(uint64_t _offset = 0ull, BuffT* _buffer = nullptr) : m_offset{_offset}, m_buffer{_buffer}
    {
        if (m_buffer)
            m_buffer->grab();
    }

    void setOffset(uint64_t _offset) { m_offset = _offset; }
    void setBuffer(BuffT* _buffer)
    {
        if (_buffer)
            _buffer->grab();
        if (m_buffer)
            m_buffer->drop();
        m_buffer = _buffer;
    }

    uint64_t getOffset() const { return m_offset; }
    BuffT* getBuffer() const { return m_buffer; }

private:
    uint64_t m_offset;
    BuffT* m_buffer;
};

template<typename AssetType>
struct asset_traits;

template<>
struct asset_traits<asset::ICPUBuffer> { using GPUObjectType = SOffsetBufferPair<video::IGPUBuffer>; };
template<>
struct asset_traits<asset::ICPUMeshBuffer> { using GPUObjectType = video::IGPUMeshBuffer; };
template<>
struct asset_traits<asset::ICPUMesh> { using GPUObjectType = video::IGPUMesh; };
template<>
struct asset_traits<asset::ICPUTexture> { using GPUObjectType = video::ITexture; };

}}

#endif //__IRR_ASSET_TRAITS_H_INCLUDED__