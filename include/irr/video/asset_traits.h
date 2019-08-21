#ifndef __IRR_ASSET_TRAITS_H_INCLUDED__
#define __IRR_ASSET_TRAITS_H_INCLUDED__

#include "irr/video/IGPUMesh.h"

namespace irr
{
namespace video
{

template<typename BuffT>
class SOffsetBufferPair : public core::IReferenceCounted
{
protected:
	virtual ~SOffsetBufferPair() {}

public:
    SOffsetBufferPair(uint64_t _offset = 0ull, core::smart_refctd_ptr<BuffT>&& _buffer = nullptr) : m_offset{_offset}, m_buffer(_buffer) {}

    inline void setOffset(uint64_t _offset) { m_offset = _offset; }
    inline void setBuffer(core::smart_refctd_ptr<BuffT>&& _buffer) { m_buffer = _buffer; }

    uint64_t getOffset() const { return m_offset; }
    BuffT* getBuffer() const { return m_buffer.get(); }

private:
    uint64_t m_offset;
    core::smart_refctd_ptr<BuffT> m_buffer;
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