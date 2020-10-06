// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_VIDEO_ASSET_TRAITS_H_INCLUDED__
#define __NBL_VIDEO_ASSET_TRAITS_H_INCLUDED__

#include "irr/asset/ICPUMesh.h"
#include "irr/video/IGPUMesh.h"
#include "irr/asset/ICPUShader.h"
#include "irr/video/IGPUShader.h"
#include "irr/asset/ICPUSpecializedShader.h"
#include "irr/video/IGPUSpecializedShader.h"
#include "irr/asset/ICPUBufferView.h"
#include "irr/video/IGPUBufferView.h"
#include "irr/asset/ICPUDescriptorSet.h"
#include "irr/video/IGPUDescriptorSet.h"
#include "irr/asset/ICPUDescriptorSetLayout.h"
#include "irr/video/IGPUDescriptorSetLayout.h"
#include "irr/asset/ICPUPipelineLayout.h"
#include "irr/video/IGPUPipelineLayout.h"
#include "irr/asset/ICPURenderpassIndependentPipeline.h"
#include "irr/video/IGPURenderpassIndependentPipeline.h"
#include "irr/asset/ICPUComputePipeline.h"
#include "irr/video/IGPUComputePipeline.h"
#include "irr/asset/ICPUSampler.h"
#include "irr/video/IGPUSampler.h"
#include "irr/asset/ICPUImageView.h"
#include "irr/video/IGPUImageView.h"


namespace irr
{
namespace video
{

// TODO: don't we already have a class for this in asset::IBuffer?
template<typename BuffT>
class IOffsetBufferPair : public core::IReferenceCounted
{
protected:
	virtual ~IOffsetBufferPair() {}

public:
    IOffsetBufferPair(uint64_t _offset = 0ull, core::smart_refctd_ptr<BuffT>&& _buffer = nullptr) : m_offset{_offset}, m_buffer(_buffer) {}

    inline void setOffset(uint64_t _offset) { m_offset = _offset; }
    inline void setBuffer(core::smart_refctd_ptr<BuffT>&& _buffer) { m_buffer = _buffer; }

    uint64_t getOffset() const { return m_offset; }
    BuffT* getBuffer() const { return m_buffer.get(); }

private:
    uint64_t m_offset;
    core::smart_refctd_ptr<BuffT> m_buffer;
};
using IGPUOffsetBufferPair = IOffsetBufferPair<video::IGPUBuffer>;

template<typename AssetType>
struct asset_traits;

template<>
struct asset_traits<asset::ICPUBuffer> { using GPUObjectType = IGPUOffsetBufferPair; };
template<>
struct asset_traits<asset::ICPUBufferView> { using GPUObjectType = video::IGPUBufferView; };
template<>
struct asset_traits<asset::ICPUImage> { using GPUObjectType = video::IGPUImage; };
template<>
struct asset_traits<asset::ICPUImageView> { using GPUObjectType = video::IGPUImageView; };
template<>
struct asset_traits<asset::ICPUSampler> { using GPUObjectType = video::IGPUSampler; };
template<>
struct asset_traits<asset::ICPUShader> { using GPUObjectType = video::IGPUShader; };
template<>
struct asset_traits<asset::ICPUSpecializedShader> { using GPUObjectType = video::IGPUSpecializedShader; };
template<>
struct asset_traits<asset::ICPUDescriptorSet> { using GPUObjectType = video::IGPUDescriptorSet; };
template<>
struct asset_traits<asset::ICPUDescriptorSetLayout> { using GPUObjectType = video::IGPUDescriptorSetLayout; };
template<>
struct asset_traits<asset::ICPUPipelineLayout> { using GPUObjectType = video::IGPUPipelineLayout; };
template<>
struct asset_traits<asset::ICPURenderpassIndependentPipeline> { using GPUObjectType = video::IGPURenderpassIndependentPipeline; };
template<>
struct asset_traits<asset::ICPUMeshBuffer> { using GPUObjectType = video::IGPUMeshBuffer; };
template<>
struct asset_traits<asset::ICPUComputePipeline> { using GPUObjectType = video::IGPUComputePipeline; };
template<>
struct asset_traits<asset::ICPUMesh> { using GPUObjectType = video::IGPUMesh; };


template<typename AssetType>
using created_gpu_object_array = core::smart_refctd_dynamic_array<core::smart_refctd_ptr<typename video::asset_traits<AssetType>::GPUObjectType> >;

}
}

#endif //__IRR_ASSET_TRAITS_H_INCLUDED__