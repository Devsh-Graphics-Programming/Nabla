// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_VIDEO_ASSET_TRAITS_H_INCLUDED_
#define _NBL_VIDEO_ASSET_TRAITS_H_INCLUDED_

#include "nbl/asset/ICPUSkeleton.h"
#include "nbl/asset/ICPUShader.h"
#include "nbl/video/IGPUShader.h"
#include "nbl/asset/ICPUBufferView.h"
#include "nbl/video/IGPUBufferView.h"
#include "nbl/asset/ICPUDescriptorSet.h"
#include "nbl/video/IGPUDescriptorSet.h"
#include "nbl/asset/ICPUDescriptorSetLayout.h"
#include "nbl/video/IGPUDescriptorSetLayout.h"
#include "nbl/asset/ICPUPipelineLayout.h"
#include "nbl/video/IGPUPipelineLayout.h"
#include "nbl/asset/ICPUComputePipeline.h"
#include "nbl/video/IGPUComputePipeline.h"
#include "nbl/asset/ICPUSampler.h"
#include "nbl/video/IGPUSampler.h"
#include "nbl/asset/ICPUImageView.h"
#include "nbl/video/IGPUImageView.h"
#include "nbl/asset/ICPUAnimationLibrary.h"
#include "nbl/video/IGPUAnimationLibrary.h"
#include "nbl/asset/ICPUAccelerationStructure.h"
#include "nbl/video/IGPUAccelerationStructure.h"


namespace nbl
{
namespace video
{

// TODO: don't we already have a class for this in asset::IBuffer? No, its a simple struct, not a refcounted object
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
struct asset_traits<asset::ICPUDescriptorSet> { using GPUObjectType = video::IGPUDescriptorSet; };
template<>
struct asset_traits<asset::ICPUDescriptorSetLayout> { using GPUObjectType = video::IGPUDescriptorSetLayout; };
template<>
struct asset_traits<asset::ICPUPipelineLayout> { using GPUObjectType = video::IGPUPipelineLayout; };
// TODO: figure this out
//template<>
//struct asset_traits<asset::ICPUSkeleton> { using GPUObjectType = video::IGPUSkeleton; };
template<>
struct asset_traits<asset::ICPUComputePipeline> { using GPUObjectType = video::IGPUComputePipeline; };
template<>
struct asset_traits<asset::ICPUAnimationLibrary> { using GPUObjectType = video::IGPUAnimationLibrary; };
template<>
struct asset_traits<asset::ICPUAccelerationStructure> { using GPUObjectType = video::IGPUAccelerationStructure; };


template<typename AssetType>
using created_gpu_object_array = core::smart_refctd_dynamic_array<core::smart_refctd_ptr<typename video::asset_traits<AssetType>::GPUObjectType> >;

}
}

#endif