// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_VIDEO_ASSET_TRAITS_H_INCLUDED_
#define _NBL_VIDEO_ASSET_TRAITS_H_INCLUDED_

#include "nbl/asset/ICPUShader.h"
#include "nbl/video/IGPUShader.h"
#include "nbl/asset/ICPUBufferView.h"
#include "nbl/video/IGPUBufferView.h"
#include "nbl/asset/ICPUDescriptorSet.h"
#include "nbl/video/IGPUDescriptorSet.h"
#include "nbl/asset/ICPUComputePipeline.h"
#include "nbl/video/IGPUComputePipeline.h"
#include "nbl/asset/ICPUGraphicsPipeline.h"
#include "nbl/video/IGPUGraphicsPipeline.h"
#include "nbl/asset/ICPUSampler.h"
#include "nbl/video/IGPUSampler.h"
#include "nbl/asset/ICPUImageView.h"
#include "nbl/video/IGPUImageView.h"
#include "nbl/asset/ICPUAccelerationStructure.h"
#include "nbl/video/IGPUAccelerationStructure.h"


namespace nbl::video
{
template<asset::Asset AssetType>
struct asset_traits;


template<>
struct asset_traits<asset::ICPUSampler>
{
	// the asset type
	using asset_t = asset::ICPUSampler;
	// we don't need to descend during DFS into other assets
	constexpr static inline bool HasChildren = false;
	// the video type
	using video_t = IGPUSampler;
	// lookup type
	using lookup_t = const video_t*;
};

template<>
struct asset_traits<asset::ICPUShader>
{
	// the asset type
	using asset_t = asset::ICPUShader;
	// we don't need to descend during DFS into other assets
	constexpr static inline bool HasChildren = false;
	// the video type
	using video_t = IGPUShader;
	// lookup type
	using lookup_t = const video_t*;
};

template<>
struct asset_traits<asset::ICPUDescriptorSetLayout>
{
	// the asset type
	using asset_t = asset::ICPUDescriptorSetLayout;
	// DS layout can have immutable samplers
	constexpr static inline bool HasChildren = true;
	// the video type
	using video_t = IGPUDescriptorSetLayout;
	// lookup type
	using lookup_t = const video_t*;
};

template<>
struct asset_traits<asset::ICPUPipelineLayout>
{
	// the asset type
	using asset_t = asset::ICPUPipelineLayout;
	// Pipeline Layout references Descriptor Set Layouts
	constexpr static inline bool HasChildren = true;
	// the video type
	using video_t = IGPUPipelineLayout;
	// lookup type
	using lookup_t = const video_t*;
};

template<>
struct asset_traits<asset::ICPUPipelineCache>
{
	// the asset type
	using asset_t = asset::ICPUPipelineCache;
	// we don't need to descend during DFS into other assets
	constexpr static inline bool HasChildren = false;
	// the video type
	using video_t = IGPUPipelineCache;
	// lookup type
	using lookup_t = const video_t*;
};

template<>
struct asset_traits<asset::ICPUComputePipeline>
{
	// the asset type
	using asset_t = asset::ICPUComputePipeline;
	// Pipeline Layout references Descriptor Set Layouts
	constexpr static inline bool HasChildren = true;
	// the video type
	using video_t = IGPUComputePipeline;
	// lookup type
	using lookup_t = const video_t*;
};


template<>
struct asset_traits<asset::ICPURenderpass>
{
	// the asset type
	using asset_t = asset::ICPURenderpass;
	// we don't need to descend during DFS into other assets
	constexpr static inline bool HasChildren = false;
	// the video type
	using video_t = IGPURenderpass;
	// lookup type
	using lookup_t = const video_t*;
};

template<>
struct asset_traits<asset::ICPUGraphicsPipeline>
{
	// the asset type
	using asset_t = asset::ICPUGraphicsPipeline;
	// we reference a pipeline layout and a renderpass
	constexpr static inline bool HasChildren = true;
	// the video type
	using video_t = IGPUGraphicsPipeline;
	// lookup type
	using lookup_t = const video_t*;
};


template<>
struct asset_traits<asset::ICPUBuffer>
{
	// the asset type
	using asset_t = asset::ICPUBuffer;
	// we don't need to descend during DFS into other assets
	constexpr static inline bool HasChildren = false;
	// the video type
	using video_t = IGPUBuffer;
	// lookup type
	using lookup_t = const video_t*;
};

template<>
struct asset_traits<asset::ICPUBufferView>
{
	// the asset type
	using asset_t = asset::ICPUBufferView;
	// depends on ICPUBuffer
	constexpr static inline bool HasChildren = true;
	// the video type
	using video_t = IGPUBufferView;
	// lookup type
	using lookup_t = const video_t*;
};


template<>
struct asset_traits<asset::ICPUImage>
{
	// the asset type
	using asset_t = asset::ICPUImage;
	// we don't need to descend during DFS into other assets
	constexpr static inline bool HasChildren = false;
	// the video type
	using video_t = IGPUImage;
	// lookup type
	using lookup_t = const video_t*;
};

template<>
struct asset_traits<asset::ICPUImageView>
{
	// the asset type
	using asset_t = asset::ICPUImageView;
	// depends on ICPUImage
	constexpr static inline bool HasChildren = true;
	// the video type
	using video_t = IGPUImageView;
	// lookup type
	using lookup_t = const video_t*;
};


template<>
struct asset_traits<asset::ICPUBottomLevelAccelerationStructure>
{
	// the asset type
	using asset_t = asset::ICPUBottomLevelAccelerationStructure;
	// we don't need to descend during DFS into other assets
	constexpr static inline bool HasChildren = true;
	// the video type
	using video_t = IGPUImageView;
	// lookup type
	using lookup_t = const video_t*;
};

template<>
struct asset_traits<asset::ICPUTopLevelAccelerationStructure>
{
	// the asset type
	using asset_t = asset::ICPUTopLevelAccelerationStructure;
	// depends on ICPUBottomLevelAccelerationStructure
	constexpr static inline bool HasChildren = true;
	// the video type
	using video_t = IGPUTopLevelAccelerationStructure;
	// lookup type
	using lookup_t = const video_t*;
};


template<>
struct asset_traits<asset::ICPUDescriptorSet>
{
	// the asset type
	using asset_t = asset::ICPUDescriptorSet;
	// depends on a lot of `IDescriptor` ICPU... types
	constexpr static inline bool HasChildren = true;
	// the video type
	using video_t = IGPUDescriptorSet;
	// lookup type
	using lookup_t = const video_t*;
};


/* TODO
template<>
struct asset_traits<asset::ICPUFramebuffer>;
*/

// Every other ICPU type not present in the list here is deprecated


// Slight wrapper to allow copyable smart pointers
template<asset::Asset AssetType>
struct asset_cached_t final
{
	private:
		using this_t = asset_cached_t<AssetType>;
		using video_t = typename asset_traits<AssetType>::video_t;
		constexpr static inline bool RefCtd = core::ReferenceCounted<video_t>;

	public:
		inline asset_cached_t() = default;
		inline asset_cached_t(const this_t& other) : asset_cached_t() {operator=(other);}
		inline asset_cached_t(this_t&&) = default;

		// special wrapping to make smart_refctd_ptr copyable
		inline this_t& operator=(const this_t& rhs)
		{
			if constexpr (RefCtd)
				value = core::smart_refctd_ptr<video_t>(rhs.value.get());
			else
				value = video_t(rhs.value);
			return *this;
		}
		inline this_t& operator=(this_t&&) = default;

		inline bool operator==(const this_t& other) const
		{
			return value==other.value;
		}

		explicit inline operator bool() const
		{
			return bool(get());
		}

		inline const auto& get() const
		{
			if constexpr (RefCtd)
				return value.get();
			else
				return value;
		}

		using type = std::conditional_t<RefCtd,core::smart_refctd_ptr<video_t>,video_t>;
		type value = {};
};
}

namespace std
{
template<nbl::asset::Asset AssetType>
struct hash<nbl::video::asset_cached_t<AssetType>>
{
	inline size_t operator()(const nbl::video::asset_cached_t<AssetType>& entry) const noexcept
	{
		return 0ull;
	}
};
}
#endif