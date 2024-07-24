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

//! Pipelines
template<>
struct asset_traits<asset::ICPUShader>
{
	// the asset type
	using asset_t = asset::ICPUShader;
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
	// the video type
	using video_t = IGPUPipelineLayout;
	// lookup type
	using lookup_t = const video_t*;
};
/*
/*
template<>
struct asset_traits<asset::ICPUDescriptorSetLayout> { using GPUObjectType = IGPUDescriptorSetLayout; };

template<>
struct asset_traits<asset::ICPUComputePipeline> { using GPUObjectType = IGPUComputePipeline; };
*/

template<>
struct asset_traits<asset::ICPUBuffer>
{
	using asset_t = asset::ICPUBuffer;
	using video_t = asset::SBufferRange<IGPUBuffer>;
	// lookup type
	using lookup_t = video_t;
};

template<>
struct asset_traits<asset::ICPUBufferView>
{
	// the asset type
	using asset_t = asset::ICPUBufferView;
	// the video type
	using video_t = IGPUBufferView;
	// lookup type
	using lookup_t = const video_t*;
};


/*
template<>
struct asset_traits<asset::ICPUImage> { using GPUObjectType = IGPUImage; };

template<>
struct asset_traits<asset::ICPUImageView> { using GPUObjectType = IGPUImageView; };
*/

template<>
struct asset_traits<asset::ICPUSampler>
{
	// the asset type
	using asset_t = asset::ICPUSampler;
	// the video type
	using video_t = IGPUSampler;
	// lookup type
	using lookup_t = const video_t*;
};

/*
template<>
struct asset_traits<asset::ICPUShader> { using GPUObjectType = IGPUShader; };

template<>
struct asset_traits<asset::ICPUDescriptorSet> { using GPUObjectType = IGPUDescriptorSet; };

template<>
struct asset_traits<asset::ICPUAccelerationStructure> { using GPUObjectType = IGPUAccelerationStructure; };
*/


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