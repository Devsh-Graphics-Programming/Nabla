// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_ASSET_I_ASSET_H_INCLUDED_
#define _NBL_ASSET_I_ASSET_H_INCLUDED_

#include "nbl/core/decl/smart_refctd_ptr.h"

#include <string>

namespace nbl::asset
{

class IAssetManager;

//! An abstract data type class representing an interface of any kind of cpu-objects with capability of being cached
/**
	An Asset is a CPU object which needs to be loaded for caching it to RAM memory,
	or needs to be filled with Asset-data if it has to be created by hand,
	and then may be converted to GPU object, but it isn't necessary in each case! For instance
	ICPUBuffer which is an Asset opens a nice way to represent a plain byte array, but doesn't need
	to be converted into GPU object!

	@see ICPUBuffer

	Actually an Asset is a class deriving from it that can be anything like cpu-side meshes scenes, texture data and material pipelines. 
	There are different asset types you can find at IAsset::E_TYPE. 
	IAsset doesn't provide direct instantiation (virtual destructor), much like IReferenceCounted.

	Asset type's naming-convention is ICPU_x, where _x is a name of an Asset, eg. ICPUBuffer, ICPUMesh e.t.c

	Every asset must be loaded by a particular class derived from IAssetLoader.
	
	@see IAssetLoader
	@see IAssetManager
	@see IAssetWriter
	@see IReferenceCounted
*/

class IAsset : virtual public core::IReferenceCounted
{
	public:
		/**
			Values of E_TYPE represents an Asset type.
		
			Types are provided, so known is that the type you're casting to is right, eg.
			If there is an Asset represeting ICPUBuffer, after calling a function returning
			a type, it should return you a type associated with it, so
		
			\code{.cpp}
			IAsset* asset;
			if(asset->getType() == ET_BUFFER)
				static_cast<ICPUBuffer*>(asset) // it is safe
			\endcode

			@see IAsset

			ET_BUFFER represents ICPUBuffer
			ET_IMAGE represents ICPUTexture
			ET_SUB_MESH represents
			ET_MESH represents
			ET_SKELETON represents
			ET_ANIMATION_LIBRARY represents
			ET_SHADER represents
			ET_SPECIALIZED_SHADER represents
			ET_MESH_DATA_DESCRIPTOR represents
			ET_GRAPHICS_PIPELINE represents
			ET_SCENE represents

			Pay attention that an Asset type represents one single bit, so there is a limit to 64 bits.

		*/
		enum E_TYPE : uint64_t
		{
			ET_BUFFER = 1ull<<0,								//!< asset::ICPUBuffer
			ET_BUFFER_VIEW = 1ull<<1,                           //!< asset::ICPUBufferView
			ET_SAMPLER = 1ull<<2,                               //!< asset::ICPUSampler
			ET_IMAGE = 1ull<<3,									//!< asset::ICPUImage
			ET_IMAGE_VIEW = 1ull<<4,			                //!< asset::ICPUImageView
			ET_DESCRIPTOR_SET = 1ull<<5,                        //!< asset::ICPUDescriptorSet
			ET_DESCRIPTOR_SET_LAYOUT = 1ull<<6,                 //!< asset::ICPUDescriptorSetLayout
			ET_SKELETON = 1ull<<7,							    //!< asset::ICPUSkeleton
			ET_ANIMATION_LIBRARY = 1ull<<8,						//!< asset::ICPUAnimationLibrary
			ET_PIPELINE_LAYOUT = 1ull<<9,						//!< asset::ICPUPipelineLayout
			ET_SHADER = 1ull<<10,								//!< asset::ICPUShader
			ET_RENDERPASS_INDEPENDENT_PIPELINE = 1ull<<12,		//!< asset::ICPURenderpassIndependentPipeline
			ET_RENDERPASS = 1ull<<13,							//!< asset::ICPURenderpass
			ET_FRAMEBUFFER = 1ull<<14,							//!< asset::ICPUFramebuffer
			ET_GRAPHICS_PIPELINE = 1ull<<15,					//!< asset::ICPUGraphicsPipeline
			ET_BOTOM_LEVEL_ACCELERATION_STRUCTURE = 1ull<<16,	//!< asset::ICPUBottomLevelAccelerationStructure
			ET_TOP_LEVEL_ACCELERATION_STRUCTURE = 1ull<<17,		//!< asset::ICPUTopLevelAccelerationStructure
			ET_SUB_MESH = 1ull<<18,							    //!< DEPRECATED asset::ICPUMeshBuffer
			ET_MESH = 1ull<<19,								    //!< DEPRECATED asset::ICPUMesh
			ET_COMPUTE_PIPELINE = 1ull<<20,                     //!< asset::ICPUComputePipeline
			ET_PIPELINE_CACHE = 1ull<<21,						//!< asset::ICPUPipelineCache
			ET_SCENE = 1ull<<22,								//!< reserved, to implement later
			ET_IMPLEMENTATION_SPECIFIC_METADATA = 1ull<<31u,    //!< lights, etc.
			//! Reserved special value used for things like terminating lists of this enum

			ET_TERMINATING_ZERO = 0
		};
		constexpr static size_t ET_STANDARD_TYPES_COUNT = 23u;

		//! Returns a representaion of an Asset type in decimal system
		/**
			Each value is returned from the range 0 to (ET_STANDARD_TYPES_COUNT - 1) to provide
			easy way to handle array indexing. For instance ET_SUB_IMAGE returns 1 and ET_MESH
			returns 4.
		*/
		static uint32_t typeFlagToIndex(E_TYPE _type)
		{
			return hlsl::findLSB(static_cast<uint64_t>(_type));
		}

		//! Returns reinterpreted Asset for an Asset expecting full pointer type Asset
		/**
			assetType is an Asset the rootAsset will be assigned to after
			interpretation process. So if your full pointer Asset is an 
			ICPUImage you can attempt to interpate passing rootAsset
			as it. 

			It will perform assert if the attempt fails.
		*/
		template<typename assetType>
		static assetType* castDown(core::add_const_if_const_t<assetType,IAsset>* rootAsset) // maybe call it something else and not make static?
		{
			if (!rootAsset)
				return nullptr;
			assetType* typedAsset = rootAsset->getAssetType()!=assetType::AssetType ? nullptr:static_cast<assetType*>(rootAsset);
			#ifdef _NBL_DEBUG
				assert(typedAsset);
			#endif
			return typedAsset;
		}
		//! Smart pointer variant
		template<typename assetType>
		static inline core::smart_refctd_ptr<assetType> castDown(const core::smart_refctd_ptr<core::add_const_if_const_t<assetType,IAsset> >& rootAsset)
		{
			return core::smart_refctd_ptr<assetType>(castDown<assetType>(rootAsset.get()));
		}
		template<typename assetType>
		static inline core::smart_refctd_ptr<assetType> castDown(core::smart_refctd_ptr<core::add_const_if_const_t<assetType,IAsset> >&& rootAsset)
		{
			if (!castDown<assetType>(rootAsset.get()))
				return nullptr;
			return core::smart_refctd_ptr_static_cast<assetType>(std::move(rootAsset));
		}

		//! To be implemented by derived classes. Returns a type of an Asset
		virtual E_TYPE getAssetType() const = 0;

		//! creates a copy of the asset, duplicating dependant resources up to a certain depth (default duplicate everything)
        virtual core::smart_refctd_ptr<IAsset> clone(uint32_t _depth = ~0u) const = 0;

		//!
		inline bool isMutable() const {return m_mutable;}

		//!
		virtual size_t getDependantCount() const = 0;
		inline IAsset* getDependant(const size_t ix)
		{
			if (ix<getDependantCount())
				return getDependant_impl(ix);
			return nullptr;
		}

    protected:
		inline IAsset() = default;
		//! Pure virtual destructor to ensure no instantiation
		NBL_API2 virtual ~IAsset() = 0;

		virtual IAsset* getDependant_impl(const size_t ix) = 0;

	private:
		friend IAssetManager;
		bool m_mutable = true;
};

template<typename T>
concept Asset = std::is_base_of_v<IAsset,T>;

}

#endif
