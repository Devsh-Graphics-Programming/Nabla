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

	Actually an Asset is a class deriving from it that can be anything like cpu-side meshes scenes, texture data and material pipelines, 
	but must be serializable into/from .baw file format, unless it comes from an extension (nbl::ext), 
	so forth cached. There are different asset types you can find at IAsset::E_TYPE. 
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
		enum E_MUTABILITY : uint32_t
		{
			EM_MUTABLE = 0u,
			EM_CPU_PERSISTENT = 0b01u,
			EM_IMMUTABLE = 0b11u,
		};

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

		//!
		inline IAsset() : isDummyObjectForCacheAliasing{false}, m_mutability{EM_MUTABLE} {}

		//! Returns correct size reserved associated with an Asset and its data
		/**
			Some containers like std::vector reserves usually more memory than they actually need. 
			Similar behaviour appears here and it is actually necessary to reserve the correct amount of memory when writing to file.
			The value returned can be greater than memory actually needed and that symbolizes the name "conservative".

			Additionally the size is used to determine compression level while writing process is performed.
			As you expect, the bigger the size returned the more likely it is to be compressed with a more expensive (slower) algorithm.
		*/
		virtual size_t conservativeSizeEstimate() const = 0; // TODO: this shouldn't be a method of IAsset but BlobSerializable ?

		//! creates a copy of the asset, duplicating dependant resources up to a certain depth (default duplicate everything)
        virtual core::smart_refctd_ptr<IAsset> clone(uint32_t _depth = ~0u) const = 0;

		// TODO: `_other` should probably be const qualified!
		inline bool restoreFromDummy(IAsset* _other, uint32_t _levelsBelow = (~0u))
		{
			assert(getAssetType() == _other->getAssetType());

			if (!canBeRestoredFrom(_other))
				return false;

			restoreFromDummy_impl(_other, _levelsBelow);
			isDummyObjectForCacheAliasing = false;
			return true;
		}

		inline bool willBeRestoredFrom(const IAsset* _other) const
		{
			assert(getAssetType() == _other->getAssetType());

			if (getMutability() != EM_MUTABLE)
				return false;
			if (_other->getMutability() != EM_MUTABLE)
				return false;
			if (!isADummyObjectForCache())
				return false;
			if (_other->isADummyObjectForCache())
				return false;

			return true;
		}


		inline E_MUTABILITY getMutability() const { return m_mutability; }
		inline bool isMutable() const { return getMutability() == EM_MUTABLE; }
		inline bool canBeConvertedToDummy() const { return !isADummyObjectForCache() && getMutability() < EM_CPU_PERSISTENT; }

		// TODO: add a null and type check here, delegate rest to an `impl`
		virtual bool canBeRestoredFrom(const IAsset* _other) const = 0;

		// returns if `this` is dummy or any of its dependencies up to `_levelsBelow` levels below
		inline bool isAnyDependencyDummy(uint32_t _levelsBelow = ~0u) const
		{
			if (isADummyObjectForCache())
				return true;

			return _levelsBelow ? isAnyDependencyDummy_impl(_levelsBelow) : false;
		}

    protected:
		inline static void restoreFromDummy_impl_call(IAsset* _this_child, IAsset* _other_child, uint32_t _levelsBelow)
		{
			_this_child->restoreFromDummy_impl(_other_child, _levelsBelow);
		}

		virtual void restoreFromDummy_impl(IAsset* _other, uint32_t _levelsBelow) = 0;

		// returns if any of `this`'s up to `_levelsBelow` levels below is dummy
		virtual bool isAnyDependencyDummy_impl(uint32_t _levelsBelow) const { return false; }

        inline void clone_common(IAsset* _clone) const
        {
            assert(!isDummyObjectForCacheAliasing);
            _clone->isDummyObjectForCacheAliasing = false;
            _clone->m_mutability = EM_MUTABLE;
        }
		inline bool isImmutable_debug()
		{
			const bool imm = getMutability() == EM_IMMUTABLE;
			//_NBL_DEBUG_BREAK_IF(imm);
			return imm;
		}

	private:
		friend IAssetManager;

	protected:
		bool isDummyObjectForCacheAliasing; //!< A bool for setting whether Asset is in dummy state. @see convertToDummyObject(uint32_t referenceLevelsBelowToConvert)

		E_MUTABILITY m_mutability;

		//! To be implemented by base classes, dummies must retain references to other assets
		//! but cleans up all other resources which are not assets.
		/**
			Dummy object is an object which is converted to GPU object or which is about to be converted to GPU object.
			Take into account that\b convertToDummyObject(uint32_t referenceLevelsBelowToConvert) itself doesn't perform exactly converting to GPU object\b. 

			@see IAssetManager::convertAssetToEmptyCacheHandle(IAsset* _asset, core::smart_refctd_ptr<core::IReferenceCounted>&& _gpuObject)

			If an Asset is being converted to a GPU object, its resources are no longer needed in RAM memory,
			so everything it has allocated becomes deleted, but the Asset itself remains untouched, so that is the 
			pointer for an Asset and the memory allocated for that pointer. It's because it's needed as a key by some 
			functions that find GPU objects. It involves all CPU objects (Assets).

			So an Asset signed as dummy becomes GPU object and deletes some resources in RAM memory.

			@param referenceLevelsBelowToConvert says how many times to recursively call `convertToDummyObject` on its references.
		*/
		virtual void convertToDummyObject(uint32_t referenceLevelsBelowToConvert=0u) = 0;

        inline void convertToDummyObject_common(uint32_t referenceLevelsBelowToConvert)
        {
			if (canBeConvertedToDummy())
				isDummyObjectForCacheAliasing = true;
        }

		//! Checks if the object is either not dummy or dummy but in some cache for a purpose
		inline bool isInValidState() { return !isDummyObjectForCacheAliasing /* || !isCached TODO*/; }

		//! Pure virtual destructor to ensure no instantiation
		NBL_API2 virtual ~IAsset() = 0;

	public:
		//! To be implemented by derived classes. Returns a type of an Asset
		virtual E_TYPE getAssetType() const = 0;

		//! Returning isDummyObjectForCacheAliasing, specifies whether Asset in dummy state
		inline bool isADummyObjectForCache() const { return isDummyObjectForCacheAliasing; }
};

}

#endif
