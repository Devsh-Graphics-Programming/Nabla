// Copyright (C) 2024-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
#ifndef _NBL_VIDEO_C_ASSET_CONVERTER_INCLUDED_
#define _NBL_VIDEO_C_ASSET_CONVERTER_INCLUDED_


#include "nbl/asset/utils/ISPIRVOptimizer.h"
#include "nbl/video/utilities/IUtilities.h"
#include "nbl/video/asset_traits.h"
#include "nbl/builtin/hlsl/cpp_compat.hlsl"


namespace nbl::video
{
/*
* This whole class assumes all assets you are converting will be used in read-only mode by the Device.
* It's a valid assumption for everything from pipelines to shaders, but not for descriptors (with exception of samplers) and their sets.
* 
* Only Descriptors (by proxy their backing objects) and their Sets can have their contents changed after creation.
* 
* What this converter does is it computes hashes and compares equality based on the contents of an IAsset, not the pointer!
* With some sane limits, its not going to compare the contents of an ICPUImage or ICPUBuffer.
* 
* Therefore if you don't want some resource to be deduplicated you need to "explicitly" let us know via `SInputs::getDependantUniqueCopyGroupID`.
*/
class CAssetConverter : public core::IReferenceCounted
{
	public:
		// Shader, DSLayout, PipelineLayout, Compute Pipeline
		// Renderpass, Graphics Pipeline
		// Buffer, BufferView, Sampler, Image, Image View, Bottom Level AS, Top Level AS, Descriptor Set, Framebuffer  
		// Buffer -> SRange, patched usage, owner(s)
		// BufferView -> SRange, promoted format
		// Sampler -> Clamped Params (only aniso, really)
		// Image -> this, patched usage, promoted format
		// Image View -> ref to patched Image, patched usage, promoted format
		// Descriptor Set -> unique layout, 
		using supported_asset_types = core::type_list<
			asset::ICPUSampler,
			asset::IShader,
			asset::ICPUBuffer,
#ifdef NBL_ACCELERATION_STRUCTURE_CONVERSION
			asset::ICPUBottomLevelAccelerationStructure,
			asset::ICPUTopLevelAccelerationStructure,
#endif
			asset::ICPUImage,
			asset::ICPUBufferView,
			asset::ICPUImageView,
			asset::ICPUDescriptorSetLayout,
			asset::ICPUPipelineLayout,
			asset::ICPUPipelineCache,
			asset::ICPUComputePipeline,
			asset::ICPURenderpass,
			asset::ICPUGraphicsPipeline,
			asset::ICPUDescriptorSet
			//asset::ICPUFramebuffer doesn't exist yet XD
		>;

		struct SCreationParams
		{
			inline bool valid() const
			{
				if (!device)
					return false;

				return true;
			}

			// required not null
			ILogicalDevice* device = nullptr;
			// optional
			core::smart_refctd_ptr<const asset::ISPIRVOptimizer> optimizer = {};
		};
		static inline core::smart_refctd_ptr<CAssetConverter> create(SCreationParams&& params)
		{
			if (!params.valid())
				return nullptr;
		#ifndef _NBL_DEBUG
			if (!params.optimizer)
			{
				using pass_e = asset::ISPIRVOptimizer::E_OPTIMIZER_PASS;
				// shall we do others?
				const pass_e passes[] = {pass_e::EOP_STRIP_DEBUG_INFO};
				params.optimizer = core::make_smart_refctd_ptr<asset::ISPIRVOptimizer>(passes);
			}
		#endif
			return core::smart_refctd_ptr<CAssetConverter>(new CAssetConverter(std::move(params)),core::dont_grab);
		}
		// When getting dependents, the creation parameters of GPU objects will be produced and patched appropriately.
		// `patch_t` uses CRTP to inherit from `patch_impl_t` to provide default `operator==` and `update_hash()` definition.
		// The default specialization kicks in for any `AssetType` that has nothing possible to patch (e.g. Descriptor Set Layout).
		template<asset::Asset AssetType>
		struct patch_impl_t
		{
#define PATCH_IMPL_BOILERPLATE(ASSET_TYPE) using this_t = patch_impl_t<ASSET_TYPE>; \
			public: \
				inline patch_impl_t() = default; \
				inline patch_impl_t(const this_t& other) = default; \
				inline patch_impl_t(this_t&& other) = default; \
				inline this_t& operator=(const this_t& other) = default; \
				inline this_t& operator=(this_t&& other) = default; \
				patch_impl_t(const ASSET_TYPE* asset); \
				bool valid(const ILogicalDevice* device)

				PATCH_IMPL_BOILERPLATE(AssetType);

			protected:
				// there's nothing to combine, so combining always produces the input successfully
				inline std::pair<bool,this_t> combine(const this_t& other) const
				{
					return {true,*this};
				}
		};
		template<>
		struct NBL_API2 patch_impl_t<asset::ICPUSampler>
		{
			public:
				PATCH_IMPL_BOILERPLATE(asset::ICPUSampler);

				uint8_t anisotropyLevelLog2 = 6;
				
			protected:
				inline std::pair<bool,this_t> combine(const this_t& other) const
				{
					// The only reason why someone would have a different level to creation parameters is
					// because the HW doesn't support that level and the level gets clamped. So must be same.
					if (anisotropyLevelLog2!=other.anisotropyLevelLog2)
						return {false,{}}; // invalid
					return {true,*this};
				}
		};
		template<>
		struct NBL_API2 patch_impl_t<asset::ICPUBuffer>
		{
			public:
				PATCH_IMPL_BOILERPLATE(asset::ICPUBuffer);

				using usage_flags_t = IGPUBuffer::E_USAGE_FLAGS;
				core::bitflag<usage_flags_t> usage = usage_flags_t::EUF_NONE;

			protected:
				inline std::pair<bool,this_t> combine(const this_t& other) const
				{
					this_t retval = *this;
					retval.usage |= other.usage;
					return {true,retval};
				}
		};
		struct NBL_API2 acceleration_structure_patch_base
		{
			public:
				enum class BuildPreference : uint8_t
				{
					None = 0,
					FastTrace = 1,
					FastBuild = 2,
					Invalid = 3
				};

				//! select build flags
				uint8_t allowUpdate : 1 = false;
				uint8_t allowCompaction : 1 = false;
				uint8_t allowDataAccess : 1 = false;
				BuildPreference preference : 2 = BuildPreference::Invalid;
				uint8_t lowMemory : 1 = false;
				//! things that control the build
				uint8_t hostBuild : 1 = false; // DO NOT USE, will get overriden to false anyway
				uint8_t compactAfterBuild : 1 = false;

			protected:
				bool valid(const ILogicalDevice* device);
				
				template<typename CRTP>
				std::pair<bool,CRTP> combine_impl(const CRTP& _this, const CRTP& other) const
				{
					if (_this.preference!=other.preference || _this.preference==BuildPreference::Invalid)
						return {false,_this};
					CRTP retval = _this;
					retval.allowUpdate |= other.allowUpdate;
					retval.allowCompaction |= other.allowCompaction;
					retval.allowDataAccess |= other.allowDataAccess;
					retval.lowMemory |= other.lowMemory;
					// Host Builds are presumed to be "beter quality" and lower staging resource pressure,
					// we may change the behaviour here in the future
					retval.hostBuild |= other.hostBuild;
					retval.compactAfterBuild |= other.compactAfterBuild;
					return {true,retval};
				}
		};
		template<>
		struct NBL_API2 patch_impl_t<asset::ICPUBottomLevelAccelerationStructure> : acceleration_structure_patch_base
		{
			public:
				PATCH_IMPL_BOILERPLATE(asset::ICPUBottomLevelAccelerationStructure);

				using build_flags_t = asset::ICPUBottomLevelAccelerationStructure::BUILD_FLAGS;
				core::bitflag<build_flags_t> getBuildFlags(const asset::ICPUBottomLevelAccelerationStructure* blas) const;

			protected:
				inline std::pair<bool,this_t> combine(const this_t& other) const
				{
					return combine_impl<this_t>(*this,other);
				}
		};
		template<>
		struct NBL_API2 patch_impl_t<asset::ICPUTopLevelAccelerationStructure> : acceleration_structure_patch_base
		{
			public:
				PATCH_IMPL_BOILERPLATE(asset::ICPUTopLevelAccelerationStructure);

				using build_flags_t = asset::ICPUTopLevelAccelerationStructure::BUILD_FLAGS;
				core::bitflag<build_flags_t> getBuildFlags(const asset::ICPUTopLevelAccelerationStructure* tlas) const;

			protected:
				inline std::pair<bool,this_t> combine(const this_t& other) const
				{
					return combine_impl<this_t>(*this,other);
				}
		};
		template<>
		struct NBL_API2 patch_impl_t<asset::ICPUImage>
		{
			public:
				PATCH_IMPL_BOILERPLATE(asset::ICPUImage);

				using usage_flags_t = IGPUImage::E_USAGE_FLAGS;
				constexpr static inline usage_flags_t UsagesThatPreventFormatPromotion = usage_flags_t::EUF_RENDER_ATTACHMENT_BIT|usage_flags_t::EUF_INPUT_ATTACHMENT_BIT;
				// make our promotion policy explicit
				inline bool canAttemptFormatPromotion() const
				{
					// if there exist views of the image that reinterpret cast its texel blocks, stop promotion, aliasing can't work with promotion!
					if (mutableFormat)
						return false;
					// we don't support promoting formats in renderpasses' attachment descriptions, so stop it here too
					if (!usageFlags.hasAnyFlag(UsagesThatPreventFormatPromotion))
						return false;
					if (!stencilUsage.hasAnyFlag(UsagesThatPreventFormatPromotion))
						return false;
					return true;
				}

				// the most important thing about an image
				asset::E_FORMAT format = asset::EF_UNKNOWN;
				// but we also track separate dpeth and stencil usage flags
				core::bitflag<usage_flags_t> usageFlags = usage_flags_t::EUF_NONE;
				core::bitflag<usage_flags_t> stencilUsage = usage_flags_t::EUF_NONE;
				// moar stuff
				uint32_t mutableFormat : 1 = false;
				uint32_t cubeCompatible : 1 = false;
				uint32_t _3Dbut2DArrayCompatible : 1 = false;
				// we sort of ignore that at the end if the format doesn't stay block compressed
				uint32_t uncompressedViewOfCompressed : 1 = false;
				// Extra metadata needed for format promotion, if you want any of them (except for `linearlySampled` and `depthCompareSampledImage`)
				// as anything other than the default values, use explicit input roots with patches. Otherwise if `format` is not supported by device
				// the view can get promoted to a format that doesn't have these usage capabilities.
				uint32_t linearlySampled : 1 = false;
				uint32_t storageAtomic : 1 = false;
				uint32_t storageImageLoadWithoutFormat : 1 = false;
				uint32_t depthCompareSampledImage : 1 = false;
				// all converted images default to optimal!
				uint32_t linearTiling : 1 = false;
				// aside from format promotion, we can also promote images to have a fuller mip chain and recompute it
				uint32_t mipLevels : 7 = 0;
				uint32_t recomputeMips : 16 = false;

			protected:
				inline std::pair<bool,this_t> combine(const this_t& other) const
				{
					this_t retval = *this;
					// changing tiling would mess up everything to do with format validation
					if (linearTiling!=other.linearTiling)
						return {false,retval};

					// combine usage flags
					retval.usageFlags |= other.usageFlags;
					retval.stencilUsage |= other.stencilUsage;
					// creation flag relevant for format promotion
					retval.mutableFormat |= other.mutableFormat;
					// and meta-usages
					retval.linearlySampled |= other.linearlySampled;
					retval.storageAtomic |= other.storageAtomic;
					retval.storageImageLoadWithoutFormat |= other.storageImageLoadWithoutFormat;
					retval.depthCompareSampledImage |= other.depthCompareSampledImage;
					// Patches only differ by format if it was promoted, and you might ask yourself:
					// "What if due to different usages `this` and `other` get promoted to different formats?"
					// `valid` does not promote the format, format gets promoted AFTER the whole DFS pass
					if (format!=other.format) // during the DFS phase formats will match, if we're here, we're in a subsequent phase
					{
						// During non-DFS phase, `other` is always an immediate temporary patch, without promoted format
						// and a matching `this` must always be a superset of `other` for format promotion to remain valid!
						if (memcmp(this,&retval,sizeof(retval))!=0) // no usages were added
							return {false,retval};
					}
					// rest of creation flags
					retval.cubeCompatible |= other.cubeCompatible;
					retval._3Dbut2DArrayCompatible |= other._3Dbut2DArrayCompatible;
					retval.uncompressedViewOfCompressed |= other.uncompressedViewOfCompressed;
					// We don't touch `mipLevels` or `recomputeMips` because during DFS they're uninitialized
					// and during post-DFS phase, `this` is the already patched entry we're merging to, so always takes precedence
					// Because merge is only called on identical asset and groupID handles, SInputs callback is called with same parameters always
					// therefore we don't need to think about how patches with different `mipLevels` or `recomputeMips` values would merge.
					return {true,retval};
				}
		};
		template<>
		struct NBL_API2 patch_impl_t<asset::ICPUBufferView>
		{
			public:
				PATCH_IMPL_BOILERPLATE(asset::ICPUBufferView);

				uint8_t stbo : 1 = false;
				uint8_t utbo : 1 = false;
				uint8_t mustBeZero : 6 = 0;

			protected:
				inline std::pair<bool,this_t> combine(const this_t& other) const
				{
					this_t retval = *this;
					retval.stbo |= other.stbo;
					retval.utbo |= other.utbo;
					return {true,retval};
				}
		};
		template<>
		struct NBL_API2 patch_impl_t<asset::ICPUImageView>
		{
			private:
				using this_t = patch_impl_t<asset::ICPUImageView>;

			public:
				inline patch_impl_t() = default;
				inline patch_impl_t(const this_t& other) = default;
				inline patch_impl_t(this_t&& other) = default;
				inline this_t& operator=(const this_t& other) = default;
				inline this_t& operator=(this_t&& other) = default;

				using usage_flags_t = IGPUImage::E_USAGE_FLAGS;
				// slightly weird constructor because it deduces the metadata from subusages, so need the subusages right away, not patched later
				patch_impl_t(const asset::ICPUImageView* asset, const core::bitflag<usage_flags_t> extraSubUsages=usage_flags_t::EUF_NONE);

				bool valid(const ILogicalDevice* device);

				//
				inline bool formatFollowsImage() const
				{
					return originalFormat==asset::EF_UNKNOWN;
				}

				// just because we record all subusages we can find, doesn't mean we will set them on the created image
				core::bitflag<usage_flags_t> subUsages = usage_flags_t::EUF_NONE;
				// Extra metadata needed for format promotion, if you want any of them (except for `linearlySampled` and `depthCompareSampledImage`)
				// as anything other than the default values, use explicit input roots with patches. Otherwise if `format` is not supported by device
				// the view can get promoted to a format that doesn't have these usage capabilities.
				uint8_t linearlySampled : 1 = false;
				uint8_t storageAtomic : 1 = false;
				uint8_t storageImageLoadWithoutFormat : 1 = false;
				uint8_t depthCompareSampledImage : 1 = false;

			protected:
				uint8_t invalid : 1 = false;
				// to not mess with hashing and comparison
				uint8_t padding : 3 = 0;
				// normally wouldn't store that but we don't provide a ref/pointer to the asset when combining or checking validity, treat member as impl detail
				asset::E_FORMAT originalFormat = asset::EF_UNKNOWN;

				inline std::pair<bool,this_t> combine(const this_t& other) const
				{
					assert(padding==0);
					if (invalid || other.invalid)
						return {false,*this};

					this_t retval = *this;
					// So we have two patches of the same image view, ergo they were the same format.
					// If one mutates and other doesn't its because of added usages that preclude, so make us immutable again.
					if (formatFollowsImage() && !other.formatFollowsImage())
						retval.originalFormat = other.originalFormat;
					// When combining usages, we already:
					// - require that two patches' formats were identical
					// - require that each patch be valid in on its own
					// therefore both patches' usages are valid for the format at the time of combining
					retval.subUsages |= other.subUsages;
					retval.linearlySampled |= other.linearlySampled;
					retval.storageAtomic |= other.storageAtomic;
					retval.storageImageLoadWithoutFormat |= other.storageImageLoadWithoutFormat;
					retval.depthCompareSampledImage |= other.depthCompareSampledImage;
					return {true,retval};
				}
		};
		template<>
		struct NBL_API2 patch_impl_t<asset::ICPUPipelineLayout>
		{
			public:
				PATCH_IMPL_BOILERPLATE(asset::ICPUPipelineLayout);

				using shader_stage_t = hlsl::ShaderStage;
				std::array<core::bitflag<shader_stage_t>,asset::CSPIRVIntrospector::MaxPushConstantsSize> pushConstantBytes = {shader_stage_t::ESS_UNKNOWN};
				
			protected:
				inline std::pair<bool,this_t> combine(const this_t& other) const
				{
					if (invalid || other.invalid)
						return {false,*this};
					this_t retval = *this;
					for (auto byte=0; byte!=pushConstantBytes.size(); byte++)
						retval.pushConstantBytes[byte] |= other.pushConstantBytes[byte];
					return {true,retval};
				}

				bool invalid = true;
		};
#undef PATCH_IMPL_BOILERPLATE
		// The default specialization provides simple equality operations and hash operations, this will work as long as your patch_impl_t doesn't:
		// - use a container like `core::vector<T>`, etc.
		// - use pointers to other objects or arrays whose contents must be analyzed
		template<asset::Asset AssetType>
		struct patch_t final : patch_impl_t<AssetType>
		{
			using this_t = patch_t<AssetType>;
			using base_t = patch_impl_t<AssetType>;

			// forwarding
			using base_t::base_t;
			inline patch_t(const this_t& other) : base_t(other) {}
			inline patch_t(this_t&& other) : base_t(std::move(other)) {}
			inline patch_t(base_t&& other) : base_t(std::move(other)) {}

			inline this_t& operator=(const this_t& other) = default;
			inline this_t& operator=(this_t&& other) = default;

			// The assumption is we'll only ever be combining valid patches together.
			// Returns: whether the combine op was a success, DOESN'T MEAN the result is VALID!
			inline std::pair<bool,this_t> combine(const this_t& other) const
			{
				//assert(base_t::valid() && other.valid());
				return base_t::combine(other);
			}

			// actual new methods
			inline bool operator==(const patch_t<AssetType>& other) const
			{
				if (std::is_empty_v<base_t>)
					return true; 
				return memcmp(this,&other,sizeof(base_t))==0;
			}
		};
		// A class to accelerate our hash computations
		class CHashCache final : public core::IReferenceCounted
		{
			public:
				//
				template<asset::Asset AssetType>
				struct lookup_t
				{
					const AssetType* asset = nullptr;
					const patch_t<AssetType>* patch = {};
				};

			private:
				//
				template<asset::Asset AssetType>
				struct key_t
				{
					core::smart_refctd_ptr<const AssetType> asset = {};
					patch_t<AssetType> patch = {};
				};
				template<asset::Asset AssetType>
				struct HashEquals
				{
					using is_transparent = void;

					inline size_t operator()(const key_t<AssetType>& key) const
					{
						return operator()(lookup_t<AssetType>{key.asset.get(),&key.patch});
					}
					inline size_t operator()(const lookup_t<AssetType>& lookup) const
					{
						core::blake3_hasher hasher;
						hasher << ptrdiff_t(lookup.asset);
						hasher << *lookup.patch;
						// put long hash inside a small hash
						return std::hash<core::blake3_hash_t>()(static_cast<core::blake3_hash_t>(hasher));
					}

					inline bool operator()(const key_t<AssetType>& lhs, const key_t<AssetType>& rhs) const
					{
						return lhs.asset.get()==rhs.asset.get() && lhs.patch==rhs.patch;
					}
					inline bool operator()(const key_t<AssetType>& lhs, const lookup_t<AssetType>& rhs) const
					{
						return lhs.asset.get()==rhs.asset && rhs.patch && lhs.patch==*rhs.patch;
					}
					inline bool operator()(const lookup_t<AssetType>& lhs, const key_t<AssetType>& rhs) const
					{
						return lhs.asset==rhs.asset.get() && lhs.patch && *lhs.patch==rhs.patch;
					}
				};
				template<asset::Asset AssetType>
				using container_t = core::unordered_map<key_t<AssetType>,core::blake3_hash_t,HashEquals<AssetType>,HashEquals<AssetType>>;

			public:
				static const core::blake3_hash_t NoContentHash;

				inline CHashCache() = default;

				//
				template<asset::Asset AssetType>
				inline container_t<AssetType>::iterator find(const lookup_t<AssetType>& assetAndPatch)
				{
					return std::get<container_t<AssetType>>(m_containers).find<lookup_t<AssetType>>(assetAndPatch);
				}
				template<asset::Asset AssetType>
				inline container_t<AssetType>::const_iterator find(const lookup_t<AssetType>& assetAndPatch) const
				{
					return std::get<container_t<AssetType>>(m_containers).find<lookup_t<AssetType>>(assetAndPatch);
				}
				template<asset::Asset AssetType>
				inline container_t<AssetType>::const_iterator end() const
				{
					return std::get<container_t<AssetType>>(m_containers).end();
				}

				//
				class IPatchOverride
				{
					public:
						virtual const patch_t<asset::ICPUSampler>* operator()(const lookup_t<asset::ICPUSampler>&) const = 0;
						virtual const patch_t<asset::ICPUBuffer>* operator()(const lookup_t<asset::ICPUBuffer>&) const = 0;
#ifdef NBL_ACCELERATION_STRUCTURE_CONVERSION
						virtual const patch_t<asset::ICPUBottomLevelAccelerationStructure>* operator()(const lookup_t<asset::ICPUBottomLevelAccelerationStructure>&) const = 0;
						virtual const patch_t<asset::ICPUTopLevelAccelerationStructure>* operator()(const lookup_t<asset::ICPUTopLevelAccelerationStructure>&) const = 0;
#endif
						virtual const patch_t<asset::ICPUImage>* operator()(const lookup_t<asset::ICPUImage>&) const = 0;
						virtual const patch_t<asset::ICPUBufferView>* operator()(const lookup_t<asset::ICPUBufferView>&) const = 0;
						virtual const patch_t<asset::ICPUImageView>* operator()(const lookup_t<asset::ICPUImageView>&) const = 0;
						virtual const patch_t<asset::ICPUPipelineLayout>* operator()(const lookup_t<asset::ICPUPipelineLayout>&) const = 0;

						// certain items are not patchable, so there's no `patch_t` with non zero size
						inline const patch_t<asset::IShader>* operator()(const lookup_t<asset::IShader>& unpatchable) const
						{
							return unpatchable.patch;
						}
						inline const patch_t<asset::ICPUDescriptorSetLayout>* operator()(const lookup_t<asset::ICPUDescriptorSetLayout>& unpatchable) const
						{
							return unpatchable.patch;
						}
						inline const patch_t<asset::ICPURenderpass>* operator()(const lookup_t<asset::ICPURenderpass>& unpatchable) const
						{
							return unpatchable.patch;
						}
						inline const patch_t<asset::ICPUDescriptorSet>* operator()(const lookup_t<asset::ICPUDescriptorSet>& unpatchable) const
						{
							return unpatchable.patch;
						}

						// while other things are top level assets in the graph and `operator()` would never be called on their patch
				};
				// `cacheMistrustLevel` is how deep from `asset` do we start trusting the cache to contain correct non stale hashes
				template<asset::Asset AssetType>
				inline core::blake3_hash_t hash(const lookup_t<AssetType>& lookup, const IPatchOverride* patchOverride, const uint32_t cacheMistrustLevel=0)
				{
					if (!patchOverride || !lookup.asset || !lookup.patch)// || !lookup.patch->valid()) we assume any patch gotten is valid (to not have a dependancy on the device)
						return NoContentHash;

					// consult cache
					auto foundIt = find(lookup);
					auto& container = std::get<container_t<AssetType>>(m_containers);
					const bool found = foundIt!=container.end();
					// if found and we trust then return the cached hash
					if (cacheMistrustLevel==0 && found)
						return foundIt->second;

					// proceed with full hash computation
					core::blake3_hasher hasher = {};
					// We purposefully don't hash asset pointer, we hash the contents instead
					hash_impl impl = {{
							.hashCache = this,
							.patchOverride = patchOverride,
							.nextMistrustLevel = cacheMistrustLevel ? (cacheMistrustLevel-1):0,
							.hasher  = hasher
					}};
					// failed to hash (missing required deps), so return invalid hash
					// but don't eject stale entry, this may have been a mistake
					if (!impl(lookup))
						return NoContentHash;
					const auto retval = static_cast<core::blake3_hash_t>(hasher);
					assert(retval!=NoContentHash);

					if (found) // replace stale entry
						foundIt->second = retval;
					else // insert new entry
					{
						auto [insertIt,inserted] = container.emplace(
							key_t<AssetType>{
								.asset = core::smart_refctd_ptr<const AssetType>(lookup.asset),
								.patch = *lookup.patch
							},
							retval
						);
						assert(inserted && HashEquals<AssetType>()(insertIt->first,lookup) && insertIt->second==retval);
					}
					return retval;
				}

				// Its fastest to erase if you know your patch
				template<asset::Asset AssetType>
				inline bool erase(const lookup_t<AssetType>& what)
				{
					return std::get<container_t<AssetType>>(m_containers).erase(what)>0;
				}
				// Warning: Linear Search! Super slow!
				template<asset::Asset AssetType>
				inline bool erase(const AssetType* asset)
				{
					// TODO: improve by cycling through possible patches when the set of possibilities is small
					return core::erase_if(std::get<container_t<AssetType>>(m_containers),[asset](const auto& entry)->bool
						{
							auto const& [key,value] = entry;
							return key.asset==asset;
						}
					);
				}
				// TODO: `eraseStale(const IAsset*)` which erases a subgraph?
				// An asset being pointed to can mutate and that would invalidate the hash, this recomputes all hashes.
				void eraseStale(const IPatchOverride* patchOverride);
				// Clear the cache for a given type
				template<asset::Asset AssetType>
				inline void clear()
				{
					std::get<container_t<AssetType>>(m_containers).clear();
				}
				// Clear the caches completely
				inline void clear()
				{
					core::for_each_in_tuple(m_containers,[](auto& container)->void{container.clear();});
				}

				// only public to allow inheritance later in the cpp file
				struct hash_impl_base
				{
					CHashCache* hashCache;
					const IPatchOverride* patchOverride;
					const uint32_t nextMistrustLevel;
					core::blake3_hasher& hasher;
				};

			private:
				inline ~CHashCache() = default;

				// only public to allow inheritance later in the cpp file
				struct NBL_API2 hash_impl : hash_impl_base
				{
					bool operator()(lookup_t<asset::ICPUSampler>);
					bool operator()(lookup_t<asset::IShader>);
					bool operator()(lookup_t<asset::ICPUBuffer>);
					bool operator()(lookup_t<asset::ICPUBottomLevelAccelerationStructure>);
					bool operator()(lookup_t<asset::ICPUTopLevelAccelerationStructure>);
					bool operator()(lookup_t<asset::ICPUImage>);
					bool operator()(lookup_t<asset::ICPUBufferView>);
					bool operator()(lookup_t<asset::ICPUImageView>);
					bool operator()(lookup_t<asset::ICPUDescriptorSetLayout>);
					bool operator()(lookup_t<asset::ICPUPipelineLayout>);
					bool operator()(lookup_t<asset::ICPUPipelineCache>);
					bool operator()(lookup_t<asset::ICPUComputePipeline>);
					bool operator()(lookup_t<asset::ICPURenderpass>);
					bool operator()(lookup_t<asset::ICPUGraphicsPipeline>);
					bool operator()(lookup_t<asset::ICPUDescriptorSet>);
				};

				//
				core::tuple_transform_t<container_t,supported_asset_types> m_containers;
		};
		// Typed Cache (for a particular AssetType)
		class CCacheBase
		{
			public:
				// Make it clear to users that we don't look up just by the asset content hash
				struct key_t
				{
					inline key_t(const core::blake3_hash_t& contentHash, const size_t uniqueCopyGroupID) : value(contentHash)
					{
						reinterpret_cast<size_t*>(value.data)[0] ^= uniqueCopyGroupID;
					}

					inline bool operator==(const key_t&) const = default;

					// The blake3 hash is quite fat (256bit), so we don't actually store a full asset ref for comparison.
					// Assuming a uniform distribution of keys and perfect hashing, we'd expect a collision on average every 2^256 asset loads.
					// Or if you actually calculate the P(X>1) for any reasonable number of asset loads (k trials), the Poisson CDF will be pratically 0.
					core::blake3_hash_t value;
				};

			protected:
				struct ForwardHash
				{
					inline size_t operator()(const key_t& key) const
					{
						return std::hash<core::blake3_hash_t>()(key.value);
					}
				};
		};
		template<asset::Asset AssetType>
        class CCache final : public CCacheBase
        {
			public:
				// typedefs
				using forward_map_t = core::unordered_map<key_t,asset_cached_t<AssetType>,ForwardHash>;
				using reverse_map_t = core::unordered_map<typename asset_traits<AssetType>::lookup_t,key_t>;


				//
				inline CCache() = default;
				inline CCache(const CCache&) = default;
				inline CCache(CCache&&) = default;
				inline ~CCache() = default;

				inline CCache& operator=(const CCache&) = default;
				inline CCache& operator=(CCache&&) = default;

				// no point returning iterators to inserted positions, they're not stable
				inline bool insert(const key_t& _key, const asset_cached_t<AssetType>& _gpuObj)
				{
					auto [unused0,insertedF] = m_forwardMap.emplace(_key,_gpuObj);
					if (!insertedF)
						return false;
					auto [unused1,insertedR] = m_reverseMap.emplace(_gpuObj.get(),_key);
					assert(insertedR);
					return true;
				}

				//
				inline size_t size() const
				{
					assert(m_forwardMap.size()==m_reverseMap.size());
					return m_forwardMap.size();
				}

				//
				inline forward_map_t::const_iterator forwardMapEnd() const {return m_forwardMap.end();}
				inline reverse_map_t::const_iterator reverseMapEnd() const {return m_reverseMap.end();}

				// fastest lookup
				inline forward_map_t::const_iterator find(const key_t& _key) const {return m_forwardMap.find(_key);}
				inline reverse_map_t::const_iterator find(asset_traits<AssetType>::lookup_t gpuObject) const {return m_reverseMap.find(gpuObject);}

				// fastest erase
				inline bool erase(forward_map_t::const_iterator fit, reverse_map_t::const_iterator rit)
				{
					if (fit->first!=rit->second || fit->second.get()!=rit->first)
						return false;
					m_reverseMap.erase(rit);
					m_forwardMap.erase(fit);
					return true;
				}
				inline bool erase(forward_map_t::const_iterator it)
				{
					return erase(it,find(it->second));
				}
				inline bool erase(reverse_map_t::const_iterator it)
				{
					return erase(find(it->second),it);
				}

				//
				inline void merge(const CCache<AssetType>& other)
				{
					m_forwardMap.insert(other.m_forwardMap.begin(),other.m_forwardMap.end());
					m_reverseMap.insert(other.m_reverseMap.begin(),other.m_reverseMap.end());
				}

			private:
				friend class CAssetConverter;

				forward_map_t m_forwardMap;
				reverse_map_t m_reverseMap;
        };

		// A meta class to encompass all the Assets you might want to convert at once
        struct SInputs
        {
			// Normally all references to the same IAsset* would spawn the same IBackendObject*.
			// You need to tell us if an asset needs multiple copies, separate for each user. The return value of this function dictates what copy of the asset each user gets.
			// Each unique integer value returned for a given input `dependant` "spawns" a new copy.
			// Note that the group ID is the same size as a pointer, so you can e.g. cast a pointer of the user (parent reference) to size_t and use that for a unique copy for the user.
			// Note that we also call it with `user=={nullptr,0xdeadbeefBADC0FFEull}` for each entry in `SInputs::assets`.
			// NOTE: You might get extra copies within the same group ID due to inability to patch entries
			virtual inline size_t getDependantUniqueCopyGroupID(const size_t usersGroupCopyID, const asset::IAsset* user, const asset::IAsset* dependant) const
			{
				return 0;
			}

			// If you want concurrent sharing return a list here, REMEMBER THAT IF YOU DON'T INCLUDE THE LATER QUEUE FAMILIES USED in `SConvertParams` you'll fail!
			virtual inline std::span<const uint32_t> getSharedOwnershipQueueFamilies(const size_t groupCopyID, const asset::ICPUBuffer* buffer, const patch_t<asset::ICPUBuffer>& patch) const
			{
				return {};
			}

			// this a weird signature, but its for the IGPUBuffer backing an acceleration structure
			virtual inline std::span<const uint32_t> getSharedOwnershipQueueFamilies(const size_t groupCopyID, const asset::ICPUBottomLevelAccelerationStructure* blas, const patch_t<asset::ICPUBottomLevelAccelerationStructure>& patch) const
			{
				return {};
			}
			virtual inline std::span<const uint32_t> getSharedOwnershipQueueFamilies(const size_t groupCopyID, const asset::ICPUTopLevelAccelerationStructure* tlas, const patch_t<asset::ICPUTopLevelAccelerationStructure>& patch) const
			{
				return {};
			}

			virtual inline std::span<const uint32_t> getSharedOwnershipQueueFamilies(const size_t groupCopyID, const asset::ICPUImage* buffer, const patch_t<asset::ICPUImage>& patch) const
			{
				return {};
			}

			// most plain PNG, JPG, etc. loaders don't produce images with mip chains/tails
			virtual inline uint8_t getMipLevelCount(const size_t groupCopyID, const asset::ICPUImage* image, const patch_t<asset::ICPUImage>& patch) const
			{
				assert(image);
				const auto& params = image->getCreationParameters();
				// failure/no-change balue
				const auto origCount = params.mipLevels;
				const auto format = patch.format;
				// unlikely anyone needs mipmaps without this usage
				if (!patch.usageFlags.hasFlags(IGPUImage::E_USAGE_FLAGS::EUF_SAMPLED_BIT))
					return 0;
				// makes no sense to have a mip-map of integer values, and we can't encode into BC formats (yet)
				if (asset::isIntegerFormat(format) || asset::isBlockCompressionFormat(format))
					return origCount;
				// original image did already have a mip tail, or we don't have any base level data
				if (origCount!=1 || image->getRegions(0).empty())
					return origCount;
				// ok lets do a full one then
				const auto maxExtent = std::max<uint32_t>(std::max<uint32_t>(params.extent.width,params.extent.height),params.extent.depth);
				assert(maxExtent>0);
				return hlsl::findMSB(maxExtent)+1;
			}

			// Bitfield of which mip levels will get recomputed, gets called AFTER `getMipLevelCount` on the same image asset instance
			// Bit 0 is mip level 1, Bit N is mip level N+1, as the base cannot be recomputed
			virtual inline uint16_t needToRecomputeMips(const size_t groupCopyID, const asset::ICPUImage* image, const patch_t<asset::ICPUImage>& patch) const
			{
				assert(image);
				const auto format = patch.format;
				// makes no sense to have a mip-map of integer values, and we can't encode into BC formats (yet)
				if (asset::isIntegerFormat(format) || asset::isBlockCompressionFormat(format))
					return 0;
				// base mip level has data to use as source
				if (image->getRegions(0).empty())
					return 0;
				// any mip level is completely empty (mip levels with any data will NOT be recomputed) and has a mip level with data preceeding it
				uint16_t retval = 0;
				const uint16_t mipCount = patch.mipLevels;
				for (uint16_t l=1; l<mipCount; l++)
				if (image->getRegions(l).empty())
					retval |= 0x1u<<(l-1);
				return retval;
			}

			// Typed Range of Inputs of the same type
            template<asset::Asset AssetType>
            using asset_span_t = std::span<const typename asset_traits<AssetType>::asset_t* const>;
            template<asset::Asset AssetType>
            using patch_span_t = std::span<const patch_t<AssetType>>;

			// can be `nullptr` and even equal to `this`
			const CAssetConverter* readCache = nullptr;

			// recommended you set this
			system::logger_opt_ptr logger = nullptr;

			// A type-sorted non-polymorphic list of "root assets"
			core::tuple_transform_t<asset_span_t,supported_asset_types> assets = {};
			// Optional: Whatever is not in `patches` will generate a default patch
			core::tuple_transform_t<patch_span_t,supported_asset_types> patches = {};

			// optional, useful for shaders
			asset::IShaderCompiler::CCache* readShaderCache = nullptr;
			asset::IShaderCompiler::CCache* writeShaderCache = nullptr;
			IGPUPipelineCache* pipelineCache = nullptr;
        };
		// Split off from inputs because only assets that build on IPreHashed need uploading
		struct SConvertParams
		{
			// By default the last to queue to touch a GPU object will own it after any transfer or compute operations are complete.
			// If you want to record a pipeline barrier that will release ownership to another family, override this.
			// The overload for the IGPUBuffer may be called with a hash belonging to a Acceleration Structure, this means that its the storage buffer backing the AS
			virtual inline uint32_t getFinalOwnerQueueFamily(const IGPUBuffer* buffer, const core::blake3_hash_t& createdFrom)
			{
				return IQueue::FamilyIgnored;
			}
			virtual inline uint32_t getFinalOwnerQueueFamily(const IGPUImage* image, const core::blake3_hash_t& createdFrom, const uint8_t mipLevel)
			{
				return IQueue::FamilyIgnored;
			}
			// You can choose what layout the images get transitioned to at the end of an upload
			// (the images that don't get uploaded to can be transitioned from UNDEFINED without needing any work here)
			virtual inline IGPUImage::LAYOUT getFinalLayout(const IGPUImage* image, const core::blake3_hash_t& createdFrom, const uint8_t mipLevel)
			{
				using layout_t = IGPUImage::LAYOUT;
				using flags_t = IGPUImage::E_USAGE_FLAGS;
				const auto usages = image->getCreationParameters().usage;
				if (usages.hasFlags(flags_t::EUF_RENDER_ATTACHMENT_BIT) || usages.hasFlags(flags_t::EUF_TRANSIENT_ATTACHMENT_BIT))
					return layout_t::ATTACHMENT_OPTIMAL;
				if (usages.hasFlags(flags_t::EUF_SAMPLED_BIT) || usages.hasFlags(flags_t::EUF_INPUT_ATTACHMENT_BIT))
					return layout_t::READ_ONLY_OPTIMAL;
				// best guess
				return layout_t::GENERAL;
			}
			// By default we always insert into the cache
			virtual inline bool writeCache(const CCacheBase::key_t& createdFrom)
			{
				return true;
			}

			// One queue is for copies, another is for mip map generation and Acceleration Structure building
			// SCRATCH COMMAND BUFFERS MUST BE DIFFERENT (for submission/non-idling efficiency)
			SIntendedSubmitInfo* transfer = {};
			SIntendedSubmitInfo* compute = {};
			// required for Buffer or Image upload operations
			IUtilities* utilities = nullptr;
			// optional, last submit (compute, transfer if no compute needed) signals these in addition to the scratch semaphore
			std::span<const IQueue::SSubmitInfo::SSemaphoreInfo> extraSignalSemaphores = {};
			// specific to mip-map recomputation, these are okay defaults for the size of our Descriptor Indexed temporary descriptor set
			uint32_t sampledImageBindingCount = 1<<10;
			uint32_t storageImageBindingCount = 11<<10;
			// specific to Acceleration Structure Build, they need to be at least as large as the largest amount of scratch required for an AS build
			CAsyncSingleBufferSubAllocatorST</*using 32bit cause who uses 4GB of scratch for a build!?*/>* scratchForDeviceASBuild = nullptr;
			std::pmr::memory_resource* scratchForHostASBuild = nullptr;
			// needs to service allocations without limit, unlike the above where failure will just force a flush and performance of already queued up builds
			IDeviceMemoryAllocator* compactedASAllocator = nullptr;
			// How many extra threads you want to use for AS Builds
			uint16_t extraHostASBuildThreads = 0;
		};
        struct SReserveResult final
        {
				template<asset::Asset AssetType>
				using vector_t = core::vector<asset_cached_t<AssetType>>;

			public:
				template<asset::Asset AssetType>
				using staging_cache_t = core::unordered_map<typename asset_traits<AssetType>::video_t*,typename CCache<AssetType>::key_t>;

				inline SReserveResult(SReserveResult&&) = default;
				inline SReserveResult(const SReserveResult&) = delete;
				inline ~SReserveResult() = default;
				inline SReserveResult& operator=(const SReserveResult&) = delete;
				inline SReserveResult& operator=(SReserveResult&&) = default;

				// What queues you'll need to run the submit
				// WARNING: Uploading image region data for depth or stencil formats requires that the transfer queue has GRAPHICS capability!
				// https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkCmdCopyBufferToImage.html#VUID-vkCmdCopyBufferToImage-commandBuffer-07739
				inline core::bitflag<IQueue::FAMILY_FLAGS> getRequiredQueueFlags() const {return m_queueFlags;}

				// This is just enough memory to build the Acceleration Structures one by one waiting for each Device Build to complete inbetween. If 0 there are no Device AS Builds or Compactions to perform.
				inline uint64_t getMinASBuildScratchSize(const bool forHostOps) const
				{
					assert(m_minASBuildScratchSize[forHostOps]<=m_maxASBuildScratchSize[forHostOps]);
					return m_minASBuildScratchSize[forHostOps];
				}
				// Enough memory to build and compact all the Acceleration Structures at once, obviously respecting order of BLAS (build->compact) -> TLAS (build->compact)
				inline uint64_t getMaxASBuildScratchSize(const bool forHostOps) const
				{
					assert(m_minASBuildScratchSize[forHostOps]<=m_maxASBuildScratchSize[forHostOps]);
					return m_maxASBuildScratchSize[forHostOps];
				}
				// tells you if you need to provide a valid `SConvertParams::scratchForDeviceASBuild`
				inline bool willDeviceASBuild() const {return getMinASBuildScratchSize(false)>0;}
				// tells you if you need to provide a valid `SConvertParams::scratchForHostASBuild`
				inline bool willHostASBuild() const
				{
					const bool retval = getMinASBuildScratchSize(true)>0;
					assert(!retval); // host builds not supported yet
					return retval;
				}
				// tells you if you need to provide a valid `SConvertParams::compactedASAllocator`
				inline bool willCompactAS() const
				{
					assert(!m_willCompactSomeAS || willDeviceASBuild() || willHostASBuild());
					return m_willCompactSomeAS;
				}

				//
				inline operator bool() const {return bool(m_converter);}

				// Until `convert` is called, the Buffers and Images are not filled with content and Acceleration Structures are not built, unless found in the `SInput::readCache`
				// WARNING: The Acceleration Structure Pointer WILL CHANGE after calling `convert` if its patch dictates that it will be compacted! (since AS can't resize)
				// TODO: we could also return per-object semaphore values when object is ready for use (would have to propagate two semaphores up through dependants)
				template<asset::Asset AssetType>
				std::span<const asset_cached_t<AssetType>> getGPUObjects() const {return std::get<vector_t<AssetType>>(m_gpuObjects);}

				// If you ever need to look up the content hashes of the assets AT THE TIME you converted them
				// REMEMBER it can have stale hashes (asset or its dependants mutated since hash computed),
				// then you can get hash mismatches or plain wrong hashes.
				CHashCache* getHashCache() {return m_hashCache.get();}
				const CHashCache* getHashCache() const {return m_hashCache.get();}

				// useful for virtual function implementations in `SConvertParams`
				template<asset::Asset AssetType>
				const auto& getStagingCache() const {return std::get<staging_cache_t<AssetType>>(m_stagingCaches);}

				// You only get to call this once if successful, it submits right away (no potentially left-over commands in open scratch buffers) because Asset Conversion is meant to be a heavy-weight operation.
				// Leaving the final commands dangling in the `SIntendedSubmitInfo` members of `SConvertParams` creates fairly fragile and pessimistic scheduling (ensuring compute waits on transfer) and a complex API for the user. 
				// IMPORTANT: Barriers are NOT automatically issued AFTER the last command to touch a converted resource unless Queue Family Ownership needs to be released!
				// Therefore, unless you synchronise the submissions of future workloads using converted resources using semaphore wait on respective scratch or extra signal semaphores, YOU NEED TO RECORD THE PIPELINE BARRIERS YOURSELF!
				// **If there were QFOT Releases done, you need to record pipeline barriers with QFOT acquire yourself anyway!**
				// We only record pipeline barriers AFTER the last command if the image layout was meant to change.
				// TL;DR Syncrhonize access to converted contents with the retured semaphore signal value if in doubt.
				inline ISemaphore::future_t<IQueue::RESULT> convert(SConvertParams& params)
				{
					auto enqueueSuccess = m_converter->convert_impl(*this,params);
					// leveraging implementation details nastily, another way is `retval.blocking() || retval.copy()`
					if (reinterpret_cast<const IQueue::RESULT&>(enqueueSuccess)==IQueue::RESULT::SUCCESS)
					{
						// wipe after success
						core::for_each_in_tuple(m_stagingCaches,[](auto& stagingCache)->void{stagingCache.clear();});
						// disallow a double run
						m_converter = nullptr;
					}
					return enqueueSuccess;
				}

			private:
				friend class CAssetConverter;

				inline SReserveResult() = default;

				// we need to remember a few things so that `convert` can work seamlessly
				core::smart_refctd_ptr<CAssetConverter> m_converter = nullptr;
				system::logger_opt_smart_ptr m_logger = nullptr;

				//
				core::smart_refctd_ptr<CHashCache> m_hashCache = nullptr;

				// for every entry in the input array, we have this mapped 1:1
				core::tuple_transform_t<vector_t,supported_asset_types> m_gpuObjects = {};
				
				// we don't insert into the writeCache until conversions are successful
				core::tuple_transform_t<staging_cache_t,supported_asset_types> m_stagingCaches;
        // converted IShaders do not have any object that hold a smartptr into them, so we have to persist them in this vector to prevent m_stagingCacheds hold a raw dangling pointer into them
				core::vector<core::smart_refctd_ptr<asset::IShader>> m_shaders;

				// need a more explicit list of GPU objects that need device-assisted conversion
				template<asset::Asset AssetType>
				struct SConversionRequestBase
				{
					// canonical asset (the one that provides content)
					core::smart_refctd_ptr<const AssetType> canonical;
					// gpu object to transfer canonical's data to or build it from
					asset_traits<AssetType>::video_t* gpuObj;
				};
				using SConvReqBuffer = SConversionRequestBase<asset::ICPUBuffer>;
				core::vector<SConvReqBuffer> m_bufferConversions;
				struct SConvReqImage : SConversionRequestBase<asset::ICPUImage>
				{
					uint16_t recomputeMips = 0;
				};
				core::vector<SConvReqImage> m_imageConversions;
				template<typename CPUAccelerationStructure>// requires std::is_base_of_v<asset::ICPUAccelerationStructure,CPUAccelerationStructure>
				struct SConvReqAccelerationStructure : SConversionRequestBase<CPUAccelerationStructure>
				{
					constexpr static inline uint64_t WontCompact = (0x1ull<<48)-1;
					inline bool compact() const {return compactedASWriteOffset!=WontCompact;}

					using build_f = typename CPUAccelerationStructure::BUILD_FLAGS;
					inline void setBuildFlags(const build_f _flags) {buildFlags = static_cast<uint16_t>(_flags);}
					inline build_f getBuildFlags() const {return static_cast<build_f>(buildFlags);}


					uint64_t compactedASWriteOffset : 48 = WontCompact;
					uint64_t buildFlags : 16 = static_cast<uint16_t>(build_f::NONE);
				};
				core::vector<SConvReqAccelerationStructure<asset::ICPUBottomLevelAccelerationStructure>> m_blasConversions[2];
				core::vector<SConvReqAccelerationStructure<asset::ICPUTopLevelAccelerationStructure>> m_tlasConversions[2];

				// 0 for device builds, 1 for host builds
				uint64_t m_minASBuildScratchSize[2] = {0,0};
				uint64_t m_maxASBuildScratchSize[2] = {0,0};
				// We do all compactions on the Device for simplicity
				uint8_t m_willCompactSomeAS : 1 = false;

				//
				core::bitflag<IQueue::FAMILY_FLAGS> m_queueFlags = IQueue::FAMILY_FLAGS::NONE;
        };
		// First Pass: Explore the DAG of Assets and "gather" patch infos and create equivalent GPU Objects.
		NBL_API2 SReserveResult reserve(const SInputs& inputs);

		// Only const methods so others are not able to insert things made by e.g. different devices
		template<asset::Asset AssetType>
		inline const CCache<AssetType>& getCache() const
		{
			return std::get<CCache<AssetType>>(m_caches);
		}

		//
		inline void merge(const CAssetConverter* other)
		{
			std::apply([&](auto&... caches)->void{
				(..., caches.merge(std::get<std::remove_reference_t<decltype(caches)>>(other->m_caches)));
			},m_caches);
		}

    protected:
        inline CAssetConverter(const SCreationParams& params) : m_params(std::move(params)) {}
        virtual inline ~CAssetConverter() = default;
		
		template<asset::Asset AssetType>
		inline CCache<AssetType>& getCache()
		{
			return std::get<CCache<AssetType>>(m_caches);
		}

		friend struct SReserveResult;
		NBL_API2 ISemaphore::future_t<IQueue::RESULT> convert_impl(SReserveResult& reservations, SConvertParams& params);

        SCreationParams m_params;
		core::tuple_transform_t<CCache,supported_asset_types> m_caches;
};


// nothing to do
template<asset::Asset AssetType>
inline CAssetConverter::patch_impl_t<AssetType>::patch_impl_t(const AssetType* asset) {}
// always valid
template<asset::Asset AssetType>
inline bool CAssetConverter::patch_impl_t<AssetType>::valid(const ILogicalDevice* device) { return true; }

}
#endif