// Copyright (C) 2024-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
#include "nbl/video/utilities/CAssetConverter.h"

#include <type_traits>


using namespace nbl::core;
using namespace nbl::asset;


// if you end up specializing `patch_t` for any type because its non trivial and starts needing weird stuff done with memory, you need to spec this as well
namespace nbl
{
template<asset::Asset AssetType, typename Dummy>
struct core::blake3_hasher::update_impl<video::CAssetConverter::patch_t<AssetType>,Dummy>
{
	static inline void __call(blake3_hasher& hasher, const video::CAssetConverter::patch_t<AssetType>& input)
	{
		// empty classes are still sizeof==1
		if (std::is_empty_v<video::CAssetConverter::patch_impl_t<AssetType>>)
			return;
		hasher.update(&input,sizeof(input));
	}
};

namespace video
{
// No asset has a 0 length input to the hash function
const core::blake3_hash_t CAssetConverter::CHashCache::NoContentHash = static_cast<core::blake3_hash_t>(core::blake3_hasher());

CAssetConverter::patch_impl_t<ICPUSampler>::patch_impl_t(const ICPUSampler* sampler) : anisotropyLevelLog2(sampler->getParams().AnisotropicFilter) {}
bool CAssetConverter::patch_impl_t<ICPUSampler>::valid(const ILogicalDevice* device)
{
	if (anisotropyLevelLog2>5) // unititialized
		return false;
	const auto& limits = device->getPhysicalDevice()->getLimits();
	if (anisotropyLevelLog2>limits.maxSamplerAnisotropyLog2)
		anisotropyLevelLog2 = limits.maxSamplerAnisotropyLog2;
	return true;
}


CAssetConverter::patch_impl_t<ICPUShader>::patch_impl_t(const ICPUShader* shader) : stage(shader->getStage()) {}
bool CAssetConverter::patch_impl_t<ICPUShader>::valid(const ILogicalDevice* device)
{
	const auto& features = device->getEnabledFeatures();
	switch (stage)
	{
		// supported always
		case IGPUShader::E_SHADER_STAGE::ESS_VERTEX:
		case IGPUShader::E_SHADER_STAGE::ESS_FRAGMENT:
		case IGPUShader::E_SHADER_STAGE::ESS_COMPUTE:
			return true;
		case IGPUShader::E_SHADER_STAGE::ESS_TESSELLATION_CONTROL:
		case IGPUShader::E_SHADER_STAGE::ESS_TESSELLATION_EVALUATION:
			if (features.tessellationShader)
				return true;
			break;
		case IGPUShader::E_SHADER_STAGE::ESS_GEOMETRY:
			if (features.geometryShader)
				return true;
			break;
		case IGPUShader::E_SHADER_STAGE::ESS_TASK:
//			if (features.taskShader)
//				return true;
			break;
		case IGPUShader::E_SHADER_STAGE::ESS_MESH:
//			if (features.meshShader)
//				return true;
			break;
		case IGPUShader::E_SHADER_STAGE::ESS_RAYGEN:
		case IGPUShader::E_SHADER_STAGE::ESS_ANY_HIT:
		case IGPUShader::E_SHADER_STAGE::ESS_CLOSEST_HIT:
		case IGPUShader::E_SHADER_STAGE::ESS_MISS:
		case IGPUShader::E_SHADER_STAGE::ESS_INTERSECTION:
		case IGPUShader::E_SHADER_STAGE::ESS_CALLABLE:
			if (features.rayTracingPipeline)
				return true;
			break;
		default:
			break;
	}
	return false;
}

CAssetConverter::patch_impl_t<ICPUBuffer>::patch_impl_t(const ICPUBuffer* buffer) : usage(buffer->getUsageFlags()) {}
bool CAssetConverter::patch_impl_t<ICPUBuffer>::valid(const ILogicalDevice* device)
{
	const auto& features = device->getEnabledFeatures();
	if (usage.hasFlags(usage_flags_t::EUF_CONDITIONAL_RENDERING_BIT_EXT) && !features.conditionalRendering)
		return false;
	if ((usage.hasFlags(usage_flags_t::EUF_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT)||usage.hasFlags(usage_flags_t::EUF_ACCELERATION_STRUCTURE_STORAGE_BIT)) && !features.accelerationStructure)
		return false;
	if (usage.hasFlags(usage_flags_t::EUF_SHADER_BINDING_TABLE_BIT) && !features.rayTracingPipeline)
		return false;
	// good default
	usage |= usage_flags_t::EUF_INLINE_UPDATE_VIA_CMDBUF;
	return true;
}

bool CAssetConverter::acceleration_structure_patch_base::valid(const ILogicalDevice* device)
{
	// note that we don't check the validity of things we don't patch, all the instance and geometry data, but it will be checked by the driver anyway during creation/build
	if (preference==BuildPreference::Invalid) // unititialized or just wrong
		return false;
	// potentially skip a lot of work and allocations
	const auto& features = device->getEnabledFeatures();
	if (!features.accelerationStructure)
		return false;
	// 
	if (isMotion && !features.rayTracingMotionBlur)
		return false;
	// just make the flags agree/canonicalize
	allowCompaction = allowCompaction || compactAfterBuild;
	// can always build with the device
	if (hostBuild)
#ifdef NBL_ACCELERATION_STRUCTURE_CONVERSION_HOST_READY
	if (!features.accelerationStructureHostCommands)
#endif
	{
		if (auto logger=device->getLogger();logger)
			logger->log("Host Acceleration Structure Builds are not yet supported!",system::ILogger::ELL_ERROR);
		hostBuild = false;
	}
	return true;
}
CAssetConverter::patch_impl_t<ICPUBottomLevelAccelerationStructure>::patch_impl_t(const ICPUBottomLevelAccelerationStructure* blas)
{
	const auto flags = blas->getBuildFlags();
	// straight up invalid
	if (flags.hasFlags(build_flags_t::PREFER_FAST_TRACE_BIT|build_flags_t::PREFER_FAST_BUILD_BIT))
		return;

	isMotion = blas->usesMotion();
	allowUpdate = flags.hasFlags(build_flags_t::ALLOW_UPDATE_BIT);
	allowCompaction = flags.hasFlags(build_flags_t::ALLOW_COMPACTION_BIT);
	if (flags.hasFlags(build_flags_t::PREFER_FAST_TRACE_BIT))
		preference = BuildPreference::FastTrace;
	else if (flags.hasFlags(build_flags_t::PREFER_FAST_BUILD_BIT))
		preference = BuildPreference::FastBuild;
	else
		preference = BuildPreference::None;
	lowMemory = flags.hasFlags(build_flags_t::LOW_MEMORY_BIT);
	allowDataAccess = flags.hasFlags(build_flags_t::ALLOW_DATA_ACCESS);
}
auto CAssetConverter::patch_impl_t<ICPUBottomLevelAccelerationStructure>::getBuildFlags(const ICPUBottomLevelAccelerationStructure* blas) const -> core::bitflag<build_flags_t>
{
	constexpr build_flags_t OverridableMask = build_flags_t::LOW_MEMORY_BIT|build_flags_t::PREFER_FAST_TRACE_BIT|build_flags_t::PREFER_FAST_BUILD_BIT|build_flags_t::ALLOW_COMPACTION_BIT|build_flags_t::ALLOW_UPDATE_BIT|build_flags_t::ALLOW_DATA_ACCESS;
	auto flags = blas->getBuildFlags()&(~OverridableMask);
	if (lowMemory)
		flags |= build_flags_t::LOW_MEMORY_BIT;
	if (allowDataAccess)
		flags |= build_flags_t::ALLOW_DATA_ACCESS;
	if (allowCompaction)
		flags |= build_flags_t::ALLOW_COMPACTION_BIT;
	if (allowUpdate)
		flags |= build_flags_t::ALLOW_UPDATE_BIT;
	switch (preference)
	{
		case acceleration_structure_patch_base::BuildPreference::FastTrace:
			flags |= build_flags_t::PREFER_FAST_TRACE_BIT;
			break;
		case acceleration_structure_patch_base::BuildPreference::FastBuild:
			flags |= build_flags_t::PREFER_FAST_BUILD_BIT;
			break;
		default:
			break;
	}
	return flags;
}
bool CAssetConverter::patch_impl_t<ICPUBottomLevelAccelerationStructure>::valid(const ILogicalDevice* device)
{
	// on a second thought, if someone asked for BLAS with data access, they probably intend to use it
	const auto& limits = device->getPhysicalDevice()->getLimits();
	if (allowDataAccess && !limits.rayTracingPositionFetch)
		return false;
	return acceleration_structure_patch_base::valid(device);
}
CAssetConverter::patch_impl_t<ICPUTopLevelAccelerationStructure>::patch_impl_t(const ICPUTopLevelAccelerationStructure* tlas)
{
	const auto flags = tlas->getBuildFlags();
	// straight up invalid
	if (tlas->getInstances().empty())
		return;
	if (flags.hasFlags(build_flags_t::PREFER_FAST_TRACE_BIT|build_flags_t::PREFER_FAST_BUILD_BIT))
		return;

	isMotion = tlas->usesMotion();
	allowUpdate = flags.hasFlags(build_flags_t::ALLOW_UPDATE_BIT);
	allowCompaction = flags.hasFlags(build_flags_t::ALLOW_COMPACTION_BIT);
	if (flags.hasFlags(build_flags_t::PREFER_FAST_TRACE_BIT))
		preference = BuildPreference::FastTrace;
	else if (flags.hasFlags(build_flags_t::PREFER_FAST_BUILD_BIT))
		preference = BuildPreference::FastBuild;
	else
		preference = BuildPreference::None;
	lowMemory = flags.hasFlags(build_flags_t::LOW_MEMORY_BIT);
	maxInstances = tlas->getInstances().size();
}
auto CAssetConverter::patch_impl_t<ICPUTopLevelAccelerationStructure>::getBuildFlags(const ICPUTopLevelAccelerationStructure* tlas) const -> core::bitflag<build_flags_t>
{
	constexpr build_flags_t OverridableMask = build_flags_t::LOW_MEMORY_BIT|build_flags_t::PREFER_FAST_TRACE_BIT|build_flags_t::PREFER_FAST_BUILD_BIT|build_flags_t::ALLOW_COMPACTION_BIT|build_flags_t::ALLOW_UPDATE_BIT;
	auto flags = tlas->getBuildFlags()&(~OverridableMask);
	if (lowMemory)
		flags |= build_flags_t::LOW_MEMORY_BIT;
	if (allowCompaction)
		flags |= build_flags_t::ALLOW_COMPACTION_BIT;
	if (allowUpdate)
		flags |= build_flags_t::ALLOW_UPDATE_BIT;
	switch (preference)
	{
		case acceleration_structure_patch_base::BuildPreference::FastTrace:
			flags |= build_flags_t::PREFER_FAST_TRACE_BIT;
			break;
		case acceleration_structure_patch_base::BuildPreference::FastBuild:
			flags |= build_flags_t::PREFER_FAST_BUILD_BIT;
			break;
		default:
			break;
	}
	return flags;
}
bool CAssetConverter::patch_impl_t<ICPUTopLevelAccelerationStructure>::valid(const ILogicalDevice* device)
{
	return acceleration_structure_patch_base::valid(device);
}

// smol utility function
template<typename Patch>
void deduceMetaUsages(Patch& patch, const core::bitflag<IGPUImage::E_USAGE_FLAGS> usages, const E_FORMAT originalFormat, const bool hasDepthAspect=true)
{
	// Impossible to deduce this without knowing all the sampler & view combos that will be used,
	// because under descriptor indexing we can use any sampler with any view in SPIR-V.
	// Also because we don't know what Descriptor Sets will be used with what Pipelines!
	if (usages.hasFlags(IGPUImage::E_USAGE_FLAGS::EUF_SAMPLED_BIT))
	{
		if (isDepthOrStencilFormat(originalFormat))
		{
			if (hasDepthAspect)
				patch.depthCompareSampledImage = true;
		}
		// Also have no info about any blit cmd that could use this view's image as source
		else if (!asset::isIntegerFormat(originalFormat))
			patch.linearlySampled = true;
	}
	// same stuff for storage images
	if (usages.hasFlags(IGPUImage::E_USAGE_FLAGS::EUF_STORAGE_BIT))
	{
		// Deducing this in another way would seriously hinder our ability to do format promotions.
		// To ensure the view doesn't get promoted away from device-feature dependant atomic storage capable formats, use explicit patches upon input! 
		patch.storageAtomic = originalFormat==EF_R32_UINT;
	}
}

CAssetConverter::patch_impl_t<ICPUImage>::patch_impl_t(const ICPUImage* image)
{
	const auto& params = image->getCreationParameters();
	format = params.format;
	usageFlags = params.usage;
	if (isDepthOrStencilFormat(format))
	{
		const bool hasStencil = !isDepthOnlyFormat(format);
		if (hasStencil)
			stencilUsage = params.actualStencilUsage();
		const bool hasDepth = !isStencilOnlyFormat(format);
		bool foundDepthReg = !hasDepth;
		bool foundStencilReg = !hasStencil;
		for (const auto& region : image->getRegions())
		{
			foundDepthReg = foundDepthReg || region.imageSubresource.aspectMask.hasFlags(IGPUImage::EAF_DEPTH_BIT);
			foundStencilReg = foundStencilReg || region.imageSubresource.aspectMask.hasFlags(IGPUImage::EAF_STENCIL_BIT);
			if (foundDepthReg && foundStencilReg)
				break;
		}
	}
	else if(!image->getRegions().empty())
		usageFlags |= usage_flags_t::EUF_TRANSFER_DST_BIT;
	//
	using create_flags_t = IGPUImage::E_CREATE_FLAGS;
	mutableFormat = params.flags.hasFlags(create_flags_t::ECF_MUTABLE_FORMAT_BIT);
	cubeCompatible = params.flags.hasFlags(create_flags_t::ECF_CUBE_COMPATIBLE_BIT);
	_3Dbut2DArrayCompatible = params.flags.hasFlags(create_flags_t::ECF_2D_ARRAY_COMPATIBLE_BIT);
	uncompressedViewOfCompressed = params.flags.hasFlags(create_flags_t::ECF_BLOCK_TEXEL_VIEW_COMPATIBLE_BIT);
	// meta usages only matter for promotion
	if (canAttemptFormatPromotion())
		deduceMetaUsages(*this,usageFlags|stencilUsage,format);
}
bool CAssetConverter::patch_impl_t<ICPUImage>::valid(const ILogicalDevice* device)
{
	const auto& features = device->getEnabledFeatures();
	// usages we don't have features for
	if (usageFlags.hasFlags(usage_flags_t::EUF_SHADING_RATE_ATTACHMENT_BIT))// && !features.shadingRate)
		return false;
	if (usageFlags.hasFlags(usage_flags_t::EUF_FRAGMENT_DENSITY_MAP_BIT) && !features.fragmentDensityMap)
		return false;
	const auto* physDev = device->getPhysicalDevice();
	if (storageImageLoadWithoutFormat && !physDev->getLimits().shaderStorageImageReadWithoutFormat)
		return false;
	// We're not going to check if the format is creatable for a given usage nad metausages, because we possibly haven't collected all the usages yet.
	// So the Image format promotion happens in another pass, just after the DFS descent.
	return true;
}

CAssetConverter::patch_impl_t<ICPUBufferView>::patch_impl_t(const ICPUBufferView* view) {}
bool CAssetConverter::patch_impl_t<ICPUBufferView>::valid(const ILogicalDevice* device)
{
	// note that we don't check the validity of things we don't patch, so offset alignment, size and format
	// we could check if the format and usage make sense, but it will be checked by the driver anyway
	return true;
}

CAssetConverter::patch_impl_t<ICPUImageView>::patch_impl_t(const ICPUImageView* view, const core::bitflag<usage_flags_t> extraSubUsages) :
	subUsages(view->getCreationParameters().actualUsages()|extraSubUsages)
{
	const auto& params = view->getCreationParameters();
	originalFormat = params.format;
	if (originalFormat==params.image->getCreationParameters().format)
	{
		// meta usages only matter for promotion (because only non-mutable format/non-aliased views can be promoted)
		// we only promote if format is the same as the base image and we are are not using it for renderpass attachments
		if (!subUsages.hasAnyFlag(patch_impl_t<ICPUImage>::UsagesThatPreventFormatPromotion))
		{
			deduceMetaUsages(*this,subUsages,originalFormat,params.subresourceRange.aspectMask.hasFlags(IGPUImage::E_ASPECT_FLAGS::EAF_DEPTH_BIT));
			// allow to format to mutate with base image's
			originalFormat = EF_UNKNOWN;
		}
	}
}
bool CAssetConverter::patch_impl_t<ICPUImageView>::valid(const ILogicalDevice* device)
{
	invalid = true;
	// check exotic usages against device caps
	const auto& features = device->getEnabledFeatures();
	if (subUsages.hasFlags(usage_flags_t::EUF_SHADING_RATE_ATTACHMENT_BIT))// && !features.fragmentDensityMap)
		return false;
	if (subUsages.hasFlags(usage_flags_t::EUF_FRAGMENT_DENSITY_MAP_BIT) && !features.fragmentDensityMap)
		return false;
	const auto* physDev = device->getPhysicalDevice();
	if (storageImageLoadWithoutFormat && !physDev->getLimits().shaderStorageImageReadWithoutFormat)
		return false;
	// now check for the format (if the format is immutable)
	if (originalFormat!=EF_UNKNOWN)
	{
		// normally we wouldn't check for usages being valid, but combine needs to know about validity before combining and producing a wider set of usages
		// we cull bad instances instead (uses of the view), it wont catch 100% of cases though!
		IPhysicalDevice::SFormatImageUsages::SUsage usages = {subUsages};
		usages.linearlySampledImage = linearlySampled;
		if (usages.storageImage) // we require this anyway
			usages.storageImageStoreWithoutFormat = true;
		usages.storageImageAtomic = storageAtomic;
		usages.storageImageLoadWithoutFormat = storageImageLoadWithoutFormat;
		usages.depthCompareSampledImage = depthCompareSampledImage;
		invalid = (physDev->getImageFormatUsagesOptimalTiling()[originalFormat]&usages)!=usages || (physDev->getImageFormatUsagesLinearTiling()[originalFormat]&usages)!=usages;
	}
	else
		invalid = false;
	return !invalid;
}

CAssetConverter::patch_impl_t<ICPUPipelineLayout>::patch_impl_t(const ICPUPipelineLayout* pplnLayout) : patch_impl_t()
{
	const auto pc = pplnLayout->getPushConstantRanges();
	for (auto it=pc.begin(); it!=pc.end(); it++)
	if (it->stageFlags!=shader_stage_t::ESS_UNKNOWN)
	{
		if (it->offset>=pushConstantBytes.size())
			return;
		const auto end = it->offset+it->size;
		if (end<it->offset || end>pushConstantBytes.size())
			return;
		for (auto byte=it->offset; byte<end; byte++)
			pushConstantBytes[byte] = it->stageFlags;
	}
	invalid = false;
}
bool CAssetConverter::patch_impl_t<ICPUPipelineLayout>::valid(const ILogicalDevice* device)
{
	const auto& limits = device->getPhysicalDevice()->getLimits();
	for (auto byte=limits.maxPushConstantsSize; byte<pushConstantBytes.size(); byte++)
	if (pushConstantBytes[byte]!=shader_stage_t::ESS_UNKNOWN)
		return false;
	return !invalid;
}

// not sure if useful enough to move to core utils
template<typename T, typename TypeList>
struct index_of;
template<typename T>
struct index_of<T,core::type_list<>> : std::integral_constant<size_t,0> {};
template<typename T, typename... Us>
struct index_of<T,core::type_list<T,Us...>> : std::integral_constant<size_t,0> {};
template<typename T, typename U, typename... Us>
struct index_of<T,core::type_list<U,Us...>> : std::integral_constant<size_t,1+index_of<T,core::type_list<Us...>>::value> {};
template<typename T, typename TypeList>
inline constexpr size_t index_of_v = index_of<T,TypeList>::value;


//
template<Asset AssetType>
struct instance_t
{
	inline operator instance_t<IAsset>() const
	{
		return instance_t<IAsset>{.asset=asset,.uniqueCopyGroupID=uniqueCopyGroupID};
	}

	inline bool operator==(const instance_t<AssetType>&) const = default;

	//
	const AssetType* asset = nullptr;
	size_t uniqueCopyGroupID = 0xdeadbeefBADC0FFEull;
};

//
template<typename CRTP>
class AssetVisitor : public CRTP
{
	public:
		using AssetType = CRTP::AssetType;

		bool operator()()
		{
			if (!instance.asset)
				return false;
			return impl(instance,patch);
		}

		const instance_t<AssetType> instance;
		const CAssetConverter::patch_t<AssetType>& patch;

	protected:
		template<Asset DepType, typename... ExtraArgs>
		bool descend(const DepType* dep, CAssetConverter::patch_t<DepType>&& candidatePatch, ExtraArgs&&... extraArgs)
		{
			assert(dep);
			return bool(
				CRTP::descend_impl(
					instance,patch,
					{dep,CRTP::getDependantUniqueCopyGroupID(instance.uniqueCopyGroupID,instance.asset,dep)},
					std::move(candidatePatch),
					std::forward<ExtraArgs>(extraArgs)...
				)
			);
		}

	private:
		// there is no `impl()` overload taking `ICPUBottomLevelAccelerationStructure` same as there is no `ICPUmage`
		inline bool impl(const instance_t<ICPUTopLevelAccelerationStructure>& instance, const CAssetConverter::patch_t<ICPUTopLevelAccelerationStructure>& userPatch)
		{
			const auto blasInstances = instance.asset->getInstances();
			if (blasInstances.empty())
				return false;
			for (size_t i=0; i<blasInstances.size(); i++)
			{
				const auto* blas = blasInstances[i].getBase().blas.get();
				// TODO: can one disable instances during builds?
				if (!blas)
					return false;
				CAssetConverter::patch_t<ICPUBottomLevelAccelerationStructure> patch = {blas};
				if (!descend(blas,std::move(patch),i))
					return false;
			}
			return true;
		}
		inline bool impl(const instance_t<ICPUBufferView>& instance, const CAssetConverter::patch_t<ICPUBufferView>& userPatch)
		{
			const auto* dep = instance.asset->getUnderlyingBuffer();
			if (!dep)
				return false;
			CAssetConverter::patch_t<ICPUBuffer> patch = {dep};
			if (userPatch.utbo)
				patch.usage |= IGPUBuffer::E_USAGE_FLAGS::EUF_UNIFORM_TEXEL_BUFFER_BIT;
			if (userPatch.stbo)
				patch.usage |= IGPUBuffer::E_USAGE_FLAGS::EUF_STORAGE_TEXEL_BUFFER_BIT;
			return descend<ICPUBuffer>(dep,std::move(patch));
		}
		inline bool impl(const instance_t<ICPUImageView>& instance, const CAssetConverter::patch_t<ICPUImageView>& userPatch)
		{
			const auto& params = instance.asset->getCreationParameters();
			const auto* dep = params.image.get();
			if (!dep)
				return false;
			CAssetConverter::patch_t<ICPUImage> patch = { dep };
			// any other aspects than stencil?
			if (params.subresourceRange.aspectMask.value&(~IGPUImage::E_ASPECT_FLAGS::EAF_STENCIL_BIT))
				patch.usageFlags |= userPatch.subUsages;
			// stencil aspect?
			if (params.subresourceRange.aspectMask.hasFlags(IGPUImage::E_ASPECT_FLAGS::EAF_STENCIL_BIT))
				patch.stencilUsage |= userPatch.subUsages;
			// view format doesn't mutate with image and format was actually different than base image
			// NOTE: `valid()` hasn't been called on `patch` yet, so format not promoted yet!
			if (!userPatch.formatFollowsImage() && params.format!=patch.format)
			{
				patch.mutableFormat = true;
				if (isBlockCompressionFormat(patch.format) && getFormatClass(params.format)!=getFormatClass(patch.format))
					patch.uncompressedViewOfCompressed = true;
			}
			// rest of create flags
			switch (params.viewType)
			{
				case IGPUImageView::E_TYPE::ET_CUBE_MAP:
					[[fallthrough]];
				case IGPUImageView::E_TYPE::ET_CUBE_MAP_ARRAY:
					patch.cubeCompatible = true;
					break;
				case IGPUImageView::E_TYPE::ET_2D:
					[[fallthrough]];
				case IGPUImageView::E_TYPE::ET_2D_ARRAY:
					if (dep->getCreationParameters().type==IImage::E_TYPE::ET_3D)
						patch._3Dbut2DArrayCompatible = true;
					break;
				default:
					break;
			}
			//
			patch.linearlySampled |= userPatch.linearlySampled;
			patch.storageAtomic |= userPatch.storageAtomic;
			patch.storageImageLoadWithoutFormat |= userPatch.storageImageLoadWithoutFormat;
			patch.depthCompareSampledImage |= userPatch.depthCompareSampledImage;
			// decision about whether to extend mipchain depends on format promotion (away from Block Compressed) so done in separate pass after DFS
			return descend<ICPUImage>(dep,std::move(patch));
		}
		inline bool impl(const instance_t<ICPUDescriptorSetLayout>& instance, const CAssetConverter::patch_t<ICPUDescriptorSetLayout>& userPatch)
		{
			const auto samplers = instance.asset->getImmutableSamplers();
			for (size_t i=0; i<samplers.size(); i++)
			{
				const auto sampler = samplers[i].get();
				if (!sampler || !descend(sampler,{sampler},i))
					return false;
			}
			return true;
		}
		inline bool impl(const instance_t<ICPUPipelineLayout>& instance, const CAssetConverter::patch_t<ICPUPipelineLayout>& userPatch)
		{
			// individual DS layouts are optional
			for (auto i=0; i<ICPUPipelineLayout::DESCRIPTOR_SET_COUNT; i++)
			{
				if (auto layout=instance.asset->getDescriptorSetLayout(i); layout)
				{
					if (!descend(layout,{layout},i))
						return false;
				}
				else
					CRTP::template nullOptional<ICPUDescriptorSetLayout>();
			}
			return true;
		}
		inline bool impl(const instance_t<ICPUComputePipeline>& instance, const CAssetConverter::patch_t<ICPUComputePipeline>& userPatch)
		{
			const auto* asset = instance.asset;
			const auto* layout = asset->getLayout();
			if (!layout || !descend(layout,{layout}))
				return false;
			const auto& specInfo = asset->getSpecInfo();
			const auto* shader = specInfo.shader;
			if (!shader)
				return false;
			CAssetConverter::patch_t<ICPUShader> patch = {shader};
			constexpr auto stage = IGPUShader::E_SHADER_STAGE::ESS_COMPUTE;
			patch.stage = stage;
			if (!descend(shader,std::move(patch),stage,specInfo))
				return false;
			return true;
		}
		inline bool impl(const instance_t<ICPUGraphicsPipeline>& instance, const CAssetConverter::patch_t<ICPUGraphicsPipeline>& userPatch)
		{
			const auto* asset = instance.asset;
			const auto* layout = asset->getLayout();
			if (!layout || !descend(layout,{layout}))
				return false;
			const auto* rpass = asset->getRenderpass();
			if (!rpass || !descend(rpass,{rpass}))
				return false;
			using stage_t = ICPUShader::E_SHADER_STAGE;
			for (stage_t stage : {stage_t::ESS_VERTEX,stage_t::ESS_TESSELLATION_CONTROL,stage_t::ESS_TESSELLATION_EVALUATION,stage_t::ESS_GEOMETRY,stage_t::ESS_FRAGMENT})
			{
				const auto& specInfo = asset->getSpecInfo(stage);
				const auto* shader = specInfo.shader;
				if (!shader)
				{
					if (stage==stage_t::ESS_VERTEX) // required
						return false;
					CRTP::template nullOptional<ICPUShader>();
					continue;
				}
				CAssetConverter::patch_t<ICPUShader> patch = {shader};
				patch.stage = stage;
				if (!descend(shader,std::move(patch),stage,specInfo))
					return false;
			}
			return true;
		}
		inline bool impl(const instance_t<ICPUDescriptorSet>& instance, const CAssetConverter::patch_t<ICPUDescriptorSet>& userPatch)
		{
			const auto* asset = instance.asset;
			const auto* layout = asset->getLayout();
			if (!layout || !descend(layout,{layout}))
				return false;
			for (auto i=0u; i<static_cast<uint32_t>(IDescriptor::E_TYPE::ET_COUNT); i++)
			{
				const auto type = static_cast<IDescriptor::E_TYPE>(i);
				const auto allInfos = asset->getDescriptorInfoStorage(type);
				if (allInfos.empty())
					continue;
				const auto& redirect = layout->getDescriptorRedirect(type);
				const auto bindingCount = redirect.getBindingCount();
				// go over every binding
				for (auto j=0; j<bindingCount; j++)
				{
					const IDescriptorSetLayoutBase::CBindingRedirect::storage_range_index_t storageRangeIx(j);
					const auto binding = redirect.getBinding(storageRangeIx);
					const uint32_t count = redirect.getCount(storageRangeIx);
					// this is where the descriptors have their flattened place in a unified array
					const auto storageBaseOffset = redirect.getStorageOffset(storageRangeIx);
					const auto* infos = allInfos.data()+storageBaseOffset.data;
					for (uint32_t el=0u; el<count; el++)
					{
						const auto& info = infos[el];
						if (auto untypedDesc=info.desc.get(); untypedDesc) // written descriptors are optional
						switch (IDescriptor::GetTypeCategory(type))
						{
							case IDescriptor::EC_BUFFER:
							{
								auto buffer = static_cast<const ICPUBuffer*>(untypedDesc);
								CAssetConverter::patch_t<ICPUBuffer> patch = {buffer};
								switch(type)
								{
									case IDescriptor::E_TYPE::ET_UNIFORM_BUFFER:
									case IDescriptor::E_TYPE::ET_UNIFORM_BUFFER_DYNAMIC:
										patch.usage |= IGPUBuffer::E_USAGE_FLAGS::EUF_UNIFORM_BUFFER_BIT;
										break;
									case IDescriptor::E_TYPE::ET_STORAGE_BUFFER:
									case IDescriptor::E_TYPE::ET_STORAGE_BUFFER_DYNAMIC:
										patch.usage |= IGPUBuffer::E_USAGE_FLAGS::EUF_STORAGE_BUFFER_BIT;
										break;
									default:
										assert(false);
										return false;
								}
								if (!descend(buffer,std::move(patch),type,binding,el,info.info.buffer))
									return false;
								break;
							}
							case IDescriptor::EC_SAMPLER:
							{
								auto sampler = static_cast<const ICPUSampler*>(untypedDesc);
								if (!descend(sampler,{sampler},type,binding,el))
									return false;
								break;
							}
							case IDescriptor::EC_IMAGE:
							{
								auto imageView = static_cast<const ICPUImageView*>(untypedDesc);
								IGPUImage::E_USAGE_FLAGS usage = IGPUImage::E_USAGE_FLAGS::EUF_NONE; // silence a warning
								switch (type)
								{
									case IDescriptor::E_TYPE::ET_COMBINED_IMAGE_SAMPLER:
										{
											const auto* sampler = info.info.combinedImageSampler.sampler.get();
											if (!descend(sampler,{sampler},type,binding,el))
												return false;
										}
										[[fallthrough]];
									case IDescriptor::E_TYPE::ET_SAMPLED_IMAGE:
										usage = IGPUImage::E_USAGE_FLAGS::EUF_SAMPLED_BIT;
										break;
									case IDescriptor::E_TYPE::ET_STORAGE_IMAGE:
										usage = IGPUImage::E_USAGE_FLAGS::EUF_STORAGE_BIT;
										break;
									case IDescriptor::E_TYPE::ET_INPUT_ATTACHMENT:
										usage = IGPUImage::E_USAGE_FLAGS::EUF_INPUT_ATTACHMENT_BIT;
										break;
									default:
										assert(false);
										break;
								}
								if (!descend(imageView,{imageView,usage},type,binding,el,info.info.image.imageLayout))
									return false;
								break;
							}
							case IDescriptor::EC_BUFFER_VIEW:
							{
								auto bufferView = static_cast<const ICPUBufferView*>(untypedDesc);
								CAssetConverter::patch_t<ICPUBufferView> patch = {bufferView};
								switch (type)
								{
									case IDescriptor::E_TYPE::ET_UNIFORM_TEXEL_BUFFER:
										patch.utbo = true;
										break;
									case IDescriptor::E_TYPE::ET_STORAGE_TEXEL_BUFFER:
										patch.stbo = true;
										break;
									default:
										assert(false);
										return false;
								}
								if (!descend(bufferView,std::move(patch),type,binding,el))
									return false;
								break;
							}
							case IDescriptor::EC_ACCELERATION_STRUCTURE:
							{
								auto tlas = static_cast<const ICPUTopLevelAccelerationStructure*>(untypedDesc);
								if (!descend(tlas,{tlas},type,binding,el,storageBaseOffset))
									return false;
								break;
							}
							default:
								assert(false);
								return false;
						}
						CRTP::template nullOptional<IDescriptor>();
					}
				}
			}
			return true;
		}
};


//
struct patch_index_t
{
	inline bool operator==(const patch_index_t&) const = default;
	inline bool operator!=(const patch_index_t&) const = default;

	explicit inline operator bool() const {return operator!=({});}

	uint64_t value = 0xdeadbeefBADC0FFEull;
};
// This cache stops us traversing an asset with the same user group and patch more than once.
template<asset::Asset AssetType>
class dfs_cache
{
	public:
		// Maps `instance_t` to `patchIndex`, makes sure the find can handle polymorphism of assets
		struct HashEquals
		{
			using is_transparent = void;

			inline size_t operator()(const instance_t<IAsset>& entry) const
			{
				return ptrdiff_t(entry.asset)^entry.uniqueCopyGroupID;
			}

			inline size_t operator()(const instance_t<AssetType>& entry) const
			{
				// its very important to cast the derived AssetType to IAsset because otherwise pointers won't match
				return operator()(instance_t<IAsset>(entry));
			}
	
			inline bool operator()(const instance_t<IAsset>& lhs, const instance_t<AssetType>& rhs) const
			{
				return lhs==instance_t<IAsset>(rhs);
			}
			inline bool operator()(const instance_t<AssetType>& lhs, const instance_t<IAsset>& rhs) const
			{
				return instance_t<IAsset>(lhs)==rhs;
			}
			inline bool operator()(const instance_t<AssetType>& lhs, const instance_t<AssetType>& rhs) const
			{
				return instance_t<IAsset>(lhs)==instance_t<IAsset>(rhs);
			}
		};
		using key_map_t = core::unordered_map<instance_t<AssetType>,patch_index_t,HashEquals,HashEquals>;

	private:
		// Implement both finding the first node for an instance with a compatible patch, and merging patches plus inserting new nodes if nothing can be merged
		template<bool ReplaceWithCombined>
		inline patch_index_t* impl(const instance_t<AssetType>& instance, const CAssetConverter::patch_t<AssetType>& soloPatch)
		{
			// get all the existing patches for the same (AssetType*,UniqueCopyGroupID)
			auto found = instances.find(instance);
			if (found==instances.end())
				return nullptr;
			// we don't want to pass the device to this function, it just assumes the patch will be valid without touch-ups
			//assert(soloPatch.valid(device));
			patch_index_t* pIndex = &found->second;
			// iterate over linked list
			while (*pIndex)
			{
				// get our candidate
				auto& candidate = nodes[pIndex->value];
				// found a thing, try-combine the patches
				auto [success,combined] = candidate.patch.combine(soloPatch);
				if (success)
				{
					if constexpr (ReplaceWithCombined)
					{
						// change the patch to a combined version
						candidate.patch = std::move(combined);
						break;
					}
					else
					{
						// to truly "find" a patched entry, the candidate needs to equal combined
						if (candidate.patch==combined)
							break;
					}
				}
				// else try the next one
				pIndex = &candidate.next;
			}
			return pIndex;
		}

	public:
		// Find the first node for an instance with a compatible patch and return its index
		inline std::pair<patch_index_t,bool> insert(const instance_t<AssetType>& instance, CAssetConverter::patch_t<AssetType>&& soloPatch)
		{
			const patch_index_t newPatchIndex = {nodes.size()};
			auto pIndex = impl<true>(instance,soloPatch);
			// found a linked list for this entry
			if (pIndex)
			{
				// found a patch and merged it
				if (*pIndex)
					return {*pIndex,false};
				// nothing mergable found, make old TAIL point to new node about to be inserted
				*pIndex = newPatchIndex;
			}
			else // there isn't even a linked list head for this entry
			{
				// phmap's unordered map doesn't even care about the found hint
				instances.emplace(instance,newPatchIndex);
			}
			// both non-mergable and completely not found cases fall through here
			nodes.emplace_back(std::move(soloPatch),CAssetConverter::CHashCache::NoContentHash);
			return {newPatchIndex,true};
		}
		// Find the first node for an instance with a compatible patch and return its index
		inline patch_index_t find(const instance_t<AssetType>& instance, const CAssetConverter::patch_t<AssetType>& soloPatch) const
		{
			if (const patch_index_t* pIndex=const_cast<dfs_cache<AssetType>*>(this)->impl<false>(instance,soloPatch); pIndex)
				return *pIndex;
			return {};
		}

		template<typename What>
		inline void for_each(What what)
		{
			for (auto& entry : instances)
			{
				const auto& instance = entry.first;
				auto patchIx = entry.second;
				assert(instance.asset || !patchIx);
				for (; patchIx; patchIx=nodes[patchIx.value].next)
					what(instance,nodes[patchIx.value]);
			}
		}

		// not a multi-map anymore because order of insertion into an equal range needs to be stable, so I just make it a linked list explicitly
		key_map_t instances;
		// node struct
		struct created_t
		{
			CAssetConverter::patch_t<AssetType> patch = {};
			core::blake3_hash_t contentHash = {};
			asset_cached_t<AssetType> gpuObj = {};
			patch_index_t next = {};
		};
		// all entries refer to patch by index, so its stable against vector growth
		core::vector<created_t> nodes;
};

//
struct input_metadata_t
{
	inline bool operator==(const input_metadata_t&) const = default;
	inline bool operator!=(const input_metadata_t&) const = default;

	explicit inline operator bool() const {return operator!=({});}

	size_t uniqueCopyGroupID = 0xdeadbeefBADC0FFEull;
	patch_index_t patchIndex = {};
};
// polymorphic
struct patched_instance_t
{
	instance_t<IAsset> instance;
	patch_index_t patchIx;
};
//
template<Asset AssetT>
class DFSVisitor
{
	protected:
		using AssetType = AssetT;

		template<Asset DepType>
		inline size_t getDependantUniqueCopyGroupID(const size_t usersGroupCopyID, const AssetType* user, const DepType* dep) const
		{
			return inputs.getDependantUniqueCopyGroupID(usersGroupCopyID,user,dep);
		}
		
		// impl
		template<Asset DepType, typename... IgnoredArgs>
		bool descend_impl(
			const instance_t<AssetType>& user, const CAssetConverter::patch_t<AssetType>& userPatch,
			const instance_t<DepType>& dep, CAssetConverter::patch_t<DepType>&& soloPatch,
			IgnoredArgs&&... ignoredArgs // DFS doesn't need to know
		)
		{
			return bool(descend_impl_impl<DepType>({user.asset,user.uniqueCopyGroupID},dep,std::move(soloPatch)));
		}

		// do nothing
		template<typename T>
		inline void nullOptional() {}

	public:
		// returns `input_metadata_t` which you can `bool(input_metadata_t)` to find out if patch was valid and merge was successful
		template<Asset DepType>
		input_metadata_t descend_impl_impl(const instance_t<IAsset>& user, const instance_t<DepType>& dep, CAssetConverter::patch_t<DepType>&& soloPatch)
		{
			// skip invalid inputs silently
			if (!soloPatch.valid(device))
			{
				inputs.logger.log(
					"Asset %p used by %p in group %d has an invalid initial patch and won't be converted!",
					system::ILogger::ELL_ERROR,dep.asset,user.asset,user.uniqueCopyGroupID
				);
				return {};
			}
			// special checks (normally the GPU object creation will fail, but these are common pitfall paths, so issue errors earlier for select problems)
			if constexpr (std::is_same_v<DepType,ICPUShader>)
			if (dep.asset->getContentType()==ICPUShader::E_CONTENT_TYPE::ECT_GLSL)
			{
				inputs.logger.log("Asset Converter doesn't support converting GLSL shaders! Asset %p won't be converted (GLSL is deprecated in Nabla)",system::ILogger::ELL_ERROR,dep.asset);
				return {};
			}
			if constexpr (std::is_same_v<DepType,ICPUBuffer>)
			if (dep.asset->getSize()>device->getPhysicalDevice()->getLimits().maxBufferSize)
			{
				inputs.logger.log(
					"Requested buffer size %zu is larger than the Physical Device's maxBufferSize Limit! Asset %p won't be converted",
					system::ILogger::ELL_ERROR,dep.asset->getSize(),dep.asset
				);
				return {};
			}
			// debug print
			inputs.logger.log("Asset (%p,%d) is used by (%p,%d)",system::ILogger::ELL_DEBUG,dep.asset,dep.uniqueCopyGroupID,user.asset,user.uniqueCopyGroupID);

			// now see if we visited already
			auto& dfsCache = std::get<dfs_cache<DepType>>(dfsCaches);
			// try to insert a new instance with a new patch, we'll be told if something mergable already existed
			auto [patchIndex,inserted] = dfsCache.insert(dep,std::move(soloPatch));
			// Only when we don't find a compatible patch entry do we carry on with the DFS
			if constexpr (asset_traits<DepType>::HasChildren)
			if (inserted)
				stack.emplace(instance_t<IAsset>{dep.asset,dep.uniqueCopyGroupID},patchIndex);
			// return the metadata
			return {.uniqueCopyGroupID=dep.uniqueCopyGroupID,.patchIndex=patchIndex};
		}

		const CAssetConverter::SInputs& inputs;
		ILogicalDevice* device;
		core::tuple_transform_t<dfs_cache,CAssetConverter::supported_asset_types>& dfsCaches;
		core::stack<patched_instance_t>& stack;
};

// because we need to iterate over all BLAS' patches to check is one of them got patched with Motion
class CheckBLASPatchMotions
{
	public:
		using AssetType = ICPUTopLevelAccelerationStructure;

		const CAssetConverter::SInputs& inputs;
		const dfs_cache<ICPUBottomLevelAccelerationStructure>& visitedBLASes;
		bool isMotion = false;

	protected:
		template<typename DepType>
		void nullOptional() const {}

		inline size_t getDependantUniqueCopyGroupID(const size_t usersGroupCopyID, const AssetType* user, const ICPUBottomLevelAccelerationStructure* dep) const
		{
			return inputs.getDependantUniqueCopyGroupID(usersGroupCopyID,user,dep);
		}

		bool descend_impl(
			const instance_t<AssetType>& user, const CAssetConverter::patch_t<AssetType>& userPatch,
			const instance_t<ICPUBottomLevelAccelerationStructure>& dep, const CAssetConverter::patch_t<ICPUBottomLevelAccelerationStructure>& soloPatch,
			const uint32_t instanceIndex // not the custom index, its literally just an ordinal in `getInstances()`
		)
		{
			// find matching patch in dfsCache
			const auto patchIx = visitedBLASes.find(dep,soloPatch);
			// must be found, must have been visited
			assert(bool(patchIx));
			// want to stop the visits after finding first BLAS with motion
			if (visitedBLASes.nodes[patchIx.value].patch.isMotion)
			{
				isMotion = true;
				return false;
			}
			return true;
		}
};

// go forth and find first patch that matches
class PatchOverride final : public CAssetConverter::CHashCache::IPatchOverride
{
		template<Asset AssetType>
		using lookup_t = CAssetConverter::CHashCache::lookup_t<AssetType>;
		template<Asset AssetType>
		using patch_t = CAssetConverter::patch_t<AssetType>;

		template<Asset AssetType>
		inline const patch_t<AssetType>* impl(const lookup_t<AssetType>& lookup) const
		{
			const auto& dfsCache = std::get<dfs_cache<AssetType>>(dfsCaches);
			const auto patchIx = dfsCache.find({lookup.asset,uniqueCopyGroupID},*lookup.patch);
			if (patchIx)
				return &dfsCache.nodes[patchIx.value].patch;
			assert(false); // really shouldn't happen because DFS descent should have explored ALL!
			return lookup.patch;
		}

	public:
		const CAssetConverter::SInputs& inputs;
		core::tuple_transform_t<dfs_cache,CAssetConverter::supported_asset_types>& dfsCaches;
		mutable size_t uniqueCopyGroupID;

		inline explicit PatchOverride(decltype(inputs) _inputs, decltype(dfsCaches) _dfsCaches, const size_t _uniqueCopyGroupID)
			: inputs(_inputs), dfsCaches(_dfsCaches), uniqueCopyGroupID(_uniqueCopyGroupID)
		{}


		inline const patch_t<ICPUSampler>* operator()(const lookup_t<ICPUSampler>& lookup) const override {return impl(lookup);}
		inline const patch_t<ICPUShader>* operator()(const lookup_t<ICPUShader>& lookup) const override {return impl(lookup);}
		inline const patch_t<ICPUBuffer>* operator()(const lookup_t<ICPUBuffer>& lookup) const override {return impl(lookup);}
		inline const patch_t<ICPUBottomLevelAccelerationStructure>* operator()(const lookup_t<ICPUBottomLevelAccelerationStructure>& lookup) const override {return impl(lookup);}
		inline const patch_t<ICPUTopLevelAccelerationStructure>* operator()(const lookup_t<ICPUTopLevelAccelerationStructure>& lookup) const override {return impl(lookup);}
		inline const patch_t<ICPUImage>* operator()(const lookup_t<ICPUImage>& lookup) const override {return impl(lookup);}
		inline const patch_t<ICPUBufferView>* operator()(const lookup_t<ICPUBufferView>& lookup) const override {return impl(lookup);}
		inline const patch_t<ICPUImageView>* operator()(const lookup_t<ICPUImageView>& lookup) const override {return impl(lookup);}
		inline const patch_t<ICPUPipelineLayout>* operator()(const lookup_t<ICPUPipelineLayout>& lookup) const override {return impl(lookup);}
};

template<Asset AssetT>
class HashVisit : public CAssetConverter::CHashCache::hash_impl_base
{
	public:
		using AssetType = AssetT;

	protected:
		template<Asset DepType>
		inline size_t getDependantUniqueCopyGroupID(const size_t usersGroupCopyID, const AssetType* user, const DepType* dep) const
		{
			return static_cast<const PatchOverride*>(patchOverride)->inputs.getDependantUniqueCopyGroupID(usersGroupCopyID,user,dep);
		}

		template<Asset DepType, typename... ExtraArgs>
		inline bool descend_impl(
			const instance_t<AssetType>& user, const CAssetConverter::patch_t<AssetType>& userPatch, // unused for this visit type
			const instance_t<DepType>& dep, const CAssetConverter::patch_t<DepType>& soloPatch,
			ExtraArgs&&... extraArgs // its just easier to hash most of those in `hash_impl::operator()`
		)
		{
			assert(hashCache && patchOverride);
			// find dependency compatible patch
			const CAssetConverter::patch_t<DepType>* found = patchOverride->operator()({dep.asset,&soloPatch});
			if (!found)
				return false;
			// hash dep
			static_cast<const PatchOverride*>(patchOverride)->uniqueCopyGroupID = dep.uniqueCopyGroupID;
			const auto depHash = hashCache->hash<DepType>({dep.asset,found},patchOverride,nextMistrustLevel);
			// check if hash failed
			if (depHash==CAssetConverter::CHashCache::NoContentHash)
				return false;
			// add dep hash to own
			hasher << depHash;
			// handle the few things we want to handle here
			if constexpr (sizeof...(extraArgs)>0)
			{
				auto argTuple = std::tuple<const ExtraArgs&...>(extraArgs...);
				const auto& arg0 = std::get<0>(argTuple);
				if constexpr (sizeof...(extraArgs)>1)
				{
					const auto& arg1 = std::get<1>(argTuple);
					// hash the spec info
					if constexpr (std::is_same_v<decltype(arg1),const ICPUShader::SSpecInfo&>)
					{
						hasher << arg1.entryPoint;
						for (const auto& specConstant : *arg1.entries)
						{
							hasher << specConstant.first;
							hasher.update(specConstant.second.data,specConstant.second.size);
						}
						hasher << arg1.requiredSubgroupSize;
						switch (arg0)
						{
							case IShader::E_SHADER_STAGE::ESS_COMPUTE:
								hasher << arg1.requireFullSubgroups;
								break;
							default:
								break;
						}
					}
				}
			}
			return true;
		}

		template<typename T>
		inline void nullOptional()
		{
			// just put something in the hash to denote that there's a null node in our graph
			hasher << '\0';
		}
};
bool CAssetConverter::CHashCache::hash_impl::operator()(lookup_t<ICPUSampler> lookup)
{
	auto patchedParams = lookup.asset->getParams();
	patchedParams.AnisotropicFilter = lookup.patch->anisotropyLevelLog2;
	hasher.update(&patchedParams,sizeof(patchedParams));
	return true;
}
bool CAssetConverter::CHashCache::hash_impl::operator()(lookup_t<ICPUShader> lookup)
{
	const auto* asset = lookup.asset;

	hasher << lookup.patch->stage;
	const auto type = asset->getContentType();
	hasher << type;
	// if not SPIR-V then own path matters
	if (type!=ICPUShader::E_CONTENT_TYPE::ECT_SPIRV)
		hasher << asset->getFilepathHint();
	const auto* content = asset->getContent();
	if (!content || content->getContentHash()==NoContentHash)
		return false;
	// we're not using the buffer directly, just its contents
	hasher << content->getContentHash();
	return true;
}
bool CAssetConverter::CHashCache::hash_impl::operator()(lookup_t<ICPUBuffer> lookup)
{
	auto patchedParams = lookup.asset->getCreationParams();
	assert(lookup.patch->usage.hasFlags(patchedParams.usage));
	patchedParams.usage = lookup.patch->usage;
	hasher.update(&patchedParams,sizeof(patchedParams)) << lookup.asset->getContentHash();
	return true;
}
bool CAssetConverter::CHashCache::hash_impl::operator()(lookup_t<ICPUBottomLevelAccelerationStructure> lookup)
{
	hasher << lookup.patch->isMotion;
	// overriden flags
	hasher << lookup.patch->getBuildFlags(lookup.asset);
	// extras from the patch
	hasher << lookup.patch->hostBuild;
	hasher << lookup.patch->compactAfterBuild;
	// finally the contents
	if (lookup.asset->getContentHash()==NoContentHash)
		return false;
	hasher << lookup.asset->getContentHash();
	return true;
}
bool CAssetConverter::CHashCache::hash_impl::operator()(lookup_t<ICPUTopLevelAccelerationStructure> lookup)
{
	hasher << lookup.patch->isMotion;
	// overriden flags
	const auto* asset = lookup.asset;
	hasher << lookup.patch->getBuildFlags(asset);
	// extras from the patch
	hasher << lookup.patch->hostBuild;
	hasher << lookup.patch->compactAfterBuild;
	hasher << (lookup.patch->isMotion ? lookup.patch->maxInstances:0u);
	const auto instances = asset->getInstances();
	hasher << instances.size();
	AssetVisitor<HashVisit<ICPUTopLevelAccelerationStructure>> visitor = {
		*this,
		{asset,static_cast<const PatchOverride*>(patchOverride)->uniqueCopyGroupID},
		*lookup.patch
	};
	if (!visitor())
		return false;
	// important two passes do not give identical data due to variable length polymorphic array being hashed
	for (const auto& instance : instances)
		hasher << instance.getType();
	for (const auto& instance : instances)
	{
		std::visit([&](const auto& typedInstance)->void
			{
				using instance_t = std::decay_t<decltype(typedInstance)>;
				// the BLAS pointers (the BLAS contents already get hashed via asset visitor and `getDependent`, its only the metadate we need to hash
				hasher.update(&typedInstance,offsetof(instance_t,base)+offsetof(ICPUTopLevelAccelerationStructure::Instance,blas));
			},
			instance.instance
		);
	}
	return true;
}
bool CAssetConverter::CHashCache::hash_impl::operator()(lookup_t<ICPUImage> lookup)
{
	// failed promotion
	if (lookup.patch->format==EF_UNKNOWN)
		return false;
	// extras from the patch
	hasher << lookup.patch->linearTiling;
	hasher << lookup.patch->recomputeMips;
	// overriden creation params
	const auto format = lookup.patch->format;
	hasher << format;
	hasher << lookup.patch->mipLevels;
	hasher << lookup.patch->usageFlags;
	hasher << lookup.patch->stencilUsage;
	// NOTE: We don't hash the usage metada from the patch! Because it doesn't matter.
	// The meta usages help us not merge incompatible patches together and not mis-promote a format
	const auto& origParams = lookup.asset->getCreationParameters();
	// sanity checks
	assert(lookup.patch->usageFlags.hasFlags(origParams.depthUsage));
	if (isDepthOrStencilFormat(format) && !isDepthOnlyFormat(format))
	{
		assert(lookup.patch->stencilUsage.hasFlags(origParams.actualStencilUsage()));
	}
	// non patchable params
	hasher << origParams.type;
	hasher << origParams.samples;
	hasher.update(&origParams.extent,sizeof(origParams.extent));
	hasher << origParams.arrayLayers;
	// now proceed to patch
	using create_flags_t = IGPUImage::E_CREATE_FLAGS;
	auto creationFlags = origParams.flags;
	if (lookup.patch->mutableFormat)
		creationFlags |= create_flags_t::ECF_MUTABLE_FORMAT_BIT;
	if (lookup.patch->cubeCompatible)
		creationFlags |= create_flags_t::ECF_CUBE_COMPATIBLE_BIT;
	if (lookup.patch->_3Dbut2DArrayCompatible)
		creationFlags |= create_flags_t::ECF_2D_ARRAY_COMPATIBLE_BIT;
	if (lookup.patch->uncompressedViewOfCompressed)
		creationFlags |= create_flags_t::ECF_BLOCK_TEXEL_VIEW_COMPATIBLE_BIT;
	hasher << creationFlags;
	// finally the contents
	if (lookup.asset->getContentHash()==NoContentHash)
		return false;
	hasher << lookup.asset->getContentHash();
	return true;
}
bool CAssetConverter::CHashCache::hash_impl::operator()(lookup_t<ICPUBufferView> lookup)
{
	const auto* asset = lookup.asset;
	AssetVisitor<HashVisit<ICPUBufferView>> visitor = {
		*this,
		{asset,static_cast<const PatchOverride*>(patchOverride)->uniqueCopyGroupID},
		*lookup.patch
	};
	if (!visitor())
		return false;
	// NOTE: We don't hash the usage metada from the patch! Because it doesn't matter.
	// The view usage in the patch helps us propagate and patch during DFS, but no more.
	hasher << asset->getFormat();
	hasher << asset->getOffsetInBuffer();
	hasher << asset->getByteSize();
	return true;
}
bool CAssetConverter::CHashCache::hash_impl::operator()(lookup_t<ICPUImageView> lookup)
{
	const auto* asset = lookup.asset;
	AssetVisitor<HashVisit<ICPUImageView>> visitor = {
		*this,
		{asset,static_cast<const PatchOverride*>(patchOverride)->uniqueCopyGroupID},
		*lookup.patch
	};
	if (!visitor())
		return false;
	auto patchedParams = asset->getCreationParameters();
	// NOTE: We don't hash the usage metada from the patch! Because it doesn't matter.
	// The view usage in the patch helps us propagate and patch during DFS, but no more.
	patchedParams.subUsages = lookup.patch->subUsages;
	const auto& imageParams = patchedParams.image->getCreationParameters();
	// kinda restore the format, but we don't set to new "promoted" format because we hash INPUTS not OUTPUTS
	if (lookup.patch->formatFollowsImage())
		patchedParams.format = imageParams.format;
	hasher.update(&patchedParams,sizeof(patchedParams));
	return true;
}
bool CAssetConverter::CHashCache::hash_impl::operator()(lookup_t<ICPUDescriptorSetLayout> lookup)
{
	const auto* asset = lookup.asset;
	// visit and hash all the immutable samplers
	AssetVisitor<HashVisit<ICPUDescriptorSetLayout>> visitor = {
		*this,
		{asset,static_cast<const PatchOverride*>(patchOverride)->uniqueCopyGroupID},
		*lookup.patch
	};
	if (!visitor())
		return false;

	using storage_range_index_t = ICPUDescriptorSetLayout::CBindingRedirect::storage_range_index_t;
	// but also need to hash the binding they are at
	const auto& immutableSamplerRedirects = asset->getImmutableSamplerRedirect();
	const auto count = immutableSamplerRedirects.getBindingCount();
	for (auto i=0u; i<count; i++)
	{
		const storage_range_index_t storageRangeIx(i);
		hasher << immutableSamplerRedirects.getBinding(storageRangeIx).data;
	}
	// then all the other bindings
	for (uint32_t t=0u; t<static_cast<uint32_t>(IDescriptor::E_TYPE::ET_COUNT); t++)
	{
		const auto type = static_cast<IDescriptor::E_TYPE>(t);
		hasher << type;
		const auto& redirect = asset->getDescriptorRedirect(type);
		const auto count = redirect.getBindingCount();
		for (auto i=0u; i<count; i++)
		{
			const storage_range_index_t storageRangeIx(i);
			hasher << redirect.getBinding(storageRangeIx).data;
			hasher << redirect.getCreateFlags(storageRangeIx);
			hasher << redirect.getStageFlags(storageRangeIx);
			hasher << redirect.getCount(storageRangeIx);
		}
	}
	return true;
}
bool CAssetConverter::CHashCache::hash_impl::operator()(lookup_t<ICPUPipelineLayout> lookup)
{
	const auto* asset = lookup.asset;
	// visit and hash all the set layouts
	AssetVisitor<HashVisit<ICPUPipelineLayout>> visitor = {
		*this,
		{asset,static_cast<const PatchOverride*>(patchOverride)->uniqueCopyGroupID},
		*lookup.patch
	};
	if (!visitor())
		return false;
	// the pc byte ranges
	hasher << std::span(lookup.patch->pushConstantBytes);
	return true;
}
bool CAssetConverter::CHashCache::hash_impl::operator()(lookup_t<ICPUPipelineCache> lookup)
{
	for (const auto& entry : lookup.asset->getEntries())
	{
		hasher << entry.first.deviceAndDriverUUID;
		if (entry.first.meta)
			hasher.update(entry.first.meta->data(),entry.first.meta->size());
	}
	if (lookup.asset->getContentHash()==NoContentHash)
		return false;
	hasher << lookup.asset->getContentHash();
	return true;
}
bool CAssetConverter::CHashCache::hash_impl::operator()(lookup_t<ICPUComputePipeline> lookup)
{
	const auto* asset = lookup.asset;
	//
	AssetVisitor<HashVisit<ICPUComputePipeline>> visitor = {
		*this,
		{asset,static_cast<const PatchOverride*>(patchOverride)->uniqueCopyGroupID},
		*lookup.patch
	};
	if (!visitor())
		return false;
	return true;
}
bool CAssetConverter::CHashCache::hash_impl::operator()(lookup_t<ICPURenderpass> lookup)
{
	const auto* asset = lookup.asset;

	hasher << asset->getDepthStencilAttachmentCount();
	hasher << asset->getColorAttachmentCount();
	hasher << asset->getSubpassCount();
	hasher << asset->getDependencyCount();
	hasher << asset->getViewMaskMSB();
	const ICPURenderpass::SCreationParams& params = asset->getCreationParameters();
	{
		auto hashLayout = [&](const E_FORMAT format, const IImage::SDepthStencilLayout& layout)->void
		{
			if (!isStencilOnlyFormat(format))
				hasher << layout.depth;
			if (!isDepthOnlyFormat(format))
				hasher << layout.stencil;
		};

		for (auto i=0; i<asset->getDepthStencilAttachmentCount(); i++)
		{
			auto entry = params.depthStencilAttachments[i];
			if (!entry.valid())
				return false;
			hasher << entry.format;
			hasher << entry.samples;
			hasher << entry.mayAlias;
			auto hashOp = [&](const auto& op)->void
			{
				if (!isStencilOnlyFormat(entry.format))
					hasher << op.depth;
				if (!isDepthOnlyFormat(entry.format))
					hasher << op.actualStencilOp();
			};
			hashOp(entry.loadOp);
			hashOp(entry.storeOp);
			hashLayout(entry.format,entry.initialLayout);
			hashLayout(entry.format,entry.finalLayout);
		}
		for (auto i=0; i<asset->getColorAttachmentCount(); i++)
		{
			const auto& entry = params.colorAttachments[i];
			if (!entry.valid())
				return false;
			hasher.update(&entry,sizeof(entry));
		}
		// subpasses
		using SubpassDesc = ICPURenderpass::SCreationParams::SSubpassDescription;
		auto hashDepthStencilAttachmentRef = [&](const SubpassDesc::SDepthStencilAttachmentRef& ref)->void
		{
			hasher << ref.attachmentIndex;
			hashLayout(params.depthStencilAttachments[ref.attachmentIndex].format,ref.layout);
		};
		for (auto i=0; i<asset->getSubpassCount(); i++)
		{
			const auto& entry = params.subpasses[i];
			const auto depthStencilRenderAtt = entry.depthStencilAttachment.render;
			if (depthStencilRenderAtt.used())
			{
				hashDepthStencilAttachmentRef(depthStencilRenderAtt);
				if (entry.depthStencilAttachment.resolve.used())
				{
					hashDepthStencilAttachmentRef(entry.depthStencilAttachment.resolve);
					hasher.update(&entry.depthStencilAttachment.resolveMode,sizeof(entry.depthStencilAttachment.resolveMode));
				}
			}
			else // hash needs to care about which slots go unused
				hasher << false;
			// color attachments
			for (const auto& colorAttachment : std::span(entry.colorAttachments))
			{
				if (colorAttachment.render.used())
				{
					hasher.update(&colorAttachment.render,sizeof(colorAttachment.render));
					if (colorAttachment.resolve.used())
						hasher.update(&colorAttachment.resolve,sizeof(colorAttachment.resolve));
				}
				else // hash needs to care about which slots go unused
					hasher << false;
			}
			// input attachments
			for (auto inputIt=entry.inputAttachments; *inputIt!=SubpassDesc::InputAttachmentsEnd; inputIt++)
			{
				if (inputIt->used())
				{
					hasher << inputIt->aspectMask;
                    if (inputIt->aspectMask==IImage::EAF_COLOR_BIT)
						hashDepthStencilAttachmentRef(inputIt->asDepthStencil);
					else
						hasher.update(&inputIt->asColor,sizeof(inputIt->asColor));
				}
				else
					hasher << false;
			}
			// preserve attachments
			for (auto preserveIt=entry.preserveAttachments; *preserveIt!=SubpassDesc::PreserveAttachmentsEnd; preserveIt++)
				hasher.update(preserveIt,sizeof(SubpassDesc::SPreserveAttachmentRef));
			hasher << entry.viewMask;
			hasher << entry.flags;
		}
		// TODO: we could sort these before hashing (and creating GPU objects)
		hasher.update(params.dependencies,sizeof(ICPURenderpass::SCreationParams::SSubpassDependency)*asset->getDependencyCount());
	}
	hasher.update(params.viewCorrelationGroup,sizeof(params.viewCorrelationGroup));

	return true;
}
bool CAssetConverter::CHashCache::hash_impl::operator()(lookup_t<ICPUGraphicsPipeline> lookup)
{
	const auto* asset = lookup.asset;
	//
	AssetVisitor<HashVisit<ICPUGraphicsPipeline>> visitor = {
		*this,
		{asset,static_cast<const PatchOverride*>(patchOverride)->uniqueCopyGroupID},
		*lookup.patch
	};
	if (!visitor())
		return false;

	const auto& params = asset->getCachedCreationParams();
	{
		for (auto i=0; i<SVertexInputParams::MAX_VERTEX_ATTRIB_COUNT; i++)
		if (params.vertexInput.enabledAttribFlags&(0x1u<<i))
		{
			const auto& attribute = params.vertexInput.attributes[i];
			hasher.update(&attribute,sizeof(SVertexInputAttribParams));
			hasher.update(&params.vertexInput.bindings+attribute.binding,sizeof(SVertexInputBindingParams));
		}
		const auto& ass = params.primitiveAssembly;
		hasher << ass.primitiveType;
		hasher << ass.primitiveRestartEnable;
		if (ass.primitiveType==E_PRIMITIVE_TOPOLOGY::EPT_PATCH_LIST)
			hasher << ass.tessPatchVertCount;
		const auto& raster = params.rasterization;
		if (!raster.rasterizerDiscard)
		{
			hasher << raster.viewportCount;
			hasher << raster.samplesLog2;
			hasher << raster.polygonMode;
			//if (raster.polygonMode==E_POLYGON_MODE::EPM_FILL) // do wireframes and point draw with face culling?
			{
				hasher << raster.faceCullingMode;
				hasher << raster.frontFaceIsCCW;
			}
			const auto& rpassParam = asset->getRenderpass()->getCreationParameters();
			const auto& depthStencilRef = rpassParam.subpasses[params.subpassIx].depthStencilAttachment.render;
			if (depthStencilRef.used())
			{
				const auto attFormat = rpassParam.depthStencilAttachments[depthStencilRef.attachmentIndex].format;
				if (!isStencilOnlyFormat(attFormat))
				{
					hasher << raster.depthCompareOp;
					hasher << raster.depthWriteEnable;
					if (raster.depthTestEnable())
					{
						hasher << raster.depthClampEnable;
						hasher << raster.depthBiasEnable;
						hasher << raster.depthBoundsTestEnable;
					}
				}
				if (raster.stencilTestEnable() && !isDepthOnlyFormat(attFormat))
				{
					if ((raster.faceCullingMode&E_FACE_CULL_MODE::EFCM_FRONT_BIT)==0)
						hasher << raster.frontStencilOps;
					if ((raster.faceCullingMode&E_FACE_CULL_MODE::EFCM_BACK_BIT)==0)
						hasher << raster.backStencilOps;
				}
			}
			hasher << raster.alphaToCoverageEnable;
			hasher << raster.alphaToOneEnable;
			if (raster.samplesLog2)
			{
				hasher << raster.minSampleShadingUnorm;
				hasher << (reinterpret_cast<const uint64_t&>(raster.sampleMask)&((0x1ull<<raster.samplesLog2)-1));
			}
		}
		for (const auto& blend : std::span(params.blend.blendParams))
		{
			if (blend.blendEnabled())
				hasher.update(&blend,sizeof(blend));
			else
				hasher << blend.colorWriteMask;
		}
		hasher << params.blend.logicOp;
	}
	hasher << params.subpassIx;
	return true;
}
bool CAssetConverter::CHashCache::hash_impl::operator()(lookup_t<ICPUDescriptorSet> lookup)
{
	const auto* asset = lookup.asset;
	//
	AssetVisitor<HashVisit<ICPUDescriptorSet>> visitor = {
		*this,
		{asset,static_cast<const PatchOverride*>(patchOverride)->uniqueCopyGroupID},
		*lookup.patch
	};
	if (!visitor())
		return false;
	//
	for (auto i=0u; i<static_cast<uint32_t>(IDescriptor::E_TYPE::ET_COUNT); i++)
	{
		const auto type = static_cast<IDescriptor::E_TYPE>(i);
		const auto infos = asset->getDescriptorInfoStorage(type);
		if (infos.empty())
			continue;
		for (const auto& info : infos)
		if (const auto* untypedDesc=info.desc.get(); untypedDesc)
		{
			core::blake3_hash_t descHash = NoContentHash;
			switch (IDescriptor::GetTypeCategory(type))
			{
				case IDescriptor::EC_BUFFER:
					hasher.update(&info.info.buffer,sizeof(info.info.buffer));
					break;
				case IDescriptor::EC_IMAGE:
					hasher.update(&info.info.image,sizeof(info.info.image));
					break;
				default:
					break;
			}
		}
	}
	return true;
}

void CAssetConverter::CHashCache::eraseStale(const IPatchOverride* patchOverride)
{
	auto rehash = [&]<typename AssetType>() -> void
	{
		auto& container = std::get<container_t<AssetType>>(m_containers);
		core::erase_if(container,[&](const auto& entry)->bool
			{
				// backup because `hash(lookup)` call will update it
				const auto oldHash = entry.second;
				const auto& key = entry.first;
				// can re-use cached hashes for dependants if we start ejecting in the correct order
				const auto newHash = hash<AssetType>({key.asset.get(),&key.patch},patchOverride,/*.cacheMistrustLevel = */1);
				return newHash!=oldHash || newHash==NoContentHash;
			}
		);
	};
	// to make the process more efficient we start ejecting from "lowest level" assets
	rehash.template operator()<ICPUSampler>();
	rehash.template operator()<ICPUDescriptorSetLayout>();
	rehash.template operator()<ICPUPipelineLayout>();
	// shaders and images depend on buffers for data sourcing
	rehash.template operator()<ICPUBuffer>();
	rehash.template operator()<ICPUBufferView>();
	rehash.template operator()<ICPUImage>();
	rehash.template operator()<ICPUImageView>();
	rehash.template operator()<ICPUBottomLevelAccelerationStructure>();
	rehash.template operator()<ICPUTopLevelAccelerationStructure>();
	// only once all the descriptor types have been hashed, we can hash sets
	rehash.template operator()<ICPUDescriptorSet>();
	// naturally any pipeline depends on shaders and pipeline cache
	rehash.template operator()<ICPUShader>();
	rehash.template operator()<ICPUPipelineCache>();
	rehash.template operator()<ICPUComputePipeline>();
	// graphics pipeline needs a renderpass
	rehash.template operator()<ICPURenderpass>();
	rehash.template operator()<ICPUGraphicsPipeline>();
//	rehash.template operator()<ICPUFramebuffer>();
}


//
template<Asset AssetT>
class GetDependantVisitBase
{
	public:
		using AssetType = AssetT;

		const CAssetConverter::SInputs& inputs;
		core::tuple_transform_t<dfs_cache,CAssetConverter::supported_asset_types>& dfsCaches;

	protected:
		template<Asset DepType>
		inline size_t getDependantUniqueCopyGroupID(const size_t usersGroupCopyID, const AssetType* user, const DepType* dep) const
		{
			return inputs.getDependantUniqueCopyGroupID(usersGroupCopyID,user,dep);
		}

		template<Asset DepType>
		asset_cached_t<DepType>::type getDependant(const instance_t<DepType>& dep, const CAssetConverter::patch_t<DepType>& soloPatch) const
		{
			const auto& dfsCache = std::get<dfs_cache<DepType>>(dfsCaches);
			// find matching patch in dfsCache
			const auto patchIx = dfsCache.find(dep,soloPatch);
			// if not found
			if (!patchIx)
				return {}; // nullptr basically
			// grab gpu object from the node of patch index
			return dfsCache.nodes[patchIx.value].gpuObj.value;
		}

		template<typename DepType>
		void nullOptional() const {}
};
template<Asset AssetType>
class GetDependantVisit;

template<>
class GetDependantVisit<ICPUTopLevelAccelerationStructure> : public GetDependantVisitBase<ICPUTopLevelAccelerationStructure>
{
	public:
		CAssetConverter::SReserveResult::SConvReqTLAS::cpu_to_gpu_blas_map_t* instanceMap;

	protected:
		bool descend_impl(
			const instance_t<AssetType>& user, const CAssetConverter::patch_t<AssetType>& userPatch,
			const instance_t<ICPUBottomLevelAccelerationStructure>& dep, const CAssetConverter::patch_t<ICPUBottomLevelAccelerationStructure>& soloPatch,
			const uint32_t instanceIndex // not the custom index, its literally just an ordinal in `getInstances()`
		)
		{
			auto depObj = getDependant<ICPUBottomLevelAccelerationStructure>(dep,soloPatch);
			if (!depObj)
				return false;
			instanceMap->operator[](dep.asset) = std::move(depObj);
			return true;
		}
};

template<>
class GetDependantVisit<ICPUBufferView> : public GetDependantVisitBase<ICPUBufferView>
{
	public:
		SBufferRange<IGPUBuffer> underlying = {};

	protected:
		bool descend_impl(
			const instance_t<AssetType>& user, const CAssetConverter::patch_t<AssetType>& userPatch,
			const instance_t<ICPUBuffer>& dep, const CAssetConverter::patch_t<ICPUBuffer>& soloPatch
		)
		{
			auto depObj = getDependant<ICPUBuffer>(dep,soloPatch);
			if (!depObj)
				return false;
			underlying = {
				.offset = user.asset->getOffsetInBuffer(),
				.size = user.asset->getByteSize(),
				.buffer = std::move(depObj)
			};
			return underlying.isValid();
		}
};
template<>
class GetDependantVisit<ICPUImageView> : public GetDependantVisitBase<ICPUImageView>
{
	public:
		core::smart_refctd_ptr<IGPUImage> image = {};
		uint8_t oldMipCount = 0;

	protected:
		bool descend_impl(
			const instance_t<AssetType>& user, const CAssetConverter::patch_t<AssetType>& userPatch,
			const instance_t<ICPUImage>& dep, const CAssetConverter::patch_t<ICPUImage>& soloPatch
		)
		{
			image = getDependant<ICPUImage>(dep,soloPatch);
			if (!image)
				return false;
			oldMipCount = dep.asset->getCreationParameters().mipLevels;
			return true;
		}
};
template<>
class GetDependantVisit<ICPUDescriptorSetLayout> : public GetDependantVisitBase<ICPUDescriptorSetLayout>
{
	public:
		core::smart_refctd_ptr<IGPUSampler>* const outImmutableSamplers;

	protected:
		bool descend_impl(
			const instance_t<AssetType>& user, const CAssetConverter::patch_t<AssetType>& userPatch,
			const instance_t<ICPUSampler>& dep, const CAssetConverter::patch_t<ICPUSampler>& soloPatch,
			const uint32_t immutableSamplerStorageOffset
		)
		{
			auto depObj = getDependant<ICPUSampler>(dep,soloPatch);
			if (!depObj)
				return false;
			outImmutableSamplers[immutableSamplerStorageOffset] = std::move(depObj);
			return true;
		}
};
template<>
class GetDependantVisit<ICPUPipelineLayout> : public GetDependantVisitBase<ICPUPipelineLayout>
{
	public:
		core::smart_refctd_ptr<IGPUDescriptorSetLayout> dsLayouts[4] = {};

	protected:
		bool descend_impl(
			const instance_t<AssetType>& user, const CAssetConverter::patch_t<AssetType>& userPatch,
			const instance_t<ICPUDescriptorSetLayout>& dep, const CAssetConverter::patch_t<ICPUDescriptorSetLayout>& soloPatch,
			const uint32_t setIndex
		)
		{
			auto depObj = getDependant<ICPUDescriptorSetLayout>(dep,soloPatch);
			if (!depObj)
				return false;
			dsLayouts[setIndex] = std::move(depObj);
			return true;
		}
};
template<>
class GetDependantVisit<ICPUComputePipeline> : public GetDependantVisitBase<ICPUComputePipeline>
{
	public:
//		using AssetType = ICPUComputePipeline;

		inline auto& getSpecInfo(const IShader::E_SHADER_STAGE stage)
		{
			assert(hlsl::bitCount(stage)==1);
			return specInfo[hlsl::findLSB(stage)];
		}

		// ok to do non owning since some cache owns anyway
		IGPUPipelineLayout* layout = nullptr;
		// has to be public to allow for initializer list constructor
		std::array<IGPUShader::SSpecInfo,/*hlsl::mpl::findMSB<ESS_COUNT>::value*/sizeof(IShader::E_SHADER_STAGE)*8> specInfo = {};

	protected:
		bool descend_impl(
			const instance_t<ICPUComputePipeline>& user, const CAssetConverter::patch_t<ICPUComputePipeline>& userPatch,
			const instance_t<ICPUPipelineLayout>& dep, const CAssetConverter::patch_t<ICPUPipelineLayout>& soloPatch
		)
		{
			auto depObj = getDependant<ICPUPipelineLayout>(dep,soloPatch);
			if (!depObj)
				return false;
			layout = depObj.get();
			return true;
		}
		bool descend_impl(
			const instance_t<ICPUComputePipeline>& user, const CAssetConverter::patch_t<ICPUComputePipeline>& userPatch,
			const instance_t<ICPUShader>& dep, const CAssetConverter::patch_t<ICPUShader>& soloPatch,
			const IShader::E_SHADER_STAGE stage, const IShader::SSpecInfo<const ICPUShader>& inSpecInfo
		)
		{
			auto depObj = getDependant<ICPUShader>(dep,soloPatch);
			if (!depObj)
				return false;
			getSpecInfo(stage) = {
				.entryPoint = inSpecInfo.entryPoint,
				.shader = depObj.get(),
				.entries = inSpecInfo.entries,
				.requiredSubgroupSize = inSpecInfo.requiredSubgroupSize,
				.requireFullSubgroups = inSpecInfo.requireFullSubgroups
			};
			return true;
		}
};
template<>
class GetDependantVisit<ICPUGraphicsPipeline> : public GetDependantVisitBase<ICPUGraphicsPipeline>
{
	public:
//		using AssetType = ICPUGraphicsPipeline;

		inline auto& getSpecInfo(const IShader::E_SHADER_STAGE stage)
		{
			assert(hlsl::bitCount(stage)==1);
			return specInfo[hlsl::findLSB(stage)];
		}

		// ok to do non owning since some cache owns anyway
		IGPUPipelineLayout* layout = nullptr;
		// has to be public to allow for initializer list constructor
		std::array<IGPUShader::SSpecInfo,/*hlsl::mpl::findMSB<ESS_COUNT>::value*/sizeof(IShader::E_SHADER_STAGE)*8> specInfo = {};
		// optionals (done this way because inheritance chain with templated class hides protected methods)
		IGPURenderpass* renderpass = nullptr;

	protected:
		bool descend_impl(
			const instance_t<ICPUGraphicsPipeline>& user, const CAssetConverter::patch_t<ICPUGraphicsPipeline>& userPatch,
			const instance_t<ICPUPipelineLayout>& dep, const CAssetConverter::patch_t<ICPUPipelineLayout>& soloPatch
		)
		{
			auto depObj = getDependant<ICPUPipelineLayout>(dep,soloPatch);
			if (!depObj)
				return false;
			layout = depObj.get();
			return true;
		}
		bool descend_impl(
			const instance_t<ICPUGraphicsPipeline>& user, const CAssetConverter::patch_t<ICPUGraphicsPipeline>& userPatch,
			const instance_t<ICPUShader>& dep, const CAssetConverter::patch_t<ICPUShader>& soloPatch,
			const IShader::E_SHADER_STAGE stage, const IShader::SSpecInfo<const ICPUShader>& inSpecInfo
		)
		{
			auto depObj = getDependant<ICPUShader>(dep,soloPatch);
			if (!depObj)
				return false;
			getSpecInfo(stage) = {
				.entryPoint = inSpecInfo.entryPoint,
				.shader = depObj.get(),
				.entries = inSpecInfo.entries,
				.requiredSubgroupSize = inSpecInfo.requiredSubgroupSize,
				.requireFullSubgroups = 0
			};
			return true;
		}
		bool descend_impl(
			const instance_t<ICPUGraphicsPipeline>& user, const CAssetConverter::patch_t<ICPUGraphicsPipeline>& userPatch,
			const instance_t<ICPURenderpass>& dep, const CAssetConverter::patch_t<ICPURenderpass>& soloPatch
		)
		{
			auto depObj = getDependant<ICPURenderpass>(dep,soloPatch);
			if (!depObj)
				return false;
			renderpass = depObj.get();
			return true;
		}
};
template<>
class GetDependantVisit<ICPUDescriptorSet> : public GetDependantVisitBase<ICPUDescriptorSet>
{
	public:
		// returns if there are any writes to do
		bool finalizeWrites(IGPUDescriptorSet* dstSet)
		{
			if (writes.empty())
				return false;
			// now infos can't move in memory anymore
			auto baseInfoPtr = infos.data();
			for (auto& write : writes)
			{
				write.dstSet = dstSet;
				write.info = baseInfoPtr+reinterpret_cast<const size_t&>(write.info);
			}
			return true;
		}

		core::smart_refctd_ptr<IGPUDescriptorSetLayout> layout = {};
		// okay to do non-owning, cache has ownership
		core::vector<IGPUDescriptorSet::SWriteDescriptorSet> writes = {};
		core::vector<IGPUDescriptorSet::SDescriptorInfo> infos = {};
		core::vector<IGPUDescriptorSetLayout::CBindingRedirect::storage_offset_t> potentialTLASRewrites = {};
		// has to be public because of aggregate init, but its only for internal usage!
		uint32_t lastBinding;
		uint32_t lastElement;
		// special state to pass around
		IGPUSampler* lastCombinedSampler;

	protected:
		bool descend_impl(
			const instance_t<AssetType>& user, const CAssetConverter::patch_t<AssetType>& userPatch,
			const instance_t<ICPUDescriptorSetLayout>& dep, const CAssetConverter::patch_t<ICPUDescriptorSetLayout>& soloPatch
		)
		{
			auto depObj = getDependant<ICPUDescriptorSetLayout>(dep,soloPatch);
			if (!depObj)
				return false;
			layout = std::move(depObj);
			// set initial state
			lastBinding = 0xdeadbeefu;
			lastElement = 0xdeadbeefu;
			lastCombinedSampler = nullptr;
			// could reserve `writes` and `infos`
			return true;
		}
		template<Asset DepType, typename... ExtraArgs> requires std::is_base_of_v<IDescriptor,DepType>
		bool descend_impl(
			const instance_t<AssetType>& user, const CAssetConverter::patch_t<AssetType>& userPatch,
			const instance_t<DepType>& dep, const CAssetConverter::patch_t<DepType>& soloPatch,
			const IDescriptor::E_TYPE type,
			const IDescriptorSetLayoutBase::CBindingRedirect::binding_number_t binding,
			const uint32_t element, ExtraArgs&&... extraArgs
		)
		{
			auto depObj = getDependant<DepType>(dep,soloPatch);
			if (!depObj)
				return false;
			// special path for handling combined samplers (remeber them for the image call)
			if constexpr (std::is_same_v<DepType,ICPUSampler>)
			if (type==IDescriptor::E_TYPE::ET_COMBINED_IMAGE_SAMPLER)
			{
				lastCombinedSampler = depObj.get();
				return true;
			}
			// a bit of RLE
			if (binding.data!=lastBinding || element!=(lastElement+1))
			{
				writes.push_back({
					.dstSet = nullptr, // will be patched later
					.binding = binding.data,
					.arrayElement = element,
					.count = 1,
					.info = reinterpret_cast<const IGPUDescriptorSet::SDescriptorInfo*>(infos.size()) // patch base ptr later
				});
				lastBinding = binding.data;
			}
			else
				writes.back().count++;
			lastElement = element;
			//
			auto& outInfo = infos.emplace_back();
			// extra stuff
			auto argTuple = std::tuple<const ExtraArgs&...>(extraArgs...);
			if constexpr (std::is_same_v<DepType,ICPUBuffer>)
			{
				if (IDescriptor::GetTypeCategory(type)==IDescriptor::E_CATEGORY::EC_BUFFER)
				{
					//outInfo.info.buffer = std::get<0>(argTuple);
					outInfo.info.buffer.offset = std::get<0>(argTuple).offset;
					outInfo.info.buffer.size = std::get<0>(argTuple).size;
				}
			}
			// mark potential TLAS rewrites (with compaction) so we don't have to scan entire descriptor set for potentially compacted TLASes
			if constexpr (std::is_same_v<DepType,ICPUTopLevelAccelerationStructure>)
			if (depObj->getPendingBuildVer()==0) // means not built yet, so compactable by next `convert` run
			{
				auto storageOffset = std::get<0>(argTuple);
				storageOffset.data += element;
				potentialTLASRewrites.push_back(storageOffset);
			}
			if constexpr (std::is_same_v<DepType,ICPUImageView>)
			{
				outInfo.info.image.imageLayout = std::get<0>(argTuple);
				if (type==IDescriptor::E_TYPE::ET_COMBINED_IMAGE_SAMPLER)
				{
					assert(lastCombinedSampler);
					outInfo.info.combinedImageSampler.sampler = smart_refctd_ptr<IGPUSampler>(lastCombinedSampler);
					lastCombinedSampler = nullptr; // for debuggability
				}
			}
			outInfo.desc = std::move(depObj);
			return true;
		}
};


// Needed both for reservation and conversion
class MetaDeviceMemoryAllocator final
{
	public:
		MetaDeviceMemoryAllocator(IDeviceMemoryAllocator* _allocator, system::logger_opt_ptr _logger) : m_allocator(_allocator), m_logger(_logger) {}

		// a somewhat structured uint64_t
		struct MemoryRequirementBin
		{
			inline bool operator==(const MemoryRequirementBin&) const = default;

			// We order our requirement bins from those that can be allocated from the most memory types to those that can only be allocated from one
			inline bool operator<(const MemoryRequirementBin& other) const
			{
				if (needsDeviceAddress!=other.needsDeviceAddress)
					return needsDeviceAddress;
				return hlsl::bitCount(compatibileMemoryTypeBits)<hlsl::bitCount(other.compatibileMemoryTypeBits);
			}

			uint64_t compatibileMemoryTypeBits : 32 = 0;
			uint64_t needsDeviceAddress : 1 = 0;
		};
	
		template<Asset AssetType>
		bool request(asset_cached_t<AssetType>* pGpuObj, const uint32_t memoryTypeConstraint=~0u)
		{
			auto* gpuObj = pGpuObj->get();
			const IDeviceMemoryBacked::SDeviceMemoryRequirements& memReqs = gpuObj->getMemoryReqs();
			// overconstrained
			if ((memReqs.memoryTypeBits&memoryTypeConstraint)==0)
			{
				m_logger.log("Overconstrained the Memory Type Index bitmask %d with %d for %s",system::ILogger::ELL_ERROR,memReqs.memoryTypeBits,memoryTypeConstraint,gpuObj->getObjectDebugName());
				pGpuObj->value = nullptr;
				return false;
			}
			//
			bool needsDeviceAddress = false;
			if constexpr (std::is_same_v<std::remove_pointer_t<decltype(gpuObj)>,IGPUBuffer>)
			{
				const auto usage = gpuObj->getCreationParams().usage;
				needsDeviceAddress = usage.hasFlags(IGPUBuffer::E_USAGE_FLAGS::EUF_SHADER_DEVICE_ADDRESS_BIT);
				// stops us needing weird awful code to ensure buffer storing AS has alignment of at least 256
				assert(!usage.hasFlags(IGPUBuffer::E_USAGE_FLAGS::EUF_ACCELERATION_STRUCTURE_STORAGE_BIT) || memReqs.alignmentLog2>=8);
			}
			// allocate right away those that need their own allocation
			if (memReqs.requiresDedicatedAllocation)
			{
				// allocate and bind right away
				auto allocation = m_allocator->allocate(memReqs,gpuObj);
				if (!allocation.isValid())
				{
					m_logger.log("Failed to allocate and bind dedicated memory for %s",system::ILogger::ELL_ERROR,gpuObj->getObjectDebugName());
					pGpuObj->value = nullptr;
					return false;
				}
			}
			else
			{
				// make the creation conditional upon allocation success
				const MemoryRequirementBin reqBin = {
					.compatibileMemoryTypeBits = memReqs.memoryTypeBits&memoryTypeConstraint,
					.needsDeviceAddress = needsDeviceAddress,
					// we ignore this for now, because we can't know how many `DeviceMemory` objects we have left to make, so just join everything by default
					//.refersDedicatedAllocation = memReqs.prefersDedicatedAllocation
				};
				allocationRequests[reqBin].emplace_back(pGpuObj);
			}
			return true;
		}

		//
		void finalize()
		{
			auto getAsBase = [](const memory_backed_ptr_variant_t& var) -> const IDeviceMemoryBacked*
			{
				switch (var.index())
				{
					case 0:
						return std::get<asset_cached_t<ICPUBuffer>*>(var)->get();
					case 1:
						return std::get<asset_cached_t<ICPUImage>*>(var)->get();
					default:
						assert(false);
						break;
				}
				return nullptr;
			};
			// sort each bucket by size from largest to smallest with pessimized allocation size due to alignment
			for (auto& bin : allocationRequests)
				std::sort(bin.second.begin(),bin.second.end(),[getAsBase](const memory_backed_ptr_variant_t& lhs, const memory_backed_ptr_variant_t& rhs)->bool
					{
						const auto& lhsReqs = getAsBase(lhs)->getMemoryReqs();
						const auto& rhsReqs = getAsBase(rhs)->getMemoryReqs();
						const size_t lhsWorstSize = lhsReqs.size+(0x1ull<<lhsReqs.alignmentLog2)-1;
						const size_t rhsWorstSize = rhsReqs.size+(0x1ull<<rhsReqs.alignmentLog2)-1;
						return lhsWorstSize>rhsWorstSize;
					}
				);

			// lets define our order of memory type usage
			auto* device = m_allocator->getDeviceForAllocations();
			const auto& memoryProps = device->getPhysicalDevice()->getMemoryProperties();
			core::vector<uint32_t> memoryTypePreference(memoryProps.memoryTypeCount);
			std::iota(memoryTypePreference.begin(),memoryTypePreference.end(),0);
			std::sort(memoryTypePreference.begin(),memoryTypePreference.end(),
				[&memoryProps](const uint32_t leftIx, const uint32_t rightIx)->bool
				{
					const auto& leftType = memoryProps.memoryTypes[leftIx];
					const auto& rightType = memoryProps.memoryTypes[rightIx];

					using flags_t = IDeviceMemoryAllocation::E_MEMORY_PROPERTY_FLAGS;
					const auto& leftTypeFlags = leftType.propertyFlags;
					const auto& rightTypeFlags = rightType.propertyFlags;

					// we want to try types that device local first, then non-device local
					const bool leftDeviceLocal = leftTypeFlags.hasFlags(flags_t::EMPF_DEVICE_LOCAL_BIT);
					const bool rightDeviceLocal = rightTypeFlags.hasFlags(flags_t::EMPF_DEVICE_LOCAL_BIT);
					if (leftDeviceLocal!=rightDeviceLocal)
						return leftDeviceLocal;

					// then we want to allocate from largest heap to smallest
					// TODO: actually query the amount of free memory using VK_EXT_memory_budget
					const size_t leftHeapSize = memoryProps.memoryHeaps[leftType.heapIndex].size;
					const size_t rightHeapSize = memoryProps.memoryHeaps[rightType.heapIndex].size;
					if (leftHeapSize<rightHeapSize)
						return true;
					else if (leftHeapSize!=rightHeapSize)
						return false;

					// within those types we want to first do non-mappable
					const bool leftMappable = leftTypeFlags.value&(flags_t::EMPF_HOST_READABLE_BIT|flags_t::EMPF_HOST_WRITABLE_BIT);
					const bool rightMappable = rightTypeFlags.value&(flags_t::EMPF_HOST_READABLE_BIT|flags_t::EMPF_HOST_WRITABLE_BIT);
					if (leftMappable!=rightMappable)
						return rightMappable;

					// then non-coherent
					const bool leftCoherent = leftTypeFlags.hasFlags(flags_t::EMPF_HOST_COHERENT_BIT);
					const bool rightCoherent = rightTypeFlags.hasFlags(flags_t::EMPF_HOST_COHERENT_BIT);
					if (leftCoherent!=rightCoherent)
						return rightCoherent;

					// then non-cached
					const bool leftCached = leftTypeFlags.hasFlags(flags_t::EMPF_HOST_CACHED_BIT);
					const bool rightCached = rightTypeFlags.hasFlags(flags_t::EMPF_HOST_CACHED_BIT);
					if (leftCached!=rightCached)
						return rightCached;

					// otherwise equal
					return false;
				}
			);
			
			// go over our preferred memory types and try to service allocations from them
			core::vector<size_t> offsetsTmp;
			for (const auto memTypeIx : memoryTypePreference)
			{
				// we could try to service multiple requirements with the same allocation, but we probably don't need to try so hard
				for (auto& reqBin : allocationRequests)
				if (reqBin.first.compatibileMemoryTypeBits&(0x1<<memTypeIx))
				{
					auto& binItems = reqBin.second;
					const auto binItemCount = reqBin.second.size();
					if (!binItemCount)
						continue;

					// the `std::exclusive_scan` syntax is more effort for this
					{
						offsetsTmp.resize(binItemCount);
						offsetsTmp[0] = 0;
						for (size_t i=0; true;)
						{
							const auto* memBacked = getAsBase(binItems[i]);
							const auto& memReqs = memBacked->getMemoryReqs();
							// round up the offset to get the correct alignment
							offsetsTmp[i] = core::roundUp(offsetsTmp[i],0x1ull<<memReqs.alignmentLog2);
							// record next offset
							if (i<binItemCount-1)
								offsetsTmp[++i] = offsetsTmp[i]+memReqs.size;
							else
								break;
						}
					}
					// to replace
					core::vector<memory_backed_ptr_variant_t> failures;
					failures.reserve(binItemCount);
					// ...
					using allocate_flags_t = IDeviceMemoryAllocation::E_MEMORY_ALLOCATE_FLAGS;
					IDeviceMemoryAllocator::SAllocateInfo info = {
						.size = 0xdeadbeefBADC0FFEull, // set later
						.flags = reqBin.first.needsDeviceAddress ? allocate_flags_t::EMAF_DEVICE_ADDRESS_BIT:allocate_flags_t::EMAF_NONE,
						.memoryTypeIndex = memTypeIx,
						.dedication = nullptr
					};
					// allocate in progression of combined allocations, while trying allocate as much as possible in a single allocation
					auto binItemsIt = binItems.begin();
					for (auto firstOffsetIt=offsetsTmp.begin(); firstOffsetIt!=offsetsTmp.end(); )
					for (auto nextOffsetIt=offsetsTmp.end(); nextOffsetIt>firstOffsetIt; nextOffsetIt--)
					{
						const size_t combinedCount = std::distance(firstOffsetIt,nextOffsetIt);
						const size_t lastIx = combinedCount-1;
						// if we take `combinedCount` starting at `firstItem` their allocation would need this size
						info.size = (firstOffsetIt[lastIx]-*firstOffsetIt)+getAsBase(binItemsIt[lastIx])->getMemoryReqs().size;
						auto allocation = m_allocator->allocate(info);
						if (allocation.isValid())
						{
							// bind everything
							for (auto i=0; i<combinedCount; i++)
							{
								const auto& toBind = binItems[i];
								bool bindSuccess = false;
								const IDeviceMemoryBacked::SMemoryBinding binding = {
									.memory = allocation.memory.get(),
									// base allocation offset, plus relative offset for this batch
									.offset = allocation.offset+firstOffsetIt[i]-*firstOffsetIt
								};
								switch (toBind.index())
								{
									case 0:
										{
											const ILogicalDevice::SBindBufferMemoryInfo info =
											{
												.buffer = std::get<asset_cached_t<ICPUBuffer>*>(toBind)->get(),
												.binding = binding
											};
											bindSuccess = device->bindBufferMemory(1,&info);
										}
										break;
									case 1:
										{
											const ILogicalDevice::SBindImageMemoryInfo info =
											{
												.image = std::get<asset_cached_t<ICPUImage>*>(toBind)->get(),
												.binding = binding
											};
											bindSuccess = device->bindImageMemory(std::span(&info,1));
										}
										break;
									default:
										break;
								}
								assert(bindSuccess);
							}
							// move onto next batch
							firstOffsetIt = nextOffsetIt;
							binItemsIt += combinedCount;
							break;
						}
						// we're unable to allocate even for a single item with a dedicated allocation, skip trying then
						else if (combinedCount==1)
						{
							firstOffsetIt = nextOffsetIt;
							failures.push_back(std::move(*binItemsIt));
							binItemsIt++;
							break;
						}
					}
					// leave only the failures behind
					binItems = std::move(failures);
				}
			}

			// If we failed to allocate and bind memory from any heap, need to wipe the GPU Obj as a failure
			for (const auto& reqBin : allocationRequests)
			for (auto& req : reqBin.second)
			{
				const auto asBacked = getAsBase(req);
				assert(!asBacked->getBoundMemory().isValid());
				switch (req.index())
				{
					case 0:
						*std::get<asset_cached_t<ICPUBuffer>*>(req) = {};
						break;
					case 1:
						*std::get<asset_cached_t<ICPUImage>*>(req) = {};
						break;
					default:
						assert(false);
						break;
				}
				m_logger.log("Allocation and Binding of Device Memory for \"%\" failed, deleting GPU object.",system::ILogger::ELL_ERROR,asBacked->getObjectDebugName());
			}
			allocationRequests.clear();
		}
	
	private:
		IDeviceMemoryAllocator* m_allocator;
		system::logger_opt_ptr m_logger;
		// Because we store node pointer we can both get the `IDeviceMemoryBacked*` to bind to, and also zero out the cache entry if allocation unsuccessful
		// for this we require that the data storage for the dfsCaches' nodes does not change between request and finalizeAllocations
		using memory_backed_ptr_variant_t = std::variant<asset_cached_t<ICPUBuffer>*,asset_cached_t<ICPUImage>*>;
		core::map<MemoryRequirementBin,core::vector<memory_backed_ptr_variant_t>> allocationRequests;
};

// for dem ReBAR goodies
bool canHostWriteToMemoryRange(const IDeviceMemoryBacked::SMemoryBinding& binding, const size_t length)
{
	assert(binding.isValid());
	const auto* memory = binding.memory;
	const auto& mappedRange = memory->getMappedRange();
	return memory->isCurrentlyMapped() && memory->getCurrentMappingAccess().hasFlags(IDeviceMemoryAllocation::EMCAF_WRITE) && mappedRange.offset<=binding.offset && binding.offset+length<=mappedRange.offset+mappedRange.length;
}

//
template<asset::Asset AssetType>
struct unique_conversion_t
{
	const AssetType* canonicalAsset = nullptr;
	patch_index_t patchIndex = {};
	size_t firstCopyIx : 40 = 0u;
	size_t copyCount : 24 = 1u;
};

//
inline void setDebugName(const CAssetConverter* conv, IBackendObject* gpuObj, const core::blake3_hash_t& contentHash, const uint64_t uniqueCopyGroupID)
{
	std::ostringstream debugName;
	debugName << "Created by Converter ";
	debugName << std::hex;
	debugName << conv;
	debugName << " from Asset with hash ";
	for (const auto& byte : contentHash.data)
		debugName << uint32_t(byte) << " ";
	debugName << "for Group " << uniqueCopyGroupID;
	gpuObj->setObjectDebugName(debugName.str().c_str());
}

// Map from ContentHash to canonical asset & patch and the list of uniqueCopyGroupIDs
template<asset::Asset AssetType>
struct conversions_t
{
	public:
		// Go through the dfsCache and work out each entry's content hashes, so that we can carry out unique conversions.
		void gather(core::tuple_transform_t<dfs_cache,CAssetConverter::supported_asset_types>& dfsCaches, CAssetConverter::CHashCache* hashCache, const CAssetConverter::CCache<AssetType>* readCache)
		{
			auto& dfsCache = std::get<dfs_cache<AssetType>>(dfsCaches);
			dfsCache.for_each([&](const instance_t<AssetType>& instance, dfs_cache<AssetType>::created_t& created)->void
				{
					// compute the hash or look it up if it exists
					// We mistrust every dependency such that the eject/update if needed.
					// Its really important that the Deduplication gets performed Bottom-Up
					auto& contentHash = created.contentHash;
					PatchOverride patchOverride(*inputs,dfsCaches,instance.uniqueCopyGroupID);
					contentHash = hashCache->hash<AssetType>(
						{instance.asset,&created.patch},
						&patchOverride,
						/*.mistrustLevel =*/ 1
					);
					// failed to hash all together (only possible reason is failure of `PatchGetter` to provide a valid patch)
					if (contentHash==CAssetConverter::CHashCache::NoContentHash)
					{
						inputs->logger.log("Could not compute hash for asset %p in group %d, maybe an IPreHashed dependant's content hash is missing?",system::ILogger::ELL_ERROR,instance.asset,instance.uniqueCopyGroupID);
						return;
					}
					const auto hashAsU64 = reinterpret_cast<const uint64_t*>(contentHash.data);
					{
						inputs->logger.log("Asset (%p,%d) has hash %8llx%8llx%8llx%8llx",system::ILogger::ELL_DEBUG,instance.asset,instance.uniqueCopyGroupID,hashAsU64[0],hashAsU64[1],hashAsU64[2],hashAsU64[3]);
					}
					// if we have a read cache, lets retry looking the item up!
					if (readCache)
					{
						// We can't look up "near misses" (supersets of patches) because they'd have different hashes
						// and we can't afford to split hairs like finding overlapping buffer ranges, etc.
						// Stuff like that would require a completely different hashing/lookup strategy (or multiple fake entries).
						const auto found = readCache->find({contentHash,instance.uniqueCopyGroupID});
						if (found!=readCache->forwardMapEnd())
						{
							created.gpuObj = found->second;
							inputs->logger.log(
								"Asset (%p,%d) with hash %8llx%8llx%8llx%8llx found its GPU Object in Read Cache",system::ILogger::ELL_DEBUG,
								instance.asset,instance.uniqueCopyGroupID,hashAsU64[0],hashAsU64[1],hashAsU64[2],hashAsU64[3]
							);
							return;
						}
					}
					// The conversion request we insert needs an instance asset whose unconverted dependencies don't have missing content
					// SUPER SIMPLIFICATION: because we hash and search for readCache items bottom up (BFS), we don't need a stack (DFS) here!
					// Any dependant that's not getting a GPU object due to missing content or GPU cache object for its cache, will show up later during `getDependant`
					// An additional optimization would be to improve the `PatchGetter` to check dependants (only deps) during hashing for missing dfs cache gpu Object (no read cache) and no conversion request.
					auto* isPrehashed = dynamic_cast<const IPreHashed*>(instance.asset);
					if (isPrehashed && isPrehashed->missingContent())
					{
						inputs->logger.log(
							"PreHashed Asset (%p,%d) with hash %8llx%8llx%8llx%8llx has missing content and no GPU Object in Read Cache!",system::ILogger::ELL_ERROR,
							instance.asset,instance.uniqueCopyGroupID,hashAsU64[0],hashAsU64[1],hashAsU64[2],hashAsU64[3]
						);
						return;
					}
					// then de-duplicate the conversions needed
					const patch_index_t patchIx = {static_cast<uint64_t>(std::distance(dfsCache.nodes.data(),&created))};
					auto [inSetIt,inserted] = contentHashToCanonical.emplace(contentHash,unique_conversion_t<AssetType>{.canonicalAsset=instance.asset,.patchIndex=patchIx});
					if (!inserted)
					{
						// If an element prevented insertion, the patch must be identical!
						// Because the conversions don't care about groupIDs, the patches may be identical but not the same object in memory.
						assert(inSetIt->second.patchIndex==patchIx || dfsCache.nodes[inSetIt->second.patchIndex.value].patch==dfsCache.nodes[patchIx.value].patch);
						inSetIt->second.copyCount++;
					}
				}
			);
			
			// work out mapping of `conversionRequests` to multiple GPU objects and their copy groups via counting sort
			{
				// assign storage offsets via exclusive scan and put the `uniqueGroupID` mappings in sorted order
				auto exclScanConvReqs = [&]()->size_t
				{
					size_t sum = 0;
					for (auto& entry : contentHashToCanonical)
					{
						entry.second.firstCopyIx = sum;
						sum += entry.second.copyCount;
					}
					return sum;
				};
				gpuObjUniqueCopyGroupIDs.resize(exclScanConvReqs());
				//
				dfsCache.for_each([&](const instance_t<AssetType>& instance, dfs_cache<AssetType>::created_t& created)->void
					{
						if (created.gpuObj)
							return;
						auto found = contentHashToCanonical.find(created.contentHash);
						// may not find things because of unconverted dummy deps
						if (found!=contentHashToCanonical.end())
							gpuObjUniqueCopyGroupIDs[found->second.firstCopyIx++] = instance.uniqueCopyGroupID;
						else
						{
							inputs->logger.log(
								"No conversion request made for Asset %p in group %d, its impossible to convert.",
								system::ILogger::ELL_ERROR,instance.asset,instance.uniqueCopyGroupID
							);
						}
					}
				);
				// `{conversionRequests}.firstCopyIx` needs to be brought back down to exclusive scan form
				exclScanConvReqs();
			}

			// we now know the size of out output array
			gpuObjects.resize(gpuObjUniqueCopyGroupIDs.size());
		}

		//
		template<bool GPUObjectWhollyImmutable=false>
		void assign(const core::blake3_hash_t& contentHash, const size_t baseIx, const size_t copyIx, asset_cached_t<AssetType>::type&& gpuObj, const AssetType* asset=nullptr)
		{
			const auto hashAsU64 = reinterpret_cast<const uint64_t*>(contentHash.data);
			if constexpr (GPUObjectWhollyImmutable) // including any deps!
			if (copyIx==1) // Only warn once to reduce log spam
				inputs->logger.log(
					"Why are you creating multiple Objects for asset content %8llx%8llx%8llx%8llx, when they are a readonly GPU Object Type with no dependants!?",
					system::ILogger::ELL_PERFORMANCE,hashAsU64[0],hashAsU64[1],hashAsU64[2],hashAsU64[3]
				);
			//
			if (!gpuObj)
			{
				inputs->logger.log(
					"Failed to create GPU Object for asset content %8llx%8llx%8llx%8llx",
					system::ILogger::ELL_ERROR,hashAsU64[0],hashAsU64[1],hashAsU64[2],hashAsU64[3]
				);
				return;
			}
			auto output = gpuObjects.data()+copyIx+baseIx;
			output->value = std::move(gpuObj);
			const uint64_t uniqueCopyGroupID = gpuObjUniqueCopyGroupIDs[copyIx+baseIx];
			if constexpr (std::is_same_v<AssetType,ICPUBuffer> || std::is_same_v<AssetType,ICPUImage>)
			{
				const auto constrainMask = inputs->constrainMemoryTypeBits(uniqueCopyGroupID,asset,contentHash,output->value.get());
				if (!deferredAllocator->request(output,constrainMask))
					return;
			}
			// set debug names on everything!
			setDebugName(conv,output->get(),contentHash,uniqueCopyGroupID);
		}

		// Since the dfsCache has the original asset pointers as keys, we map in reverse (multiple `instance_t` can map to the same unique content hash and GPU object)
		void propagateToCaches(dfs_cache<AssetType>& dfsCache, CAssetConverter::SReserveResult::staging_cache_t<AssetType>& stagingCache)
		{
			assert(gpuObjUniqueCopyGroupIDs.empty());
			dfsCache.for_each([&](const instance_t<AssetType>& instance, dfs_cache<AssetType>::created_t& created)->void
				{
					// already found in read cache and not converted
					if (created.gpuObj)
						return;

					const auto uniqueCopyGroupID = instance.uniqueCopyGroupID;
					const auto& contentHash = created.contentHash;
					const auto hashAsU64 = reinterpret_cast<const uint64_t*>(contentHash.data);

					auto found = contentHashToCanonical.find(contentHash);
					// can happen if deps were unconverted dummies
					if (found==contentHashToCanonical.end())
					{
						if (contentHash!=CAssetConverter::CHashCache::NoContentHash)
							inputs->logger.log(
								"Could not find GPU Object for Asset %p in group %ull with Content Hash %8llx%8llx%8llx%8llx",
								system::ILogger::ELL_ERROR, instance.asset, uniqueCopyGroupID, hashAsU64[0], hashAsU64[1], hashAsU64[2], hashAsU64[3]
							);
						return;
					}
					// unhashables were not supposed to be added to conversion requests
					assert(contentHash!=CAssetConverter::CHashCache::NoContentHash);

					const auto copyIx = found->second.firstCopyIx++;
					auto& gpuObj = gpuObjects[copyIx];
					if (!gpuObj)
					{
						inputs->logger.log(
							"Creation of GPU Object (or its dependents) for Content Hash %8llx%8llx%8llx%8llx Copy Index %d from Canonical Asset %p Failed.",
							system::ILogger::ELL_ERROR, hashAsU64[0], hashAsU64[1], hashAsU64[2], hashAsU64[3], copyIx, found->second.canonicalAsset
						);
						return;
					}
					// insert into staging cache
					stagingCache.emplace(gpuObj.get(),CAssetConverter::SReserveResult::staging_cache_key<AssetType>{gpuObj.value,typename CAssetConverter::CCache<AssetType>::key_t(contentHash,uniqueCopyGroupID)});
					// propagate back to dfsCache
					created.gpuObj = std::move(gpuObj);
				}
			);
		}

		const CAssetConverter* conv;
		const CAssetConverter::SInputs* inputs;
		MetaDeviceMemoryAllocator* deferredAllocator;
		core::unordered_map<core::blake3_hash_t,unique_conversion_t<AssetType>> contentHashToCanonical;
		core::vector<size_t> gpuObjUniqueCopyGroupIDs;
		core::vector<asset_cached_t<AssetType>> gpuObjects;
};

//
auto CAssetConverter::reserve(const SInputs& inputs) -> SReserveResult
{
	auto* const device = m_params.device;
	if (inputs.readCache && inputs.readCache->m_params.device!=m_params.device)
	{
		inputs.logger.log("Read Cache's owning device %p not compatible with this cache's owning device %p.",system::ILogger::ELL_ERROR,inputs.readCache->m_params.device,m_params.device);
		return {};
	}
	if (inputs.pipelineCache && inputs.pipelineCache->getOriginDevice()!=device)
	{
		inputs.logger.log("Pipeline Cache's owning device %p not compatible with this cache's owning device %p.",system::ILogger::ELL_ERROR,inputs.pipelineCache->getOriginDevice(),m_params.device);
		return {};
	}

	SReserveResult retval = {};
	
	// this will allow us to look up the conversion parameter (actual patch for an asset) and therefore write the GPUObject to the correct place in the return value
	core::vector<input_metadata_t> inputsMetadata[core::type_list_size_v<supported_asset_types>];
	// One would think that we first need an (AssetPtr,Patch) -> ContentHash map and then a ContentHash -> GPUObj map to
	// save ourselves iterating over redundant assets. The truth is that we going from a ContentHash to GPUObj is blazing fast.
	core::tuple_transform_t<dfs_cache,supported_asset_types> dfsCaches = {};

	{
		// gather all dependencies (DFS graph search) and patch, this happens top-down
		// do not deduplicate/merge assets at this stage, only patch GPU creation parameters
		{
			// stack is nice and polymorphic
			core::stack<patched_instance_t> stack = {};

			// initialize stacks
			auto initialize = [&]<typename AssetType>(const std::span<const AssetType* const> assets)->void
			{
				const auto count = assets.size();
				const auto& patches = std::get<SInputs::patch_span_t<AssetType>>(inputs.patches);
				// size and fill the result array with nullptr
				std::get<SReserveResult::vector_t<AssetType>>(retval.m_gpuObjects).resize(count);
				// size the final patch mapping
				auto& metadata = inputsMetadata[index_of_v<AssetType,supported_asset_types>];
				metadata.resize(count);
				for (size_t i=0; i<count; i++)
				if (auto asset=assets[i]; asset) // skip invalid inputs silently
				{
					patch_t<AssetType> patch = {asset};
					if (i<patches.size())
					{
						// derived patch has to be valid
						if (!patch.valid(device))
							continue;
						// the overriden one too
						auto overidepatch = patches[i];
						if (!overidepatch.valid(device))
							continue;
						// the combination must be a success (doesn't need to be valid though)
						bool combineSuccess;
						std::tie(combineSuccess,patch) = patch.combine(overidepatch);
						if (!combineSuccess)
							continue;
					}
					const size_t uniqueGroupID = inputs.getDependantUniqueCopyGroupID(0xdeadbeefBADC0FFEull,nullptr,asset);
					metadata[i] = DFSVisitor<AssetType>{
						.inputs = inputs,
						.device = device,
						.dfsCaches = dfsCaches,
						.stack = stack
					}.descend_impl_impl<AssetType>({},{asset,uniqueGroupID},std::move(patch));
				}
			};
			core::for_each_in_tuple(inputs.assets,initialize);

			// wrap in templated lambda
			auto visit = [&]<Asset AssetType>(const patched_instance_t& user)->void
			{
				// we don't use the result yet
				const bool success = AssetVisitor<DFSVisitor<AssetType>>{
					{
						.inputs = inputs,
						.device = device,
						.dfsCaches = dfsCaches,
						.stack = stack
					},
					// construct a casted instance type
					{static_cast<const AssetType*>(user.instance.asset),user.instance.uniqueCopyGroupID},
					// This is fairly risky, because its a reference to a vector element while we're pushing new elements to a vector during DFS
					// however we have a DAG and AssetType cannot depend on the same AssetType and we don't recurse inside `visit` so we never grow our own vector.
					std::get<dfs_cache<AssetType>>(dfsCaches).nodes[user.patchIx.value].patch
				}();
			};
			// Perform Depth First Search of the Asset Graph
			while (!stack.empty())
			{
				auto entry = stack.top();
				stack.pop();
				// everything we popped has already been cached in dfsCache, now time to go over dependents
				switch (entry.instance.asset->getAssetType())
				{
					case ICPUDescriptorSetLayout::AssetType:
						visit.template operator()<ICPUDescriptorSetLayout>(entry);
						break;
					case ICPUPipelineLayout::AssetType:
						visit.template operator()<ICPUPipelineLayout>(entry);
						break;
					case ICPUComputePipeline::AssetType:
						visit.template operator()<ICPUComputePipeline>(entry);
						break;
					case ICPUGraphicsPipeline::AssetType:
						visit.template operator()<ICPUGraphicsPipeline>(entry);
						break;
					case ICPUDescriptorSet::AssetType:
						visit.template operator()<ICPUDescriptorSet>(entry);
						break;
					case ICPUBufferView::AssetType:
						visit.template operator()<ICPUBufferView>(entry);
						break;
					case ICPUImageView::AssetType:
						visit.template operator()<ICPUImageView>(entry);
						break;
					case ICPUTopLevelAccelerationStructure::AssetType:
						visit.template operator()<ICPUTopLevelAccelerationStructure>(entry);
						break;
					// these assets have no dependants, should have never been pushed on the stack
					default:
						assert(false);
						break;
				}
			}
			// special pass to promote image formats
			std::get<dfs_cache<ICPUImage>>(dfsCaches).for_each([device,&inputs](const instance_t<ICPUImage>& instance, dfs_cache<ICPUImage>::created_t& created)->void
				{
					auto& patch = created.patch;
					const auto* physDev = device->getPhysicalDevice();
					const bool canPromoteFormat = patch.canAttemptFormatPromotion();
					// return true is success
					auto promoteFormat = [=]()->E_FORMAT
					{
						const auto origFormat = instance.asset->getCreationParameters().format;
						// Why don't we check format creation possibility for non-promotable images?
						if (canPromoteFormat)
							return origFormat;
						// We'd have to track (extended) usages from views with mutated formats separately from usages from views of the same format.
						// And mutable format creation flag will always preclude ANY format promotion, therefore all usages come from views that have the same initial format!
						IPhysicalDevice::SImageFormatPromotionRequest req = {
							.originalFormat = origFormat,
							.usages = {patch.usageFlags|patch.stencilUsage}
						};
						req.usages.linearlySampledImage = patch.linearlySampled;
						if (req.usages.storageImage) // we require this anyway
							req.usages.storageImageStoreWithoutFormat = true;
						req.usages.storageImageAtomic = patch.storageAtomic;
						req.usages.storageImageLoadWithoutFormat = patch.storageImageLoadWithoutFormat;
						req.usages.depthCompareSampledImage = patch.depthCompareSampledImage;
						const auto format = physDev->promoteImageFormat(req,static_cast<IGPUImage::TILING>(patch.linearTiling));
						if (format==EF_UNKNOWN)
						{
							inputs.logger.log(
								"ICPUImage %p in group %d with NEXT patch index %d cannot be created with its original format due to its usages and failed to promote to a different format!",
								system::ILogger::ELL_ERROR,instance.asset,instance.uniqueCopyGroupID,created.next
							);
						}
						return format;
					};
					// first promote try
					patch.format = promoteFormat();
					if (patch.format==EF_UNKNOWN)
						return;
					// after promoted format is known we can proceed with mip tail extenion and tagging if mipmaps get recomputed
					patch.mipLevels = inputs.getMipLevelCount(instance.uniqueCopyGroupID,instance.asset,patch);
					// important to call AFTER the mipchain length is known
					patch.recomputeMips = inputs.needToRecomputeMips(instance.uniqueCopyGroupID,instance.asset,patch);
					// zero out invalid return values
					for (uint16_t l=1; l<patch.mipLevels; l++)
					{
						const auto levelMask = 0x1<<(l-1);
						if ((patch.recomputeMips&levelMask)==0)
							continue;
						const auto prevLevel = l-1;
						const auto prevLevelMask = 0x1<<(prevLevel-1);
						// marked as recompute but has no source data on previous level
						const bool noPrevRecompute = prevLevel==0 || (patch.recomputeMips&prevLevelMask)==0;
						if (noPrevRecompute && !instance.asset->getRegions(l).empty())
						{
							inputs.logger.log(
								"`SInputs::needToRecomputeMips` callback erroneously marked mip level %d of ICPUImage %p in group %d with NEXT patch index %d for recomputation, no source data available! Unmarking.",
								system::ILogger::ELL_ERROR,l,instance.asset,instance.uniqueCopyGroupID,created.next
							);
							patch.recomputeMips ^= levelMask;
						}
					}
					// also trim anything above
					patch.recomputeMips &= (0x1u<<(patch.mipLevels-1))-1;
					// If any mip level will be recomputed we need to sample from others. Stencil can't be written to with a storage image, so only add to regular usage.
					if (patch.recomputeMips)
					{
						patch.usageFlags |= IGPUImage::EUF_SAMPLED_BIT;
						// usage changed
						const auto firstFormat = patch.format;
						patch.format = promoteFormat();
						// if failed then
						if (patch.format==EF_UNKNOWN)
						{
							// undo our offending change
							patch.recomputeMips = 0;
							// and restore the original promotion
							patch.format = firstFormat;
						}
					}
				}
			);
			// special pass to propagate Motion Acceleration Structure flag upwards from BLAS to referencing TLAS
			std::get<dfs_cache<ICPUTopLevelAccelerationStructure>>(dfsCaches).for_each([device,&inputs,&dfsCaches](const instance_t<ICPUTopLevelAccelerationStructure>& assetInstance, dfs_cache<ICPUTopLevelAccelerationStructure>::created_t& created)->void
				{
					auto& patch = created.patch;
					// we already have motion, can stop searching
					if (patch.isMotion)
						return;
					auto visitor = AssetVisitor<CheckBLASPatchMotions>{
						{
							.inputs = inputs,
							.visitedBLASes = std::get<dfs_cache<ICPUBottomLevelAccelerationStructure>>(dfsCaches)
						},
						// construct a casted instance type
						{assetInstance.asset,assetInstance.uniqueCopyGroupID},
						patch
					};
					// don't care about success, I've abused the termination criteria, will return false sometimes
					visitor();
					// I don't need to check if the new patch is valid, because we checked if the Motion Raytracing feature is enabled when checking BLASes for validity
					patch.isMotion = visitor.isMotion;
				}
			);
		}
		//! `inputsMetadata` is now constant!
		//! `dfsCache` keys are now constant!

		// can now spawn our own hash cache
		retval.m_hashCache = core::make_smart_refctd_ptr<CHashCache>();

		MetaDeviceMemoryAllocator deferredAllocator(inputs.allocator ? inputs.allocator:device,inputs.logger);

		// BLAS and TLAS creation is somewhat delayed by buffer creation and allocation
		struct DeferredASCreationParams
		{
			asset_cached_t<ICPUBuffer> storage = {};
			uint64_t scratchSize = 0;
			uint64_t buildSize = 0;
		};
		core::vector<DeferredASCreationParams> accelerationStructureParams[2];
		// Deduplication, Creation and Propagation
		auto dedupCreateProp = [&]<Asset AssetType>()->conversions_t<AssetType>
		{
			// This map contains the assets by-hash, identical asset+patch hash the same.
			// It only has entries for GPU objects that need to be created
			conversions_t<AssetType> conversionRequests = {this,&inputs,&deferredAllocator};

			//
			const CCache<AssetType>* readCache = inputs.readCache ? (&std::get<CCache<AssetType>>(inputs.readCache->m_caches)):nullptr;
			conversionRequests.gather(dfsCaches,retval.m_hashCache.get(),readCache);
			
			//
			GetDependantVisitBase<AssetType> visitBase = {
				.inputs = inputs,
				.dfsCaches = dfsCaches
			};

			// Dispatch to correct creation of GPU objects
			auto& dfsCache = std::get<dfs_cache<AssetType>>(dfsCaches);
			if constexpr (std::is_same_v<AssetType,ICPUSampler>)
			{
				for (auto& entry : conversionRequests.contentHashToCanonical)
				for (auto i=0ull; i<entry.second.copyCount; i++)
					conversionRequests.template assign<true>(entry.first,entry.second.firstCopyIx,i,device->createSampler(entry.second.canonicalAsset->getParams()));
			}
			if constexpr (std::is_same_v<AssetType,ICPUBuffer>)
			{
				for (auto& entry : conversionRequests.contentHashToCanonical)
				for (auto i=0ull; i<entry.second.copyCount; i++)
				{
					const ICPUBuffer* asset = entry.second.canonicalAsset;
					const auto& patch = dfsCache.nodes[entry.second.patchIndex.value].patch;
					//
					IGPUBuffer::SCreationParams params = {};
					params.size = asset->getSize();
					params.usage = patch.usage;
					// concurrent ownership if any
					const auto outIx = i+entry.second.firstCopyIx;
					const auto uniqueCopyGroupID = conversionRequests.gpuObjUniqueCopyGroupIDs[outIx];
					const auto queueFamilies =  inputs.getSharedOwnershipQueueFamilies(uniqueCopyGroupID,asset,patch);
					params.queueFamilyIndexCount = queueFamilies.size();
					params.queueFamilyIndices = queueFamilies.data();
					// if creation successful, we will request some memory allocation to bind to, and if thats okay we preliminarily request a conversion
					conversionRequests.assign(entry.first,entry.second.firstCopyIx,i,device->createBuffer(std::move(params)),asset);
				}
			}
			if constexpr (std::is_same_v<AssetType,ICPUBottomLevelAccelerationStructure> || std::is_same_v<AssetType,ICPUTopLevelAccelerationStructure>)
			{
				using mem_prop_f = IDeviceMemoryAllocation::E_MEMORY_PROPERTY_FLAGS;
				const auto deviceBuildMemoryTypes = device->getPhysicalDevice()->getMemoryTypeBitsFromMemoryTypeFlags(mem_prop_f::EMPF_DEVICE_LOCAL_BIT);
				const auto hostBuildMemoryTypes = device->getPhysicalDevice()->getMemoryTypeBitsFromMemoryTypeFlags(mem_prop_f::EMPF_DEVICE_LOCAL_BIT|mem_prop_f::EMPF_HOST_WRITABLE_BIT|mem_prop_f::EMPF_HOST_CACHED_BIT);
				
				constexpr bool IsTLAS = std::is_same_v<AssetType,ICPUTopLevelAccelerationStructure>;
				accelerationStructureParams[IsTLAS].resize(conversionRequests.gpuObjects.size());
				for (auto& entry : conversionRequests.contentHashToCanonical)
				for (auto i=0ull; i<entry.second.copyCount; i++)
				{
					const auto* as = entry.second.canonicalAsset;
					const auto patchIx = entry.second.patchIndex.value;
					const auto& patch = dfsCache.nodes[patchIx].patch;
					const bool motionBlur = patch.isMotion;
					const auto buildFlags = patch.getBuildFlags(as);
					const auto outIx = i+entry.second.firstCopyIx;
					const auto uniqueCopyGroupID = conversionRequests.gpuObjUniqueCopyGroupIDs[outIx];
					// prevent CPU hangs by making sure allocator big enough to service us in worst case
					const auto minScratchAllocSize = patch.hostBuild ? inputs.scratchForHostASBuildMinAllocSize:inputs.scratchForDeviceASBuildMinAllocSize;
					uint64_t buildSize = 0;
					auto incrementBuildSize = [minScratchAllocSize,&buildSize](const uint64_t size, const uint32_t alignment)->void
					{
						// account for fragmentation and misalignment
						buildSize += hlsl::max<uint64_t>(size,minScratchAllocSize)+hlsl::max<uint32_t>(minScratchAllocSize,alignment)*2;
					};
					ILogicalDevice::AccelerationStructureBuildSizes sizes = {};
					const auto hashAsU64 = reinterpret_cast<const uint64_t*>(entry.first.data);
					{
						if constexpr (IsTLAS)
						{
							// TLAS can't check for the BLASes existing yet, because they haven't had their backing buffers allocated yet
							const auto instanceCount = as->getInstances().size();
							sizes = device->getAccelerationStructureBuildSizes(patch.hostBuild,buildFlags,motionBlur,instanceCount);
							// all instances need to be aligned to 16 bytes so alignment irrelevant (everything can be tightly packed) and implicit
							const uint64_t worstCaseInstanceSize = motionBlur ? IGPUTopLevelAccelerationStructure::DevicePolymorphicInstance::LargestUnionMemberSize:sizeof(IGPUTopLevelAccelerationStructure::DeviceStaticInstance);
							// worst case approximation is fine here
							incrementBuildSize(worstCaseInstanceSize*instanceCount,16);
							incrementBuildSize(sizeof(uint64_t)*instanceCount,alignof(uint64_t));
						}
						else
						{
							const uint32_t* pPrimitiveCounts = as->getGeometryPrimitiveCounts().data();
							if (buildFlags.hasFlags(ICPUBottomLevelAccelerationStructure::BUILD_FLAGS::GEOMETRY_TYPE_IS_AABB_BIT))
							{
								const auto geoms = as->getAABBGeometries();
								sizes = device->getAccelerationStructureBuildSizes(patch.hostBuild,buildFlags,motionBlur,geoms,pPrimitiveCounts);
								for (const auto& geom : geoms)
								if (const auto aabbCount=*(pPrimitiveCounts++); aabbCount)
									incrementBuildSize(aabbCount*geom.stride,alignof(float));
							}
							else
							{
								const auto geoms = as->getTriangleGeometries();
								sizes = device->getAccelerationStructureBuildSizes(patch.hostBuild,buildFlags,motionBlur,geoms,pPrimitiveCounts);
								for (const auto& geom : geoms)
								if (const auto triCount=*(pPrimitiveCounts++); triCount)
								{
									auto size = geom.vertexStride*(geom.vertexData[1] ? 2:1)*(geom.maxVertex+1);
									uint16_t alignment = hlsl::max(0x1u<<hlsl::findLSB(geom.vertexStride),32u);
									if (geom.hasTransform())
									{
										size = core::alignUp(size,alignof(float))+sizeof(hlsl::float32_t3x4);
										alignment = hlsl::max<uint16_t>(alignof(float),alignment);
									}
									uint16_t indexSize = 0;
									switch (geom.indexType)
									{
										case E_INDEX_TYPE::EIT_16BIT:
											indexSize = sizeof(uint16_t);
											break;
										case E_INDEX_TYPE::EIT_32BIT:
											indexSize = sizeof(uint32_t);
											break;
										default:
											break;
									}
									if (indexSize)
									{
										size = core::alignUp(size,indexSize)+triCount*3*indexSize;
										alignment = hlsl::max<uint16_t>(indexSize,alignment);
									}
									//inputs.logger.log("%p Triangle Data Size %d Align %d",system::ILogger::ELL_DEBUG,as,size,alignment);
									incrementBuildSize(size,alignment);
								}
							}
						}
					}
					if (buildSize==0 || sizes.buildScratchSize==0)
					{
						inputs.logger.log(
							"Build Size Input is 0 or failed the call to `ILogicalDevice::getAccelerationStructureBuildSizes` for Acceleration Structure %8llx%8llx%8llx%8llx",
							system::ILogger::ELL_ERROR,hashAsU64[0],hashAsU64[1],hashAsU64[2],hashAsU64[3]
						);
						continue;
					}
					//
					incrementBuildSize(sizes.buildScratchSize,device->getPhysicalDevice()->getLimits().minAccelerationStructureScratchOffsetAlignment);
					//inputs.logger.log("%p Scratch Size %d Combined %d",system::ILogger::ELL_DEBUG,as,sizes.buildScratchSize,buildSize);

					// we need to save the buffer in a side-channel for later
					auto& out = accelerationStructureParams[IsTLAS][entry.second.firstCopyIx+i];
					// this is where it gets a bit weird, we need to create a buffer to back the acceleration structure
					{
						IGPUBuffer::SCreationParams params = {};
						constexpr size_t MinASBufferAlignment = 256u;
						params.size = core::roundUp(sizes.accelerationStructureSize,MinASBufferAlignment);
						params.usage = IGPUBuffer::E_USAGE_FLAGS::EUF_ACCELERATION_STRUCTURE_STORAGE_BIT|IGPUBuffer::E_USAGE_FLAGS::EUF_SHADER_DEVICE_ADDRESS_BIT;
						// concurrent ownership if any
						const auto queueFamilies = inputs.getSharedOwnershipQueueFamilies(uniqueCopyGroupID,as,patch);
						params.queueFamilyIndexCount = queueFamilies.size();
						params.queueFamilyIndices = queueFamilies.data();
						out.storage.value = device->createBuffer(std::move(params));
						if (out.storage)
						{
							nbl::video::setDebugName(this,out.storage.value.get(),entry.first,uniqueCopyGroupID);
							if (!deferredAllocator.request(&out.storage,patch.hostBuild ? hostBuildMemoryTypes:deviceBuildMemoryTypes))
								continue;
						}
					}
					out.scratchSize = sizes.buildScratchSize;
					out.buildSize = buildSize;
				}
			}
			if constexpr (std::is_same_v<AssetType,ICPUImage>)
			{
				for (auto& entry : conversionRequests.contentHashToCanonical)
				for (auto i=0ull; i<entry.second.copyCount; i++)
				{
					const ICPUImage* asset = entry.second.canonicalAsset;
					const auto& node = dfsCache.nodes[entry.second.patchIndex.value];
					const auto& patch = node.patch;
					//
					IGPUImage::SCreationParams params = {};
					params = asset->getCreationParameters();
					// deal with format
					params.format = patch.format;
					const auto& allowedBaseFormatUsages = device->getPhysicalDevice()->getImageFormatUsagesOptimalTiling()[params.format];
					if (allowedBaseFormatUsages==IPhysicalDevice::SFormatImageUsages::SUsage{})
					{
						const auto hashAsU64 = reinterpret_cast<const uint64_t*>(node.contentHash.data);
						inputs.logger.log(
							"Image Format %d is wholly unsupported by the device, cannot create Image with asset hash %8llx%8llx%8llx%8llx",
							system::ILogger::ELL_ERROR,params.format,hashAsU64[0],hashAsU64[1],hashAsU64[2],hashAsU64[3]
						);
						continue;
					}
					//
					params.mipLevels = patch.mipLevels;
					// patch creation params
					using create_flags_t = IGPUImage::E_CREATE_FLAGS;
					if (patch.mutableFormat)
						params.flags |= create_flags_t::ECF_MUTABLE_FORMAT_BIT;
					if (patch.cubeCompatible)
						params.flags |= create_flags_t::ECF_CUBE_COMPATIBLE_BIT;
					if (patch._3Dbut2DArrayCompatible)
						params.flags |= create_flags_t::ECF_2D_ARRAY_COMPATIBLE_BIT;
					if (patch.uncompressedViewOfCompressed)
						params.flags |= create_flags_t::ECF_BLOCK_TEXEL_VIEW_COMPATIBLE_BIT;
					params.usage = patch.usageFlags;
					// Now add STORAGE USAGE to creation parameters if mip-maps need to be recomputed
					if (patch.recomputeMips)
					{
						params.usage |= IGPUImage::E_USAGE_FLAGS::EUF_STORAGE_BIT;
						// formats like SRGB etc. can't be stored to
						if (!allowedBaseFormatUsages.storageImage)
						{
							// but image views with type-punned formats that are store-able can be created
							params.flags |= create_flags_t::ECF_MUTABLE_FORMAT_BIT;
							// making UINT views of whole block compressed textures requires some special care (even though we can't encode yet)
							if (isBlockCompressionFormat(patch.format))
								params.flags |= create_flags_t::ECF_2D_ARRAY_COMPATIBLE_BIT;
						}
					}
					params.stencilUsage = patch.stencilUsage;
					// time to check if the format supports all the usages or not
					{
						IPhysicalDevice::SFormatImageUsages::SUsage finalUsages(params.usage|params.stencilUsage);
						finalUsages.linearlySampledImage = patch.linearlySampled;
						finalUsages.storageImageAtomic = patch.storageAtomic;
						finalUsages.storageImageLoadWithoutFormat = patch.storageImageLoadWithoutFormat;
						finalUsages.depthCompareSampledImage = patch.depthCompareSampledImage;
						// we have some usages not allowed on this base format, so they must have been added for views with different formats
						if ((finalUsages&allowedBaseFormatUsages)!=finalUsages)
						{
							// but for this a mutable format and extended usage creation flag is needed!
							params.flags |= create_flags_t::ECF_EXTENDED_USAGE_BIT;
							params.flags |= create_flags_t::ECF_MUTABLE_FORMAT_BIT; // Question: do we always add it, or require it be present?
						}
					}
					// concurrent ownership if any
					const auto outIx = i+entry.second.firstCopyIx;
					const auto uniqueCopyGroupID = conversionRequests.gpuObjUniqueCopyGroupIDs[outIx];
					const auto queueFamilies =  inputs.getSharedOwnershipQueueFamilies(uniqueCopyGroupID,asset,patch);
					params.queueFamilyIndexCount = queueFamilies.size();
					params.queueFamilyIndices = queueFamilies.data();
					// gpu image specifics
					params.tiling = static_cast<IGPUImage::TILING>(patch.linearTiling);
					params.preinitialized = false;
					// if creation successful, we will request some memory allocation to bind to
					conversionRequests.assign(entry.first,entry.second.firstCopyIx,i,device->createImage(std::move(params)),asset);
				}
			}
			if constexpr (std::is_same_v<AssetType,ICPUBufferView>)
			{
				for (auto& entry : conversionRequests.contentHashToCanonical)
				{
					const ICPUBufferView* asset = entry.second.canonicalAsset;
					const auto& patch = dfsCache.nodes[entry.second.patchIndex.value].patch;
					for (auto i=0ull; i<entry.second.copyCount; i++)
					{
						const auto outIx = i+entry.second.firstCopyIx;
						const auto uniqueCopyGroupID = conversionRequests.gpuObjUniqueCopyGroupIDs[outIx];
						AssetVisitor<GetDependantVisit<ICPUBufferView>> visitor = {
							{visitBase},
							{asset,uniqueCopyGroupID},
							patch
						};
						if (!visitor())
							continue;
						// no format promotion for buffer views
						conversionRequests.assign(entry.first,entry.second.firstCopyIx,i,device->createBufferView(visitor.underlying,asset->getFormat()));
					}
				}
			}
			if constexpr (std::is_same_v<AssetType,ICPUImageView>)
			{
				for (auto& entry : conversionRequests.contentHashToCanonical)
				{
					const ICPUImageView* asset = entry.second.canonicalAsset;
					const auto& cpuParams = asset->getCreationParameters();
					const auto& patch = dfsCache.nodes[entry.second.patchIndex.value].patch;
					for (auto i=0ull; i<entry.second.copyCount; i++)
					{
						const auto outIx = i+entry.second.firstCopyIx;
						const auto uniqueCopyGroupID = conversionRequests.gpuObjUniqueCopyGroupIDs[outIx];
						AssetVisitor<GetDependantVisit<ICPUImageView>> visitor = {
							{visitBase},
							{asset,uniqueCopyGroupID},
							patch
						};
						if (!visitor())
							continue;
						// format of the underlying image
						const auto& imageParams = visitor.image->getCreationParameters();
						const auto baseFormat = imageParams.format;
						//
						IGPUImageView::SCreationParams params = {};
						params.flags = cpuParams.flags;
						// EXPERIMENTAL: Only restrict ourselves to an explicit usage list if our view's format prevents all parent image's usages!
						//const auto& validUsages = device->getPhysicalDevice()->getImageFormatUsages(visitor.image->getTiling());
						//const auto& allowedForViewsFormat = validUsages[cpuParams.format];
						//const IPhysicalDevice::SFormatImageUsages::SUsage allImageUsages(imageParams.usage|imageParams.stencilUsage);
						//if (!allImageUsages.isSubsetOf(allowedForViewsFormat))
							params.subUsages = patch.subUsages;
						params.image = std::move(visitor.image);
						params.viewType = cpuParams.viewType;
						// does the format get promoted
						params.format = patch.formatFollowsImage() ? baseFormat:cpuParams.format;
						memcpy(&params.components,&cpuParams.components,sizeof(params.components));
						params.subresourceRange = cpuParams.subresourceRange;
						// if underlying image had mip-chain extended then we extend our own
						if (imageParams.mipLevels!=visitor.oldMipCount)
							params.subresourceRange.levelCount = imageParams.mipLevels-params.subresourceRange.baseMipLevel;
						conversionRequests.assign(entry.first,entry.second.firstCopyIx,i,device->createImageView(std::move(params)));
					}
				}
			}
			if constexpr (std::is_same_v<AssetType,ICPUShader>)
			{
				ILogicalDevice::SShaderCreationParameters createParams = {
					.optimizer = m_params.optimizer.get(),
					.readCache = inputs.readShaderCache,
					.writeCache = inputs.writeShaderCache
				};
				for (auto& entry : conversionRequests.contentHashToCanonical)
				for (auto i=0ull; i<entry.second.copyCount; i++)
				{
					createParams.cpushader = entry.second.canonicalAsset;
					conversionRequests.assign(entry.first,entry.second.firstCopyIx,i,device->createShader(createParams));
				}
			}
			if constexpr (std::is_same_v<AssetType,ICPUDescriptorSetLayout>)
			{
				for (auto& entry : conversionRequests.contentHashToCanonical)
				{
					const ICPUDescriptorSetLayout* asset = entry.second.canonicalAsset;
					// there is no patching possible for this asset
					using storage_range_index_t = ICPUDescriptorSetLayout::CBindingRedirect::storage_range_index_t;
					// rebuild bindings from CPU info
					core::vector<IGPUDescriptorSetLayout::SBinding> bindings;
					bindings.reserve(asset->getTotalBindingCount());
					for (uint32_t t=0u; t<static_cast<uint32_t>(IDescriptor::E_TYPE::ET_COUNT); t++)
					{
						const auto type = static_cast<IDescriptor::E_TYPE>(t);
						const auto& redirect = asset->getDescriptorRedirect(type);
						const auto count = redirect.getBindingCount();
						for (auto i=0u; i<count; i++)
						{
							const storage_range_index_t storageRangeIx(i);
							const auto binding = redirect.getBinding(storageRangeIx);
							bindings.push_back(IGPUDescriptorSetLayout::SBinding{
								.binding = binding.data,
								.type = type,
								.createFlags = redirect.getCreateFlags(storageRangeIx),
								.stageFlags = redirect.getStageFlags(storageRangeIx),
								.count = redirect.getCount(storageRangeIx),
								.immutableSamplers = nullptr
							});
						}
					}
					// to let us know what binding has immutables, and set up a mapping
					core::vector<core::smart_refctd_ptr<IGPUSampler>> immutableSamplers(asset->getImmutableSamplers().size());
					{
						const auto& immutableSamplerRedirects = asset->getImmutableSamplerRedirect();
						auto outImmutableSamplers = immutableSamplers.data();
						for (auto j=0u; j<immutableSamplerRedirects.getBindingCount(); j++)
						{
							const storage_range_index_t storageRangeIx(j);
							// assuming the asset was validly created, the binding must exist
							const auto binding = immutableSamplerRedirects.getBinding(storageRangeIx);
							// TODO: optimize this, the `bindings` are sorted within a given type (can do binary search in SAMPLER and COMBINED)
							auto outBinding = std::find_if(bindings.begin(),bindings.end(),[=](const IGPUDescriptorSetLayout::SBinding& item)->bool{return item.binding==binding.data;});
							// the binding must be findable, otherwise above code logic is wrong
							assert(outBinding!=bindings.end());
							// set up the mapping
							outBinding->immutableSamplers = immutableSamplers.data()+immutableSamplerRedirects.getStorageOffset(storageRangeIx).data;
						}
					}
					//
					for (auto i=0ull; i<entry.second.copyCount; i++)
					{
						const auto outIx = i+entry.second.firstCopyIx;
						const auto uniqueCopyGroupID = conversionRequests.gpuObjUniqueCopyGroupIDs[outIx];
						// visit the immutables, can't be factored out because depending on groupID the dependant might change
						AssetVisitor<GetDependantVisit<ICPUDescriptorSetLayout>> visitor = {
							{
								visitBase,
								immutableSamplers.data()
							},
							{asset,uniqueCopyGroupID},
							{} // no patch
						};
						if (!visitor())
							continue;
						conversionRequests.assign(entry.first,entry.second.firstCopyIx,i,device->createDescriptorSetLayout(bindings));
					}
				}
			}
			if constexpr (std::is_same_v<AssetType,ICPUPipelineLayout>)
			{
				core::vector<asset::SPushConstantRange> pcRanges;
				pcRanges.reserve(CSPIRVIntrospector::MaxPushConstantsSize);
				for (auto& entry : conversionRequests.contentHashToCanonical)
				{
					const ICPUPipelineLayout* asset = entry.second.canonicalAsset;
					const auto& patch = dfsCache.nodes[entry.second.patchIndex.value].patch;
					// time for some RLE
					{
						pcRanges.clear();
						asset::SPushConstantRange prev = {
							.stageFlags = IGPUShader::E_SHADER_STAGE::ESS_UNKNOWN,
							.offset = 0,
							.size = 0
						};
						for (auto byte=0u; byte<patch.pushConstantBytes.size(); byte++)
						{
							const auto current = patch.pushConstantBytes[byte].value;
							if (current!=prev.stageFlags)
							{
								if (prev.stageFlags)
								{
									prev.size = byte-prev.offset;
									pcRanges.push_back(prev);
								}
								prev.stageFlags = current;
								prev.offset = byte;
							}
						}
						if (prev.stageFlags)
						{
							prev.size = CSPIRVIntrospector::MaxPushConstantsSize-prev.offset;
							pcRanges.push_back(prev);
						}
					}
					for (auto i=0ull; i<entry.second.copyCount; i++)
					{
						const auto outIx = i+entry.second.firstCopyIx;
						const auto uniqueCopyGroupID = conversionRequests.gpuObjUniqueCopyGroupIDs[outIx];
						AssetVisitor<GetDependantVisit<ICPUPipelineLayout>> visitor = {
							{visitBase},
							{asset,uniqueCopyGroupID},
							patch
						};
						if (!visitor())
							continue;
						auto layout = device->createPipelineLayout(pcRanges,std::move(visitor.dsLayouts[0]),std::move(visitor.dsLayouts[1]),std::move(visitor.dsLayouts[2]),std::move(visitor.dsLayouts[3]));
						conversionRequests.assign(entry.first,entry.second.firstCopyIx,i,std::move(layout));
					}
				}
			}
			if constexpr (std::is_same_v<AssetType,ICPUPipelineCache>)
			{
				for (auto& entry : conversionRequests.contentHashToCanonical)
				{
					const ICPUPipelineCache* asset = entry.second.canonicalAsset;
					// there is no patching possible for this asset
					for (auto i=0ull; i<entry.second.copyCount; i++)
					{
						// since we don't have dependants we don't care about our group ID
						// we create threadsafe pipeline caches, because we have no idea how they may be used
						conversionRequests.template assign<true>(entry.first,entry.second.firstCopyIx,i,device->createPipelineCache(asset,false));
					}
				}
			}
			if constexpr (std::is_same_v<AssetType,ICPUComputePipeline>)
			{
				for (auto& entry : conversionRequests.contentHashToCanonical)
				{
					const ICPUComputePipeline* asset = entry.second.canonicalAsset;
					// there is no patching possible for this asset
					for (auto i=0ull; i<entry.second.copyCount; i++)
					{
						const auto outIx = i+entry.second.firstCopyIx;
						const auto uniqueCopyGroupID = conversionRequests.gpuObjUniqueCopyGroupIDs[outIx];
						AssetVisitor<GetDependantVisit<ICPUComputePipeline>> visitor = {
							{visitBase},
							{asset,uniqueCopyGroupID},
							{}
						};
						if (!visitor())
							continue;
						// ILogicalDevice::createComputePipelines is rather aggressive on the spec constant validation, so we create one pipeline at a time
						core::smart_refctd_ptr<IGPUComputePipeline> ppln;
						{
							// no derivatives, special flags, etc.
							IGPUComputePipeline::SCreationParams params = {};
							params.layout = visitor.layout;
							// while there are patches possible for shaders, the only patch which can happen here is changing a stage from UNKNOWN to COMPUTE
							params.shader = visitor.getSpecInfo(IShader::E_SHADER_STAGE::ESS_COMPUTE);
							device->createComputePipelines(inputs.pipelineCache,{&params,1},&ppln);
						}
						conversionRequests.assign(entry.first,entry.second.firstCopyIx,i,std::move(ppln));
					}
				}
			}
			if constexpr (std::is_same_v<AssetType,ICPURenderpass>)
			{
				for (auto& entry : conversionRequests.contentHashToCanonical)
				{
					const ICPURenderpass* asset = entry.second.canonicalAsset;
					// there is no patching possible for this asset
					for (auto i=0ull; i<entry.second.copyCount; i++)
					{
						// since we don't have dependants we don't care about our group ID
						// we create threadsafe pipeline caches, because we have no idea how they may be used
						conversionRequests.template assign<true>(entry.first,entry.second.firstCopyIx,i,device->createRenderpass(asset->getCreationParameters()));
					}
				}
			}
			if constexpr (std::is_same_v<AssetType,ICPUGraphicsPipeline>)
			{
				core::vector<IGPUShader::SSpecInfo> tmpSpecInfo;
				tmpSpecInfo.reserve(5);
				for (auto& entry : conversionRequests.contentHashToCanonical)
				{
					const ICPUGraphicsPipeline* asset = entry.second.canonicalAsset;
					// there is no patching possible for this asset
					for (auto i=0ull; i<entry.second.copyCount; i++)
					{
						const auto outIx = i+entry.second.firstCopyIx;
						const auto uniqueCopyGroupID = conversionRequests.gpuObjUniqueCopyGroupIDs[outIx];
						AssetVisitor<GetDependantVisit<ICPUGraphicsPipeline>> visitor = {
							{visitBase},
							{asset,uniqueCopyGroupID},
							{}
						};
						if (!visitor())
							continue;
						// ILogicalDevice::createComputePipelines is rather aggressive on the spec constant validation, so we create one pipeline at a time
						core::smart_refctd_ptr<IGPUGraphicsPipeline> ppln;
						{
							// no derivatives, special flags, etc.
							IGPUGraphicsPipeline::SCreationParams params = {};
							bool depNotFound = false;
							{
								params.layout = visitor.layout;
								params.renderpass = visitor.renderpass;
								// while there are patches possible for shaders, the only patch which can happen here is changing a stage from UNKNOWN to match the slot here
								tmpSpecInfo.clear();
								using stage_t = ICPUShader::E_SHADER_STAGE;
								for (stage_t stage : {stage_t::ESS_VERTEX,stage_t::ESS_TESSELLATION_CONTROL,stage_t::ESS_TESSELLATION_EVALUATION,stage_t::ESS_GEOMETRY,stage_t::ESS_FRAGMENT})
								{
									auto& info = visitor.getSpecInfo(stage);
									if (info.shader)
										tmpSpecInfo.push_back(std::move(info));
								}
								params.shaders = tmpSpecInfo;
							}
							params.cached = asset->getCachedCreationParams();
							device->createGraphicsPipelines(inputs.pipelineCache,{&params,1},&ppln);
							conversionRequests.assign(entry.first,entry.second.firstCopyIx,i,std::move(ppln));
						}
					}
				}
			}
			if constexpr (std::is_same_v<AssetType,ICPUDescriptorSet>)
			{
				// Why we're not grouping multiple descriptor sets into few pools and doing 1 pool per descriptor set.
				// Descriptor Pools have large up-front slots reserved for all descriptor types, if we were to merge 
				// multiple descriptor sets to be allocated from one pool, dropping any set wouldn't result in the
				// reclamation of the memory used, it would at most (with the FREE pool create flag) return to pool. 
				for (auto& entry : conversionRequests.contentHashToCanonical)
				{
					const ICPUDescriptorSet* asset = entry.second.canonicalAsset;
					for (auto i=0ull; i<entry.second.copyCount; i++)
					{
						const auto outIx = i+entry.second.firstCopyIx;
						const auto uniqueCopyGroupID = conversionRequests.gpuObjUniqueCopyGroupIDs[outIx];
						AssetVisitor<GetDependantVisit<ICPUDescriptorSet>> visitor = {
							{visitBase},
							{asset,uniqueCopyGroupID},
							{}
						};
						if (!visitor())
							continue;
						const auto* layout = visitor.layout.get();
						const bool hasUpdateAfterBind = layout->needUpdateAfterBindPool();
						using pool_flags_t = IDescriptorPool::E_CREATE_FLAGS;
						auto pool = device->createDescriptorPoolForDSLayouts(
							hasUpdateAfterBind ? pool_flags_t::ECF_UPDATE_AFTER_BIND_BIT:pool_flags_t::ECF_NONE,{&layout,1}
						);
						core::smart_refctd_ptr<IGPUDescriptorSet> ds;
						if (pool)
						{
							ds = pool->createDescriptorSet(std::move(visitor.layout));
							if (ds && visitor.finalizeWrites(ds.get()) && !device->updateDescriptorSets(visitor.writes,{}))
							{
								inputs.logger.log("Failed to write Descriptors into Descriptor Set's bindings!",system::ILogger::ELL_ERROR);
								// fail
								ds = nullptr;
							}
							else
							for (const auto storageIx : visitor.potentialTLASRewrites)
								retval.m_potentialTLASRewrites.insert({ds.get(),storageIx});
						}
						else
							inputs.logger.log("Failed to create Descriptor Pool suited for Layout %s",system::ILogger::ELL_ERROR,layout->getObjectDebugName());
						conversionRequests.assign(entry.first,entry.second.firstCopyIx,i,std::move(ds));
					}
				}
			}

			// clear what we don't need
			if constexpr (!std::is_base_of_v<IAccelerationStructure,AssetType>)
				conversionRequests.gpuObjUniqueCopyGroupIDs.clear();
			// This gets deferred till AFTER the Buffer Memory Allocations and Binding
			if constexpr (!std::is_base_of_v<IAccelerationStructure,AssetType> && !std::is_base_of_v<IDeviceMemoryBacked,typename asset_traits<AssetType>::video_t>)
			{
				conversionRequests.propagateToCaches(std::get<dfs_cache<AssetType>>(dfsCaches),std::get<SReserveResult::staging_cache_t<AssetType>>(retval.m_stagingCaches));
				return {};
			}
			return conversionRequests;
		};
		// scope so the conversion requests go our of scope early
		{
			// The order of these calls is super important to go BOTTOM UP in terms of hashing and conversion dependants.
			// Both so we can hash in O(Depth) and not O(Depth^2) but also so we have all the possible dependants ready.
			// If two Asset chains are independent then we order them from most catastrophic failure to least.
			auto bufferConversions = dedupCreateProp.template operator()<ICPUBuffer>();
			auto blasConversions = dedupCreateProp.template operator()<ICPUBottomLevelAccelerationStructure>();
			auto tlasConversions = dedupCreateProp.template operator()<ICPUTopLevelAccelerationStructure>();
			auto imageConversions = dedupCreateProp.template operator()<ICPUImage>();
			// now allocate the memory for buffers and images
			deferredAllocator.finalize();

			// enqueue successfully created buffers for conversion
			for (auto& entry : bufferConversions.contentHashToCanonical)
			for (auto i=0ull; i<entry.second.copyCount; i++)
			if (auto& gpuBuff=bufferConversions.gpuObjects[i+entry.second.firstCopyIx].value; gpuBuff)
			{
				auto [where,inserted] = retval.m_bufferConversions.insert({gpuBuff.get(),core::smart_refctd_ptr<const ICPUBuffer>(entry.second.canonicalAsset)});
				assert(inserted);
			}
			bufferConversions.propagateToCaches(std::get<dfs_cache<ICPUBuffer>>(dfsCaches),std::get<SReserveResult::staging_cache_t<ICPUBuffer>>(retval.m_stagingCaches));
			// Deal with Deferred Creation of Acceleration structures
			{
				auto createAccelerationStructures = [&]<typename AccelerationStructure>(conversions_t<AccelerationStructure>& requests)->void
				{
					constexpr bool IsTLAS = std::is_same_v<AccelerationStructure,ICPUTopLevelAccelerationStructure>;
					//
					std::conditional_t<IsTLAS,SReserveResult::SConvReqTLASMap,SReserveResult::SConvReqBLASMap>* pConversions;
					if constexpr (IsTLAS)
						pConversions = retval.m_tlasConversions;
					else
						pConversions = retval.m_blasConversions;
					// we enqueue the conversions AFTER making sure that the BLAS / TLAS can actually be created
					for (auto& entry : requests.contentHashToCanonical)
					for (auto i=0ull; i<entry.second.copyCount; i++)
					{
						const auto reqIx = entry.second.firstCopyIx+i;
						if (const auto& deferredParams=accelerationStructureParams[IsTLAS][reqIx]; deferredParams.storage)
						{
							const auto* canonical = entry.second.canonicalAsset;
							const auto& dfsNode = std::get<dfs_cache<AccelerationStructure>>(dfsCaches).nodes[entry.second.patchIndex.value];
							const auto& patch = dfsNode.patch;
							// create the AS
							const auto bufSz = deferredParams.storage.get()->getSize();
							IGPUAccelerationStructure::SCreationParams baseParams;
							{
								using create_f = IGPUAccelerationStructure::SCreationParams::FLAGS;
								baseParams = {
									.bufferRange = {.offset=0,.size=bufSz,.buffer=deferredParams.storage.value},
									.flags = patch.isMotion ? create_f::MOTION_BIT:create_f::NONE
								};
							}
							smart_refctd_ptr<typename asset_traits<AccelerationStructure>::video_t> as;
							CAssetConverter::SReserveResult::SConvReqTLAS::cpu_to_gpu_blas_map_t blasInstanceMap;
							if constexpr (IsTLAS)
							{
								// check if the BLASes we want to use for the instances were successfully allocated and created
								AssetVisitor<GetDependantVisit<ICPUTopLevelAccelerationStructure>> visitor = {
									{inputs,dfsCaches,&blasInstanceMap},
									{canonical,requests.gpuObjUniqueCopyGroupIDs[reqIx]},
									patch
								};
								if (!visitor())
								{
									const auto hashAsU64 = reinterpret_cast<const uint64_t*>(entry.first.data);
									inputs.logger.log(
										"Failed to find all GPU Bottom Level Acceleration Structures needed to build TLAS %8llx%8llx%8llx%8llx",
										system::ILogger::ELL_ERROR,hashAsU64[0],hashAsU64[1],hashAsU64[2],hashAsU64[3]
									);
									continue;
								}
								as = device->createTopLevelAccelerationStructure({std::move(baseParams),patch.maxInstances});
							}
							else
								as = device->createBottomLevelAccelerationStructure(std::move(baseParams));
							if (!as)
							{
								inputs.logger.log("Failed to Create Acceleration Structure.",system::ILogger::ELL_ERROR);
								continue;
							}
							// file the request for conversion
							auto& request = pConversions[patch.hostBuild][as.get()];
							request.canonical = smart_refctd_ptr<const AccelerationStructure>(canonical);
							request.scratchSize = deferredParams.scratchSize;
							request.compact = patch.compactAfterBuild;
							request.buildFlags = static_cast<uint16_t>(patch.getBuildFlags(canonical).value);
							request.buildSize = deferredParams.buildSize;
							if constexpr (IsTLAS)
								request.instanceMap = std::move(blasInstanceMap);
							requests.assign(entry.first,entry.second.firstCopyIx,i,std::move(as));
						}
					}
					requests.gpuObjUniqueCopyGroupIDs.clear();
				};
				createAccelerationStructures.template operator()<ICPUBottomLevelAccelerationStructure>(blasConversions);
				blasConversions.propagateToCaches(std::get<dfs_cache<ICPUBottomLevelAccelerationStructure>>(dfsCaches),std::get<SReserveResult::staging_cache_t<ICPUBottomLevelAccelerationStructure>>(retval.m_stagingCaches));
				createAccelerationStructures.template operator()<ICPUTopLevelAccelerationStructure>(tlasConversions);
				tlasConversions.propagateToCaches(std::get<dfs_cache<ICPUTopLevelAccelerationStructure>>(dfsCaches),std::get<SReserveResult::staging_cache_t<ICPUTopLevelAccelerationStructure>>(retval.m_stagingCaches));
			}
			// enqueue successfully created images with data to upload for conversion
			auto& dfsCacheImages = std::get<dfs_cache<ICPUImage>>(dfsCaches);
			for (auto& entry : imageConversions.contentHashToCanonical)
			for (auto i=0ull; i<entry.second.copyCount; i++)
			{
				const auto* cpuImg = entry.second.canonicalAsset;
				if (auto& gpuImg=imageConversions.gpuObjects[i+entry.second.firstCopyIx].value; gpuImg && !cpuImg->getRegions().empty())
				{
					const bool recomputeMips = dfsCacheImages.nodes[entry.second.patchIndex.value].patch.recomputeMips;
					auto [where,inserted] = retval.m_imageConversions.insert({gpuImg.get(),SReserveResult::SConvReqImage{core::smart_refctd_ptr<const ICPUImage>(cpuImg),recomputeMips}});
					assert(inserted);
				}
			}
			imageConversions.propagateToCaches(dfsCacheImages,std::get<SReserveResult::staging_cache_t<ICPUImage>>(retval.m_stagingCaches));
		}
		dedupCreateProp.template operator()<ICPUBufferView>();
		dedupCreateProp.template operator()<ICPUImageView>();
		dedupCreateProp.template operator()<ICPUShader>();
		dedupCreateProp.template operator()<ICPUSampler>();
		dedupCreateProp.template operator()<ICPUDescriptorSetLayout>();
		dedupCreateProp.template operator()<ICPUPipelineLayout>();
		dedupCreateProp.template operator()<ICPUPipelineCache>();
		dedupCreateProp.template operator()<ICPUComputePipeline>();
		dedupCreateProp.template operator()<ICPURenderpass>();
		dedupCreateProp.template operator()<ICPUGraphicsPipeline>();
		dedupCreateProp.template operator()<ICPUDescriptorSet>();
//		dedupCreateProp.template operator()<ICPUFramebuffer>();
	}

	// write out results
	auto finalize = [&]<typename AssetType>(const std::span<const AssetType* const> assets)->void
	{
		const auto count = assets.size();
		//
		const auto& metadata = inputsMetadata[index_of_v<AssetType,supported_asset_types>];
		const auto& dfsCache = std::get<dfs_cache<AssetType>>(dfsCaches);
		const auto& stagingCache = std::get<SReserveResult::staging_cache_t<AssetType>>(retval.m_stagingCaches);
		auto& results = std::get<SReserveResult::vector_t<AssetType>>(retval.m_gpuObjects);
		for (size_t i=0; i<count; i++)
		if (auto asset=assets[i]; asset)
		{
			const auto uniqueCopyGroupID = metadata[i].uniqueCopyGroupID;
			// simple and easy to find all the associated items
			if (!metadata[i].patchIndex)
			{
				inputs.logger.log("No valid patch could be created for Root Asset %p in group %d",system::ILogger::ELL_ERROR,asset,uniqueCopyGroupID);
				continue;
			}
			const auto& found = dfsCache.nodes[metadata[i].patchIndex.value];
			// write it out to the results
			if (const auto& gpuObj=found.gpuObj; gpuObj)
			{
				results[i] = gpuObj;
#ifdef _NBL_DEBUG
				// if something with this content hash is in the stagingCache, then it must match the `found->gpuObj`
				if (auto finalCacheIt=stagingCache.find(gpuObj.get()); finalCacheIt!=stagingCache.end())
				{
					const bool matches = finalCacheIt->second.cacheKey==typename CCache<AssetType>::key_t(found.contentHash,uniqueCopyGroupID);
					assert(matches);
				}
#endif
			}
			else
				inputs.logger.log("No GPU Object could be found or created for Root Asset %p in group %d",system::ILogger::ELL_ERROR,asset,uniqueCopyGroupID);
		}
	};
	core::for_each_in_tuple(inputs.assets,finalize);

	// A failed conversion can cause dangling GPU object pointers, and needless work for objects which will die soon after, so prune with a Top-Down pass anything thats not reachable from a root
	{
		// we use a genious trick, if someone else is using the GPU object, the refcount must obviously be greater than 1
		auto pruneStaging = [&]<Asset AssetType>()->void
		{
			auto& stagingCache = std::get<SReserveResult::staging_cache_t<AssetType>>(retval.m_stagingCaches);
			phmap::erase_if(stagingCache,[&retval](const auto& entry)->bool
				{
					if (entry.first->getReferenceCount()==1)
					{
						// I know what I'm doing, the hashmap is being annoying not letting you look up with const pointer key a non const pointer hashmap
						auto* gpuObj = const_cast<asset_traits<AssetType>::video_t*>(entry.first);
						if constexpr (std::is_same_v<AssetType,ICPUBuffer>)
							retval.m_bufferConversions.erase(gpuObj);
						if constexpr (std::is_same_v<AssetType,ICPUBottomLevelAccelerationStructure>)
						for (auto i=0; i<2; i++)
							retval.m_blasConversions[i].erase(gpuObj);
						if constexpr (std::is_same_v<AssetType,ICPUTopLevelAccelerationStructure>)
						for (auto i=0; i<2; i++)
							retval.m_tlasConversions[i].erase(gpuObj);
						if constexpr (std::is_same_v<AssetType,ICPUImage>)
							retval.m_imageConversions.erase(gpuObj);
						// TODO: erase from `retval.m_gpuObjects` as well
						return true;
					}
					// still referenced, keep it around
					return false;
				}
			);
		};
		// The order these are called is paramount, the Higher Level User needs to die to let go of dependants and make our Garbage Collection work
//		pruneStaging.template operator()<ICPUFramebuffer>();
		pruneStaging.template operator()<ICPUDescriptorSet>();
		pruneStaging.template operator()<ICPUGraphicsPipeline>();
		pruneStaging.template operator()<ICPURenderpass>();
		pruneStaging.template operator()<ICPUComputePipeline>();
		pruneStaging.template operator()<ICPUPipelineCache>();
		pruneStaging.template operator()<ICPUPipelineLayout>();
		pruneStaging.template operator()<ICPUDescriptorSetLayout>();
		pruneStaging.template operator()<ICPUSampler>();
		pruneStaging.template operator()<ICPUShader>();
		pruneStaging.template operator()<ICPUImageView>();
		pruneStaging.template operator()<ICPUBufferView>();
		pruneStaging.template operator()<ICPUImage>();
		pruneStaging.template operator()<ICPUTopLevelAccelerationStructure>();
		pruneStaging.template operator()<ICPUBottomLevelAccelerationStructure>();
		pruneStaging.template operator()<ICPUBuffer>();
	}

	// only now get the queue flags
	{
		using q_fam_f = IQueue::FAMILY_FLAGS;
		// acceleration structures, get scratch size
		auto computeAccelerationStructureScratchSizes = [device,&retval]<typename AccelerationStructure>()->void
		{
			constexpr bool IsTLAS = std::is_same_v<AccelerationStructure,ICPUTopLevelAccelerationStructure>;
			const auto& limits = device->getPhysicalDevice()->getLimits();
			const auto minScratchAlignment = limits.minAccelerationStructureScratchOffsetAlignment;
			// index 0 is device build, 1 is host build
			size_t scratchSizeFullParallelBuild[2] = {0,0};
			//
			const std::conditional_t<IsTLAS,SReserveResult::SConvReqTLASMap,SReserveResult::SConvReqBLASMap>* pConversions;
			if constexpr (IsTLAS)
				pConversions = retval.m_tlasConversions;
			else
				pConversions = retval.m_blasConversions;
			// we collect the stats AFTER making sure only needed TLAS and BLAS will be built
			for (auto i=0; i<2; i++)
			for (auto req : pConversions[i])
			{
				const auto buildSize = req.second.buildSize;
				// sizes for building 1-by-1 vs parallel, note that BLAS and TLAS can't be built concurrently
				retval.m_minASBuildScratchSize[i] = core::max(retval.m_minASBuildScratchSize[i],buildSize);
				scratchSizeFullParallelBuild[i] = core::alignUp(scratchSizeFullParallelBuild[i],minScratchAlignment)+buildSize;
				// note that in order to compact an AS you need to allocate a buffer range whose size is known only after the build
				if (req.second.compact)
				{
					const auto asSize = req.first->getCreationParams().bufferRange.size;
					assert(core::is_aligned_to(asSize,256));
					retval.m_compactedASMaxMemory += asSize;
				}
			}
			// TLAS and BLAS can't build concurrently
			retval.m_maxASBuildScratchSize[0] = core::max(scratchSizeFullParallelBuild[0],retval.m_maxASBuildScratchSize[0]);
			retval.m_maxASBuildScratchSize[1] = core::max(scratchSizeFullParallelBuild[1],retval.m_maxASBuildScratchSize[1]);
		};
		computeAccelerationStructureScratchSizes.template operator()<ICPUBottomLevelAccelerationStructure>();
		computeAccelerationStructureScratchSizes.template operator()<ICPUTopLevelAccelerationStructure>();
		if (retval.willDeviceASBuild() || retval.willCompactAS())
			retval.m_queueFlags |= IQueue::FAMILY_FLAGS::COMPUTE_BIT;
		// images are trickier, we can't finish iterating until all possible flags are there
		for (auto it=retval.m_imageConversions.begin(); !retval.m_queueFlags.hasFlags(q_fam_f::TRANSFER_BIT|q_fam_f::COMPUTE_BIT|q_fam_f::GRAPHICS_BIT) && it!=retval.m_imageConversions.end(); it++)
		{
			const auto boundMemory = it->first->getBoundMemory();
			assert(boundMemory.isValid());
			// Note: with `host_image_copy` this will get conditional
			{
				retval.m_queueFlags |= IQueue::FAMILY_FLAGS::TRANSFER_BIT;
				// Best effort guess, without actually looking at all regions
				const auto& params = it->first->getCreationParameters();
				// https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkCmdCopyBufferToImage.html#VUID-vkCmdCopyBufferToImage-commandBuffer-07739
				if (isDepthOrStencilFormat(params.format) && (params.depthUsage|params.stencilUsage).hasFlags(IGPUImage::E_USAGE_FLAGS::EUF_TRANSFER_DST_BIT))
					retval.m_queueFlags |= IQueue::FAMILY_FLAGS::GRAPHICS_BIT;
				if (it->second.recomputeMips)
					retval.m_queueFlags |= IQueue::FAMILY_FLAGS::COMPUTE_BIT;
			}
		}
		// buffer conversions
		for (auto it=retval.m_bufferConversions.begin(); !retval.m_queueFlags.hasFlags(q_fam_f::TRANSFER_BIT) && it!=retval.m_bufferConversions.end(); it++)
		{
			const auto boundMemory = it->first->getBoundMemory();
			assert(boundMemory.isValid());
			if (!canHostWriteToMemoryRange(boundMemory,it->first->getSize()))
				retval.m_queueFlags |= IQueue::FAMILY_FLAGS::TRANSFER_BIT;
		}
	}

	retval.m_converter = core::smart_refctd_ptr<CAssetConverter>(this);
	retval.m_logger = system::logger_opt_smart_ptr(core::smart_refctd_ptr<system::ILogger>(inputs.logger.get()));
	return retval;
}

//
ISemaphore::future_t<IQueue::RESULT> CAssetConverter::convert_impl(SReserveResult& reservations, SConvertParams& params)
{
	ISemaphore::future_t<IQueue::RESULT> retval = IQueue::RESULT::OTHER_ERROR;
	system::logger_opt_ptr logger = reservations.m_logger.get().get();
	if (!reservations.m_converter)
	{
		logger.log("Cannot call convert on an unsuccessful reserve result! Or are you attempting to do a double run of `convert` ?",system::ILogger::ELL_ERROR);
		return retval;
	}
	assert(reservations.m_converter.get()==this);
	auto device = m_params.device;

	auto hostBufferXferIt = reservations.m_bufferConversions.begin();
	core::vector<ILogicalDevice::MappedMemoryRange> memoryHostFlushRanges;
	memoryHostFlushRanges.reserve(reservations.m_bufferConversions.size());
	auto hostUploadBuffers = [&](auto&& pred)->void
	{
		for (; hostBufferXferIt!=reservations.m_bufferConversions.end() && pred(); hostBufferXferIt++)
		{
			IGPUBuffer* buff = hostBufferXferIt->first;
			const size_t size = buff->getSize();
			const auto boundMemory = buff->getBoundMemory();
			if (!canHostWriteToMemoryRange(boundMemory,size))
				continue;
			auto* const memory = boundMemory.memory;
			const IDeviceMemoryAllocation::MemoryRange range = {boundMemory.offset,size};
			memcpy(reinterpret_cast<uint8_t*>(memory->getMappedPointer())+range.offset,hostBufferXferIt->second->getPointer(),size);
			// let go of canonical asset (may free RAM)
			hostBufferXferIt->second = nullptr;
			if (memory->haveToMakeVisible())
				memoryHostFlushRanges.emplace_back(memory,range.offset,range.length,ILogicalDevice::MappedMemoryRange::align_non_coherent_tag);
		}
		if (!memoryHostFlushRanges.empty())
		{
			device->flushMappedMemoryRanges(memoryHostFlushRanges);
			memoryHostFlushRanges.clear();
		}
	};

	// wipe gpu item in staging cache (this may drop it as well if it was made for only a root asset == no users)
	core::unordered_map<const IBackendObject*, uint32_t> outputReverseMap;
	core::for_each_in_tuple(reservations.m_gpuObjects,[&outputReverseMap](const auto& gpuObjects)->void
		{
			uint32_t i = 0;
			for (const auto& gpuObj : gpuObjects)
				outputReverseMap[gpuObj.value.get()] = i++;
		}
	);
	auto markFailure = [&reservations,&outputReverseMap,logger]<Asset AssetType>(const char* message, smart_refctd_ptr<const AssetType>* canonical, typename SReserveResult::staging_cache_t<AssetType>::mapped_type* cacheNode)->void
	{
		// wipe the smart pointer to the canonical, make sure we release that memory ASAP if no other user is around
		*canonical = nullptr;
		// also drop the smart pointer from the output array so failures release memory quickly
		const auto foundIx = outputReverseMap.find(cacheNode->gpuRef.get());
		if (foundIx!=outputReverseMap.end())
		{
			auto& resultOutput = std::get<SReserveResult::vector_t<AssetType>>(reservations.m_gpuObjects);
			resultOutput[foundIx->second].value = nullptr;
			outputReverseMap.erase(foundIx);
		}
		logger.log("%s failed for \"%s\"",system::ILogger::ELL_ERROR,message,cacheNode->gpuRef->getObjectDebugName());
		// drop smart pointer 
		cacheNode->gpuRef = nullptr;
	};

	// want to check if deps successfully exist
	struct SMissingDependent
	{
		// This only checks if whether we had to convert and failed, but the dependent might be in readCache of one or more converters, so if in doubt assume its okay
		inline operator bool() const {return wasInStaging && gotWiped;}

		bool wasInStaging;
		bool gotWiped;
	};
	auto missingDependent = [&reservations]<Asset AssetType>(const typename asset_traits<AssetType>::video_t* dep)->SMissingDependent
	{
		const auto& stagingCache = std::get<SReserveResult::staging_cache_t<AssetType>>(reservations.m_stagingCaches);
		const auto found = stagingCache.find(dep);
		SMissingDependent retval = {.wasInStaging=found!=stagingCache.end()};
		retval.gotWiped = retval.wasInStaging && !found->second.gpuRef;
		return retval;
	};

	// Descriptor Sets need their TLAS descriptors substituted if they've been compacted
	core::unordered_map<const IGPUTopLevelAccelerationStructure*,smart_refctd_ptr<IGPUTopLevelAccelerationStructure>> compactedTLASMap;
	// Anything to do?
	if (reservations.m_queueFlags.value!=IQueue::FAMILY_FLAGS::NONE)
	{
		// whether we actually get around to doing that depends on validity and success of transfers
		const bool shouldDoSomeCompute = reservations.m_queueFlags.hasFlags(IQueue::FAMILY_FLAGS::COMPUTE_BIT);
		auto invalidIntended = [device,logger](const IQueue::FAMILY_FLAGS flag, const SIntendedSubmitInfo* intended)->bool
		{
			if (!intended || !intended->valid())
			{
				logger.log("Invalid `SIntendedSubmitInfo` for queue capability %d!",system::ILogger::ELL_ERROR,flag);
				return true;
			}
			const auto* queue = intended->queue;
			if (queue->getOriginDevice()!=device)
			{
				logger.log("Provided Queue's device %p doesn't match CAssetConverter's device %p!",system::ILogger::ELL_ERROR,queue->getOriginDevice(),device);
				return true;
			}
			const auto& qFamProps = device->getPhysicalDevice()->getQueueFamilyProperties();
			if (!qFamProps[queue->getFamilyIndex()].queueFlags.hasFlags(flag))
			{
				logger.log("Provided Queue %p in Family %d does not have the required capabilities %d!",system::ILogger::ELL_ERROR,queue,queue->getFamilyIndex(),flag);
				return true;
			}
			return false;
		};
		// If the compute queue will be used, the compute Intended Submit Info must be valid
		if (shouldDoSomeCompute && invalidIntended(IQueue::FAMILY_FLAGS::COMPUTE_BIT,params.compute))
			return retval;
		// the flag check stops us derefercing an invalid pointer
		const auto computeFamily = shouldDoSomeCompute ? params.compute->queue->getFamilyIndex():IQueue::FamilyIgnored;

		// unfortunately can't count on large ReBAR heaps so we can't require the `scratchBuffer` to be mapped and writable
		uint8_t* deviceASBuildScratchPtr = nullptr;
		// check things necessary for building Acceleration Structures
		if (reservations.willDeviceASBuild())
		{
			if (!params.scratchForDeviceASBuild)
			{
				logger.log("An Acceleration Structure will be built on Device but no scratch allocator provided!",system::ILogger::ELL_ERROR);
				return retval;
			}
			using buffer_usage_f = IGPUBuffer::E_USAGE_FLAGS;
			constexpr buffer_usage_f asBuildInputFlags = buffer_usage_f::EUF_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT|buffer_usage_f::EUF_SHADER_DEVICE_ADDRESS_BIT;
			constexpr buffer_usage_f asBuildScratchFlags = buffer_usage_f::EUF_STORAGE_BUFFER_BIT|buffer_usage_f::EUF_SHADER_DEVICE_ADDRESS_BIT;
			auto* scratchBuffer = params.scratchForDeviceASBuild->getBuffer();
			const auto& scratchParams = scratchBuffer->getCachedCreationParams();
			if (!scratchParams.canBeUsedByQueueFamily(computeFamily))
			{
				logger.log("Acceleration Structure Scratch Device Memory Allocator has concurrent sharing but not usable by Compute Family %d!",system::ILogger::ELL_ERROR,computeFamily);
				return retval;
			}
			// we use the scratch allocator both for scratch and uploaded geometry data
			if (!scratchBuffer->getCreationParams().usage.hasFlags(asBuildScratchFlags|asBuildInputFlags))
			{
				logger.log("An Acceleration Structure will be built on Device but scratch buffer doesn't have required usage flags!",system::ILogger::ELL_ERROR);
				return retval;
			}
			const auto& addrAlloc = params.scratchForDeviceASBuild->getAddressAllocator();
			// could have used an address allocator trait to work this out, same verbosity
			if (addrAlloc.get_allocated_size()+addrAlloc.get_free_size()<reservations.m_minASBuildScratchSize[0])
			{
				logger.log("Acceleration Structure Scratch Device Memory Allocator not large enough!",system::ILogger::ELL_ERROR);
				return retval;
			}
			// this alignment is probably bigger than required by any Build Input
			const auto minScratchAlignment = device->getPhysicalDevice()->getLimits().minAccelerationStructureScratchOffsetAlignment;
			if (addrAlloc.max_alignment()<minScratchAlignment)
			{
				logger.log("Accceleration Structure Scratch Device Memory Allocator cannot allocate with Physical Device's minimum required AS-build scratch alignment %u",system::ILogger::ELL_ERROR,minScratchAlignment);
				return retval;
			}
		// TODO: check scratchForDeviceASBuildMinAllocSize
			// returns non-null pointer if the buffer is writeable directly byt the host
			deviceASBuildScratchPtr = reinterpret_cast<uint8_t*>(scratchBuffer->getBoundMemory().memory->getMappedPointer());
			// Need to use Transfer Queue and copy via staging buffer
			if (!deviceASBuildScratchPtr)
			{
				if (!params.transfer || !params.transfer->queue)
				{
					logger.log("Transfers required for Acceleration Structure Builds, but no valid queue given!", system::ILogger::ELL_ERROR);
					return retval;
				}
				const auto transferFamily = params.transfer->queue->getFamilyIndex();
				// But don't want to have to do QFOTs between Transfer and Queue Families then
				if (transferFamily!=computeFamily)
				if (!scratchParams.isConcurrentSharing() || !scratchParams.canBeUsedByQueueFamily(transferFamily))
				{
					logger.log("Acceleration Structure Scratch Device Memory Allocator not mapped and not concurrently share-able by Transfer Family %d!",system::ILogger::ELL_ERROR,transferFamily);
					return retval;
				}
				if (!scratchBuffer->getCreationParams().usage.hasFlags(buffer_usage_f::EUF_TRANSFER_DST_BIT))
				{
					logger.log("Acceleration Structure Scratch Device Memory Allocator not mapped and doesn't the transfer destination usage flag!",system::ILogger::ELL_ERROR);
					return retval;
				}
				// Right now we copy from staging to scratch, but in the future we may use the staging buffer directly to skip an extra copy on small enough geometries
				if (!params.utilities->getDefaultUpStreamingBuffer()->getBuffer()->getCreationParams().usage.hasFlags(asBuildInputFlags|buffer_usage_f::EUF_TRANSFER_SRC_BIT))
				{
					logger.log("An Acceleration Structure will be built on Device but Default UpStreaming Buffer from IUtilities doesn't have required usage flags!", system::ILogger::ELL_ERROR);
					return retval;
				}
			}
		}
		// the elusive and exotic host builds
		if (reservations.willHostASBuild())
		{
			if (!params.scratchForHostASBuild)
			{
				logger.log("An Acceleration Structure will be built on the Host but no Scratch Memory Allocator provided!", system::ILogger::ELL_ERROR);
				return retval;
			}
			// TODO: check everything else when we actually support host builds
		}
		// and compacting
		if (reservations.willCompactAS())
		{
			if (!params.compactedASAllocator)
				logger.log("Acceleration Structures will be compacted using the ILogicalDevice as the memory allocator!", system::ILogger::ELL_WARNING);
			// note that can't check the compacted AS allocator being large enough against `reservations.m_compactedASMaxMemory`
		}

		//
		const auto reqQueueFlags = reservations.getRequiredQueueFlags(deviceASBuildScratchPtr);
		bool shouldDoSomeTransfer = reqQueueFlags.hasFlags(IQueue::FAMILY_FLAGS::TRANSFER_BIT);
		{
			// If the transfer queue will be used, the transfer Intended Submit Info must be valid and utilities must be provided
			auto reqTransferQueueCaps = IQueue::FAMILY_FLAGS::TRANSFER_BIT;
			// Depth/Stencil transfers need Graphics Capabilities, so make sure the queue chosen for transfers also has them!
			if (reservations.m_queueFlags.hasFlags(IQueue::FAMILY_FLAGS::GRAPHICS_BIT))
				reqTransferQueueCaps |= IQueue::FAMILY_FLAGS::GRAPHICS_BIT;
			if (shouldDoSomeTransfer && invalidIntended(reqTransferQueueCaps,params.transfer))
				return retval;
		}
		const auto transferFamily = shouldDoSomeTransfer ? params.transfer->queue->getFamilyIndex():IQueue::FamilyIgnored;

		// The current begun Xfer and Compute commandbuffer changing because of submit of Xfer or Compute would be a royal mess to deal with
		if (shouldDoSomeTransfer && shouldDoSomeCompute)
		{
			core::unordered_set<const IGPUCommandBuffer*> uniqueCmdBufs;
			for (const auto& scratch : params.transfer->scratchCommandBuffers)
				uniqueCmdBufs.insert(scratch.cmdbuf);
			for (const auto& scratch : params.compute->scratchCommandBuffers)
				uniqueCmdBufs.insert(scratch.cmdbuf);
			if (uniqueCmdBufs.size()!=params.compute->scratchCommandBuffers.size()+params.transfer->scratchCommandBuffers.size())
			{
				logger.log("The Compute `SIntendedSubmit` Scratch Command Buffers cannot be idential to Transfer's!",system::ILogger::ELL_ERROR);
				return retval;
			}
		}
		const bool uniQueue = !shouldDoSomeTransfer || !shouldDoSomeCompute || params.transfer->queue->getNativeHandle()==params.compute->queue->getNativeHandle();

		//
		if (shouldDoSomeTransfer && (!params.utilities || params.utilities->getLogicalDevice()!=device))
		{
			logger.log("Transfer Capability required for this conversion and no compatible `utilities` provided!",system::ILogger::ELL_ERROR);
			return retval;
		}

		//
		core::bitflag<IQueue::FAMILY_FLAGS> submitsNeeded = IQueue::FAMILY_FLAGS::NONE;

		//
		constexpr uint32_t QueueFamilyInvalid = 0xffffffffu;
		auto checkOwnership = [&](auto* gpuObj, const uint32_t nextQueueFamily, const uint32_t currentQueueFamily)->auto
		{
			// didn't ask for a QFOT
			if (nextQueueFamily==IQueue::FamilyIgnored)
				return IQueue::FamilyIgnored;
			// we already own
			if (nextQueueFamily==currentQueueFamily)
				return IQueue::FamilyIgnored;
			const auto& params = gpuObj->getCachedCreationParams();
			// silently skip ownership transfer
			if (params.isConcurrentSharing())
			{
				if (params.canBeUsedByQueueFamily(currentQueueFamily))
				{
					logger.log("Previous Queue Family %d not in the concurrent sharing set of IDeviceMemoryBacked %s",system::ILogger::ELL_ERROR,gpuObj->getObjectDebugName());
					return QueueFamilyInvalid;
				}
				if (params.canBeUsedByQueueFamily(nextQueueFamily))
				{
					logger.log("Next Queue Family %d not in the concurrent sharing set of IDeviceMemoryBacked %s",system::ILogger::ELL_ERROR,gpuObj->getObjectDebugName());
					return QueueFamilyInvalid;
				}
				return IQueue::FamilyIgnored;
			}
			return nextQueueFamily;
		};
		using ownership_op_t = IGPUCommandBuffer::SOwnershipTransferBarrier::OWNERSHIP_OP;

		// unify logging
		auto pipelineBarrier = [logger]/*<typename... Args>*/(
			const IQueue::SSubmitInfo::SCommandBufferInfo* const cmdbufInfo,
			const IGPUCommandBuffer::SPipelineBarrierDependencyInfo & info,
			const char* failMessage/*, Args&&... args*/
		)->bool
		{
			if (!cmdbufInfo->cmdbuf->pipelineBarrier(E_DEPENDENCY_FLAGS::EDF_NONE,info))
			{
				logger.log(failMessage,system::ILogger::ELL_ERROR/*,std::forward<Args>(args)...*/);
				return false;
			}
			return true;
		};

		// some state so we don't need to look later
		auto xferCmdBuf = shouldDoSomeTransfer ? params.transfer->getCommandBufferForRecording():nullptr;

		//
		auto findInStaging = [&reservations]<Asset AssetType>(const typename asset_traits<AssetType>::video_t* gpuObj)->auto
		{
			auto& stagingCache = std::get<SReserveResult::staging_cache_t<AssetType>>(reservations.m_stagingCaches);
			const auto found = stagingCache.find(gpuObj);
			assert(found!=stagingCache.end());
			return found;
		};

		using buffer_mem_barrier_t = IGPUCommandBuffer::SBufferMemoryBarrier<IGPUCommandBuffer::SOwnershipTransferBarrier>;
		// upload Buffers
		auto& buffersToUpload = reservations.m_bufferConversions;
		{
			core::vector<buffer_mem_barrier_t> finalReleases;
			finalReleases.reserve(buffersToUpload.size());
			// do the uploads
			if (!buffersToUpload.empty() && xferCmdBuf)
			{
				xferCmdBuf->cmdbuf->beginDebugMarker("Asset Converter Upload Buffers START");
				xferCmdBuf->cmdbuf->endDebugMarker();
			}
			for (auto& item : buffersToUpload)
			{
				auto* buffer = item.first;
				const size_t size = buffer->getCreationParams().size;
				// host will upload
				if (canHostWriteToMemoryRange(buffer->getBoundMemory(),size))
					continue;
				auto pFound = &findInStaging.template operator()<ICPUBuffer>(buffer)->second;
				//
				const auto ownerQueueFamily = checkOwnership(buffer,params.getFinalOwnerQueueFamily(buffer,pFound->cacheKey.value),transferFamily);
				if (ownerQueueFamily==QueueFamilyInvalid)
				{
					markFailure("invalid Final Queue Family given by user callback",&item.second,pFound);
					continue;
				}
				// do the upload
				const SBufferRange<IGPUBuffer> range = {.offset=0,.size=size,.buffer=core::smart_refctd_ptr<IGPUBuffer>(buffer)};
				const bool success = params.utilities->updateBufferRangeViaStagingBuffer(*params.transfer,range,item.second->getPointer());
				// current recording buffer may have changed
				xferCmdBuf = params.transfer->getCommandBufferForRecording();
				if (!success)
				{
					markFailure("Data Upload",&item.second,pFound);
					continue;
				}
				// let go of canonical asset (may free RAM)
				item.second = nullptr;
				submitsNeeded |= IQueue::FAMILY_FLAGS::TRANSFER_BIT;
				// enqueue ownership release if necessary
				if (ownerQueueFamily!=IQueue::FamilyIgnored)
					finalReleases.push_back({
						.barrier = {
							.dep = {
								.srcStageMask = PIPELINE_STAGE_FLAGS::COPY_BIT,
								.srcAccessMask = ACCESS_FLAGS::TRANSFER_WRITE_BIT
								// leave rest empty, we can release whenever after the copies and before the semaphore signal
							},
							.ownershipOp = ownership_op_t::RELEASE,
							.otherQueueFamilyIndex = ownerQueueFamily
						},
						.range = range
					});
			}
			if (!buffersToUpload.empty() && xferCmdBuf)
			{
				xferCmdBuf->cmdbuf->beginDebugMarker("Asset Converter Upload Buffers END");
				xferCmdBuf->cmdbuf->endDebugMarker();
			}
			// release ownership
			if (!finalReleases.empty())
				pipelineBarrier(xferCmdBuf,{.memBarriers={},.bufBarriers=finalReleases},"Ownership Releases of Buffers Failed");
		}

		const auto* physDev = device->getPhysicalDevice();
		
		// whenever transfer needs to do a submit overflow because it ran out of memory for streaming, we can already submit the recorded compute shader dispatches
		auto computeCmdBuf = shouldDoSomeCompute ? params.compute->getCommandBufferForRecording():nullptr;
		auto drainCompute = [&params,shouldDoSomeTransfer,&computeCmdBuf](const std::span<const IQueue::SSubmitInfo::SSemaphoreInfo> extraSignal={})->auto
		{
			if (!computeCmdBuf || computeCmdBuf->cmdbuf->empty())
				return IQueue::RESULT::SUCCESS;
			// before we overflow submit we need to inject extra wait semaphores
			auto& waitSemaphoreSpan = params.compute->waitSemaphores;
			std::unique_ptr<IQueue::SSubmitInfo::SSemaphoreInfo[]> patchedWaits;
			// the transfer scratch semaphore value, is from the last submit, not the future value we're enqueing all the deferred memory releases with
			if (shouldDoSomeTransfer)
			{
				if (waitSemaphoreSpan.empty())
					waitSemaphoreSpan = {&params.transfer->scratchSemaphore,1};
				else
				{
					const auto origCount = waitSemaphoreSpan.size();
					patchedWaits.reset(new IQueue::SSubmitInfo::SSemaphoreInfo[origCount+1]);
					std::copy(waitSemaphoreSpan.begin(),waitSemaphoreSpan.end(),patchedWaits.get());
					patchedWaits[origCount] = params.transfer->scratchSemaphore;
					waitSemaphoreSpan = {patchedWaits.get(),origCount+1};
				}
			}
			// don't worry about resetting old `waitSemaphores` because they get cleared to an empty span after overflow submit
            IQueue::RESULT res = params.compute->submit(computeCmdBuf,extraSignal);
            if (res!=IQueue::RESULT::SUCCESS)
                return res;
			// set to empty so we don't grow over and over again
			waitSemaphoreSpan = {};
            if (!params.compute->beginNextCommandBuffer(computeCmdBuf))
                return IQueue::RESULT::OTHER_ERROR;
			return res;
		};

		// We want to be doing Host operations while stalled for GPU, compose our overflow callback on top of what's already there, only if we need to ofc 
		std::function<void(const ISemaphore::SWaitInfo&)> origXferStallCallback;
		if (shouldDoSomeTransfer)
		{
			origXferStallCallback = std::move(params.transfer->overflowCallback);
			params.transfer->overflowCallback = [device,&hostUploadBuffers,&origXferStallCallback,&drainCompute](const ISemaphore::SWaitInfo& tillScratchResettable)->void
			{
				drainCompute();
				if (origXferStallCallback)
					origXferStallCallback(tillScratchResettable);
				hostUploadBuffers([device,&tillScratchResettable]()->bool{return device->waitForSemaphores({&tillScratchResettable,1},false,0)==ISemaphore::WAIT_RESULT::TIMEOUT;});
			};
		}
		// when overflowing compute resources, we need to submit the Xfer before submitting Compute
		auto drainBoth = [&params,&xferCmdBuf,&drainCompute](const std::span<const IQueue::SSubmitInfo::SSemaphoreInfo> extraSignal={})->auto
		{
			if (xferCmdBuf && !xferCmdBuf->cmdbuf->empty())
				params.transfer->overflowSubmit(xferCmdBuf);
			return drainCompute();
		};

		auto& imagesToUpload = reservations.m_imageConversions;
		if (!imagesToUpload.empty())
		{
			//
			constexpr auto MaxMipLevelsPastBase = 16;
			constexpr auto SrcMipBinding = 1;
			constexpr auto DstMipBinding = 1;
			core::smart_refctd_ptr<SubAllocatedDescriptorSet> dsAlloc;
			if (shouldDoSomeCompute)
			{
				// TODO: add mip-map recomputation sampler to the patch params (wrap modes and border color), and hash & cache them during creation
				const auto repeatSampler = device->createSampler({
					// default everything
				});
				using binding_create_flags_t = IGPUDescriptorSetLayout::SBindingBase::E_CREATE_FLAGS;
				constexpr auto BindingFlags = SubAllocatedDescriptorSet::RequiredBindingFlags;
				// need at least as many elements in descriptor array as scratch buffers, and no more than total images
				const uint32_t imageCount = imagesToUpload.size();
				const uint32_t computeMultiBufferingCount = params.compute->scratchCommandBuffers.size();
				const IGPUDescriptorSetLayout::SBinding bindings[3] = {
					{.binding=0,.type=IDescriptor::E_TYPE::ET_SAMPLER,.createFlags=BindingFlags,.stageFlags=IGPUShader::E_SHADER_STAGE::ESS_COMPUTE,.count=1,.immutableSamplers=&repeatSampler},
					{
						.binding = SrcMipBinding,
						.type = IDescriptor::E_TYPE::ET_SAMPLED_IMAGE,
						.createFlags = BindingFlags,
						.stageFlags = IGPUShader::E_SHADER_STAGE::ESS_COMPUTE,
						.count = std::min(std::max(computeMultiBufferingCount,params.sampledImageBindingCount),imageCount)
					},
					{
						.binding = DstMipBinding,
						.type = IDescriptor::E_TYPE::ET_STORAGE_IMAGE,
						.createFlags = BindingFlags,
						.stageFlags = IGPUShader::E_SHADER_STAGE::ESS_COMPUTE,
						.count = std::min(std::max(MaxMipLevelsPastBase*computeMultiBufferingCount,params.storageImageBindingCount),MaxMipLevelsPastBase*imageCount)
					}
				};
				auto layout = device->createDescriptorSetLayout(bindings);
				auto pool = device->createDescriptorPoolForDSLayouts(IDescriptorPool::ECF_UPDATE_AFTER_BIND_BIT,{&layout.get(),1});
				dsAlloc = core::make_smart_refctd_ptr<SubAllocatedDescriptorSet>(pool->createDescriptorSet(std::move(layout)));
			}
			auto quickWriteDescriptor = [device,logger,&dsAlloc](const uint32_t binding, const uint32_t arrayElement, core::smart_refctd_ptr<IGPUImageView> view)->bool
			{
				if (arrayElement==SubAllocatedDescriptorSet::invalid_value)
				{
					logger.log("Failed to allocate from binding %d in the Suballocated Descriptor Sets!",system::ILogger::ELL_ERROR,binding);
					return false;
				}
				auto* ds = dsAlloc->getDescriptorSet();
				IGPUDescriptorSet::SDescriptorInfo info = {};
				info.desc = std::move(view);
				info.info.image.imageLayout = IGPUImage::LAYOUT::GENERAL;
				const IGPUDescriptorSet::SWriteDescriptorSet write = {
					.dstSet = ds,
					.binding = binding,
					.arrayElement = arrayElement,
					.count = 1,
					.info = &info
				};
				if (!device->updateDescriptorSets({&write,1},{}))
				{
					logger.log("Failed to write to binding %d element %d in the Suballocated Descriptor Set, despite allocation success!",system::ILogger::ELL_ERROR,binding,arrayElement);
					return false;
				}
				return true;
			};

			// because of the layout transitions (TODO: conditional when host_image_copy gets implemented)
			params.transfer->scratchSemaphore.stageMask |= PIPELINE_STAGE_FLAGS::ALL_COMMANDS_BITS;
// TODO:: Shall we rewrite? e.g. we upload everything first, extra submit for QFOT pipeline barrier & transition in overflow callback, then record compute commands, and submit them, plus their final QFOTs
			// Lets analyze sync cases:
			// - Single Queue = Semaphore Signal is sufficient, 
			// - Two distinct Queues = no barrier, semaphore signal-wait is sufficient
			// - Two distinct Queue Families Exclusive Sharing mode = QFOT necessary
			core::vector<IGPUCommandBuffer::SImageMemoryBarrier<IGPUCommandBuffer::SOwnershipTransferBarrier>> transferBarriers;
			core::vector<IGPUCommandBuffer::SImageMemoryBarrier<IGPUCommandBuffer::SOwnershipTransferBarrier>> computeBarriers;
			transferBarriers.reserve(MaxMipLevelsPastBase);
			computeBarriers.reserve(MaxMipLevelsPastBase);
			// finally go over the images
			xferCmdBuf->cmdbuf->beginDebugMarker("Asset Converter Upload Images START");
			xferCmdBuf->cmdbuf->endDebugMarker();
			for (auto& item : imagesToUpload)
			{
				// basiscs
				auto& cpuImg = item.second.canonical;
				auto* image = item.first;
				auto pFound = &findInStaging.template operator()<ICPUImage>(image)->second;
				// get params
				const auto& creationParams = image->getCreationParameters();
				const auto format = creationParams.format;
				using aspect_flags_t = IGPUImage::E_ASPECT_FLAGS;
				const core::bitflag<aspect_flags_t> aspects = isDepthOrStencilFormat(format) ? static_cast<aspect_flags_t>(
						(isDepthOnlyFormat(format) ? aspect_flags_t::EAF_NONE:aspect_flags_t::EAF_STENCIL_BIT)|(isStencilOnlyFormat(format) ? aspect_flags_t::EAF_NONE:aspect_flags_t::EAF_DEPTH_BIT)
					):aspect_flags_t::EAF_COLOR_BIT;
				// allocate the offset in the binding array and write the source image view into the descriptor set
				auto srcIx = SubAllocatedDescriptorSet::invalid_value;
				// clean up the allocation if we fail to make it to the end of loop for whatever reason
				// cannot do `multi_deallocate` with future semaphore value right away, because we don't know the last submit to use this descriptor, yet. 
				auto deallocSrc = core::makeRAIIExiter([SrcMipBinding,&dsAlloc,&srcIx]()->void{
					if (srcIx!=SubAllocatedDescriptorSet::invalid_value)
						dsAlloc->multi_deallocate(SrcMipBinding,1,&srcIx,{});
				});
				IGPUImageView::E_TYPE viewType = IGPUImageView::E_TYPE::ET_2D_ARRAY;
				// create Mipmapping source Image View, allocate its place in the descriptor set and write it
				if (item.second.recomputeMips)
				{
					switch (creationParams.type)
					{
						case IGPUImage::E_TYPE::ET_1D:
							viewType = IGPUImageView::E_TYPE::ET_1D_ARRAY;
							break;
						case IGPUImage::E_TYPE::ET_3D:
							viewType = IGPUImageView::E_TYPE::ET_3D;
							break;
						default:
							break;
					}
					// creating and hashing those ahead of time makes no sense, because all `imagesToUpload` have different hashes
					auto srcView = device->createImageView({
						.flags = IGPUImageView::ECF_NONE,
						.subUsages = IGPUImage::E_USAGE_FLAGS::EUF_SAMPLED_BIT,
						.image = core::smart_refctd_ptr<IGPUImage>(image),
						.viewType = viewType,
						.format = format,
						.subresourceRange = {}
					});
					// its our own resource, it will eventually be free
					while (dsAlloc->multi_allocate(SrcMipBinding,1,&srcIx)!=0)
					{
						if (drainBoth()!=IQueue::RESULT::SUCCESS)
							break;
						//params.compute->overflowCallback(); // erm what semaphore would we even be waiting for? TODO: need an event handler/timeline method to give lowest latch event/semaphore value
						dsAlloc->cull_frees();
					}
					if (!quickWriteDescriptor(SrcMipBinding,srcIx,std::move(srcView)))
					{
						markFailure("Source Mip Level Descriptor Write",&cpuImg,pFound);
						continue;
					}
				}
				// there might be some QFOT releases from transfer to compute which need to happen before we execute Compute
				auto drain = [&]()->bool
				{
					// Transfer and Compute barriers get recorded for image individually (see the TODO why its horrible)
					// so we only need to worry about QFOTs for current image if they even exist
					if (item.second.recomputeMips && !transferBarriers.empty())
					{
						// so now we need a immeidate QFOT Release cause we already recorded some compute mipmapping for current image
						if (pipelineBarrier(xferCmdBuf,{.memBarriers={},.bufBarriers={},.imgBarriers=transferBarriers},"Recording QFOT Release from Transfer Queue Family after overflow failed"))
						{
							// a transfer microsubmit just for QFOT Release will need to be performed before Compute
							if (drainBoth()!=IQueue::RESULT::SUCCESS)
								return false;
							transferBarriers.clear();
						}
						else
						{
							markFailure("Image QFOT Pipeline Barrier",&cpuImg,pFound);
							return false;
						}
						return true;
					}
					else // Transfer just got submitted due to staging buffer overflow, compute has all the data required to start
						return drainCompute()==IQueue::RESULT::SUCCESS;
				};
				//
				using layout_t = IGPUImage::LAYOUT;
				// record optional transitions to transfer/mip recompute layout and optional transfers, then transitions to desired layout after transfer
				{
					transferBarriers.clear();
					computeBarriers.clear();
					const bool concurrentSharing = image->getCachedCreationParams().isConcurrentSharing();
					uint8_t lvl = 0;
					const auto recomputeMipMask = item.second.recomputeMips;
					bool _prevRecompute = false;
					for (; lvl<creationParams.mipLevels; lvl++)
					{
						// always start with a new struct to not get stale/old value bugs
						IGPUCommandBuffer::SImageMemoryBarrier<IGPUCommandBuffer::SOwnershipTransferBarrier> tmp =
						{
							.barrier = {
								.dep = {
									// first usage doesn't need to sync against anything, so leave src default
									.srcStageMask = PIPELINE_STAGE_FLAGS::NONE,
									.srcAccessMask = ACCESS_FLAGS::NONE
								} // fill the rest later
							},
							.image = image,
							.subresourceRange = {
								.aspectMask = aspects,
								.baseMipLevel = lvl,
								// we'll always do one level at a time
								.levelCount = 1,
								// all the layers
								.baseArrayLayer = 0,
								.layerCount = creationParams.arrayLayers
							},
							// first use, can transition away from undefined straight into what we want
							.oldLayout = layout_t::UNDEFINED
						};
						auto& barrier = tmp.barrier;
						// if any op, it will always be a release (Except acquisition of first source mip in compute)
						barrier.ownershipOp = ownership_op_t::RELEASE;
						// if we're recomputing this mip level 
						const bool recomputeMip = lvl && (recomputeMipMask&(0x1u<<(lvl-1)));
						// query final layout from callback
						const auto finalLayout = params.getFinalLayout(image,pFound->cacheKey.value,lvl);
						// get region data for upload
						auto regions = cpuImg->getRegions(lvl);
						// basic error checks
						const bool prevRecomputed = _prevRecompute;
						_prevRecompute = false;
						if (finalLayout==layout_t::UNDEFINED && !regions.empty() && !recomputeMip)
						{
							logger.log("What are you doing requesting layout UNDEFINED for mip level % of image %s after Upload or Mip Recomputation!?",system::ILogger::ELL_ERROR,lvl,image->getObjectDebugName());
							break;
						}
						const auto suggestedFinalOwner = params.getFinalOwnerQueueFamily(image,pFound->cacheKey.value,lvl);
						// if we'll recompute the mipmap, then do the layout transition on the compute queue (there's one less potential QFOT)
						if (recomputeMip)
						{
							// query final owner from callback
							const auto finalOwnerQueueFamily = checkOwnership(image,suggestedFinalOwner,computeFamily);
							if (finalOwnerQueueFamily==QueueFamilyInvalid)
								break;
							// layout transition from UNDEFINED to GENERAL
							{
								// both src and dst will be used and have layout general in the dispatch to come
								barrier.dep.dstStageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT;
								tmp.newLayout = layout_t::GENERAL;
								// destination level
								{
									// empty mips are written by the compute shader in the first dispatch
									barrier.dep.dstAccessMask = ACCESS_FLAGS::STORAGE_WRITE_BIT;
									// first usage acquires ownership so we're good
								}
								// source and destination level barrier
								decltype(tmp) preComputeBarriers[2] = { tmp,tmp };
								{
									auto& source = preComputeBarriers[1];
									// we will read from this level
									source.barrier.dep.dstAccessMask = ACCESS_FLAGS::STORAGE_READ_BIT|ACCESS_FLAGS::SAMPLED_READ_BIT;
									// now what happened before depends on:
									if (prevRecomputed)
									{
										// compute wrote into this image level we will use as source
										source.barrier.dep.srcStageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT;
										source.barrier.dep.srcAccessMask = ACCESS_FLAGS::STORAGE_WRITE_BIT;
										// no ownership acquire, we already have it acquired
									}
									else 
									{
										// transfer filled this image level
										if (uniQueue)
										{
											source.barrier.dep.srcStageMask = PIPELINE_STAGE_FLAGS::COPY_BIT;
											source.barrier.dep.srcAccessMask = ACCESS_FLAGS::TRANSFER_WRITE_BIT;
										}
										// else, no source masks because semaphore wait is right behind us
										else if (!concurrentSharing && computeFamily!=transferFamily)
										{
											source.barrier.otherQueueFamilyIndex = transferFamily;
											// exemption to the rule, this time we acquire if necessary
											source.barrier.ownershipOp = ownership_op_t::ACQUIRE;
										}
										// Theoretically we could exclude this subresource from the barrier if there's no QFOT and transfer and compute are different queues
										// but we have another subresource with subset stage flags in the barrier anyway.
									}
									// should have already been in GENERAL by now
									source.oldLayout = layout_t::GENERAL;
								}
								if (!pipelineBarrier(computeCmdBuf,{.memBarriers={},.bufBarriers={},.imgBarriers=preComputeBarriers},"Failed to record pre-mipmapping-dispatch pipeline barrier!"))
									break;
								submitsNeeded |= IQueue::FAMILY_FLAGS::COMPUTE_BIT;
							}
							//
							{
								// If format cannot be stored directly, alias the texel block to a compatible `uint_t` format and we'll encode manually.
								auto storeFormat = format;
								if (!physDev->getImageFormatUsages(image->getTiling())[format].storageImage)
								switch (image->getTexelBlockInfo().getBlockByteSize())
								{
									case 1:
										storeFormat = asset::EF_R8_UINT;
										break;
									case 2:
										storeFormat = asset::EF_R16_UINT;
										break;
									case 4:
										storeFormat = asset::EF_R32_UINT;
										break;
									case 8:
										storeFormat = asset::EF_R32G32_UINT;
										break;
									case 16:
										storeFormat = asset::EF_R32G32B32A32_UINT;
										break;
									case 32:
										storeFormat = asset::EF_R64G64B64A64_UINT;
										break;
									default:
										assert(false);
										break;
								}
								// no point caching this view, has to be created individually for each mip level with modified format
								auto dstView = device->createImageView({
									.flags = IGPUImageView::ECF_NONE,
									.subUsages = IGPUImage::E_USAGE_FLAGS::EUF_STORAGE_BIT,
									.image = core::smart_refctd_ptr<IGPUImage>(image),
									.viewType = viewType,
									.format = format,
									.subresourceRange = {
										.aspectMask = IGPUImage::EAF_COLOR_BIT,
										.baseMipLevel = lvl,
										.levelCount = 1
									}
								});
								// TODO: move to blit ext
								struct PushConstant
								{
									uint64_t srcIx : 22;
									uint64_t srcMipLevel : 20;
									uint64_t dstIx : 22;
								} pc;
								pc.srcIx = srcIx;
								pc.srcMipLevel = lvl-1;
								// allocate slot for the output image view and write it
								{
									auto dstIx = SubAllocatedDescriptorSet::invalid_value;
									for (uint32_t i=0; dsAlloc->try_multi_allocate(DstMipBinding,1,&dstIx)!=0; i++)
									{
										if (i) // don't submit on first fail
										if (!drain())
											break;
										dsAlloc->cull_frees();
									}
									if (quickWriteDescriptor(DstMipBinding,dstIx,std::move(dstView)))
									{
										pc.dstIx = dstIx; // before gets wiped
										dsAlloc->multi_deallocate(DstMipBinding,1,&dstIx,params.compute->getFutureScratchSemaphore());
									}
									else
									{
										dsAlloc->multi_deallocate(DstMipBinding,1,&dstIx,{});
										break;
									}
									pc.dstIx = dstIx;
								}
							}
// TODO: push constants, and dispatch compute shader
							_prevRecompute = true;
							// record deferred layout transitions and QFOTs
							{
								// protect against anything that may overlap our compute and optional layout transition end
								barrier.dep = barrier.dep.nextBarrier(PIPELINE_STAGE_FLAGS::ALL_COMMANDS_BITS,ACCESS_FLAGS::MEMORY_READ_BITS|ACCESS_FLAGS::MEMORY_WRITE_BITS);
								tmp.oldLayout = tmp.newLayout;
								tmp.newLayout = finalLayout;
								// if the queue is ignored, nothing will happen
								barrier.otherQueueFamilyIndex = finalOwnerQueueFamily;
								computeBarriers.push_back(tmp);
							}
						}
						else
						{
							// query final owner from callback
							const auto finalOwnerQueueFamily = checkOwnership(image,suggestedFinalOwner,transferFamily);
							if (finalOwnerQueueFamily==QueueFamilyInvalid)
								break;
							// a non-recomputed mip level can either be empty or have content
							if (regions.empty())
							{
								// nothing needs to be done, evidently user wants to transition this mip level themselves
								if (finalLayout!=layout_t::UNDEFINED)
									continue;
								// such an action makes no sense but we'll respect it
								if (finalOwnerQueueFamily!=IQueue::FamilyIgnored)
								{
									// issue a warning, because your application code will just be more verbose (you still have to place an almost identical barrier on a queue in the acquiring family)
									logger.log(
										"It makes no sense to split-transition mip-level %d of image %s to layout %d with a QFOT to %d as no queue owns it yet! Just keep it in UNDEFINED and do the transition on the final owning queue yourself without the ownership acquire.",
										system::ILogger::ELL_PERFORMANCE,lvl,image->getObjectDebugName(),finalLayout,finalOwnerQueueFamily
									);
									// ownership release just needs to happen before semaphore signal
									barrier.dep.dstStageMask = PIPELINE_STAGE_FLAGS::NONE;
									barrier.dep.dstAccessMask = ACCESS_FLAGS::NONE;
								}
								else
								{
									// protect against anything that may overlap our layout transition end
									barrier.dep.dstStageMask = PIPELINE_STAGE_FLAGS::ALL_COMMANDS_BITS;
									barrier.dep.dstAccessMask = ACCESS_FLAGS::MEMORY_READ_BITS|ACCESS_FLAGS::MEMORY_WRITE_BITS;
								}
								// go straight to final layout
								tmp.newLayout = finalLayout;
								// if the queue is ignored, nothing will happen
								barrier.otherQueueFamilyIndex = finalOwnerQueueFamily;
							}
							else
							{
								// need to transition layouts before data upload
								barrier.dep.dstStageMask = PIPELINE_STAGE_FLAGS::COPY_BIT;
								barrier.dep.dstAccessMask = ACCESS_FLAGS::TRANSFER_WRITE_BIT;
								// whether next mip will need to read from this one to recompute itself
								const bool sourceForNextMipCompute = item.second.recomputeMips&(0x1u<<lvl);
								// keep in general layout to avoid a transfer->general transition
								tmp.newLayout = sourceForNextMipCompute ? layout_t::GENERAL : layout_t::TRANSFER_DST_OPTIMAL;
								// fire off the pipeline barrier so we can start uploading right away
								if (!pipelineBarrier(xferCmdBuf,{.memBarriers={},.bufBarriers={},.imgBarriers={&tmp,1}},"Initial Pre-Image-Region-Upload Layout Transition failed!"))
									break;
								// first use owns
								submitsNeeded |= IQueue::FAMILY_FLAGS::TRANSFER_BIT;
								// start recording uploads
								{
									const auto oldImmediateSubmitSignalValue = params.transfer->scratchSemaphore.value;
									if (!params.utilities->updateImageViaStagingBuffer(*params.transfer,cpuImg->getBuffer()->getPointer(),cpuImg->getCreationParameters().format,image,tmp.newLayout,regions))
									{
										logger.log("Image Region Upload failed!", system::ILogger::ELL_ERROR);
										break;
									}
									// stall callback is only called if multiple buffering of scratch commandbuffers fails, we also want to submit compute if transfer was submitted
									if (oldImmediateSubmitSignalValue != params.transfer->scratchSemaphore.value)
									{
										// and our recording scratch commandbuffer most likely changed
										xferCmdBuf = params.transfer->getCommandBufferForRecording();
										if (!drain())
											break;
									}
								}
								// new layout becomes old
								tmp.oldLayout = tmp.newLayout;
								// good initial variable value from initialization
								assert(barrier.otherQueueFamilyIndex == IQueue::FamilyIgnored);
								// slightly different post-barriers are needed post-upload
								if (sourceForNextMipCompute)
								{
									// If submitting to same queue, then we use compute commandbuffer to perform the barrier between Xfer and compute stages.
									// also do this if no QFOT, because no barrier needed at all as layout stays unchanged and semaphore signal-wait perform big memory barriers
									if (uniQueue || computeFamily==transferFamily || concurrentSharing)
										continue;
									// stay in the same layout, no transition (both match)
									tmp.newLayout = layout_t::GENERAL;
									barrier.otherQueueFamilyIndex = computeFamily;
									// we only sync with semaphore signal
									barrier.dep = barrier.dep.nextBarrier(PIPELINE_STAGE_FLAGS::NONE, ACCESS_FLAGS::NONE);
								}
								else // no usage from compute command buffer
								{
									tmp.newLayout = finalLayout;
									// there shouldn't actually be a QFOT because its the same family or concurrent sharing
									if (finalOwnerQueueFamily != IQueue::FamilyIgnored)
									{
										// QFOT release must only sync against semaphore signal
										barrier.dep = barrier.dep.nextBarrier(PIPELINE_STAGE_FLAGS::NONE, ACCESS_FLAGS::NONE);
									}
									else
									{
										// There is no layout transition to perform, and no QFOT release, all memory and execution dependency will be done externally
										if (tmp.newLayout == tmp.oldLayout)
											continue;
										// Otherwise protect against anything that may overlap our layout transition end if no QFOT
										barrier.dep = barrier.dep.nextBarrier(PIPELINE_STAGE_FLAGS::ALL_COMMANDS_BITS, ACCESS_FLAGS::MEMORY_READ_BITS | ACCESS_FLAGS::MEMORY_WRITE_BITS);
									}
									barrier.otherQueueFamilyIndex = finalOwnerQueueFamily;
								}
							}
							// we need a layout transition anyway
							transferBarriers.push_back(tmp);
						}
					}
					// failed in the for-loop
					if (lvl != creationParams.mipLevels)
					{
						markFailure("Compute Mip Mapping",&cpuImg,pFound);
						continue;
					}
					// let go of canonical asset (may free RAM)
					cpuImg = nullptr;
				}
				// here we only record barriers that do final layout transitions and release ownership to final queue family
				if (!transferBarriers.empty())
				{
					if (!pipelineBarrier(xferCmdBuf,{.memBarriers={},.bufBarriers={},.imgBarriers=transferBarriers},"Final Pipeline Barrier recording to Transfer Command Buffer failed"))
					{
						markFailure("Image Data Upload Pipeline Barrier",&cpuImg,pFound);
						continue;
					}
					// even if no uploads performed, we do layout transitions on empty images from Xfer Queue
					submitsNeeded |= IQueue::FAMILY_FLAGS::TRANSFER_BIT;
				}
				if (!computeBarriers.empty())
				{
					// the RAII exiter does an immediate "failure deallocation" without any semaphore dependant deferral, so preempt it here
					dsAlloc->multi_deallocate(SrcMipBinding,1,&srcIx,params.compute->getFutureScratchSemaphore());
					if (!pipelineBarrier(computeCmdBuf,{.memBarriers={},.bufBarriers={},.imgBarriers=computeBarriers},"Final Pipeline Barrier recording to Compute Command Buffer failed"))
					{
						markFailure("Compute Mip Mapping Pipeline Barrier",&cpuImg,pFound);
						continue;
					}
				}
			}
			xferCmdBuf->cmdbuf->beginDebugMarker("Asset Converter Upload Images END");
			xferCmdBuf->cmdbuf->endDebugMarker();
			imagesToUpload.clear();
		}

		// Host builds are unsupported
		assert(reservations.m_blasConversions[1].empty() && reservations.m_tlasConversions[1].empty());

		// Acceleration Structures
		if (reservations.willDeviceASBuild())
		{
			// we release BLAS and TLAS Storage Buffer ownership at the same time, because BLASes about to be released may need to be read by TLAS builds
			core::vector<buffer_mem_barrier_t> ownershipTransfers;

			// Device Builds
			auto& blasesToBuild = reservations.m_blasConversions[0];
			auto& tlasesToBuild = reservations.m_tlasConversions[0];
			const auto blasCount = blasesToBuild.size();
			const auto tlasCount = tlasesToBuild.size();
			ownershipTransfers.reserve(blasCount+tlasCount);


			// Right now we build all BLAS first, then all TLAS
			// (didn't fancy horrible concurrency managment taking compactions into account)
			auto queryPool = device->createQueryPool({.queryCount=hlsl::max<uint32_t>(blasCount,tlasCount),.queryType=IQueryPool::ACCELERATION_STRUCTURE_COMPACTED_SIZE});
			
			// leftover for TLAS builds
			using compacted_blas_map_t = unordered_map<const IGPUBottomLevelAccelerationStructure*,smart_refctd_ptr<IGPUBottomLevelAccelerationStructure>>;
			compacted_blas_map_t compactedBLASMap;
			bool failedBLASBarrier = false;
			// returns a map of compacted Acceleration Structures
			auto buildAndCompactASes = [&]<typename AccelerationStructure>(auto& asesToBuild)->unordered_map<const AccelerationStructure*,smart_refctd_ptr<AccelerationStructure>>
			{
				const auto asCount = asesToBuild.size();
				if (asCount==0)
					return {};
				
				constexpr bool IsTLAS = std::is_same_v<AccelerationStructure,IGPUTopLevelAccelerationStructure>;
				using CPUAccelerationStructure = std::conditional_t<IsTLAS,ICPUTopLevelAccelerationStructure,ICPUBottomLevelAccelerationStructure>;

				core::vector<const IGPUAccelerationStructure*> compactions;
				// 0xffFFffFFu when not releasing ownership, otherwise index into `ownershipTransfers` where the ownership release for the old buffer was
				core::vector<uint32_t> compactedOwnershipReleaseIndices;
				compactions.reserve(asCount);
				compactedOwnershipReleaseIndices.reserve(asCount);
				// build
				{
					auto* scratchBuffer = params.scratchForDeviceASBuild->getBuffer();
					core::vector<ILogicalDevice::MappedMemoryRange> flushRanges;
					const bool manualFlush = scratchBuffer->getBoundMemory().memory->haveToMakeVisible();
					if (deviceASBuildScratchPtr && manualFlush) // TLAS builds do max 2 writes each and BLAS do much more anyway
						flushRanges.reserve(asCount*2);
					// lambdas!
					auto streamDataToScratch = [&](const size_t offset, const size_t size,IUtilities::IUpstreamingDataProducer& callback) -> bool
					{
						if (deviceASBuildScratchPtr)
						{
							callback(deviceASBuildScratchPtr+offset,0ull,size);
							if (manualFlush)
								flushRanges.emplace_back(scratchBuffer->getBoundMemory().memory,offset,size,ILogicalDevice::MappedMemoryRange::align_non_coherent_tag);
							return true;
						}
						else
						{
							const SBufferRange<IGPUBuffer> range={.offset=offset,.size=size,.buffer=smart_refctd_ptr<IGPUBuffer>(scratchBuffer)};
							const bool retval = params.utilities->updateBufferRangeViaStagingBuffer(*params.transfer,range,callback);
							// current recording buffer may have changed
							xferCmdBuf = params.transfer->getCommandBufferForRecording();
							return retval;
						}
					};
					//
					core::vector<typename AccelerationStructure::DeviceBuildInfo> buildInfos;
					buildInfos.reserve(asCount);
					using build_range_info_t = std::conditional_t<IsTLAS,typename AccelerationStructure::BuildRangeInfo,const typename AccelerationStructure::BuildRangeInfo*>;
					core::vector<build_range_info_t> rangeInfos;
					rangeInfos.reserve(asCount);
					using scratch_allocator_t = std::remove_reference_t<decltype(*params.scratchForDeviceASBuild)>;
					using addr_t = typename scratch_allocator_t::size_type;
					core::vector<addr_t> allocOffsets;
					allocOffsets.reserve(asCount);
					core::vector<addr_t> allocSizes;
					allocSizes.reserve(asCount);
					// BLAS and TLAS specific things
					core::vector<IGPUBottomLevelAccelerationStructure::BuildRangeInfo> geometryRangeInfo;
					core::vector<IGPUBottomLevelAccelerationStructure::Triangles<IGPUBuffer>> triangles;
					core::vector<IGPUBottomLevelAccelerationStructure::AABBs<IGPUBuffer>> aabbs;
					core::vector<smart_refctd_ptr<const IGPUBottomLevelAccelerationStructure>> trackedBLASes;
					if constexpr (IsTLAS)
						trackedBLASes.reserve(asCount);
					else // would have to count total geometries in BLASes to initialize properly, and we probably don't want to over-reserve
					{
						geometryRangeInfo.reserve(asCount);
						triangles.reserve(asCount);
						aabbs.reserve(asCount);
					}
					//
					core::vector<addr_t> alignments;
					alignments.reserve(asCount*2);
					constexpr auto GeometryIsAABBFlag = IGPUBottomLevelAccelerationStructure::BUILD_FLAGS::GEOMETRY_TYPE_IS_AABB_BIT;
					auto recordBuildCommands = [&]()->void
					{
						bool success = !buildInfos.empty();
						// Lets analyze sync cases:
						// - Mapped Host write = no barrier, flush & optional submit sufficient
						// - Single Queue = Global Memory Barrier
						// - Two distinct Queues = no barrier, semaphore signal-wait is sufficient
						// - Two distinct Queue Families Exclusive Sharing mode = QFOT necessary but we require concurrent sharing on the scratch buffer !
						if (success)
						{
							const asset::SMemoryBarrier readGeometryOrInstanceInASBuildBarrier = {
								// the last use of the source BLAS could have been a build or a compaction
								.srcStageMask = PIPELINE_STAGE_FLAGS::COPY_BIT,
								.srcAccessMask = ACCESS_FLAGS::TRANSFER_WRITE_BIT,
								.dstStageMask = PIPELINE_STAGE_FLAGS::ACCELERATION_STRUCTURE_BUILD_BIT,
								.dstAccessMask = ACCESS_FLAGS::STORAGE_READ_BIT
							};
							success = !uniQueue || deviceASBuildScratchPtr || pipelineBarrier(computeCmdBuf,{.memBarriers={&readGeometryOrInstanceInASBuildBarrier,1}},"Pipeline Barriers of Acceleration Structure backing Buffers failed!");
						}
						//
						constexpr bool IsTLAS = std::is_same_v<AccelerationStructure,IGPUTopLevelAccelerationStructure>;
						if (success)
						{
							// rewrite the based pointers
							if constexpr (IsTLAS)
							for (auto& info : buildInfos)
							{
								const auto offset = info.trackedBLASes.data();
								const auto correctPtr = trackedBLASes.data()+reinterpret_cast<const size_t&>(offset);
								info.trackedBLASes = {reinterpret_cast<const IGPUBottomLevelAccelerationStructure** const&>(correctPtr),info.trackedBLASes.size()};
							}
							else
							{
								for (auto& info : buildInfos)
								{
									if (info.buildFlags.hasFlags(GeometryIsAABBFlag))
										info.aabbs = aabbs.data()+reinterpret_cast<const size_t&>(info.aabbs);
									else
										info.triangles = triangles.data()+reinterpret_cast<const size_t&>(info.triangles);
								}
								for (auto& rangeInfo : rangeInfos)
									rangeInfo = geometryRangeInfo.data()+reinterpret_cast<const size_t&>(rangeInfo);
							}
							success = computeCmdBuf->cmdbuf->buildAccelerationStructures({buildInfos},rangeInfos.data());
						}
						// account for the in-progress allocation (we may be called from an overflow submit)
						const auto oldAllocCount = allocOffsets.size()-alignments.size();
						if (success)
						{
							submitsNeeded |= IQueue::FAMILY_FLAGS::COMPUTE_BIT;
							// queue up a deferred allocation
							if (oldAllocCount)
								params.scratchForDeviceASBuild->multi_deallocate(oldAllocCount,allocOffsets.data(),allocSizes.data(),params.compute->getFutureScratchSemaphore());
						}
						else
						{
							// release right away
							if (oldAllocCount)
								params.scratchForDeviceASBuild->multi_deallocate(oldAllocCount,allocOffsets.data(),allocSizes.data());
							for (const auto& info : buildInfos)
							{
								const auto stagingFound = findInStaging.template operator()<CPUAccelerationStructure>(info.dstAS);
								smart_refctd_ptr<const CPUAccelerationStructure> dummy; // already null at this point
								markFailure("AS Build Command Recording",&dummy,&stagingFound->second);
							}
						}
						allocOffsets.erase(allocOffsets.begin(),allocOffsets.begin()+oldAllocCount);
						allocSizes.erase(allocSizes.begin(),allocSizes.begin()+oldAllocCount);
						buildInfos.clear();
						rangeInfos.clear();
						if constexpr (IsTLAS)
							trackedBLASes.clear();
						else
						{
							geometryRangeInfo.clear();
							triangles.clear();
							aabbs.clear();
						}
					};

					computeCmdBuf->cmdbuf->beginDebugMarker("Asset Converter Build Acceleration Structures START");
					computeCmdBuf->cmdbuf->endDebugMarker();
					const auto& limits = physDev->getLimits();
					for (auto& asToBuild : asesToBuild)
					{
						auto& canonical = asToBuild.second.canonical;
						const auto as = asToBuild.first;
						const auto pFound = &findInStaging.template operator()<CPUAccelerationStructure>(as)->second;
						const auto& backingRange = as->getCreationParams().bufferRange;
						// checking ownership for the future on old buffer, but compacted will be made with same sharing creation parameters
						const auto finalOwnerQueueFamily = checkOwnership(backingRange.buffer.get(),params.getFinalOwnerQueueFamily(as,pFound->cacheKey.value),computeFamily);
						if (finalOwnerQueueFamily==QueueFamilyInvalid)
						{
							markFailure("invalid Final Queue Family given by user callback",&canonical,pFound);
							continue;
						}
						// clean up the allocation if we fail to make it to the end of loop for whatever reason
						alignments.clear();
						auto allocCount = 0;
						auto deallocSrc = core::makeRAIIExiter([&params,&allocOffsets,&allocSizes,&alignments,&allocCount]()->void
							{
								const auto beginIx = allocSizes.size()-allocCount;
								// if got to end of loop queue up the release of memory, otherwise release right away
								if (allocCount)
									params.scratchForDeviceASBuild->multi_deallocate(allocCount,allocOffsets.data()+beginIx,allocSizes.data()+beginIx);
								allocOffsets.resize(beginIx);
								allocSizes.resize(beginIx);
								alignments.clear();
							}
						);
						allocSizes.push_back(asToBuild.second.scratchSize);
						alignments.push_back(limits.minAccelerationStructureScratchOffsetAlignment);
						const bitflag<typename AccelerationStructure::BUILD_FLAGS> buildFlags = asToBuild.second.getBuildFlags();
						if constexpr (IsTLAS)
						{
							const auto instances = canonical->getInstances();
							// gather total input size and check dependants exist
							size_t instanceDataSize = 0;
							bool dependsOnBLASBuilds = false;
							const auto& instanceMap = asToBuild.second.instanceMap;
							for (const auto& instance : instances)
							{
								auto found = instanceMap.find(instance.getBase().blas.get());
								assert(instanceMap.end()!=found);
								const auto depInfo = missingDependent.template operator()<ICPUBottomLevelAccelerationStructure>(found->second.get());
								if (depInfo)
								{
									instanceDataSize = 0;
									break;
								}
								if (depInfo.wasInStaging)
									dependsOnBLASBuilds = true;
								instanceDataSize += ITopLevelAccelerationStructure::getInstanceSize(instance.getType());
							}
							// problem with building some Dependent BLASes
							if (failedBLASBarrier && dependsOnBLASBuilds)
							{
								markFailure("building BLASes which current TLAS build wants to instance",&canonical,pFound);
								continue;
							}
							// problem with finding the dependents (BLASes)
							if (instanceDataSize==0)
							{
								markFailure("finding valid Dependant GPU BLASes for TLAS build",&canonical,pFound);
								continue;
							}
							allocSizes.push_back(instanceDataSize);
							alignments.push_back(16);
							if (as->usesMotion())
							{
								allocSizes.push_back(sizeof(void*)*instances.size());
								alignments.push_back(alignof(uint64_t));
							}
						}
						else
						{
							const uint32_t* pPrimitiveCounts = canonical->getGeometryPrimitiveCounts().data();
							if (buildFlags.hasFlags(GeometryIsAABBFlag))
							{
								for (const auto& geom : canonical->getAABBGeometries())
								if (const auto aabbCount=*(pPrimitiveCounts++); aabbCount)
								{
									allocSizes.push_back(aabbCount*geom.stride);
									alignments.push_back(alignof(float));
								}
							}
							else
							{
								for (const auto& geom : canonical->getTriangleGeometries())
								if (const auto triCount=*(pPrimitiveCounts++); triCount)
								{
									auto size = geom.vertexStride*(geom.vertexData[1] ? 2:1)*(geom.maxVertex+1);
									uint16_t alignment = hlsl::max(0x1u<<hlsl::findLSB(geom.vertexStride),32u);
									if (geom.hasTransform())
									{
										size = core::alignUp(size,alignof(float))+sizeof(hlsl::float32_t3x4);
										alignment = hlsl::max<uint16_t>(alignof(float),alignment);
									}
									uint16_t indexSize = 0u;
									switch (geom.indexType)
									{
										case E_INDEX_TYPE::EIT_16BIT:
											indexSize = alignof(uint16_t);
											break;
										case E_INDEX_TYPE::EIT_32BIT:
											indexSize = alignof(uint32_t);
											break;
										default:
											break;
									}
									if (indexSize)
									{
										size = core::alignUp(size,indexSize)+triCount*3*indexSize;
										alignment = hlsl::max<uint16_t>(indexSize,alignment);
									}
									allocSizes.push_back(size);
									alignments.push_back(alignment);
									const auto tmp = asToBuild.second.scratchSize;
									//logger.log("%p Triangle Data Size %d Align %d Scratch Size %d",system::ILogger::ELL_DEBUG,canonical.get(),size,alignment,tmp);
								}
							}
						}
						allocOffsets.resize(allocSizes.size(),scratch_allocator_t::invalid_value);
						// allocate out scratch or submit overflow, if fail then flush and keep trying till space is made
						auto* offsets = allocOffsets.data()+allocOffsets.size()-alignments.size();
						const auto* sizes = allocSizes.data()+allocSizes.size()-alignments.size();
						//logger.log("%p Combined Size %d",system::ILogger::ELL_DEBUG,canonical.get(),std::accumulate(sizes,sizes+alignments.size(),0));
						for (uint32_t t=0; params.scratchForDeviceASBuild->multi_allocate(alignments.size(),offsets,sizes,alignments.data())!=0; t++)
						{
							if (t==1) // don't flush right away cause allocator not defragmented yet
							{
								recordBuildCommands();
								// the submit overflow deallocates old offsets and erases them from the temp arrays, pointer changes
								offsets = allocOffsets.data();
								sizes = allocSizes.data();
								// if writing to scratch directly, flush the writes
								if (!flushRanges.empty())
								{
									device->flushMappedMemoryRanges(flushRanges);
									flushRanges.clear();
								}
								drainCompute();
							}
							// we may be preventing ourselves from allocating memory, with one successful allocation still being alive and fragmenting our allocator
							params.scratchForDeviceASBuild->multi_deallocate(alignments.size(),offsets,sizes);
							std::fill_n(offsets,alignments.size(),scratch_allocator_t::invalid_value);
						}
						// now upon a failure, our allocations will need to be deallocated
						allocCount = alignments.size();
						// prepare build infos
						typename AccelerationStructure::DeviceBuildInfo buildInfo;
						buildInfo.scratch = {.offset=offsets[0],.buffer=smart_refctd_ptr<IGPUBuffer>(scratchBuffer)};
						buildInfo.buildFlags = buildFlags;
						buildInfo.dstAS = as;
						// abortion backup
						bool success = true;
						const auto geometryRangeInfoOffset = geometryRangeInfo.size();
						const auto trianglesOffset = triangles.size();
						const auto aabbsOffset = aabbs.size();
						const size_t trackedBLASesOffset = trackedBLASes.size();
						if constexpr (IsTLAS)
						{
							const auto instances = canonical->getInstances();
							const auto instanceCount = static_cast<uint32_t>(instances.size());
							// stream the instance/geometry input in
							{
								struct FillInstances : IUtilities::IUpstreamingDataProducer
								{
									uint32_t operator()(void* dst, const size_t offsetInRange, const uint32_t blockSize) override
									{
										using blas_ref_t = IGPUBottomLevelAccelerationStructure::device_op_ref_t;
										assert(offsetInRange%16==0);
											
										uint32_t bytesWritten = 0;
										while (instanceIndex<instances.size())
										{
											const auto& instance = instances[instanceIndex++];
											const auto type = instance.getType();
											const auto size = ITopLevelAccelerationStructure::getInstanceSize(type);
											const auto newWritten = bytesWritten+size;
											if (newWritten>blockSize)
												break;
											auto found = instanceMap->find(instance.getBase().blas.get());
											auto blas = found->second.get();
											if (auto found=compactedBLASMap->find(blas); found!=compactedBLASMap->end())
												blas = found->second.get();
											trackedBLASes->emplace_back(blas);
											dst = IGPUTopLevelAccelerationStructure::writeInstance(dst,instance,blas->getReferenceForDeviceOperations());
											bytesWritten = newWritten;
										}
										return bytesWritten;
									}

									const compacted_blas_map_t* compactedBLASMap;
									core::vector<smart_refctd_ptr<const IGPUBottomLevelAccelerationStructure>>* trackedBLASes;
									SReserveResult::SConvReqTLAS::cpu_to_gpu_blas_map_t* instanceMap;
									std::span<const ICPUTopLevelAccelerationStructure::PolymorphicInstance> instances;
									uint32_t instanceIndex = 0;
								};
								FillInstances fillInstances;
								fillInstances.compactedBLASMap = &compactedBLASMap;
								fillInstances.trackedBLASes = &trackedBLASes;
								fillInstances.instanceMap = &asToBuild.second.instanceMap;
								fillInstances.instances = instances;
								success = streamDataToScratch(offsets[1],sizes[1],fillInstances);
								// provoke refcounting bugs right away
								asToBuild.second.instanceMap.clear();
							}
							if (success && as->usesMotion())
							{
								struct FillInstancePointers : IUtilities::IUpstreamingDataProducer
								{
									uint32_t operator()(void* dst, const size_t offsetInRange, const uint32_t blockSize) override
									{
										constexpr uint32_t ptr_sz = sizeof(uint64_t);

										const uint32_t count = blockSize/ptr_sz;
										assert(offsetInRange%ptr_sz==0);
										const uint32_t baseInstance = static_cast<uint32_t>(offsetInRange)/ptr_sz;
										for (uint32_t i=0; i<count; i++)
										{
											const auto type = instances[baseInstance+i].getType();
											reinterpret_cast<uint64_t*>(dst)[i] = IGPUTopLevelAccelerationStructure::encodeTypeInAddress(type,instanceAddress);
											instanceAddress += ITopLevelAccelerationStructure::getInstanceSize(type);
										}
										return count*ptr_sz;
									}

									std::span<const ICPUTopLevelAccelerationStructure::PolymorphicInstance> instances;
									uint64_t instanceAddress;
								};
								FillInstancePointers fillInstancePointers;
								fillInstancePointers.instances = instances;
								fillInstancePointers.instanceAddress = scratchBuffer->getDeviceAddress()+offsets[1];
								success = streamDataToScratch(offsets[2],sizes[2],fillInstancePointers);
							}
							//
							buildInfo.instanceDataTypeEncodedInPointersLSB = as->usesMotion();
							// note we don't build directly from staging, because only very small inputs could come from there and they'd impede the transfer efficiency of the larger ones
							buildInfo.instanceData = {.offset=offsets[as->usesMotion() ? 2:1],.buffer=smart_refctd_ptr<IGPUBuffer>(scratchBuffer)};
							// be based cause vectors can grow
							using p_p_BLAS_t = const IGPUBottomLevelAccelerationStructure**;
							buildInfo.trackedBLASes = {reinterpret_cast<const p_p_BLAS_t&>(trackedBLASesOffset),trackedBLASes.size()-trackedBLASesOffset};
							// no special extra byte offset into the instance buffer
							rangeInfos.emplace_back(instanceCount,0u);
						}
						else
						{
							buildInfo.geometryCount = canonical->getGeometryCount();
							const auto* offsetIt = offsets+1;
							const auto* sizeIt = sizes+1;
							const auto primitiveCounts = canonical->getGeometryPrimitiveCounts();
							for (const auto count : primitiveCounts)
								geometryRangeInfo.push_back({
									.primitiveCount = count,
									.primitiveByteOffset = 0,
									.firstVertex = 0,
									.transformByteOffset = 0
								});	
							const uint32_t* pPrimitiveCounts = primitiveCounts.data();
							IUtilities::CMemcpyUpstreamingDataProducer memcpyCallback;
							if (buildFlags.hasFlags(GeometryIsAABBFlag))
							{
								for (const auto& geom : canonical->getAABBGeometries())
								if (const auto aabbCount=*(pPrimitiveCounts++); aabbCount)
								{
									auto offset = *(offsetIt++);
									memcpyCallback.data = reinterpret_cast<const uint8_t*>(geom.data.buffer->getPointer())+geom.data.offset;
									if (!streamDataToScratch(offset,*(sizeIt++),memcpyCallback))
										break;
									aabbs.push_back({
										.data = {.offset=offset,.buffer=smart_refctd_ptr<const IGPUBuffer>(scratchBuffer)},
										.stride = geom.stride,
										.geometryFlags = geom.geometryFlags
									});
								}
								buildInfo.aabbs = reinterpret_cast<const IGPUBottomLevelAccelerationStructure::AABBs<IGPUBuffer>* const&>(aabbsOffset);
							}
							else
							{
								for (const auto& geom : canonical->getTriangleGeometries())
								if (const auto triCount=*(pPrimitiveCounts++); triCount)
								{
									auto& outGeom = triangles.emplace_back();
									const auto origSize = *(sizeIt++);
									const auto origOffset = *(offsetIt++);
									auto offset = origOffset;
									auto size = geom.vertexStride*(geom.maxVertex+1);
									for (auto i=0; i<2; i++)
									if (geom.vertexData[i]) // could assert that it must be true for i==0
									{
										outGeom.vertexData[i] = {.offset=offset,.buffer=smart_refctd_ptr<const IGPUBuffer>(scratchBuffer)};
										memcpyCallback.data = reinterpret_cast<const uint8_t*>(geom.vertexData[i].buffer->getPointer())+geom.vertexData[i].offset;
										if (!streamDataToScratch(offset,size,memcpyCallback))
											break;
										offset += size;
									}
									if (geom.hasTransform())
									{
										offset = core::alignUp(offset,alignof(float));
										outGeom.transform = {.offset=offset,.buffer=smart_refctd_ptr<const IGPUBuffer>(scratchBuffer)};
										memcpyCallback.data = &geom.transform;
										if (!streamDataToScratch(offset,sizeof(geom.transform),memcpyCallback))
											break;
										offset += sizeof(geom.transform);
									}
									switch (geom.indexType)
									{
										case E_INDEX_TYPE::EIT_16BIT: [[fallthrough]];
										case E_INDEX_TYPE::EIT_32BIT:
										{
											const auto alignment = geom.indexType==E_INDEX_TYPE::EIT_16BIT ? alignof(uint16_t):alignof(uint32_t);
											offset = core::alignUp(offset,alignment);
											outGeom.indexData = {.offset=offset,.buffer=smart_refctd_ptr<const IGPUBuffer>(scratchBuffer)};
											size = triCount*3*alignment;
											memcpyCallback.data = reinterpret_cast<const uint8_t*>(geom.indexData.buffer->getPointer())+geom.indexData.offset;
											success = streamDataToScratch(offset,size,memcpyCallback);
											offset += size;
											break;
										}
										default:
											break;
									}
									assert(offset-origOffset<=origSize);
									if (!success)
										break;
									outGeom.maxVertex = geom.maxVertex;
									outGeom.vertexStride = geom.vertexStride;
									outGeom.vertexFormat = geom.vertexFormat;
									outGeom.indexType = geom.indexType;
									outGeom.geometryFlags = geom.geometryFlags;
								}
								buildInfo.triangles = reinterpret_cast<const IGPUBottomLevelAccelerationStructure::Triangles<IGPUBuffer>* const&>(trianglesOffset);
							}
							success = pPrimitiveCounts==primitiveCounts.data()+primitiveCounts.size();
							rangeInfos.push_back(reinterpret_cast<const IGPUBottomLevelAccelerationStructure::BuildRangeInfo* const&>(geometryRangeInfoOffset));
						}
						if (!success)
						{
							rangeInfos.resize(buildInfos.size());
							geometryRangeInfo.resize(geometryRangeInfoOffset);
							triangles.resize(trianglesOffset);
							aabbs.resize(aabbsOffset);
							trackedBLASes.resize(trackedBLASesOffset);
							markFailure("Uploading Input Data for Accleration Structure build failed",&canonical,pFound);
							continue;
						}
						buildInfos.emplace_back(std::move(buildInfo));
						allocCount = 0;
						// let go of canonical asset (may free RAM)
						canonical = nullptr;
						//
						const bool willCompact = asToBuild.second.compact;
						if (willCompact)
							compactions.push_back(as);
						// enqueue ownership release if necessary
						if (finalOwnerQueueFamily!=IQueue::FamilyIgnored)
						{
							if (willCompact)
								compactedOwnershipReleaseIndices.push_back(ownershipTransfers.size());
							ownershipTransfers.push_back({
								.barrier = {
									.dep = {
										.srcStageMask = PIPELINE_STAGE_FLAGS::ACCELERATION_STRUCTURE_BUILD_BIT,
										.srcAccessMask = ACCESS_FLAGS::ACCELERATION_STRUCTURE_WRITE_BIT
										// leave rest empty, we can release whenever after the copies and before the semaphore signal
									},
									.ownershipOp = ownership_op_t::RELEASE,
									.otherQueueFamilyIndex = finalOwnerQueueFamily
								},
								.range = backingRange
							});
						}
						else if (willCompact)
							compactedOwnershipReleaseIndices.push_back(~0u);
					}
					// finish the last batch
					recordBuildCommands();
					computeCmdBuf->cmdbuf->beginDebugMarker("Asset Converter Build Acceleration Structures END");
					computeCmdBuf->cmdbuf->endDebugMarker();
					// provoke refcounting bugs
					asesToBuild.clear();
					// flush all ranged before potential submit
					if (!flushRanges.empty())
					{
						device->flushMappedMemoryRanges(flushRanges);
						flushRanges.clear();
					}
				}

				// Not messing around with listing AS backing buffers individually, ergonomics of that are null 
				const asset::SMemoryBarrier readASInASCompactBarrier = {
					.srcStageMask = PIPELINE_STAGE_FLAGS::ACCELERATION_STRUCTURE_BUILD_BIT,
					.srcAccessMask = ACCESS_FLAGS::ACCELERATION_STRUCTURE_WRITE_BIT,
					// TODO: do queries or query retrieval have a stage?
					.dstStageMask = PIPELINE_STAGE_FLAGS::ACCELERATION_STRUCTURE_COPY_BIT,
					.dstAccessMask = ACCESS_FLAGS::ACCELERATION_STRUCTURE_READ_BIT
				};
				if (!compactions.empty() && 
					pipelineBarrier(computeCmdBuf,{.memBarriers={&readASInASCompactBarrier,1}},"Failed to sync Acceleration Structure builds with compactions!") &&
					computeCmdBuf->cmdbuf->resetQueryPool(queryPool.get(),0,compactions.size()) &&
					computeCmdBuf->cmdbuf->writeAccelerationStructureProperties(compactions,IQueryPool::TYPE::ACCELERATION_STRUCTURE_COMPACTED_SIZE,queryPool.get(),0)
				)
				{
					// clean AS builds, pipeline barrier, query reset and writes need to get executed before we start waiting on the results
					drainBoth();
					// get queries
					core::vector<size_t> sizes(compactions.size());
					if (!device->getQueryPoolResults(queryPool.get(),0,compactions.size(),sizes.data(),sizeof(size_t),bitflag(IQueryPool::RESULTS_FLAGS::WAIT_BIT)|IQueryPool::RESULTS_FLAGS::_64_BIT))
					{
						logger.log("Failed to Query %sLevelAccelerationStructure compacted sizes, skipping compaction!",system::ILogger::ELL_ERROR,IsTLAS ? "Top":"Bottom");
						return {};
					}
					//
					auto logFail = [logger](const char* msg, const IGPUAccelerationStructure* as)->void
					{
						logger.log("Failed to %s for \"%s\"",system::ILogger::ELL_ERROR,msg,as->getObjectDebugName());
					};
					// try to allocate memory for 
					core::vector<asset_cached_t<ICPUBuffer>> backingBuffers(compactions.size());
					{
						MetaDeviceMemoryAllocator deferredAllocator(params.compactedASAllocator ? params.compactedASAllocator:device,logger);
						// create
						for (size_t i=0; i<compactions.size(); i++)
						{
							const auto* as = static_cast<const AccelerationStructure*>(compactions[i]);
							assert(as);
							// silently skip if not worth it
							if (!params.confirmCompact(sizes[i],as))
							{
								logger.log("Compaction not confirmed for \"%s\" would be compacted size is %d, original %d.",system::ILogger::ELL_DEBUG,as->getObjectDebugName(),sizes[i],as->getCreationParams().bufferRange.size);
								continue;
							}
							// create backing buffer and request an allocation for it
							{
								const auto* oldBuffer = as->getCreationParams().bufferRange.buffer.get();
								assert(oldBuffer);
								// This is a Spec limit/rpomise we don't even expose it
								constexpr size_t MinASBufferAlignment = 256u;
								using usage_f = IGPUBuffer::E_USAGE_FLAGS;
								IGPUBuffer::SCreationParams creationParams = { {.size=core::roundUp(sizes[i],MinASBufferAlignment),.usage=usage_f::EUF_ACCELERATION_STRUCTURE_STORAGE_BIT|usage_f::EUF_SHADER_DEVICE_ADDRESS_BIT},{}};
								// same sharing setup as the previous AS buffer
								creationParams.queueFamilyIndexCount = oldBuffer->getCachedCreationParams().queueFamilyIndexCount;
								creationParams.queueFamilyIndices = oldBuffer->getCachedCreationParams().queueFamilyIndices;
								auto buf = device->createBuffer(std::move(creationParams));
								if (!buf)
								{
									logFail("create Buffer backing the Compacted Acceleration Structure",as);
									continue;
								}
								auto bufReqs = buf->getMemoryReqs();
								backingBuffers[i].value = std::move(buf);
								// allocate new memory - definitely don't want to be raytracing from across the PCIE slot
								if (!deferredAllocator.request(backingBuffers.data()+i,physDev->getDeviceLocalMemoryTypeBits()))
								{
									logFail("request of a Memory Allocation for the Buffer backing the Compacted Acceleration Structure",as);
									continue;
								}
							}
						}
						// allocate memory for the buffers
						deferredAllocator.finalize();
						unordered_map<const AccelerationStructure*,smart_refctd_ptr<AccelerationStructure>> retval;
						retval.reserve(compactions.size());
						// recreate Acceleration Structures
						for (size_t i=0; i<compactions.size(); i++)
						if (backingBuffers[i])
						{
							const auto* srcAS = static_cast<const AccelerationStructure*>(compactions[i]);
							auto& backingBuffer = backingBuffers[i].value;
							if (!backingBuffer->getBoundMemory().isValid())
							{
								logFail("allocate Memory for the Buffer backing the Compacted Acceleration Structure",srcAS);
								continue;
							}
							smart_refctd_ptr<AccelerationStructure> compactedAS;
							{
								typename AccelerationStructure::SCreationParams creationParams = {srcAS->getCreationParams()};
								creationParams.bufferRange = {.offset=0,.size=sizes[i],.buffer=std::move(backingBuffer)};
								if constexpr (IsTLAS)
								{
									creationParams.maxInstanceCount = srcAS->getMaxInstanceCount();
									compactedAS = device->createTopLevelAccelerationStructure(std::move(creationParams));
								}
								else
									compactedAS = device->createBottomLevelAccelerationStructure(std::move(creationParams));
							}
							if (!compactedAS)
							{
								logFail("create the Compacted Acceleration Structure",srcAS);
								continue;
							}
							// set the debug name
							{
								std::string debugName = srcAS->getObjectDebugName();
								debugName += " compacted";
								compactedAS->setObjectDebugName(debugName.c_str());
							}
							// record compaction
							if (!computeCmdBuf->cmdbuf->copyAccelerationStructure<AccelerationStructure>({.src=srcAS,.dst=compactedAS.get(),.compact=true}))
							{
								logFail("record Acceleration Structure compaction",compactedAS.get());
								continue;
							}
							// modify the ownership release to be for the final compacted AS
							if (const auto ix=compactedOwnershipReleaseIndices[i]; ix<ownershipTransfers.size())
								ownershipTransfers[ix].range = compactedAS->getCreationParams().bufferRange;
							// swap out the conversion result
							const auto foundIx = outputReverseMap.find(srcAS);
							if (foundIx!=outputReverseMap.end())
							{
								auto& resultOutput = std::get<SReserveResult::vector_t<CPUAccelerationStructure>>(reservations.m_gpuObjects);
								resultOutput[foundIx->second].value = compactedAS;
							}
							// overwrite staging cache
							auto pFound = findInStaging.template operator()<CPUAccelerationStructure>(srcAS);
							pFound->second.gpuRef = compactedAS;
							// insert into compaction map
							retval[srcAS] = std::move(compactedAS);
						}
						return retval;
					}
					computeCmdBuf->cmdbuf->beginDebugMarker("Asset Converter Compact Acceleration Structures START");
					computeCmdBuf->cmdbuf->endDebugMarker();
					computeCmdBuf->cmdbuf->beginDebugMarker("Asset Converter Compact Acceleration Structures END");
					computeCmdBuf->cmdbuf->endDebugMarker();
				}
				return {};
			};

			// compacted BLASes need to be substituted in cache and TLAS Build Inputs
			compactedBLASMap = buildAndCompactASes.template operator()<IGPUBottomLevelAccelerationStructure>(blasesToBuild);
			// Device TLAS builds
			if (tlasCount)
			{
				// either we built no BLASes (remember we could retrieve already built ones from cache)
				if (blasCount)
				{
					// Or we barrier for the previous compactions or builds (a single pipeline barrier to ensure BLASes build before TLASes is needed)
					const asset::SMemoryBarrier readBLASInTLASBuildBarrier = {
						// the last use of the source BLAS could have been a build or a compaction
						.srcStageMask = PIPELINE_STAGE_FLAGS::ACCELERATION_STRUCTURE_BUILD_BIT|PIPELINE_STAGE_FLAGS::ACCELERATION_STRUCTURE_COPY_BIT,
						.srcAccessMask = ACCESS_FLAGS::ACCELERATION_STRUCTURE_WRITE_BIT,
						.dstStageMask = PIPELINE_STAGE_FLAGS::ACCELERATION_STRUCTURE_BUILD_BIT,
						.dstAccessMask = ACCESS_FLAGS::ACCELERATION_STRUCTURE_READ_BIT
					};
					// submit because we want to launch BLAS builds in a separate submit, so the scratch semaphore can signal and free the scratch and more is available for TLAS builds
					if (pipelineBarrier(computeCmdBuf,{.memBarriers={&readBLASInTLASBuildBarrier,1}},"Failed to sync BLAS with TLAS build!"))
						drainBoth();
					else
						failedBLASBarrier = true;
				}
				compactedTLASMap = buildAndCompactASes.template operator()<IGPUTopLevelAccelerationStructure>(tlasesToBuild);
			}

			// release ownership
			if (!ownershipTransfers.empty())
				pipelineBarrier(computeCmdBuf,{.memBarriers={},.bufBarriers=ownershipTransfers},"Ownership Releases of Acceleration Structure backing Buffers failed!");
		}

		const bool computeSubmitIsNeeded = submitsNeeded.hasFlags(IQueue::FAMILY_FLAGS::COMPUTE_BIT);
		// first submit transfer
		if (submitsNeeded.hasFlags(IQueue::FAMILY_FLAGS::TRANSFER_BIT))
		{
			// if there's still a compute submit to perform, then we will signal the extra semaphores from there
			constexpr auto emptySignalSpan = std::span<const IQueue::SSubmitInfo::SSemaphoreInfo>{};
			if (params.transfer->submit(xferCmdBuf,computeSubmitIsNeeded ? emptySignalSpan:params.extraSignalSemaphores)!=IQueue::RESULT::SUCCESS)
				return retval;
			// leave open for next user
			params.transfer->beginNextCommandBuffer(xferCmdBuf);
			// set the future
			if (!computeSubmitIsNeeded)
				retval.set({params.transfer->scratchSemaphore.semaphore,params.transfer->scratchSemaphore.value});
		}
		// reset original callback
		if (bool(origXferStallCallback))
			params.transfer->overflowCallback = std::move(origXferStallCallback);
		
		// Its too dangerous to leave an Intended Transfer Submit hanging around that needs to be submitted for Compute to make forward progress outside of this utility,
		// and doing transfer-signals-after-compute-wait timeline sema tricks are not and option because:
		// - Bricking the Compute Queue if user forgets to submit the transfers!
		// - Opaquely blocking any workload they would submit between now and when they submit the coonvert result open Compute scratch
		// - Violating the spec if anyone submits binary semaphore or fence signals on the Compute Queue between now and when 
		//   the open Transfer Scratch Command Buffer is submitted (e.g. swapchain or some other external Native use of Vulkan)
		if (computeSubmitIsNeeded)
		{
			// we may as well actually submit the compute commands instead of doing a silly empty submit to sync compute with transfer
			// the only other option is to literally have the coupling between transfer and compute explicit in the public api
			if (drainCompute(params.extraSignalSemaphores)!=IQueue::RESULT::SUCCESS)
				return retval;
			retval.set({params.compute->scratchSemaphore.semaphore,params.compute->scratchSemaphore.value});
		}
	}
	
	// finish host tasks if not done yet
	hostUploadBuffers([]()->bool{return true;});
	// in the future we'll also finish host image copies

	// check dependents before inserting into cache
	if (reservations.m_queueFlags.value!=IQueue::FAMILY_FLAGS::NONE)
	{
		auto checkDependents = [&]<Asset AssetType>()->void
		{
			auto& stagingCache = std::get<SReserveResult::staging_cache_t<AssetType>>(reservations.m_stagingCaches);
			phmap::erase_if(stagingCache,[&](auto& item)->bool
				{
					auto* pGpuObj = item.first;
					// rescan all the GPU objects and find out if they depend on anything that failed, if so add to failure set
					bool depsMissing = false;
					if constexpr (std::is_same_v<AssetType,ICPUBufferView>)
						depsMissing = missingDependent.template operator()<ICPUBuffer>(pGpuObj->getUnderlyingBuffer());
					if constexpr (std::is_same_v<AssetType,ICPUImageView>)
						depsMissing = missingDependent.template operator()<ICPUImage>(pGpuObj->getCreationParameters().image.get());
					if constexpr (std::is_same_v<AssetType,ICPUDescriptorSet>)
					{
						const IGPUDescriptorSetLayout* layout = pGpuObj->getLayout();
						// check samplers
						{
							const auto count = layout->getTotalMutableCombinedSamplerCount();
							const auto* samplers = pGpuObj->getAllMutableCombinedSamplers();
							for (auto i=0u; !depsMissing && i<count; i++)
							if (samplers[i])
								depsMissing = missingDependent.template operator()<ICPUSampler>(samplers[i].get());
						}
						for (auto i=0u; !depsMissing && i<static_cast<uint32_t>(asset::IDescriptor::E_TYPE::ET_COUNT); i++)
						{
							const auto type = static_cast<asset::IDescriptor::E_TYPE>(i);
							const auto count = layout->getTotalDescriptorCount(type);
							auto* psDescriptors = pGpuObj->getAllDescriptors(type);
							if (!psDescriptors)
								continue;
							for (auto i=0u; !depsMissing && i<count; i++)
							{
								auto* untypedDesc = psDescriptors[i].get();
								if (untypedDesc)
								switch (asset::IDescriptor::GetTypeCategory(type))
								{
									case asset::IDescriptor::EC_BUFFER:
										depsMissing = missingDependent.template operator()<ICPUBuffer>(static_cast<const IGPUBuffer*>(untypedDesc));
										break;
									case asset::IDescriptor::EC_SAMPLER:
										depsMissing = missingDependent.template operator()<ICPUSampler>(static_cast<const IGPUSampler*>(untypedDesc));
										break;
									case asset::IDescriptor::EC_IMAGE:
										depsMissing = missingDependent.template operator()<ICPUImageView>(static_cast<const IGPUImageView*>(untypedDesc));
										break;
									case asset::IDescriptor::EC_BUFFER_VIEW:
										depsMissing = missingDependent.template operator()<ICPUBufferView>(static_cast<const IGPUBufferView*>(untypedDesc));
										break;
									case asset::IDescriptor::EC_ACCELERATION_STRUCTURE:
										depsMissing = missingDependent.template operator()<ICPUTopLevelAccelerationStructure>(static_cast<const IGPUTopLevelAccelerationStructure*>(untypedDesc));
										break;
									default:
										assert(false);
										depsMissing = true;
										break;
								}
							}
						}
					}
					if (depsMissing)
					{
						smart_refctd_ptr<const AssetType> dummy;
						// I know what I'm doing (breaking the promise of the `erase_if` to not mutate the inputs)
						markFailure("because conversion of a dependant failed!",&dummy,&item.second);
					}
					return depsMissing;
				}
			);
		};
		// Bottom up, only go over types we could actually break via missing upload/build (i.e. pipelines are unbreakable)
		// A built TLAS cannot be queried about the BLASes it contains, so just trust the pre-TLAS-build input validation did its job
		checkDependents.template operator()<ICPUBufferView>();
		checkDependents.template operator()<ICPUImageView>();
		checkDependents.template operator()<ICPUDescriptorSet>();
//		mergeCache.template operator()<ICPUFramebuffer>();
		// overwrite the compacted TLASes in Descriptor Sets
		if (auto& tlasRewriteSet=reservations.m_potentialTLASRewrites; !tlasRewriteSet.empty())
		{
			core::vector<IGPUDescriptorSet::SWriteDescriptorSet> writes;
			writes.reserve(tlasRewriteSet.size());
			core::vector<IGPUDescriptorSet::SDescriptorInfo> infos(tlasRewriteSet.size());
			auto* pInfo = infos.data();
			for (auto& entry : tlasRewriteSet)
			{
				auto* const dstSet = entry.dstSet;
				// we need to check if the descriptor set itself didn't get deleted in the meantime
				if (missingDependent.template operator()<ICPUDescriptorSet>(dstSet))
					continue;
				// rewtrieve the binding from the TLAS
				const auto* const tlas = static_cast<const IGPUTopLevelAccelerationStructure*>(dstSet->getAllDescriptors(IDescriptor::E_TYPE::ET_ACCELERATION_STRUCTURE)[entry.storageOffset.data].get());
				assert(tlas);
				// only rewrite if successfully compacted
				if (const auto foundCompacted=compactedTLASMap.find(tlas); foundCompacted!=compactedTLASMap.end())
				{
					pInfo->desc = foundCompacted->second;
					using redirect_t = IDescriptorSetLayoutBase::CBindingRedirect;
					const redirect_t& redirect = dstSet->getLayout()->getDescriptorRedirect(IDescriptor::E_TYPE::ET_ACCELERATION_STRUCTURE);
					const auto bindingRange = redirect.findBindingStorageIndex(entry.storageOffset);
					const auto firstElementOffset = redirect.getStorageOffset(bindingRange);
					writes.push_back({
						.dstSet = dstSet,
						.binding = redirect.getBinding(bindingRange).data,
						.arrayElement = entry.storageOffset.data-firstElementOffset.data,
						.count = 1,
						.info = pInfo++
					});
				}
			}
			// if the descriptor write fails, we make the Descriptor Sets behave as-if the TLAS build failed (dep is missing)
			if (!writes.empty() && !device->updateDescriptorSets(writes,{}))
				logger.log("Failed to write one of the compacted TLASes into a Descriptor Set, all Descriptor Sets will still use non-compacted TLASes",system::ILogger::ELL_ERROR);
		}
	}

	// insert items into cache if overflows handled fine and commandbuffers ready to be recorded
	core::for_each_in_tuple(reservations.m_stagingCaches,[&]<typename AssetType>(SReserveResult::staging_cache_t<AssetType>& stagingCache)->void
	{
		auto& cache = std::get<CCache<AssetType>>(m_caches);
		cache.m_forwardMap.reserve(cache.m_forwardMap.size()+stagingCache.size());
		cache.m_reverseMap.reserve(cache.m_reverseMap.size()+stagingCache.size());
		for (auto& item : stagingCache)
		if (item.second.gpuRef) // not wiped
		{
			// We have success now, but ask callback if we write to the new cache.
			if (!params.writeCache(item.second.cacheKey)) // TODO: let the user know the pointer to the GPU Object too?
				continue;
			asset_cached_t<AssetType> cached;
			cached.value = std::move(item.second.gpuRef);
			cache.m_reverseMap.emplace(item.first,item.second.cacheKey);
			cache.m_forwardMap.emplace(item.second.cacheKey,std::move(cached));
		}
		// provoke refcounting bugs ASAP
		stagingCache.clear();
	});

	// no submit was necessary, so should signal the extra semaphores from the host
	if (!retval.blocking())
	for (const auto& info : params.extraSignalSemaphores)
		info.semaphore->signal(info.value);
	retval.set(IQueue::RESULT::SUCCESS);
	return retval;
}

#if 0
	// Lots of extra work, is why we didn't want to pursue it:
	// - TLAS builds should happen semi-concurrently to BLAS, but need to know what TLAS needs what BLAS to finish (scheduling)
	//   + also device TLAS builds should know what Host Built BLAS they depend on, so that `pool.work()` is called until the BLAS's associated deferred op signals COMPLETE
	// - any AS should enqueue in a weird way with a sort of RLE, we allocate scratch until we can't then build whatever we can
	// - the list of outstanding BLAS and TLAS to build should get updated periodically
	// - overflow callbacks should call back into the BLAS and TLAS enqueuers and `pool.work()`
	struct ASBuilderPool
	{
		public:
			struct Worker
			{
				public:
					inline Worker(const ASBuilderPool* _pool) : pool(_pool), pushCount(0), executor(execute) {}
					inline ~Worker() {executor.join();}

					inline void push(smart_refctd_ptr<IDeferredOperation>&& task)
					{
						std::lock_guard(queueLock);
						tasks.push_back(std::move(task));
						pushCount.fetch_add(1);
						pushCount.notify_one();
					}

				private:
					inline void execute()
					{
						uint64_t oldTaskCount = 0;
						uint32_t taskIx = 0;
						while (pool->stop.test())
						{
							while (pushCount.load())
								pushCount.wait(oldTaskCount);
							size_t taskCount;
							IDeferredOperation* task;
							// grab the task under a lock so we're not in danger of vector reallocating
							{
								std::lock_guard(queueLock);
								taskCount = tasks.size();
								task = tasks[taskIx].get();
							}
							switch (task->execute())
							{
								case IDeferredOperation::STATUS::THREAD_IDLE:
									taskIx++; // next task
									break;
								default:
								{
									std::lock_guard(queueLock);
									tasks.erase(tasks.begin()+taskIx);
									break;
								}
							}
							if (taskIx>=taskCount)
								taskIx = 0;
						}
					}

					std::mutex queueLock;
					const ASBuilderPool* pool;
					std::atomic_uint64_t pushCount;
					std::thread executor;
					core::vector<smart_refctd_ptr<IDeferredOperation>> tasks;
			};

			inline ASBuilderPool(const uint16_t _workerCount, system::logger_opt_ptr _logger) : stop(), workerCount(_workerCount), nextWorkerPush(0), logger(_logger)
			{
				workers = std::make_unique<Worker[]>(workerCount);
			}
			inline ~ASBuilderPool()
			{
				finish();
			}

			inline void finish()
			{
				while (work()) {}
				stop.test_and_set();
				stop.notify_one();
				workers = nullptr;
			}

			struct Build
			{
				smart_refctd_ptr<IDeferredOperation> op;
				// WRONG: for every deferred op, there are multiple `gpuObj` and `hash` that get built by it
				IGPUAccelerationStructure* gpuObj;
				core::blake3_hash_t* hash;
			};
			inline void push(Build&& build)
			{
				auto op = build.op.get();
				if (!op->isPending())
				{
					logger.log("Host Acceleration Structure failed for \"%s\"",system::ILogger::ELL_ERROR,build.gpuObj->getObjectDebugName());
					// change the content hash on the reverse map to a NoContentHash
					*build.hash = CHashCache::NoContentHash;
					return;
				}
				// there's no true best way to pick the worker with least work
				for (uint16_t i=0; i<min<uint16_t>(op->getMaxConcurrency()-1,workerCount); i++)
					workers[(nextWorkerPush++)%workerCount].push(smart_refctd_ptr<IDeferredOperation>(op));
				buildsInProgress.push_back(std::move(build));
			}

			inline bool empty() const {return buildsInProgress.empty();}

			// The idea is to somehow get the overflow callbacks to call this
			inline bool work()
			{
				if (empty())
					return;
				auto build = buildsInProgress.begin()+buildIx;
				switch (build->op->execute())
				{
					case IDeferredOperation::STATUS::THREAD_IDLE:
						buildIx++; // next task
						break;
					case IDeferredOperation::STATUS::_ERROR:
						logger.log("Host Acceleration Structure failed for \"%s\"",system::ILogger::ELL_ERROR,build->gpuObj->getObjectDebugName());
						// change the content hash on the reverse map to a NoContentHash
						*build->hash = CHashCache::NoContentHash;
						[[fallthrough]];
					default:
					{
						buildsInProgress.erase(build);
						break;
					}
				}
				if (buildIx>=buildsInProgress.size())
					buildIx = 0;
				return buildsInProgress.empty();
			}

			std::atomic_flag stop;

		private:
			uint16_t workerCount;
			uint16_t nextWorkerPush = 0;
			system::logger_opt_ptr logger;
			std::unique_ptr<Worker[]> workers;
			core::vector<Build> buildsInProgress;
			uint32_t buildIx = 0;
	};
	ASBuilderPool hostBuilders(params.extraHostASBuildThreads,logger);

	// crappy pseudocode
	auto hostBLASConvIt = reservations.m_blasConversions[1].begin();
	auto hostBLASConvEnd = reservations.m_blasConversions[1].end();
	while (hostBLASConvIt!=hostBLASConvEnd)
	{
		auto op = device->createDeferredOperation();
		if (!op)
			error, mark failure in staging;
		core::vector<IGPUBottomLevelAccelerationStructure::HostBuildInfo> infos;
		core::vector<IGPUBottomLevelAccelerationStructure::BuildRangeInfo> ranges;
		for (; hostBLASConvIt!=hostBLASConvEnd; hostBLASConvIt++)
		{
			void* scratch = hostBLASConvIt->scratchSize;
			if (!scratch)
			{
				if (infos.empty() && hostBuilders.empty())
					error mark failure in staging, can't even enqueue 1 build';
				else
					break;
			}

			auto asset = hostBLASConvIt->canonical;
			asset->getGeometryPrimitiveCounts();
			ranges.push_back({
				.primitiveCount = 0,
				.primitiveByteOffset = 0,
				.firstVertex = 0,
				.transformByteOffset = 0
			});
		}
		if (!device->buildAccelerationStructures(op.get(),infos,ranges.data()))
			continue;
	}
#endif
}
}