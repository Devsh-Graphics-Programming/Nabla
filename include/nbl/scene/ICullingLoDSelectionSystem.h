// Copyright (C) 2019 - DevSH Graphics Programming Sp. z O.O.
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors

#ifndef __NBL_SCENE_I_CULLING_LOD_SELECTION_SYSTEM_H_INCLUDED__
#define __NBL_SCENE_I_CULLING_LOD_SELECTION_SYSTEM_H_INCLUDED__

#include "nbl/scene/ITransformTree.h"
#include "nbl/scene/ILevelOfDetailLibrary.h"

namespace nbl::scene
{

// register many <renderpass_t,node_t,lod_table_t> to be rendered
// TODO: Make LoDTable entries contain pointers to bindpose matrices and joint AABBs (culling of skinned models can only occur after LoD choice)
// SPECIAL: when registering skeletons, allocate the registrations contiguously to get a free translation table for skinning
// but reserve a per_view_data_t=<MVP,chosen_lod_t> for the output
// keep in `sparse_vector` to make contiguous
class ICullingLoDSelectionSystem : public virtual core::IReferenceCounted
{
	public:
		#define nbl_glsl_DispatchIndirectCommand_t asset::DispatchIndirectCommand_t
		#include "nbl/builtin/glsl/culling_lod_selection/dispatch_indirect_params.glsl"
		#undef nbl_glsl_DispatchIndirectCommand_t
		struct DispatchIndirectParams : nbl_glsl_culling_lod_selection_dispatch_indirect_params_t
		{
		};

		//
		static core::smart_refctd_ptr<video::IGPUBuffer> createDispatchIndirectBuffer(video::ILogicalDevice* logicalDevice)
		{
            video::IGPUBuffer::SCreationParams params;
            params.usage = core::bitflag(asset::IBuffer::EUF_STORAGE_BUFFER_BIT)|asset::IBuffer::EUF_INDIRECT_BUFFER_BIT;
            return logicalDevice->createDeviceLocalGPUBufferOnDedMem(params,sizeof(DispatchIndirectParams));
		}

		// Instance Redirect buffer holds a `uvec2` of `{instanceGUID,perViewPerInstanceDataID}` for each instace of a drawcall
		static core::smart_refctd_ptr<video::IGPUBuffer> createInstanceRedirectBuffer(video::ILogicalDevice* logicalDevice, const uint32_t maxTotalDrawcallInstances)
		{
            video::IGPUBuffer::SCreationParams params;
            params.usage = core::bitflag(asset::IBuffer::EUF_STORAGE_BUFFER_BIT)|asset::IBuffer::EUF_VERTEX_BUFFER_BIT;
            return logicalDevice->createDeviceLocalGPUBufferOnDedMem(params,sizeof(uint32_t)*2u*maxTotalDrawcallInstances);
		}

		// lod_table
		// lod_info
		// drawcall AABB SSBO

		// dispatch indirect params [input]
		// instance list [input]
		// lod drawcall offsets [working mem]
		// lod drawcall counts [working mem]
		// prefix sum scratch memory

		// drawcall pool [inout]
		// transient per view data [output]
		// transient per instance attribute [output]
		// draw count pool [inout]

		// custom DS
		struct Params
		{
			video::IGPUCommandBuffer* cmdbuf; // must already be in recording state
			const video::IGPUBuffer* indirectDispatchParams;
			const video::IGPUDescriptorSet* lodTableAndDrawcallDS;
			const video::IGPUDescriptorSet* transientInputDS;
			const video::IGPUDescriptorSet* transientOutputDS;
			const video::IGPUDescriptorSet* customDS;
			uint32_t directInstanceCount; // set as 0u for indirect dispatch
		};
		void processInstancesAndFillIndirectDraws(const Params& params)
		{
			auto cmdbuf = params.cmdbuf;
			const auto indirectDispatchParams = params.indirectDispatchParams;
#if 0
			if (params.directInstanceCount)
			{
				cmdbuf->bindComputePipeline(directInstanceCullAndLoDSelect);
				cmdbuf->bindDescriptorSets(EPBP_COMPUTE,directInstanceCullAndLoDSelectLayout.get(),0u,4u,&params.lodTableAndDrawcallDS);
				//cmdbuf->dispatch(,1u,1u);
			}
			else
			{
				cmdbuf->bindComputePipeline(indirectInstanceCullAndLoDSelect);
				cmdbuf->bindDescriptorSets(EPBP_COMPUTE,indirectInstanceCullAndLoDSelectLayout.get(),0u,4u,&params.lodTableAndDrawcallDS);
				cmdbuf->dispatchIndirect(indirectDispatchParams,offsetof(DispatchIndirectParams,instanceCullAndLoDSelect));
			}
			//barrier(compute->compute);
			cmdbuf->bindComputePipeline(instanceDrawCull);
			cmdbuf->bindDescriptorSets(EPBP_COMPUTE,sharedPipelineLayout.get(),0u,4u,&params.lodTableAndDrawcallDS);
			cmdbuf->dispatchIndirect(indirectDispatchParams,offsetof(DispatchIndirectParams,instanceDrawCull));
			//barrier(compute->compute);
			cmdbuf->bindComputePipeline(drawInstanceCountPrefixSum);
			cmdbuf->dispatchIndirect(indirectDispatchParams,offsetof(DispatchIndirectParams,drawInstanceCountPrefixSum));
			//barrier(compute -> compute);
#endif
			cmdbuf->bindComputePipeline(instanceRefCountingSortScatter.get());
			cmdbuf->dispatchIndirect(indirectDispatchParams,offsetof(DispatchIndirectParams,instanceRefCountingSortScatter));
			/*
			if (features.drawIndirectCount)
			{
				cmdbuf->bindComputePipeline(drawCompact);
				cmdbuf->dispatchIndirect(indirectDispatchParams,offsetof(DispatchIndirectParams,drawCompact));
			}
			barrier(compute -> compute|indirect|vertex_input);
			*/
		}
	protected:
		core::smart_refctd_ptr<video::IGPUPipelineLayout> directInstanceCullAndLoDSelectLayout,indirectInstanceCullAndLoDSelectLayout,sharedPipelineLayout;
		core::smart_refctd_ptr<video::IGPUComputePipeline> directInstanceCullAndLoDSelect,indirectInstanceCullAndLoDSelect,instanceDrawCull,drawInstanceCountPrefixSum,instanceRefCountingSortScatter,drawCompact;
};


} // end namespace nbl::scene

#endif

