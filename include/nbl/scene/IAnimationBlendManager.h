// Copyright (C) 2019 - DevSH Graphics Programming Sp. z O.O.
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors

#ifndef _NBL_SCENE_I_ANIMATION_BLEND_MANAGER_H_INCLUDED_
#define _NBL_SCENE_I_ANIMATION_BLEND_MANAGER_H_INCLUDED_

#include "nbl/core/declarations.h"

#include "nbl/video/declarations.h"

#include "nbl/scene/ITransformTreeManager.h"

namespace nbl::scene
{

/**
* Time to think about animation library.
* 
* Models reference bones using per-vertex (uint8 usually) attributes, which means a single vertex is influenced by max 4 bones.
* Also to keep data compact bones are usually referenced locally (to allow 8bit indices in the vertices).
* A glTF skin helps to keep the bone/joint ID values down by providing a "remapping" table in the "skin" (`sceneNodeTransform[skinJoints[vertexJointID]]`)
* In order to allow for instancing of skinned meshes, we must implement a translation table as well.
* 
* It should be considered whether to pack instance data into a single SSBO as a range of `uvec4` OR whether to make the instance data contain a pointer/reference to a joint translation table. 
* 
* 
* Each Node should keep a linked list (head and tail for easy iteration and addition) of animation blends that affect it as well as blend update frequencies, via auxilary properties.
* Blends should be user removable (by the CPU) and auto removable (by the GPU).
* In the case of user removal a complex shader with forward progressing locks needs to be launched.
* GPU Autoremove is trivial as there's no data races to prevent because the blend would be removed during regular list traversal
* However the list of autoremoved blends would need to be maintained and periodically downloaded to free the blends in the CPU-side pool allocator
* We can leave GPU autoremove of finished blends as a TODO for much later
* 
* The way that this system is supposed to work is that the ITransformTreeManager is supposed to construct an Indirect Dispatch + compact Node List for the next frame's animation updates.
* The reason for this weird feedback loop and one-frame delay is because there's no sane way to keep contiguous lists of nodes bucketed by their update frequencies.
* This means a TODO of passing three optional node update frequency [readonly], animation blend dispatch indirect[coherent readwrite] and node output list [writeonly] buffers via `ITreeTransformManager::GlobalTransformUpdateParams`
* 
* Also this allows us to cull the animation updates to only the nodes whose global transforms we want to know.
* 
* This means that when you `IAnimationBlendManager::computeBlends`, you feed it an indirect dispatch buffer and a compact node list (with first uint denoting the count).
* The animation blends should be written directly to the relative transforms of the nodes and their timestamps modified (so you need acess to some of the ITransformTree's property pool descriptor).
* They shouldn't write directly to global transforms (even if node has no parent) because that would be overwritten if the recompute timestamp isnt written,
* but if the recompute timestamp is written, subsequent relative transform modifications won't be picked up in a later recomputation of global transforms.
**/

// TODO: move out to separate header
class IAnimationBlendSystem : public virtual core::IReferenceCounted
{
	public:
		using blend_id_t = uint32_t;

		// TODO: move to GLSL header and include it
		struct nbl_glsl_animation_blend_t
		{
			blend_id_t next;
			// one less indirection is good
			video::IGPUAnimationLibrary::keyframe_t keyframeOffset_type;
			video::IGPUAnimationLibrary::timestamp_t timestampBeginOffset;
			video::IGPUAnimationLibrary::timestamp_t timestampEndOffset;
			uint16_t weight;
			// mutable state
			// we will support programmble (non constant) weights in the far future
			uint16_t animationTimestampsPerSystemTimestamp; // how many uints to advance for every uint ticked (speed), if NaN then dont perform the blend (dont evaluate)
			video::IGPUAnimationLibrary::timestamp_t currentAnimationFrame; // might need a supporting fraction to allow interpolation (really slow speeds)
			uint32_t keyframeBinarySearchHint; // does it help, is it needed?
		};
		struct Blend : nbl_glsl_animation_blend_t
		{
			enum E_TYPE : uint32_t
			{
				// will run once (TODO: and self-delete)
				ET_FINISH=0,
				// will run over and over
				ET_LOOP=1,
				ET_PING_PONG_FORWARD=2,
				ET_PING_PONG_BACKWARD=3,
				ET_COUNT
			};
			
			inline video::IGPUAnimationLibrary::keyframe_t getKeyframeOffset() const {return core::bitfieldExtract(keyframeOffset_type,0,32-TypeBits());}
			inline E_TYPE getType() const {return static_cast<E_TYPE>(core::bitfieldExtract(keyframeOffset_type,32-TypeBits(),TypeBits()));}
		private:
			static inline int32_t TypeBits() {return hlsl::findMSB(ET_COUNT);}
		};

		// creation
        static inline core::smart_refctd_ptr<IAnimationBlendManager> create(core::smart_refctd_ptr<ITransformTree>&& tree, core::smart_refctd_ptr<video::IGPUAnimationLibrary>&& _animationLibrary)
        {
			if (true) // TODO: some checks and validation before creating?
				return nullptr;

			// TODO: allocate node blend update frequency bufferview (R8_UINT), and initialize to 120 FPS

			// TODO: allocate per-node list tail and head buffer and fill them with 0xfffffffu (invalid value) [could call `clearBlends`]

			auto* abm = new IAnimationBlendSystem(std::move(tree),std::move(_animationLibrary));
            return core::smart_refctd_ptr<IAnimationBlendSystem>(abm,core::dont_grab);
        }

		// TODO: might need to move these to the `IAnimationBlendManager` 
		// TODO: do as a param struct, allow for animation speed to be sourced from a GPU bufffer as well
		void startBlends(const blend_id_t* begin, const blend_id_t* end, const float* animationTimestampsPerSystemTimestamp)
		{
			// easy enough, just set the `animationTimestampsPerSystemTimestamp` to whatever it was supposed to be
		}
		// remove from a contiguous list in GPU memory
		void pauseBlends(const blend_id_t* begin, const blend_id_t* end)
		{
			// easy enough, just set the `animationTimestampsPerSystemTimestamp` to NaN
		}

		// TODO: do as a param struct, allow for the frame seeked timestamp to be sourced from a GPU bufffer as well
		void seekBlends(const blend_id_t* begin, const blend_id_t* end, const video::IGPUAnimationLibrary::timestamp_t* frame)
		{
			// easy enough, just set the `currentAnimationFrame` to `frame
		}

		// TODO: clearBlends (remove all/everything), setNodeUpdateFrequency

	protected:
		IAnimationBlendSystem(core::smart_refctd_ptr<ITransformTree>&& tree, core::smart_refctd_ptr<video::IGPUAnimationLibrary>&& _animationLibrary) : m_tree(std::move(tree)), m_animationLibrary(std::move(_animationLibrary))
		{
		}
		~IAnimationBlendSystem()
		{
			// everything drops itself automatically
		}

		core::smart_refctd_ptr<ITransformTree> m_tree;
		core::smart_refctd_ptr<video::IGPUAnimationLibrary> m_animationLibrary;
		core::smart_refctd_ptr<video::IGPUBufferView> m_nodeBlendUpdateFrequencies;
		core::smart_refctd_ptr<video::IGPUBuffer> m_nodeBlendList;
		// TODO: PRoperty Pool for the blends
		// TODO: Descriptor sets for the different pipelines
		//core::smart_refctd_ptr<video::IGPUDescriptorSet> m_animationDS; // animation library (keyframes + animations) + registered blends + active blend IDs
		// ? core::smart_refctd_ptr<video::IGPUBuffer> m_dispatchIndirectCommandBuffer;
};

class IAnimationBlendManager : public virtual core::IReferenceCounted
{
	public:
		// creation
        static inline core::smart_refctd_ptr<IAnimationBlendManager> create(core::smart_refctd_ptr<video::ILogicalDevice>&& _device)
        {
			if (true) // TODO: some checks and validation before creating?
				return nullptr;

			auto* abm = new IAnimationBlendManager(std:::move(_device));
            return core::smart_refctd_ptr<IAnimationBlendManager>(abm,core::dont_grab);
        }

		struct ParamsBase
		{
			// TODO: probably cmdbuf, fence, etc.
			const ITransformTree::node_t* nodes;
			uint32_t count;
		};
		struct AddBlendsParams : ParamsBase
		{
			const IAnimationBlendSystem::blend_id_t* outBlends;
			// required, from this we will initialize the three members of `nbl_glsl_animation_blend_t`
			const video::IGPUAnimationLibrary::animation_t* animations;
			// if null, intialize to 1.f
			const uint16_t* animationTimestampsPerSystemTimestamp = nullptr;
			// if null, intialize to the first timestamp read from `timestamps[anims.data[animations[i]].timestampOffset]`
			const video::IGPUAnimationLibrary::timestamp_t* currentAnimationFrame = nullptr;
		};
		// Each blend is added to a particular node, if you have multiple `node` references we cannot guarantee the order they will be added to the per-node linked list
		void addBlends(const AddBlendsParams& params)
		{
			// launch a `CPropertyPoolHandler`-esque shader to initialize the propertries of a blend properly and add it to a node's linked list
		}

		struct RemoveBlendsParams : ParamsBase
		{
			const IAnimationBlendSystem::blend_id_t* blends;
		};
		//
		void removeBlends(const RemoveBlendsParams& params)
		{
			// launch a compute shader that mutexes on a node but can always make forward progress, to remove the blends from the linked list 
		}

		// removes all blends (resets linked list) for the given nodes
		void removeAllBlends(const ParamsBase& params)
		{
			// launch a compute shader that initializes the head and tail to default values for the nodes given in the list 
		}

		// TODO: periodically we need to download the list of blends which have self-removed on the GPU, in order to free them from the CPU Pool Allocator 
#if 0 // out of date API idea
		//
		struct Params
		{
			video::IGPUCommandBuffer* cmdbuf; // must already be in recording state
			// for signalling when to drop a temporary descriptor set
			video::IGPUFence* fence;
			IAnimationBlendSystem* system;
			union
			{
				struct
				{
					video::IGPUBuffer* buffer;
					uint64_t offset;
				} dispatchIndirect;
				struct
				{
				private:
					uint64_t dummy;
				public:
					uint32_t nodeCount;
				} dispatchDirect;
			};
			struct BarrierParams
			{
				// TODO: what to set queue family indices to if we don't plan on a transfer by default?
				uint32_t srcQueueFamilyIndex;
				uint32_t dstQueueFamilyIndex;
				asset::E_PIPELINE_STAGE_FLAGS dstStages = asset::PIPELINE_STAGE_FLAGS::ALL_COMMANDS_BIT;
				asset::E_ACCESS_FLAGS dstAccessMask = asset::EAF_ALL_ACCESSES_BIT_DEVSH;
			} finalBarrier = {};
			asset::SBufferBinding<video::IGPUBuffer> outRelativeTFormUpdateIndirectParameters;
			// first uint in the buffer tells us how many ModificationRequestRanges we have
			// second uint in the buffer tells us how many total requests we have
			// rest is filled wtih ModificationRequestRange
			asset::SBufferBinding<video::IGPUBuffer> outRequestRanges;
			// this one is filled with RelativeTransformModificationRequest
			asset::SBufferBinding<video::IGPUBuffer> outModificationRequests;
		};
		void computeBlends(ITransformTree::timestamp_t newTimestamp, const Params& params)
		{
			// TODO: Do what ITransformTreeManager and CPropertyPoolHandler do
		}
#endif
	protected:
		IAnimationBlendManager(core::smart_refctd_ptr<video::ILogicalDevice>&& _device) : m_device(std::move(_device))
		{
		}
		~IAnimationBlendManager()
		{
			// everything drops itself automatically
		}

		core::smart_refctd_ptr<video::ILogicalDevice> m_device;
		core::smart_refctd_ptr<video::IGPUComputePipeline> m_addBlendsPipeline,m_removeBlendsPipeline,m_computeBlendsPipeline;
};



} // end namespace nbl::scene

#endif

