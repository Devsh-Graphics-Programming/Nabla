// Copyright (C) 2019 - DevSH Graphics Programming Sp. z O.O.
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors

#ifndef _NBL_SCENE_I_TREE_TRANSFORM_MANAGER_H_INCLUDED_
#define _NBL_SCENE_I_TREE_TRANSFORM_MANAGER_H_INCLUDED_

#include "nbl/core/declarations.h"
#include "nbl/video/declarations.h"

#include "nbl/core/definitions.h"

#include "nbl/scene/ITransformTree.h"

namespace nbl::scene
{

// TODO: rewrite
#if 0
#define uint uint32_t
#define int int32_t
#define uvec4 core::vectorSIMDu32
#include "nbl/builtin/glsl/transform_tree/relative_transform_modification.glsl"
#include "nbl/builtin/glsl/transform_tree/modification_request_range.glsl"
#undef uvec4
#undef int
#undef uint

class ITransformTreeManager : public virtual core::IReferenceCounted
{
		template<uint32_t BindingCount>
		static inline auto createDescriptorSetLayout(video::ILogicalDevice* device)
		{
			video::IGPUDescriptorSetLayout::SBinding bnd[BindingCount];
			bnd[0].binding = 0u;
			bnd[0].count = 1u;
			bnd[0].type = asset::IDescriptor::E_TYPE::ET_STORAGE_BUFFER;
			bnd[0].stageFlags = asset::IShader::ESS_COMPUTE;
			bnd[0].samplers = nullptr;
			for (auto i = 1u; i < BindingCount; i++)
			{
				bnd[i] = bnd[i - 1u];
				bnd[i].binding = i;
			}
			return device->createDescriptorSetLayout(bnd, bnd + BindingCount);
		}
		template<uint32_t BindingCount>
		static inline void updateDescriptorSet(
			video::ILogicalDevice* device, video::IGPUDescriptorSet* set,
			std::array<asset::SBufferBinding<video::IGPUBuffer>, BindingCount>&& bufferBindings
		)
		{
			video::IGPUDescriptorSet::SWriteDescriptorSet writes[BindingCount];
			video::IGPUDescriptorSet::SDescriptorInfo infos[BindingCount];
			for (auto i = 0u; i < BindingCount; i++)
			{
				infos[i].desc = std::move(bufferBindings[i].buffer);
				infos[i].info.buffer.offset = bufferBindings[i].offset;
				infos[i].info.buffer.size = asset::SBufferRange<video::IGPUBuffer>::WholeBuffer;
				writes[i].dstSet = set;
				writes[i].binding = i;
				writes[i].arrayElement = 0u;
				writes[i].count = 1u;
				writes[i].descriptorType = asset::IDescriptor::E_TYPE::ET_STORAGE_BUFFER;
				writes[i].info = infos+i;
			}
			device->updateDescriptorSets(BindingCount, writes, 0u, nullptr);
		}
	public:
		struct RelativeTransformModificationRequest : nbl_glsl_transform_tree_relative_transform_modification_t
		{
			public:
				enum E_TYPE : uint32_t
				{
					ET_OVERWRITE=_NBL_BUILTIN_TRANSFORM_TREE_RELATIVE_TRANSFORM_MODIFICATION_T_E_TYPE_OVERWRITE_, // exchange the value, `This(vertex)`
					ET_CONCATENATE_AFTER=_NBL_BUILTIN_TRANSFORM_TREE_RELATIVE_TRANSFORM_MODIFICATION_T_E_TYPE_CONCATENATE_AFTER_, // apply transform after, `This(Previous(vertex))`
					ET_CONCATENATE_BEFORE=_NBL_BUILTIN_TRANSFORM_TREE_RELATIVE_TRANSFORM_MODIFICATION_T_E_TYPE_CONCATENATE_BEFORE_, // apply transform before, `Previous(This(vertex))`
					ET_WEIGHTED_ACCUMULATE=_NBL_BUILTIN_TRANSFORM_TREE_RELATIVE_TRANSFORM_MODIFICATION_T_E_TYPE_WEIGHTED_ACCUMULATE_, // add to existing value, `(Previous+This)(vertex)`
					ET_COUNT=_NBL_BUILTIN_TRANSFORM_TREE_RELATIVE_TRANSFORM_MODIFICATION_T_E_TYPE_COUNT_
				};
				RelativeTransformModificationRequest() = default;
				RelativeTransformModificationRequest(const E_TYPE type, const core::matrix3x4SIMD& _preweightedModification)
				{
					constexpr uint32_t log2ET_COUNT = 2u;
					static_assert(ET_COUNT<=(0x1u<<log2ET_COUNT),"Need to rewrite the type encoding routine!");
				
					//
					*reinterpret_cast<core::matrix3x4SIMD*>(data) = _preweightedModification;

					// stuff the bits into x and z components of scale (without a rotation) 
					// clear then bitwise-or
					data[0][0] &= 0xfffffffeu;
					data[0][0] |= type&0x1u;
					data[2][2] &= 0xfffffffeu;
					data[2][2] |= (type>>1u)&0x1u;
				}
				RelativeTransformModificationRequest(const E_TYPE type, const core::matrix3x4SIMD& _modification, const float weight) : RelativeTransformModificationRequest(type,_modification*weight) {}

				inline E_TYPE getType() const
				{
					return static_cast<E_TYPE>(nbl_glsl_transform_tree_relative_transform_modification_t_getType(*this));
				}
		};

		// creation
		static inline core::smart_refctd_ptr<ITransformTreeManager> create(video::IUtilities* utils, video::IQueue* uploadQueue)
		{
			auto device = utils->getLogicalDevice();
			auto system = device->getPhysicalDevice()->getSystem();
			auto createShader = [&system,&device]<core::StringLiteral Path>(asset::IShader::E_SHADER_STAGE type=asset::IShader::ESS_COMPUTE) -> core::smart_refctd_ptr<video::IGPUSpecializedShader>
			{
				auto loadBuiltinData = [&](const std::string _path) -> core::smart_refctd_ptr<const nbl::system::IFile>
				{
					nbl::system::ISystem::future_t<core::smart_refctd_ptr<nbl::system::IFile>> future;
					system->createFile(future, system::path(_path), core::bitflag(nbl::system::IFileBase::ECF_READ) | nbl::system::IFileBase::ECF_MAPPABLE);
					if (future.wait())
						return future.copy();
					return nullptr;
				};

				auto glslFile = loadBuiltinData(Path.value);
				core::smart_refctd_ptr<asset::ICPUBuffer> glsl;
				{
					glsl = core::make_smart_refctd_ptr<asset::ICPUBuffer>(glslFile->getSize());
					memcpy(glsl->getPointer(), glslFile->getMappedPointer(), glsl->getSize());
				}
				auto shader = device->createShader(core::make_smart_refctd_ptr<asset::ICPUShader>(core::smart_refctd_ptr(glsl), type, asset::IShader::E_CONTENT_TYPE::ECT_GLSL, "????"));
				return device->createSpecializedShader(shader.get(),{nullptr,nullptr,"main"});
			};

			auto updateRelativeSpec = createShader.operator()<core::StringLiteral("nbl/builtin/glsl/transform_tree/relative_transform_update.comp")>();
			auto recomputeGlobalSpec = createShader.operator()<core::StringLiteral("nbl/builtin/glsl/transform_tree/global_transform_update.comp")>();
			auto recomputeGlobalAndNormalSpec = createShader.operator()<core::StringLiteral("nbl/builtin/glsl/transform_tree/global_transform_and_normal_matrix_update.comp")>();
			auto debugDrawVertexSpec = createShader.operator()<core::StringLiteral("nbl/builtin/glsl/transform_tree/debug.vert")>(asset::IShader::ESS_VERTEX);
			auto debugDrawFragmentSpec = createShader.operator()<core::StringLiteral("nbl/builtin/material/debug/vertex_normal/specialized_shader.frag")>(asset::IShader::ESS_FRAGMENT);
			if (!updateRelativeSpec || !recomputeGlobalSpec || !recomputeGlobalAndNormalSpec || !debugDrawVertexSpec || !debugDrawFragmentSpec)
				return nullptr;

			const auto& limits = device->getPhysicalDevice()->getLimits();
			core::vector<uint8_t> tmp(getDefaultValueBufferOffset(limits,~0u));
			uint8_t* fillData = tmp.data();
			{
				*reinterpret_cast<ITransformTree::parent_t*>(fillData+getDefaultValueBufferOffset(limits,ITransformTree::parent_prop_ix)) = ITransformTree::invalid_node;
				*reinterpret_cast<ITransformTree::relative_transform_t*>(fillData+getDefaultValueBufferOffset(limits,ITransformTree::relative_transform_prop_ix)) = core::matrix3x4SIMD();
				*reinterpret_cast<ITransformTree::modified_stamp_t*>(fillData+getDefaultValueBufferOffset(limits,ITransformTree::modified_stamp_prop_ix)) = ITransformTree::initial_modified_timestamp;
				*reinterpret_cast<ITransformTree::recomputed_stamp_t*>(fillData+getDefaultValueBufferOffset(limits,ITransformTree::recomputed_stamp_prop_ix)) = ITransformTree::initial_recomputed_timestamp;
			}			
			
			video::IGPUBuffer::SCreationParams defaultFillValuesBufferCreationParams = {};
			defaultFillValuesBufferCreationParams.size = tmp.size();
			defaultFillValuesBufferCreationParams.usage = core::bitflag<video::IGPUBuffer::E_USAGE_FLAGS>(video::IGPUBuffer::E_USAGE_FLAGS::EUF_TRANSFER_DST_BIT);
			auto defaultFillValues = utils->createFilledDeviceLocalBufferOnDedMem(uploadQueue,std::move(defaultFillValuesBufferCreationParams),fillData);
			defaultFillValues->setObjectDebugName("ITransformTreeManager::m_defaultFillValues");
			tmp.resize(sizeof(uint16_t)*DebugIndexCount);
			uint16_t* debugIndices = reinterpret_cast<uint16_t*>(tmp.data());
			{
				std::fill_n(debugIndices,24u,0u);
				debugIndices[0] = 0b000;
				debugIndices[1] = 0b001;
				debugIndices[2] = 0b001;
				debugIndices[3] = 0b011;
				debugIndices[4] = 0b011;
				debugIndices[5] = 0b010;
				debugIndices[6] = 0b010;
				debugIndices[7] = 0b000;
				debugIndices[8] = 0b000;
				debugIndices[9] = 0b100;
				debugIndices[10] = 0b001;
				debugIndices[11] = 0b101;
				debugIndices[12] = 0b010;
				debugIndices[13] = 0b110;
				debugIndices[14] = 0b011;
				debugIndices[15] = 0b111;
				debugIndices[16] = 0b100;
				debugIndices[17] = 0b101;
				debugIndices[18] = 0b101;
				debugIndices[19] = 0b111;
				debugIndices[20] = 0b100;
				debugIndices[21] = 0b110;
				debugIndices[22] = 0b110;
				debugIndices[23] = 0b111;
				debugIndices[24] = 8u;
				debugIndices[25] = 9u;
			}
			
			video::IGPUBuffer::SCreationParams debugIndexBufferCreationParams = {};
			debugIndexBufferCreationParams.size = tmp.size();
			debugIndexBufferCreationParams.usage = core::bitflag(video::IGPUBuffer::E_USAGE_FLAGS::EUF_TRANSFER_DST_BIT) | video::IGPUBuffer::E_USAGE_FLAGS::EUF_INDEX_BUFFER_BIT;
			auto debugIndexBuffer = utils->createFilledDeviceLocalBufferOnDedMem(uploadQueue,std::move(debugIndexBufferCreationParams),fillData);

			auto updateLocalDsLayout = createUpdateLocalTransformsDescriptorSetLayout(device);
			auto recomputeGlobalDsLayout = createRecomputeGlobalTransformsDescriptorSetLayout(device);
			auto debugDrawDsLayout = createDebugDrawDescriptorSetLayout(device);

			auto pipelinesWithoutNormalMatrices = createPipelines<ITransformTreeWithoutNormalMatrices>(
				device,updateRelativeSpec,recomputeGlobalSpec,
				debugDrawVertexSpec,debugDrawFragmentSpec,
				updateLocalDsLayout,recomputeGlobalDsLayout,debugDrawDsLayout
			);
			auto pipelinesWithNormalMatrices = createPipelines<ITransformTreeWithNormalMatrices>(
				device,updateRelativeSpec,recomputeGlobalAndNormalSpec,
				debugDrawVertexSpec,debugDrawFragmentSpec,
				updateLocalDsLayout,recomputeGlobalDsLayout,debugDrawDsLayout
			);
			if (!pipelinesWithoutNormalMatrices.isValid() || !pipelinesWithNormalMatrices.isValid())
				return nullptr;

			auto* ttm = new ITransformTreeManager(
				core::smart_refctd_ptr<video::ILogicalDevice>(device),
				std::move(pipelinesWithoutNormalMatrices),std::move(pipelinesWithNormalMatrices),
				std::move(defaultFillValues),std::move(debugIndexBuffer)
			);
			return core::smart_refctd_ptr<ITransformTreeManager>(ttm,core::dont_grab);
		}

		static inline constexpr uint32_t TransferCount = 4u;
		struct RequestBase
		{
			ITransformTree* tree;
		};
		//
		struct TransferRequest : RequestBase
		{
			asset::SBufferRange<video::IGPUBuffer> nodes = {};
			// if not present we set these properties to defaults (no parent and identity transform)
			asset::SBufferBinding<video::IGPUBuffer> parents = {};
			asset::SBufferBinding<video::IGPUBuffer> relativeTransforms = {};
		};
		inline bool setupTransfers(const TransferRequest& request, video::CPropertyPoolHandler::TransferRequest* transfers)
		{
			if (!request.tree)
				return false;

			const auto* pool = request.tree->getNodePropertyPool();

			for (auto i=0u; i<TransferCount; i++)
			{
				transfers[i].elementCount = request.nodes.size/sizeof(ITransformTree::node_t);
				transfers[i].srcAddressesOffset = video::IPropertyPool::invalid; // iota input indices
				transfers[i].dstAddressesOffset = request.nodes.offset;
			}
			transfers[0].setFromPool(pool,ITransformTree::parent_prop_ix);
			transfers[0].flags = request.parents.buffer ? video::CPropertyPoolHandler::TransferRequest::EF_NONE:video::CPropertyPoolHandler::TransferRequest::EF_FILL;
			transfers[0].buffer = request.parents.buffer ? request.parents:getDefaultValueBufferBinding(ITransformTree::parent_prop_ix);
			transfers[1].setFromPool(pool,ITransformTree::relative_transform_prop_ix);
			transfers[1].flags = request.relativeTransforms.buffer ? video::CPropertyPoolHandler::TransferRequest::EF_NONE:video::CPropertyPoolHandler::TransferRequest::EF_FILL;
			transfers[1].buffer = request.relativeTransforms.buffer ? request.relativeTransforms:getDefaultValueBufferBinding(ITransformTree::relative_transform_prop_ix);
			transfers[2].setFromPool(pool,ITransformTree::modified_stamp_prop_ix);
			transfers[2].flags = video::CPropertyPoolHandler::TransferRequest::EF_FILL;
			transfers[2].buffer = getDefaultValueBufferBinding(ITransformTree::modified_stamp_prop_ix);
			transfers[3].setFromPool(pool,ITransformTree::recomputed_stamp_prop_ix);
			transfers[3].flags = video::CPropertyPoolHandler::TransferRequest::EF_FILL;
			transfers[3].buffer = getDefaultValueBufferBinding(ITransformTree::recomputed_stamp_prop_ix);
			return true;
		}
		
		//
		struct UpstreamRequestBase : RequestBase
		{
			video::CPropertyPoolHandler::UpStreamingRequest::Source parents = {};
			video::CPropertyPoolHandler::UpStreamingRequest::Source relativeTransforms = {};
		};
		struct UpstreamRequest : UpstreamRequestBase
		{
			core::SRange<const ITransformTree::node_t> nodes = {nullptr,nullptr};
		};
		inline bool setupTransfers(const UpstreamRequest& request, video::CPropertyPoolHandler::UpStreamingRequest* upstreams)
		{
			if (!request.tree)
				return false;

			const auto* pool = request.tree->getNodePropertyPool();
			for (auto i=0u; i<TransferCount; i++)
			{
				upstreams[i].elementCount = request.nodes.size();
				upstreams[i].srcAddresses = nullptr;
				upstreams[i].dstAddresses = request.nodes.begin();
			}
			if (request.nodes.empty())
				return true;

			upstreams[0].setFromPool(pool,ITransformTree::parent_prop_ix);
			if (request.parents.device2device || request.parents.data)
			{
				upstreams[0].fill = false;
				upstreams[0].source = request.parents;
			}
			else
			{
				upstreams[0].fill = true;
				upstreams[0].source.buffer = getDefaultValueBufferBinding(ITransformTree::parent_prop_ix);
			}
			upstreams[1].setFromPool(pool,ITransformTree::relative_transform_prop_ix);
			if (request.relativeTransforms.device2device || request.relativeTransforms.data)
			{
				upstreams[1].fill = false;
				upstreams[1].source = request.relativeTransforms;
			}
			else
			{
				upstreams[1].fill = true;
				upstreams[1].source.buffer = getDefaultValueBufferBinding(ITransformTree::relative_transform_prop_ix);
			}
			upstreams[2].setFromPool(pool,ITransformTree::modified_stamp_prop_ix);
			upstreams[2].fill = true;
			upstreams[2].source.buffer = getDefaultValueBufferBinding(ITransformTree::modified_stamp_prop_ix);
			upstreams[3].setFromPool(pool,ITransformTree::recomputed_stamp_prop_ix);
			upstreams[3].fill = true;
			upstreams[3].source.buffer = getDefaultValueBufferBinding(ITransformTree::recomputed_stamp_prop_ix);
			return true;
		}

		struct AdditionRequestBase
		{
			public:
				// must be in recording state
				video::IGPUCommandBuffer* cmdbuf;
				video::IGPUFence* fence;
				asset::SBufferBinding<video::IGPUBuffer> scratch;
				video::StreamingTransientDataBufferMT<>* upBuff;
				video::CPropertyPoolHandler* poolHandler;
				video::IQueue* queue;
				system::logger_opt_ptr logger = nullptr;

			protected:
				inline bool isValid() const
				{
					return cmdbuf && fence && scratch.isValid() && upBuff && poolHandler && queue;
				}
		};
		//
		struct AdditionRequest : UpstreamRequestBase,AdditionRequestBase
		{
			// if the `outNodes` have values not equal to `invalid_node` then we treat them as already allocated
			// (this allows you to split allocation of nodes from setting up the transfers)
			core::SRange<ITransformTree::node_t> outNodes = {nullptr,nullptr};

			inline bool isValid() const
			{
				return AdditionRequestBase::isValid() && outNodes.begin() && outNodes.begin()<=outNodes.end();
			}
		};
		inline uint32_t addNodes(
			const AdditionRequest& request, uint32_t& waitSemaphoreCount,
			video::IGPUSemaphore* const*& semaphoresToWaitBeforeOverwrite,
			const asset::PIPELINE_STAGE_FLAGS*& stagesToWaitForPerSemaphore, 
			const std::chrono::steady_clock::time_point& maxWaitPoint=video::GPUEventWrapper::default_wait()
		)
		{
			if (!request.isValid())
				return false;
			if (request.outNodes.empty())
				return true;

			if (!request.tree->allocateNodes(request.outNodes))
				return false;

			video::CPropertyPoolHandler::UpStreamingRequest upstreams[TransferCount];
			UpstreamRequest req;
			static_cast<UpstreamRequestBase&>(req) = request;
			req.nodes = {request.outNodes.begin(),request.outNodes.end()};
			if (!setupTransfers(req,upstreams))
				return false;

			auto upstreamsPtr = upstreams;
			return request.poolHandler->transferProperties(
				request.upBuff,request.cmdbuf,request.fence,request.queue,request.scratch,upstreamsPtr,TransferCount,
				waitSemaphoreCount,semaphoresToWaitBeforeOverwrite,stagesToWaitForPerSemaphore,request.logger,maxWaitPoint
			);
		}

		//
		struct SkeletonAllocationRequest : RequestBase,AdditionRequestBase
		{
			inline bool isValid() const
			{
				return AdditionRequestBase::isValid() && skeletons.begin() && skeletons.begin()<=skeletons.end() && outNodes && parentScratch;
			}

			struct StagingRequirements
			{
				uint32_t nodeCount;
				uint32_t transformCount; // for internal usage only
				uint32_t transformScratchCount;
			};
			inline StagingRequirements computeStagingRequirements() const
			{
				StagingRequirements reqs = {0u,0u,0u};
				auto instanceCountIt = instanceCounts;
				for (auto skeleton : skeletons)
				{
					if (skeleton)
					{
						const uint32_t jointCount = skeleton->getJointCount();
						const uint32_t jointInstanceCount = (instanceCounts ? (*instanceCountIt):1u)*jointCount;
						reqs.nodeCount += jointInstanceCount;
						if (skeleton->getDefaultTransformBinding().buffer)
							reqs.transformCount += jointCount;
					}
					instanceCountIt++;
				}
				if (reqs.transformCount)
				{
					constexpr uint32_t kNodesInATransform = sizeof(ITransformTree::relative_transform_t)/sizeof(uint32_t);
					reqs.transformScratchCount = reqs.transformCount+core::roundUp(reqs.nodeCount,kNodesInATransform);
				}
				return reqs;
			}
			
			core::SRange<const asset::ICPUSkeleton*> skeletons = {nullptr,nullptr};
			// if nullptr then treated like a buffer of {1,1,...,1,1}, else needs to be same length as the skeleton range
			const uint32_t* instanceCounts = nullptr;
			// If you make the skeleton hierarchy have a real parent, you won't be able to share it amongst multiple instances of a mesh
			// also in order to render with standard shaders you'll have to cancel out the model transform of the parent for the skinning matrices.
			const ITransformTree::node_t* const* skeletonInstanceParents = nullptr;
			// the following arrays need to be sized according to `StagingRequirements`
			// if the `outNodes` have values not equal to `invalid_node` then we treat them as already allocated
			// (this allows you to split allocation of nodes from setting up the transfers)
			ITransformTree::node_t* outNodes = nullptr;
			// scratch buffers are just required to be the set size, they can be filled with garbage
			ITransformTree::node_t* parentScratch;
			// must be non null if at least one skeleton has default transforms
			ITransformTree::relative_transform_t* transformScratch = nullptr;
		};
		inline bool addSkeletonNodes(
			const SkeletonAllocationRequest& request, uint32_t& waitSemaphoreCount,
			video::IGPUSemaphore* const*& semaphoresToWaitBeforeOverwrite,
			const asset::PIPELINE_STAGE_FLAGS*& stagesToWaitForPerSemaphore, 
			const std::chrono::steady_clock::time_point& maxWaitPoint=video::GPUEventWrapper::default_wait()
		)
		{
			if (!request.isValid())
				return false;

			const auto staging = request.computeStagingRequirements();
			if (staging.nodeCount==0u)
				return true;

			if (!request.tree->allocateNodes({request.outNodes,request.outNodes+staging.nodeCount}))
				return false;

			uint32_t* const srcTransformIndices = reinterpret_cast<uint32_t*>(request.transformScratch+staging.transformCount);
			{
				auto instanceCountIt = request.instanceCounts;
				auto skeletonInstanceParentsIt = request.skeletonInstanceParents;
				auto parentsIt = request.parentScratch;
				auto transformIt = request.transformScratch;
				auto srcTransformIndicesIt = srcTransformIndices;
				uint32_t baseJointInstance = 0u;
				uint32_t baseJoint = 0u;
				for (auto skeleton : request.skeletons)
				{
					const auto instanceCount = request.instanceCounts ? (*(instanceCountIt++)):1u;
					auto* const instanceParents = request.skeletonInstanceParents ? (*(skeletonInstanceParentsIt++)):nullptr;

					const auto jointCount = skeleton ? skeleton->getJointCount():0u;
					auto instanceParentsIt = instanceParents;
					for (auto instanceID=0u; instanceID<instanceCount; instanceID++)
					{
						for (auto jointID=0u; jointID<jointCount; jointID++)
						{
							uint32_t parentID = skeleton->getParentJointID(jointID);
							if (parentID!=asset::ICPUSkeleton::invalid_joint_id)
								parentID = request.outNodes[parentID+baseJointInstance];
							else
								parentID = instanceParents ? (*instanceParentsIt):ITransformTree::invalid_node;
							*(parentsIt++) = parentID;

							if (staging.transformScratchCount)
								*(srcTransformIndicesIt++) = jointID+baseJoint;
						}
						baseJointInstance += jointCount;
						instanceParentsIt++;
					}
					if (skeleton->getDefaultTransformBinding().buffer)
					for (auto jointID=0u; jointID<jointCount; jointID++)
						*(transformIt++) = skeleton->getDefaultTransformMatrix(jointID);
					baseJoint += jointCount;
				}
			}

			video::CPropertyPoolHandler::UpStreamingRequest upstreams[TransferCount];
			UpstreamRequest req = {};
			req.tree = request.tree;
			req.nodes = {request.outNodes,request.outNodes+staging.nodeCount};
			req.parents.data = request.parentScratch;
			if (staging.transformCount)
				req.relativeTransforms.data = request.transformScratch;
			if (!setupTransfers(req,upstreams))
				return false;
			if (staging.transformCount)
				upstreams[1].srcAddresses = srcTransformIndices;

			auto upstreamsPtr = upstreams;
			return request.poolHandler->transferProperties(
				request.upBuff,request.cmdbuf,request.fence,request.queue,request.scratch,upstreamsPtr,TransferCount,
				waitSemaphoreCount,semaphoresToWaitBeforeOverwrite,stagesToWaitForPerSemaphore,request.logger,maxWaitPoint
			);
		}


		 
		//
		inline void removeNodes(ITransformTree* tree, const ITransformTree::node_t* begin, const ITransformTree::node_t* end)
		{
			// If we start wanting a contiguous range to be maintained, this will need to change
			tree->getNodePropertyPool()->freeProperties(begin,end);
		}


		//
		struct SBarrierSuggestion
		{
			static inline constexpr uint32_t MaxBufferCount = 6u; // could be less?

			//
			enum E_FLAG
			{
				// basic
				EF_PRE_RELATIVE_TFORM_UPDATE = 0x1u,
				EF_POST_RELATIVE_TFORM_UPDATE = 0x2u,
				EF_PRE_GLOBAL_TFORM_RECOMPUTE = 0x4u,
				EF_POST_GLOBAL_TFORM_RECOMPUTE = 0x8u,
				// if you plan to run recompute right after update
				EF_INBETWEEN_RLEATIVE_UPDATE_AND_GLOBAL_RECOMPUTE = EF_POST_RELATIVE_TFORM_UPDATE|EF_PRE_GLOBAL_TFORM_RECOMPUTE,
				// if you're planning to run the fused recompute and update kernel
				EF_PRE_UPDATE_AND_RECOMPUTE = EF_PRE_RELATIVE_TFORM_UPDATE|EF_PRE_GLOBAL_TFORM_RECOMPUTE,
				EF_POST_UPDATE_AND_RECOMPUTE = EF_POST_RELATIVE_TFORM_UPDATE|EF_POST_GLOBAL_TFORM_RECOMPUTE,
			};

			core::bitflag<asset::PIPELINE_STAGE_FLAGS> srcStageMask = asset::PIPELINE_STAGE_FLAGS::NONE;
			core::bitflag<asset::PIPELINE_STAGE_FLAGS> dstStageMask = asset::PIPELINE_STAGE_FLAGS::NONE;
			asset::SMemoryBarrier requestRanges = {};
			asset::SMemoryBarrier modificationRequests = {};
			asset::SMemoryBarrier relativeTransforms = {};
			asset::SMemoryBarrier modifiedTimestamps = {};
			asset::SMemoryBarrier globalTransforms = {};
			asset::SMemoryBarrier recomputedTimestamps = {};
			asset::SMemoryBarrier normalMatrices = {};
		};
		//
		static inline SBarrierSuggestion barrierHelper(const SBarrierSuggestion::E_FLAG type)
		{
			const auto rwAccessMask = asset::ACCESS_FLAGS::SHADER_WRITE_BITS|asset::ACCESS_FLAGS::SHADER_READ_BITS;

			SBarrierSuggestion barrier;
#if 0 // TODO: redo
			if (type&SBarrierSuggestion::EF_PRE_RELATIVE_TFORM_UPDATE)
			{
				// we're mostly concerned about stuff writing to buffer update reads from 
				barrier.dstStageMask |= asset::PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT;
				barrier.requestRanges.dstAccessMask |= asset::EAF_SHADER_READ_BIT;
				barrier.modificationRequests.dstAccessMask |= asset::EAF_SHADER_READ_BIT;
				// the case of update stepping on its own toes is handled by the POST case
			}
			if (type&SBarrierSuggestion::EF_POST_RELATIVE_TFORM_UPDATE)
			{
				// we're mostly concerned about relative tform update overwriting itself
				barrier.srcStageMask |= asset::PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT;
				barrier.dstStageMask |= asset::PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT;
				// we also need to barrier against any future update to the inputs overstepping our reading
				barrier.requestRanges.srcAccessMask |= asset::EAF_SHADER_READ_BIT;
				barrier.modificationRequests.srcAccessMask |= asset::EAF_SHADER_READ_BIT;
				// relative transform can be pre-post multiplied or entirely erased, we're not in charge of that
				// need to also worry about update<->update loop, so both masks are R/W
				barrier.relativeTransforms.srcAccessMask |= rwAccessMask;
				barrier.relativeTransforms.dstAccessMask |= rwAccessMask;
				// we will only overwrite
				barrier.modifiedTimestamps.srcAccessMask |= asset::EAF_SHADER_WRITE_BIT;
				// modified timestamp will be written by previous update, but also has to be read by recompute later
				barrier.modifiedTimestamps.dstAccessMask |= rwAccessMask;
				// we don't touch anything else
			}
			if (type&SBarrierSuggestion::EF_PRE_GLOBAL_TFORM_RECOMPUTE)
			{
				// we're mostly concerned about relative transform update not being finished before global transform recompute runs 
				barrier.srcStageMask |= asset::PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT;
				barrier.dstStageMask |= asset::PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT;
				barrier.relativeTransforms.srcAccessMask |= rwAccessMask;
				barrier.relativeTransforms.dstAccessMask |= asset::EAF_SHADER_READ_BIT;
				barrier.modifiedTimestamps.srcAccessMask |= asset::EAF_SHADER_WRITE_BIT;
				barrier.modifiedTimestamps.dstAccessMask |= asset::EAF_SHADER_READ_BIT;
				// the case of update stepping on its own toes is handled by the POST case
			}
			if (type&SBarrierSuggestion::EF_POST_GLOBAL_TFORM_RECOMPUTE)
			{
				// we're mostly concerned about global tform recompute overwriting itself
				barrier.srcStageMask |= asset::PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT;
				barrier.dstStageMask |= asset::PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT;
				// and future local update overwritng the inputs before recompute is done reading
				barrier.relativeTransforms.srcAccessMask |= asset::EAF_SHADER_READ_BIT;
				barrier.relativeTransforms.dstAccessMask |= rwAccessMask;
				barrier.modifiedTimestamps.srcAccessMask |= asset::EAF_SHADER_READ_BIT;
				barrier.modifiedTimestamps.dstAccessMask |= asset::EAF_SHADER_WRITE_BIT;
				// global transforms and recompute timestamps can be both read and written
				barrier.globalTransforms.srcAccessMask |= rwAccessMask;
				barrier.globalTransforms.dstAccessMask |= rwAccessMask;
				barrier.recomputedTimestamps.srcAccessMask |= rwAccessMask;
				barrier.recomputedTimestamps.dstAccessMask |= rwAccessMask;
				// global tform recompute only writes these, but its reasonable to assume someone will read them inbetween
				barrier.normalMatrices.srcAccessMask |= rwAccessMask;
				barrier.normalMatrices.dstAccessMask |= rwAccessMask;
			}
#endif
			return barrier;
		}

		//
		struct BaseParams
		{
			BaseParams() = default;

			inline BaseParams& operator=(const BaseParams& other) = default;

			video::IGPUCommandBuffer* cmdbuf; // must already be in recording state
			ITransformTree* tree;
			system::logger_opt_ptr logger = nullptr;
		};
		//
		union DispatchParams
		{
			DispatchParams()
			{
				indirect = {};
				direct.nodeCount = 0u;
			}

			inline DispatchParams& operator=(const DispatchParams& other)
			{
				if (other.indirect.isValid())
					indirect = other.indirect;
				else
				{
					indirect = {};
					direct.nodeCount = other.direct.nodeCount;
				}
				return *this;
			}

			asset::SBufferBinding<video::IGPUBuffer> indirect;
			struct
			{
				private:
					uint64_t dummy;
				public:
					uint32_t nodeCount;
			} direct;
		};
		//
		static inline core::smart_refctd_ptr<video::IGPUDescriptorSetLayout> createUpdateLocalTransformsDescriptorSetLayout(video::ILogicalDevice* device)
		{
			return createDescriptorSetLayout<2u>(device);
		}
		// first uint in the `requestRanges` buffer tells us how many ModificationRequestRanges we have
		// second uint in the `requestRanges` buffer tells us how many total requests we have
		// rest is filled wtih ModificationRequestRange
		using ModificationRequestRange = nbl_glsl_transform_tree_modification_request_range_t;
		// `modificationRequests` is filled with RelativeTransformModificationRequest
		static inline void updateUpdateLocalTransformsDescriptorSet(
			video::ILogicalDevice* device, video::IGPUDescriptorSet* updateLocalTransformsDS, 
			asset::SBufferBinding<video::IGPUBuffer>&& requestRanges,
			asset::SBufferBinding<video::IGPUBuffer>&& modificationRequests
		)
		{
			updateDescriptorSet<2>(device,updateLocalTransformsDS,{std::move(requestRanges),std::move(modificationRequests)});
		}
		inline void updateLocalTransforms(const BaseParams& baseParams, const DispatchParams& dispatchParams, const video::IGPUDescriptorSet* updateLocalTransformsDS)
		{
			const auto updatePipeline = choosePipelines(baseParams.tree).updateRelative.get();

			const video::IGPUDescriptorSet* descSets[] = {baseParams.tree->getNodePropertyPoolDescriptorSet(),updateLocalTransformsDS};
			baseParams.cmdbuf->bindDescriptorSets(asset::EPBP_COMPUTE,updatePipeline->getLayout(),0u,2u,descSets);
			
			baseParams.cmdbuf->bindComputePipeline(updatePipeline);
			dispatch(baseParams.cmdbuf,dispatchParams);
		}

		//
		static inline core::smart_refctd_ptr<video::IGPUDescriptorSetLayout> createRecomputeGlobalTransformsDescriptorSetLayout(video::ILogicalDevice* device)
		{
			return createDescriptorSetLayout<1u>(device);
		}
		// first uint in the `nodeID` buffer tells us how many nodes to update we have
		static inline void updateRecomputeGlobalTransformsDescriptorSet(
			video::ILogicalDevice* device, video::IGPUDescriptorSet* recomputeGlobalTransformsDS,
			asset::SBufferBinding<video::IGPUBuffer>&& nodeID
		)
		{
			updateDescriptorSet<1>(device,recomputeGlobalTransformsDS,{std::move(nodeID)});
		}
		inline void recomputeGlobalTransforms(const BaseParams& baseParams, const DispatchParams& dispatchParams, const video::IGPUDescriptorSet* recomputeGlobalTransformsDS)
		{
			const auto recomputePipeline = choosePipelines(baseParams.tree).recomputeGlobal.get();

			const video::IGPUDescriptorSet* descSets[] = {baseParams.tree->getNodePropertyPoolDescriptorSet(),recomputeGlobalTransformsDS};
			baseParams.cmdbuf->bindDescriptorSets(asset::EPBP_COMPUTE,recomputePipeline->getLayout(),0u,2u,descSets);
			
			baseParams.cmdbuf->bindComputePipeline(recomputePipeline);
			dispatch(baseParams.cmdbuf,dispatchParams);
		}

		// Note: Function signature subject to change
		inline void updateLocalAndRecomputeGlobalTransforms(
			const BaseParams& baseParams, const DispatchParams& updateDispatch, const DispatchParams& recomputeDispatch,
			const video::IGPUDescriptorSet* updateLocalTransformsDS, const video::IGPUDescriptorSet* recomputeGlobalTransformsDS
		)
		{
			assert(false);
#if 0 // TODO: REDO
			const core::bitflag<asset::PIPELINE_STAGE_FLAGS> renderingStages = asset::PIPELINE_STAGE_FLAGS::PRE_RASTERIZATION_SHADERS_BITS;
			updateLocalTransforms(baseParams,updateDispatch,updateLocalTransformsDS);
			{
				video::IGPUCommandBuffer::SDependencyInfo info = {};
				asset::SMemoryBarrier memoryBarrier;
				memoryBarrier.srcStageMask = asset::PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT;
				memoryBarrier.srcAccessMask = asset::ACCESS_FLAGS::MEMORY_WRITE_BITS;
				memoryBarrier.dstStageMask = asset::PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT;
				memoryBarrier.dstAccessMask = asset::ACCESS_FLAGS::MEMORY_WRITE_BITS|asset::ACCESS_FLAGS::MEMORY_READ_BITS;
				// I'm lazy but this should really be on a per-buffer granularity!
				info.memBarrierCount = 1u;
				info.memBarriers = &memoryBarrier;
				baseParams.cmdbuf->pipelineBarrier(asset::EDF_NONE,info);
			}
			recomputeGlobalTransforms(baseParams,recomputeDispatch,recomputeGlobalTransformsDS);
#endif
		}


		constexpr static inline auto DebugAABBIndices = 24u;
		constexpr static inline auto DebugLineIndices = 2u;
		constexpr static inline auto DebugIndexCount = DebugAABBIndices + DebugLineIndices;
		//
		inline const video::IGPUBuffer* getDebugIndexBuffer() const { return m_debugIndexBuffer.get(); }

		static inline constexpr uint32_t DebugNodeIDAttributeIndex = 14u;
		static inline constexpr uint32_t DebugAABBIDAttributeIndex = 15u;
		static inline constexpr uint32_t DebugNodeIDBindingIndex = DebugNodeIDAttributeIndex;
		static inline constexpr uint32_t DebugAABBIDBindingIndex = DebugAABBIDAttributeIndex;
		//
		template<class TransformTree>
		inline auto createDebugPipeline(core::smart_refctd_ptr<const video::IGPURenderpass>&& renderpass)
		{
			nbl::video::IGPUGraphicsPipeline::SCreationParams graphicsPipelineParams;
			graphicsPipelineParams.renderpassIndependent = choosePipelines<TransformTree>().debugDraw;
			graphicsPipelineParams.renderpass = std::move(renderpass);
			return m_device->createGraphicsPipeline(nullptr,std::move(graphicsPipelineParams));
		}

		//
		static inline core::smart_refctd_ptr<video::IGPUDescriptorSetLayout> createDebugDrawDescriptorSetLayout(video::ILogicalDevice* device)
		{
			return createDescriptorSetLayout<1u>(device);
		}
		static inline void updateDebugDrawDescriptorSet(video::ILogicalDevice* device, video::IGPUDescriptorSet* debugDrawDS, asset::SBufferBinding<video::IGPUBuffer>&& aabbPool)
		{
			updateDescriptorSet<1>(device,debugDrawDS,{std::move(aabbPool)});
		}
		struct DebugPushConstants
		{
			core::matrix4SIMD viewProjectionMatrix;
			core::vector4df_SIMD lineColor;
			core::vector4df_SIMD aabbColor;
		};
		inline void debugDraw(
			video::IGPUCommandBuffer* cmdbuf, const video::IGPUGraphicsPipeline* pipeline, const ITransformTree* tree,
			const video::IGPUDescriptorSet* debugDrawDS, const asset::SBufferBinding<const video::IGPUBuffer>& nodeID, const asset::SBufferBinding<const video::IGPUBuffer>& aabbID,
			const DebugPushConstants& pushConstants, const uint32_t count
		)
		{
			auto layout = choosePipelines(tree).debugDraw->getLayout();
			assert(pipeline->getRenderpassIndependentPipeline()->getLayout()==layout);

			const video::IGPUDescriptorSet* sets[] = {tree->getNodePropertyPoolDescriptorSet(),debugDrawDS};
			cmdbuf->bindDescriptorSets(asset::EPBP_GRAPHICS,layout,0u,2u,sets);
			cmdbuf->bindGraphicsPipeline(pipeline);
			cmdbuf->bindVertexBuffers(DebugNodeIDBindingIndex,1u,&nodeID);
			cmdbuf->bindVertexBuffers(DebugAABBIDBindingIndex,1u,&aabbID);
			cmdbuf->bindIndexBuffer({0u,m_debugIndexBuffer},asset::EIT_16BIT);
			cmdbuf->pushConstants(layout,asset::IShader::ESS_VERTEX,0u,sizeof(DebugPushConstants),&pushConstants);
			cmdbuf->drawIndexed(DebugIndexCount,count,0u,0u,0u);
		}

		//
		struct DescriptorSets
		{
			core::smart_refctd_ptr<video::IGPUDescriptorSet> updateLocal;
			core::smart_refctd_ptr<video::IGPUDescriptorSet> recomputeGlobal;
			core::smart_refctd_ptr<video::IGPUDescriptorSet> debugDraw;
		};
		DescriptorSets createAllDescriptorSets(video::ILogicalDevice* device)
		{
			core::smart_refctd_ptr<video::IGPUDescriptorSetLayout> layouts[] =
			{
				createUpdateLocalTransformsDescriptorSetLayout(device),
				createRecomputeGlobalTransformsDescriptorSetLayout(device),
				createDebugDrawDescriptorSetLayout(device)
			};

			auto pool = device->createDescriptorPoolForDSLayouts(video::IDescriptorPool::ECF_NONE,&layouts->get(),&layouts->get()+3u);

			DescriptorSets descSets;
			descSets.updateLocal = pool->createDescriptorSet(std::move(layouts[0]));
			descSets.recomputeGlobal = pool->createDescriptorSet(std::move(layouts[1]));
			descSets.debugDraw = pool->createDescriptorSet(std::move(layouts[2]));
			return descSets;
		}
	protected:
		static inline uint64_t getDefaultValueBufferOffset(const video::IPhysicalDevice::SLimits& limits, uint32_t prop_ix)
		{
			uint64_t offset = 0u;
			const uint64_t ssboOffsetAlignment = limits.minSSBOAlignment;
			if (prop_ix!=ITransformTree::relative_transform_prop_ix)
			{
				offset = core::roundUp(offset+sizeof(ITransformTree::relative_transform_t),ssboOffsetAlignment);
				if (prop_ix!=ITransformTree::parent_prop_ix)
				{
					offset = core::roundUp(offset+sizeof(ITransformTree::parent_t),ssboOffsetAlignment);
					if (prop_ix!=ITransformTree::modified_stamp_prop_ix)
					{
						offset = core::roundUp(offset+sizeof(ITransformTree::modified_stamp_t),ssboOffsetAlignment);
						if (prop_ix!=ITransformTree::recomputed_stamp_prop_ix)
							return core::roundUp(offset+sizeof(ITransformTree::recomputed_stamp_t),ssboOffsetAlignment);
					}
				}
			}
			return offset;
		}
		struct Pipelines
		{
			inline bool isValid() const
			{
				return updateRelative && recomputeGlobal && debugDraw;
			}

			core::smart_refctd_ptr<video::IGPUComputePipeline> updateRelative;
			core::smart_refctd_ptr<video::IGPUComputePipeline> recomputeGlobal;
			// TODO: after BaW
			// core::smart_refctd_ptr<video::IGPUComputePipeline> updateRelativeAndRecomputeGlobal;
			core::smart_refctd_ptr<video::IGPURenderpassIndependentPipeline> debugDraw;
		};
		template<class TransformTree>
		static inline Pipelines createPipelines(
			video::ILogicalDevice* device,
			const core::smart_refctd_ptr<video::IGPUSpecializedShader>& updateRelativeSpec,
			const core::smart_refctd_ptr<video::IGPUSpecializedShader>& recomputeGlobalSpec,
			const core::smart_refctd_ptr<video::IGPUSpecializedShader>& debugDrawVertexSpec,
			const core::smart_refctd_ptr<video::IGPUSpecializedShader>& debugDrawFragmentSpec,
			const core::smart_refctd_ptr<video::IGPUDescriptorSetLayout>& updateLocalDsLayout,
			const core::smart_refctd_ptr<video::IGPUDescriptorSetLayout>& recomputeGlobalDsLayout,
			const core::smart_refctd_ptr<video::IGPUDescriptorSetLayout>& debugDrawDsLayout
		)
		{
			Pipelines retval = {};

			auto poolLayout = TransformTree::createPoolDescriptorSetLayout(device);
			
			auto updateRelativeLayout = device->createPipelineLayout(nullptr,nullptr,core::smart_refctd_ptr(poolLayout),core::smart_refctd_ptr(updateLocalDsLayout));
			auto recomputeGlobalLayout = device->createPipelineLayout(nullptr,nullptr,core::smart_refctd_ptr(poolLayout),core::smart_refctd_ptr(recomputeGlobalDsLayout));
			asset::SPushConstantRange pcRange;
			pcRange.offset = 0u;
			pcRange.size = sizeof(DebugPushConstants);
			pcRange.stageFlags = asset::IShader::ESS_VERTEX;
			auto debugDrawLayout = device->createPipelineLayout(&pcRange,&pcRange+1u,core::smart_refctd_ptr(poolLayout),core::smart_refctd_ptr(debugDrawDsLayout));

			retval.updateRelative = device->createComputePipeline(nullptr,std::move(updateRelativeLayout),core::smart_refctd_ptr(updateRelativeSpec));
			retval.recomputeGlobal = device->createComputePipeline(nullptr,std::move(recomputeGlobalLayout),core::smart_refctd_ptr(recomputeGlobalSpec));
			core::smart_refctd_ptr<video::IGPURenderpassIndependentPipeline> debugDrawIndepenedentPipeline;
			{
				asset::SVertexInputParams vertexInputParams = {};
				vertexInputParams.bindings[DebugNodeIDBindingIndex].inputRate = asset::EVIR_PER_INSTANCE;
				vertexInputParams.bindings[DebugNodeIDBindingIndex].stride = sizeof(uint32_t);
				vertexInputParams.bindings[DebugAABBIDBindingIndex].inputRate = asset::EVIR_PER_INSTANCE;
				vertexInputParams.bindings[DebugAABBIDBindingIndex].stride = sizeof(uint32_t);

				vertexInputParams.attributes[DebugNodeIDAttributeIndex].binding = DebugNodeIDBindingIndex;
				vertexInputParams.attributes[DebugNodeIDAttributeIndex].format = asset::EF_R32_UINT;
				vertexInputParams.attributes[DebugAABBIDAttributeIndex].binding = DebugAABBIDBindingIndex;
				vertexInputParams.attributes[DebugAABBIDAttributeIndex].format = asset::EF_R32_UINT;

				vertexInputParams.enabledBindingFlags |= 0x1u<<DebugNodeIDBindingIndex;
				vertexInputParams.enabledBindingFlags |= 0x1u<<DebugAABBIDBindingIndex;
				vertexInputParams.enabledAttribFlags |= 0x1u<<DebugNodeIDAttributeIndex;
				vertexInputParams.enabledAttribFlags |= 0x1u<<DebugAABBIDAttributeIndex;

				asset::SBlendParams blendParams = {};
				asset::SPrimitiveAssemblyParams primitiveAssemblyParams = {};
				primitiveAssemblyParams.primitiveType = asset::EPT_LINE_LIST;
				asset::SRasterizationParams rasterizationParams = {};
				rasterizationParams.depthTestEnable = false;

				video::IGPUSpecializedShader* const debugDrawShaders[] = {debugDrawVertexSpec.get(),debugDrawFragmentSpec.get()};
				retval.debugDraw = device->createRenderpassIndependentPipeline(
					nullptr,std::move(debugDrawLayout),debugDrawShaders,debugDrawShaders+2u,vertexInputParams,blendParams,primitiveAssemblyParams,rasterizationParams
				);
			}

			return retval;
		}

		//
		ITransformTreeManager(
			core::smart_refctd_ptr<video::ILogicalDevice>&& _device,
			Pipelines&& _pipelinesWithoutNormalMatrices,
			Pipelines&& _pipelinesWithNormalMatrices,
			core::smart_refctd_ptr<video::IGPUBuffer>&& _defaultFillValues,
			core::smart_refctd_ptr<video::IGPUBuffer>&& _debugIndexBuffer
		) : m_device(std::move(_device)), m_pipelinesWithoutNormalMatrices(std::move(_pipelinesWithoutNormalMatrices)),
			m_pipelinesWithNormalMatrices(std::move(_pipelinesWithNormalMatrices)),
			m_defaultFillValues(std::move(_defaultFillValues)),
			m_debugIndexBuffer(std::move(_debugIndexBuffer)),
			m_workgroupSize(m_device->getPhysicalDevice()->getLimits().maxOptimallyResidentWorkgroupInvocations)
		{
		}
		~ITransformTreeManager()
		{
			// everything drops itself automatically
		}
		
		inline asset::SBufferBinding<video::IGPUBuffer> getDefaultValueBufferBinding(uint32_t prop_ix) const
		{
			const auto& limits = m_device->getPhysicalDevice()->getLimits();
			return {getDefaultValueBufferOffset(limits,prop_ix),m_defaultFillValues};
		}
		
		template<class TransformTree>
		inline Pipelines& choosePipelines()
		{
			return TransformTree::HasNormalMatrices ? m_pipelinesWithNormalMatrices:m_pipelinesWithoutNormalMatrices;
		}
		inline Pipelines& choosePipelines(const ITransformTree* tree)
		{
			return tree->hasNormalMatrices() ? m_pipelinesWithNormalMatrices:m_pipelinesWithoutNormalMatrices;
		}
		
		void dispatch(video::IGPUCommandBuffer* cmdbuf, const DispatchParams& dispatch)
		{
			if (dispatch.indirect.isValid())
				cmdbuf->dispatchIndirect(dispatch.indirect);
			else
			{
				const auto& limits = m_device->getPhysicalDevice()->getLimits();
				cmdbuf->dispatch(limits.computeOptimalPersistentWorkgroupDispatchSize(dispatch.direct.nodeCount,m_workgroupSize),1u,1u);
			}
		}


		core::smart_refctd_ptr<video::ILogicalDevice> m_device;
		Pipelines m_pipelinesWithoutNormalMatrices;
		Pipelines m_pipelinesWithNormalMatrices;
		core::smart_refctd_ptr<video::IGPUBuffer> m_defaultFillValues;
		const uint32_t m_workgroupSize;

		core::smart_refctd_ptr<video::IGPUBuffer> m_debugIndexBuffer;


};
#endif
} // end namespace nbl::scene

#endif

