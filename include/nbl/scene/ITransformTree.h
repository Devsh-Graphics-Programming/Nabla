// Copyright (C) 2019 - DevSH Graphics Programming Sp. z O.O.
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors

#ifndef _NBL_SCENE_I_TRANSFORM_TREE_H_INCLUDED_
#define _NBL_SCENE_I_TRANSFORM_TREE_H_INCLUDED_

#include "nbl/core/declarations.h"

#include "nbl/video/declarations.h"
#include "nbl/video/definitions.h"

namespace nbl::scene
{

#if 0 // legacy and unported
class ITransformTree : public virtual core::IReferenceCounted
{
	public:
		using node_t = uint32_t;
		static inline constexpr node_t invalid_node = video::IPropertyPool::invalid;

		using timestamp_t = video::IGPUAnimationLibrary::timestamp_t;
		// two timestamp values are reserved for initialization
		static inline constexpr timestamp_t min_timestamp = 0u;
		static inline constexpr timestamp_t max_timestamp = 0xfffffffcu;
		static inline constexpr timestamp_t initial_modified_timestamp = 0xfffffffdu;
		static inline constexpr timestamp_t initial_recomputed_timestamp = 0xfffffffeu;
		
		using parent_t = node_t;
		using relative_transform_t = core::matrix3x4SIMD;
		using modified_stamp_t = timestamp_t;
		using global_transform_t = core::matrix3x4SIMD;
		using recomputed_stamp_t = timestamp_t;

		static inline constexpr uint32_t parent_prop_ix = 0u;
		static inline constexpr uint32_t relative_transform_prop_ix = 1u;
		static inline constexpr uint32_t modified_stamp_prop_ix = 2u;
		static inline constexpr uint32_t global_transform_prop_ix = 3u;
		static inline constexpr uint32_t recomputed_stamp_prop_ix = 4u;

		//
		template<class TransformTree>
		static inline core::smart_refctd_ptr<asset::ICPUDescriptorSetLayout> createPoolDescriptorSetLayout(asset::IShader::E_SHADER_STAGE* stageAccessFlags=nullptr)
		{
			using property_pool_t = typename TransformTree::property_pool_t;
			return property_pool_t::createDescriptorSetLayout(stageAccessFlags);
		}
		template<class TransformTree>
		static inline core::smart_refctd_ptr<video::IGPUDescriptorSetLayout> createPoolDescriptorSetLayout(video::ILogicalDevice* device, asset::IShader::E_SHADER_STAGE* stageAccessFlags=nullptr)
		{
			using property_pool_t = typename TransformTree::property_pool_t;
			return property_pool_t::createDescriptorSetLayout(device,stageAccessFlags);
		}
		//
		template<class TransformTree>
		static inline core::smart_refctd_ptr<asset::ICPUDescriptorSetLayout> createRenderDescriptorSetLayout(asset::IShader::E_SHADER_STAGE* stageAccessFlags=nullptr)
		{
			asset::ICPUDescriptorSetLayout::SBinding bindings[TransformTree::RenderDescriptorSetBindingCount];
            asset::ICPUDescriptorSetLayout::fillBindingsSameType(bindings,TransformTree::RenderDescriptorSetBindingCount,asset::IDescriptor::E_TYPE::ET_STORAGE_BUFFER,nullptr,stageAccessFlags);
			return core::make_smart_refctd_ptr<asset::ICPUDescriptorSetLayout>(bindings,bindings+TransformTree::RenderDescriptorSetBindingCount);
		}
		template<class TransformTree>
		static inline core::smart_refctd_ptr<video::IGPUDescriptorSetLayout> createRenderDescriptorSetLayout(video::ILogicalDevice* device, asset::IShader::E_SHADER_STAGE* stageAccessFlags=nullptr)
		{
			video::IGPUDescriptorSetLayout::SBinding bindings[TransformTree::RenderDescriptorSetBindingCount];
			video::IGPUDescriptorSetLayout::fillBindingsSameType(bindings,TransformTree::RenderDescriptorSetBindingCount,asset::IDescriptor::E_TYPE::ET_STORAGE_BUFFER,nullptr,stageAccessFlags);
			return device->createDescriptorSetLayout(bindings,bindings+TransformTree::RenderDescriptorSetBindingCount);
		}
		
		//
		virtual bool hasNormalMatrices() const =0;
		virtual uint32_t getRenderDescriptorSetBindingCount() const =0;

		//
		virtual const video::IPropertyPool* getNodePropertyPool() const=0;

		//
		inline const auto* getNodePropertyPoolDescriptorSet() const {return m_transformHierarchyDS.get();}

		//
		inline const auto* getRenderDescriptorSet() const {return m_renderDS.get();}

		// nodes array must be initialized with invalid_node
		[[nodiscard]] inline bool allocateNodes(const core::SRange<ITransformTree::node_t>& outNodes)
		{
			if (outNodes.size()>getNodePropertyPool()->getFree())
				return false;

			return getNodePropertyPool()->allocateProperties(outNodes.begin(),outNodes.end());
		}

		// This removes all nodes in the hierarchy, if you want to remove individual nodes, use `ITransformTreeManager::removeNodes`
		inline void clearNodes()
		{
			getNodePropertyPool()->freeAllProperties();
		}

		//
		[[nodiscard]] inline bool copyGlobalTransforms(
			video::CPropertyPoolHandler* pphandler, video::IGPUCommandBuffer* const cmdbuf, video::IGPUFence* const fence,
			const asset::SBufferBinding<video::IGPUBuffer>& scratch, const asset::SBufferRange<video::IGPUBuffer>& nodes,
			const asset::SBufferBinding<video::IGPUBuffer>& srcTransforms, system::logger_opt_ptr logger
		)
		{
			video::CPropertyPoolHandler::TransferRequest request;
			request.setFromPool(getNodePropertyPool(),global_transform_prop_ix);
			request.flags = video::CPropertyPoolHandler::TransferRequest::EF_NONE;
			request.elementCount = nodes.size/sizeof(scene::ITransformTree::node_t);
			request.srcAddressesOffset = 0u;
			request.buffer = srcTransforms;
			return pphandler->transferProperties(cmdbuf,fence,scratch,{nodes.offset,nodes.buffer},&request,&request+1u,logger);
		}

		//
		[[nodiscard]] inline bool downloadGlobalTransforms(
			video::CPropertyPoolHandler* pphandler, video::IGPUCommandBuffer* const cmdbuf, video::IGPUFence* const fence,
			const asset::SBufferBinding<video::IGPUBuffer>& scratch, const asset::SBufferRange<video::IGPUBuffer>& nodes,
			const asset::SBufferBinding<video::IGPUBuffer>& dstTransforms, system::logger_opt_ptr logger
		)
		{
			video::CPropertyPoolHandler::TransferRequest request;
			request.setFromPool(getNodePropertyPool(),global_transform_prop_ix);
			request.flags = video::CPropertyPoolHandler::TransferRequest::EF_DOWNLOAD;
			request.elementCount = nodes.size/sizeof(scene::ITransformTree::node_t);
			request.srcAddressesOffset = 0u;
			request.buffer = dstTransforms;
			return pphandler->transferProperties(cmdbuf,fence,scratch,{nodes.offset,nodes.buffer},&request,&request+1u,logger);
		}

	protected:
		template<class TransformTree, typename... Args>
		static inline bool create(
			core::smart_refctd_ptr<typename TransformTree::property_pool_t>& outPool,
			core::smart_refctd_ptr<video::IGPUDescriptorSet>& outPoolDS,
			core::smart_refctd_ptr<video::IGPUDescriptorSet>& outRenderDS,
			video::ILogicalDevice* device, Args... args
		)
		{
			using property_pool_t = typename TransformTree::property_pool_t;
			outPool = property_pool_t::create(device,std::forward<Args>(args)...);
			if (!outPool)
				return false;

			video::IDescriptorPool::SCreateInfo createInfo;
			createInfo.maxDescriptorCount[static_cast<uint32_t>(asset::IDescriptor::E_TYPE::ET_STORAGE_BUFFER)] = property_pool_t::PropertyCount + TransformTree::RenderDescriptorSetBindingCount;
			createInfo.maxSets = 2;
			auto dsp = device->createDescriptorPool(std::move(createInfo));
			if (!dsp)
				return false;

			video::IGPUDescriptorSet::SWriteDescriptorSet writes[property_pool_t::PropertyCount];
			static_assert(TransformTree::RenderDescriptorSetBindingCount<=property_pool_t::PropertyCount);
			for (auto i=0u; i<property_pool_t::PropertyCount; i++)
			{
				writes[i].binding = i;
				writes[i].descriptorType = asset::IDescriptor::E_TYPE::ET_STORAGE_BUFFER;
				writes[i].count = 1u;
			}
			auto poolLayout = createPoolDescriptorSetLayout<TransformTree>(device);
			auto renderLayout = createRenderDescriptorSetLayout<TransformTree>(device);
			if (!poolLayout || !renderLayout)
				return false;

			outPoolDS = dsp->createDescriptorSet(std::move(poolLayout));
			outRenderDS = dsp->createDescriptorSet(std::move(renderLayout));
			if (!outPoolDS || !outRenderDS)
				return false;

			using DescriptorInfo = video::IGPUDescriptorSet::SDescriptorInfo;
			DescriptorInfo infos[property_pool_t::PropertyCount];
			for (auto i=0u; i<property_pool_t::PropertyCount; i++)
			{
				writes[i].dstSet = outPoolDS.get();
				writes[i].arrayElement = 0u;
				writes[i].info = infos+i;

				infos[i] = DescriptorInfo(outPool->getPropertyMemoryBlock(i));
			}
			device->updateDescriptorSets(property_pool_t::PropertyCount,writes,0u,nullptr);
			for (auto i=0u; i<property_pool_t::PropertyCount; i++)
				writes[i].dstSet = outRenderDS.get();
			infos[0] = DescriptorInfo(outPool->getPropertyMemoryBlock(global_transform_prop_ix));
			if (TransformTree::HasNormalMatrices)
				infos[1] = DescriptorInfo(outPool->getPropertyMemoryBlock(TransformTree::normal_matrix_prop_ix));
			device->updateDescriptorSets(TransformTree::RenderDescriptorSetBindingCount,writes,0u,nullptr);
			return true;
		}

		ITransformTree(core::smart_refctd_ptr<video::IGPUDescriptorSet>&& _transformHierarchyDS, core::smart_refctd_ptr<video::IGPUDescriptorSet>&& _renderDS)
			: m_transformHierarchyDS(std::move(_transformHierarchyDS)), m_renderDS(std::move(_renderDS)) {}
		virtual ~ITransformTree() =0;

		friend class ITransformTreeManager;
		inline video::IPropertyPool* getNodePropertyPool()
		{
			return const_cast<video::IPropertyPool*>(const_cast<const ITransformTree*>(this)->getNodePropertyPool());
		}

		//
		core::smart_refctd_ptr<video::IGPUDescriptorSet> m_transformHierarchyDS,m_renderDS;
		// TODO: do we keep a contiguous `node_t` array in-case we want to shortcut to full tree reevaluation when the number of relative transform modification requests > totalNodes*ratio (or overflows the temporary buffer we've provided) ?
};

class ITransformTreeWithoutNormalMatrices : public ITransformTree
{
	public:
		using property_pool_t = video::CPropertyPool<core::allocator,
			parent_t,
			relative_transform_t,modified_stamp_t,
			global_transform_t,recomputed_stamp_t
		>;

		//
		static inline core::smart_refctd_ptr<asset::ICPUDescriptorSetLayout> createPoolDescriptorSetLayout(asset::IShader::E_SHADER_STAGE* stageAccessFlags=nullptr)
		{
			return ITransformTree::createPoolDescriptorSetLayout<ITransformTreeWithoutNormalMatrices>(stageAccessFlags);
		}
		static inline core::smart_refctd_ptr<video::IGPUDescriptorSetLayout> createPoolDescriptorSetLayout(video::ILogicalDevice* device, asset::IShader::E_SHADER_STAGE* stageAccessFlags=nullptr)
		{
			return ITransformTree::createPoolDescriptorSetLayout<ITransformTreeWithoutNormalMatrices>(device,stageAccessFlags);
		}
		static inline core::smart_refctd_ptr<asset::ICPUDescriptorSetLayout> createRenderDescriptorSetLayout(asset::IShader::E_SHADER_STAGE* stageAccessFlags=nullptr)
		{
			return ITransformTree::createRenderDescriptorSetLayout<ITransformTreeWithoutNormalMatrices>(stageAccessFlags);
		}
		static inline core::smart_refctd_ptr<video::IGPUDescriptorSetLayout> createRenderDescriptorSetLayout(video::ILogicalDevice* device, asset::IShader::E_SHADER_STAGE* stageAccessFlags=nullptr)
		{
			return ITransformTree::createRenderDescriptorSetLayout<ITransformTreeWithoutNormalMatrices>(device,stageAccessFlags);
		}

		// the creation is the same as that of a `video::CPropertyPool`
		template<typename... Args>
		static inline core::smart_refctd_ptr<ITransformTreeWithoutNormalMatrices> create(video::ILogicalDevice* device, Args... args)
		{
			core::smart_refctd_ptr<property_pool_t> pool;
			core::smart_refctd_ptr<video::IGPUDescriptorSet> poolDS,renderDS;
			if (!ITransformTree::create<ITransformTreeWithoutNormalMatrices,Args...>(pool,poolDS,renderDS,device,std::forward<Args>(args)...))
				return nullptr;

			auto* ttRaw = new ITransformTreeWithoutNormalMatrices(std::move(pool),std::move(poolDS),std::move(renderDS));
			return core::smart_refctd_ptr<ITransformTreeWithoutNormalMatrices>(ttRaw,core::dont_grab);
		}
		
		//
		static constexpr inline bool HasNormalMatrices = false;
		inline bool hasNormalMatrices() const override {return false;}
		
		//
		static constexpr inline uint32_t RenderDescriptorSetBindingCount = 1u;
		inline uint32_t getRenderDescriptorSetBindingCount() const override {return RenderDescriptorSetBindingCount;}

		//
		inline const video::IPropertyPool* getNodePropertyPool() const override {return m_nodeStorage.get();}

	protected:
		ITransformTreeWithoutNormalMatrices(core::smart_refctd_ptr<property_pool_t>&& _nodeStorage, core::smart_refctd_ptr<video::IGPUDescriptorSet>&& _transformHierarchyDS, core::smart_refctd_ptr<video::IGPUDescriptorSet>&& _renderDS);
		inline ~ITransformTreeWithoutNormalMatrices() {} // everything drops itself automatically


		core::smart_refctd_ptr<property_pool_t> m_nodeStorage;
};

class ITransformTreeWithNormalMatrices : public ITransformTree
{
	public:
		struct normal_matrix_t
		{
			uint32_t compressedComponents[4];
		};

		using property_pool_t = video::CPropertyPool<core::allocator,
			parent_t,
			relative_transform_t,modified_stamp_t,
			global_transform_t,recomputed_stamp_t,
			normal_matrix_t
		>;
		static inline constexpr uint32_t normal_matrix_prop_ix = 5u;

		//
		static inline core::smart_refctd_ptr<asset::ICPUDescriptorSetLayout> createPoolDescriptorSetLayout(asset::IShader::E_SHADER_STAGE* stageAccessFlags=nullptr)
		{
			return ITransformTree::createPoolDescriptorSetLayout<ITransformTreeWithNormalMatrices>(stageAccessFlags);
		}
		static inline core::smart_refctd_ptr<video::IGPUDescriptorSetLayout> createPoolDescriptorSetLayout(video::ILogicalDevice* device, asset::IShader::E_SHADER_STAGE* stageAccessFlags=nullptr)
		{
			return ITransformTree::createPoolDescriptorSetLayout<ITransformTreeWithNormalMatrices>(device,stageAccessFlags);
		}
		static inline core::smart_refctd_ptr<asset::ICPUDescriptorSetLayout> createRenderDescriptorSetLayout(asset::IShader::E_SHADER_STAGE* stageAccessFlags=nullptr)
		{
			return ITransformTree::createRenderDescriptorSetLayout<ITransformTreeWithNormalMatrices>(stageAccessFlags);
		}
		static inline core::smart_refctd_ptr<video::IGPUDescriptorSetLayout> createRenderDescriptorSetLayout(video::ILogicalDevice* device, asset::IShader::E_SHADER_STAGE* stageAccessFlags=nullptr)
		{
			return ITransformTree::createRenderDescriptorSetLayout<ITransformTreeWithNormalMatrices>(device,stageAccessFlags);
		}

		// the creation is the same as that of a `video::CPropertyPool`
		template<typename... Args>
		static inline core::smart_refctd_ptr<ITransformTreeWithNormalMatrices> create(video::ILogicalDevice* device, Args... args)
		{
			core::smart_refctd_ptr<property_pool_t> pool;
			core::smart_refctd_ptr<video::IGPUDescriptorSet> poolDS,renderDS;
			if (!ITransformTree::create<ITransformTreeWithNormalMatrices,Args...>(pool,poolDS,renderDS,device,std::forward<Args>(args)...))
				return nullptr;

			auto* ttRaw = new ITransformTreeWithNormalMatrices(std::move(pool),std::move(poolDS),std::move(renderDS));
			return core::smart_refctd_ptr<ITransformTreeWithNormalMatrices>(ttRaw,core::dont_grab);
		}
		
		//
		static constexpr inline bool HasNormalMatrices = true;
		inline bool hasNormalMatrices() const override {return true;}

		//
		static constexpr inline uint32_t RenderDescriptorSetBindingCount = 2u;
		inline uint32_t getRenderDescriptorSetBindingCount() const override { return RenderDescriptorSetBindingCount; }

		//
		inline const video::IPropertyPool* getNodePropertyPool() const override {return m_nodeStorage.get();}

	protected:
		ITransformTreeWithNormalMatrices(core::smart_refctd_ptr<property_pool_t>&& _nodeStorage, core::smart_refctd_ptr<video::IGPUDescriptorSet>&& _transformHierarchyDS, core::smart_refctd_ptr<video::IGPUDescriptorSet>&& _renderDS);
		~ITransformTreeWithNormalMatrices() {} // everything drops itself automatically


		core::smart_refctd_ptr<property_pool_t> m_nodeStorage;
};
#endif

} // end namespace nbl::scene

#endif

