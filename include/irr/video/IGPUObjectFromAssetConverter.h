// Do not include this in headers, please
#ifndef __IRR_I_GPU_OBJECT_FROM_ASSET_CONVERTER_H_INCLUDED__
#define __IRR_I_GPU_OBJECT_FROM_ASSET_CONVERTER_H_INCLUDED__

#include "irr/core/core.h"
#include "irr/asset/asset.h"

#include "IDriver.h"
#include "IDriverMemoryBacked.h"
#include "irr/video/CGPUMesh.h"
#include "irr/video/CGPUSkinnedMesh.h"
#include "CLogger.h"
#include "irr/video/asset_traits.h"
#include "irr/core/alloc/LinearAddressAllocator.h"
#include "irr/video/IGPUPipelineCache.h"

namespace irr
{
namespace video
{

class IGPUObjectFromAssetConverter
{
    public:
        struct SParams
        {
            IGPUPipelineCache* pipelineCache = nullptr;
        };

	protected:
		asset::IAssetManager* m_assetManager;
		video::IDriver* m_driver;

		template<typename AssetType, typename iterator_type>
		struct get_asset_raw_ptr
		{
			static inline AssetType* value(iterator_type it) { return static_cast<AssetType*>(it->get()); }
		};

		template<typename AssetType>
		struct get_asset_raw_ptr<AssetType, AssetType**>
		{
			static inline AssetType* value(AssetType** it) { return *it; }
		};
		template<typename AssetType>
		struct get_asset_raw_ptr<AssetType, AssetType*const*>
		{
			static inline AssetType* value(AssetType*const* it) { return *it; }
		};

	public:
		IGPUObjectFromAssetConverter(asset::IAssetManager* _assetMgr, video::IDriver* _drv) : m_assetManager{_assetMgr}, m_driver{_drv} {}

		virtual ~IGPUObjectFromAssetConverter() = default;

		inline virtual created_gpu_object_array<asset::ICPUBuffer>				            create(asset::ICPUBuffer** const _begin, asset::ICPUBuffer** const _end, const SParams& _params);
		inline virtual created_gpu_object_array<asset::ICPUMeshBuffer>			            create(asset::ICPUMeshBuffer** const _begin, asset::ICPUMeshBuffer** const _end, const SParams& _params);
		inline virtual created_gpu_object_array<asset::ICPUMesh>				            create(asset::ICPUMesh** const _begin, asset::ICPUMesh** const _end, const SParams& _params);
		inline virtual created_gpu_object_array<asset::ICPUImage>				            create(asset::ICPUImage** const _begin, asset::ICPUImage** const _end, const SParams& _params);
        inline virtual created_gpu_object_array<asset::ICPUShader>				            create(asset::ICPUShader** const _begin, asset::ICPUShader** const _end, const SParams& _params);
        inline virtual created_gpu_object_array<asset::ICPUSpecializedShader>	            create(asset::ICPUSpecializedShader** const _begin, asset::ICPUSpecializedShader** const _end, const SParams& _params);
        inline virtual created_gpu_object_array<asset::ICPUBufferView>		                create(asset::ICPUBufferView** const _begin, asset::ICPUBufferView** const _end, const SParams& _params);
        inline virtual created_gpu_object_array<asset::ICPUDescriptorSetLayout>             create(asset::ICPUDescriptorSetLayout** const _begin, asset::ICPUDescriptorSetLayout** const _end, const SParams& _params);
        inline virtual created_gpu_object_array<asset::ICPUSampler>		                    create(asset::ICPUSampler** const _begin, asset::ICPUSampler** const _end, const SParams& _params);
        inline virtual created_gpu_object_array<asset::ICPUPipelineLayout>		            create(asset::ICPUPipelineLayout** const _begin, asset::ICPUPipelineLayout** const _end, const SParams& _params);
        inline virtual created_gpu_object_array<asset::ICPURenderpassIndependentPipeline>	create(asset::ICPURenderpassIndependentPipeline** const _begin, asset::ICPURenderpassIndependentPipeline** const _end, const SParams& _params);
        inline virtual created_gpu_object_array<asset::ICPUImageView>				        create(asset::ICPUImageView** const _begin, asset::ICPUImageView** const _end, const SParams& _params);
        inline virtual created_gpu_object_array<asset::ICPUDescriptorSet>				    create(asset::ICPUDescriptorSet** const _begin, asset::ICPUDescriptorSet** const _end, const SParams& _params);
        inline virtual created_gpu_object_array<asset::ICPUComputePipeline>				    create(asset::ICPUComputePipeline** const _begin, asset::ICPUComputePipeline** const _end, const SParams& _params);

		template<typename AssetType, typename iterator_type>
        created_gpu_object_array<AssetType> getGPUObjectsFromAssets(iterator_type _begin, iterator_type _end, const SParams& _params = {})
		{
			const auto assetCount = std::distance(_begin, _end);
			auto res = core::make_refctd_dynamic_array<created_gpu_object_array<AssetType> >(assetCount);

			core::vector<AssetType*> notFound; notFound.reserve(assetCount);
			core::vector<size_t> pos; pos.reserve(assetCount);

			for (iterator_type it = _begin; it != _end; it++)
			{
				const auto index = std::distance(_begin, it);

				//if (*it)
				//{
					auto gpu = m_assetManager->findGPUObject(get_asset_raw_ptr<AssetType, iterator_type>::value(it));
					if (!gpu)
					{
						if ((*it)->isADummyObjectForCache())
							notFound.push_back(nullptr);
						else
							notFound.push_back(get_asset_raw_ptr<AssetType, iterator_type>::value(it));
						pos.push_back(index);
					}
					else
						res->operator[](index) = core::move_and_dynamic_cast<typename video::asset_traits<AssetType>::GPUObjectType>(gpu);
				//}
				//res->operator[](index) = nullptr;
			}

			if (notFound.size())
			{
				decltype(res) created = create(notFound.data(), notFound.data()+notFound.size(), _params);
				for (size_t i=0u; i<created->size(); ++i)
				{
					auto& input = created->operator[](i);
					if (notFound[i])
						m_assetManager->convertAssetToEmptyCacheHandle(notFound[i], core::smart_refctd_ptr(input));
					res->operator[](pos[i]) = std::move(input); // ok to move because the `created` array will die after the next scope
				}
			}

			return res;
		}

	protected:
		//! TODO: Make this faster and not call any allocator
		template<typename T>
		static inline core::vector<size_t> eliminateDuplicatesAndGenRedirs(core::vector<T*>& _input)
		{
			core::vector<size_t> redirs;

			core::unordered_map<T*, size_t> firstOccur;
			size_t i = 0u;
			for (T* el : _input)
			{
				if (!el)
				{
					redirs.push_back(0xdeadbeefu);
					continue;
				}

				auto r = firstOccur.insert({ el, i });
				redirs.push_back(r.first->second);
				if (r.second)
					++i;
			}
			for (const auto& p : firstOccur)
				_input.push_back(p.first);
			_input.erase(_input.begin(), _input.begin() + (_input.size() - firstOccur.size()));
			std::sort(_input.begin(), _input.end(), [&firstOccur](T* a, T* b) { return firstOccur[a] < firstOccur[b]; });

			return redirs;
		}
};

auto IGPUObjectFromAssetConverter::create(asset::ICPUBuffer** const _begin, asset::ICPUBuffer** const _end, const SParams& _params) -> created_gpu_object_array<asset::ICPUBuffer> // TODO: improve for caches of very large buffers!!!
{
	const auto assetCount = std::distance(_begin, _end);
	auto res = core::make_refctd_dynamic_array<created_gpu_object_array<asset::ICPUBuffer> >(assetCount);


    const uint64_t alignment =
        std::max(
            std::max(m_driver->getRequiredTBOAlignment(), m_driver->getRequiredUBOAlignment()),
            std::max(m_driver->getRequiredSSBOAlignment(), _IRR_SIMD_ALIGNMENT)
        );


    core::LinearAddressAllocator<uint64_t> addrAllctr(nullptr, 0u, 0u, alignment, m_driver->getMaxBufferSize());
    uint64_t addr = 0ull;
	for (auto i=0u; i<assetCount; i++)
    {
        const uint64_t addr = addrAllctr.alloc_addr(_begin[i]->getSize(), alignment);
        assert(addr != decltype(addrAllctr)::invalid_address); // fix this to work better with really large buffers in the future
        if (addr == decltype(addrAllctr)::invalid_address)
            return {};
		res->operator[](i) = core::make_smart_refctd_ptr<typename video::asset_traits<asset::ICPUBuffer>::GPUObjectType>(addr);
    }

    auto reqs = m_driver->getDeviceLocalGPUMemoryReqs();
    reqs.vulkanReqs.size = addrAllctr.get_allocated_size();
    reqs.vulkanReqs.alignment = alignment;

	auto gpubuffer = m_driver->createGPUBufferOnDedMem(reqs);

    for (size_t i = 0u; i < res->size(); ++i)
    {
		auto& output = res->operator[](i);
		if (!output)
			continue;

		m_driver->updateBufferRangeViaStagingBuffer(gpubuffer.get(), output->getOffset(), _begin[i]->getSize(), _begin[i]->getPointer());
        output->setBuffer(core::smart_refctd_ptr(gpubuffer));
    }

    return res;
}
auto IGPUObjectFromAssetConverter::create(asset::ICPUMeshBuffer** _begin, asset::ICPUMeshBuffer** _end, const SParams& _params) -> created_gpu_object_array<asset::ICPUMeshBuffer>
{
	const size_t assetCount = std::distance(_begin, _end);
	auto res = core::make_refctd_dynamic_array<created_gpu_object_array<asset::ICPUMeshBuffer> >(assetCount);

    core::vector<asset::ICPURenderpassIndependentPipeline*> cpuPipelines;
    cpuPipelines.reserve(assetCount);
    core::vector<asset::ICPUBuffer*> cpuBuffers;
    cpuBuffers.reserve(assetCount * (asset::ICPUMeshBuffer::MAX_ATTR_BUF_BINDING_COUNT+1u));
    core::vector<asset::ICPUDescriptorSet*> cpuDescSets;
    cpuDescSets.reserve(assetCount);

    for (ptrdiff_t i = 0u; i < assetCount; ++i)
    {
        asset::ICPUMeshBuffer* cpumb = _begin[i];

        if (cpumb->getPipeline())
            cpuPipelines.push_back(cpumb->getPipeline());
        if (cpumb->getAttachedDescriptorSet())
            cpuDescSets.push_back(cpumb->getAttachedDescriptorSet());

        for (size_t b = 0ull; b < asset::ICPUMeshBuffer::MAX_ATTR_BUF_BINDING_COUNT; ++b)
        {
            if (asset::ICPUBuffer* buf = cpumb->getVertexBufferBindings()[b].buffer.get())
                cpuBuffers.push_back(buf);
        }
        if (asset::ICPUBuffer* buf = cpumb->getIndexBufferBinding()->buffer.get())
            cpuBuffers.push_back(buf);
    }

    using redirs_t = core::vector<size_t>;

    redirs_t pplnRedirs = eliminateDuplicatesAndGenRedirs(cpuPipelines);
    redirs_t bufRedirs = eliminateDuplicatesAndGenRedirs(cpuBuffers);
    redirs_t dsRedirs = eliminateDuplicatesAndGenRedirs(cpuDescSets);

    auto gpuPipelines = getGPUObjectsFromAssets<asset::ICPURenderpassIndependentPipeline>(cpuPipelines.data(), cpuPipelines.data()+cpuPipelines.size(), _params);
    auto gpuBuffers = getGPUObjectsFromAssets<asset::ICPUBuffer>(cpuBuffers.data(), cpuBuffers.data()+cpuBuffers.size(), _params);
    auto gpuDescSets = getGPUObjectsFromAssets<asset::ICPUDescriptorSet>(cpuDescSets.data(), cpuDescSets.data()+cpuDescSets.size(), _params);

    size_t pplnIter = 0ull, bufIter = 0ull, dsIter = 0ull;
    for (ptrdiff_t i = 0u; i < assetCount; ++i)
    {
        asset::ICPUMeshBuffer* cpumb = _begin[i];

        IGPURenderpassIndependentPipeline* gpuppln = nullptr;
        if (cpumb->getPipeline())
            gpuppln = (*gpuPipelines)[pplnRedirs[pplnIter++]].get();
        IGPUDescriptorSet* gpuds = nullptr;
        if (cpumb->getAttachedDescriptorSet())
            gpuds = (*gpuDescSets)[dsRedirs[dsIter++]].get();

        asset::SBufferBinding<IGPUBuffer> vtxBindings[IGPUMeshBuffer::MAX_ATTR_BUF_BINDING_COUNT];
        for (size_t b = 0ull; b < IGPUMeshBuffer::MAX_ATTR_BUF_BINDING_COUNT; ++b)
        {
            const auto& cpubnd = cpumb->getVertexBufferBindings()[b];
            if (cpubnd.buffer) {
                vtxBindings[b].offset = cpubnd.offset;
                auto& gpubuf = (*gpuBuffers)[bufRedirs[bufIter++]];
                vtxBindings[b].offset += gpubuf->getOffset();
                vtxBindings[b].buffer = core::smart_refctd_ptr<IGPUBuffer>(gpubuf->getBuffer());
            }
        }

		asset::SBufferBinding<IGPUBuffer> idxBinding;
        if (cpumb->getIndexBufferBinding()->buffer)
        {
            idxBinding.offset = cpumb->getIndexBufferBinding()->offset;
            auto& gpubuf = (*gpuBuffers)[bufRedirs[bufIter++]];
            idxBinding.offset += gpubuf->getOffset();
            idxBinding.buffer = core::smart_refctd_ptr<IGPUBuffer>(gpubuf->getBuffer());
        }

        core::smart_refctd_ptr<IGPURenderpassIndependentPipeline> gpuppln_(gpuppln);
        core::smart_refctd_ptr<IGPUDescriptorSet> gpuds_(gpuds);
        (*res)[i] = core::make_smart_refctd_ptr<IGPUMeshBuffer>(std::move(gpuppln_), std::move(gpuds_), vtxBindings, std::move(idxBinding));
        (*res)[i]->setBaseInstance(_begin[i]->getBaseInstance());
        (*res)[i]->setBaseVertex(_begin[i]->getBaseVertex());
        (*res)[i]->setIndexCount(_begin[i]->getIndexCount());
        (*res)[i]->setIndexType(_begin[i]->getIndexType());
        (*res)[i]->setInstanceCount(_begin[i]->getInstanceCount());
        memcpy((*res)[i]->getPushConstantsDataPtr(), _begin[i]->getPushConstantsDataPtr(), IGPUMeshBuffer::MAX_PUSH_CONSTANT_BYTESIZE);
        //const core::aabbox3df oldBBox = cpumb->getBoundingBox();
        //if (cpumb->getMeshBufferType() != asset::EMBT_ANIMATED_SKINNED)
        //    cpumb->recalculateBoundingBox();
        (*res)[i]->setBoundingBox(cpumb->getBoundingBox());
        //if (cpumb->getMeshBufferType() != asset::EMBT_ANIMATED_SKINNED)
        //    cpumb->setBoundingBox(oldBBox);
    }

    return res;
}
auto IGPUObjectFromAssetConverter::create(asset::ICPUMesh** const _begin, asset::ICPUMesh** const _end, const SParams& _params) -> created_gpu_object_array<asset::ICPUMesh>
{
	const size_t assetCount = std::distance(_begin, _end);
	auto res = core::make_refctd_dynamic_array<created_gpu_object_array<asset::ICPUMesh> >(assetCount);

	core::vector<asset::ICPUMeshBuffer*> cpuDeps; cpuDeps.reserve(assetCount);
	for (auto i=0u; i<assetCount; i++)
	{
		auto* _asset = _begin[i];
        for (uint32_t i = 0u; i <_asset->getMeshBufferCount(); ++i)
            cpuDeps.push_back(_asset->getMeshBuffer(i));

		auto& output = res->operator[](i);
		switch (_asset->getMeshType())
		{
			case asset::EMT_ANIMATED_SKINNED:
				output = core::make_smart_refctd_ptr<video::CGPUSkinnedMesh>(core::smart_refctd_ptr<asset::CFinalBoneHierarchy>(static_cast<asset::ICPUSkinnedMesh*>(_asset)->getBoneReferenceHierarchy()));
				break;
			default:
				output = core::make_smart_refctd_ptr<video::CGPUMesh>();
				break;
        }
        output->setBoundingBox(_asset->getBoundingBox());
    }

    core::vector<size_t> redir = eliminateDuplicatesAndGenRedirs(cpuDeps);
    auto gpuDeps = getGPUObjectsFromAssets<asset::ICPUMeshBuffer>(cpuDeps.data(), cpuDeps.data() + cpuDeps.size(), _params);
    for (size_t i=0u, j=0u; i<assetCount; ++i)
    {
		auto* _asset = _begin[i];

		auto& output = res->operator[](i);
        switch (output->getMeshType())
        {
			case asset::EMT_ANIMATED_SKINNED:
				for (uint32_t k=0u; k<_asset->getMeshBufferCount(); ++k)
				{
					static_cast<video::CGPUSkinnedMesh*>(output.get())->addMeshBuffer(core::smart_refctd_ptr(gpuDeps->operator[](redir[j])), static_cast<asset::ICPUSkinnedMeshBuffer*>((*(_begin + i))->getMeshBuffer(i))->getMaxVertexBoneInfluences());
					++j;
				}
				break;
			default:
				for (uint32_t k=0u; k<_asset->getMeshBufferCount(); ++k)
				{
					static_cast<video::CGPUMesh*>(output.get())->addMeshBuffer(core::smart_refctd_ptr(gpuDeps->operator[](redir[j])));
					++j;
				}
				break;
        }
    }

    return res;
}

auto IGPUObjectFromAssetConverter::create(asset::ICPUImage** const _begin, asset::ICPUImage** const _end, const SParams& _params) -> created_gpu_object_array<asset::ICPUImage>
{
    const auto assetCount = std::distance(_begin, _end);
    auto res = core::make_refctd_dynamic_array<created_gpu_object_array<asset::ICPUImage> >(assetCount);

    for (ptrdiff_t i = 0u; i < assetCount; ++i)
    {
        const asset::ICPUImage* cpuimg = _begin[i];
        asset::IImage::SCreationParams params = cpuimg->getCreationParameters();
        params.mipLevels = 1u + static_cast<uint32_t>(std::log2(static_cast<float>(core::max(core::max(params.extent.width, params.extent.height), params.extent.depth))));
        auto gpuimg = m_driver->createDeviceLocalGPUImageOnDedMem(std::move(params));

		auto regions = cpuimg->getRegions();
		auto count = regions.length();
		if (count)
		{
			auto tmpBuff = m_driver->createFilledDeviceLocalGPUBufferOnDedMem(cpuimg->getBuffer()->getSize(),cpuimg->getBuffer()->getPointer());
			m_driver->copyBufferToImage(tmpBuff.get(),gpuimg.get(),count,cpuimg->getRegions().begin());
            gpuimg->generateMipmaps();
		}

		res->operator[](i) = std::move(gpuimg);
    }

    return res;
}

auto IGPUObjectFromAssetConverter::create(asset::ICPUShader** const _begin, asset::ICPUShader** const _end, const SParams& _params) -> created_gpu_object_array<asset::ICPUShader>
{
    const auto assetCount = std::distance(_begin, _end);
    auto res = core::make_refctd_dynamic_array<created_gpu_object_array<asset::ICPUShader> >(assetCount);

    for (ptrdiff_t i = 0u; i < assetCount; ++i)
        res->operator[](i) = m_driver->createGPUShader(core::smart_refctd_ptr<const asset::ICPUShader>(_begin[i]));

    return res;
}

auto IGPUObjectFromAssetConverter::create(asset::ICPUSpecializedShader** const _begin, asset::ICPUSpecializedShader** const _end, const SParams& _params) -> created_gpu_object_array<asset::ICPUSpecializedShader>
{
    const auto assetCount = std::distance(_begin, _end);
    auto res = core::make_refctd_dynamic_array<created_gpu_object_array<asset::ICPUSpecializedShader> >(assetCount);

    core::vector<asset::ICPUShader*> cpuDeps;
    cpuDeps.reserve(res->size());

    asset::ICPUSpecializedShader** it = _begin;
    while (it != _end)
    {
        cpuDeps.push_back((*it)->getUnspecialized());
        ++it;
    }

    core::vector<size_t> redirs = eliminateDuplicatesAndGenRedirs(cpuDeps);
    auto gpuDeps = getGPUObjectsFromAssets<asset::ICPUShader>(cpuDeps.data(), cpuDeps.data()+cpuDeps.size(), _params);

    for (ptrdiff_t i = 0; i < assetCount; ++i)
    {
        res->operator[](i) = m_driver->createGPUSpecializedShader(gpuDeps->operator[](redirs[i]).get(), _begin[i]->getSpecializationInfo());
    }

    return res;
}

auto IGPUObjectFromAssetConverter::create(asset::ICPUBufferView** const _begin, asset::ICPUBufferView** const _end, const SParams& _params) -> created_gpu_object_array<asset::ICPUBufferView>
{
    const auto assetCount = std::distance(_begin, _end);
    auto res = core::make_refctd_dynamic_array<created_gpu_object_array<asset::ICPUBufferView> >(assetCount);

    core::vector<asset::ICPUBuffer*> cpuBufs(assetCount, nullptr);
    for (ptrdiff_t i = 0u; i < assetCount; ++i)
        cpuBufs[i] = _begin[i]->getUnderlyingBuffer();

    core::vector<size_t> redirs = eliminateDuplicatesAndGenRedirs(cpuBufs);
    auto gpuBufs = getGPUObjectsFromAssets<asset::ICPUBuffer>(cpuBufs.data(), cpuBufs.data()+cpuBufs.size(), _params);

    for (ptrdiff_t i = 0u; i < assetCount; ++i)
    {
        asset::ICPUBufferView* cpubufview = _begin[i];
        IGPUOffsetBufferPair* gpubuf = (*gpuBufs)[redirs[i]].get();
        (*res)[i] = m_driver->createGPUBufferView(gpubuf->getBuffer(), cpubufview->getFormat(), gpubuf->getOffset() + cpubufview->getOffsetInBuffer(), cpubufview->getByteSize());;
    }

    return res;
}

auto IGPUObjectFromAssetConverter::create(asset::ICPUDescriptorSetLayout** const _begin, asset::ICPUDescriptorSetLayout** const _end, const SParams& _params) -> created_gpu_object_array<asset::ICPUDescriptorSetLayout>
{
    const auto assetCount = std::distance(_begin, _end);
    auto res = core::make_refctd_dynamic_array<created_gpu_object_array<asset::ICPUDescriptorSetLayout> >(assetCount);

    core::vector<asset::ICPUSampler*> cpuSamplers;//immutable samplers
    size_t maxSamplers = 0ull;
    size_t maxBindingsPerDescSet = 0ull;
    size_t maxSamplersPerDescSet = 0u;
    for (auto dsl : core::SRange<asset::ICPUDescriptorSetLayout*>(_begin, _end))
    {
        size_t samplersInDS = 0u;
        for (const auto& bnd : dsl->getBindings()) {
            const uint32_t samplerCnt = bnd.samplers ? bnd.count : 0u;
            maxSamplers += samplerCnt;
            samplersInDS = samplerCnt;
        }
        maxBindingsPerDescSet = std::max(maxBindingsPerDescSet, dsl->getBindings().length());
        maxSamplersPerDescSet = std::max(maxSamplersPerDescSet, samplersInDS);
    }
    cpuSamplers.reserve(maxSamplers);

    for (auto dsl : core::SRange<asset::ICPUDescriptorSetLayout*>(_begin, _end))
    {
        for (const auto& bnd : dsl->getBindings())
        {
            if (bnd.samplers)
            {
                for (uint32_t i = 0u; i < bnd.count; ++i)
                    cpuSamplers.push_back(bnd.samplers[i].get());
            }
        }
    }

    core::vector<size_t> redirs = eliminateDuplicatesAndGenRedirs(cpuSamplers);
    auto gpuSamplers = getGPUObjectsFromAssets<asset::ICPUSampler>(cpuSamplers.data(), cpuSamplers.data()+cpuSamplers.size(), _params);
    size_t gpuSmplrIter = 0ull;

    using gpu_bindings_array_t = core::smart_refctd_dynamic_array<IGPUDescriptorSetLayout::SBinding>;
    auto tmpBindings = core::make_refctd_dynamic_array<gpu_bindings_array_t>(maxBindingsPerDescSet);
    using samplers_array_t = core::smart_refctd_dynamic_array<core::smart_refctd_ptr<IGPUSampler>>;
    auto tmpSamplers = core::make_refctd_dynamic_array<samplers_array_t>(maxSamplersPerDescSet*maxBindingsPerDescSet);
    for (ptrdiff_t i = 0u; i < assetCount; ++i)
    {
        core::smart_refctd_ptr<IGPUSampler>* smplr_ptr = tmpSamplers->data();
        asset::ICPUDescriptorSetLayout* cpudsl = _begin[i];
        size_t bndIter = 0ull;
        for (const auto& bnd : cpudsl->getBindings())
        {
            IGPUDescriptorSetLayout::SBinding gpubnd;
            gpubnd.binding = bnd.binding;
            gpubnd.type = bnd.type;
            gpubnd.count = bnd.count;
            gpubnd.stageFlags = bnd.stageFlags;
            gpubnd.samplers = nullptr;

            if (bnd.samplers)
            {
                for (uint32_t s = 0u; s < gpubnd.count; ++s)
                    smplr_ptr[s] = (*gpuSamplers)[redirs[gpuSmplrIter++]];
                gpubnd.samplers = smplr_ptr;
                smplr_ptr += gpubnd.count;
            }
            (*tmpBindings)[bndIter++] = gpubnd;
        }
        (*res)[i] = m_driver->createGPUDescriptorSetLayout((*tmpBindings).data(), (*tmpBindings).data()+bndIter);
    }

    return res;
}

inline created_gpu_object_array<asset::ICPUSampler> IGPUObjectFromAssetConverter::create(asset::ICPUSampler** const _begin, asset::ICPUSampler** const _end, const SParams& _params)
{
    const auto assetCount = std::distance(_begin, _end);
    auto res = core::make_refctd_dynamic_array<created_gpu_object_array<asset::ICPUSampler> >(assetCount);

    for (ptrdiff_t i = 0u; i < assetCount; ++i)
    {
        asset::ICPUSampler* cpusmplr = _begin[i];
        res->operator[](i) = m_driver->createGPUSampler(cpusmplr->getParams());
    }

    return res;
}

inline created_gpu_object_array<asset::ICPUPipelineLayout> IGPUObjectFromAssetConverter::create(asset::ICPUPipelineLayout** const _begin, asset::ICPUPipelineLayout** const _end, const SParams& _params)
{
    const auto assetCount = std::distance(_begin, _end);
    auto res = core::make_refctd_dynamic_array<created_gpu_object_array<asset::ICPUPipelineLayout> >(assetCount);

	// TODO: Deal with duplication of layouts and any other resource that can be present at different resource tree levels
	// SOLUTION: a `creationCache` object as the last parameter to the `create` function
    core::vector<asset::ICPUDescriptorSetLayout*> cpuDSLayouts;
    cpuDSLayouts.reserve(assetCount*asset::ICPUPipelineLayout::DESCRIPTOR_SET_COUNT);

    for (asset::ICPUPipelineLayout* pl : core::SRange<asset::ICPUPipelineLayout*>(_begin, _end))
    {
        for (uint32_t ds = 0u; ds < asset::ICPUPipelineLayout::DESCRIPTOR_SET_COUNT; ++ds)
            if (pl->getDescriptorSetLayout(ds))
                cpuDSLayouts.push_back(pl->getDescriptorSetLayout(ds));
    }
    core::vector<size_t> redirs = eliminateDuplicatesAndGenRedirs(cpuDSLayouts);

    auto gpuDSLayouts = getGPUObjectsFromAssets<asset::ICPUDescriptorSetLayout>(cpuDSLayouts.data(), cpuDSLayouts.data()+cpuDSLayouts.size(), _params);

    size_t dslIter = 0ull;
    for (ptrdiff_t i = 0u; i < assetCount; ++i)
    {
        asset::ICPUPipelineLayout* cpupl = _begin[i];
        IGPUDescriptorSetLayout* dsLayouts[asset::ICPUPipelineLayout::DESCRIPTOR_SET_COUNT]{};
        for (size_t ds = 0ull; ds < asset::ICPUPipelineLayout::DESCRIPTOR_SET_COUNT; ++ds)
        {
            if (cpupl->getDescriptorSetLayout(ds))
                dsLayouts[ds++] = (*gpuDSLayouts)[redirs[dslIter++]].get();
        }
        res->operator[](i) = m_driver->createGPUPipelineLayout(
            cpupl->getPushConstantRanges().begin(), cpupl->getPushConstantRanges().end(),
            core::smart_refctd_ptr<IGPUDescriptorSetLayout>(dsLayouts[0]),
            core::smart_refctd_ptr<IGPUDescriptorSetLayout>(dsLayouts[1]),
            core::smart_refctd_ptr<IGPUDescriptorSetLayout>(dsLayouts[2]),
            core::smart_refctd_ptr<IGPUDescriptorSetLayout>(dsLayouts[3])
        );
    }

    return res;
}

inline created_gpu_object_array<asset::ICPURenderpassIndependentPipeline> IGPUObjectFromAssetConverter::create(asset::ICPURenderpassIndependentPipeline** const _begin, asset::ICPURenderpassIndependentPipeline** const _end, const SParams& _params)
{
    constexpr size_t SHADER_STAGE_COUNT = asset::ICPURenderpassIndependentPipeline::SHADER_STAGE_COUNT;

    const auto assetCount = std::distance(_begin, _end);
    auto res = core::make_refctd_dynamic_array<created_gpu_object_array<asset::ICPURenderpassIndependentPipeline> >(assetCount);

    core::vector<asset::ICPUPipelineLayout*> cpuLayouts;
    cpuLayouts.reserve(assetCount);
    core::vector<asset::ICPUSpecializedShader*> cpuShaders;
    cpuShaders.reserve(assetCount*SHADER_STAGE_COUNT);

    for (ptrdiff_t i = 0u; i < assetCount; ++i)
    {
        asset::ICPURenderpassIndependentPipeline* cpuppln = _begin[i];
        cpuLayouts.push_back(cpuppln->getLayout());

        for (size_t s = 0ull; s < SHADER_STAGE_COUNT; ++s)
            if (asset::ICPUSpecializedShader* shdr = cpuppln->getShaderAtIndex(static_cast<asset::ICPURenderpassIndependentPipeline::E_SHADER_STAGE_IX>(s)))
                cpuShaders.push_back(shdr);
    }

    core::vector<size_t> layoutRedirs = eliminateDuplicatesAndGenRedirs(cpuLayouts);
    core::vector<size_t> shdrRedirs = eliminateDuplicatesAndGenRedirs(cpuShaders);

    auto gpuLayouts = getGPUObjectsFromAssets<asset::ICPUPipelineLayout>(cpuLayouts.data(), cpuLayouts.data()+cpuLayouts.size(), _params);
    auto gpuShaders = getGPUObjectsFromAssets<asset::ICPUSpecializedShader>(cpuShaders.data(), cpuShaders.data()+cpuShaders.size(), _params);

    size_t shdrIter = 0ull;
    for (ptrdiff_t i = 0u; i < assetCount; ++i)
    {
        asset::ICPURenderpassIndependentPipeline* cpuppln = _begin[i];

        IGPUPipelineLayout* layout = (*gpuLayouts)[layoutRedirs[i]].get();

        IGPUSpecializedShader* shaders[SHADER_STAGE_COUNT]{};
        size_t local_shdr_count = 0ull;
        for (size_t s = 0ull; s < SHADER_STAGE_COUNT; ++s)
            if (cpuppln->getShaderAtIndex(static_cast<asset::ICPURenderpassIndependentPipeline::E_SHADER_STAGE_IX>(s)))
                shaders[local_shdr_count++] = (*gpuShaders)[shdrRedirs[shdrIter++]].get();

        (*res)[i] = m_driver->createGPURenderpassIndependentPipeline(
            _params.pipelineCache,
            core::smart_refctd_ptr<IGPUPipelineLayout>(layout),
            shaders, shaders+local_shdr_count,
            cpuppln->getVertexInputParams(),
            cpuppln->getBlendParams(),
            cpuppln->getPrimitiveAssemblyParams(),
            cpuppln->getRasterizationParams()
        );
    }

    return res;
}

inline created_gpu_object_array<asset::ICPUImageView> IGPUObjectFromAssetConverter::create(asset::ICPUImageView** const _begin, asset::ICPUImageView** const _end, const SParams& _params)
{
    const auto assetCount = std::distance(_begin, _end);
    auto res = core::make_refctd_dynamic_array<created_gpu_object_array<asset::ICPUImageView> >(assetCount);

    core::vector<asset::ICPUImage*> cpuDeps;
    cpuDeps.reserve(res->size());

    asset::ICPUImageView** it = _begin;
    while (it != _end)
    {
        cpuDeps.push_back((*it)->getCreationParameters().image.get());
        ++it;
    }

    core::vector<size_t> redirs = eliminateDuplicatesAndGenRedirs(cpuDeps);
    auto gpuDeps = getGPUObjectsFromAssets<asset::ICPUImage>(cpuDeps.data(), cpuDeps.data() + cpuDeps.size(), _params);

    for (ptrdiff_t i = 0; i < assetCount; ++i)
    {
        const asset::ICPUImageView::SCreationParams& cpuparams = _begin[i]->getCreationParameters();
        IGPUImageView::SCreationParams params;
        memcpy(&params.components, &cpuparams.components, sizeof(params.components));
        params.flags = static_cast<IGPUImageView::E_CREATE_FLAGS>(cpuparams.flags);
        params.format = cpuparams.format;
        params.subresourceRange = cpuparams.subresourceRange;
        params.subresourceRange.levelCount = (*gpuDeps)[redirs[i]]->getCreationParameters().mipLevels - params.subresourceRange.baseMipLevel;
        params.viewType = static_cast<IGPUImageView::E_TYPE>(cpuparams.viewType);
        params.image = (*gpuDeps)[redirs[i]];
        (*res)[i] = m_driver->createGPUImageView(std::move(params));
    }

    return res;
}

inline created_gpu_object_array<asset::ICPUDescriptorSet> IGPUObjectFromAssetConverter::create(asset::ICPUDescriptorSet** const _begin, asset::ICPUDescriptorSet** const _end, const SParams& _params)
{
    const auto assetCount = std::distance(_begin, _end);
    auto res = core::make_refctd_dynamic_array<created_gpu_object_array<asset::ICPUDescriptorSet> >(assetCount);

    struct BindingDescTypePair_t{
        uint32_t binding;
        asset::E_DESCRIPTOR_TYPE descType;
        size_t count;
    };
    auto isBufferDesc = [](asset::E_DESCRIPTOR_TYPE t) {
        using namespace asset;
        switch (t) {
        case EDT_UNIFORM_BUFFER: _IRR_FALLTHROUGH;
        case EDT_STORAGE_BUFFER: _IRR_FALLTHROUGH;
        case EDT_UNIFORM_BUFFER_DYNAMIC: _IRR_FALLTHROUGH;
        case EDT_STORAGE_BUFFER_DYNAMIC:
            return true;
            break;
        default:
            return false;
            break;
        }
    };
    auto isBufviewDesc = [](asset::E_DESCRIPTOR_TYPE t) {
        using namespace asset;
        return t==EDT_STORAGE_TEXEL_BUFFER || t==EDT_STORAGE_TEXEL_BUFFER;
    };
    auto isSampledImgViewDesc = [](asset::E_DESCRIPTOR_TYPE t) {
        return t==asset::EDT_COMBINED_IMAGE_SAMPLER;
    };
    auto isStorageImgDesc = [](asset::E_DESCRIPTOR_TYPE t) {
        return t==asset::EDT_STORAGE_IMAGE;
    };

	// TODO: Deal with duplication of layouts and any other resource that can be present at different resource tree levels
	core::vector<asset::ICPUDescriptorSetLayout*> cpuLayouts;
	cpuLayouts.reserve(assetCount);
	uint32_t writeCount = 0ull;
	uint32_t descCount = 0ull;
	uint32_t bufCount = 0ull;
	uint32_t bufviewCount = 0ull;
	uint32_t sampledImgViewCount = 0ull;
	uint32_t storageImgViewCount = 0ull;
    for (ptrdiff_t i=0u; i<assetCount; i++)
    {
        asset::ICPUDescriptorSet* cpuds = _begin[i];
		cpuLayouts.push_back(cpuds->getLayout());
              
		for (auto j=0u; j<=cpuds->getMaxDescriptorBindingIndex(); j++)
		{
			const uint32_t cnt = cpuds->getDescriptors(j).length();
			if (cnt)
				writeCount++;
			descCount += cnt;

			const auto type = cpuds->getDescriptorsType(j);
			if (isBufferDesc(type))
				bufCount += cnt;
			else if (isBufviewDesc(type))
				bufviewCount += cnt;
			else if (isSampledImgViewDesc(type))
				sampledImgViewCount += cnt;
			else if (isStorageImgDesc(type))
				storageImgViewCount += cnt;
		}
    }
	
    core::vector<asset::ICPUBuffer*> cpuBuffers;
    cpuBuffers.reserve(bufCount);
    core::vector<asset::ICPUBufferView*> cpuBufviews;
    cpuBufviews.reserve(bufviewCount);
    core::vector<asset::ICPUImageView*> cpuImgViews;
    cpuImgViews.reserve(storageImgViewCount+sampledImgViewCount);
    core::vector<asset::ICPUSampler*> cpuSamplers;
    cpuSamplers.reserve(sampledImgViewCount);
    for (ptrdiff_t i=0u; i<assetCount; i++)
    {
        asset::ICPUDescriptorSet* cpuds = _begin[i];              
		for (auto j=0u; j<=cpuds->getMaxDescriptorBindingIndex(); j++)
		{
			const auto type = cpuds->getDescriptorsType(j);
			for (const auto& info : cpuds->getDescriptors(j))
			{
				if (isBufferDesc(type))
					cpuBuffers.push_back(static_cast<asset::ICPUBuffer*>(info.desc.get()));
				else if (isBufviewDesc(type))
					cpuBufviews.push_back(static_cast<asset::ICPUBufferView*>(info.desc.get()));
				else if (isSampledImgViewDesc(type))
				{
					cpuImgViews.push_back(static_cast<asset::ICPUImageView*>(info.desc.get()));
                    if (info.image.sampler)
					    cpuSamplers.push_back(info.image.sampler.get());
				}
				else if (isStorageImgDesc(type))
					cpuImgViews.push_back(static_cast<asset::ICPUImageView*>(info.desc.get()));
			}
		}
    }

	using redirs_t = core::vector<size_t>;
	redirs_t layoutRedirs = eliminateDuplicatesAndGenRedirs(cpuLayouts);
	redirs_t bufRedirs = eliminateDuplicatesAndGenRedirs(cpuBuffers);
	redirs_t bufviewRedirs = eliminateDuplicatesAndGenRedirs(cpuBufviews);
	redirs_t imgViewRedirs = eliminateDuplicatesAndGenRedirs(cpuImgViews);
	redirs_t smplrRedirs = eliminateDuplicatesAndGenRedirs(cpuSamplers);

	auto gpuLayouts = getGPUObjectsFromAssets<asset::ICPUDescriptorSetLayout>(cpuLayouts.data(), cpuLayouts.data()+cpuLayouts.size(), _params);
    auto gpuBuffers = getGPUObjectsFromAssets<asset::ICPUBuffer>(cpuBuffers.data(), cpuBuffers.data()+cpuBuffers.size(), _params);
    auto gpuBufviews = getGPUObjectsFromAssets<asset::ICPUBufferView>(cpuBufviews.data(), cpuBufviews.data()+cpuBufviews.size(), _params);
    auto gpuImgViews = getGPUObjectsFromAssets<asset::ICPUImageView>(cpuImgViews.data(), cpuImgViews.data()+cpuImgViews.size(), _params);
    auto gpuSamplers = getGPUObjectsFromAssets<asset::ICPUSampler>(cpuSamplers.data(), cpuSamplers.data()+cpuSamplers.size(), _params);

	core::vector<IGPUDescriptorSet::SWriteDescriptorSet> writes(writeCount);
	core::vector<IGPUDescriptorSet::SDescriptorInfo> descInfos(descCount);
	{
		auto write = writes.begin();
		auto info = descInfos.begin();
		//iterators
		uint32_t bi = 0u, bvi = 0u, ivi = 0u, si = 0u;
		for (ptrdiff_t i = 0u; i < assetCount; i++)
		{
			IGPUDescriptorSetLayout* gpulayout = gpuLayouts->operator[](layoutRedirs[i]).get();
			res->operator[](i) = m_driver->createGPUDescriptorSet(core::smart_refctd_ptr<IGPUDescriptorSetLayout>(gpulayout));
			auto gpuds = res->operator[](i).get();

			asset::ICPUDescriptorSet* cpuds = _begin[i];
			for (uint32_t j=0u; j<=cpuds->getMaxDescriptorBindingIndex(); j++)
			{
				auto descriptors = cpuds->getDescriptors(j);
				if (descriptors.length()==0u)
					continue;

				const auto type = cpuds->getDescriptorsType(j);
				write->dstSet = gpuds;
				write->binding = j;
				write->arrayElement = 0;
				write->count = descriptors.length();
				write->descriptorType = type;
				write->info = &(*info);
				write++;
				for (const auto& desc : descriptors)
				{
					if (isBufferDesc(type))
					{
						auto buffer = gpuBuffers->operator[](bufRedirs[bi++]);
						info->desc = core::smart_refctd_ptr<video::IGPUBuffer>(buffer->getBuffer());
						info->buffer.offset = desc.buffer.offset + buffer->getOffset();
						info->buffer.size = desc.buffer.size;
					}
					else if (isBufviewDesc(type))
						info->desc = gpuBufviews->operator[](bufviewRedirs[bvi++]);
					else if (isSampledImgViewDesc(type) || isStorageImgDesc(type))
					{
						info->desc = gpuImgViews->operator[](imgViewRedirs[ivi++]);
						info->image.imageLayout = desc.image.imageLayout;
						if (isSampledImgViewDesc(type) && desc.image.sampler)
							info->image.sampler = gpuSamplers->operator[](smplrRedirs[si++]);
					}
					info++;
				}
			}
		}
	}

	m_driver->updateDescriptorSets(writes.size(), writes.data(), 0u, nullptr);

    return res;
}

inline created_gpu_object_array<asset::ICPUComputePipeline> IGPUObjectFromAssetConverter::create(asset::ICPUComputePipeline ** const _begin, asset::ICPUComputePipeline ** const _end, const SParams& _params)
{
    const auto assetCount = std::distance(_begin, _end);
    auto res = core::make_refctd_dynamic_array<created_gpu_object_array<asset::ICPUComputePipeline> >(assetCount);

    core::vector<asset::ICPUPipelineLayout*> cpuLayouts;
    core::vector<asset::ICPUSpecializedShader*> cpuShaders;
    cpuLayouts.reserve(res->size());
    cpuShaders.reserve(res->size());

    asset::ICPUComputePipeline** it = _begin;
    while (it != _end)
    {
        cpuShaders.push_back((*it)->getShader());
        cpuLayouts.push_back((*it)->getLayout());
        ++it;
    }

    core::vector<size_t> shdrRedirs = eliminateDuplicatesAndGenRedirs(cpuShaders);
    core::vector<size_t> layoutRedirs = eliminateDuplicatesAndGenRedirs(cpuLayouts);
    auto gpuShaders = getGPUObjectsFromAssets<asset::ICPUSpecializedShader>(cpuShaders.data(), cpuShaders.data()+cpuShaders.size(), _params);
    auto gpuLayouts = getGPUObjectsFromAssets<asset::ICPUPipelineLayout>(cpuLayouts.data(), cpuLayouts.data()+cpuLayouts.size(), _params);

    for (ptrdiff_t i = 0; i < assetCount; ++i)
    {
        auto layout = (*gpuLayouts)[layoutRedirs[i]];
        auto shdr = (*gpuShaders)[shdrRedirs[i]];
        (*res)[i] = m_driver->createGPUComputePipeline(_params.pipelineCache, std::move(layout), std::move(shdr));
    }

    return res;
}

}}//irr::video

#endif //__IRR_I_GPU_OBJECT_FROM_ASSET_CONVERTER_H_INCLUDED__
