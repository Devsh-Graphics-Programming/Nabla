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

namespace irr
{
namespace video
{

class IGPUObjectFromAssetConverter
{
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

		inline virtual created_gpu_object_array<asset::ICPUBuffer>				            create(asset::ICPUBuffer** const _begin, asset::ICPUBuffer** const _end);
		inline virtual created_gpu_object_array<asset::ICPUMeshBuffer>			            create(asset::ICPUMeshBuffer** const _begin, asset::ICPUMeshBuffer** const _end);
		inline virtual created_gpu_object_array<asset::ICPUMesh>				            create(asset::ICPUMesh** const _begin, asset::ICPUMesh** const _end);
		inline virtual created_gpu_object_array<asset::ICPUTexture>				            create(asset::ICPUTexture** const _begin, asset::ICPUTexture** const _end);
        inline virtual created_gpu_object_array<asset::ICPUShader>				            create(asset::ICPUShader** const _begin, asset::ICPUShader** const _end);
        inline virtual created_gpu_object_array<asset::ICPUSpecializedShader>	            create(asset::ICPUSpecializedShader** const _begin, asset::ICPUSpecializedShader** const _end);
        inline virtual created_gpu_object_array<asset::ICPUBufferView>		                create(asset::ICPUBufferView** const _begin, asset::ICPUBufferView** const _end);
        inline virtual created_gpu_object_array<asset::ICPUDescriptorSetLayout>             create(asset::ICPUDescriptorSetLayout** const _begin, asset::ICPUDescriptorSetLayout** const _end);
        inline virtual created_gpu_object_array<asset::ICPUSampler>		                    create(asset::ICPUSampler** const _begin, asset::ICPUSampler** const _end);
        inline virtual created_gpu_object_array<asset::ICPUPipelineLayout>		            create(asset::ICPUPipelineLayout** const _begin, asset::ICPUPipelineLayout** const _end);
        inline virtual created_gpu_object_array<asset::ICPURenderpassIndependentPipeline>	create(asset::ICPURenderpassIndependentPipeline** const _begin, asset::ICPURenderpassIndependentPipeline** const _end);
        inline virtual created_gpu_object_array<asset::ICPUTextureView>				        create(asset::ICPUTextureView** const _begin, asset::ICPUTextureView** const _end);
        inline virtual created_gpu_object_array<asset::ICPUDescriptorSet>				    create(asset::ICPUDescriptorSet** const _begin, asset::ICPUDescriptorSet** const _end);


		template<typename AssetType, typename iterator_type>
		created_gpu_object_array<AssetType> getGPUObjectsFromAssets(iterator_type _begin, iterator_type _end)
		{
			const auto assetCount = std::distance(_begin, _end);
			auto res = core::make_refctd_dynamic_array<created_gpu_object_array<AssetType> >(assetCount);

			core::vector<AssetType*> notFound; notFound.reserve(assetCount);
			core::vector<size_t> pos; pos.reserve(assetCount);

			for (iterator_type it=_begin; it!=_end; it++)
			{
				const auto index = std::distance(_begin,it);

				auto gpu = m_assetManager->findGPUObject(get_asset_raw_ptr<AssetType, iterator_type>::value(it));
				if (!gpu)
				{
					notFound.push_back(get_asset_raw_ptr<AssetType,iterator_type>::value(it));
					pos.push_back(index);
				}
				else
					res->operator[](index) = core::move_and_dynamic_cast<typename video::asset_traits<AssetType>::GPUObjectType>(gpu);
			}

			if (notFound.size())
			{
				decltype(res) created = create(notFound.data(), notFound.data()+notFound.size());
				for (size_t i=0u; i<created->size(); ++i)
				{
					auto& input = created->operator[](i);
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

auto IGPUObjectFromAssetConverter::create(asset::ICPUBuffer** const _begin, asset::ICPUBuffer** const _end) -> created_gpu_object_array<asset::ICPUBuffer> // TODO: improve for caches of very large buffers!!!
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

    auto gpubuffer = core::smart_refctd_ptr<IGPUBuffer>(m_driver->createGPUBufferOnDedMem(reqs, true), core::dont_grab); // TODO: full smart pointer + streaming staging buffer

    for (size_t i = 0u; i < res->size(); ++i)
    {
		auto& output = res->operator[](i);
		if (!output)
			continue;

        output->setBuffer(core::smart_refctd_ptr(gpubuffer));
        gpubuffer->updateSubRange(video::IDriverMemoryAllocation::MemoryRange(output->getOffset(), _begin[i]->getSize()), _begin[i]->getPointer());
    }

    return res;
}
auto IGPUObjectFromAssetConverter::create(asset::ICPUMeshBuffer** _begin, asset::ICPUMeshBuffer** _end) -> created_gpu_object_array<asset::ICPUMeshBuffer>
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

    auto gpuPipelines = getGPUObjectsFromAssets<asset::ICPURenderpassIndependentPipeline>(cpuPipelines.data(), cpuPipelines.data()+cpuPipelines.size());
    auto gpuBuffers = getGPUObjectsFromAssets<asset::ICPUBuffer>(cpuBuffers.data(), cpuBuffers.data()+cpuBuffers.size());
    auto gpuDescSets = getGPUObjectsFromAssets<asset::ICPUDescriptorSet>(cpuDescSets.data(), cpuDescSets.data()+cpuDescSets.size());

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

        IGPUMeshBuffer::SBufferBinding vtxBindings[IGPUMeshBuffer::MAX_ATTR_BUF_BINDING_COUNT];
        for (size_t b = 0ull; b < IGPUMeshBuffer::MAX_ATTR_BUF_BINDING_COUNT; ++b)
        {
            const asset::ICPUMeshBuffer::SBufferBinding& cpubnd = cpumb->getVertexBufferBindings()[b];
            if (cpubnd.buffer) {
                vtxBindings[b].offset = cpubnd.offset;
                auto& gpubuf = (*gpuBuffers)[bufRedirs[bufIter++]];
                vtxBindings[b].offset += gpubuf->getOffset();
                vtxBindings[b].buffer = core::smart_refctd_ptr<IGPUBuffer>(gpubuf->getBuffer());
            }
        }

        IGPUMeshBuffer::SBufferBinding idxBinding;
        if (cpumb->getIndexBufferBinding()->buffer)
        {
            idxBinding.offset = cpumb->getIndexBufferBinding()->offset;
            auto& gpubuf = (*gpuBuffers)[bufRedirs[bufIter++]];
            idxBinding.offset += gpubuf->getOffset();
            idxBinding.buffer = core::smart_refctd_ptr<IGPUBuffer>(gpubuf->getBuffer());
        }

        (*res)[i] = core::make_smart_refctd_ptr<IGPUMeshBuffer>(std::move(gpuppln), std::move(gpuds), vtxBindings, std::move(idxBinding));
        const core::aabbox3df oldBBox = cpumb->getBoundingBox();
        if (cpumb->getMeshBufferType() != asset::EMBT_ANIMATED_SKINNED)
            cpumb->recalculateBoundingBox();
        (*res)[i]->setBoundingBox(cpumb->getBoundingBox());
        if (cpumb->getMeshBufferType() != asset::EMBT_ANIMATED_SKINNED)
            cpumb->setBoundingBox(oldBBox);
    }

    return res;
}
auto IGPUObjectFromAssetConverter::create(asset::ICPUMesh** const _begin, asset::ICPUMesh** const _end) -> created_gpu_object_array<asset::ICPUMesh>
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
    auto gpuDeps = getGPUObjectsFromAssets<asset::ICPUMeshBuffer>(cpuDeps.data(), cpuDeps.data() + cpuDeps.size());
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

auto IGPUObjectFromAssetConverter::create(asset::ICPUTexture** _begin, asset::ICPUTexture**_end) -> created_gpu_object_array<asset::ICPUTexture>
{
	const auto assetCount = std::distance(_begin, _end);
	auto res = core::make_refctd_dynamic_array<created_gpu_object_array<asset::ICPUTexture> >(assetCount);

	for (auto i=0u; i<assetCount; i++)
    {
        asset::ICPUTexture* cpuTex = _begin[i];

        auto t = m_driver->createGPUTexture(cpuTex->getType(), cpuTex->getSize(), cpuTex->getHighestMip() ? cpuTex->getHighestMip()+1 : 0, cpuTex->getColorFormat());
		if (t)
		{
			for (const asset::CImageData* img : cpuTex->getRanges())
				t->updateSubRegion(img->getColorFormat(), img->getData(), img->getSliceMin(), img->getSliceMax(), img->getSupposedMipLevel(), img->getUnpackAlignment());

			if (cpuTex->getHighestMip()==0 && t->hasMipMaps())
				t->regenerateMipMapLevels(); // todo : Compute Shader mip-mapper necessary after vulkan
		}

        res->operator[](i) = std::move(t);
    }

    return res;
}

auto IGPUObjectFromAssetConverter::create(asset::ICPUShader** const _begin, asset::ICPUShader** const _end) -> created_gpu_object_array<asset::ICPUShader>
{
    const auto assetCount = std::distance(_begin, _end);
    auto res = core::make_refctd_dynamic_array<created_gpu_object_array<asset::ICPUShader> >(assetCount);

    for (ptrdiff_t i = 0u; i < assetCount; ++i)
    {
        asset::ICPUShader* cpushader = _begin[i];
        res->operator[](i) = m_driver->createGPUShader(cpushader);
    }

    return res;
}

auto IGPUObjectFromAssetConverter::create(asset::ICPUSpecializedShader** const _begin, asset::ICPUSpecializedShader** const _end) -> created_gpu_object_array<asset::ICPUSpecializedShader>
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
    auto gpuDeps = getGPUObjectsFromAssets<asset::ICPUShader>(cpuDeps.data(), cpuDeps.data()+cpuDeps.size());

    for (ptrdiff_t i = 0; i < assetCount; ++i)
    {
        res->operator[](i) = m_driver->createGPUSpecializedShader(gpuDeps->operator[](redirs[i]).get(), _begin[i]->getSpecializationInfo());
    }

    return res;
}

auto IGPUObjectFromAssetConverter::create(asset::ICPUBufferView** const _begin, asset::ICPUBufferView** const _end) -> created_gpu_object_array<asset::ICPUBufferView>
{
    const auto assetCount = std::distance(_begin, _end);
    auto res = core::make_refctd_dynamic_array<created_gpu_object_array<asset::ICPUBufferView> >(assetCount);

    core::vector<asset::ICPUBuffer*> cpuBufs(assetCount, nullptr);
    for (ptrdiff_t i = 0u; i < assetCount; ++i)
        cpuBufs[i] = _begin[i]->getUnderlyingBuffer();

    core::vector<size_t> redirs = eliminateDuplicatesAndGenRedirs(cpuBufs);
    auto gpuBufs = getGPUObjectsFromAssets<asset::ICPUBuffer>(cpuBufs.data(), cpuBufs.data()+cpuBufs.size());

    for (ptrdiff_t i = 0u; i < assetCount; ++i)
    {
        asset::ICPUBufferView* cpubufview = _begin[i];
        IGPUOffsetBufferPair* gpubuf = (*gpuBufs)[redirs[i]].get();
        (*res)[i] = m_driver->createGPUBufferView(gpubuf->getBuffer(), cpubufview->getFormat(), gpubuf->getOffset() + cpubufview->getOffsetInBuffer(), cpubufview->getByteSize());;
    }

    return res;
}

auto IGPUObjectFromAssetConverter::create(asset::ICPUDescriptorSetLayout** const _begin, asset::ICPUDescriptorSetLayout** const _end) -> created_gpu_object_array<asset::ICPUDescriptorSetLayout>
{
    const auto assetCount = std::distance(_begin, _end);
    auto res = core::make_refctd_dynamic_array<created_gpu_object_array<asset::ICPUDescriptorSet> >(assetCount);

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
    auto gpuSamplers = getGPUObjectsFromAssets<asset::ICPUSampler>(cpuSamplers.data(), cpuSamplers.data()+cpuSamplers.size());
    size_t gpuSmplrIter = 0ull;

    using gpu_bindings_array_t = core::smart_refctd_dynamic_array<IGPUDescriptorSetLayout::SBinding>;
    auto tmpBindings = core::make_refctd_dynamic_array<gpu_bindings_array_t>(maxBindingsPerDescSet);
    using samplers_array_t = core::smart_refctd_dynamic_array<core::smart_refctd_ptr<IGPUSampler>>;
    auto tmpSamplers = core::make_refctd_dynamic_array<samplers_array_t>(maxSamplersPerDescSet);
    for (ptrdiff_t i = 0u; i < assetCount; ++i)
    {
        core::smart_refctd_ptr<IGPUSampler>* smplr_ptr = tmpSamplers->data();
        asset::ICPUDescriptorSetLayout* cpudsl = _begin[i];
        size_t bndIter = 0ull;
        for (const auto& bnd : cpudsl->getBindings())
        {
            IGPUDescriptorSetLayout::SBinding gpubnd;
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

inline created_gpu_object_array<asset::ICPUSampler> IGPUObjectFromAssetConverter::create(asset::ICPUSampler** const _begin, asset::ICPUSampler** const _end)
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

inline created_gpu_object_array<asset::ICPUPipelineLayout> IGPUObjectFromAssetConverter::create(asset::ICPUPipelineLayout** const _begin, asset::ICPUPipelineLayout** const _end)
{
    const auto assetCount = std::distance(_begin, _end);
    auto res = core::make_refctd_dynamic_array<created_gpu_object_array<asset::ICPUPipelineLayout> >(assetCount);

    core::vector<asset::ICPUDescriptorSetLayout*> cpuDSLayouts;
    cpuDSLayouts.reserve(assetCount*asset::ICPUPipelineLayout::DESCRIPTOR_SET_COUNT);

    for (asset::ICPUPipelineLayout* pl : core::SRange<asset::ICPUPipelineLayout*>(_begin, _end))
    {
        for (uint32_t ds = 0u; ds < asset::ICPUPipelineLayout::DESCRIPTOR_SET_COUNT; ++ds)
            if (pl->getDescriptorSetLayout(ds))
                cpuDSLayouts.push_back(pl->getDescriptorSetLayout(ds));
    }
    core::vector<size_t> redirs = eliminateDuplicatesAndGenRedirs(cpuDSLayouts);

    auto gpuDSLayouts = getGPUObjectsFromAssets<asset::ICPUDescriptorSetLayout>(cpuDSLayouts.data(), cpuDSLayouts.data()+cpuDSLayouts.size());

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

inline created_gpu_object_array<asset::ICPURenderpassIndependentPipeline> IGPUObjectFromAssetConverter::create(asset::ICPURenderpassIndependentPipeline** const _begin, asset::ICPURenderpassIndependentPipeline** const _end)
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

    auto gpuLayouts = getGPUObjectsFromAssets<asset::ICPUPipelineLayout>(cpuLayouts.data(), cpuLayouts.data()+cpuLayouts.size());
    auto gpuShaders = getGPUObjectsFromAssets<asset::ICPUSpecializedShader>(cpuShaders.data(), cpuShaders.data()+cpuShaders.size());

    size_t shdrIter = 0ull;
    for (ptrdiff_t i = 0u; i < assetCount; ++i)
    {
        asset::ICPURenderpassIndependentPipeline* cpuppln = _begin[i];

        IGPUPipelineLayout* layout = (*gpuLayouts)[layoutRedirs[i]].get();

        IGPUSpecializedShader* shaders[SHADER_STAGE_COUNT]{};
        for (size_t s = 0ull; s < SHADER_STAGE_COUNT; ++s)
            if (cpuppln->getShaderAtIndex(static_cast<asset::ICPURenderpassIndependentPipeline::E_SHADER_STAGE_IX>(s)))
                shaders[s] = (*gpuShaders)[shdrRedirs[shdrIter++]].get();

        (*res)[i] = m_driver->createGPURenderpassIndependentPipeline(
            core::smart_refctd_ptr<IGPUPipelineLayout>(layout),
            shaders,
            cpuppln->getVertexInputParams(),
            cpuppln->getBlendParams(),
            cpuppln->getPrimitiveAssemblyParams(),
            cpuppln->getRasterizationParams()
        );
    }

    return res;
}

inline created_gpu_object_array<asset::ICPUTextureView> IGPUObjectFromAssetConverter::create(asset::ICPUTextureView ** const _begin, asset::ICPUTextureView ** const _end)
{
    const auto assetCount = std::distance(_begin, _end);
    //TODO implement!
    return core::make_refctd_dynamic_array<created_gpu_object_array<asset::ICPUTextureView> >(assetCount, nullptr);
}

inline created_gpu_object_array<asset::ICPUDescriptorSet> IGPUObjectFromAssetConverter::create(asset::ICPUDescriptorSet** const _begin, asset::ICPUDescriptorSet** const _end)
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
    auto isTextureDesc = [](asset::E_DESCRIPTOR_TYPE t) {
        return t==asset::EDT_COMBINED_IMAGE_SAMPLER;
    };
    auto isTexviewDesc = [](asset::E_DESCRIPTOR_TYPE t) {
        return t==asset::EDT_STORAGE_IMAGE;
    };

    size_t descCount = 0ull;
    size_t bufCount = 0ull;
    size_t bufviewCount = 0ull;
    size_t texCount = 0ull;
    size_t texviewCount = 0ull;
    for (ptrdiff_t i = 0u; i < assetCount; ++i)
    {
        asset::ICPUDescriptorSet* cpuds = _begin[i];
                
        descCount += cpuds->getDescriptors().length();
        for (const auto& desc : cpuds->getDescriptors())
        {
            const size_t cnt = desc.info->size();
            bufCount += (isBufferDesc(desc.descriptorType) * cnt);
            bufviewCount += (isBufviewDesc(desc.descriptorType) * cnt);
            texCount += (isTextureDesc(desc.descriptorType) * cnt);
            texviewCount += (isTexviewDesc(desc.descriptorType) * cnt);
        }
    }

    core::vector<BindingDescTypePair_t> descInfos;
    descInfos.reserve(descCount);
    core::vector<asset::ICPUBuffer*> cpuBuffers;
    cpuBuffers.reserve(bufCount);
    core::vector<asset::ICPUBufferView*> cpuBufviews;
    cpuBufviews.reserve(bufviewCount);
    core::vector<asset::ICPUTexture*> cpuTextures;
    cpuTextures.reserve(texCount);
    core::vector<asset::ICPUTextureView*> cpuTexviews;
    cpuTexviews.reserve(texviewCount);
    core::vector<asset::ICPUSampler*> cpuSamplers;
    cpuSamplers.reserve(texCount);
    core::vector<asset::ICPUDescriptorSetLayout*> cpuLayouts;
    cpuLayouts.reserve(assetCount);

    for (ptrdiff_t i = 0u; i < assetCount; ++i)
    {
        asset::ICPUDescriptorSet* cpuds = _begin[i];

        for (const auto& desc : cpuds->getDescriptors())
        {
            const auto t = desc.descriptorType;
            descInfos.push_back({desc.binding, t, desc.info->size()});

#define PUSH_DESCRIPTORS(casttype, container) for (auto& info : (*desc.info)) { container.push_back(static_cast<casttype*>(info.desc.get())); }
            if (isBufferDesc(t))
                PUSH_DESCRIPTORS(asset::ICPUBuffer, cpuBuffers)
            else if (isBufviewDesc(t))
                PUSH_DESCRIPTORS(asset::ICPUBufferView, cpuBufviews)
            else if (isTextureDesc(t)) {
                PUSH_DESCRIPTORS(asset::ICPUTexture, cpuTextures)
                for (auto& info : (*desc.info)) {
                    if (asset::ICPUSampler* smplr = info.image.sampler.get())
                        cpuSamplers.push_back(smplr);
                }
            }
            else if (isTexviewDesc(t))
                PUSH_DESCRIPTORS(asset::ICPUTextureView, cpuTexviews)
#undef PUSH_DESCRIPTORS
        }
    }

    using redirs_t = core::vector<size_t>;
    redirs_t bufRedirs = eliminateDuplicatesAndGenRedirs(cpuBuffers);
    redirs_t bufviewRedirs = eliminateDuplicatesAndGenRedirs(cpuBufviews);
    redirs_t texRedirs = eliminateDuplicatesAndGenRedirs(cpuTextures);
    redirs_t texviewRedirs = eliminateDuplicatesAndGenRedirs(cpuTexviews);
    redirs_t smplrRedirs = eliminateDuplicatesAndGenRedirs(cpuSamplers);
    redirs_t layoutRedirs = eliminateDuplicatesAndGenRedirs(cpuLayouts);

    auto gpuBuffers = getGPUObjectsFromAssets<asset::ICPUBuffer>(cpuBuffers.data(), cpuBuffers.data()+cpuBuffers.size());
    auto gpuBufviews = getGPUObjectsFromAssets<asset::ICPUBufferView>(cpuBufviews.data(), cpuBufviews.data()+cpuBufviews.size());
    auto gpuTextures = getGPUObjectsFromAssets<asset::ICPUTexture>(cpuTextures.data(), cpuTextures.data()+cpuTextures.size());
    auto gpuTexviews = getGPUObjectsFromAssets<asset::ICPUTextureView>(cpuTexviews.data(), cpuTexviews.data()+cpuTexviews.size());
    auto gpuSamplers = getGPUObjectsFromAssets<asset::ICPUSampler>(cpuSamplers.data(), cpuSamplers.data()+cpuSamplers.size());
    auto gpuLayouts = getGPUObjectsFromAssets<asset::ICPUDescriptorSetLayout>(cpuLayouts.data(), cpuLayouts.data()+cpuLayouts.size());

    //iterators
    size_t di = 0ull;
    size_t bi=0ull, bvi=0ull, ti=0ull, tvi=0ull, si=0ull;
    for (ptrdiff_t i = 0u; i < assetCount; ++i)
    {
        asset::ICPUDescriptorSet* cpuds = _begin[i];

        using SGPUWriteDescriptorSet = IGPUDescriptorSet::SWriteDescriptorSet;
        using gpu_descriptors_array = core::smart_refctd_dynamic_array<SGPUWriteDescriptorSet>;
        auto gpudescriptors = core::make_refctd_dynamic_array<gpu_descriptors_array>(cpuds->getDescriptors().length());

        for (size_t d = 0ull; d < gpudescriptors->size(); ++d, ++di)
        {
            const auto& wrt = descInfos[d];
            auto& gpuwrt = (*gpudescriptors)[di];
            gpuwrt.binding = wrt.binding;
            gpuwrt.descriptorType = wrt.descType;
            using gpu_desc_info_array = core::smart_refctd_dynamic_array<IGPUDescriptorSet::SDescriptorInfo>;
            gpuwrt.info = core::make_refctd_dynamic_array<gpu_desc_info_array>(wrt.count);

            if (isBufferDesc(wrt.descType))
            {
                for (size_t infoIter = 0ull; infoIter < gpuwrt.info->size(); ++infoIter)
                {
                    auto& out = (*gpuwrt.info)[infoIter];
                    const auto& in = cpuds->getDescriptors().begin()[d].info->operator[](infoIter).buffer;

                    out.buffer.offset = in.offset + (*gpuBuffers)[bufRedirs[bi]]->getOffset();
                    out.buffer.size = in.size;
                    out.desc = core::smart_refctd_ptr<IGPUBuffer>((*gpuBuffers)[bufRedirs[bi]]->getBuffer());

                    ++bi;
                }
            }
            else if (isBufviewDesc(wrt.descType))
            {
                for (size_t infoIter = 0ull; infoIter < gpuwrt.info->size(); ++infoIter)
                {
                    auto& out = (*gpuwrt.info)[infoIter];
                    out.desc = (*gpuBufviews)[bufviewRedirs[bvi++]];
                }
            }
            else if (isTextureDesc(wrt.descType) || isTexviewDesc(wrt.descType))
            {
                for (size_t infoIter = 0ull; infoIter < gpuwrt.info->size(); ++infoIter)
                {
                    auto& out = (*gpuwrt.info)[infoIter];
                    const auto& in = cpuds->getDescriptors().begin()[d].info->operator[](infoIter).image;

                    out.image.imageLayout = in.imageLayout;
                    if (isTextureDesc(wrt.descType)) {
                        out.image.sampler = (*gpuSamplers)[smplrRedirs[si++]];
                        out.desc = (*gpuTextures)[texRedirs[ti++]];
                    }
                    else {
                        out.desc = (*gpuTexviews)[texviewRedirs[tvi++]];
                    }
                }
            }
        }

        IGPUDescriptorSetLayout* gpulayout = (*gpuLayouts)[layoutRedirs[i]].get();

        (*res)[i] = m_driver->createGPUDescriptorSet(core::smart_refctd_ptr<IGPUDescriptorSetLayout>(gpulayout), std::move(gpudescriptors));
    }

    return res;
}

}}//irr::video

#endif //__IRR_I_GPU_OBJECT_FROM_ASSET_CONVERTER_H_INCLUDED__
