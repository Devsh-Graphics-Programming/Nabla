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

	public:
		IGPUObjectFromAssetConverter(asset::IAssetManager* _assetMgr, video::IDriver* _drv) : m_assetManager{_assetMgr}, m_driver{_drv} {}

		virtual ~IGPUObjectFromAssetConverter() = default;

		inline virtual created_gpu_object_array<asset::ICPUBuffer>				create(asset::ICPUBuffer** const _begin, asset::ICPUBuffer** const _end);
		//inline virtual created_gpu_object_array<asset::ICPUMeshDataFormatDesc>	create(asset::ICPUMeshDataFormatDesc** const _begin, asset::ICPUMeshBuffer** const _end); // we change to ICPURenderPassIndependentPipeline
		inline virtual created_gpu_object_array<asset::ICPUMeshBuffer>			create(asset::ICPUMeshBuffer** const _begin, asset::ICPUMeshBuffer** const _end); // soon ICPUMeshBuffer will hold all its own buffer bindings
		inline virtual created_gpu_object_array<asset::ICPUMesh>				create(asset::ICPUMesh** const _begin, asset::ICPUMesh** const _end);
		inline virtual created_gpu_object_array<asset::ICPUTexture>				create(asset::ICPUTexture** const _begin, asset::ICPUTexture** const _end);
        inline virtual created_gpu_object_array<asset::ICPUShader>				create(asset::ICPUShader** const _begin, asset::ICPUShader** const _end);
        inline virtual created_gpu_object_array<asset::ICPUSpecializedShader>	create(asset::ICPUSpecializedShader** const _begin, asset::ICPUSpecializedShader** const _end);
        inline virtual created_gpu_object_array<asset::ICPUBufferView>		    create(asset::ICPUBufferView** const _begin, asset::ICPUBufferView** const _end);
        inline virtual created_gpu_object_array<asset::ICPUDescriptorSetLayout> create(asset::ICPUDescriptorSetLayout** const _begin, asset::ICPUDescriptorSetLayout** const _end);
        inline virtual created_gpu_object_array<asset::ICPUSampler>		        create(asset::ICPUSampler** const _begin, asset::ICPUSampler** const _end);
        inline virtual created_gpu_object_array<asset::ICPUPipelineLayout>		create(asset::ICPUPipelineLayout** const _begin, asset::ICPUPipelineLayout** const _end);


		template<typename AssetType>
		created_gpu_object_array<AssetType> getGPUObjectsFromAssets(AssetType* const* const _begin, AssetType* const* const _end)
		{
			const auto assetCount = std::distance(_begin, _end);
			auto res = core::make_refctd_dynamic_array<created_gpu_object_array<AssetType> >(assetCount);

			core::vector<AssetType*> notFound; notFound.reserve(assetCount);
			core::vector<size_t> pos; pos.reserve(assetCount);

			for (AssetType*const * it=_begin; it!=_end; it++)
			{
				const auto index = std::distance(_begin,it);

				auto gpu = m_assetManager->findGPUObject(*it);
				if (!gpu)
				{
					notFound.push_back(*it);
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
					redirs.push_back(0xdeadbeefbadc0ffeull);
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
	const auto assetCount = std::distance(_begin, _end);
	auto res = core::make_refctd_dynamic_array<created_gpu_object_array<asset::ICPUMeshBuffer> >(assetCount);

    struct VaoConfig //! ~@Crisspl why on earth is the create<ICPUMeshBuffer> function creating the pipelines/VAOs as well?~ Because MeshBuffer holds the index buffer, but MeshDataFormatDesc holds the other buffer bindings
    {
        VaoConfig() : noAttributes{ true }, idxbuf{ nullptr }
        {
            std::fill(oldbuffer, oldbuffer + asset::EVAI_COUNT, nullptr);
        }

        const asset::ICPUBuffer* oldbuffer[asset::EVAI_COUNT];
        asset::E_FORMAT formats[asset::EVAI_COUNT];
        size_t strides[asset::EVAI_COUNT];
        size_t offsets[asset::EVAI_COUNT];
        uint32_t divisors[asset::EVAI_COUNT];
        const asset::ICPUBuffer* idxbuf;
        bool noAttributes;
    };


	core::vector<VaoConfig> vaoConfigs(assetCount);
	core::vector<SCPUMaterial> cpumaterials(assetCount);

	constexpr auto MaxBuffersPerVAO = (asset::EVAI_COUNT + 1u);
	core::vector<asset::ICPUBuffer*> cpuBufDeps(MaxBuffersPerVAO*assetCount,nullptr);
	core::vector<asset::ICPUTexture*> cpuTexDeps(_IRR_MATERIAL_MAX_TEXTURES_*assetCount,nullptr);
	for (auto j=0u; j<assetCount; j++)
	{
		auto& output = res->operator[](j);

		auto* const _asset = _begin[j];
		const asset::ICPUMeshDataFormatDesc* origdesc = static_cast<asset::ICPUMeshDataFormatDesc*>(_asset->getMeshDataAndFormat());
		if (!origdesc)
		{
			output = nullptr;
			continue;
		}

		VaoConfig vaoConf;
		for (auto i=0u; i<asset::EVAI_COUNT; i++)
		{
			asset::E_VERTEX_ATTRIBUTE_ID attrId = static_cast<asset::E_VERTEX_ATTRIBUTE_ID>(i);
			vaoConf.oldbuffer[attrId] = origdesc->getMappedBuffer(attrId);
			if (vaoConf.oldbuffer[attrId])
			{
				cpuBufDeps[j*MaxBuffersPerVAO+i] = const_cast<asset::ICPUBuffer*>(vaoConf.oldbuffer[attrId]);
				vaoConf.noAttributes = false;
			}

			vaoConf.formats[attrId] = origdesc->getAttribFormat(attrId);
			vaoConf.strides[attrId] = origdesc->getMappedBufferStride(attrId);
			vaoConf.offsets[attrId] = origdesc->getMappedBufferOffset(attrId);
			vaoConf.divisors[attrId] = origdesc->getAttribDivisor(attrId);
		}
		vaoConf.idxbuf = origdesc->getIndexBuffer();
		if (vaoConf.idxbuf)
			cpuBufDeps[j*MaxBuffersPerVAO+asset::EVAI_COUNT] = const_cast<asset::ICPUBuffer*>(vaoConf.idxbuf);
		vaoConfigs[j] = std::move(vaoConf);

		const auto& mat = cpumaterials[j] = _asset->getMaterial();
		for (auto i=0u; i<_IRR_MATERIAL_MAX_TEXTURES_; i++)
			cpuTexDeps[_IRR_MATERIAL_MAX_TEXTURES_*j+i] = mat.getTexture(i);


        auto gpuMeshBuf = core::make_smart_refctd_ptr<IGPUMeshBuffer>();
        gpuMeshBuf->setIndexType(_asset->getIndexType());
        gpuMeshBuf->setBaseVertex(_asset->getBaseVertex());
        gpuMeshBuf->setIndexCount(_asset->getIndexCount());
        gpuMeshBuf->setIndexBufferOffset(_asset->getIndexBufferOffset());
        gpuMeshBuf->setInstanceCount(_asset->getInstanceCount());
        gpuMeshBuf->setBaseInstance(_asset->getBaseInstance());
        gpuMeshBuf->setPrimitiveType(_asset->getPrimitiveType());
        const core::aabbox3df oldBBox = _asset->getBoundingBox();
        if (_asset->getMeshBufferType() != asset::EMBT_ANIMATED_SKINNED)
            _asset->recalculateBoundingBox();
        gpuMeshBuf->setBoundingBox(_asset->getBoundingBox());
        if (_asset->getMeshBufferType() != asset::EMBT_ANIMATED_SKINNED)
            _asset->setBoundingBox(oldBBox);

        output = std::move(gpuMeshBuf);
    }

    const core::vector<size_t> bufRedir = eliminateDuplicatesAndGenRedirs(cpuBufDeps);
    const core::vector<size_t> texRedir = eliminateDuplicatesAndGenRedirs(cpuTexDeps);
    auto gpuBufDeps = getGPUObjectsFromAssets(cpuBufDeps.data(), cpuBufDeps.data() + cpuBufDeps.size());
    auto gpuTexDeps = getGPUObjectsFromAssets(cpuTexDeps.data(), cpuTexDeps.data() + cpuTexDeps.size());
    for (size_t i = 0u; i <assetCount; ++i)
    {
		auto& output = res->operator[](i);
		if (!output)
			continue;

		// TODO: All this shit is a terrible hack, REDO: @Crisspl WATCH OUT WITH THE BLOODY Shaders!
        SGPUMaterial mat;
        static_assert(sizeof(SCPUMaterial) == sizeof(SGPUMaterial), "SCPUMaterial and SGPUMaterial are NOT same sizes!");
        memcpy(&mat, &cpumaterials[i], sizeof(mat)); // this will mess up refcounting
        for (size_t k = 0u; k < _IRR_MATERIAL_MAX_TEXTURES_; ++k)
        {
            if (mat.getTexture(k))
            {
				memset(&mat.TextureLayer[k].Texture, 0, sizeof(void*)); // don't mess up reference counting
				auto redir = texRedir[i*_IRR_MATERIAL_MAX_TEXTURES_+k];
                mat.setTexture(k, core::smart_refctd_ptr(gpuTexDeps->operator[](redir)));
            }
        }
        output->getMaterial() = mat;

        const VaoConfig& vaoConf = vaoConfigs[i];
        if (!vaoConf.noAttributes)
        {
            auto vao = m_driver->createGPUMeshDataFormatDesc();
            for (size_t k = 0u; k < asset::EVAI_COUNT; ++k)
            {
                if (vaoConf.oldbuffer[k])
                {
					auto redir = bufRedir[i*MaxBuffersPerVAO+k];
					auto& buffDep = gpuBufDeps->operator[](redir);
                    vao->setVertexAttrBuffer(
                        core::smart_refctd_ptr<IGPUBuffer>(buffDep->getBuffer()), // yes construct new shared ptr, we want a grab
                        asset::E_VERTEX_ATTRIBUTE_ID(k),
                        vaoConf.formats[k],
                        vaoConf.strides[k],
                        vaoConf.offsets[k] + buffDep->getOffset(),
                        vaoConf.divisors[k]
                    );
                }
            }
            if (vaoConf.idxbuf)
            {
				auto redir = bufRedir[i*MaxBuffersPerVAO+asset::EVAI_COUNT];
				auto& buffDep = gpuBufDeps->operator[](redir);
                vao->setIndexBuffer(core::smart_refctd_ptr<IGPUBuffer>(buffDep->getBuffer())); // yes construct a new shared ptr, we want a grab
                output->setIndexBufferOffset(output->getIndexBufferOffset() + buffDep->getOffset());
            }
			output->setMeshDataAndFormat(std::move(vao));
        }
    }

    return res;
}
auto IGPUObjectFromAssetConverter::create(asset::ICPUMesh** const _begin, asset::ICPUMesh** const _end) -> created_gpu_object_array<asset::ICPUMesh>
{
	const auto assetCount = std::distance(_begin, _end);
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
    auto gpuDeps = getGPUObjectsFromAssets(cpuDeps.data(), cpuDeps.data() + cpuDeps.size());
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
    auto gpuDeps = getGPUObjectsFromAssets(cpuDeps.data(), cpuDeps.data()+cpuDeps.size());

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
    auto gpuBufs = getGPUObjectsFromAssets(cpuBufs.data(), cpuBufs.data()+cpuBufs.size());

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
    auto gpuSamplers = getGPUObjectsFromAssets(cpuSamplers.data(), cpuSamplers.data()+cpuSamplers.size());
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

    auto gpuDSLayouts = getGPUObjectsFromAssets(cpuDSLayouts.data(), cpuDSLayouts.data()+cpuDSLayouts.size());

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

}}//irr::video

#endif //__IRR_I_GPU_OBJECT_FROM_ASSET_CONVERTER_H_INCLUDED__
