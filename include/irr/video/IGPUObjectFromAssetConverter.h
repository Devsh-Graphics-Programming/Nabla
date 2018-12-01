// Do not include this in headers, please
#ifndef __IRR_I_GPU_OBJECT_FROM_ASSET_CONVERTER_H_INCLUDED__
#define __IRR_I_GPU_OBJECT_FROM_ASSET_CONVERTER_H_INCLUDED__

#include "irr/core/Types.h"
#include "irr/asset/IAssetManager.h"
#include "IDriver.h"
#include "IDriverMemoryBacked.h"
#include "SMesh.h"
#include "CSkinnedMesh.h"
#include "irr/asset/SSkinMeshBuffer.h"
#include "CLogger.h"
#include "irr/video/asset_traits.h"

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

    inline virtual core::vector<typename video::asset_traits<asset::ICPUBuffer>::GPUObjectType*> create(asset::ICPUBuffer** const _begin, asset::ICPUBuffer** const _end);
    inline virtual core::vector<typename video::asset_traits<asset::ICPUMeshBuffer>::GPUObjectType*> create(asset::ICPUMeshBuffer** const _begin, asset::ICPUMeshBuffer** const _end);
    inline virtual core::vector<typename video::asset_traits<asset::ICPUMesh>::GPUObjectType*> create(asset::ICPUMesh** const _begin, asset::ICPUMesh** const _end);
    inline virtual core::vector<typename video::asset_traits<asset::ICPUTexture>::GPUObjectType*> create(asset::ICPUTexture** const _begin, asset::ICPUTexture** const _end);

    template<typename AssetType>
    core::vector<typename video::asset_traits<AssetType>::GPUObjectType*> getGPUObjectsFromAssets(AssetType** const _begin, AssetType** const _end)
    {
        core::vector<AssetType*> notFound;
        core::vector<size_t> pos;
        core::vector<typename video::asset_traits<AssetType>::GPUObjectType*> res;
        AssetType** it = _begin;
        while (it != _end)
        {
            core::IReferenceCounted* gpu = m_assetManager->findGPUObject(*it);
            if (!gpu)
            {
                notFound.push_back(*it);
                pos.push_back(it - _begin);
            }
            else res.push_back(dynamic_cast<typename video::asset_traits<AssetType>::GPUObjectType*>(gpu));
            ++it;
        }

        core::vector<typename video::asset_traits<AssetType>::GPUObjectType*> created = create(notFound.data(), notFound.data() + notFound.size());
        for (size_t i = 0u; i < created.size(); ++i)
        {
            m_assetManager->convertAssetToEmptyCacheHandle(notFound[i], created[i]);
            res.insert(res.begin() + pos[i], created[i]);
        }

        return res;
    }

protected:
    template<typename T>
    static inline core::vector<size_t> eliminateDuplicatesAndGenRedirs(core::vector<T*>& _input)
    {
        core::vector<size_t> redirs;

        core::unordered_map<T*, size_t> firstOccur;
        size_t i = 0u;
        for (T* el : _input)
        {
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

auto IGPUObjectFromAssetConverter::create(asset::ICPUBuffer** const _begin, asset::ICPUBuffer** const _end) -> core::vector<typename video::asset_traits<asset::ICPUBuffer>::GPUObjectType*>
{
    core::vector<typename video::asset_traits<asset::ICPUBuffer>::GPUObjectType*> res;

    asset::ICPUBuffer** it = _begin;
    while (it != _end)
    {
        IDriverMemoryBacked::SDriverMemoryRequirements reqs;
        reqs.vulkanReqs.size = (*it)->getSize();
        reqs.vulkanReqs.alignment = 8;
        reqs.vulkanReqs.memoryTypeBits = 0xffffffffu;
        reqs.memoryHeapLocation = IDriverMemoryAllocation::ESMT_DEVICE_LOCAL;
        reqs.mappingCapability = IDriverMemoryAllocation::EMCF_CANNOT_MAP;
        reqs.prefersDedicatedAllocation = true;
        reqs.requiresDedicatedAllocation = true;
        IGPUBuffer* buffer = m_driver->createGPUBufferOnDedMem(reqs, true);

        if (buffer)
            buffer->updateSubRange(video::IDriverMemoryAllocation::MemoryRange(0, (*it)->getSize()), (*it)->getPointer());

        res.push_back(buffer);
        ++it;
    }

    return res;
}
auto IGPUObjectFromAssetConverter::create(asset::ICPUMeshBuffer** _begin, asset::ICPUMeshBuffer** _end) -> core::vector<typename video::asset_traits<asset::ICPUMeshBuffer>::GPUObjectType*>
{
    struct VaoConfig
    {
        VaoConfig() : noAttributes{ true }, success{ true }, idxbuf{ nullptr }
        {
            std::fill(oldbuffer, oldbuffer + scene::EVAI_COUNT, nullptr);
        }

        const asset::ICPUBuffer* oldbuffer[scene::EVAI_COUNT];
        scene::E_COMPONENTS_PER_ATTRIBUTE components[scene::EVAI_COUNT];
        scene::E_COMPONENT_TYPE componentTypes[scene::EVAI_COUNT];
        size_t strides[scene::EVAI_COUNT];
        size_t offsets[scene::EVAI_COUNT];
        uint32_t divisors[scene::EVAI_COUNT];
        const asset::ICPUBuffer* idxbuf;
        bool noAttributes;
        bool success;
    };

    core::vector<asset::ICPUBuffer*> cpuBufDeps;
    core::vector<VaoConfig> vaoConfigs;
    core::vector<typename video::asset_traits<asset::ICPUMeshBuffer>::GPUObjectType*> res;
    core::vector<SCPUMaterial> cpumaterials;

    asset::ICPUMeshBuffer** it = _begin;
    while (it != _end)
    {
        cpumaterials.push_back((*it)->getMaterial());

        scene::ICPUMeshDataFormatDesc* origdesc = static_cast<scene::ICPUMeshDataFormatDesc*>((*it)->getMeshDataAndFormat());
        //if (!origdesc)
         //   return nullptr; //todo

        VaoConfig vaoConf;

        for (size_t j = 0; j < scene::EVAI_COUNT; j++)
        {
            scene::E_VERTEX_ATTRIBUTE_ID attrId = static_cast<scene::E_VERTEX_ATTRIBUTE_ID>(j);
            vaoConf.oldbuffer[attrId] = origdesc->getMappedBuffer(attrId);
            if (vaoConf.oldbuffer[attrId])
            {
                cpuBufDeps.push_back(const_cast<asset::ICPUBuffer*>(vaoConf.oldbuffer[attrId]));
                vaoConf.noAttributes = false;
            }

            vaoConf.components[attrId] = origdesc->getAttribComponentCount(attrId);
            vaoConf.componentTypes[attrId] = origdesc->getAttribType(attrId);
            vaoConf.strides[attrId] = origdesc->getMappedBufferStride(attrId);
            vaoConf.offsets[attrId] = origdesc->getMappedBufferOffset(attrId);
            vaoConf.divisors[attrId] = origdesc->getAttribDivisor(attrId);
            if (!scene::validCombination(vaoConf.componentTypes[attrId], vaoConf.components[attrId]))
            {
                os::Printer::log("createGPUObjectFromAsset input ICPUMeshBuffer(s) have one or more invalid attribute specs!\n", ELL_ERROR);
                vaoConf.success = false;
                break;
            }
        }
        vaoConf.idxbuf = origdesc->getIndexBuffer();
        if (vaoConf.idxbuf)
            cpuBufDeps.push_back(const_cast<asset::ICPUBuffer*>(vaoConf.idxbuf));
        vaoConfigs.push_back(vaoConf);

        scene::IGPUMeshBuffer* gpuMeshBuf = new scene::IGPUMeshBuffer();
        gpuMeshBuf->setIndexType((*it)->getIndexType());
        gpuMeshBuf->setBaseVertex((*it)->getBaseVertex());
        gpuMeshBuf->setIndexCount((*it)->getIndexCount());
        gpuMeshBuf->setIndexBufferOffset((*it)->getIndexBufferOffset());
        gpuMeshBuf->setInstanceCount((*it)->getInstanceCount());
        gpuMeshBuf->setBaseInstance((*it)->getBaseInstance());
        gpuMeshBuf->setPrimitiveType((*it)->getPrimitiveType());
        const core::aabbox3df oldBBox = (*it)->getBoundingBox();
        if ((*it)->getMeshBufferType() != scene::EMBT_ANIMATED_SKINNED)
            (*it)->recalculateBoundingBox();
        gpuMeshBuf->setBoundingBox((*it)->getBoundingBox());
        if ((*it)->getMeshBufferType() != scene::EMBT_ANIMATED_SKINNED)
            (*it)->setBoundingBox(oldBBox);
        res.push_back(gpuMeshBuf);

        ++it;
    }

    core::vector<asset::ICPUTexture*> cpuTexDeps;
    for (const SCPUMaterial& m : cpumaterials)
    {
        for (uint32_t i = 0u; i < _IRR_MATERIAL_MAX_TEXTURES_; ++i)
            if (asset::ICPUTexture* t = m.getTexture(i))
                cpuTexDeps.push_back(t);
    }

    const core::vector<size_t> bufRedir = eliminateDuplicatesAndGenRedirs(cpuBufDeps);
    const core::vector<size_t> texRedir = eliminateDuplicatesAndGenRedirs(cpuTexDeps);
    core::vector<typename video::asset_traits<asset::ICPUBuffer>::GPUObjectType*> gpuBufDeps = getGPUObjectsFromAssets(cpuBufDeps.data(), cpuBufDeps.data() + cpuBufDeps.size());
    core::vector<typename video::asset_traits<asset::ICPUTexture>::GPUObjectType*> gpuTexDeps = getGPUObjectsFromAssets(cpuTexDeps.data(), cpuTexDeps.data() + cpuTexDeps.size());
    size_t j = 0u; // buffer deps iterator
    size_t t = 0u; // texture deps iterator
    for (size_t i = 0u; i < res.size(); ++i)
    {
        SGPUMaterial mat;
        static_assert(sizeof(SCPUMaterial) == sizeof(SGPUMaterial), "SCPUMaterial and SGPUMaterial are NOT same sizes!");
        memcpy(&mat, &cpumaterials[i], sizeof(mat));
        for (size_t k = 0u; k < _IRR_MATERIAL_MAX_TEXTURES_; ++k)
        {
            if (mat.getTexture(k))
            {
                mat.setTexture(k, gpuTexDeps[texRedir[t]]);
                ++t;
            }
        }
        res[i]->getMaterial() = mat;

        const VaoConfig& vaoConf = vaoConfigs[i];
        if (!vaoConf.noAttributes && vaoConf.success)
        {
            scene::IGPUMeshDataFormatDesc* vao = m_driver->createGPUMeshDataFormatDesc();
            res[i]->setMeshDataAndFormat(vao);
            vao->drop();
            for (size_t k = 0u; k < scene::EVAI_COUNT; ++k)
            {
                if (vaoConf.oldbuffer[k])
                {
                    vao->mapVertexAttrBuffer(
                        gpuBufDeps[bufRedir[j]],
                        scene::E_VERTEX_ATTRIBUTE_ID(k),
                        vaoConf.components[k],
                        vaoConf.componentTypes[k],
                        vaoConf.strides[k],
                        vaoConf.offsets[k],
                        vaoConf.divisors[k]
                    );
                    ++j;
                }
            }
            if (vaoConf.idxbuf)
            {
                vao->mapIndexBuffer(gpuBufDeps[bufRedir[j]]);
                ++j;
            }
        }
    }
    for (video::IGPUBuffer* b : gpuBufDeps)
        b->drop(); // drop after mappings

    return res;
}
auto IGPUObjectFromAssetConverter::create(asset::ICPUMesh** const _begin, asset::ICPUMesh** const _end) -> core::vector<typename video::asset_traits<asset::ICPUMesh>::GPUObjectType*>
{
    core::vector<typename video::asset_traits<asset::ICPUMesh>::GPUObjectType*> res;
    core::vector<asset::ICPUMeshBuffer*> cpuDeps;

    asset::ICPUMesh** it = _begin;
    while (it != _end)
    {
        for (uint32_t i = 0u; i < (*it)->getMeshBufferCount(); ++i)
            cpuDeps.push_back((*it)->getMeshBuffer(i));

        scene::IGPUMesh* gpumesh = nullptr;
        switch ((*it)->getMeshType())
        {
        case scene::EMT_ANIMATED_SKINNED:
            gpumesh = new scene::CGPUSkinnedMesh(static_cast<asset::ICPUSkinnedMesh*>(*it)->getBoneReferenceHierarchy());
            break;
        default:
            gpumesh = new scene::SGPUMesh();
            break;
        }
        res.push_back(gpumesh);

        ++it;
    }

    core::vector<size_t> redir = eliminateDuplicatesAndGenRedirs(cpuDeps);
    core::vector<typename video::asset_traits<asset::ICPUMeshBuffer>::GPUObjectType*> gpuDeps = getGPUObjectsFromAssets(cpuDeps.data(), cpuDeps.data() + cpuDeps.size());
    for (size_t i = 0u, j = 0u; i < res.size(); ++i)
    {
        switch (res[i]->getMeshType())
        {
        case scene::EMT_ANIMATED_SKINNED:
            for (uint32_t k = 0u; k < (*(_begin + i))->getMeshBufferCount(); ++k)
            {
                static_cast<scene::CGPUSkinnedMesh*>(res[i])->addMeshBuffer(gpuDeps[redir[j]], static_cast<asset::SCPUSkinMeshBuffer*>((*(_begin + i))->getMeshBuffer(i))->getMaxVertexBoneInfluences());
                ++j;
            }
            break;
        default:
            for (uint32_t k = 0u; k < (*(_begin + i))->getMeshBufferCount(); ++k)
            {
                static_cast<scene::SGPUMesh*>(res[i])->addMeshBuffer(gpuDeps[redir[j]]);
                ++j;
            }
            break;
        }
    }

    return res;
}

auto IGPUObjectFromAssetConverter::create(asset::ICPUTexture** _begin, asset::ICPUTexture**_end) -> core::vector<typename video::asset_traits<asset::ICPUTexture>::GPUObjectType*>
{
    core::vector<typename video::asset_traits<asset::ICPUTexture>::GPUObjectType*> res;

    asset::ICPUTexture** it = _begin;
    while (it != _end)
    {
        asset::ICPUTexture* cpuTex = *it;
        ITexture* t = m_driver->createGPUTexture(cpuTex->getType(), cpuTex->getSize(), cpuTex->getHighestMip() ? cpuTex->getHighestMip()+1 : 0, cpuTex->getColorFormat());

        for (const CImageData* img : cpuTex->getMipmaps())
        {
            t->updateSubRegion(img->getColorFormat(), img->getData(), img->getSliceMin(), img->getSliceMax(), img->getSupposedMipLevel(), img->getUnpackAlignment());
        }

        if (cpuTex->getHighestMip()==0 && t->hasMipMaps())
            t->regenerateMipMapLevels(); // todo : Compute Shader mip-mapper necessary after vulkan

        res.push_back(t);

        ++it;
    }

    return res;
}

}}//irr::video

#endif //__IRR_I_GPU_OBJECT_FROM_ASSET_CONVERTER_H_INCLUDED__