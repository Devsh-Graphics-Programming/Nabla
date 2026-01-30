// Copyright (C) 2025-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_ASSET_I_CPU_SCENE_H_INCLUDED_
#define _NBL_ASSET_I_CPU_SCENE_H_INCLUDED_


#include "nbl/core/containers/CMemoryPool.h"

#include "nbl/asset/IScene.h"
#include "nbl/asset/material_compiler3/CTrueIR.h"


namespace nbl::asset
{
// 
class NBL_API2 ICPUScene final : public IAsset, public IScene
{
        using base_t = IScene;
        using material_table_allocator_t = core::GeneralpurposeAddressAllocatorST<uint32_t>;
//        using material_table_t = core::CMemoryPool<,core::allocator,false>;

    public:
        using material_pool_t = material_compiler3::CTrueIR;
        //
        static inline core::smart_refctd_ptr<ICPUScene> create(core::smart_refctd_ptr<material_pool_t>&& ir, const uint8_t maxMorphTargetGeometryCountLog2=16)
        {
            return core::smart_refctd_ptr<ICPUScene>(new ICPUScene(std::move(ir),maxMorphTargetGeometryCountLog2),core::dont_grab);
        }

        constexpr static inline auto AssetType = ET_SCENE;
        inline E_TYPE getAssetType() const override { return AssetType; }

        inline bool valid() const override
        {
            if (!m_instances)
                return false;
            auto materialTableOffsetIt = m_instances.materials.begin();
            for (const auto& targets : m_instances.morphTargets)
            {
                const auto materialTableOffset = *(materialTableOffsetIt++);
                // TODO: check if `materialTableOffset` can be contained in `materialTable`
                if (!targets || targets->valid())
                    return false;
                const auto geoCount = targets->getGeometryExclusiveCount({});
                // TODO: check if `materialTableOffset+geoCount` can be contained in `materialTable`
                // TODO: check every material is either null or belongs in `m_materialPool`
            }
            return true;
        }

        inline core::smart_refctd_ptr<IAsset> clone(uint32_t _depth=~0u) const
        {
            const auto nextDepth = _depth ? (_depth-1):0;
            // TODO: copy the material_table state/contents!
            // the True IR isn't an asset (yet), but it probably should be?
            auto retval = create(core::smart_refctd_ptr(m_materialPool),m_maxMorphTargetGeometryCountLog2);
            if (nextDepth)
            {
                retval->m_instances.morphTargets.resize(retval->m_instances.size());
                for (auto& targets : m_instances.morphTargets)
                    retval->m_instances.morphTargets.push_back(core::move_and_static_cast<ICPUMorphTargets>(targets->clone(nextDepth)));
                retval->m_envLightTexs.reserve(m_envLightTexs.size());
                for (const auto& tex : m_envLightTexs)
                    retval->m_envLightTexs.push_back(core::move_and_static_cast<ICPUImageView>(tex->clone(nextDepth)));
            }
            else
            {
                retval->m_instances = m_instances;
                retval->m_envLightTexs = m_envLightTexs;
            }
            retval->m_instances.materials = m_instances.materials;
            retval->m_instances.initialTransforms = m_instances.initialTransforms;
            retval->m_envLightTypes = m_envLightTypes;
            return retval;
        }

        // TODO: change to CRootNode
        using material_t = material_pool_t::TypedHandle<material_pool_t::INode>;
        inline material_compiler3::CTrueIR* getMaterialPool() {return m_materialPool.get();}
        inline const material_compiler3::CTrueIR* getMaterialPool() const {return m_materialPool.get();}
        
        //
        using material_table_offset_t = uint32_t;
        material_table_offset_t allocateMaterialTable(const ICPUMorphTargets* targets)
        {
            if (!targets)
                return material_table_allocator_t::invalid_address;
            return allocateMaterialTable(targets->getGeometryExclusiveCount({}));
        }
        material_table_offset_t allocateMaterialTable(const uint32_t count)
        {
            // TODO: implement
            return material_table_allocator_t::invalid_address;
        }
        void deallocateMaterialTable(const material_table_offset_t offset, const ICPUMorphTargets* targets)
        {
            return deallocateMaterialTable(offset,targets->getGeometryExclusiveCount({}));
        }
        void deallocateMaterialTable(const material_table_offset_t offset, const uint32_t count)
        {
            // TODO: implement
        }

        // TODO: get material table pointer

        // TODO: wrap up in some ECS storage class
        struct SInstanceStorage final
        {
            public:
                inline SInstanceStorage(const size_t size=1) : morphTargets(size), materials(size), initialTransforms(size) {}

                inline void clearInitialTransforms() {initialTransforms.clear();}

                inline operator bool() const
                {
                    if (morphTargets.size()!=materials.size())
                        return false;
                    if (initialTransforms.empty())
                        return true;
                    return morphTargets.size()==initialTransforms.size();
                }

                inline size_t resize(const size_t newSize)
                {
                    morphTargets.resize(newSize);
                    materials.resize(newSize);
                    if (!initialTransforms.empty())
                        initialTransforms.resize(newSize);
                }

                inline void erase(const size_t first, const size_t last)
                {
                    morphTargets.erase(morphTargets.begin()+first,morphTargets.begin()+last);
                    materials.erase(materials.begin()+first, materials.begin()+last);
                    initialTransforms.erase(initialTransforms.begin()+first,initialTransforms.begin()+last);
                }
                inline void erase(const size_t ix) {return erase(ix,ix+1);}

                inline size_t size() const {return morphTargets.size();}
            
            private:
                friend class ICPUScene;

                core::vector<core::smart_refctd_ptr<ICPUMorphTargets>> morphTargets;
                // One material table per morph target,
                // Within each morph target, one material per geometry
                core::vector<material_table_offset_t> materials;
                core::vector<hlsl::float32_t3x4> initialTransforms;
                // TODO: animations (keyframed transforms, skeleton instance)
        };

        //
        inline SInstanceStorage& getInstances() {return m_instances;}
        inline const SInstanceStorage& getInstances() const {return m_instances;}

        inline void setInstanceInitialTransform(const uint32_t index, const hlsl::float32_t3x4& xform)
        {
            if (index<m_instances.size())
                m_instances.initialTransforms[index] = xform;
        }

        enum class EEnvLightType : uint8_t
        {
            Cubemap,
            SphereMap, // u=theta, v=phi with (0,0) being top right of image
            OctahedralMap,
            Count
        };
        //
        inline bool addEnvLight(const EEnvLightType type, core::smart_refctd_ptr<ICPUImageView>&& tex)
        {
            if (!tex)
                return false;
            using view_e = IImageViewBase::E_TYPE;
            switch (tex->getCreationParameters().viewType)
            {
                case view_e::ET_2D: [[fallthrough]];
                case view_e::ET_2D_ARRAY:
                    m_envLightTypes.push_back(type);
                    break;
                case view_e::ET_CUBE_MAP: [[fallthrough]];
                case view_e::ET_CUBE_MAP_ARRAY:
                    if (type!=EEnvLightType::Cubemap)
                        return false;
                    m_envLightTypes.push_back(EEnvLightType::Cubemap);
                    break;
                default:
                    return false;
            }
            m_envLightTexs.push_back(std::move(tex));
            return true;
        }
        //
        inline std::span<const EEnvLightType> getEnviornmentLightTypes() const {return m_envLightTypes;}
        inline std::span<const ICPUImageView* const> getEnvironmentLightTextures() const {return {&m_envLightTexs.data()->get(),m_envLightTexs.size()}; }
        // TODO: add an erase_if and erase with begin/end iterators
        inline void clearEnvLights()
        {
            m_envLightTexs.clear();
            m_envLightTypes.clear();
        }

        //
        hlsl::float32_t3 m_ambientLight;

    protected:
        inline ICPUScene(core::smart_refctd_ptr<material_pool_t>&& materialPool, const uint32_t maxMorphTargetGeometryCountLog2) : 
            m_materialPool(std::move(materialPool)), m_maxMorphTargetGeometryCountLog2(maxMorphTargetGeometryCountLog2) {}
        //
        inline void visitDependents_impl(std::function<bool(const IAsset*)> visit) const override
        {
            assert(false && "Unimplemented"); // we'd probalby be going over the: morph targets, image views, ...
        }

        //
// TODO        material_table_t m_materialTable;
        core::smart_refctd_ptr<material_pool_t> m_materialPool;
        //
        SInstanceStorage m_instances;
        //
        core::vector<core::smart_refctd_ptr<ICPUImageView>> m_envLightTexs;
        core::vector<EEnvLightType> m_envLightTypes;
        //
        const uint8_t m_maxMorphTargetGeometryCountLog2;
};
}

#endif