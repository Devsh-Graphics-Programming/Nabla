// Copyright (C) 2025-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_ASSET_I_CPU_SCENE_H_INCLUDED_
#define _NBL_ASSET_I_CPU_SCENE_H_INCLUDED_


#include "nbl/core/containers/CMemoryPool.h"

#include "nbl/asset/IScene.h"
#include "nbl/asset/ICPUMorphTargets.h"
#include "nbl/asset/material_compiler3/CTrueIR.h"


namespace nbl::asset
{
// 
class ICPUScene final : public IAsset, public IScene
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
        constexpr static inline material_table_offset_t InvalidMaterialTable = ~0u;
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
                inline SInstanceStorage(const size_t size=0) : morphTargets(size), materials(size), initialTransforms(size) {}

                inline void clearInitialTransforms() {initialTransforms.clear();}

                explicit inline operator bool() const
                {
                    if (morphTargets.size()!=materials.size())
                        return false;
                    if (initialTransforms.empty())
                        return true;
                    return morphTargets.size()==initialTransforms.size();
                }

                inline void reserve(const size_t newSize)
                {
                    morphTargets.reserve(newSize);
                    materials.reserve(newSize);
                    if (!initialTransforms.empty())
                        initialTransforms.reserve(newSize);
                }

                inline void resize(const size_t newSize, const bool forceTransformStorage=false)
                {
                    morphTargets.resize(newSize);
                    materials.resize(newSize,InvalidMaterialTable);
                    if (forceTransformStorage || !initialTransforms.empty())
                        initialTransforms.resize(newSize,ICPUGeometryCollection::SGeometryReference{}.transform);
                }

                inline void erase(const size_t first, const size_t last)
                {
                    morphTargets.erase(morphTargets.begin()+first,morphTargets.begin()+last);
                    materials.erase(materials.begin()+first, materials.begin()+last);
                    if (!initialTransforms.empty())
                        initialTransforms.erase(initialTransforms.begin()+first,initialTransforms.begin()+last);
                }
                inline void erase(const size_t ix) {return erase(ix,ix+1);}

                inline uint64_t size() const {return morphTargets.size();}
            
                inline std::span<core::smart_refctd_ptr<ICPUMorphTargets>> getMorphTargets() {return morphTargets;}
                inline std::span<const core::smart_refctd_ptr<ICPUMorphTargets>> getMorphTargets() const {return morphTargets;}
            
                inline std::span<material_table_offset_t> getMaterialTables() {return materials;}
                inline std::span<const material_table_offset_t> getMaterialTables() const {return materials;}
            
                inline std::span<hlsl::float32_t3x4> getInitialTransforms() {return initialTransforms;}
                inline std::span<const hlsl::float32_t3x4> getInitialTransforms() const {return initialTransforms;}

            private:
                friend class ICPUScene;

                core::vector<core::smart_refctd_ptr<ICPUMorphTargets>> morphTargets;
                // One material table per morph target,
                // Within each morph target, one material per geometry
                core::vector<material_table_offset_t> materials;
                core::vector<hlsl::float32_t3x4> initialTransforms;
                // TODO: animations (keyframed transforms, skeleton instance)
        };

        // utility
        class ITLASExporter
        {
            protected:
                using instance_flags_t = asset::ICPUTopLevelAccelerationStructure::INSTANCE_FLAGS;

                inline ITLASExporter(const SInstanceStorage& _storage) : m_storage(_storage) {}

                const SInstanceStorage& m_storage;

            public:
                virtual inline ICPUMorphTargets::index_t getTargetIndex(const uint32_t instanceIx) {return ICPUMorphTargets::index_t{0u};}

                virtual inline instance_flags_t getInstanceFlags(const uint32_t instanceIx, const ICPUMorphTargets::index_t targetIx)
                {
                    // TODO: could derive from the material table if we want FORCE_OPAQUE_BIT or FORCE_NO_OPAQUE_BIT but its a whole instance thing
                    return instance_flags_t::TRIANGLE_FACING_CULL_DISABLE_BIT;
                }

                virtual inline uint32_t getInstanceIndex(const uint32_t instanceIx, const ICPUMorphTargets::index_t targetIx) {return instanceIx;}

                // default
                virtual inline uint32_t getSBTOffset(const material_table_offset_t materialsBeginIndex)
                {
                    return 0;
                }

                virtual inline uint8_t getMask(const uint32_t instanceIx, const ICPUMorphTargets::index_t targetIx)
                {
                    return 0xFF;
                }

                virtual inline hlsl::float32_t3x4 getTransform(const uint32_t instanceIx, const ICPUMorphTargets::index_t targetIx)
                {
                    if (m_storage.initialTransforms.empty())
                        return hlsl::math::linalg::diagonal<hlsl::float32_t3x4>(1.f);
                    else
                        return m_storage.initialTransforms[instanceIx];
                }

                // TODO: when we allow non-polygon geometries in the collection, we need to return a named pair, one BLAS for tris and one for AABBs
                virtual core::smart_refctd_ptr<ICPUBottomLevelAccelerationStructure> getBLAS(const uint32_t instanceIx, const ICPUMorphTargets::index_t targetIx) = 0;

                struct SResult
                {
                    explicit inline operator bool() const {return instances && !instances->empty();}

                    core::smart_refctd_dynamic_array<ICPUTopLevelAccelerationStructure::PolymorphicInstance> instances = nullptr;
                    bool allInstancesValid = false;
                };
                // TODO: SBT stuff
                inline SResult operator()()
                {
                    // this is because most GPUs report 16M as max instance count, and there's only 24 bits in `instanceCustomIndex`
                    constexpr uint64_t MaxInstanceCount = 0x1u<<24;
                    const uint64_t instanceCount = m_storage.size();
                    if (instanceCount>MaxInstanceCount)
                        return {};

                    std::vector<ICPUTopLevelAccelerationStructure::PolymorphicInstance> instances;
                    instances.reserve(instanceCount*2);
                    bool allInstancesValid = true;
                    for (auto i=0u; i<instanceCount; i++)
                    {
                        // TODO: deal with SRT motion later when we add keyframed animations
                        const auto targetIx = getTargetIndex(i);
                        const auto* const targets = m_storage.morphTargets[i].get();
                        if (!targets || !targets->valid())
                        {
                            allInstancesValid = false;
                            continue;
                        }
                        const auto* const collection = targets->getTargets()[targetIx.value].geoCollection.get();
                        ICPUTopLevelAccelerationStructure::StaticInstance inst;
                        inst.base.blas = getBLAS(i,targetIx);
                        if (!inst.base.blas)
                        {
                            allInstancesValid = false;
                            continue;
                        }
                        inst.transform = getTransform(i,targetIx);
                        const uint32_t customIndex = getInstanceIndex(i,targetIx);
                        if (customIndex>=MaxInstanceCount)
                        {
                            allInstancesValid = false;
                            continue;
                        }
                        inst.base.instanceCustomIndex = customIndex;
                        inst.base.mask = getMask(i,targetIx);
                        const auto targetTableOffset = m_storage.materials[i]+targets->getGeometryExclusiveCount(targetIx);
                        const auto sbtOffset = getSBTOffset(targetTableOffset);
                        if (sbtOffset>MaxInstanceCount+collection->getGeometries().size())
                        {
                            allInstancesValid = false;
                            continue;
                        }
                        inst.base.instanceShaderBindingTableRecordOffset = sbtOffset;
                        inst.base.flags = static_cast<uint32_t>(getInstanceFlags(i,targetIx));
                        instances.emplace_back().instance = std::move(inst);
                    }
                    // TODO: adjust BLAS geometry flags according to materials set opaqueness and NO_DUPLICATE_ANY_HIT_INVOCATION_BIT
                    SResult retval = {.instances=core::make_refctd_dynamic_array<decltype(SResult::instances)>(instanceCount),.allInstancesValid=allInstancesValid};
                    std::move(instances.begin(),instances.end(),retval.instances->begin());
                    return retval;
                }
        };
        class CDefaultTLASExporter final : public ITLASExporter
        {
                using triangles_t = ICPUBottomLevelAccelerationStructure::Triangles<ICPUBuffer>;
                core::vector<triangles_t> triangleScratch;
                core::vector<uint32_t> primitiveCountScratch;

            public:
                inline CDefaultTLASExporter(const SInstanceStorage& _storage) : ITLASExporter(_storage) {}

                inline core::smart_refctd_ptr<ICPUBottomLevelAccelerationStructure> getBLAS(const uint32_t instanceIx, const ICPUMorphTargets::index_t targetIx) override
                {
                    const auto* const targets = m_storage.morphTargets[instanceIx].get();
                    const auto* const collection = targets->getTargets()[targetIx.value].geoCollection.get();
                    // TODO: use emplace so erase can be faster
                    auto& entry = m_blasCache[collection];
                    if (!entry)
                    {
                        entry = core::make_smart_refctd_ptr<ICPUBottomLevelAccelerationStructure>();
                        //
                        const auto& geometries = collection->getGeometries();
                        // deal with triangles 
                        {
                            triangleScratch.resize(geometries.size());
                            primitiveCountScratch.resize(geometries.size());
                            const auto usedScratchEnd = ICPUGeometryCollection::CBLASExporter(geometries)(triangleScratch.begin(),primitiveCountScratch.data());
                            // TODO: report some error that a there was an unsupported geometry
                            //triangleScratch.end()!=usedScratchEnd
                            const auto actualGeoCount = std::distance(triangleScratch.begin(),usedScratchEnd);
                            if (actualGeoCount==0)
                            {
                                m_blasCache.erase(m_blasCache.find(collection));
                                return nullptr;
                            }
                            auto triGeos = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<triangles_t>>(actualGeoCount);
                            std::move(triangleScratch.begin(),usedScratchEnd,triGeos->begin());
                            auto primCounts = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<uint32_t>>(actualGeoCount);
                            std::copy_n(primitiveCountScratch.data(),actualGeoCount,primCounts->data());
                            entry->setGeometries(std::move(triGeos),std::move(primCounts));
                        }
                        using build_f = ICPUBottomLevelAccelerationStructure::BUILD_FLAGS;
                        // no virtual callbacks because its easy to tell what geometry collection the BLAS came from by looking at the cache after the export
                        // TODO: Allow Update when we figure out morph targets/skinning
                        // TODO: GEOMETRY_TYPE_IS_AABB_BIT for non-polygon geometry collections
                        entry->setBuildFlags(build_f::PREFER_FAST_TRACE_BIT|build_f::ALLOW_COMPACTION_BIT);
                        entry->setContentHash(entry->computeContentHash());
                    }
                    return entry;
                }

                // when doing animations, it good to copy and reuse this with dummy BLASes but where content hashes are already the same
                core::unordered_map<const ICPUGeometryCollection*,core::smart_refctd_ptr<ICPUBottomLevelAccelerationStructure>> m_blasCache;
        };

        //
        inline SInstanceStorage& getInstances() {return m_instances;}
        inline const SInstanceStorage& getInstances() const {return m_instances;}

        enum class EEnvLightType : uint8_t
        {
            Cubemap,
            SphereMap, // u=theta, v=phi with (0,0) being top right of image
            OctahedralMap,
            Count
        };
        //
        inline bool addEnvLight(const EEnvLightType type, core::smart_refctd_ptr<const ICPUImageView>&& tex)
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
        core::vector<core::smart_refctd_ptr<const ICPUImageView>> m_envLightTexs;
        core::vector<EEnvLightType> m_envLightTypes;
        //
        const uint8_t m_maxMorphTargetGeometryCountLog2;
};
}

#endif