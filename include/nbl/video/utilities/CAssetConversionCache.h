// Copyright (C) 2018-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_VIDEO_C_ASSET_CONVERSION_CACHE_INCLUDED_
#define _NBL_VIDEO_C_ASSET_CONVERSION_CACHE_INCLUDED_
//
#include "nbl/core/declarations.h"
#include "nbl/core/alloc/LinearAddressAllocator.h"

#include <iterator>
#include <tuple>

#include "nbl/asset/asset.h"


#include "nbl/video/asset_traits.h"
#include "nbl/video/ISemaphore.h"
#include "nbl/video/ILogicalDevice.h"

#include "nbl/asset/ECommonEnums.h"


namespace nbl::video
{

    class CAssetConversionCache : public core::IReferenceCounted
    {
    public:
        template<class GPUObject> requires GPUObject::asset_t
            class CTypedAssetCache
        {
        public:
            struct Value
            {
                // for turning LVHC entry into PTSC
                inline Value(const GPUObject::patchable_params_t& _overrides, core::smart_refctd_ptr<const GPUObject::asset_t>&& _asset, const size_t assetHash)
                    : hashValueAtInsert(assetHash), asset(std::move(_asset)), overrides(_overrides)
                {
                    core::combine_hash(hashValueAtInsert, overrides);
                }
                inline Value(const GPUObject::patchable_params_t& _overrides, const GPUObject::asset_t* const _asset)
                    : Value(overrides, core::smart_refctd_ptr<const GPUObject::asset_t>(_asset), _asset->hash()) {}

                inline bool stale(IAsset::hash_cache_t* pAHC = nullptr) const
                {
                    size_t hash = asset->hash(pAHC);
                    core::combine_hash(hash, overrides);
                    return hash != hashValueAtInsert;
                }

                size_t hashValueAtInsert;
                const GPUObject::asset_t* asset;
                GPUObject::patchable_params_t converted;
            };

        protected:
            struct ReverseValue
            {
                ReverseValue(const Value&);

                size_t hashValueAtInsert;
                core::smart_refctd_ptr<const GPUObject::asset_t> asset;
            }
            struct Hasher
            {
                using is_transparent = void;

                static inline size_t operator()(const Value& key)
                {
                    return key.hashValueAtInsert;
                }
            };
            struct Equals
            {
                using is_transparent = void;

                static inline bool operator()(const Value& lhs, const Value& rhs)
                {
                    // if this redundant?
                    if (lhs.hashValueAtInsert != rhs.hashValueAtInsert)
                        return false;

                    if (lhs.overrides != rhs.overrides)
                        return false;

                    if (!lhs.asset)
                        return !rhs.asset;
                    return key.asset->equals(rhs.asset);
                }

                // clear before every insert/iterator invalidation op
                // consume after every find/insert
                //core::vector<decltype(m_storage::find)> m_staleItemsDetected;
            };
            core::unordered_map<Value, std::atomic<const GPUObject*>, Hasher, Equals> m_storage;
            core::unordered_map<core::smart_refctd_ptr<const GPUObject>, ReverseValue> m_reverseMap;
        };

    protected: 


        template <class... GPUObjects>
        class caches_t
        {
            std::tuple<CTypedAssetCache<GPUObjects>...> data;
        };
        //caches_t<IGPUBuffer> m_caches;

        CTypedAssetCache<IGPUBuffer> m_cache;

    };



    //define conversion order 
    // same order as iasset type
    // 
    //bucket assets by type
    // 
    //change func name to getDeps and change return type to range
}
#endif