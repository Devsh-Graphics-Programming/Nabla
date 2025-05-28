// Copyright (C) 2025-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_ASSET_I_MORPH_TARGETS_H_INCLUDED_
#define _NBL_ASSET_I_MORPH_TARGETS_H_INCLUDED_


#include "nbl/asset/IGeometryCollection.h"


namespace nbl::asset
{
// Unlike glTF we don't require same index buffers and maintaining isomorphisms between primitives/vertices in different geometries. But most of the use cases would have such mappings.
// The semantics are really up to you, these are just collections of Geometries which can be swapped out or interpolated between.
// Note: A LoD set can also be viewed as a morph shape per LoD, while a Motion Blur BLAS can be viewed as representing the interval between two consecutive Morph Targets. 
template<class BufferType>
class NBL_API2 IMorphTargets : public virtual core::IReferenceCounted
{
    public:
        struct index_t
        {
            explicit inline index_t(uint32_t _value) : value(_value) {}

            inline operator bool() const {return value!=(~0u);}

            uint32_t value = ~0u;
        };

        inline uint32_t getTargetCount() const
        {
            return static_cast<uint32_t>(m_morphGeometries.size());
        }

        template<typename Scalar, uint16_t Degree> requires std::is_floating_point_v<Scalar>
        struct SInterpolants
        {
            index_t indices[Degree];
            Scalar weights[Degree-1];
        };

        template<typename Scalar> requires std::is_floating_point_v<Scalar>
        inline SInterpolants<Scalar> getLinearBlend(const Scalar blend) const
        {
            SInterpolants<Scalar> retval;
            if (!m_morphGeometries.empty())
            {
                const Scalar maxMorph = getMorphShapeCount();
                retval.indices[0] = index_t(hlsl::clamp<Scalar>(blend,Scalar(0),maxMorph));
                retval.indices[1] = index_t(hlsl::min<Scalar>(retval.indices[0].value+Scalar(1),maxMorph));
                retval.weights[0] = blend-Scalar(retval.indices[0].value);
            }
            return retval;
        }

        struct STarget
        {
            core::smart_refctd_ptr<IGeometryCollection<BufferType>> geoCollection = {};
            // The geometry may be using a smaller set of joint/bone IDs which need to be remapped to a larger or common skeleton.
            // Ignored if the collection is not skinned.
            SDataView jointRedirectView = {};
        };
        inline const core::vector<STarget>& getTargets() const {return m_targets;}

    protected:
        virtual ~IMorphShapes() = default;

        //
        inline core::vector<STarget>& getTargets() {return m_targets;}

        //
        core::vector<STarget> m_targets;
        // no point keeping an overall AABB, because the only reason to do that is to skip animation/indexing logic all together
};
}

#endif