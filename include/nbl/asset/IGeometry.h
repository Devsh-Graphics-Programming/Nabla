// Copyright (C) 2018-2021 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_ASSET_I_GEOMETRY_H_INCLUDED_
#define _NBL_ASSET_I_GEOMETRY_H_INCLUDED_


#include "nbl/builtin/hlsl/shapes/aabb.hlsl"

#include "nbl/asset/IAsset.h"
#include "nbl/asset/format/decodePixels.h"
#include "nbl/asset/format/encodePixels.h"


namespace nbl::asset
{
class IGeometryBase : public virtual core::IReferenceCounted
{
    public:
        enum class PrimitiveType : uint8_t
        {
            Triangles = 0,
            Lines = 2,
            Points = 3,
            // do not overengineer
//            AABBs = 1,
            // LSS, Beziers etc.
        };

        // using `nbl::hlsl::` concepts instead of `std::` so that `nbl::hlsl::float16_t` can be used
        union SAABBStorage
        {
            hlsl::shapes::AABB<4,hlsl::float64_t> f64 = hlsl::shapes::AABB<4,hlsl::float64_t>::create();
            hlsl::shapes::AABB<4,uint64_t> u64;
            hlsl::shapes::AABB<4,int64_t> s64;
            hlsl::shapes::AABB<4,hlsl::float32_t> f32;
            hlsl::shapes::AABB<4,uint32_t> u32;
            hlsl::shapes::AABB<4,int32_t> s32;
            hlsl::shapes::AABB<4,hlsl::float16_t> f16;
            hlsl::shapes::AABB<4,uint16_t> u16;
            hlsl::shapes::AABB<4,int16_t> s16;
            hlsl::shapes::AABB<4,uint8_t> u8;
            hlsl::shapes::AABB<4,int8_t> s8;
        }; 
        struct SDataViewBase
        {
            // mostly checking validity of the format
            inline operator bool() const {return format==EF_UNKNOWN || isBlockCompressionFormat(format) && isDepthOrStencilFormat(format);}

            //
            inline bool isFormatted() const {return format!=EF_UNKNOWN && bool(*this);}

            // Useful for checking if something can be used as an index
            inline bool isFormattedScalarInteger() const
            {
                if (isFormatted())
                switch (format)
                {
                    case EF_R8_SINT: [fallthrough];
                    case EF_R8_UINT: [fallthrough];
                    case EF_R16_SINT: [fallthrough];
                    case EF_R16_UINT: [fallthrough];
                    case EF_R32_SINT: [fallthrough];
                    case EF_R32_UINT: [fallthrough];
                    case EF_R64_SINT: [fallthrough];
                    case EF_R64_UINT:
                        return true;
                    default:
                        break;
                }
                return false;
            }

            //
            inline uint32_t getStride() const
            {
                if (isFormatted())
                    return getFormatClassBlockBytesize(getFormatClass(format));
                return stride;
            }

            // optional, really only meant for formatted views
            SAABBStorage encodedDataRange = {};
            // 0 means no fixed stride, totally variable data inside
            uint32_t stride = 0;
            // format takes precedence over stride
            E_FORMAT format = EF_UNKNOWN;
            // If format is UNORM or SNORM, is the vertex data relative to the AABB (range) of the stream
            uint8_t normRelativeCompression : 1 = false;
        };

        virtual const SAABBStorage& getAABB() const = 0;

    protected:
        inline IGeometryBase() {}
};

// A geometry should map 1:1 to a BLAS geometry, Meshlet or a Drawcall in API terms
template<class BufferType>
class NBL_API2 IGeometry : public IGeometryBase
{
    public:
        struct SDataView
        {
            inline operator bool() const {return src && composed;}

            inline uint64_t getElementCount() const
            {
                if (!this->operator bool())
                    return 0ull;
                const auto stride = getStride();
                if (stride==0)
                    return 0ull;
                return src.length/stride;
            }

            SDataViewBase composed = {};
            SBufferRange<BufferType> src = {};
        };
        //
        inline const SDataView& getPositionView() const {return m_positionView;}

        // depends on indexing, primitive type, etc.
        virtual uint64_t getPrimitiveCount() const = 0;

        // This is the upper bound on the local Joint IDs that the geometry uses
        virtual uint32_t getJointCount() const = 0;

        //
        inline bool isSkinned() const {return getJointCount()>0;}

        // Providing Per-Joint Bind-Pose-Space AABBs is optional for a skinned geometry
        inline const SDataView* getJointOBBView() const
        {
            if (isSkinned())
                return &m_jointOBBView;
            return nullptr;
        }

    protected:
        virtual ~IGeometry() = default;

        // needs to be hidden because of mutability checking
        inline bool setPositionView(SDataView&& view)
        {
            // need a formatted view for the positions
            if (!view || !view.composed.isFormatted())
                return false;
            m_positionView = std::move(view);
            return true;
        }
        // 
        inline bool setJointOBBView(SDataView&& view)
        {
            // want to set, not clear the AABBs
            if (view)
            {
                // An OBB is a affine transform of a [0,1]^3 unit cube to the Oriented Bounding Box
                // Needs to be formatted, each element is a row of that matrix
                if (!view.composed.isFormatted() || getFormatChannelCount(view.composed.format)!=4)
                   return false;
                // The range doesn't need to be exact, just large enough
                if (view.getElementCount()<getJointCount()*3)
                    return false;
            }
            m_jointOBBView = std::move(view);
            return true;
        }

        // Everyone needs a position attribute (even if its somehow transformed into a primitive
        SDataView m_positionView = {};
        // The conservative OBBs of each joint's influence on the primitive positions.
        // OBB chosen because AABB would need to be in joint-space and we don't wish to care about bind poses here.
        // Note that orientation and translation of the bindpose be partially recovered from the OBB.
        SDataView m_jointOBBView = {};
};


// for geometries which can be indexed with an index buffer
template<class BufferType>
class NBL_API2 IIndexableGeometry : public IGeometry
{
    public:
        inline const SDataView& getIndexView() const {return m_indexView;}

        inline const uint64_t getIndexCount() const
        {
            return m_indexView.getElementCount();
        }

    protected:
        // Needs to be hidden because ICPU base class shall check mutability
        inline bool setIndexView(SDataView&& view)
        {
            if (view.isFormattedScalarInteger())
            {
                m_indexView = std::move(strm);
                return true;
            }
            return false;
        }

        //
        SDataView m_indexView = {};
};

}
#endif