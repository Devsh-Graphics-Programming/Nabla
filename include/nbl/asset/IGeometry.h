// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_ASSET_I_GEOMETRY_H_INCLUDED_
#define _NBL_ASSET_I_GEOMETRY_H_INCLUDED_


#include "nbl/builtin/hlsl/shapes/aabb.hlsl"

#include "nbl/asset/IAsset.h"


namespace nbl::asset
{
class IGeometryBase : public virtual core::IReferenceCounted
{
    public:
        enum class EPrimitiveType : uint8_t
        {
            Polygon = 0,
            // do not overengineer
//            AABBs = 1,
            // LSS, Beziers etc.
        };
        //
        virtual EPrimitiveType getPrimitiveType() const = 0;

        //
        enum class EAABBFormat : uint8_t
        {
            F64,
            U64,
            S64,
            F32,
            U32,
            S32,
            F16,
            U16,
            U16_NORM,
            S16,
            S16_NORM,
            U8,
            U8_NORM,
            S8,
            S8_NORM,
            BitCount=4
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

            //
            template<typename Visitor>
            inline void visitAABB(Visitor& visitor)
            {
                switch (newFormat)
                {
                    case EAABBFormat::F64:
                        visitor(encodedDataRange.f64);
                        break;
                    case EAABBFormat::U64:
                        visitor(encodedDataRange.u64);
                        break;
                    case EAABBFormat::S64:
                        visitor(encodedDataRange.s64);
                        break;
                    case EAABBFormat::F32:
                        visitor(encodedDataRange.f32);
                        break;
                    case EAABBFormat::U32:
                        visitor(encodedDataRange.u32);
                        break;
                    case EAABBFormat::S32:
                        visitor(encodedDataRange.s32);
                        break;
                    case EAABBFormat::F16:
                        visitor(encodedDataRange.f16);
                        break;
                    case EAABBFormat::U16: [[fallthrough]];
                    case EAABBFormat::U16_NORM:
                        visitor(encodedDataRange.u16);
                        break;
                    case EAABBFormat::S16: [[fallthrough]];
                    case EAABBFormat::S16_NORM:
                        visitor(encodedDataRange.s16);
                        break;
                    case EAABBFormat::U8: [[fallthrough]];
                    case EAABBFormat::U8_NORM:
                        visitor(encodedDataRange.u8);
                        break;
                    case EAABBFormat::S8: [[fallthrough]];
                    case EAABBFormat::S8_NORM:
                        visitor(encodedDataRange.s8);
                        break;
                    default:
                        break;
                }
            }
            template<typename Visitor>
            inline void visitAABB(const Visitor& visitor) const
            {
                auto tmp = [&visitor](const auto& aabb)->void{visitor(aabb);};
                const_cast<typename std::decay_t<decltype(*this)>*>(this)->visitAABB(tmp);
            }

            //
            inline void resetRange(const EAABBFormat newFormat)
            {
                rangeFormat = newFormat;
                auto tmp = [](auto& aabb)->void{aabb = aabb.clear();};
                visitAABB(tmp);
            }
            inline void resetRange() {resetRange(rangeFormat);}

            //
            template<typename AABB>
            inline AABB getRange() const
            {
                AABB retval = AABB::create();
                auto tmp = [&retval](const auto& aabb)->void
                {
                    retval.minVx = aabb.minVx;
                    retval.maxVx = aabb.maxVx;
                };
                visitAABB(tmp);
                return retval;
            }

            // optional, really only meant for formatted views
            SAABBStorage encodedDataRange = {};
            // 0 means no fixed stride, totally variable data inside
            uint32_t stride = 0;
            // Format takes precedence over stride
            // Note :If format is UNORM or SNORM, the vertex data is relative to the AABB (range)
            E_FORMAT format = EF_UNKNOWN;
            // tells you which `encodedDataRange` union member to access
            EAABBFormat rangeFormat : int(EAABBFormat::BitCount) = EAABBFormat::F64;
        };

        virtual const SAABBStorage& getAABB() const = 0;

    protected:
        virtual inline ~IGeometryBase() = default;
};

// A geometry should map 1:1 to a BLAS geometry, Meshlet or a Drawcall in API terms
template<class BufferType>
class NBL_API2 IGeometry : public IGeometryBase
{
    public:
        struct SDataView
        {
            inline operator bool() const {return src && composed;}

            //
            explicit inline operator SBufferBinding<const BufferType>() const
            {
                if (*this)
                    return {.offset=src.offset,.buffer=smart_refctd_ptr(src.buffer)};
                return {};
            }

            inline uint64_t getElementCount() const
            {
                if (!this->operator bool())
                    return 0ull;
                const auto stride = getStride();
                if (stride==0)
                    return 0ull;
                return src.length/stride;
            }
            
            //
            template<typename Index=uint32_t, typename U=BufferType> requires (std::is_same_v<U,BufferType> && std::is_same_v<U,ICPUBuffer>)
            inline const void* getPointer(const Index elIx=0) const
            {
                return const_cast<typename std::decay_t<decltype(*this)>*>(this)->getPointer<U>(elIx);
            }
            template<typename Index=uint32_t, typename U=BufferType> requires (std::is_same_v<U,BufferType> && std::is_same_v<U,ICPUBuffer>)
            inline void* getPointer(const Index elIx=0)
            {
                if (*this)
                    return reinterpret_cast<uint8_t*>(src.buffer->getPointer())+src.offset+elIx*getStride();
                return nullptr;
            }

            //
            template<typename V, typename Index=uint32_t, typename U=BufferType> requires (hlsl::concepts::Vector<V> && std::is_same_v<U,BufferType> && std::is_same_v<U,ICPUBuffer>)
            inline bool decodeElement(const Index elIx, V& v) const
            {
                if (!composed.isFormatted())
                    return false;
                using code_t = std::conditional_t<hlsl::concepts::FloatingPointVector<V>,hlsl::float64_t,std::conditional_t<hlsl::concepts::SignedIntVector<V>,int64_t,uint64_t>>;
                code_t tmp[4];
                if (const auto* src=getPointer<Index>(elIx); src)
                {
                    const void* srcArr[4] = {src,nullptr};
                    assert(!isScaledFormat(composed.format)); // handle this by improving the decode functions, not adding workarounds here
                    if (decodePixels<code_t>(composed.format,srcArr,tmp,0,0))
                    {
                        if (isNormalizedFormat(composed.format))
                        {
                            using traits = hlsl::vector_traits<V>;
                            const auto range = composed.getRange<hlsl::shapes::AABB<traits::Dimension,traits::scalar_type>>();
                            v = v*(range.maxVx-range.minVx)+range.minVx;
                        }
                        return true;
                    }
                }
                return false;
            }
            
            //
            template<typename V, typename Index=uint32_t, typename U=BufferType> requires (hlsl::concepts::Vector<V> && std::is_same_v<U,BufferType> && std::is_same_v<U,ICPUBuffer>)
            inline void encodeElement(const Index elIx, const V& v)
            {
                if (!composed.isFormatted())
                    return false;
                void* const out = getPointer<Index>(elIx);
                if (!out)
                    return false;
                using traits = hlsl::vector_traits<V>;
                using code_t = std::conditional_t<hlsl::concepts::FloatingPointVector<V>,hlsl::float64_t,std::conditional_t<hlsl::concepts::SignedIntVector<V>,int64_t,uint64_t>>;
                code_t tmp[traits::Dimension];
                const auto range = composed.getRange<traits::Dimension,traits::scalar_type>>();
                for (auto i=0u; i<traits::Dimension; i++)
                {
                    if (isNormalizedFormat(composed.format))
                        tmp[i] = v[i]*(range.maxVx[i]-range.minVx[i])+range.minVx[i];
                    else
                        tmp[i] = v[i];
                }
                assert(!isScaledFormat(composed.format)); // handle this by improving the decode functions, not adding workarounds here
                if (encodePixels<code_t>(composed.format,out,tmp))
                    return true;
                return false;
            }

            //
            inline SDataView clone(uint32_t _depth=~0u) const
            {
                SDataView retval;
                retval.composed = composed;
                retval.src.offset = src.offset;
                retval.src.size = src.size;
                if (_depth)
                    retval.src.buffer = core::smart_refctd_ptr_static_cast<ICPUBuffer>(src.buffer->clone(_depth));
                else
                    retval.src.buffer = core::smart_refctd_ptr(src.buffer);
                return retval;
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
        virtual inline ~IGeometry() = default;

        // needs to be hidden because of mutability checking
        inline bool setPositionView(SDataView&& view)
        {
            if (!view)
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
class NBL_API2 IIndexableGeometry : public IGeometry<BufferType>
{
    public:
        inline const SDataView& getIndexView() const {return m_indexView;}

        inline const uint64_t getIndexCount() const
        {
            return m_indexView.getElementCount();
        }

    protected:
        virtual ~IIndexableGeometry() = default;

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