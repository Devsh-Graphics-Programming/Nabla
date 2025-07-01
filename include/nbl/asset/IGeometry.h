// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_ASSET_I_GEOMETRY_H_INCLUDED_
#define _NBL_ASSET_I_GEOMETRY_H_INCLUDED_


#include "nbl/builtin/hlsl/shapes/aabb.hlsl"

#include "nbl/asset/IAsset.h"
#include "nbl/asset/format/EFormat.h"


namespace nbl::asset
{
class IGeometryBase : public virtual core::IReferenceCounted
{
    public:
        // used for same purpose as and overlaps `IAsset::valid()`
        virtual bool valid() const = 0;

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
        //
        static inline EAABBFormat getMatchingAABBFormat(const E_FORMAT attributeFormat)
        {
            if (isBlockCompressionFormat(attributeFormat))
                return EAABBFormat::BitCount;
            if (isFloatingPointFormat(attributeFormat))
            {
                const auto maxVal = getFormatMaxValue<double>(attributeFormat,0);
                if (maxVal>hlsl::numeric_limits<hlsl::float32_t>::max)
                    return EAABBFormat::F64;
                if (maxVal>hlsl::numeric_limits<hlsl::float16_t>::max)
                    return EAABBFormat::F32;
                return EAABBFormat::F16;
            }
            else if (isNormalizedFormat(attributeFormat))
            {
                const auto precision = getFormatPrecision<float>(attributeFormat,0,0.f);
                const auto minVal = getFormatMinValue<float>(attributeFormat,0);
                if (minVal<-0.f)
                    return precision<getFormatPrecision<float>(EF_R8_SNORM,0,0.f) ? EAABBFormat::S16_NORM:EAABBFormat::S8_NORM;
                else
                    return precision<getFormatPrecision<float>(EF_R8_UNORM,0,0.f) ? EAABBFormat::U16_NORM:EAABBFormat::U8_NORM;
            }
            else if (isIntegerFormat(attributeFormat))
            {
                if (isSignedFormat(attributeFormat))
                {
                    const auto maxVal = getFormatMaxValue<int64_t>(attributeFormat,0);
                    if (maxVal>hlsl::numeric_limits<int32_t>::max)
                        return EAABBFormat::S64;
                    else if (maxVal>hlsl::numeric_limits<int16_t>::max)
                        return EAABBFormat::S32;
                    else if (maxVal>hlsl::numeric_limits<int8_t>::max)
                        return EAABBFormat::S16;
                    return EAABBFormat::S8;
                }
                else
                {
                    const auto maxVal = getFormatMaxValue<uint64_t>(attributeFormat,0);
                    if (maxVal>hlsl::numeric_limits<uint32_t>::max)
                        return EAABBFormat::U64;
                    else if (maxVal>hlsl::numeric_limits<uint16_t>::max)
                        return EAABBFormat::U32;
                    else if (maxVal>hlsl::numeric_limits<uint8_t>::max)
                        return EAABBFormat::U16;
                    return EAABBFormat::U8;

                }
            }
            return EAABBFormat::BitCount;
        }
        // using `nbl::hlsl::` concepts instead of `std::` so that `nbl::hlsl::float16_t` can be used
        union SAABBStorage
        {
            template<typename Visitor>
            inline void visit(const EAABBFormat format, Visitor&& visitor)
            {
                switch (format)
                {
                    case EAABBFormat::F64:
                        visitor(f64);
                        break;
                    case EAABBFormat::U64:
                        visitor(u64);
                        break;
                    case EAABBFormat::S64:
                        visitor(s64);
                        break;
                    case EAABBFormat::F32:
                        visitor(f32);
                        break;
                    case EAABBFormat::U32:
                        visitor(u32);
                        break;
                    case EAABBFormat::S32:
                        visitor(s32);
                        break;
                    case EAABBFormat::F16:
                        visitor(f16);
                        break;
                    case EAABBFormat::U16: [[fallthrough]];
                    case EAABBFormat::U16_NORM:
                        visitor(u16);
                        break;
                    case EAABBFormat::S16: [[fallthrough]];
                    case EAABBFormat::S16_NORM:
                        visitor(s16);
                        break;
                    case EAABBFormat::U8: [[fallthrough]];
                    case EAABBFormat::U8_NORM:
                        visitor(u8);
                        break;
                    case EAABBFormat::S8: [[fallthrough]];
                    case EAABBFormat::S8_NORM:
                        visitor(s8);
                        break;
                    default:
                        break;
                }
            }
            template<typename Visitor>
            inline void visit(const EAABBFormat format, Visitor&& visitor) const
            {
                const_cast<SAABBStorage*>(this)->visit(format,std::forward<Visitor>(visitor));
            }

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
            inline operator bool() const {return format==EF_UNKNOWN || !isBlockCompressionFormat(format) && !isDepthOrStencilFormat(format);}

            //
            inline bool isFormatted() const {return format!=EF_UNKNOWN && bool(*this);}

            // Useful for checking if something can be used as an index
            inline bool isFormattedScalarInteger() const
            {
                if (isFormatted())
                switch (format)
                {
                    case EF_R8_SINT: [[fallthrough]];
                    case EF_R8_UINT: [[fallthrough]];
                    case EF_R16_SINT: [[fallthrough]];
                    case EF_R16_UINT: [[fallthrough]];
                    case EF_R32_SINT: [[fallthrough]];
                    case EF_R32_UINT: [[fallthrough]];
                    case EF_R64_SINT: [[fallthrough]];
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
            inline void visitAABB(Visitor&& visitor) {encodedDataRange.visit(rangeFormat,std::forward<Visitor>(visitor));}
            template<typename Visitor>
            inline void visitAABB(Visitor&& visitor) const {encodedDataRange.visit(rangeFormat,std::forward<Visitor>(visitor));}

            //
            inline void resetRange(const EAABBFormat newFormat)
            {
                rangeFormat = newFormat;
                auto tmp = [](auto& aabb)->void{aabb = aabb.create();};
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

        virtual EAABBFormat getAABBFormat() const = 0;
        virtual const SAABBStorage& getAABB() const = 0;
        template<typename Visitor>
        inline void visitAABB(Visitor&& visitor) const
        {
            getAABB().visit(getAABBFormat(),std::forward<Visitor>(visitor));
        }

    protected:
        virtual inline ~IGeometryBase() = default;
};

// a thing to expose `clone()` conditionally via inheritance and `conditional_t`
namespace impl
{
class NBL_FORCE_EBO NBL_NO_VTABLE INotCloneable {};
}

// A geometry should map 1:1 to a BLAS geometry, Meshlet or a Drawcall in API terms
template<class BufferType>
class IGeometry : public std::conditional_t<std::is_same_v<BufferType,ICPUBuffer>,IAsset,impl::INotCloneable>, public IGeometryBase
{
    public:
        //
        virtual inline bool valid() const override
        {
            if (!m_positionView)
                return false;
            if (getPrimitiveCount()==0)
                return false;
            // joint OBBs are optional
            return true;
        }

        struct SDataView
        {
            inline operator bool() const {return src && composed;}

            //
            explicit inline operator SBufferBinding<const BufferType>() const
            {
                if (*this)
                    return {.offset=src.offset,.buffer=core::smart_refctd_ptr(src.buffer)};
                return {};
            }

            inline uint64_t getElementCount() const
            {
                if (!this->operator bool())
                    return 0ull;
                const auto stride = composed.getStride();
                if (stride==0)
                    return 0ull;
                return src.size/stride;
            }
            
            //
            template<typename Index=uint32_t, typename U=BufferType> requires (std::is_same_v<U,BufferType> && std::is_same_v<U,ICPUBuffer>)
            inline const void* getPointer(const Index elIx=0) const
            {
                if (*this)
                    return reinterpret_cast<const uint8_t*>(src.buffer->getPointer())+src.offset+elIx*composed.getStride();
                return nullptr;
            }
            template<typename Index=uint32_t, typename U=BufferType> requires (std::is_same_v<U,BufferType> && std::is_same_v<U,ICPUBuffer>)
            inline void* getPointer(const Index elIx=0)
            {
                if (*this)
                    return reinterpret_cast<uint8_t*>(src.buffer->getPointer())+src.offset+elIx*composed.getStride();
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
                        using traits = hlsl::vector_traits<V>;
                        const auto range = composed.getRange<hlsl::shapes::AABB<traits::Dimension,typename traits::scalar_type>>();
                        for (auto i=0u; i<traits::Dimension; i++)
                        {
                            if (isNormalizedFormat(composed.format))
                            {
                                v[i] = tmp[i] * (range.maxVx[i] - range.minVx[i]) + range.minVx[i];
                            }
                            else
                                v[i] = tmp[i];
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
                const auto range = composed.getRange<traits::Dimension,typename traits::scalar_type>();
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
class IIndexableGeometry : public IGeometry<BufferType>
{
    protected:
        using SDataView = IGeometry<BufferType>::SDataView;

    public:
        // index buffer is optional so no override of `valid()`

        inline const SDataView& getIndexView() const {return m_indexView;}

        inline const uint64_t getIndexCount() const
        {
            return m_indexView.getElementCount();
        }

    protected:
        virtual inline ~IIndexableGeometry() = default;

        // Needs to be hidden because ICPU base class shall check mutability
        inline bool setIndexView(SDataView&& view)
        {
            if (!view || view.composed.isFormattedScalarInteger())
            {
                m_indexView = std::move(view);
                return true;
            }
            return false;
        }

        //
        SDataView m_indexView = {};
};

}

//
namespace nbl::core
{
template<typename Dummy>
struct blake3_hasher::update_impl<asset::IGeometryBase::SDataViewBase,Dummy>
{
	static inline void __call(blake3_hasher& hasher, const asset::IGeometryBase::SDataViewBase& input)
	{
        hasher << input.stride;
        hasher << input.format;
        hasher << input.rangeFormat;
        input.visitAABB([&hasher](auto& aabb)->void{hasher.update(&aabb,sizeof(aabb));});
	}
};
}
#endif