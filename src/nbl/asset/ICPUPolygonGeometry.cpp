#include "nbl/asset/ICPUPolygonGeometry.h"

#include <ranges>

using namespace nbl;
using namespace asset;


template<uint8_t Order> requires (Order>0)
class CListIndexingCB final : public IPolygonGeometryBase::IIndexingCallback
{
        template<typename OutT>
        static void operator_impl(SContext<OutT>& ctx)
        {
            auto indexOfIndex = ctx.beginPrimitive*3;
            for (const auto end=ctx.endPrimitive*3; indexOfIndex!=end; indexOfIndex+=3)
                ctx.streamOut(indexOfIndex,std::ranges::iota_view{0,int(Order)});
        }

    public:
        uint8_t degree_impl() const override {return Order;}
        uint8_t rate_impl() const override {return Order;}
        void operator()(SContext<uint8_t>& ctx) const override {operator_impl(ctx);}
        void operator()(SContext<uint16_t>& ctx) const override {operator_impl(ctx);}
        void operator()(SContext<uint32_t>& ctx) const override {operator_impl(ctx);}

        E_PRIMITIVE_TOPOLOGY knownTopology() const override
        {
            switch (Order)
            {
                case 1:
                    return EPT_POINT_LIST;
                case 2:
                    return EPT_LINE_LIST;
                case 3:
                    return EPT_TRIANGLE_LIST;
                default:
                    return EPT_PATCH_LIST;
            }
        }
};
auto IPolygonGeometryBase::PointList() -> IIndexingCallback*
{
    static CListIndexingCB<1> singleton;
    return &singleton;
}
auto IPolygonGeometryBase::LineList() -> IIndexingCallback*
{
    static CListIndexingCB<2> singleton;
    return &singleton;
}
auto IPolygonGeometryBase::TriangleList() -> IIndexingCallback*
{
    static CListIndexingCB<3> singleton;
    return &singleton;
}
auto IPolygonGeometryBase::QuadList() -> IIndexingCallback*
{
    static CListIndexingCB<4> singleton;
    return &singleton;
}

class CTriangleStripIndexingCB final : public IPolygonGeometryBase::IIndexingCallback
{
        template<typename OutT>
        static void operator_impl(SContext<OutT>& ctx)
        {
            uint64_t indexOfIndex;
            if (ctx.beginPrimitive==0)
            {
                ctx.streamOut(0,std::ranges::iota_view{0,3});
                indexOfIndex = 3;
            }
            else
                indexOfIndex = ctx.beginPrimitive+2;
            const int32_t perm[] = {-1,-2,0};
            for (const auto end=ctx.endPrimitive+2; std::cmp_not_equal(indexOfIndex, end); indexOfIndex++)
                ctx.streamOut(indexOfIndex,perm);
        }

    public:
        inline uint8_t degree_impl() const override { return 3; }
        inline uint8_t rate_impl() const override { return 1; }
        void operator()(SContext<uint8_t>& ctx) const override { operator_impl(ctx); }
        void operator()(SContext<uint16_t>& ctx) const override { operator_impl(ctx); }
        void operator()(SContext<uint32_t>& ctx) const override { operator_impl(ctx); }

        E_PRIMITIVE_TOPOLOGY knownTopology() const override {return EPT_TRIANGLE_STRIP;}
};
auto IPolygonGeometryBase::TriangleStrip() -> IIndexingCallback*
{
    static CTriangleStripIndexingCB singleton;
    return &singleton;
}

class CTriangleFanIndexingCB final : public IPolygonGeometryBase::IIndexingCallback
{
        template<typename OutT>
        static void operator_impl(SContext<OutT>& ctx)
        {
            uint64_t indexOfIndex;
            if (ctx.beginPrimitive==0)
            {
                ctx.streamOut(0,std::ranges::iota_view{0,3});
                indexOfIndex = 3;
            }
            else
                indexOfIndex = ctx.beginPrimitive+2;
            int32_t perm[] = {0x7eadbeefu,-1,0};
            for (const auto end=ctx.endPrimitive+2; std::cmp_not_equal(indexOfIndex, end); indexOfIndex++)
            {
                // first index is always global 0
                perm[0] = -indexOfIndex;
                ctx.streamOut(indexOfIndex,perm);
            }
        }

    public:
        inline uint8_t degree_impl() const override {return 3;}
        inline uint8_t rate_impl() const override {return 1;}
        void operator()(SContext<uint8_t>& ctx) const override { operator_impl(ctx); }
        void operator()(SContext<uint16_t>& ctx) const override { operator_impl(ctx); }
        void operator()(SContext<uint32_t>& ctx) const override { operator_impl(ctx); }

        E_PRIMITIVE_TOPOLOGY knownTopology() const override {return EPT_TRIANGLE_FAN;}
};
auto IPolygonGeometryBase::TriangleFan() -> IIndexingCallback*
{
    static CTriangleFanIndexingCB singleton;
    return &singleton;
}