#include "nbl/asset/ICPUPolygonGeometry.h"

using namespace nbl;
using namespace asset;


template<uint8_t Order> requires (Order>0)
class CListIndexingCB final : public IPolygonGeometryBase::IIndexingCallback
{
    inline uint8_t degree_impl() const override {return Order;}
    inline uint8_t reuseCount_impl() const override {return 0;}
    inline void operator_impl(const SContext& ctx) const override
    {
        if (ctx.indexBuffer)
            memcpy(ctx.out,ctx.indexBuffer+(ctx.newIndexID<<ctx.indexSizeLog2),Order<<ctx.indexSizeLog2);
        else
        for (auto i=0u; i<Order; i++)
        {
            ctx.setOutput(ctx.newIndexID+i);
            ctx.out += ctx.indexSize();
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

class CTriangleStripIndexingCB final : public IIndexingCallback
{
    inline uint8_t degree_impl() const override {return 3;}
    inline uint8_t reuseCount_impl() const override {return 2;}
    inline void operator_impl(const SContext& ctx) const override
    {
        // two immediately previous and current
        memcpy(out,indexBuffer+(newIndexID-2)*indexSize,indexSize*3);
    }
};
auto IPolygonGeometryBase::TriangleStrip() -> IIndexingCallback*
{
    static CTriangleStripIndexingCB singleton;
    return &singleton;
}

class CTriangleFanIndexingCB final : public IIndexingCallback
{
    inline uint8_t degree_impl() const override {return 3;}
    inline uint8_t reuseCount_impl() const override {return 2;}
    inline void operator_impl(const SContext& ctx) const override
    {
        // first index is always 0
        memset(ctx.out,0,ctx.indexSize());
        ctx.out += ctx.indexSize();
        // immediately previous and current
        if (ctx.indexBuffer)
            memcpy(ctx.out,ctx.indexBuffer+((ctx.newIndexID-1)<<ctx.indexSizeLog2),2u<<ctx.indexSizeLog2);
        else
        {
            ctx.setOutput(ctx.newIndexID-1);
            ctx.setOutput(ctx.newIndexID);
        }
    }
};
auto IPolygonGeometryBase::TriangleFan() -> IIndexingCallback*
{
    static CTriangleFanIndexingCB singleton;
    return &singleton;
}