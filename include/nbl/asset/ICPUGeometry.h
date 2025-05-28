// Copyright (C) 2025-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_ASSET_I_CPU_POLYGON_GEOMETRY_H_INCLUDED_
#define _NBL_ASSET_I_CPU_POLYGON_GEOMETRY_H_INCLUDED_


#include "nbl/asset/IPolygonGeometry.h"


namespace nbl::asset
{
//
class NBL_API2 ICPUPolygonGeometry : public IAsset, public IPolygonGeometry<ICPUBuffer>
{
    public:
        inline ICPUPolygonGeometry() = default;
        
        constexpr static inline auto AssetType = ET_POLYGON_GEOMETRY;
        inline E_TYPE getAssetType() const override {return AssetType;}

        // TODO: Asset methods

#if 0
        // don't want to play more inheritance games and add methods to `SDataView`
        static xxx(SDataView)
        {
            //
        }
#endif
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

// TODO: set IPolygonGeometry members
};

}
#endif