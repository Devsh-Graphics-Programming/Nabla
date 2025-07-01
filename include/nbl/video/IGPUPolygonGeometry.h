// Copyright (C) 2018-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_VIDEO_I_GPU_POLYGON_GEOMETRY_H_INCLUDED_
#define _NBL_VIDEO_I_GPU_POLYGON_GEOMETRY_H_INCLUDED_


#include "nbl/asset/IPolygonGeometry.h"

#include "nbl/video/IGPUBuffer.h"


namespace nbl::video
{

//
class IGPUPolygonGeometry final : public asset::IPolygonGeometry<const IGPUBuffer>
{
        using base_t = asset::IPolygonGeometry<const IGPUBuffer>;

	public:
        using SDataView = base_t::SDataView;
        struct SCreationParams
        {
            SDataView positionView = {};
            SDataView jointOBBView = {};
            SDataView indexView = {};
            const IIndexingCallback* indexing = nullptr;
            SDataView normalView = {};
            std::span<const SJointWeight> jointWeightViews = {};
            std::span<const SDataView> auxAttributeViews = {};
            uint32_t jointCount = 0;
        };
        static inline core::smart_refctd_ptr<IGPUPolygonGeometry> create(SCreationParams&& params)
        {
            auto retval = core::smart_refctd_ptr<IGPUPolygonGeometry>(new IGPUPolygonGeometry(),core::dont_grab);
            retval->m_positionView = params.positionView;
            if (params.jointCount)
                retval->m_jointOBBView = params.jointOBBView;
            retval->m_indexView = params.indexView;
            retval->m_indexing = params.indexing;
            if (params.jointCount)
                retval->m_jointWeightViews.insert(retval->m_jointWeightViews.begin(),params.jointWeightViews.begin(),params.jointWeightViews.end());
            retval->m_normalView = params.normalView;
            retval->m_auxAttributeViews.insert(retval->m_auxAttributeViews.begin(),params.auxAttributeViews.begin(),params.auxAttributeViews.end());
            retval->m_jointCount = params.jointCount;
            if (!retval->valid())
                return nullptr;
            return retval;
        }

        // passthrough
#if 0
        inline const SDataView& getNormalView() const {return base_t::getNormalView();}
        inline const core::vector<SJointWeight>& getJointWeightViews() const {return base_t::getJointWeightViews();}
        inline const core::vector<SDataView>& getAuxAttributeViews() const {return base_t::getAuxAttributeViews();}        
#endif

	private:
        inline IGPUPolygonGeometry() = default;
        inline ~IGPUPolygonGeometry() = default;
};

}

#endif