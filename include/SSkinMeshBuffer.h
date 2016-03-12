// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __I_SKIN_MESH_BUFFER_H_INCLUDED__
#define __I_SKIN_MESH_BUFFER_H_INCLUDED__

#include "IMeshBuffer.h"


namespace irr
{
namespace scene
{


struct SCPUSkinMeshBuffer : public ICPUMeshBuffer
{
	//! Default constructor
	SCPUSkinMeshBuffer() : normalAttrId(EVAI_ATTR3), indexValMin(0), indexValMax(0), BoundingBoxNeedsRecalculated(true)
	{
		#ifdef _DEBUG
		setDebugName("SCPUSkinMeshBuffer");
		#endif
	}

    inline void setIndexRange(const uint32_t &minBeforeBaseVxAdd, const uint32_t &maxBeforeBaseVxAdd)
    {
        indexValMin = minBeforeBaseVxAdd;
        indexValMax = maxBeforeBaseVxAdd;
    }

    inline const uint32_t& getIndexMinBound() const {return indexValMin;}
    inline const uint32_t& getIndexMaxBound() const {return indexValMax;}

    inline const E_VERTEX_ATTRIBUTE_ID& getNormalAttributeIx() const {return normalAttrId;}
    inline void setNormalAttributeIx(const E_VERTEX_ATTRIBUTE_ID& attrId)
    {
        if (attrId>=EVAI_COUNT)
#ifdef _DEBUG
        {
            //os::Printer::log("MeshBuffer setPositionAttributeIx attribute ID out of range!\n",ELL_ERROR);
            return;
        }
#else
            return;
#endif // _DEBUG

        normalAttrId = attrId;
    }


	//! Call this after changing the positions of any vertex.
	void boundingBoxNeedsRecalculated(void) { BoundingBoxNeedsRecalculated = true; }



    E_VERTEX_ATTRIBUTE_ID normalAttrId;

    uint32_t indexValMin;
    uint32_t indexValMax;

	core::matrix4 Transformation;

	bool BoundingBoxNeedsRecalculated;
};


} // end namespace scene
} // end namespace irr

#endif

