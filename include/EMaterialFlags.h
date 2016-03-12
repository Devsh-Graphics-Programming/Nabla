// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __E_MATERIAL_FLAGS_H_INCLUDED__
#define __E_MATERIAL_FLAGS_H_INCLUDED__

namespace irr
{
namespace video
{

	//! Material flags
	enum E_MATERIAL_FLAG
	{
		//! Draw as wireframe or filled triangles? Default: false
		EMF_WIREFRAME = 0x1,

		//! Draw as point cloud or filled triangles? Default: false
		EMF_POINTCLOUD = 0x2,

		//! Is the ZBuffer enabled? Default: true
		EMF_ZBUFFER = 0x10,

		//! May be written to the zbuffer or is it readonly. Default: true
		/** This flag is ignored, if the material type is a transparent type. */
		EMF_ZWRITE_ENABLE = 0x20,

		//! Is backface culling enabled? Default: true
		EMF_BACK_FACE_CULLING = 0x40,

		//! Is frontface culling enabled? Default: false
		/** Overrides EMF_BACK_FACE_CULLING if both are enabled. */
		EMF_FRONT_FACE_CULLING = 0x80,

		//! AntiAliasing mode
		EMF_ANTI_ALIASING = 0x4000,

		//! ColorMask bits, for enabling the color planes
		EMF_COLOR_MASK = 0x8000,

		//! ColorMaterial enum for vertex color interpretation
		EMF_COLOR_MATERIAL = 0x10000,

		//! Flag for blend operation
		EMF_BLEND_OPERATION = 0x40000,

		//! Flag for polygon offset
		EMF_POLYGON_OFFSET = 0x80000
	};

} // end namespace video
} // end namespace irr


#endif // __E_MATERIAL_FLAGS_H_INCLUDED__

