// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __E_MESH_WRITER_ENUMS_H_INCLUDED__
#define __E_MESH_WRITER_ENUMS_H_INCLUDED__

#include "irrTypes.h"
#include "irrMacros.h"

namespace irr
{
namespace scene
{

	//! An enumeration for all supported types of built-in mesh writers
	/** A scene mesh writers is represented by a four character code
	such as 'irrm' or 'coll' instead of simple numbers, to avoid
	name clashes with external mesh writers.*/
	enum EMESH_WRITER_TYPE
	{
		//! STL mesh writer for .stl files
		EMWT_STL          = MAKE_IRR_ID('s','t','l',0),

		//! OBJ mesh writer for .obj files
		EMWT_OBJ          = MAKE_IRR_ID('o','b','j',0),

		//! PLY mesh writer for .ply files
		EMWT_PLY          = MAKE_IRR_ID('p','l','y',0),

		//! BAW mesh writer for .baw files (custom BaW format)
		EMWT_BAW		  = MAKE_IRR_ID('b', 'a', 'w', 0)
	};


	//! flags configuring mesh writing
	enum E_MESH_WRITER_FLAGS
	{
		//! no writer flags
		EMWF_NONE = 0,

		//! write in a way that consumes less disk space
		EMWF_WRITE_COMPRESSED = 0x2,

		//! write in binary format rather than text
		EMWF_WRITE_BINARY = 0x4
	};

} // end namespace scene
} // end namespace irr


#endif // __E_MESH_WRITER_ENUMS_H_INCLUDED__

