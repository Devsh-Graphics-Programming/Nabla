// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __I_GPU_MESH_BUFFER_H_INCLUDED__
#define __I_GPU_MESH_BUFFER_H_INCLUDED__

#include <algorithm>

#include "irr/asset/asset.h"
#include "IGPUBuffer.h"

namespace irr
{
namespace video
{
	// will be replaced by graphics pipeline layout object
	class IGPUMeshDataFormatDesc : public asset::IMeshDataFormatDesc<video::IGPUBuffer>
	{
	};

	class IGPUMeshBuffer final : public asset::IMeshBuffer<video::IGPUBuffer>
	{
	};

} // end namespace video
} // end namespace irr



#endif


