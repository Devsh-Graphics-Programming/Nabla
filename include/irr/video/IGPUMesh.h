// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_VIDEO_I_GPU_MESH_H_INCLUDED__
#define __NBL_VIDEO_I_GPU_MESH_H_INCLUDED__

#include "irr/asset/IMesh.h"
#include "irr/video/IGPUMeshBuffer.h"

namespace irr
{
namespace video
{

class IGPUMesh : public asset::IMesh<IGPUMeshBuffer>
{
	public:
		virtual asset::E_MESH_TYPE getMeshType() const override { return asset::EMT_NOT_ANIMATED; }
};

}
}

#endif