#ifndef __IRR_I_GPU_MESH_H_INCLUDED__
#define __IRR_I_GPU_MESH_H_INCLUDED__

#include "irr/asset/IMesh.h"
#include "irr/video/IGPUMeshBuffer.h"

namespace irr
{
namespace video
{

    using IGPUMesh = asset::IMesh<video::IGPUMeshBuffer>;

}
}

#endif //__IRR_I_GPU_MESH_H_INCLUDED__