#ifndef __IRR_I_GPU_MESH_H_INCLUDED__
#define __IRR_I_GPU_MESH_H_INCLUDED__

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

#endif //__IRR_I_GPU_MESH_H_INCLUDED__