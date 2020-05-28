#ifndef __IRR_C_CPU_MESH_PACKER_H_INCLUDED__
#define __IRR_C_CPU_MESH_PACKER_H_INCLUDED__

#include <irr/asset/ICPUMesh.h>
#include <irr/asset/IMeshPacker.h>

namespace irr 
{ 
namespace asset
{

class CCPUMeshPacker : public IMeshPacker<ICPUMesh>
{
public:
	virtual std::optional<std::pair<ICPUMesh*, DrawElementsIndirectCommand_t>> packMeshes(core::vector<ICPUMesh*>& meshes) override;

};

}
}

#endif