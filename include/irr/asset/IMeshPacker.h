#ifndef __IRR_I_MESH_PACKER_H_INCLUDED__
#define __IRR_I_MESH_PACKER_H_INCLUDED__
#include <cstdint>
#include <optional>

namespace irr
{
namespace asset
{
//where should I move it?
struct DrawElementsIndirectCommand_t
{
    uint32_t count;
    uint32_t instanceCount;
    uint32_t firstIndex;
    uint32_t baseVertex;
    uint32_t baseInstance;
};

template <typename MeshType>
class IMeshPacker
{
public:
    virtual std::optional<std::pair<MeshType* ,DrawElementsIndirectCommand_t>> packMeshes(core::vector<MeshType*>& meshes) = 0;

protected:
    virtual ~IMeshPacker() {}

    /*static_assert(std::is_base_of<IMesh<ICPUMeshBuffer>, MeshType>::value ||
                  std::is_base_of<IMesh<IGPUMeshBuffer>, MeshType>::value, "");*/
};

}
}

#endif