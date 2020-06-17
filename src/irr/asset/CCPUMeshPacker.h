#ifndef __IRR_C_CPU_MESH_PACKER_H_INCLUDED__
#define __IRR_C_CPU_MESH_PACKER_H_INCLUDED__

#include <irr/asset/ICPUMesh.h>
#include <irr/asset/IMeshPacker.h>

namespace irr 
{ 
namespace asset
{

class CCPUMeshPacker final : public IMeshPacker<ICPUMeshBuffer>
{
public:
	CCPUMeshPacker(const SVertexInputParams& preDefinedLayout, const AllocationParams& allocParams, uint16_t maxIndexCountPerMDIData = std::numeric_limits<uint16_t>::max())
		:IMeshPacker<ICPUMeshBuffer>(preDefinedLayout, allocParams, maxIndexCountPerMDIData) {}

	virtual std::optional<CCPUMeshPacker::PackedMeshBufferData> alloc(const ICPUMeshBuffer* meshBuffer) override;
	virtual PackedMeshBuffer<ICPUMeshBuffer> commit(const core::vector<std::pair<const ICPUMeshBuffer*, PackedMeshBufferData>>& meshBuffers) override;

};

}
}

#endif