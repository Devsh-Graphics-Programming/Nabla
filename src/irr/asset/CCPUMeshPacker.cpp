#include "irr/asset/CCPUMeshPacker.h"
#include "irrlicht.h"

namespace irr 
{ 
namespace asset 
{

std::optional<std::pair<ICPUMesh*, DrawElementsIndirectCommand_t>> CCPUMeshPacker::packMeshes(core::vector<ICPUMesh*>& meshes)
{
	core::vector<ICPUMeshBuffer*> meshBuffers;

	for (auto* mesh : meshes)
	{
		for (uint32_t i = 0; i < mesh->getMeshBufferCount(); i++)
		{
			meshBuffers.push_back(mesh->getMeshBuffer(i));
		}
	}

	if (meshBuffers.empty())
	{
		_IRR_DEBUG_BREAK_IF(true);
		return {};
	}

	/*
	Requirements for input mesh buffers:
		- the same number and size of attributes
		- all meshbufers have indexed triangle list (temporary)
	*/

	//validation
	for (auto meshBuffer : meshBuffers)
	{
		auto* pipeline = meshBuffer->getPipeline();
		auto* layout = pipeline->getLayout();

		if (meshBuffer->getIndexBufferBinding()->buffer.get() == nullptr || 
			pipeline->getPrimitiveAssemblyParams().primitiveType == EPT_TRIANGLE_LIST)
		{
			_IRR_DEBUG_BREAK_IF(true);
			return {};
		}
	}

	for (int i = 1; i < meshBuffers.size(); i++)
	{
		const auto& vxInputParamsRef = meshBuffers[0]->getPipeline()->getVertexInputParams();
		const auto& vxInputParams = meshBuffers[i]->getPipeline()->getVertexInputParams();

		if (std::memcmp(&vxInputParamsRef, &vxInputParams, sizeof(SVertexInputParams)))
		{
			_IRR_DEBUG_BREAK_IF(true);
			return {};
		}
	}

	return {};
}

}
}