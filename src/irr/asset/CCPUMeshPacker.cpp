#include "irr/asset/CCPUMeshPacker.h"
#include "irrlicht.h"

namespace irr 
{ 
namespace asset 
{

std::optional<CCPUMeshPacker::PackedMeshBufferData> CCPUMeshPacker::alloc(const ICPUMeshBuffer* meshBuffer)
{
	if (meshBuffer == nullptr)
		return std::nullopt;

	/*
	Requirements for input mesh buffers:
		- attributes bound to the same binding must have identical format
		- all meshbufers have indexed triangle list (temporary)
	*/

	//validation
	//TODO: remove this condition
	{
		auto* pipeline = meshBuffer->getPipeline();

		if (const_cast<ICPUMeshBuffer*>(meshBuffer)->getIndexBufferBinding()->buffer.get() == nullptr || 
			pipeline->getPrimitiveAssemblyParams().primitiveType != EPT_TRIANGLE_LIST)
		{
			_IRR_DEBUG_BREAK_IF(true);
			return std::nullopt;
		}
	}

	//validation
	const auto& mbVtxInputParams = meshBuffer->getPipeline()->getVertexInputParams();
	for (uint16_t attrBit = 0x0001, location = 0; location < 16; attrBit <<= 1, location++)
	{
		if (!(attrBit & mbVtxInputParams.enabledAttribFlags))
			continue;

		//assert((attrBit & m_outVtxInputParams.enabledAttribFlags));

		if (mbVtxInputParams.attributes[location].format != m_outVtxInputParams.attributes[location].format ||
			mbVtxInputParams.bindings[mbVtxInputParams.attributes[location].binding].inputRate != m_outVtxInputParams.bindings[location].inputRate)
		{
			_IRR_DEBUG_BREAK_IF(true);
			return std::nullopt;
		}
	}

	size_t addr = m_MDIDataAlctr.alloc_addr(1u, 1u);

	//TODO: divide into multiple mdi structs if(idxCnt > m_maxIndexCountPerMDIData)
	DrawElementsIndirectCommand_t* mdiBuffPtr = static_cast<DrawElementsIndirectCommand_t*>(m_outMDIData.get()->getPointer());
	*(mdiBuffPtr + addr) = 
	{
		static_cast<uint32_t>(meshBuffer->getIndexCount()), 
		static_cast<uint32_t>(meshBuffer->getInstanceCount()), 
		0, //because mesh buffers may be send to `commit` in any order, `firstIndex` and `baseVertex` should be set durning commit stage I guess
		0, 
		static_cast<uint32_t>(meshBuffer->getBaseInstance())
	};

	PackedMeshBufferData result{ addr * sizeof(DrawElementsIndirectCommand_t), 1u };
	return result;
}

CCPUMeshPacker::PackedMeshBuffer<ICPUMeshBuffer> CCPUMeshPacker::commit(const core::vector<std::pair<const ICPUMeshBuffer*, PackedMeshBufferData>>& meshBuffers)
{
	return {};
}

}
}