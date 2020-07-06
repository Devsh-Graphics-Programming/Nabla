#ifndef __IRR_C_CPU_MESH_PACKER_H_INCLUDED__
#define __IRR_C_CPU_MESH_PACKER_H_INCLUDED__

#include <irr/asset/ICPUMesh.h>
#include <irr/asset/IMeshPacker.h>
#include <irr/core/math/intutil.h>

namespace irr 
{ 
namespace asset
{

using namespace meshPackerUtil;

template <typename MDIStructType = DrawElementsIndirectCommand_t>
class CCPUMeshPacker final : public IMeshPacker<ICPUMeshBuffer, MDIStructType>
{
public:
	CCPUMeshPacker(const SVertexInputParams& preDefinedLayout, const AllocationParams& allocParams, uint16_t minTriangleCountPerMDIData = 256u, uint16_t maxTriangleCountPerMDIData = 1024u)
		:IMeshPacker<ICPUMeshBuffer, MDIStructType>(preDefinedLayout, allocParams, minTriangleCountPerMDIData, maxTriangleCountPerMDIData)
	{
		m_outMDIData = core::make_smart_refctd_ptr<ICPUBuffer>(allocParams.MDIDataBuffSupportedSizeInBytes);
		m_outIdxBuffer = core::make_smart_refctd_ptr<ICPUBuffer>(allocParams.indexBuffSupportedSizeInBytes);
	}

	template <typename Iterator>
	ReservedAllocationMeshBuffers alloc(const Iterator begin, const Iterator end);
	virtual void commit() override;

	ICPUBuffer* getMultiDrawIndirectBuffer() { return m_outMDIData.get(); }

private: 
	core::smart_refctd_ptr<ICPUBuffer> m_outMDIData;
	core::smart_refctd_ptr<ICPUBuffer> m_outIdxBuffer;
};

template <typename MDIStructType>
//`Iterator` may be only an Iterator or pointer to pointer
template <typename Iterator>
ReservedAllocationMeshBuffers CCPUMeshPacker<MDIStructType>::alloc(const Iterator begin, const Iterator end)
{

	for (auto it = begin; it != end; it++)
	{
		if (*it == nullptr)
			return invalidReservedAllocationMeshBuffers;
	}
	

	/*
	Requirements for input mesh buffers:
		- attributes bound to the same binding must have identical format
		- all meshbufers have indexed triangle list (temporary)
	*/

	//validation
	//TODO: remove this condition
	for(auto it = begin; it != end; it++)
	{
		auto* pipeline = (*it)->getPipeline();

		auto a = (*it)->getIndexBufferBinding()->buffer;

		if ((*it)->getIndexBufferBinding()->buffer.get() == nullptr ||
			pipeline->getPrimitiveAssemblyParams().primitiveType != EPT_TRIANGLE_LIST)
		{
			_IRR_DEBUG_BREAK_IF(true);
			return invalidReservedAllocationMeshBuffers;
		}
	}

	//validation
	for (auto it = begin; it != end; it++)
	{
		const auto& mbVtxInputParams = (*it)->getPipeline()->getVertexInputParams();
		for (uint16_t attrBit = 0x0001, location = 0; location < 16; attrBit <<= 1, location++)
		{
			if (!(attrBit & mbVtxInputParams.enabledAttribFlags))
				continue;

			//assert((attrBit & m_outVtxInputParams.enabledAttribFlags));

			if (mbVtxInputParams.attributes[location].format != m_outVtxInputParams.attributes[location].format ||
				mbVtxInputParams.bindings[mbVtxInputParams.attributes[location].binding].inputRate != m_outVtxInputParams.bindings[location].inputRate)
			{
				_IRR_DEBUG_BREAK_IF(true);
				return invalidReservedAllocationMeshBuffers;
			}
		}
	}
	
	const size_t idxCnt = std::accumulate(begin, end, 0ull, [](size_t init, ICPUMeshBuffer* mb) { return init + mb->getIndexCount(); });
	const size_t vtxCnt = std::accumulate(begin, end, 0ull, [](size_t init, ICPUMeshBuffer* mb) { return init + mb->calcVertexCount(); });

	const uint32_t minIdxCntPerPatch = m_minTriangleCountPerMDIData * 3;
	const uint32_t possibleMDIStructsNeededCnt = core::roundUp<uint32_t>(idxCnt, minIdxCntPerPatch);

	uint32_t MDIAllocAddr       = INVALID_ADDRESS;
	uint32_t idxAllocAddr       = INVALID_ADDRESS;
	uint32_t vtxAllocAddr       = INVALID_ADDRESS;
	uint32_t perInsVtxAllocAddr = INVALID_ADDRESS;

	//actually it will not work at all if m_MDIDataAlctrResSpc == nullptr or m_idxBuffAlctrResSpc == nullptr.. TODO
	if (m_MDIDataAlctrResSpc)
	{
		MDIAllocAddr = m_MDIDataAlctr.alloc_addr(possibleMDIStructsNeededCnt, 1u);
		if (MDIAllocAddr == INVALID_ADDRESS)
			return invalidReservedAllocationMeshBuffers;
	}
	
	if (m_idxBuffAlctrResSpc)
	{
		idxAllocAddr = m_idxBuffAlctr.alloc_addr(idxCnt, 1u);
		if (idxAllocAddr == INVALID_ADDRESS)
			return invalidReservedAllocationMeshBuffers;
	}
	
	if (m_vtxBuffAlctrResSpc)
	{
		vtxAllocAddr = m_vtxBuffAlctr.alloc_addr(vtxCnt, 1u);
		if (vtxAllocAddr == INVALID_ADDRESS)
			return invalidReservedAllocationMeshBuffers;
	}
	
	if (m_perInsVtxBuffAlctrResSpc)
	{
		perInsVtxAllocAddr = m_perInsVtxBuffAlctr.alloc_addr(vtxCnt, 1u);
		if (perInsVtxAllocAddr == INVALID_ADDRESS)
			return invalidReservedAllocationMeshBuffers;
	}

	ReservedAllocationMeshBuffers result{
		MDIAllocAddr,
		possibleMDIStructsNeededCnt,
		perInsVtxAllocAddr,
		perInsVtxAllocAddr == INVALID_ADDRESS ? 0u : vtxCnt,
		idxAllocAddr,
		idxCnt,
		vtxAllocAddr,
		vtxAllocAddr == INVALID_ADDRESS ? 0u : vtxCnt
	};
	return result;
}

template <typename MDIStructType>
void CCPUMeshPacker<MDIStructType>::commit()
{
	//for (int i = 0; i < MDIStructsNeeded; i++)
	//{
	//	uint32_t idxCnt = 0u;

	//	if (i == MDIStructsNeeded - 1)
	//	{
	//		idxCnt = static_cast<uint32_t>(meshBuffer->getIndexCount()) % maxIdxCntPerPatch;

	//		//TODO: test for this case
	//		if (idxCnt == 0)
	//			idxCnt = maxIdxCntPerPatch;
	//	}
	//	else
	//	{
	//		idxCnt = maxIdxCntPerPatch;
	//	}

	//	*(mdiBuffPtr + MDIAllocAddr + i) =
	//	{
	//		idxCnt,
	//		static_cast<uint32_t>(meshBuffer->getInstanceCount()),
	//		idxAllocAddr + i * maxIdxCntPerPatch,
	//		0 /*TODO*/,
	//		static_cast<uint32_t>(meshBuffer->getBaseInstance()) /*TODO*/
	//	};
	//}

	return;
}

}
}

#endif