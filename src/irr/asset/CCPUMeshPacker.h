#ifndef __IRR_C_CPU_MESH_PACKER_H_INCLUDED__
#define __IRR_C_CPU_MESH_PACKER_H_INCLUDED__

#include <irr/asset/ICPUMesh.h>
#include <irr/asset/IMeshPacker.h>

namespace irr 
{ 
namespace asset
{

using namespace meshPackerUtil;

template <typename MDIStructType = DrawElementsIndirectCommand_t>
class CCPUMeshPacker final : public IMeshPacker<ICPUMeshBuffer, MDIStructType>
{
public:
	CCPUMeshPacker(const SVertexInputParams& preDefinedLayout, const AllocationParams& allocParams, uint16_t maxIndexCountPerMDIData = std::numeric_limits<uint16_t>::max())
		:IMeshPacker<ICPUMeshBuffer, MDIStructType>(preDefinedLayout, allocParams, maxIndexCountPerMDIData) 
	{
		m_outMDIData = core::make_smart_refctd_ptr<ICPUBuffer>(allocParams.MDIDataBuffSupportedSizeInBytes);
		m_outIdxBuffer = core::make_smart_refctd_ptr<ICPUBuffer>(allocParams.indexBuffSupportedSizeInBytes);
	}

	virtual PackedMeshBufferData alloc(const ICPUMeshBuffer const* meshBuffer) override;
	virtual void commit() override;

	ICPUBuffer* getMultiDrawIndirectBuffer() { return m_outMDIData.get(); }

private: 
	core::smart_refctd_ptr<ICPUBuffer> m_outMDIData;
	core::smart_refctd_ptr<ICPUBuffer> m_outIdxBuffer;
};

template <typename MDIStructType>
PackedMeshBufferData CCPUMeshPacker<MDIStructType>::alloc(const ICPUMeshBuffer const* meshBuffer)
{
	if (meshBuffer == nullptr)
		return invalidPackedMeshBufferData;

	/*
	Requirements for input mesh buffers:
		- attributes bound to the same binding must have identical format
		- all meshbufers have indexed triangle list (temporary)
	*/

	//validation
	//TODO: remove this condition
	{
		auto* pipeline = meshBuffer->getPipeline();

		auto a = meshBuffer->getIndexBufferBinding().buffer;

		if (meshBuffer->getIndexBufferBinding().buffer.get() == nullptr ||
			pipeline->getPrimitiveAssemblyParams().primitiveType != EPT_TRIANGLE_LIST)
		{
			_IRR_DEBUG_BREAK_IF(true);
			return invalidPackedMeshBufferData;
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
			return invalidPackedMeshBufferData;
		}
	}

	auto MDIAllocAddr = m_MDIDataAlctr.alloc_addr(1u, 1u);
	if (MDIAllocAddr == m_MDIDataAlctr.invalid_address)
		return invalidPackedMeshBufferData;

	auto idxAllocAddr = m_idxBuffAlctr.alloc_addr(meshBuffer->getIndexCount(), 1u);
	if (idxAllocAddr == m_idxBuffAlctr.invalid_address)
		return invalidPackedMeshBufferData;

	//TODO: divide into multiple mdi structs if(idxCnt > m_maxIndexCountPerMDIData)
	MDIStructType* mdiBuffPtr = static_cast<MDIStructType*>(m_outMDIData.get()->getPointer());
	*(mdiBuffPtr + MDIAllocAddr) =
	{
		static_cast<uint32_t>(meshBuffer->getIndexCount()),
		static_cast<uint32_t>(meshBuffer->getInstanceCount()),
		idxAllocAddr * sizeof(uint16_t),
		0,
		static_cast<uint32_t>(meshBuffer->getBaseInstance())
	};



	PackedMeshBufferData result{ MDIAllocAddr * sizeof(MDIStructType), 1u };
	return result;
}

template <typename MDIStructType>
void CCPUMeshPacker<MDIStructType>::commit()
{
	return;
}

}
}

#endif