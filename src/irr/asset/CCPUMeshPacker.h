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

	//needs to be called before first commit
	void createOutputBuffers();

	template <typename Iterator>
	PackedMeshBufferData commit(const Iterator begin, const Iterator end, ReservedAllocationMeshBuffers ramb);

	inline PackedMeshBuffer<ICPUBuffer>& getPackedMeshBuffer() { return outputBuffer; };

private:
	PackedMeshBuffer<ICPUBuffer> outputBuffer;

	//configures indices and MDI structs (implementation is not ready yet)
	template<typename IndexType>
	PackedMeshBufferData processMeshBuffer(ICPUMeshBuffer* inputMeshBuffer, ReservedAllocationMeshBuffers& ramb);

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
		for (uint16_t attrBit = 0x0001, location = 0; location < SVertexInputParams::MAX_ATTR_BUF_BINDING_COUNT; attrBit <<= 1, location++)
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
void CCPUMeshPacker<MDIStructType>::createOutputBuffers()
{
	if (outputBuffer.indexBuffer.buffer != nullptr)
		return;

	//TODO: redo after safe_shrink fix
	outputBuffer.MDIDataBuffer = core::make_smart_refctd_ptr<ICPUBuffer>(m_allocParams.MDIDataBuffSupportedSizeInBytes);
	outputBuffer.indexBuffer.buffer = core::make_smart_refctd_ptr<ICPUBuffer>(m_allocParams.indexBuffSupportedSizeInBytes);
	outputBuffer.perVertexVtxBuffer.buffer = core::make_smart_refctd_ptr<ICPUBuffer>(m_allocParams.vertexBuffSupportedSizeInBytes);
	if(m_perInstVtxSize)
		outputBuffer.perInstanceVtxBuffer.buffer = core::make_smart_refctd_ptr<ICPUBuffer>(m_allocParams.perInstanceVertexBuffSupportedSizeInBytes);
	outputBuffer.perVertexVtxBuffer.offset = 0u;
	outputBuffer.perInstanceVtxBuffer.offset = 1u;
}

template<typename MDIStructType>
template<typename IndexType>
PackedMeshBufferData CCPUMeshPacker<MDIStructType>::processMeshBuffer(ICPUMeshBuffer* inputMeshBuffer, ReservedAllocationMeshBuffers& ramb)
{
	MDIStructType* mdiBuffPtr = static_cast<MDIStructType*>(outputBuffer.MDIDataBuffer->getPointer()) + ramb.mdiAllocationOffset;
	uint16_t* indexBuffPtr = static_cast<uint16_t*>(outputBuffer.indexBuffer.buffer->getPointer()) + ramb.indexAllocationOffset;

	const uint64_t currMeshBufferIdxCnt = inputMeshBuffer->getIndexCount();
	const size_t MDIStructsNeeded = core::roundUp<size_t>(currMeshBufferIdxCnt, m_maxTriangleCountPerMDIData);

	uint32_t firstIdxForCurrMeshBatch = ramb.indexAllocationOffset;

	//assuming it's never EIT_UNKNOWN
	IndexType* idxBuffBatchBegin = static_cast<IndexType*>(inputMeshBuffer->getIndexBufferBinding()->buffer->getPointer());
	IndexType* idxBuffBatchEnd = nullptr;

	/*struct ReservedAllocationMeshBuffers
	{
		uint32_t mdiAllocationOffset;
		uint32_t mdiAllocationReservedSize;
		uint32_t instanceAllocationOffset;
		uint32_t instanceAllocationReservedSize;
		uint32_t indexAllocationOffset;
		uint32_t indexAllocationReservedSize;
		uint32_t vertexAllocationOffset;
		uint32_t vertexAllocationReservedSize;
	}*/

	//set vertex data
	//memcpy()

	for (uint64_t i = 0; i < MDIStructsNeeded; i++)
	{
		idxBuffBatchEnd = idxBuffBatchBegin + 
			((i == (MDIStructsNeeded - 1)) ? currMeshBufferIdxCnt % m_maxTriangleCountPerMDIData : m_maxTriangleCountPerMDIData);
		const uint32_t minIdxVal = *std::min_element(idxBuffBatchBegin, idxBuffBatchEnd);

		MDIStructType MDIData;
		MDIData.count = idxBuffBatchEnd - idxBuffBatchBegin;
		MDIData.instanceCount = inputMeshBuffer->getInstanceCount();
		MDIData.firstIndex = firstIdxForCurrMeshBatch;
		MDIData.baseVertex = *std::min_element(idxBuffBatchBegin, idxBuffBatchEnd);
		MDIData.baseInstance = 0u; /*TODO*/

		//set mdi structs
		memcpy(mdiBuffPtr, &MDIData, sizeof(MDIStructType));
		
		//set indices
		//TODO?: if constexpr for EIT
		for (IndexType* ptr = idxBuffBatchBegin; ptr != idxBuffBatchEnd; ptr++)
		{
			*indexBuffPtr = *ptr - minIdxVal;
			indexBuffPtr++;
		}

		firstIdxForCurrMeshBatch += MDIData.count;
		idxBuffBatchBegin = idxBuffBatchEnd;

		mdiBuffPtr++;
		indexBuffPtr += MDIData.count;
	}

	return {};
}

template <typename MDIStructType>
template <typename Iterator>
PackedMeshBufferData CCPUMeshPacker<MDIStructType>::commit(const Iterator begin, const Iterator end, ReservedAllocationMeshBuffers ramb)
{
	//AFTER SAFE SHRINK FIX TODO LIST:
	//multiple ReservedAllocationMeshbuffers on input
	//new size for buffers (obviously)
	//fix shit, so Iterator may be an iterator to core::smart_refct_ptr too
	//handle case where (maxIdx - minIdx) > 0xFFFF 

	uint8_t* perVertexBuffPtr = static_cast<uint8_t*>(outputBuffer.perVertexVtxBuffer.buffer->getPointer()) + (ramb.vertexAllocationOffset * m_vtxSize);
	uint8_t* perInstVtxBuffPtr = static_cast<uint8_t*>(outputBuffer.perInstanceVtxBuffer.buffer->getPointer()) + (ramb.instanceAllocationOffset * m_perInstVtxSize);

		//deinterleave vertices
	//I think I've should validate input mesh buffers again..
	for (uint16_t attrBit = 0x0001, location = 0; location < SVertexInputParams::MAX_ATTR_BUF_BINDING_COUNT; attrBit <<= 1, location++)
	{
		if (!(attrBit & m_outVtxInputParams.enabledAttribFlags))
			continue;

		//TODO: if, for case where currently processed mesh buffer doesn't have given attr (important!)
		
		SVertexInputAttribParams attrib = m_outVtxInputParams.attributes[location];
		SVertexInputBindingParams attribBinding = m_outVtxInputParams.bindings[attrib.binding];

		const size_t attrSize = asset::getTexelOrBlockBytesize(static_cast<E_FORMAT>(attrib.format));
		const size_t stride = (attribBinding.stride) == 0 ? attrSize : attribBinding.stride;

			//this is where vertices are deinterleaved and copied into output vertex buffers
		for (auto it = begin; it != end; it++)
		{
			const size_t vtxCnt = (*it)->calcVertexCount();
			uint8_t* attrPtr = static_cast<uint8_t*>((*it)->getVertexBufferBindings()[attrib.binding].buffer->getPointer()) + attrib.relativeOffset;

			switch (attribBinding.inputRate)
			{
			case EVIR_PER_VERTEX:
			{
				for (uint64_t i = 0; i < vtxCnt; i++)
				{
					//assert((perVertexBuffPtr + attrSize) <= ((ramb.vertexAllocationOffset * m_vtxSize) + ramb.vertexAllocationReservedSize));
					memcpy(perVertexBuffPtr, attrPtr, attrSize);
					perVertexBuffPtr += attrSize;
					attrPtr += stride;
				}
				break;
			}
			case EVIR_PER_INSTANCE:
			{
				for (uint64_t i = 0; i < /*fix*/vtxCnt; i++)
				{
					//assert((perInstBuffPtr + attrSize) <= ((ramb.instanceAllocationOffset * m_perInstVtxSize) + ramb.instanceAllocationReservedSize));
					memcpy(perInstVtxBuffPtr, attrPtr, attrSize);
					perInstVtxBuffPtr += attrSize;
					attrPtr += stride;
				}
				break;
			}
			}
		}
	}

	for (auto it = begin; it != end; it++)
	{
		//there indices and MDI data will be set
		((*it)->getIndexType() == EIT_16BIT) ? processMeshBuffer<uint16_t>(*it, ramb) : processMeshBuffer<uint32_t>(*it, ramb);
	}

	return {};
}

}
}

#endif