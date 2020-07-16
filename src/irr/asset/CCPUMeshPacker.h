#ifndef __IRR_C_CPU_MESH_PACKER_H_INCLUDED__
#define __IRR_C_CPU_MESH_PACKER_H_INCLUDED__

#include <irr/asset/ICPUMesh.h>
#include <irr/asset/IMeshPacker.h>
#include <irr/core/math/intutil.h>

//AFTER SAFE SHRINK FIX TODO LIST:
//1. new size for buffers (obviously)
//2. fix shit, so Iterator may be an iterator to core::smart_refct_ptr too
//3. handle case where (maxIdx - minIdx) > 0xFFFF
//4. fix handling of per instance attributes 
//5. make it work for multiple `alloc` calls
//6. provide `free` method
//7. assertions on buffer overflow
//8. extendend tests

namespace irr 
{ 
namespace asset
{

template <typename MDIStructType = DrawElementsIndirectCommand_t>
class CCPUMeshPacker final : public IMeshPacker<ICPUMeshBuffer, MDIStructType>
{
public:
	CCPUMeshPacker(const SVertexInputParams& preDefinedLayout, const MeshPackerBase::AllocationParams& allocParams, uint16_t minTriangleCountPerMDIData = 256u, uint16_t maxTriangleCountPerMDIData = 1024u)
		:IMeshPacker<ICPUMeshBuffer, MDIStructType>(preDefinedLayout, allocParams, minTriangleCountPerMDIData, maxTriangleCountPerMDIData)
	{}

	template <typename Iterator>
	MeshPackerBase::ReservedAllocationMeshBuffers alloc(const Iterator begin, const Iterator end);

	//needs to be called before first commit
	void instantiateDataStorage();

	template <typename Iterator>
	MeshPackerBase::PackedMeshBufferData commit(const Iterator begin, const Iterator end, MeshPackerBase::ReservedAllocationMeshBuffers& ramb);

	inline MeshPackerBase::PackedMeshBuffer<ICPUBuffer>& getPackedMeshBuffer() { return outputBuffer; };

private:
	//configures indices and MDI structs (implementation is not ready yet)
	template<typename IndexType>
	uint32_t processMeshBuffer(ICPUMeshBuffer* inputMeshBuffer, MeshPackerBase::ReservedAllocationMeshBuffers& ramb);

private: 
	MeshPackerBase::PackedMeshBuffer<ICPUBuffer> outputBuffer;

};

template <typename MDIStructType>
//`Iterator` may be only an Iterator or pointer to pointer
template <typename Iterator>
MeshPackerBase::ReservedAllocationMeshBuffers CCPUMeshPacker<MDIStructType>::alloc(const Iterator begin, const Iterator end)
{
	/*
	Requirements for input mesh buffers:
		- attributes bound to the same binding must have identical format
		- all meshbufers have indexed triangle list (temporary)
	*/

	//validation
	//TODO: remove this condition
	for(auto it = begin; it != end; it++)
	{
		assert(!(*it == nullptr));

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
	
	uint32_t possibleMDIStructsNeededCnt = 0u;
	for (auto it = begin; it != end; it++)
		possibleMDIStructsNeededCnt += ((*it)->getIndexCount() + minIdxCntPerPatch - 1) / minIdxCntPerPatch;

	uint32_t MDIAllocAddr       = INVALID_ADDRESS;
	uint32_t idxAllocAddr       = INVALID_ADDRESS;
	uint32_t vtxAllocAddr       = INVALID_ADDRESS;
	uint32_t perInsVtxAllocAddr = INVALID_ADDRESS;

	MDIAllocAddr = m_MDIDataAlctr.alloc_addr(possibleMDIStructsNeededCnt, 1u);
	if (MDIAllocAddr == INVALID_ADDRESS)
	{
		_IRR_DEBUG_BREAK_IF(true);
		return invalidReservedAllocationMeshBuffers;
	}
		
	
	idxAllocAddr = m_idxBuffAlctr.alloc_addr(idxCnt, 1u);
	if (idxAllocAddr == INVALID_ADDRESS)
	{
		_IRR_DEBUG_BREAK_IF(true);
		return invalidReservedAllocationMeshBuffers;
	}
	
	
	if (m_vtxBuffAlctrResSpc)
	{
		vtxAllocAddr = m_vtxBuffAlctr.alloc_addr(vtxCnt, 1u);
		if (vtxAllocAddr == INVALID_ADDRESS)
		{
			_IRR_DEBUG_BREAK_IF(true);
			return invalidReservedAllocationMeshBuffers;
		}
			
	}
	
	if (m_perInsVtxBuffAlctrResSpc)
	{
		perInsVtxAllocAddr = m_perInsVtxBuffAlctr.alloc_addr(vtxCnt * m_vtxSize, 1u);
		if (perInsVtxAllocAddr == INVALID_ADDRESS)
		{
			_IRR_DEBUG_BREAK_IF(true);
			return invalidReservedAllocationMeshBuffers;
		}
	}

	MeshPackerBase::ReservedAllocationMeshBuffers result{
		MDIAllocAddr,
		possibleMDIStructsNeededCnt,
		perInsVtxAllocAddr,
		perInsVtxAllocAddr == INVALID_ADDRESS ? 0u : vtxCnt, //TODO
		idxAllocAddr,
		idxCnt,
		vtxAllocAddr,
		vtxAllocAddr == INVALID_ADDRESS ? 0u : vtxCnt
	};
	return result;
}

template <typename MDIStructType>
void CCPUMeshPacker<MDIStructType>::instantiateDataStorage()
{
	//TODO: redo after safe_shrink fix
	outputBuffer.MDIDataBuffer = core::make_smart_refctd_ptr<ICPUBuffer>(m_allocParams.MDIDataBuffSupportedCnt * sizeof(MDIStructType));
	outputBuffer.indexBuffer.buffer = core::make_smart_refctd_ptr<ICPUBuffer>(m_allocParams.indexBuffSupportedCnt * sizeof(uint16_t));

	core::smart_refctd_ptr<ICPUBuffer> unifiedVtxBuff = core::make_smart_refctd_ptr<ICPUBuffer>(m_allocParams.vertexBuffSupportedCnt * m_vtxSize);
	core::smart_refctd_ptr<ICPUBuffer> unifiedInsBuff = core::make_smart_refctd_ptr<ICPUBuffer>(m_allocParams.perInstanceVertexBuffSupportedCnt * m_perInstVtxSize);

	outputBuffer.vertexInputParams = m_outVtxInputParams;

	//divide unified vtx buffers
	//proportions: sizeOfAttr1 : sizeOfAttr2 : ... : sizeOfAttrN
	std::array<uint32_t, SVertexInputParams::MAX_ATTR_BUF_BINDING_COUNT> attrSizeArray;

	uint32_t perVtxAttrSizeSum = 0u;
	uint32_t perInsAttrSizeSum = 0u;
	uint16_t activeAttribCnt = 0u;

	for (uint16_t attrBit = 0x0001, location = 0; location < SVertexInputParams::MAX_ATTR_BUF_BINDING_COUNT; attrBit <<= 1, location++)
	{
		auto attrib = m_outVtxInputParams.attributes[location];
		auto binding = m_outVtxInputParams.bindings[attrib.binding];

		if (!(attrBit & m_outVtxInputParams.enabledAttribFlags))
		{
			attrSizeArray[location] = 0u;
			continue;
		}
		
		attrSizeArray[location] = asset::getTexelOrBlockBytesize(static_cast<E_FORMAT>(attrib.format));

		if (binding.inputRate == EVIR_PER_VERTEX)
			perVtxAttrSizeSum += attrSizeArray[location];
		else
			perInsAttrSizeSum += attrSizeArray[location];

		activeAttribCnt++;
	}
	
	uint32_t perVtxUnitVal = (m_allocParams.vertexBuffSupportedCnt * m_vtxSize + perVtxAttrSizeSum - 1) / perVtxAttrSizeSum; //round up??
	uint32_t perInsUnitVal = 0u;
	if(m_perInstVtxSize)
		perInsUnitVal = (m_allocParams.perInstanceVertexBuffSupportedCnt * m_vtxSize + perInsAttrSizeSum - 1) / perInsAttrSizeSum; //round up??

	size_t perVtxBuffOffset = 0ull;
	size_t perInsBuffOffset = 0ull;

	for (uint16_t attrBit = 0x0001, location = 0; location < SVertexInputParams::MAX_ATTR_BUF_BINDING_COUNT; attrBit <<= 1, location++)
	{
		if (m_outVtxInputParams.enabledAttribFlags & attrBit)
		{
			auto attrib = m_outVtxInputParams.attributes[location];
			auto binding = m_outVtxInputParams.bindings[attrib.binding]; 

			if (binding.inputRate == EVIR_PER_VERTEX)
			{
				outputBuffer.vertexBufferBindings[location] = { perVtxBuffOffset,  unifiedVtxBuff };
				perVtxBuffOffset += attrSizeArray[location] * perVtxUnitVal;
			}
			else if (binding.inputRate == EVIR_PER_INSTANCE)
			{
				//TODO #4
				outputBuffer.vertexBufferBindings[location] = { perInsBuffOffset,  unifiedVtxBuff };
				//perVtxBuffOffset += attrSizeArray[location] * m_vtxSize; fix
			}

		}
	}

}

template<typename MDIStructType>
template<typename IndexType>
uint32_t CCPUMeshPacker<MDIStructType>::processMeshBuffer(ICPUMeshBuffer* inputMeshBuffer, MeshPackerBase::ReservedAllocationMeshBuffers& ramb)
{
	MDIStructType* mdiBuffPtr = static_cast<MDIStructType*>(outputBuffer.MDIDataBuffer->getPointer()) + ramb.mdiAllocationOffset;
	uint16_t* indexBuffPtr = static_cast<uint16_t*>(outputBuffer.indexBuffer.buffer->getPointer()) + ramb.indexAllocationOffset;

	const uint64_t currMeshBufferIdxCnt = inputMeshBuffer->getIndexCount();
	const size_t MDIStructsNeeded = (currMeshBufferIdxCnt + m_maxTriangleCountPerMDIData - 1) / m_maxTriangleCountPerMDIData;

	uint32_t firstIdxForCurrMeshBatch = ramb.indexAllocationOffset;

	//assuming it's never EIT_UNKNOWN
	IndexType* idxBuffBatchBegin = static_cast<IndexType*>(inputMeshBuffer->getIndexBufferBinding()->buffer->getPointer());
	IndexType* idxBuffBatchEnd = nullptr;

	for (uint64_t i = 0; i < MDIStructsNeeded; i++)
	{
		idxBuffBatchEnd = idxBuffBatchBegin;

		if (i == (MDIStructsNeeded - 1))
		{
			if (currMeshBufferIdxCnt % m_maxTriangleCountPerMDIData == 0)
				idxBuffBatchEnd += m_maxTriangleCountPerMDIData;
			else
				idxBuffBatchEnd += currMeshBufferIdxCnt % m_maxTriangleCountPerMDIData;
		}
		else
		{
			idxBuffBatchEnd += m_maxTriangleCountPerMDIData;
		}

		const uint32_t minIdxVal = *std::min_element(idxBuffBatchBegin, idxBuffBatchEnd);

		MDIStructType MDIData;
		MDIData.count = idxBuffBatchEnd - idxBuffBatchBegin;
		MDIData.instanceCount = inputMeshBuffer->getInstanceCount();
		MDIData.firstIndex = firstIdxForCurrMeshBatch;
		MDIData.baseVertex = ramb.vertexAllocationOffset + minIdxVal; //possible overflow?
		MDIData.baseInstance = 0u; //TODO #4

		//set mdi structs
		memcpy(mdiBuffPtr, &MDIData, sizeof(MDIStructType));
		
		//set indices
		for (IndexType* ptr = idxBuffBatchBegin; ptr != idxBuffBatchEnd; ptr++)
		{
			*indexBuffPtr = *ptr - minIdxVal;
			indexBuffPtr++;
		}

		firstIdxForCurrMeshBatch += MDIData.count;
		idxBuffBatchBegin = idxBuffBatchEnd;

		mdiBuffPtr++;
	}

	return MDIStructsNeeded;
}

template <typename MDIStructType>
template <typename Iterator>
MeshPackerBase::PackedMeshBufferData CCPUMeshPacker<MDIStructType>::commit(const Iterator begin, const Iterator end, MeshPackerBase::ReservedAllocationMeshBuffers& ramb)
{
	if(!outputBuffer.isValid()) return{};

	//TODO: distinct bindings for all of the attribs!
	//TODO: case where processed mb doesn't have give attrib

		//deinterleave vertices
	//I think I've should validate input mesh buffers again..
	for (uint16_t attrBit = 0x0001, location = 0; location < SVertexInputParams::MAX_ATTR_BUF_BINDING_COUNT; attrBit <<= 1, location++)
	{
		if (!(attrBit & m_outVtxInputParams.enabledAttribFlags))
			continue;
		
		SBufferBinding<ICPUBuffer>& vtxBuffBind = outputBuffer.vertexBufferBindings[location];
		uint8_t* vtxBuffPtr = static_cast<uint8_t*>(vtxBuffBind.buffer->getPointer()) + ((ramb.vertexAllocationOffset + vtxBuffBind.offset));

		//TODO: if, for case where currently processed mesh buffer doesn't have given attr (important!)
		
		SVertexInputAttribParams attrib = m_outVtxInputParams.attributes[location];
		const size_t attrSize = asset::getTexelOrBlockBytesize(static_cast<E_FORMAT>(attrib.format));

			//this is where vertices are deinterleaved and copied into output vertex buffers
		for (auto it = begin; it != end; it++)
		{
			uint16_t MBEnabledAttribFlags = (*it)->getPipeline()->getVertexInputParams().enabledAttribFlags;
			const size_t vtxCnt = (*it)->calcVertexCount();

			if (!(attrBit & MBEnabledAttribFlags))
			{
				_IRR_DEBUG_BREAK_IF(true);
				//TODO
			}
			else
			{
				SVertexInputAttribParams MBAttrib = (*it)->getPipeline()->getVertexInputParams().attributes[location];
				SVertexInputBindingParams attribBinding = (*it)->getPipeline()->getVertexInputParams().bindings[MBAttrib.binding];
				uint8_t* attrPtr = (*it)->getAttribPointer(location);
				const size_t stride = (attribBinding.stride) == 0 ? attrSize : attribBinding.stride;

				switch (attribBinding.inputRate)
				{
				case EVIR_PER_VERTEX:
				{
					for (uint64_t i = 0; i < vtxCnt; i++)
					{
						//assert((perVertexBuffPtr + attrSize) <= ((ramb.vertexAllocationOffset * m_vtxSize) + ramb.vertexAllocationReservedSize));
						memcpy(vtxBuffPtr, attrPtr, attrSize);
						vtxBuffPtr += attrSize;
						attrPtr += stride;
					}
					break;
				}
				case EVIR_PER_INSTANCE:
				{
					//not implemented yet
					_IRR_DEBUG_BREAK_IF(true);
					assert(false);
					assert(m_perInstVtxSize);

					for (uint64_t i = 0; i < /*fix*/vtxCnt; i++)
					{
						//assert((perInstBuffPtr + attrSize) <= ((ramb.instanceAllocationOffset * m_perInstVtxSize) + ramb.instanceAllocationReservedSize));
						memcpy(vtxBuffPtr, attrPtr, attrSize);
						vtxBuffPtr += attrSize;
						attrPtr += stride;
					}
					break;
				}
				}
			}
			
		}
	}

	PackedMeshBufferData output{ ramb.mdiAllocationOffset, 0u };
	uint32_t MDIStructsCreatedSum = 0u;

	for (auto it = begin; it != end; it++)
	{
		//there indices and MDI data are being set
		const uint32_t MDIStructsCreated = ((*it)->getIndexType() == EIT_16BIT) ? processMeshBuffer<uint16_t>(*it, ramb) : processMeshBuffer<uint32_t>(*it, ramb);
		MDIStructsCreatedSum += MDIStructsCreated;
		ramb.mdiAllocationOffset += MDIStructsCreated;
		ramb.indexAllocationOffset += (*it)->getIndexCount();
		ramb.vertexAllocationOffset += (*it)->calcVertexCount();
	}

	ramb = invalidReservedAllocationMeshBuffers;
	output.mdiParameterCount = MDIStructsCreatedSum;
	return output;
}

}
}

#endif