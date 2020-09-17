#ifndef __IRR_C_CPU_MESH_PACKER_H_INCLUDED__
#define __IRR_C_CPU_MESH_PACKER_H_INCLUDED__

#include <irr/asset/ICPUMesh.h>
#include <irr/asset/IMeshPacker.h>
#include <irr/core/math/intutil.h>

//AFTER SAFE SHRINK FIX TODO LIST:
//1. new size for buffers (obviously)
//3. handle case where (maxIdx - minIdx) > 0xFFFF
//4. fix handling of per instance attributes 
//5. make it work for multiple `alloc` calls
//6. provide `free` method
//7. assertions on buffer overflow
//8. extendend tests
/*9. packing sponza this way works incorrectly (it is all good if I change #1 and #2 to 5000u), 
	 clue: vertices from some buffers are being used by previous buffers (buffer with index 4 use vertices of buffer with index 5 for example)
	 {
		allocationParams.indexBuffSupportedCnt = 20000000u;
		allocationParams.indexBufferMinAllocSize = 1u;			//#1
		allocationParams.vertexBuffSupportedCnt = 20000000u;
		allocationParams.vertexBufferMinAllocSize = 1u;         //#2
		allocationParams.MDIDataBuffSupportedCnt = 20000u;
		allocationParams.MDIDataBuffMinAllocSize = 1u;
		allocationParams.perInstanceVertexBuffSupportedCnt = 0u;
		allocationParams.perInstanceVertexBufferMinAllocSize = 0u;

		asset::CCPUMeshPacker packer(inputParams, allocationParams, std::numeric_limits<uint16_t>::max(), std::numeric_limits<uint16_t>::max());

		core::vector<MeshPackerBase::ReservedAllocationMeshBuffers> resData;
		resData.reserve(mesh_raw->getMeshBufferCount());

		core::vector<asset::ICPUMeshBuffer*> meshBuffers(mbCount);
		for (uint32_t i = 0u; i < mbCount; i++)
			meshBuffers[i] = mesh_raw->getMeshBuffer(i);

		for (uint32_t i = 0u; i < mbCount; i++)
		{
			resData.emplace_back(packer.alloc(meshBuffers.begin() + i, meshBuffers.begin() + i + 1));
			assert(resData[i].isValid());
		}


		packer.instantiateDataStorage();

		for (uint32_t i = 0u; i < mbCount; i++)
		{
			pmbData.emplace_back(packer.commit(meshBuffers.begin() + i, meshBuffers.begin() + i + 1, resData[i]));
			assert(pmbData[i].isValid());
		}

		packedMeshBuffer = packer.getPackedMeshBuffer();
		assert(packedMeshBuffer.isValid());
	}
*/

namespace irr 
{ 
namespace asset
{

template <typename MDIStructType = DrawElementsIndirectCommand_t>
class CCPUMeshPacker final : public IMeshPacker<ICPUMeshBuffer, MDIStructType>
{
	using base_t = IMeshPacker<ICPUMeshBuffer, MDIStructType>;
	using Triangle = typename base_t::Triangle;
	using TriangleBatch = typename base_t::TriangleBatch;

public:
	CCPUMeshPacker(const SVertexInputParams& preDefinedLayout, const MeshPackerBase::AllocationParams& allocParams, uint16_t minTriangleCountPerMDIData = 256u, uint16_t maxTriangleCountPerMDIData = 1024u)
		:IMeshPacker<ICPUMeshBuffer, MDIStructType>(preDefinedLayout, allocParams, minTriangleCountPerMDIData, maxTriangleCountPerMDIData)
	{}

	template <typename Iterator>
	MeshPackerBase::ReservedAllocationMeshBuffers alloc(const Iterator begin, const Iterator end);

	//needs to be called before first `commit`
	void instantiateDataStorage();

	template <typename Iterator>
	MeshPackerBase::PackedMeshBufferData commit(const Iterator begin, const Iterator end, MeshPackerBase::ReservedAllocationMeshBuffers& ramb);

	inline MeshPackerBase::PackedMeshBuffer<ICPUBuffer>& getPackedMeshBuffer() { return outputBuffer; };

protected:
	core::vector<typename base_t::TriangleBatch> constructTriangleBatches(ICPUMeshBuffer& meshBuffer) override;

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
		//assert(!(*it == nullptr));

		auto* pipeline = (*it)->getPipeline();

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
	
	size_t idxCnt = 0u;
	size_t vtxCnt = 0u;
	for (auto it = begin; it != end; it++)
	{
		ICPUMeshBuffer& mb = **it;
		idxCnt += mb.getIndexCount();
		vtxCnt += mb.calcVertexCount();
	}

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

		m_MDIDataAlctr.free_addr(MDIAllocAddr, possibleMDIStructsNeededCnt);

		return invalidReservedAllocationMeshBuffers;
	}
	
	if (m_vtxBuffAlctrResSpc)
	{
		vtxAllocAddr = m_vtxBuffAlctr.alloc_addr((idxCnt + 1u) / 2u, 1u);
		if (vtxAllocAddr == INVALID_ADDRESS)
		{
			_IRR_DEBUG_BREAK_IF(true);

			m_MDIDataAlctr.free_addr(MDIAllocAddr, possibleMDIStructsNeededCnt);
			m_idxBuffAlctr.free_addr(idxAllocAddr, idxCnt);

			return invalidReservedAllocationMeshBuffers;
		}
	}
	
	if (m_perInsVtxBuffAlctrResSpc)
	{
		//wont work for meshes with per instance attributes
		_IRR_DEBUG_BREAK_IF(true);

		perInsVtxAllocAddr = m_perInsVtxBuffAlctr.alloc_addr((idxCnt + 1u) / 2u, 1u);
		if (perInsVtxAllocAddr == INVALID_ADDRESS)
		{
			_IRR_DEBUG_BREAK_IF(true);

			m_MDIDataAlctr.free_addr(MDIAllocAddr, possibleMDIStructsNeededCnt);
			m_idxBuffAlctr.free_addr(idxAllocAddr, idxCnt);
			m_vtxBuffAlctr.free_addr(vtxAllocAddr, (idxCnt + 1u) / 2u);

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
		vtxAllocAddr == INVALID_ADDRESS ? 0u : (idxCnt + 1u) / 2u
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

	uint32_t vtxBufferOffset = 0u;
	uint32_t maxVtxCnt = m_allocParams.vertexBuffSupportedCnt;

	for (uint16_t attrBit = 0x0001, location = 0; location < SVertexInputParams::MAX_ATTR_BUF_BINDING_COUNT; attrBit <<= 1, location++)
	{
		if (m_outVtxInputParams.enabledAttribFlags & attrBit)
		{
			auto attrib = m_outVtxInputParams.attributes[location];
			auto binding = m_outVtxInputParams.bindings[attrib.binding]; 

			if (binding.inputRate == EVIR_PER_VERTEX)
			{
				outputBuffer.vertexBufferBindings[location] = { vtxBufferOffset,  unifiedVtxBuff };
				vtxBufferOffset += asset::getTexelOrBlockBytesize(static_cast<E_FORMAT>(attrib.format)) * maxVtxCnt;
			}
			else if (binding.inputRate == EVIR_PER_INSTANCE)
			{
				_IRR_DEBUG_BREAK_IF(true);
				//TODO #4
				//outputBuffer.vertexBufferBindings[location] = { perInsBuffOffset,  unifiedVtxBuff };
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

template<typename MDIStructType>
auto CCPUMeshPacker<MDIStructType>::constructTriangleBatches(ICPUMeshBuffer& meshBuffer) -> core::vector<typename base_t::TriangleBatch>
{
	const size_t idxCnt = meshBuffer.getIndexCount();
	const uint32_t triCnt = idxCnt / 3;
	_IRR_DEBUG_BREAK_IF(idxCnt % 3 != 0);

	const uint32_t batchCount = (triCnt + m_maxTriangleCountPerMDIData - 1) / m_maxTriangleCountPerMDIData;

	core::vector<TriangleBatch> output(batchCount);

	for(uint32_t i = 0u; i < batchCount; i++)
	{
		if (i == (batchCount - 1))
		{
			if (triCnt % m_maxTriangleCountPerMDIData)
			{
				output[i].triangles = core::vector<Triangle>(triCnt % m_maxTriangleCountPerMDIData);
				continue;
			}
		}

		output[i].triangles = core::vector<Triangle>(m_maxTriangleCountPerMDIData);
	}

	//struct TriangleMortonCodePair
	//{
	//	Triangle triangle;
	//	//uint64_t mortonCode; TODO after benchmarks
	//};

	//TODO: triangle reordering
	
	auto* srcIdxBuffer = meshBuffer.getIndexBufferBinding();
	uint32_t* idxBufferPtr32Bit = static_cast<uint32_t*>(srcIdxBuffer->buffer->getPointer()) + (srcIdxBuffer->offset / sizeof(uint32_t)); //will be changed after benchmarks
	uint16_t* idxBufferPtr16Bit = static_cast<uint16_t*>(srcIdxBuffer->buffer->getPointer()) + (srcIdxBuffer->offset / sizeof(uint16_t));
	for (TriangleBatch& batch : output)
	{
		for (Triangle& tri : batch.triangles)
		{
			if (meshBuffer.getIndexType() == EIT_32BIT)
			{
				tri.oldIndices[0] = *idxBufferPtr32Bit;
				tri.oldIndices[1] = *(++idxBufferPtr32Bit);
				tri.oldIndices[2] = *(++idxBufferPtr32Bit);
				idxBufferPtr32Bit++;
			}
			else if (meshBuffer.getIndexType() == EIT_16BIT)
			{

				tri.oldIndices[0] = *idxBufferPtr16Bit;
				tri.oldIndices[1] = *(++idxBufferPtr16Bit);
				tri.oldIndices[2] = *(++idxBufferPtr16Bit);
				idxBufferPtr16Bit++;
			}
		}
	}

	return output;
}

template <typename MDIStructType>
template <typename Iterator>
MeshPackerBase::PackedMeshBufferData CCPUMeshPacker<MDIStructType>::commit(const Iterator begin, const Iterator end, MeshPackerBase::ReservedAllocationMeshBuffers& ramb)
{
	MDIStructType* mdiBuffPtr = static_cast<MDIStructType*>(outputBuffer.MDIDataBuffer->getPointer()) + ramb.mdiAllocationOffset;
	uint16_t* indexBuffPtr = static_cast<uint16_t*>(outputBuffer.indexBuffer.buffer->getPointer()) + ramb.indexAllocationOffset;
	size_t verticesAddedToUnifiedBufferCnt = 0ull;

	uint32_t MDIStructsAddedCnt = 0u;

	size_t batchFirstIdx = ramb.indexAllocationOffset;
	size_t batchBaseVtx = ramb.vertexAllocationOffset;

	for (Iterator it = begin; it != end; it++)
	{
		const size_t idxCnt = (*it)->getIndexCount();
		core::vector<TriangleBatch> triangleBatches = constructTriangleBatches(**it);

		for (TriangleBatch& batch : triangleBatches)
		{
			core::unordered_map<uint32_t, uint16_t> usedVertices;
			core::vector<Triangle> newIdxTris = batch.triangles;

			uint32_t newIdx = 0u;
			for (uint32_t i = 0u; i < batch.triangles.size(); i++)
			{
				const Triangle& triangle = batch.triangles[i];
				for (int32_t j = 0; j < 3; j++)
				{
					const uint32_t oldIndex = triangle.oldIndices[j];
					auto result = usedVertices.insert(std::make_pair(oldIndex, newIdx));

					newIdxTris[i].oldIndices[j] = result.second ? newIdx++ : result.first->second;			
				}
			}
			
			//TODO: cache optimization

			//copy indices into unified index buffer
			for (size_t i = 0; i < batch.triangles.size(); i++)
			{
				for (int j = 0; j < 3; j++)
				{
					*indexBuffPtr = newIdxTris[i].oldIndices[j];
					indexBuffPtr++;
				}
			}

			//copy deinterleaved vertices into unified vertex buffer
			for (uint16_t attrBit = 0x0001, location = 0; location < SVertexInputParams::MAX_ATTR_BUF_BINDING_COUNT; attrBit <<= 1, location++)
			{
				if (!(m_outVtxInputParams.enabledAttribFlags & attrBit))
					continue;

				SVertexInputAttribParams attrib = m_outVtxInputParams.attributes[location];
				SVertexInputAttribParams MBAttrib = (*it)->getPipeline()->getVertexInputParams().attributes[location];

				SVertexInputBindingParams attribBinding = (*it)->getPipeline()->getVertexInputParams().bindings[MBAttrib.binding];
				uint8_t* attrPtr = (*it)->getAttribPointer(location);
				const size_t attrSize = asset::getTexelOrBlockBytesize(static_cast<E_FORMAT>(attrib.format));
				const size_t stride = (attribBinding.stride) == 0 ? attrSize : attribBinding.stride;

				SBufferBinding<ICPUBuffer>& vtxBuffBind = outputBuffer.vertexBufferBindings[location];
				uint8_t* outBuffAttrPtr = static_cast<uint8_t*>(vtxBuffBind.buffer->getPointer()) + vtxBuffBind.offset;
				outBuffAttrPtr += (ramb.vertexAllocationOffset + verticesAddedToUnifiedBufferCnt) * attrSize;

				//if (location == 0)
				//{
				//	for (auto index : usedVertices)
				//	{
				//		std::cout << '\n' << index.first << ' ' << index.second << std::endl;
				//	}

				//	std::cout << "-------------\n";
				//}
				
				for (auto index : usedVertices)
				{
					const uint8_t* attrSrc = attrPtr + (index.first * stride);
					uint8_t* vtxAttrDest = outBuffAttrPtr + (index.second * attrSize);
					memcpy(vtxAttrDest, attrSrc, attrSize);	
				}
			}

			verticesAddedToUnifiedBufferCnt += usedVertices.size();

			//construct mdi data
			MDIStructType MDIData;
			MDIData.count = batch.triangles.size() * 3;
			MDIData.instanceCount = (*it)->getInstanceCount();
			MDIData.firstIndex = batchFirstIdx;
			MDIData.baseVertex = batchBaseVtx; //possible overflow?
			MDIData.baseInstance = 0u; //TODO #4

			*mdiBuffPtr = MDIData;
			mdiBuffPtr++;
			MDIStructsAddedCnt++;

			batchFirstIdx += 3 * batch.triangles.size();
			batchBaseVtx += usedVertices.size();
		}
	}

	return { ramb.mdiAllocationOffset, MDIStructsAddedCnt };
}

}
}

#endif