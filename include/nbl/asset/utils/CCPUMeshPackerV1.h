// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_C_CPU_MESH_PACKER_H_INCLUDED__
#define __NBL_ASSET_C_CPU_MESH_PACKER_H_INCLUDED__

#include <nbl/asset/ICPUMesh.h>
#include <nbl/asset/utils/IMeshPacker.h>
#include <nbl/core/math/intutil.h>

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

		asset::CCPUMeshPackerV1 packer(inputParams, allocationParams, std::numeric_limits<uint16_t>::max(), std::numeric_limits<uint16_t>::max());

		core::vector<IMeshPackerBase::ReservedAllocationMeshBuffers> resData;
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

		packedMeshBuffer = packer.getPackerDataStore();
		assert(packedMeshBuffer.isValid());
	}
10. test `getPackerCreationParamsFromMeshBufferRange`
*/

namespace nbl 
{ 
namespace asset
{

template <typename MDIStructType = DrawElementsIndirectCommand_t>
class CCPUMeshPackerV1 final : public IMeshPacker<ICPUMeshBuffer, MDIStructType>
{
	using base_t = IMeshPacker<ICPUMeshBuffer, MDIStructType>;
	using Triangle = typename base_t::Triangle;
	using TriangleBatch = typename base_t::TriangleBatch;

public:
	struct AllocationParams : IMeshPackerBase::AllocationParamsCommon
	{
		size_t perInstanceVertexBuffSupportedSize = 33554432ull;         /*  32MB*/
		size_t perInstanceVertexBufferMinAllocSize = 32ull;
	};

	struct ReservedAllocationMeshBuffers
	{
		uint32_t mdiAllocationOffset;
		uint32_t mdiAllocationReservedSize;
		uint32_t instanceAllocationOffset;
		uint32_t instanceAllocationReservedSize;
		uint32_t indexAllocationOffset;
		uint32_t indexAllocationReservedSize;
		uint32_t vertexAllocationOffset;
		uint32_t vertexAllocationReservedSize;

		inline bool isValid()
		{
			return this->mdiAllocationOffset != core::GeneralpurposeAddressAllocator<uint32_t>::invalid_address;
		}
	};

	struct PackerDataStore : base_t::template PackerDataStoreCommon<ICPUBuffer>
	{
		SBufferBinding<ICPUBuffer> vertexBufferBindings[SVertexInputParams::MAX_ATTR_BUF_BINDING_COUNT] = {};
		SBufferBinding<ICPUBuffer> indexBuffer;

		SVertexInputParams vertexInputParams;
	};

	template <typename Iterator>
	struct MeshPackerConfigParams
	{
		SVertexInputParams vertexInputParams;
		core::SRange<void, Iterator> belongingMeshes; // pointers to sections of `sortedMeshBuffersOut`
	};

public:
	CCPUMeshPackerV1(const SVertexInputParams& preDefinedLayout, const AllocationParams& allocParams, uint16_t minTriangleCountPerMDIData = 256u, uint16_t maxTriangleCountPerMDIData = 1024u);

	~CCPUMeshPackerV1()
	{
		_NBL_ALIGNED_FREE(m_perInsVtxBuffAlctrResSpc);
	}

	template <typename MeshBufferIterator>
	ReservedAllocationMeshBuffers alloc(const MeshBufferIterator mbBegin, const MeshBufferIterator mbEnd);

	//needs to be called before first `commit`
	void instantiateDataStorage();

	template <typename MeshBufferIterator>
	IMeshPackerBase::PackedMeshBufferData commit(const MeshBufferIterator mbBegin, const MeshBufferIterator mbEnd, ReservedAllocationMeshBuffers& ramb);

	inline PackerDataStore& getPackerDataStore() { return m_output; };

	//TODO: update comment
	// returns number of distinct mesh packers needed to pack the meshes and a sorted list of meshes by the meshpacker ID they should be packed into, as well as the parameters for the packers
	// `packerParamsOut` should be big enough to fit `std::distance(begin,end)` entries, the return value will tell you how many were actually written
	template<typename Iterator>
	static uint32_t getPackerCreationParamsFromMeshBufferRange(const Iterator begin, const Iterator end, Iterator sortedMeshBuffersOut,
		MeshPackerConfigParams<Iterator>* packerParamsOut);

private:
	//configures indices and MDI structs (implementation is not ready yet)
	template<typename IndexType>
	uint32_t processMeshBuffer(ICPUMeshBuffer* inputMeshBuffer, ReservedAllocationMeshBuffers& ramb);

	static bool cmpVtxInputParams(const SVertexInputParams& lhs, const SVertexInputParams& rhs);

private: 
	PackerDataStore m_output;

	uint32_t m_vtxSize;
	uint32_t m_perInstVtxSize;
	const AllocationParams m_allocParams;

	void* m_perInsVtxBuffAlctrResSpc;
	core::GeneralpurposeAddressAllocator<uint32_t> m_perInsVtxBuffAlctr;

	_NBL_STATIC_INLINE_CONSTEXPR ReservedAllocationMeshBuffers invalidReservedAllocationMeshBuffers{ INVALID_ADDRESS, 0, 0, 0, 0, 0, 0, 0 };

};

template <typename MDIStructType>
CCPUMeshPackerV1<MDIStructType>::CCPUMeshPackerV1(const SVertexInputParams& preDefinedLayout, const AllocationParams& allocParams, uint16_t minTriangleCountPerMDIData, uint16_t maxTriangleCountPerMDIData)
	:IMeshPacker<ICPUMeshBuffer, MDIStructType>(minTriangleCountPerMDIData, maxTriangleCountPerMDIData),
	 m_allocParams(allocParams),
	 m_perInsVtxBuffAlctrResSpc(nullptr)
	 
{
	m_output.vertexInputParams.enabledAttribFlags = preDefinedLayout.enabledAttribFlags;
	m_output.vertexInputParams.enabledBindingFlags = preDefinedLayout.enabledAttribFlags;
	memcpy(m_output.vertexInputParams.attributes, preDefinedLayout.attributes, sizeof(m_output.vertexInputParams.attributes));

	//1 attrib enabled == 1 binding
	for (uint16_t attrBit = 0x0001, location = 0; location < SVertexInputParams::MAX_ATTR_BUF_BINDING_COUNT; attrBit <<= 1, location++)
	{
		if (m_output.vertexInputParams.enabledAttribFlags & attrBit)
		{
			m_output.vertexInputParams.attributes[location].binding = location;
			m_output.vertexInputParams.attributes[location].relativeOffset = 0u;
			m_output.vertexInputParams.bindings[location].stride = getTexelOrBlockBytesize(static_cast<E_FORMAT>(m_output.vertexInputParams.attributes[location].format));
			m_output.vertexInputParams.bindings[location].inputRate = preDefinedLayout.bindings[preDefinedLayout.attributes[location].binding].inputRate;
		}
	}

	m_vtxSize = calcVertexSize(preDefinedLayout, E_VERTEX_INPUT_RATE::EVIR_PER_VERTEX);
	//TODO: allow for mesh buffers with only per instance parameters
	assert(m_vtxSize);

	m_perInstVtxSize = calcVertexSize(preDefinedLayout, E_VERTEX_INPUT_RATE::EVIR_PER_INSTANCE);
	if (m_perInstVtxSize)
	{
		m_perInsVtxBuffAlctrResSpc = _NBL_ALIGNED_MALLOC(core::GeneralpurposeAddressAllocator<uint32_t>::reserved_size(alignof(std::max_align_t), allocParams.perInstanceVertexBuffSupportedSize / m_perInstVtxSize, allocParams.perInstanceVertexBufferMinAllocSize), _NBL_SIMD_ALIGNMENT);
		_NBL_DEBUG_BREAK_IF(m_perInsVtxBuffAlctrResSpc == nullptr);
		assert(m_perInsVtxBuffAlctrResSpc != nullptr);
		m_perInsVtxBuffAlctr = core::GeneralpurposeAddressAllocator<uint32_t>(m_perInsVtxBuffAlctrResSpc, 0u, 0u, alignof(std::max_align_t), allocParams.perInstanceVertexBuffSupportedSize / m_perInstVtxSize, allocParams.perInstanceVertexBufferMinAllocSize);
	}

	initializeCommonAllocators(
		{
			allocParams.indexBuffSupportedCnt,
			m_vtxSize ? allocParams.vertexBuffSupportedSize / m_vtxSize : 0ull,
			allocParams.MDIDataBuffSupportedCnt,
			allocParams.indexBufferMinAllocSize,
			allocParams.vertexBufferMinAllocSize,
			allocParams.MDIDataBuffMinAllocSize
		}
	);
}

template <typename MDIStructType>
//`Iterator` may be only an Iterator or pointer to pointer
template <typename MeshBufferIterator>
typename CCPUMeshPackerV1<MDIStructType>::ReservedAllocationMeshBuffers CCPUMeshPackerV1<MDIStructType>::alloc(const MeshBufferIterator mbBegin, const MeshBufferIterator mbEnd)
{
	/*
	Requirements for input mesh buffers:
		- attributes bound to the same binding must have identical format
		- all meshbufers have indexed triangle list (temporary)
	*/

	//validation
	//TODO: remove this condition
	for(auto it = mbBegin; it != mbEnd; it++)
	{
		//assert(!(*it == nullptr));

		auto* pipeline = (*it)->getPipeline();

		if ((*it)->getIndexBufferBinding().buffer.get() == nullptr ||
			pipeline->getPrimitiveAssemblyParams().primitiveType != EPT_TRIANGLE_LIST)
		{
			_NBL_DEBUG_BREAK_IF(true);
			return invalidReservedAllocationMeshBuffers;
		}
	}

	//validation
	for (auto it = mbBegin; it != mbEnd; it++)
	{
		const auto& mbVtxInputParams = (*it)->getPipeline()->getVertexInputParams();
		for (uint16_t attrBit = 0x0001, location = 0; location < SVertexInputParams::MAX_ATTR_BUF_BINDING_COUNT; attrBit <<= 1, location++)
		{
			if (!(attrBit & mbVtxInputParams.enabledAttribFlags))
				continue;

			//assert((attrBit & m_output.vertexInputParams.enabledAttribFlags));

			if (mbVtxInputParams.attributes[location].format != m_output.vertexInputParams.attributes[location].format ||
				mbVtxInputParams.bindings[mbVtxInputParams.attributes[location].binding].inputRate != m_output.vertexInputParams.bindings[location].inputRate)
			{
				_NBL_DEBUG_BREAK_IF(true);
				return invalidReservedAllocationMeshBuffers;
			}
		}
	}
	
	size_t idxCnt = 0u;
	size_t vtxCnt = 0u;
	for (auto it = mbBegin; it != mbEnd; it++)
	{
		ICPUMeshBuffer* mb = *it;
		idxCnt += mb->getIndexCount();
		vtxCnt += IMeshManipulator::upperBoundVertexID(mb);
	}

	const uint32_t minIdxCntPerPatch = m_minTriangleCountPerMDIData * 3;
	
	uint32_t possibleMDIStructsNeededCnt = 0u;
	for (auto it = mbBegin; it != mbEnd; it++)
		possibleMDIStructsNeededCnt += ((*it)->getIndexCount() + minIdxCntPerPatch - 1) / minIdxCntPerPatch;

	uint32_t MDIAllocAddr       = INVALID_ADDRESS;
	uint32_t idxAllocAddr       = INVALID_ADDRESS;
	uint32_t vtxAllocAddr       = INVALID_ADDRESS;
	uint32_t perInsVtxAllocAddr = INVALID_ADDRESS;

	MDIAllocAddr = m_MDIDataAlctr.alloc_addr(possibleMDIStructsNeededCnt, 1u);
	if (MDIAllocAddr == INVALID_ADDRESS)
	{
		_NBL_DEBUG_BREAK_IF(true);
		return invalidReservedAllocationMeshBuffers;
	}
	
	idxAllocAddr = m_idxBuffAlctr.alloc_addr(idxCnt, 1u);
	if (idxAllocAddr == INVALID_ADDRESS)
	{
		_NBL_DEBUG_BREAK_IF(true);

		m_MDIDataAlctr.free_addr(MDIAllocAddr, possibleMDIStructsNeededCnt);

		return invalidReservedAllocationMeshBuffers;
	}
	
	if (m_vtxBuffAlctrResSpc)
	{
		vtxAllocAddr = m_vtxBuffAlctr.alloc_addr((idxCnt + 1u) / 2u, 1u);
		if (vtxAllocAddr == INVALID_ADDRESS)
		{
			_NBL_DEBUG_BREAK_IF(true);

			m_MDIDataAlctr.free_addr(MDIAllocAddr, possibleMDIStructsNeededCnt);
			m_idxBuffAlctr.free_addr(idxAllocAddr, idxCnt);

			return invalidReservedAllocationMeshBuffers;
		}
	}
	
	if (m_perInsVtxBuffAlctrResSpc)
	{
		//wont work for meshes with per instance attributes
		_NBL_DEBUG_BREAK_IF(true);

		perInsVtxAllocAddr = m_perInsVtxBuffAlctr.alloc_addr((idxCnt + 1u) / 2u, 1u);
		if (perInsVtxAllocAddr == INVALID_ADDRESS)
		{
			_NBL_DEBUG_BREAK_IF(true);

			m_MDIDataAlctr.free_addr(MDIAllocAddr, possibleMDIStructsNeededCnt);
			m_idxBuffAlctr.free_addr(idxAllocAddr, idxCnt);
			m_vtxBuffAlctr.free_addr(vtxAllocAddr, (idxCnt + 1u) / 2u);

			return invalidReservedAllocationMeshBuffers;
		}
	}

	ReservedAllocationMeshBuffers result{
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
void CCPUMeshPackerV1<MDIStructType>::instantiateDataStorage()
{
	//TODO: redo after safe_shrink fix
	m_output.MDIDataBuffer = core::make_smart_refctd_ptr<ICPUBuffer>(m_allocParams.MDIDataBuffSupportedCnt * sizeof(MDIStructType));
	m_output.indexBuffer.buffer = core::make_smart_refctd_ptr<ICPUBuffer>(m_allocParams.indexBuffSupportedCnt * sizeof(uint16_t));

	core::smart_refctd_ptr<ICPUBuffer> unifiedVtxBuff = core::make_smart_refctd_ptr<ICPUBuffer>(m_allocParams.vertexBuffSupportedSize);
	core::smart_refctd_ptr<ICPUBuffer> unifiedInsBuff = core::make_smart_refctd_ptr<ICPUBuffer>(m_allocParams.perInstanceVertexBuffSupportedSize);

	m_output.vertexInputParams = m_output.vertexInputParams;

	//divide unified vtx buffers
	//proportions: sizeOfAttr1 : sizeOfAttr2 : ... : sizeOfAttrN
	std::array<uint32_t, SVertexInputParams::MAX_ATTR_BUF_BINDING_COUNT> attrSizeArray;

	uint32_t vtxBufferOffset = 0u;
	uint32_t maxVtxCnt = m_allocParams.vertexBuffSupportedSize / m_vtxSize;

	for (uint16_t attrBit = 0x0001, location = 0; location < SVertexInputParams::MAX_ATTR_BUF_BINDING_COUNT; attrBit <<= 1, location++)
	{
		if (m_output.vertexInputParams.enabledAttribFlags & attrBit)
		{
			auto attrib = m_output.vertexInputParams.attributes[location];
			auto binding = m_output.vertexInputParams.bindings[attrib.binding]; 

			if (binding.inputRate == EVIR_PER_VERTEX)
			{
				m_output.vertexBufferBindings[location] = { vtxBufferOffset,  unifiedVtxBuff };
				vtxBufferOffset += asset::getTexelOrBlockBytesize(static_cast<E_FORMAT>(attrib.format)) * maxVtxCnt;
			}
			else if (binding.inputRate == EVIR_PER_INSTANCE)
			{
				_NBL_DEBUG_BREAK_IF(true);
				//TODO #4
				//m_output.vertexBufferBindings[location] = { perInsBuffOffset,  unifiedVtxBuff };
				//perVtxBuffOffset += attrSizeArray[location] * m_vtxSize; fix
			}

		}
	}

}

template<typename MDIStructType>
template<typename IndexType>
uint32_t CCPUMeshPackerV1<MDIStructType>::processMeshBuffer(ICPUMeshBuffer* inputMeshBuffer, CCPUMeshPackerV1<MDIStructType>::ReservedAllocationMeshBuffers& ramb)
{
	MDIStructType* mdiBuffPtr = static_cast<MDIStructType*>(m_output.MDIDataBuffer->getPointer()) + ramb.mdiAllocationOffset;
	uint16_t* indexBuffPtr = static_cast<uint16_t*>(m_output.indexBuffer.buffer->getPointer()) + ramb.indexAllocationOffset;

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
template <typename MeshBufferIterator>
IMeshPackerBase::PackedMeshBufferData CCPUMeshPackerV1<MDIStructType>::commit(const MeshBufferIterator mbBegin, const MeshBufferIterator mbEnd, CCPUMeshPackerV1<MDIStructType>::ReservedAllocationMeshBuffers& ramb)
{
	MDIStructType* mdiBuffPtr = static_cast<MDIStructType*>(m_output.MDIDataBuffer->getPointer()) + ramb.mdiAllocationOffset;
	uint16_t* indexBuffPtr = static_cast<uint16_t*>(m_output.indexBuffer.buffer->getPointer()) + ramb.indexAllocationOffset;
	size_t verticesAddedToUnifiedBufferCnt = 0ull;

	uint32_t MDIStructsAddedCnt = 0u;

	size_t batchFirstIdx = ramb.indexAllocationOffset;
	size_t batchBaseVtx = ramb.vertexAllocationOffset;

	for (auto it = mbBegin; it != mbEnd; it++)
	{
		const size_t idxCnt = (*it)->getIndexCount();
		core::vector<TriangleBatch> triangleBatches = constructTriangleBatches(*it);

		for (TriangleBatch& batch : triangleBatches)
		{
			core::unordered_map<uint32_t, uint16_t> usedVertices = constructNewIndicesFromTriangleBatch(batch, indexBuffPtr);

			//copy deinterleaved vertices into unified vertex buffer
			for (uint16_t attrBit = 0x0001, location = 0; location < SVertexInputParams::MAX_ATTR_BUF_BINDING_COUNT; attrBit <<= 1, location++)
			{
				if (!(m_output.vertexInputParams.enabledAttribFlags & attrBit))
					continue;

				SVertexInputAttribParams attrib = m_output.vertexInputParams.attributes[location];
				SBufferBinding<ICPUBuffer>& vtxBuffBind = m_output.vertexBufferBindings[location];
				uint8_t* dstAttrPtr = static_cast<uint8_t*>(vtxBuffBind.buffer->getPointer()) + vtxBuffBind.offset;
				const size_t attrSize = asset::getTexelOrBlockBytesize(static_cast<E_FORMAT>(attrib.format));
				dstAttrPtr += (ramb.vertexAllocationOffset + verticesAddedToUnifiedBufferCnt) * attrSize;

				deinterleaveAndCopyAttribute(*it, location, usedVertices, dstAttrPtr);
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

template <typename MDIStructType>
static bool CCPUMeshPackerV1<MDIStructType>::cmpVtxInputParams(const SVertexInputParams& lhs, const SVertexInputParams& rhs)
{
	if (lhs.enabledAttribFlags != rhs.enabledAttribFlags)
		return false;

	for (uint16_t attrBit = 0x0001, location = 0; location < SVertexInputParams::MAX_ATTR_BUF_BINDING_COUNT; attrBit <<= 1, location++)
	{
		if (!(attrBit & lhs.enabledAttribFlags))
			continue;

		if (lhs.attributes[location].format != rhs.attributes[location].format ||
			lhs.bindings[lhs.attributes[location].binding].inputRate != rhs.bindings[rhs.attributes[location].binding].inputRate)
			return false;
	}

	return true;
}

template <typename MDIStructType>
template <typename Iterator>
static uint32_t CCPUMeshPackerV1<MDIStructType>::getPackerCreationParamsFromMeshBufferRange(const Iterator begin, const Iterator end, Iterator sortedMeshBuffersOut,
	MeshPackerConfigParams<Iterator>* packerParamsOut)
{
	assert(begin <= end);
	if (begin == end)
		return 0;

	uint32_t packersNeeded = 1u;

	IMeshPackerBase::MeshPackerConfigParams<Iterator> firstInpuParams
	{
		(*begin)->getPipeline()->getVertexInputParams(),
		SRange<void, Iterator>(sortedMeshBuffersOut, sortedMeshBuffersOut)
	};
	memcpy(packerParamsOut, &firstInpuParams, sizeof(SVertexInputParams));

	//fill array
	auto test1 = std::distance(begin, end);
	auto* packerParamsOutEnd = packerParamsOut + 1u;
	for (Iterator it = begin + 1; it != end; it++)
	{
		auto& currMeshVtxInputParams = (*it)->getPipeline()->getVertexInputParams();

		bool alreadyInserted = false;
		for (auto* packerParamsIt = packerParamsOut; packerParamsIt != packerParamsOutEnd; packerParamsIt++)
		{
			alreadyInserted = cmpVtxInputParams(packerParamsIt->vertexInputParams, currMeshVtxInputParams);

			if (alreadyInserted)
				break;
		}

		if (!alreadyInserted)
		{
			packersNeeded++;

			IMeshPackerBase::MeshPackerConfigParams<Iterator> configParams
			{
				currMeshVtxInputParams,
				SRange<void, Iterator>(sortedMeshBuffersOut, sortedMeshBuffersOut)
			};
			memcpy(packerParamsOutEnd, &configParams, sizeof(SVertexInputParams));
			packerParamsOutEnd++;
		}
	}

	auto getIndexOfArrayElement = [&](const SVertexInputParams& vtxInputParams) -> int32_t
	{
		int32_t offset = 0u;
		for (auto* it = packerParamsOut; it != packerParamsOutEnd; it++, offset++)
		{
			if (cmpVtxInputParams(vtxInputParams, it->vertexInputParams))
				return offset;

			if (it == packerParamsOut - 1)
				return -1;
		}
	};

	//sort meshes by SVertexInputParams
	const Iterator sortedMeshBuffersOutEnd = sortedMeshBuffersOut + std::distance(begin, end);

	std::copy(begin, end, sortedMeshBuffersOut);
	std::sort(sortedMeshBuffersOut, sortedMeshBuffersOutEnd,
		[&](const MeshBufferType* lhs, const MeshBufferType* rhs)
		{
			return getIndexOfArrayElement(lhs->getPipeline()->getVertexInputParams()) < getIndexOfArrayElement(rhs->getPipeline()->getVertexInputParams());
		}
	);

	//set ranges
	Iterator sortedMeshBuffersIt = sortedMeshBuffersOut;
	for (auto* inputParamsIt = packerParamsOut; inputParamsIt != packerParamsOutEnd; inputParamsIt++)
	{
		Iterator firstMBForThisRange = sortedMeshBuffersIt;
		Iterator lastMBForThisRange = sortedMeshBuffersIt;
		for (Iterator it = firstMBForThisRange; it != sortedMeshBuffersOutEnd; it++)
		{
			if (!cmpVtxInputParams(inputParamsIt->vertexInputParams, (*it)->getPipeline()->getVertexInputParams()))
			{
				lastMBForThisRange = it;
				break;
			}
		}

		if (inputParamsIt == packerParamsOutEnd - 1)
			lastMBForThisRange = sortedMeshBuffersOutEnd;

		inputParamsIt->belongingMeshes = SRange<void, Iterator>(firstMBForThisRange, lastMBForThisRange);
		sortedMeshBuffersIt = lastMBForThisRange;
	}

	return packersNeeded;
}

}
}

#endif