// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_C_CPU_MESH_PACKER_H_INCLUDED__
#define __NBL_ASSET_C_CPU_MESH_PACKER_H_INCLUDED__

#include <nbl/asset/ICPUMesh.h>
#include <nbl/asset/utils/IMeshPacker.h>
#include <nbl/core/math/intutil.h>

namespace nbl 
{ 
namespace asset
{

#if 0 // REWRITE
template <typename MDIStructType = DrawElementsIndirectCommand_t>
class CCPUMeshPackerV1 final : public IMeshPacker<ICPUMeshBuffer, MDIStructType>
{
	using base_t = IMeshPacker<ICPUMeshBuffer, MDIStructType>;
	using Triangle = typename base_t::Triangle;
	using TriangleBatches = typename base_t::TriangleBatches;
	using IdxBufferParams = typename base_t::IdxBufferParams;

public:
	struct AllocationParams : IMeshPackerBase::AllocationParamsCommon
	{
		// Maximum byte size of per instance vertex data allocation
		size_t perInstanceVertexBuffSupportedByteSize = 33554432ull;         /*  32MB*/

		// Minimum bytes of per instance vertex data allocated per allocation
		size_t perInstanceVertexBufferMinAllocByteSize = 32ull;
	};

	struct ReservedAllocationMeshBuffers : IMeshPackerBase::ReservedAllocationMeshBuffersBase
	{
		uint32_t instanceAllocationOffset;
		uint32_t instanceAllocationReservedSize;
		uint32_t vertexAllocationOffset;
		uint32_t vertexAllocationReservedSize;
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
		if(isInstancingEnabled)
			_NBL_ALIGNED_FREE(m_perInsVtxBuffAlctrResSpc);
	}

	template <typename MeshBufferIterator>
	ReservedAllocationMeshBuffers alloc(const MeshBufferIterator mbBegin, const MeshBufferIterator mbEnd);

	void free(const ReservedAllocationMeshBuffers& ramb)
	{
		if (ramb.indexAllocationOffset != base_t::INVALID_ADDRESS)
			base_t::m_idxBuffAlctr.free_addr(ramb.indexAllocationOffset, ramb.indexAllocationReservedCnt);

		if (ramb.mdiAllocationOffset != base_t::INVALID_ADDRESS)
			base_t::m_MDIDataAlctr.free_addr(ramb.mdiAllocationOffset, ramb.mdiAllocationReservedCnt);

		if (ramb.vertexAllocationOffset != base_t::INVALID_ADDRESS)
			base_t::m_vtxBuffAlctr.free_addr(ramb.vertexAllocationOffset, ramb.vertexAllocationReservedSize);

		if (ramb.instanceAllocationOffset != base_t::INVALID_ADDRESS)
			base_t::m_vtxBuffAlctr.free_addr(ramb.instanceAllocationOffset, ramb.instanceAllocationReservedSize);
	}

	//needs to be called before first `commit`
	void instantiateDataStorage();

	template <typename MeshBufferIterator>
	IMeshPackerBase::PackedMeshBufferData commit(const MeshBufferIterator mbBegin, const MeshBufferIterator mbEnd, ReservedAllocationMeshBuffers& ramb, core::aabbox3df* aabbs);

	inline PackerDataStore& getPackerDataStore() { return m_output; };

	// returns number of distinct mesh packers needed to pack the meshes and a sorted list of meshes by the meshpacker ID they should be packed into, as well as the parameters for the packers
	// `packerParamsOut` should be big enough to fit `std::distance(begin,end)` entries, the return value will tell you how many were actually written
	template<typename Iterator>
	static uint32_t getPackerCreationParamsFromMeshBufferRange(const Iterator begin, const Iterator end, Iterator sortedMeshBuffersOut,
		MeshPackerConfigParams<Iterator>* packerParamsOut);

	//! shrinks byte size of all output buffers, so they are large enough to fit currently allocated contents. Call this function before `instantiateDataStorage`
	virtual void shrinkOutputBuffersSize() override
	{
		base_t::shrinkOutputBuffersSize();

		uint32_t perInsBuffNewSize = m_perInsVtxBuffAlctr.safe_shrink_size(0u, base_t::alctrTraits::max_alignment(m_perInsVtxBuffAlctr));

		if (isInstancingEnabled)
		{
			const void* oldReserved = base_t::alctrTraits::getReservedSpacePtr(m_perInsVtxBuffAlctr);
			m_perInsVtxBuffAlctr = core::GeneralpurposeAddressAllocator<uint32_t>(perInsBuffNewSize, std::move(m_perInsVtxBuffAlctr), _NBL_ALIGNED_MALLOC(base_t::alctrTraits::reserved_size(perInsBuffNewSize, m_perInsVtxBuffAlctr), _NBL_SIMD_ALIGNMENT));
			_NBL_ALIGNED_FREE(const_cast<void*>(oldReserved));
		}
	}

private:
	static bool cmpVtxInputParams(const SVertexInputParams& lhs, const SVertexInputParams& rhs);

private: 
	PackerDataStore m_output;

	uint32_t m_vtxSize;
	uint32_t m_perInsVtxSize;

	bool isInstancingEnabled;
	void* m_perInsVtxBuffAlctrResSpc;
	core::GeneralpurposeAddressAllocator<uint32_t> m_perInsVtxBuffAlctr;

	_NBL_STATIC_INLINE_CONSTEXPR ReservedAllocationMeshBuffers invalidReservedAllocationMeshBuffers{ base_t::INVALID_ADDRESS, 0, 0, 0, 0, 0, 0, 0 };

};

template <typename MDIStructType>
CCPUMeshPackerV1<MDIStructType>::CCPUMeshPackerV1(const SVertexInputParams& preDefinedLayout, const AllocationParams& allocParams, uint16_t minTriangleCountPerMDIData, uint16_t maxTriangleCountPerMDIData)
	:IMeshPacker<ICPUMeshBuffer, MDIStructType>(minTriangleCountPerMDIData, maxTriangleCountPerMDIData),
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

	m_vtxSize = base_t::calcVertexSize(preDefinedLayout, E_VERTEX_INPUT_RATE::EVIR_PER_VERTEX);

	m_perInsVtxSize = base_t::calcVertexSize(preDefinedLayout, E_VERTEX_INPUT_RATE::EVIR_PER_INSTANCE);
	if (m_perInsVtxSize)
	{
		isInstancingEnabled = true;
		m_perInsVtxBuffAlctrResSpc = _NBL_ALIGNED_MALLOC(core::GeneralpurposeAddressAllocator<uint32_t>::reserved_size(alignof(std::max_align_t), allocParams.perInstanceVertexBuffSupportedByteSize / m_perInsVtxSize, allocParams.perInstanceVertexBufferMinAllocByteSize), _NBL_SIMD_ALIGNMENT);
		assert(m_perInsVtxBuffAlctrResSpc != nullptr);
		m_perInsVtxBuffAlctr = core::GeneralpurposeAddressAllocator<uint32_t>(m_perInsVtxBuffAlctrResSpc, 0u, 0u, alignof(std::max_align_t), allocParams.perInstanceVertexBuffSupportedByteSize / m_perInsVtxSize, allocParams.perInstanceVertexBufferMinAllocByteSize);
	}
	else
	{
		isInstancingEnabled = false;
	}

	base_t::initializeCommonAllocators(
		{
			allocParams.indexBuffSupportedCnt,
			m_vtxSize ? allocParams.vertexBuffSupportedByteSize / m_vtxSize : 0ull,
			allocParams.MDIDataBuffSupportedCnt,
			allocParams.indexBufferMinAllocCnt,
			allocParams.vertexBufferMinAllocByteSize,
			allocParams.MDIDataBuffMinAllocCnt
		}
	);
}

template <typename MDIStructType>
//`Iterator` may be only an Iterator or pointer to pointer
//allocation should be happening even if processed mesh buffer doesn't have attribute that was declared in pre defined `SVertexInputParams`, if mesh buffer has any attributes that are not declared in pre defined `SVertexInputParams` then these should be always ignored
/*
	Requirements for input mesh buffers:
		- attributes bound to the same binding must have identical format
*/
template <typename MeshBufferIterator>
typename CCPUMeshPackerV1<MDIStructType>::ReservedAllocationMeshBuffers CCPUMeshPackerV1<MDIStructType>::alloc(const MeshBufferIterator mbBegin, const MeshBufferIterator mbEnd)
{
	//validation
	for (auto it = mbBegin; it != mbEnd; it++)
	{
		const auto& mbVtxInputParams = (*it)->getPipeline()->getVertexInputParams();
		for (uint16_t attrBit = 0x0001, location = 0; location < SVertexInputParams::MAX_ATTR_BUF_BINDING_COUNT; attrBit <<= 1, location++)
		{
			if (!(m_output.vertexInputParams.enabledAttribFlags & mbVtxInputParams.enabledAttribFlags & attrBit))
				continue;

			if (mbVtxInputParams.attributes[location].format != m_output.vertexInputParams.attributes[location].format ||
				mbVtxInputParams.bindings[mbVtxInputParams.attributes[location].binding].inputRate != m_output.vertexInputParams.bindings[location].inputRate)
			{
				assert(false);
				return invalidReservedAllocationMeshBuffers;
			}
		}
	}
	
	size_t idxCnt = 0u;
	size_t vtxCnt = 0u;
	size_t perInsVtxCnt = 0u;
	for (auto it = mbBegin; it != mbEnd; it++)
	{
		ICPUMeshBuffer* mb = *it;
		idxCnt += base_t::calcIdxCntAfterConversionToTriangleList(mb);
		vtxCnt += base_t::calcVertexCountBoundWithBatchDuplication(mb);
		perInsVtxCnt += mb->getInstanceCount();
	}

	const uint32_t minIdxCntPerPatch = base_t::m_minTriangleCountPerMDIData * 3;
	
	uint32_t possibleMDIStructsNeededCnt = 0u;
	for (auto it = mbBegin; it != mbEnd; it++)
		possibleMDIStructsNeededCnt += ((*it)->getIndexCount() + minIdxCntPerPatch - 1) / minIdxCntPerPatch;

	uint32_t MDIAllocAddr       = base_t::INVALID_ADDRESS;
	uint32_t idxAllocAddr       = base_t::INVALID_ADDRESS;
	uint32_t vtxAllocAddr       = base_t::INVALID_ADDRESS;
	uint32_t perInsVtxAllocAddr = base_t::INVALID_ADDRESS;

	MDIAllocAddr = base_t::m_MDIDataAlctr.alloc_addr(possibleMDIStructsNeededCnt, 1u);
	if (MDIAllocAddr == base_t::INVALID_ADDRESS)
	{
		_NBL_DEBUG_BREAK_IF(true);
		return invalidReservedAllocationMeshBuffers;
	}
	
	idxAllocAddr = base_t::m_idxBuffAlctr.alloc_addr(idxCnt, 1u);
	if (idxAllocAddr == base_t::INVALID_ADDRESS)
	{
		_NBL_DEBUG_BREAK_IF(true);

		base_t::m_MDIDataAlctr.free_addr(MDIAllocAddr, possibleMDIStructsNeededCnt);

		return invalidReservedAllocationMeshBuffers;
	}

	bool arePerVtxAttribsEnabled = base_t::alctrTraits::get_total_size(base_t::m_vtxBuffAlctr) == 0 ? false : true;
	if (arePerVtxAttribsEnabled)
	{
		vtxAllocAddr = base_t::m_vtxBuffAlctr.alloc_addr(vtxCnt * m_vtxSize, 1u);
		if (vtxAllocAddr == base_t::INVALID_ADDRESS)
		{
			_NBL_DEBUG_BREAK_IF(true);

			base_t::m_MDIDataAlctr.free_addr(MDIAllocAddr, possibleMDIStructsNeededCnt);
			base_t::m_idxBuffAlctr.free_addr(idxAllocAddr, idxCnt);

			return invalidReservedAllocationMeshBuffers;
		}
	}
	
	if (isInstancingEnabled)
	{
		perInsVtxAllocAddr = m_perInsVtxBuffAlctr.alloc_addr(perInsVtxCnt * m_perInsVtxSize, 1u);
		if (perInsVtxAllocAddr == base_t::INVALID_ADDRESS)
		{
			_NBL_DEBUG_BREAK_IF(true);

			base_t::m_MDIDataAlctr.free_addr(MDIAllocAddr, possibleMDIStructsNeededCnt);
			base_t::m_idxBuffAlctr.free_addr(idxAllocAddr, idxCnt);
			base_t::m_vtxBuffAlctr.free_addr(vtxAllocAddr, vtxCnt * m_vtxSize);

			return invalidReservedAllocationMeshBuffers;
		}
	}

	ReservedAllocationMeshBuffers result;
	result.mdiAllocationOffset = MDIAllocAddr;
	result.mdiAllocationReservedCnt = possibleMDIStructsNeededCnt;
	result.indexAllocationOffset = idxAllocAddr;
	result.indexAllocationReservedCnt = idxCnt;
	result.instanceAllocationOffset = perInsVtxAllocAddr;
	result.instanceAllocationReservedSize = perInsVtxAllocAddr == base_t::INVALID_ADDRESS ? 0u : perInsVtxCnt * m_perInsVtxSize;
	result.vertexAllocationOffset = vtxAllocAddr;
	result.vertexAllocationReservedSize = vtxAllocAddr == base_t::INVALID_ADDRESS ? 0u : vtxCnt * m_vtxSize;

	return result;
}

template <typename MDIStructType>
void CCPUMeshPackerV1<MDIStructType>::instantiateDataStorage()
{
	const size_t MDIDataBuffSupportedByteSize = base_t::alctrTraits::get_total_size(base_t::m_MDIDataAlctr) * sizeof(MDIStructType);
	const size_t idxBuffSupportedByteSize = base_t::alctrTraits::get_total_size(base_t::m_idxBuffAlctr) * sizeof(uint16_t);
	const size_t vtxBuffSupportedByteSize = base_t::alctrTraits::get_total_size(base_t::m_vtxBuffAlctr);
	const size_t perInsBuffSupportedByteSize = base_t::alctrTraits::get_total_size(base_t::m_vtxBuffAlctr);

	m_output.MDIDataBuffer = core::make_smart_refctd_ptr<ICPUBuffer>(MDIDataBuffSupportedByteSize);
	m_output.indexBuffer.buffer = core::make_smart_refctd_ptr<ICPUBuffer>(idxBuffSupportedByteSize);

	core::smart_refctd_ptr<ICPUBuffer> unifiedVtxBuff = core::make_smart_refctd_ptr<ICPUBuffer>(vtxBuffSupportedByteSize);
	core::smart_refctd_ptr<ICPUBuffer> unifiedInsBuff = core::make_smart_refctd_ptr<ICPUBuffer>(perInsBuffSupportedByteSize);

	//divide unified vtx buffers
	//proportions: sizeOfAttr1 : sizeOfAttr2 : ... : sizeOfAttrN
	std::array<uint32_t, SVertexInputParams::MAX_ATTR_BUF_BINDING_COUNT> attrSizeArray;

	uint32_t vtxBufferOffset = 0u;
	const uint32_t maxVtxCnt = m_vtxSize == 0u ? 0u : vtxBuffSupportedByteSize / m_vtxSize;

	uint32_t perInsBuffOffset = 0u;
	const uint32_t maxPerInsVtxCnt = m_perInsVtxSize == 0u ? 0u : perInsBuffSupportedByteSize / m_perInsVtxSize;

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
				m_output.vertexBufferBindings[location] = { perInsBuffOffset,  unifiedInsBuff };
				perInsBuffOffset += asset::getTexelOrBlockBytesize(static_cast<E_FORMAT>(attrib.format)) * maxPerInsVtxCnt;
			}
		}
	}

}

template <typename MDIStructType>
template <typename MeshBufferIterator>
IMeshPackerBase::PackedMeshBufferData CCPUMeshPackerV1<MDIStructType>::commit(const MeshBufferIterator mbBegin, const MeshBufferIterator mbEnd, CCPUMeshPackerV1<MDIStructType>::ReservedAllocationMeshBuffers& ramb, core::aabbox3df* aabbs)
{
	MDIStructType* mdiBuffPtr = static_cast<MDIStructType*>(m_output.MDIDataBuffer->getPointer()) + ramb.mdiAllocationOffset;
	uint16_t* indexBuffPtr = static_cast<uint16_t*>(m_output.indexBuffer.buffer->getPointer()) + ramb.indexAllocationOffset;
	size_t verticesAddedToUnifiedBufferCnt = 0ull;
	size_t instancesAddedCnt = 0ull;

	uint32_t MDIStructsAddedCnt = 0u;

	size_t batchFirstIdx = ramb.indexAllocationOffset;
	size_t batchBaseVtx = ramb.vertexAllocationOffset;

	for (auto it = mbBegin; it != mbEnd; it++)
	{
		const auto mbPrimitiveType = (*it)->getPipeline()->getPrimitiveAssemblyParams().primitiveType;

		IdxBufferParams idxBufferParams = base_t::createNewIdxBufferParamsForNonTriangleListTopologies(*it);

		TriangleBatches triangleBatches = base_t::constructTriangleBatches(*it, idxBufferParams, aabbs);
		const auto& mbVtxInputParams = (*it)->getPipeline()->getVertexInputParams();

		const uint32_t batchCnt = triangleBatches.ranges.size() - 1u;
		for (uint32_t i = 0u; i < batchCnt; i++)
		{
			auto batchBegin = triangleBatches.ranges[i];
			auto batchEnd = triangleBatches.ranges[i + 1];

			const uint32_t triangleInBatchCnt = std::distance(batchBegin, batchEnd);
			const uint32_t idxInBatchCnt = 3 * triangleInBatchCnt;

			core::unordered_map<uint32_t, uint16_t> usedVertices = base_t::constructNewIndicesFromTriangleBatchAndUpdateUnifiedIndexBuffer(triangleBatches, i, indexBuffPtr);

			//copy deinterleaved vertices into unified vertex buffer
			for (uint16_t attrBit = 0x0001, location = 0; location < SVertexInputParams::MAX_ATTR_BUF_BINDING_COUNT; attrBit <<= 1, location++)
			{
				if (!(m_output.vertexInputParams.enabledAttribFlags & mbVtxInputParams.enabledAttribFlags & attrBit))
					continue;

				SVertexInputAttribParams attrib = m_output.vertexInputParams.attributes[location];
				SBufferBinding<ICPUBuffer>& vtxBuffBind = m_output.vertexBufferBindings[location];
				const E_VERTEX_INPUT_RATE inputRate = m_output.vertexInputParams.bindings[attrib.binding].inputRate;
				uint8_t* dstAttrPtr = static_cast<uint8_t*>(vtxBuffBind.buffer->getPointer()) + vtxBuffBind.offset;
				const size_t attrSize = asset::getTexelOrBlockBytesize(static_cast<E_FORMAT>(attrib.format));

				if (inputRate == EVIR_PER_VERTEX)
				{
					dstAttrPtr += (ramb.vertexAllocationOffset + verticesAddedToUnifiedBufferCnt) * attrSize;
					base_t::deinterleaveAndCopyAttribute(*it, location, usedVertices, dstAttrPtr);
				}
				else if (inputRate == EVIR_PER_INSTANCE)
				{
					dstAttrPtr += (ramb.instanceAllocationOffset + instancesAddedCnt) * attrSize;
					base_t::deinterleaveAndCopyPerInstanceAttribute(*it, location, dstAttrPtr);
				}
			}

			//construct mdi data
			MDIStructType MDIData;
			MDIData.count = idxInBatchCnt;
			MDIData.instanceCount = isInstancingEnabled ? (*it)->getInstanceCount() : 1u;
			MDIData.firstIndex = batchFirstIdx;
			MDIData.baseVertex = batchBaseVtx; //possible overflow?
			MDIData.baseInstance = isInstancingEnabled ? instancesAddedCnt : 0u;

			*mdiBuffPtr = MDIData;
			mdiBuffPtr++;
			MDIStructsAddedCnt++;

			batchFirstIdx += idxInBatchCnt;
			batchBaseVtx += usedVertices.size();

			verticesAddedToUnifiedBufferCnt += usedVertices.size();
		}

		instancesAddedCnt += (*it)->getInstanceCount();
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

	typename IMeshPackerBase::MeshPackerConfigParams<Iterator> firstInpuParams
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

			typename IMeshPackerBase::MeshPackerConfigParams<Iterator> configParams
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
		[&](const ICPUMeshBuffer* lhs, const ICPUMeshBuffer* rhs)
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
#endif
}
}

#endif
