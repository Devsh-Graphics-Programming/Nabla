// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "nbl/core/declarations.h"

#include "COverdrawMeshOptimizer.h"

#include <vector>
#include <functional>

#include "CMeshManipulator.h"

namespace nbl::asset
{

void COverdrawMeshOptimizer::createOptimized(asset::ICPUMeshBuffer* _outbuffer, const asset::ICPUMeshBuffer* _inbuffer, float _threshold, const system::logger_opt_ptr& logger)
{
	if (!_outbuffer || !_inbuffer)
		return;

	const asset::E_INDEX_TYPE indexType = _inbuffer->getIndexType();
	const size_t indexSize = indexType == asset::EIT_16BIT ? sizeof(uint16_t):sizeof(uint32_t);

	const uint32_t idxCount = _inbuffer->getIndexCount();
	const void* const inIndices = _inbuffer->getIndices();
	if (idxCount==0u || indexType==asset::EIT_UNKNOWN || !inIndices)
	{
		logger.log("Overdraw optimization: no index buffer -- mesh buffer left unchanged.");
		return;
	}

	void* const outIndices = _outbuffer->getIndices();
	if (_outbuffer->getIndexCount()!=idxCount || _outbuffer->getIndexType()!=indexType || !outIndices)
	{
		logger.log("Overdraw optimization: output meshbuffer's index buffer does not match input -- mesh buffer left unchanged.");
		return;
	}

	void* indexCopy = nullptr;
	const uint16_t* inIndices16 = reinterpret_cast<const uint16_t*>(inIndices);
	const uint32_t* inIndices32 = reinterpret_cast<const uint32_t*>(inIndices);
	// TODO: more accure check for overlap
	if (_outbuffer->getIndexBufferBinding().buffer==_inbuffer->getIndexBufferBinding().buffer)
	{
		const size_t dataSize = indexSize*idxCount;
		indexCopy = _NBL_ALIGNED_MALLOC(dataSize,indexSize);
		memcpy(outIndices,inIndices,dataSize);
		inIndices16 = reinterpret_cast<const uint16_t*>(dataSize);
		inIndices32 = reinterpret_cast<const uint32_t*>(dataSize);
	}
	uint16_t* outIndices16 = reinterpret_cast<uint16_t*>(outIndices);
	uint32_t* outIndices32 = reinterpret_cast<uint32_t*>(outIndices);

	const uint32_t vertexCount = IMeshManipulator::upperBoundVertexID(_inbuffer);
	core::vector<core::vectorSIMDf> vertexPositions(vertexCount);
	for (uint32_t i=0u; i<vertexCount; ++i)
		_inbuffer->getAttribute(vertexPositions[i],_inbuffer->getPositionAttributeIx(),i);

	uint32_t* const hardClusters = reinterpret_cast<uint32_t*>(_NBL_ALIGNED_MALLOC((idxCount/3)*sizeof(uint32_t),_NBL_SIMD_ALIGNMENT));
	const size_t hardClusterCount = indexType == asset::EIT_16BIT ?
		genHardBoundaries(hardClusters, inIndices16, idxCount, vertexCount) :
		genHardBoundaries(hardClusters, inIndices32, idxCount, vertexCount);

	uint32_t* const softClusters = reinterpret_cast<uint32_t*>(_NBL_ALIGNED_MALLOC((idxCount/3+1)*sizeof(uint32_t),_NBL_SIMD_ALIGNMENT));
	const size_t softClusterCount = indexType == asset::EIT_16BIT ?
		genSoftBoundaries(softClusters, inIndices16, idxCount, vertexCount, hardClusters, hardClusterCount, _threshold) :
		genSoftBoundaries(softClusters, inIndices32, idxCount, vertexCount, hardClusters, hardClusterCount, _threshold);

	ClusterSortData* sortedData = (ClusterSortData*)_NBL_ALIGNED_MALLOC(softClusterCount*sizeof(ClusterSortData),_NBL_SIMD_ALIGNMENT);
	if (indexType == asset::EIT_16BIT)
		calcSortData(sortedData, inIndices16, idxCount, vertexPositions, softClusters, softClusterCount);
	else
		calcSortData(sortedData, inIndices32, idxCount, vertexPositions, softClusters, softClusterCount);

	std::stable_sort(sortedData, sortedData+softClusterCount, std::greater<ClusterSortData>()); // TODO: use core::radix_sort

	auto reorderIndices = [&](auto* out, const auto* in)
	{
		for (size_t it = 0, jt = 0; it < softClusterCount; ++it)
		{
			const uint32_t cluster = sortedData[it].cluster;

			size_t start = softClusters[cluster];
			size_t end = (cluster+1<softClusterCount) ? softClusters[cluster+1]:idxCount/3;

			for (size_t i = start; i < end; ++i)
			{
				out[jt++] = in[3 * i + 0];
				out[jt++] = in[3 * i + 1];
				out[jt++] = in[3 * i + 2];
			}
		}
	};
	if (indexType==asset::EIT_16BIT)
		reorderIndices(outIndices16,inIndices16);
	else
		reorderIndices(outIndices32,inIndices32);

	if (indexCopy)
		_NBL_ALIGNED_FREE(indexCopy);
	_NBL_ALIGNED_FREE(hardClusters);
	_NBL_ALIGNED_FREE(softClusters);
	_NBL_ALIGNED_FREE(sortedData);
}

template<typename IdxT>
size_t COverdrawMeshOptimizer::genHardBoundaries(uint32_t* _dst, const IdxT* _indices, size_t _idxCount, size_t _vtxCount)
{
	size_t* cacheTimestamps = (size_t*)_NBL_ALIGNED_MALLOC(sizeof(size_t)*_vtxCount,_NBL_SIMD_ALIGNMENT);
	memset(cacheTimestamps, 0, sizeof(size_t)*_vtxCount);

	size_t timestamp = CACHE_SIZE + 1;

	const size_t faceCount = _idxCount / 3;

	size_t retval = 0u;
	for (size_t i = 0u; i < faceCount; ++i)
	{
		size_t misses = updateCache(_indices[3*i + 0], _indices[3*i + 1], _indices[3*i + 2], cacheTimestamps, timestamp);

		// when all three vertices are not in the cache it's usually relatively safe to assume that this is a new patch in the mesh
		// that is disjoint from previous vertices; sometimes it might come back to reference existing vertices but that frequently
		// suggests an inefficiency in the vertex cache optimization algorithm
		// usually the first triangle has 3 misses unless it's degenerate - thus we make sure the first cluster always starts with 0
		if (!i || misses == 3)
			_dst[retval++] = (uint32_t)i;
	}

	_NBL_ALIGNED_FREE(cacheTimestamps);

	return retval;
}

template<typename IdxT>
size_t COverdrawMeshOptimizer::genSoftBoundaries(uint32_t* _dst, const IdxT* _indices, size_t _idxCount, size_t _vtxCount, const uint32_t* _clusters, size_t _clusterCount, float _threshold)
{
	size_t* cacheTimestamps = (size_t*)_NBL_ALIGNED_MALLOC(sizeof(size_t)*_vtxCount,_NBL_SIMD_ALIGNMENT);
	memset(cacheTimestamps, 0, sizeof(size_t)*_vtxCount);

	size_t timestamp = 0u;

	size_t retval = 0u;

	for (size_t i = 0u; i < _clusterCount; ++i)
	{
		const size_t start = _clusters[i];
		const size_t end = (i + 1 < _clusterCount) ? _clusters[i+1] : _idxCount/3;

		_NBL_DEBUG_BREAK_IF(start > end);

		timestamp += CACHE_SIZE + 1; // reset cache

		size_t clusterMisses = 0u; // cluster ACMR
		for (size_t j = start; j < end; ++j)
			clusterMisses += updateCache(_indices[j*3 + 0], _indices[j*3 + 1], _indices[j*3 + 2], cacheTimestamps, timestamp);

		const float clusterThreshold = _threshold * (float(clusterMisses) / (end - start));

		_dst[retval++] = (uint32_t)start;

		timestamp += CACHE_SIZE + 1; // reset cache

		size_t runningMisses = 0u;
		size_t runningFaces = 0u;
		for (size_t j = start; j < end; ++j)
		{
			runningMisses += updateCache(_indices[j*3 + 0], _indices[j*3 + 1], _indices[j*3 + 2], cacheTimestamps, timestamp);
			++runningFaces;

			if (float(runningMisses)/runningFaces <= clusterThreshold)
			{
				// we have reached the target ACMR with the current triangle so we need to start a new cluster on the next one
				// note that this may mean that we add 'end` to destination for the last triangle, which will imply that the last
				// cluster is empty; however, the 'pop back' after the loop will clean it up
				_dst[retval++] = (uint32_t)j + 1;

				timestamp += CACHE_SIZE + 1; // reset cache

				runningMisses = 0u;
				runningFaces = 0u;
			}
		}

		// each time we reach the target ACMR we flush the cluster
		// this means that the last cluster is by definition not very good - there are frequent cases where we are left with a few triangles
		// in the last cluster, producing a very bad ACMR and significantly penalizing the overall results
		// thus we remove the last cluster boundary, merging the last complete cluster with the last incomplete one
		// there are sometimes cases when the last cluster is actually good enough - in which case the code above would have added 'end'
		// to the cluster boundary array which we need to remove anyway - this code will do that automatically
		if (_dst[retval - 1] != start)
			--retval;
	}

	_NBL_ALIGNED_FREE(cacheTimestamps);

	_NBL_DEBUG_BREAK_IF(retval < _clusterCount || retval > _idxCount/3u)

	return retval;
}

template<typename IdxT>
void COverdrawMeshOptimizer::calcSortData(ClusterSortData* _dst, const IdxT* _indices, size_t _idxCount, const core::vector<core::vectorSIMDf>& _positions, const uint32_t* _clusters, size_t _clusterCount)
{
	core::vectorSIMDf meshCentroid;
	for (size_t i = 0u; i < _idxCount; ++i)
		meshCentroid += _positions[_indices[i]];
	meshCentroid /= (float)_idxCount;

	for (size_t cluster = 0; cluster < _clusterCount; ++cluster)
	{
		// TODO: why are the fucking clusters only 1 triangle !?!?!?
		const size_t begin = _clusters[cluster] * 3;
		const size_t end = (_clusterCount > cluster + 1) ? _clusters[cluster+1] * 3 : _idxCount;
		_NBL_DEBUG_BREAK_IF(begin > end);

		float clusterArea = 0.f;
		core::vectorSIMDf clusterCentroid;
		core::vectorSIMDf clusterNormal;

		for (size_t i = begin; i < end; i += 3)
		{
			const core::vectorSIMDf& p0 = _positions[_indices[i+0]];
			const core::vectorSIMDf& p1 = _positions[_indices[i+1]];
			const core::vectorSIMDf& p2 = _positions[_indices[i+2]];

			const core::vectorSIMDf normal = core::cross(p1 - p0,p2 - p0);

			const auto area = core::length(normal);

			clusterCentroid += (p0 + p1 + p2) * (area / 3.f);
			clusterNormal += normal;
			clusterArea += area[0];
		}

		const float invClusterArea = !clusterArea ? 0.f : 1.f/clusterArea;

		clusterCentroid *= invClusterArea;
		clusterNormal = core::normalize(clusterNormal);

		core::vectorSIMDf centroidVec = clusterCentroid - meshCentroid;

		_dst[cluster].cluster = (uint32_t)cluster;
		_dst[cluster].dot = core::dot(centroidVec,clusterNormal)[0];
	}
}

size_t COverdrawMeshOptimizer::updateCache(uint32_t _a, uint32_t _b, uint32_t _c, size_t* _cacheTimestamps, size_t& _timestamp)
{
	size_t cacheMisses = 0u;

	if (_timestamp - _cacheTimestamps[_a] > CACHE_SIZE)
	{
		_cacheTimestamps[_a] = _timestamp++;
		++cacheMisses;
	}
	if (_timestamp - _cacheTimestamps[_b] > CACHE_SIZE)
	{
		_cacheTimestamps[_b] = _timestamp++;
		++cacheMisses;
	}
	if (_timestamp - _cacheTimestamps[_c] > CACHE_SIZE)
	{
		_cacheTimestamps[_c] = _timestamp++;
		++cacheMisses;
	}

	return cacheMisses;
}

} // nbl::asset
