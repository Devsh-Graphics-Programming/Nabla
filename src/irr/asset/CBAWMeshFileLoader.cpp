// Copyright (C) 2018 Krzysztof "Criss" Szenk
// This file is part of the "Irrlicht Engine" and "Build A World".
// For conditions of distribution and use, see copyright notice in irrlicht.h
// and on http://irrlicht.sourceforge.net/forum/viewtopic.php?f=2&t=49672

#include "CBAWMeshFileLoader.h"

#include <stack>

#include "os.h"
#include "CMemoryFile.h"
#include "CFinalBoneHierarchy.h"
#include "irr/asset/IAssetManager.h"
#include "irr/asset/bawformat/legacy/CBAWLegacy.h"
#include "irr/asset/bawformat/legacy/CBAWVersionUpFunctions.h"
#include "irr/video/CGPUMesh.h"
#include "irr/video/CGPUSkinnedMesh.h"

#include "lz4/lib/lz4.h"
#undef Bool
#include "lzma/C/LzmaDec.h"

namespace irr
{
namespace asset
{

struct LzmaMemMngmnt
{
        static void *alloc(ISzAllocPtr, size_t _size) { return _IRR_ALIGNED_MALLOC(_size,_IRR_SIMD_ALIGNMENT); }
        static void release(ISzAllocPtr, void* _addr) { _IRR_ALIGNED_FREE(_addr); }
    private:
        LzmaMemMngmnt() {}
};


CBAWMeshFileLoader::~CBAWMeshFileLoader()
{
}

CBAWMeshFileLoader::CBAWMeshFileLoader(IAssetManager* _manager) : m_manager(_manager), m_fileSystem(_manager->getFileSystem())
{
#ifdef _IRR_DEBUG
	setDebugName("CBAWMeshFileLoader");
#endif
}

SAssetBundle CBAWMeshFileLoader::loadAsset(io::IReadFile* _file, const asset::IAssetLoader::SAssetLoadParams& _params, asset::IAssetLoader::IAssetLoaderOverride* _override, uint32_t _hierarchyLevel)
{
#ifdef _IRR_DEBUG
    auto time = std::chrono::high_resolution_clock::now();
#endif // _IRR_DEBUG

	SContext ctx{
        asset::IAssetLoader::SAssetLoadContext{
            _params,
            _file
        }
    };

    ctx.inner.mainFile = tryCreateNewestFormatVersionFile(ctx.inner.mainFile, _override, std::make_integer_sequence<uint64_t, _IRR_BAW_FORMAT_VERSION>{});

    BlobHeaderLatest* headers = nullptr;

    auto exitRoutine = [&] {
        if (ctx.inner.mainFile != _file) // if mainFile is temparary memory file created just to update format to the newest version
            ctx.inner.mainFile->drop();
        ctx.releaseLoadedObjects();
        if (headers)
            _IRR_ALIGNED_FREE(headers);
    };
    auto exiter = core::makeRAIIExiter(exitRoutine);

    if (!verifyFile<asset::BAWFileVn<_IRR_BAW_FORMAT_VERSION>>(ctx))
    {
        return {};
    }

    uint32_t blobCnt{};
	uint32_t* offsets = nullptr;
    if (!validateHeaders<asset::BAWFileVn<_IRR_BAW_FORMAT_VERSION>, asset::BlobHeaderVn<_IRR_BAW_FORMAT_VERSION>>(&blobCnt, &offsets, (void**)&headers, ctx))
    {
        return {};
    }

	const uint32_t BLOBS_FILE_OFFSET = asset::BAWFileVn<_IRR_BAW_FORMAT_VERSION>{ {}, blobCnt }.calcBlobsOffset();

	core::unordered_map<uint64_t, SBlobData>::iterator meshBlobDataIter;

	for (uint32_t i = 0; i < blobCnt; ++i)
	{
		SBlobData data(headers + i, BLOBS_FILE_OFFSET + offsets[i]);
		const core::unordered_map<uint64_t, SBlobData>::iterator it = ctx.blobs.insert(std::make_pair(headers[i].handle, std::move(data))).first;
		if (data.header->blobType == asset::Blob::EBT_MESH || data.header->blobType == asset::Blob::EBT_SKINNED_MESH)
			meshBlobDataIter = it;
	}
	_IRR_ALIGNED_FREE(offsets);

    const std::string rootCacheKey = ctx.inner.mainFile->getFileName().c_str();

	asset::BlobLoadingParams params{
        this,
        m_manager,
        m_fileSystem,
        ctx.inner.mainFile->getFileName()[ctx.inner.mainFile->getFileName().size()-1] == '/' ? ctx.inner.mainFile->getFileName() : ctx.inner.mainFile->getFileName()+"/",
        ctx.inner.params,
        _override,
		{}
    };
	core::stack<SBlobData*> toLoad, toFinalize;
	toLoad.push(&meshBlobDataIter->second);
    toLoad.top()->hierarchyLvl = 0u;
	while (!toLoad.empty())
	{
		SBlobData* data = toLoad.top();
		toLoad.pop();

		const uint64_t handle = data->header->handle;
        const uint32_t size = data->header->blobSizeDecompr;
        const uint32_t blobType = data->header->blobType;
        const std::string thisCacheKey = genSubAssetCacheKey(rootCacheKey, handle);
        const uint32_t hierLvl = data->hierarchyLvl;

        uint8_t decrKey[16];
        size_t decrKeyLen = 16u;
        uint32_t attempt = 0u;
        const void* blob = nullptr;
        // todo: supposedFilename arg is missing (empty string) - what is it?
        while (_override->getDecryptionKey(decrKey, decrKeyLen, attempt, ctx.inner.mainFile, "", thisCacheKey, ctx.inner, hierLvl))
        {
            if (!((data->header->compressionType & asset::Blob::EBCT_AES128_GCM) && decrKeyLen != 16u))
                blob = data->heapBlob = tryReadBlobOnStack(*data, ctx, decrKey);
            if (blob)
                break;
            ++attempt;
        }

		if (!blob)
		{
            return {};
		}

		core::unordered_set<uint64_t> deps = ctx.loadingMgr.getNeededDeps(blobType, blob);
        for (auto it = deps.begin(); it != deps.end(); ++it)
        {
            if (ctx.createdObjs.find(*it) == ctx.createdObjs.end())
            {
                toLoad.push(&ctx.blobs[*it]);
                toLoad.top()->hierarchyLvl = hierLvl+1u;
            }
        }

        auto foundBundle = _override->findCachedAsset(thisCacheKey, nullptr, ctx.inner, hierLvl).getContents();
        if (foundBundle.first!=foundBundle.second)
        {
            ctx.createdObjs[handle] = toAddrUsedByBlobsLoadingMgr(foundBundle.first->get(), blobType);
            continue;
        }

		bool fail = !(ctx.createdObjs[handle] = ctx.loadingMgr.instantiateEmpty(blobType, blob, size, params));

		if (fail)
		{
            return {};
		}

		if (!deps.size())
		{
            void* obj = ctx.createdObjs[handle];
			ctx.loadingMgr.finalize(blobType, obj, blob, size, ctx.createdObjs, params);
            _IRR_ALIGNED_FREE(data->heapBlob);
			blob = data->heapBlob = nullptr;
            insertAssetIntoCache(ctx, _override, obj, blobType, hierLvl, thisCacheKey);
		}
		else
			toFinalize.push(data);
	}

	void* retval = nullptr;
	while (!toFinalize.empty())
	{
		SBlobData* data = toFinalize.top();
		toFinalize.pop();

		const void* blob = data->heapBlob;
		const uint64_t handle = data->header->handle;
		const uint32_t size = data->header->blobSizeDecompr;
		const uint32_t blobType = data->header->blobType;
        const uint32_t hierLvl = data->hierarchyLvl;
        const std::string thisCacheKey = genSubAssetCacheKey(rootCacheKey, handle);

		retval = ctx.loadingMgr.finalize(blobType, ctx.createdObjs[handle], blob, size, ctx.createdObjs, params); // last one will always be mesh
        if (!toFinalize.empty()) // don't cache root-asset (mesh) as sub-asset because it'll be cached by asset manager directly (and there's only one IAsset::cacheKey)
            insertAssetIntoCache(ctx, _override, retval, blobType, hierLvl, thisCacheKey);
	}

	// flip meshes if needed
	const core::matrix3x4SIMD xFlip(
		-1.f,0.f,0.f,0.f,
		 0.f,1.f,0.f,0.f,
		 0.f,0.f,1.f,0.f
	);
	while (!params.meshesToFlip.empty())
	{
		auto mesh = params.meshesToFlip.top();
		params.meshesToFlip.pop();

		if (mesh->getReferenceCount() < 2)
			continue;

		auto bbox = mesh->getBoundingBox();
		bbox.MinEdge.X *= -1.f;
		bbox.MaxEdge.X *= -1.f;
		mesh->setBoundingBox(bbox);
		
		bool isSkinnedMesh = mesh->getMeshType()==asset::EMT_ANIMATED_SKINNED;
		if (isSkinnedMesh)
		{
			auto sm = static_cast<CCPUSkinnedMesh*>(mesh.get());
			const auto* fbhRef = sm->getBoneReferenceHierarchy();

			const auto boneCount = fbhRef->getBoneCount();
			const CFinalBoneHierarchy::BoneReferenceData* bones = fbhRef->getBoneData();
			core::vector<core::stringc> boneNames(boneCount);
			for (auto k = 0; k < boneCount; k++)
				boneNames[k] = fbhRef->getBoneName(k);
			auto fbhCopy = core::make_smart_refctd_ptr<CFinalBoneHierarchy>(
				bones, bones + boneCount,
				boneNames.data(), boneNames.data() + boneCount,
				fbhRef->getBoneTreeLevelEnd(), fbhRef->getBoneTreeLevelEnd() + fbhRef->getHierarchyLevels(),
				fbhRef->getKeys(), fbhRef->getKeys() + fbhRef->getKeyFrameCount(),
				fbhRef->getInterpolatedAnimationData(), fbhRef->getInterpolatedAnimationData() + fbhRef->getAnimationCount(),
				fbhRef->getNonInterpolatedAnimationData(), fbhRef->getNonInterpolatedAnimationData() + fbhRef->getAnimationCount(),
				!fbhRef->flipsXOnOutput()
			);

			// flip
            auto newbones = const_cast<CFinalBoneHierarchy::BoneReferenceData*>(const_cast<const CFinalBoneHierarchy*>(fbhCopy.get())->getBoneData());
            for (auto j=0u; j<fbhCopy->getBoneCount(); j++)
            {
                auto& bone = newbones[j];
				bone.PoseBindMatrix = core::concatenateBFollowedByA(bone.PoseBindMatrix,xFlip);
				bone.MinBBoxEdge[0] = -bone.MinBBoxEdge[0];
				bone.MaxBBoxEdge[0] = -bone.MaxBBoxEdge[0];
            }

			sm->setBoneReferenceHierarchy(std::move(fbhCopy));
		}

		for (auto i = 0ull; i < mesh->getMeshBufferCount(); ++i)
		{
			auto mb = mesh->getMeshBuffer(i);
			auto* originalMeshFormat = mb->getMeshDataAndFormat();
			const auto positionAttribute = mb->getPositionAttributeIx();
			const auto normalAttribute = mb->getNormalAttributeIx();
			auto normalBuffer = core::smart_refctd_ptr<ICPUBuffer>(const_cast<ICPUBuffer*>(originalMeshFormat->getMappedBuffer(normalAttribute)));
			const bool hasNormal = normalBuffer && normalBuffer->getPointer();

			auto copy = m_manager->getMeshManipulator()->createMeshBufferDuplicate(mb);
			auto meshFormat = copy->getMeshDataAndFormat();

			meshFormat->setIndexBuffer(core::smart_refctd_ptr<ICPUBuffer>(const_cast<ICPUBuffer*>(originalMeshFormat->getIndexBuffer())));
			for (uint32_t i=0u; i<EVAI_COUNT; i++)
			{
				auto attrId = static_cast<E_VERTEX_ATTRIBUTE_ID>(i);
				const ICPUBuffer* oldBuf = originalMeshFormat->getMappedBuffer(attrId);
				if (!oldBuf || i==positionAttribute || i==normalAttribute)
					continue;

				meshFormat->setVertexAttrBuffer(
					core::smart_refctd_ptr<ICPUBuffer>(const_cast<ICPUBuffer*>(oldBuf)), attrId, originalMeshFormat->getAttribFormat(attrId),
					originalMeshFormat->getMappedBufferStride(attrId), originalMeshFormat->getMappedBufferOffset(attrId),
					originalMeshFormat->getAttribDivisor(attrId)
				);
			}

			mb->setMeshDataAndFormat(core::smart_refctd_ptr<IMeshDataFormatDesc<ICPUBuffer> >(meshFormat));

			// do the flip
			const auto positionFormat = originalMeshFormat->getAttribFormat(positionAttribute);
			// we cannot handle the vertices changing places relative to the bone (no unsigned position formats)
			assert(!isSkinnedMesh || asset::isSignedFormat(positionFormat) || asset::isFloatingPointFormat(positionFormat));
			const auto positionDivisor = originalMeshFormat->getAttribDivisor(positionAttribute);
			assert(positionDivisor<2u);

			const auto normalFormat = originalMeshFormat->getAttribFormat(normalAttribute);
			const auto normalDivisor = originalMeshFormat->getAttribDivisor(normalAttribute);
			assert(normalDivisor<2u);

			const auto vertexCount = mb->calcVertexCount();
			const auto instanceCount = mb->getInstanceCount();

			// get attributes from copy meshbuffer
			// flip attributes
			// set attributes on meshbuffer (writes to new buffers)
			auto flipAndCopyAttribute_impl = [&](auto& tmpStorage, auto divisor, auto attrID)
			{
				const auto count = divisor ? instanceCount:vertexCount;
				for (std::remove_const<decltype(count)>::type ix = 0; ix < count; ix++)
				{
					copy->getAttribute(tmpStorage, attrID, ix);
					tmpStorage[0] = -tmpStorage[0];
					mb->setAttribute(tmpStorage, attrID, ix);
				}
			};
			auto flipAndCopyAttribute = [&](auto divisor, auto attrID, auto attrFormat)
			{
				if (asset::isIntegerFormat(attrFormat))
				{
					// whole bunch of integer hacks make it work perfectly
					uint32_t out[] = { 0, 0, 0, 1 };
					flipAndCopyAttribute_impl(out,divisor,attrID);
				}
				else
				{
					core::vectorSIMDf out(0.f, 0.f, 0.f, 1.f);
					flipAndCopyAttribute_impl(out,divisor,attrID);
				}
			};

			flipAndCopyAttribute(positionDivisor, positionAttribute, positionFormat);
			if (hasNormal)
				flipAndCopyAttribute(normalDivisor, normalAttribute, normalFormat);

			// drop the copies we don't use (implicit)

			auto bbox = mb->getBoundingBox();
			bbox.MinEdge.X *= -1.f;
			bbox.MaxEdge.X *= -1.f;
			if (!asset::isSignedFormat(positionFormat))
			{
				assert(asset::isNormalizedFormat(positionFormat)); // TODO: a function that will give us the MAX range for integer formats
				auto range = 1.f;
				bbox.MinEdge.X += range;
				bbox.MaxEdge.X += range;
			}
			mb->setBoundingBox(bbox);
		}
	}

	ctx.releaseAllButThisOne(meshBlobDataIter); // call drop on all loaded objects except mesh

#ifdef _IRR_DEBUG
	std::ostringstream tmpString("Time to load ");
	tmpString.seekp(0, std::ios_base::end);
	tmpString << "BAW file: " << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now()-time).count() << "us";
	os::Printer::log(tmpString.str());
#endif // _IRR_DEBUG

	asset::ICPUMesh* mesh = reinterpret_cast<asset::ICPUMesh*>(retval);
		
    return SAssetBundle({core::smart_refctd_ptr<asset::IAsset>(mesh,core::dont_grab)});
}

bool CBAWMeshFileLoader::safeRead(io::IReadFile * _file, void * _buf, size_t _size) const
{
	if (_file->getPos() + _size > _file->getSize())
		return false;
	_file->read(_buf, _size);
	return true;
}

bool CBAWMeshFileLoader::decompressLzma(void* _dst, size_t _dstSize, const void* _src, size_t _srcSize) const
{
	SizeT dstSize = _dstSize;
	SizeT srcSize = _srcSize - LZMA_PROPS_SIZE;
	ELzmaStatus status;
	ISzAlloc alloc{&asset::LzmaMemMngmnt::alloc, &asset::LzmaMemMngmnt::release};
	const SRes res = LzmaDecode((Byte*)_dst, &dstSize, (const Byte*)(_src)+LZMA_PROPS_SIZE, &srcSize, (const Byte*)_src, LZMA_PROPS_SIZE, LZMA_FINISH_ANY, &status, &alloc);
	if (res != SZ_OK)
		return false;
	return true;
}

bool CBAWMeshFileLoader::decompressLz4(void * _dst, size_t _dstSize, const void * _src, size_t _srcSize) const
{
	int res = LZ4_decompress_safe((const char*)_src, (char*)_dst, _srcSize, _dstSize);
	return res >= 0;
}

}} // irr::scene
