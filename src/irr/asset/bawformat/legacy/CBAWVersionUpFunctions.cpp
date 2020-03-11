#include "CBAWVersionUpFunctions.h"

namespace irr
{
    namespace asset
    {
		template<>
		io::IReadFile* CBAWMeshFileLoader::createConvertIntoVer_spec<3>(SContext & _ctx, io::IReadFile * _baw2file, asset::IAssetLoader::IAssetLoaderOverride * _override, const CommonDataTuple<2> & _common)
		{
			uint32_t blobCnt{};
			BlobHeaderVn<2> * headers = nullptr;
			uint32_t* offsets = nullptr;
			uint32_t baseOffsetv2{};
			uint32_t baseOffsetv3{};
			std::tie(blobCnt, headers, offsets, baseOffsetv2, baseOffsetv3) = _common;

			io::CMemoryWriteFile* const baw3mem = new io::CMemoryWriteFile(0u, _baw2file->getFileName());

			std::vector<uint32_t> newoffsets(blobCnt);
			int32_t offsetDiff = 0;

			const uint64_t zeroBuffer[] = {0,0,0,0};
			for (uint32_t i = 0u; i < blobCnt; ++i)
			{
				BlobHeaderVn<2>& hdr = headers[i];
				const uint32_t offset = offsets[i];
				uint32_t& newoffset = newoffsets[i];

				newoffset = offset + offsetDiff;

				uint8_t stackmem[1u << 10];
				uint32_t attempt = 0u;
				uint8_t decrKey[16];
				size_t decrKeyLen = 16u;

				void* blob = nullptr;

				/* to state blob's/asset's hierarchy level we'd have to load (and possibly decrypt and decompress) the blob
				 however we don't need to do this here since we know format's version (baw v1) and so we can be sure that hierarchy level for FinalBoneHierarchy is 3 (won't be true for v2 to v3 conversion)
				*/

				auto fetchRawBlob = [&](uint32_t hierarchyLevel)
				{
					while (_override->getDecryptionKey(decrKey, decrKeyLen, attempt, _baw2file, "", genSubAssetCacheKey(_baw2file->getFileName().c_str(), hdr.handle), _ctx.inner, hierarchyLevel))
					{
						if (!((hdr.compressionType & asset::Blob::EBCT_AES128_GCM) && decrKeyLen != 16u))
							blob = tryReadBlobOnStack<BlobHeaderVn<2>>(SBlobData_t<BlobHeaderVn<2>>(&hdr, baseOffsetv2 + offset), _ctx, decrKey, stackmem, sizeof(stackmem));
						if (blob)
							break;
						++attempt;
					}
				};

				const uint32_t absOffset = baseOffsetv3 + newoffset;
				baw3mem->seek(absOffset);

				const auto prevBlobSz = hdr.effectiveSize();
				switch (hdr.blobType)
				{
					case asset::Blob::EBT_MESH:
					{
						constexpr uint32_t MESH_HIERARCHY_LVL = 0u;
						fetchRawBlob(MESH_HIERARCHY_LVL);
						if (!blob)
							break;

						baw3mem->write(blob, offsetof(MeshBlobV3,meshFlags));
						baw3mem->write(zeroBuffer, sizeof(MeshBlobV3::meshFlags));
						baw3mem->write(reinterpret_cast<uint8_t*>(blob)+offsetof(MeshBlobV3,meshFlags), hdr.blobSizeDecompr-offsetof(MeshBlobV3,meshFlags));

						const auto sizeTakingAllBuffersIntoAcount = hdr.blobSizeDecompr + sizeof(asset::MeshBlobV3) - sizeof(asset::legacyv2::MeshBlobV2);
						hdr.blobSizeDecompr = hdr.blobSize = sizeTakingAllBuffersIntoAcount;

						break;
					}
					case asset::Blob::EBT_SKINNED_MESH:
					{
						constexpr uint32_t SKINNED_MESH_HIERARCHY_LVL = 0u;
						fetchRawBlob(SKINNED_MESH_HIERARCHY_LVL);
						if (!blob)
							break;

						baw3mem->write(blob, offsetof(SkinnedMeshBlobV3, meshFlags));
						baw3mem->write(zeroBuffer, sizeof(SkinnedMeshBlobV3::meshFlags));
						baw3mem->write(reinterpret_cast<uint8_t*>(blob) + offsetof(SkinnedMeshBlobV3, meshFlags), hdr.blobSizeDecompr - offsetof(SkinnedMeshBlobV3, meshFlags));

						const auto sizeTakingAllBuffersIntoAcount = hdr.blobSizeDecompr + sizeof(asset::SkinnedMeshBlobV3) - sizeof(asset::legacyv2::SkinnedMeshBlobV2);
						hdr.blobSizeDecompr = hdr.blobSize = sizeTakingAllBuffersIntoAcount;

						break;
					}
					case asset::Blob::EBT_MESH_BUFFER:
					{
						constexpr uint32_t MESH_BUFFER_HIERARCHY_LVL = 1u;
						fetchRawBlob(MESH_BUFFER_HIERARCHY_LVL);
						if (!blob)
							break;

						hdr.blobSizeDecompr = hdr.blobSize = sizeof(asset::MeshBufferBlobV3);
						baw3mem->write(blob,offsetof(MeshBufferBlobV3,normalAttrId));
						reinterpret_cast<uint32_t&>(stackmem[0]) = EVAI_ATTR3;
						baw3mem->write(stackmem,sizeof(uint32_t));

						break;
					}
					case asset::Blob::EBT_SKINNED_MESH_BUFFER:
					{
						constexpr uint32_t SKINNED_MESH_BUFFER_HIERARCHY_LVL = 1u;
						fetchRawBlob(SKINNED_MESH_BUFFER_HIERARCHY_LVL);
						if (!blob)
							break;

						hdr.blobSizeDecompr = hdr.blobSize = sizeof(asset::SkinnedMeshBufferBlobV3);
						baw3mem->write(blob,offsetof(SkinnedMeshBufferBlobV3,normalAttrId));
						reinterpret_cast<uint32_t&>(stackmem[0]) = EVAI_ATTR3;
						baw3mem->write(stackmem,sizeof(uint32_t));

						break;
					}
					case asset::Blob::EBT_FINAL_BONE_HIERARCHY:
					{
						constexpr uint32_t FINAL_BONE_HIERARCHY_HIERARCHY_LVL = 1u;
						fetchRawBlob(FINAL_BONE_HIERARCHY_HIERARCHY_LVL);
						if (!blob)
							break;

						static_assert(offsetof(FinalBoneHierarchyBlobV3, finalBoneHierarchyFlags) == 0); // to make sure finalBoneHierarchyFlags in first member in the struct
						baw3mem->write(zeroBuffer, sizeof(FinalBoneHierarchyBlobV3::finalBoneHierarchyFlags));
						baw3mem->write(reinterpret_cast<uint8_t*>(blob), hdr.blobSizeDecompr);

						const auto sizeTakingAllBuffersIntoAcount = hdr.blobSizeDecompr + sizeof(asset::FinalBoneHierarchyBlobV3) - sizeof(asset::legacyv2::FinalBoneHierarchyBlobV2);
						hdr.blobSizeDecompr = hdr.blobSize = sizeTakingAllBuffersIntoAcount;

						break;
					}
				}

				if (!blob)
					continue;


				hdr.compressionType = asset::Blob::EBCT_RAW;
				core::XXHash_256(reinterpret_cast<uint8_t*>(baw3mem->getPointer()) + absOffset, hdr.blobSize, hdr.blobHash);
				offsetDiff += static_cast<int32_t>(hdr.blobSizeDecompr) - prevBlobSz;

				if (blob!=stackmem)
					_IRR_ALIGNED_FREE(blob);
			}
			uint64_t fileHeader[4]{ 0u, 0u, 0u, 3u/*baw v3*/ };
			memcpy(fileHeader, BAWFileV3::HEADER_STRING, strlen(BAWFileV3::HEADER_STRING));
			baw3mem->seek(0u);
			baw3mem->write(fileHeader, sizeof(fileHeader));
			baw3mem->write(&blobCnt, 4);
			baw3mem->write(_ctx.iv, 16);
			baw3mem->write(newoffsets.data(), newoffsets.size() * 4);
			baw3mem->write(headers, blobCnt * sizeof(headers[0])); // blob header in v2 and in v3 is exact same thing, so we can do this

			uint8_t stackmem[1u << 13]{};
			size_t newFileSz = 0u;
			for (uint32_t i = 0u; i < blobCnt; ++i)
			{
				uint32_t sz = headers[i].effectiveSize();
				void* blob = nullptr;
				
				switch (headers[i].blobType)
				{
					case asset::Blob::EBT_MESH:
					case asset::Blob::EBT_SKINNED_MESH:
					case asset::Blob::EBT_MESH_BUFFER:
					case asset::Blob::EBT_SKINNED_MESH_BUFFER:
					case asset::Blob::EBT_FINAL_BONE_HIERARCHY:
						sz = 0u;
						break;
					default:
					{
						_baw2file->seek(baseOffsetv2 + offsets[i]);
						if (sz <= sizeof(stackmem))
							blob = stackmem;
						else
							blob = _IRR_ALIGNED_MALLOC(sz, _IRR_SIMD_ALIGNMENT);

						_baw2file->read(blob, sz);
						break;
					}
				}

				baw3mem->seek(baseOffsetv3 + newoffsets[i]);
				baw3mem->write(blob, sz);

				if (blob && blob != stackmem)
					_IRR_ALIGNED_FREE(blob);

				newFileSz = baseOffsetv3 + newoffsets[i] + sz;
			}

			_IRR_ALIGNED_FREE(offsets);
			_IRR_ALIGNED_FREE(headers);

			auto ret = new io::CMemoryReadFile(baw3mem->getPointer(), baw3mem->getSize(), _baw2file->getFileName());
			baw3mem->drop();
			return ret;
		}

        template<>
        io::IReadFile* CBAWMeshFileLoader::createConvertIntoVer_spec<2>(SContext& _ctx, io::IReadFile* _baw1file, asset::IAssetLoader::IAssetLoaderOverride* _override, const CommonDataTuple<1>& _common)
        {
            uint32_t blobCnt{};
			BlobHeaderVn<1>* headers = nullptr;
            uint32_t* offsets = nullptr;
            uint32_t baseOffsetv1{};
            uint32_t baseOffsetv2{};
            std::tie(blobCnt, headers, offsets, baseOffsetv1, baseOffsetv2) = _common;

            io::CMemoryWriteFile* const baw2mem = new io::CMemoryWriteFile(0u, _baw1file->getFileName());

            std::vector<uint32_t> newoffsets(blobCnt);
            int32_t offsetDiff = 0;
            for (uint32_t i = 0u; i < blobCnt; ++i)
            {
				BlobHeaderVn<1>& hdr = headers[i];
                const uint32_t offset = offsets[i];
                uint32_t& newoffset = newoffsets[i];

                newoffset = offset + offsetDiff;

                if (hdr.blobType == asset::Blob::EBT_FINAL_BONE_HIERARCHY)
                {
                    const uint32_t offset = offsets[i];

                    uint8_t stackmem[1u << 10];
                    uint32_t attempt = 0u;
                    uint8_t decrKey[16];
                    size_t decrKeyLen = 16u;
                    void* blob = nullptr;
                    /* to state blob's/asset's hierarchy level we'd have to load (and possibly decrypt and decompress) the blob
                    however we don't need to do this here since we know format's version (baw v1) and so we can be sure that hierarchy level for FinalBoneHierarchy is 3 (won't be true for v2 to v3 conversion)
                    */
                    constexpr uint32_t FINALBONEHIERARCHY_HIERARCHY_LVL = 3u;
                    while (_override->getDecryptionKey(decrKey, decrKeyLen, attempt, _baw1file, "", genSubAssetCacheKey(_baw1file->getFileName().c_str(), hdr.handle), _ctx.inner, FINALBONEHIERARCHY_HIERARCHY_LVL))
                    {
                        if (!((hdr.compressionType & asset::Blob::EBCT_AES128_GCM) && decrKeyLen != 16u))
                            blob = tryReadBlobOnStack<BlobHeaderVn<1>>(SBlobData_t<BlobHeaderVn<1>>(&hdr, baseOffsetv1 + offset), _ctx, decrKey, stackmem, sizeof(stackmem));
                        if (blob)
                            break;
                        ++attempt;
                    }

                    // patch the blob
                    if (blob)
                    {
                        auto* hierarchy = reinterpret_cast<legacyv1::FinalBoneHierarchyBlobV1*>(blob);

                        auto bones = reinterpret_cast<CFinalBoneHierarchy::BoneReferenceData*>(hierarchy->getBoneData());
                        for (auto j = 0u; j < hierarchy->boneCount; j++)
                        {
                            core::matrix3x4SIMD mtx = bones[j].PoseBindMatrix;
                            const auto* in = mtx.rows[0].pointer;
                            auto* out = bones[j].PoseBindMatrix.rows[0].pointer;
                            out[0] = in[0];
                            out[4] = in[1];
                            out[8] = in[2];
                            out[1] = in[3];
                            out[5] = in[4];
                            out[9] = in[5];
                            out[2] = in[6];
                            out[6] = in[7];
                            out[10] = in[8];
                            out[3] = in[9];
                            out[7] = in[10];
                            out[11] = in[11];
                        }

                        const uint32_t absOffset = baseOffsetv2 + newoffset;
                        baw2mem->seek(absOffset);
                        baw2mem->write(blob, hdr.blobSizeDecompr);
                    }
                    else
                        baw2mem->seek(baseOffsetv2 + newoffset + hdr.blobSizeDecompr);

                    offsetDiff += static_cast<int32_t>(hdr.blobSizeDecompr) - static_cast<int32_t>(hdr.effectiveSize());
                    hdr.compressionType = asset::Blob::EBCT_RAW;
                    core::XXHash_256(blob, hdr.blobSizeDecompr, hdr.blobHash);
                    hdr.blobSize = hdr.blobSizeDecompr;

                    if (blob!=stackmem)
                        _IRR_ALIGNED_FREE(blob);
                }
            }
            uint64_t fileHeader[4]{ 0u, 0u, 0u, 2u/*baw v2*/ };
            memcpy(fileHeader, BAWFileVn<2>::HEADER_STRING, strlen(BAWFileVn<2>::HEADER_STRING));
            baw2mem->seek(0u);
            baw2mem->write(fileHeader, sizeof(fileHeader));
            baw2mem->write(&blobCnt, 4);
            baw2mem->write(_ctx.iv, 16);
            baw2mem->write(newoffsets.data(), newoffsets.size() * 4);
            baw2mem->write(headers, blobCnt * sizeof(headers[0])); // blob header in v1 and in v2 is exact same thing, so we can do this

            uint8_t stackmem[1u << 13]{};
            size_t newFileSz = 0u;
            for (uint32_t i = 0u; i < blobCnt; ++i)
            {
                uint32_t sz = headers[i].effectiveSize();
                void* blob = nullptr;
                if (headers[i].blobType == asset::Blob::EBT_FINAL_BONE_HIERARCHY)
                {
                    sz = 0u;
                }
                else
                {
                    _baw1file->seek(baseOffsetv1 + offsets[i]);
                    if (sz <= sizeof(stackmem))
                        blob = stackmem;
                    else
                        blob = _IRR_ALIGNED_MALLOC(sz, _IRR_SIMD_ALIGNMENT);

                    _baw1file->read(blob, sz);
                }

                baw2mem->seek(baseOffsetv2 + newoffsets[i]);
                baw2mem->write(blob, sz);

                if (headers[i].blobType != asset::Blob::EBT_DATA_FORMAT_DESC && blob != stackmem)
                    _IRR_ALIGNED_FREE(blob);

                newFileSz = baseOffsetv2 + newoffsets[i] + sz;
            }

            _IRR_ALIGNED_FREE(offsets);
            _IRR_ALIGNED_FREE(headers);

            auto ret = new io::CMemoryReadFile(baw2mem->getPointer(), baw2mem->getSize(), _baw1file->getFileName());
            baw2mem->drop();
            return ret;
        }

        template<>
        io::IReadFile* CBAWMeshFileLoader::createConvertIntoVer_spec<1>(SContext& _ctx, io::IReadFile* _baw0file, asset::IAssetLoader::IAssetLoaderOverride* _override, const CommonDataTuple<0>& _common)
        {
            uint32_t blobCnt{};
            BlobHeaderVn<0>* headers = nullptr;
            uint32_t* offsets = nullptr;
            uint32_t baseOffsetv0{};
            uint32_t baseOffsetv1{};
            std::tie(blobCnt, headers, offsets, baseOffsetv0, baseOffsetv1) = _common;

            io::CMemoryWriteFile* const baw1mem = new io::CMemoryWriteFile(0u, _baw0file->getFileName());

            std::vector<uint32_t> newoffsets(blobCnt);
            int32_t offsetDiff = 0;
            for (uint32_t i = 0u; i < blobCnt; ++i)
            {
                BlobHeaderVn<0>& hdr = headers[i];
                const uint32_t offset = offsets[i];
                uint32_t& newoffset = newoffsets[i];

                newoffset = offset + offsetDiff;

                bool adjustDiff = false;
                uint32_t prevBlobSz{};
                if (hdr.blobType == asset::Blob::EBT_DATA_FORMAT_DESC)
                {
                    uint8_t stackmem[1u << 10];
                    uint32_t attempt = 0u;
                    uint8_t decrKey[16];
                    size_t decrKeyLen = 16u;
                    void* blob = nullptr;
                    /* to state blob's/asset's hierarchy level we'd have to load (and possibly decrypt and decompress) the blob
                    however we don't need to do this here since we know format's version (baw v0) and so we can be sure that hierarchy level for mesh data descriptors is 2
                    */
                    constexpr uint32_t ICPUMESHDATAFORMATDESC_HIERARCHY_LVL = 2u;
                    while (_override->getDecryptionKey(decrKey, decrKeyLen, attempt, _baw0file, "", genSubAssetCacheKey(_baw0file->getFileName().c_str(), hdr.handle), _ctx.inner, ICPUMESHDATAFORMATDESC_HIERARCHY_LVL))
                    {
                        if (!((hdr.compressionType & asset::Blob::EBCT_AES128_GCM) && decrKeyLen != 16u))
                            blob = tryReadBlobOnStack<BlobHeaderVn<0>>(SBlobData_t<BlobHeaderVn<0>>(&hdr, baseOffsetv0 + offset), _ctx, decrKey, stackmem, sizeof(stackmem));
                        if (blob)
                            break;
                        ++attempt;
                    }
                    assert(!blob || blob == stackmem); // its a fixed size blob so should always use stack

                    const uint32_t absOffset = baseOffsetv1 + newoffset;
                    baw1mem->seek(absOffset);
                    baw1mem->write(
                        asset::MeshDataFormatDescBlobV1(reinterpret_cast<asset::legacyv0::MeshDataFormatDescBlobV0*>(blob)[0]).getData(),
                        sizeof(asset::MeshDataFormatDescBlobV1)
                    );

                    prevBlobSz = hdr.effectiveSize();
                    hdr.compressionType = asset::Blob::EBCT_RAW;
                    core::XXHash_256(reinterpret_cast<uint8_t*>(baw1mem->getPointer()) + absOffset, sizeof(asset::MeshDataFormatDescBlobV1), hdr.blobHash);
                    hdr.blobSizeDecompr = hdr.blobSize = sizeof(asset::MeshDataFormatDescBlobV1);

                    adjustDiff = true;
                }
                if (adjustDiff)
                    offsetDiff += static_cast<int32_t>(sizeof(asset::MeshDataFormatDescBlobV1)) - static_cast<int32_t>(prevBlobSz);
            }
            uint64_t fileHeader[4]{ 0u, 0u, 0u, 1u/*baw v1*/ };
            memcpy(fileHeader, BAWFileVn<1>::HEADER_STRING, strlen(BAWFileVn<1>::HEADER_STRING));
            baw1mem->seek(0u);
            baw1mem->write(fileHeader, sizeof(fileHeader));
            baw1mem->write(&blobCnt, 4);
            baw1mem->write(_ctx.iv, 16);
            baw1mem->write(newoffsets.data(), newoffsets.size() * 4);
            baw1mem->write(headers, blobCnt * sizeof(headers[0])); // blob header in v0 and in v1 is exact same thing, so we can do this

            uint8_t stackmem[1u << 13]{};
            size_t newFileSz = 0u;
            for (uint32_t i = 0u; i < blobCnt; ++i)
            {
                uint32_t sz = headers[i].effectiveSize();
                void* blob = nullptr;
                if (headers[i].blobType == asset::Blob::EBT_DATA_FORMAT_DESC)
                {
                    sz = 0u;
                }
                else
                {
                    _baw0file->seek(baseOffsetv0 + offsets[i]);
                    if (sz <= sizeof(stackmem))
                        blob = stackmem;
                    else
                        blob = _IRR_ALIGNED_MALLOC(sz, _IRR_SIMD_ALIGNMENT);

                    _baw0file->read(blob, sz);
                }

                baw1mem->seek(baseOffsetv1 + newoffsets[i]);
                baw1mem->write(blob, sz);

                if (headers[i].blobType != asset::Blob::EBT_DATA_FORMAT_DESC && blob != stackmem)
                    _IRR_ALIGNED_FREE(blob);

                newFileSz = baseOffsetv1 + newoffsets[i] + sz;
            }

            _IRR_ALIGNED_FREE(offsets);
            _IRR_ALIGNED_FREE(headers);

            auto ret = new io::CMemoryReadFile(baw1mem->getPointer(), baw1mem->getSize(), _baw0file->getFileName());
            baw1mem->drop();
            return ret;
        }

    }
}