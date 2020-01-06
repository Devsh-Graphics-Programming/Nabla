// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#include "irr/core/core.h"

#ifdef _IRR_COMPILE_WITH_OBJ_LOADER_

#include "IFileSystem.h"
#include "COBJMeshFileLoader.h"
#include "irr/asset/IMeshManipulator.h"
#include "IVideoDriver.h"
#include "irr/video/CGPUMesh.h"
#include "irr/asset/normal_quantization.h"
#include "IReadFile.h"
#include "os.h"
#include "irr/asset/IAssetManager.h"

/*
namespace std
{
    template <>
    struct hash<irr::asset::SObjVertex>
    {
        std::size_t operator()(const irr::asset::SObjVertex& k) const
        {
            using std::size_t;
            using std::hash;

            return hash(k.normal32bit)^
                    (reinterpret_cast<const uint32_t&>(k.pos[0])*4996156539000000107ull)^
                    (reinterpret_cast<const uint32_t&>(k.pos[1])*620612627000000023ull)^
                    (reinterpret_cast<const uint32_t&>(k.pos[2])*1231379668000000199ull)^
                    (reinterpret_cast<const uint32_t&>(k.uv[0])*1099543332000000001ull)^
                    (reinterpret_cast<const uint32_t&>(k.uv[1])*1123461104000000009ull);
        }
    };

}
*/

#define NEW_SHADERS

namespace irr
{
namespace asset
{

//#ifdef _IRR_DEBUG
#define _IRR_DEBUG_OBJ_LOADER_
//#endif

static const uint32_t WORD_BUFFER_LENGTH = 512;


//! Constructor
COBJMeshFileLoader::COBJMeshFileLoader(IAssetManager* _manager) : AssetManager(_manager), FileSystem(_manager->getFileSystem())
{
#ifdef _IRR_DEBUG
	setDebugName("COBJMeshFileLoader");
#endif
}


//! destructor
COBJMeshFileLoader::~COBJMeshFileLoader()
{
}

asset::SAssetBundle COBJMeshFileLoader::loadAsset(io::IReadFile* _file, const asset::IAssetLoader::SAssetLoadParams& _params, asset::IAssetLoader::IAssetLoaderOverride* _override, uint32_t _hierarchyLevel)
{
    SContext ctx(
        asset::IAssetLoader::SAssetLoadContext{
            _params,
            _file
        },
		_hierarchyLevel,
        _override
    );

	const long filesize = _file->getSize();
	if (!filesize)
        return {};

	const uint32_t WORD_BUFFER_LENGTH = 512u;
    char tmpbuf[WORD_BUFFER_LENGTH]{};

	uint32_t smoothingGroup=0;

	const io::path fullName = _file->getFileName();
	const std::string relPath = (io::IFileSystem::getFileDir(fullName)+"/").c_str();

    //value_type: directory from which .mtl (pipeline) was loaded and the pipeline
    core::unordered_map<std::string, std::pair<std::string, core::smart_refctd_ptr<ICPURenderpassIndependentPipeline>>> pipelines;

    std::string fileContents;
    fileContents.resize(filesize);
	char* const buf = fileContents.data();
	_file->read(buf, filesize);
	const char* const bufEnd = buf+filesize;

	// Process obj information
	const char* bufPtr = buf;
	std::string grpName, mtlName;

	auto performActionBasedOnOrientationSystem = [&](auto performOnRightHanded, auto performOnLeftHanded)
	{
		if (_params.loaderFlags & E_LOADER_PARAMETER_FLAGS::ELPF_RIGHT_HANDED_MESHES)
			performOnRightHanded();
		else
			performOnLeftHanded();
	};


    struct vec3 {
        float data[3];
    };
    struct vec2 {
        float data[2];
    };
    core::vector<vec3> vertexBuffer;
    core::vector<vec3> normalsBuffer;
    core::vector<vec2> textureCoordBuffer;

    core::vector<core::smart_refctd_ptr<ICPUMeshBuffer>> submeshes;
    core::vector<core::vector<uint32_t>> indices;
    core::vector<SObjVertex> vertices;
    core::map<SObjVertex, uint32_t> map_vtx2ix;
    core::vector<bool> recalcNormals;
    core::vector<bool> submeshWasLoadedFromCache;
	while(bufPtr != bufEnd)
	{
		switch(bufPtr[0])
		{
		case 'm':	// mtllib (material)
		{
			if (ctx.useMaterials)
			{
				bufPtr = goAndCopyNextWord(tmpbuf, bufPtr, WORD_BUFFER_LENGTH, bufEnd);
#ifdef _IRR_DEBUG_OBJ_LOADER_
				os::Printer::log("Reading material _file",tmpbuf);
#endif

                SAssetLoadParams loadParams;
                auto bundle = interm_getAssetInHierarchy(AssetManager, tmpbuf, loadParams, _hierarchyLevel+ICPUMesh::PIPELINE_HIERARCHYLEVELS_BELOW);
                for (auto it = bundle.getContents().first; it != bundle.getContents().second; ++it)
                {
                    auto pipeln = core::smart_refctd_ptr_static_cast<ICPURenderpassIndependentPipeline>(*it);
                    auto metadata = static_cast<const CMTLPipelineMetadata*>(pipeln->getMetadata());
                    std::string mtlfilepath = tmpbuf;

                    decltype(pipelines)::value_type::second_type val{std::move(mtlfilepath), std::move(pipeln)};
                    pipelines.insert({metadata->getMaterial().name, std::move(val)});
                }
			}
		}
			break;

		case 'v':               // v, vn, vt
			switch(bufPtr[1])
			{
			case ' ':          // vertex
				{
					vec3 vec;
					bufPtr = readVec3(bufPtr, vec.data, bufEnd);
					performActionBasedOnOrientationSystem([&]() {vec.data[0] = -vec.data[0];}, [&]() {});
					vertexBuffer.push_back(vec);
				}
				break;

			case 'n':       // normal
				{
					vec3 vec;
					bufPtr = readVec3(bufPtr, vec.data, bufEnd);
					performActionBasedOnOrientationSystem([&]() {vec.data[0] = -vec.data[0]; }, [&]() {});
					normalsBuffer.push_back(vec);
				}
				break;

			case 't':       // texcoord
				{
					vec2 vec;
					bufPtr = readUV(bufPtr, vec.data, bufEnd);
					textureCoordBuffer.push_back(vec);
				}
				break;
			}
			break;

		case 'g': // group name
            bufPtr = goNextWord(bufPtr, bufEnd);
			break;
		case 's': // smoothing can be a group or off (equiv. to 0)
			{
				bufPtr = goAndCopyNextWord(tmpbuf, bufPtr, WORD_BUFFER_LENGTH, bufEnd);
#ifdef _IRR_DEBUG_OBJ_LOADER_
	os::Printer::log("Loaded smoothing group start",tmpbuf, ELL_DEBUG);
#endif
				if (core::stringc("off")==tmpbuf)
					smoothingGroup=0u;
				else
                    sscanf(tmpbuf,"%u",&smoothingGroup);
			}
			break;

		case 'u': // usemtl
			// get name of material
			{
				bufPtr = goAndCopyNextWord(tmpbuf, bufPtr, WORD_BUFFER_LENGTH, bufEnd);
#ifdef _IRR_DEBUG_OBJ_LOADER_
	os::Printer::log("Loaded material start",tmpbuf, ELL_DEBUG);
#endif
				mtlName=tmpbuf;

                if (ctx.useMaterials && !ctx.useGroups)
                {
                    asset::IAsset::E_TYPE types[] {asset::IAsset::ET_SUB_MESH, (asset::IAsset::E_TYPE)0u };
                    auto mb_bundle = _override->findCachedAsset(genKeyForMeshBuf(ctx, _file->getFileName().c_str(), mtlName, grpName), types, ctx.inner, _hierarchyLevel+ICPUMesh::MESHBUFFER_HIERARCHYLEVELS_BELOW);
                    auto mbs = mb_bundle.getContents();
                    if (mbs.first!= mbs.second)
                    {
                        submeshes.push_back(core::smart_refctd_ptr_static_cast<ICPUMeshBuffer>(*mbs.first));
                    }
                    else
                    {
                        submeshes.push_back(core::make_smart_refctd_ptr<ICPUMeshBuffer>());
                        auto found = pipelines.find(mtlName);
                        assert(found != pipelines.end());
#ifndef _IRR_DEBUG
                        if (found != pipelines.end())
                        {
#endif
                            auto& pipeln = found->second;
                            //cloning pipeline because it will be edited (vertex input params)
                            //note shallow copy (depth=0), i.e. only pipeline is cloned, but all its sub-assets are taken from original object
                            submeshes.back()->setPipeline(core::smart_refctd_ptr_static_cast<ICPURenderpassIndependentPipeline>(pipeln.second->clone(0u)));
                            auto metadata = static_cast<const CMTLPipelineMetadata*>(pipeln.second->getMetadata());
                            memcpy(
                                submeshes.back()->getPushConstantsDataPtr()+pipeln.second->getLayout()->getPushConstantRanges().begin()[0].offset,
                                &metadata->getMaterial().std140PackedData,
                                sizeof(CMTLPipelineMetadata::SMtl::std140PackedData)
                            );
                            //also make/find descriptor set
                            std::string dsCacheKey = pipeln.first + "?" + mtlName + "?_ds";
                            auto ds_bundle = _override->findCachedAsset(dsCacheKey, types, ctx.inner, _hierarchyLevel+ICPUMesh::DESC_SET_HIERARCHYLEVELS_BELOW);
                            auto ds_bundle_contents = ds_bundle.getContents();
                            core::smart_refctd_ptr<ICPUDescriptorSet> ds3;
                            if (ds_bundle_contents.first != ds_bundle_contents.second)
                            {
                                ds3 = core::smart_refctd_ptr_static_cast<ICPUDescriptorSet>(ds_bundle_contents.first[0]);
                            }
                            else
                            {
                                auto relDir = (io::IFileSystem::getFileDir(pipeln.first.c_str()) + "/");
                                images_set_t images = loadImages(relDir.c_str(), metadata->getMaterial(), _hierarchyLevel + ICPUMesh::IMAGE_HIERARCHYLEVELS_BELOW);
                                ds3 = makeDescSet(images, pipeln.second->getLayout()->getDescriptorSetLayout(3u));

                                SAssetBundle bundle{ds3};
                                _override->insertAssetIntoCache(bundle, dsCacheKey, ctx.inner, _hierarchyLevel+ICPUMesh::DESC_SET_HIERARCHYLEVELS_BELOW);
                            }
                            submeshes.back()->setAttachedDescriptorSet(std::move(ds3));
#ifndef _IRR_DEBUG
                        }
#endif
                        SAssetBundle bundle{submeshes.back()};
                        _override->insertAssetIntoCache(bundle, genKeyForMeshBuf(ctx, _file->getFileName().c_str(), mtlName, grpName), ctx.inner, _hierarchyLevel+ICPUMesh::MESHBUFFER_HIERARCHYLEVELS_BELOW);
                        interm_setAssetMutable(AssetManager, submeshes.back().get(), true); //insertion into cache makes asset immutable, so temporarily make it mutable because it has to be adjusted more
                    }
                    indices.emplace_back();
                    recalcNormals.push_back(false);
                    submeshWasLoadedFromCache.push_back(mbs.first!=mbs.second);
                }
			}
			break;

		case 'f':               // face
		{
			SObjVertex v;

			// get all vertices data in this face (current line of obj _file)
			const core::stringc wordBuffer = copyLine(bufPtr, bufEnd);
			const char* linePtr = wordBuffer.c_str();
			const char* const endPtr = linePtr+wordBuffer.size();

			core::vector<uint32_t> faceCorners;
			faceCorners.reserve(3ull);

			// read in all vertices
			linePtr = goNextWord(linePtr, endPtr);
			while (0 != linePtr[0])
			{
				// Array to communicate with retrieveVertexIndices()
				// sends the buffer sizes and gets the actual indices
				// if index not set returns -1
				int32_t Idx[3];
				Idx[1] = Idx[2] = -1;

				// read in next vertex's data
				uint32_t wlength = copyWord(tmpbuf, linePtr, WORD_BUFFER_LENGTH, endPtr);
				// this function will also convert obj's 1-based index to c++'s 0-based index
				retrieveVertexIndices(tmpbuf, Idx, tmpbuf+wlength+1, vertexBuffer.size(), textureCoordBuffer.size(), normalsBuffer.size());
				v.pos[0] = vertexBuffer[Idx[0]].data[0];
				v.pos[1] = vertexBuffer[Idx[0]].data[1];
				v.pos[2] = vertexBuffer[Idx[0]].data[2];
				//set texcoord
				if ( -1 != Idx[1] )
                {
					v.uv[0] = textureCoordBuffer[Idx[1]].data[0];
					v.uv[1] = textureCoordBuffer[Idx[1]].data[1];
                }
				else
                {
					v.uv[0] = std::numeric_limits<float>::quiet_NaN();
					v.uv[1] = std::numeric_limits<float>::quiet_NaN();
                }
                //set normal
				if ( -1 != Idx[2] )
                {
					core::vectorSIMDf simdNormal;
					simdNormal.set(normalsBuffer[Idx[2]].data);
                    simdNormal.makeSafe3D();
					v.normal32bit = asset::quantizeNormal2_10_10_10(simdNormal);
                }
				else
				{
					v.normal32bit = 0;
                    recalcNormals.back() = true;
				}

				uint32_t ix;
				auto vtx_ix = map_vtx2ix.find(v);
				if (vtx_ix != map_vtx2ix.end())
					ix = vtx_ix->second;
				else
				{
					ix = vertices.size();
					vertices.push_back(v);
					map_vtx2ix.insert({v, ix});
				}

				faceCorners.push_back(ix);

				// go to next vertex
				linePtr = goNextWord(linePtr, endPtr);
			}

            assert(faceCorners.size()==3ull);
			performActionBasedOnOrientationSystem
			(
				[](){},
				[&]() 
				{
                    std::swap(faceCorners.front(), faceCorners.back());
				}
			);
            indices.back().insert(indices.back().end(), faceCorners.begin(), faceCorners.end());
		}
		break;

		case '#': // comment
		default:
			break;
		}	// end switch(bufPtr[0])
		// eat up rest of line
		bufPtr = goNextLine(bufPtr, bufEnd);
	}	// end while(bufPtr && (bufPtr-buf<filesize))

    core::vector<core::vectorSIMDf> newNormals;
    auto doRecalcNormals = [&vertices,&newNormals](const core::vector<uint32_t>& _ixs) {
        memset(newNormals.data(), 0, sizeof(core::vectorSIMDf)*newNormals.size());

        auto minmax = std::minmax_element(_ixs.begin(), _ixs.end());
        const uint32_t maxsz = *minmax.second - *minmax.first;
        const uint32_t min = *minmax.first;
        
        newNormals.resize(maxsz, core::vectorSIMDf(0.f));
        for (size_t i = 0ull; i < _ixs.size(); i += 3ull)
        {
            core::vectorSIMDf v1, v2, v3;
            v1.set(vertices[_ixs[i-min+0u]].pos);
            v2.set(vertices[_ixs[i-min+1u]].pos);
            v3.set(vertices[_ixs[i-min+2u]].pos);
            v1.makeSafe3D();
            v2.makeSafe3D();
            v3.makeSafe3D();
            core::vectorSIMDf normal(core::plane3dSIMDf(v1, v2, v3).getNormal());
            newNormals[_ixs[i-min+0u]] += normal;
            newNormals[_ixs[i-min+1u]] += normal;
            newNormals[_ixs[i-min+2u]] += normal;
        }
        for (uint32_t ix : _ixs)
            vertices[ix].normal32bit = asset::quantizeNormal2_10_10_10(newNormals[ix-min]);
    };

    constexpr uint32_t POSITION = 0u;
    constexpr uint32_t UV       = 2u;
    constexpr uint32_t NORMAL   = 3u;
    constexpr uint32_t BND_NUM  = 0u;
    {
        uint64_t ixBufOffset = 0ull;
        for (size_t i = 0ull; i < submeshes.size(); ++i)
        {
            if (submeshWasLoadedFromCache[i])
                continue;
            if (recalcNormals[i])
                doRecalcNormals(indices[i]);

            submeshes[i]->setIndexCount(indices[i].size());
            submeshes[i]->setIndexType(EIT_32BIT);
            submeshes[i]->getIndexBufferBinding()->offset = ixBufOffset;
            ixBufOffset += indices[i].size()*4ull;

            const bool hasUV = !core::isnan(vertices[indices[i][0]].uv[0]);
            SVertexInputParams vtxParams;
            vtxParams.enabledAttribFlags = (1u<<POSITION) | (1u<<NORMAL) | (hasUV ? (1u<<UV) : 0u);
            vtxParams.enabledBindingFlags = 1u<<BND_NUM;
            vtxParams.bindings[BND_NUM].stride = sizeof(SObjVertex);
            vtxParams.bindings[BND_NUM].inputRate = EVIR_PER_VERTEX;
            //position
            vtxParams.attributes[POSITION].binding = BND_NUM;
            vtxParams.attributes[POSITION].format = EF_R32G32B32_SFLOAT;
            vtxParams.attributes[POSITION].relativeOffset = offsetof(SObjVertex, pos);
            //normal
            vtxParams.attributes[NORMAL].binding = BND_NUM;
            vtxParams.attributes[NORMAL].format = EF_A2B10G10R10_SNORM_PACK32;
            vtxParams.attributes[NORMAL].relativeOffset = offsetof(SObjVertex, normal32bit);
            //uv
            if (hasUV)
            {
                vtxParams.attributes[UV].binding = BND_NUM;
                vtxParams.attributes[UV].format = EF_R32G32_SFLOAT;
                vtxParams.attributes[UV].relativeOffset = offsetof(SObjVertex, uv);
            }
            submeshes[i]->getPipeline()->getVertexInputParams() = vtxParams;
        }

        core::smart_refctd_ptr<ICPUBuffer> vtxBuf = core::make_smart_refctd_ptr<ICPUBuffer>(vertices.size() * sizeof(SObjVertex));
        memcpy(vtxBuf->getPointer(), vertices.data(), vtxBuf->getSize());

        auto ixBuf = core::make_smart_refctd_ptr<ICPUBuffer>(ixBufOffset);
        for (size_t i = 0ull; i < submeshes.size(); ++i)
        {
            if (submeshWasLoadedFromCache[i])
                continue;

            submeshes[i]->getIndexBufferBinding()->buffer = ixBuf;
            uint64_t offset = submeshes[i]->getIndexBufferBinding()->offset;
            memcpy(reinterpret_cast<uint8_t*>(ixBuf->getPointer())+offset, indices[i].data(), indices[i].size()*4ull);

            SBufferBinding<ICPUBuffer> vtxBufBnd;
            vtxBufBnd.offset = 0ull;
            vtxBufBnd.buffer = vtxBuf;

            submeshes[i]->getVertexBufferBindings()[BND_NUM] = vtxBufBnd;
        }
    }

    auto mesh = core::make_smart_refctd_ptr<CCPUMesh>();
    for (auto& submesh : submeshes)
        mesh->addMeshBuffer(std::move(submesh));

	if (mesh->getMeshBufferCount())
		mesh->recalculateBoundingBox(true);
	else
        return {};
    
    //at the very end, when meshbuffers are finished, set them back as immutable
    for (uint32_t i = 0u; i < mesh->getMeshBufferCount(); ++i)
        interm_setAssetMutable(AssetManager, mesh->getMeshBuffer(i), false);

	return SAssetBundle({std::move(mesh)});
}

auto COBJMeshFileLoader::loadImages(const char* _relDir, const CMTLPipelineMetadata::SMtl& _mtl, uint32_t _hierarchyLvl) -> images_set_t
{
    std::array<core::smart_refctd_ptr<ICPUImage>, CMTLPipelineMetadata::SMtl::EMP_COUNT> images;

    std::string relDir = _relDir;
    relDir += '/';
    for (uint32_t i = 0u; i < images.size(); ++i)
    {
        SAssetLoadParams lp;
        if (_mtl.maps[i].size())
        {
            auto bundle = interm_getAssetInHierarchy(AssetManager, relDir + _mtl.maps[i], lp, _hierarchyLvl);
            if (!bundle.isEmpty())
                images[i] = core::smart_refctd_ptr_static_cast<ICPUImage>(bundle.getContents().first[0]);
        }
    }

    auto allCubemapFacesAreSameSizeAndFormat = [](const core::smart_refctd_ptr<ICPUImage>* _faces) {
        const VkExtent3D sz = (*_faces)->getCreationParameters().extent;
        const E_FORMAT fmt = (*_faces)->getCreationParameters().format;
        for (uint32_t i = 1u; i < 6u; ++i)
        {
            const auto& img = _faces[i];
            if (!img)
                continue;

            if (img->getCreationParameters().format != fmt)
                return false;
            const VkExtent3D sz_ = img->getCreationParameters().extent;
            if (sz.width != sz_.width || sz.height != sz_.height || sz.depth != sz_.depth)
                return false;
        }
        return true;
    };
    //make reflection cubemap
    if (images[CMTLPipelineMetadata::SMtl::EMP_REFL_POSX])
    {
        assert(allCubemapFacesAreSameSizeAndFormat(images.data() + CMTLPipelineMetadata::SMtl::EMP_REFL_POSX));

        size_t bufSz = 0ull;
        //assuming all cubemap layer images are same size and same format
        const size_t alignment = core::findLSB(images[CMTLPipelineMetadata::SMtl::EMP_REFL_POSX]->getRegions().begin()->bufferRowLength);
        core::vector<ICPUImage::SBufferCopy> regions_;
        regions_.reserve(6ull);
        for (uint32_t i = CMTLPipelineMetadata::SMtl::EMP_REFL_POSX; i < CMTLPipelineMetadata::SMtl::EMP_REFL_POSX + 6u; ++i)
        {
            assert(images[i]);
#ifndef _IRR_DEBUG
            if (images[i])
            {
#endif
                //assuming each image has just 1 region
                assert(images[i]->getRegions().length()==1ull);

                regions_.push_back(images[i]->getRegions().begin()[0]);
                regions_.back().bufferOffset = core::roundUp(regions_.back().bufferOffset, alignment);
                regions_.back().imageSubresource.baseArrayLayer = (i - CMTLPipelineMetadata::SMtl::EMP_REFL_POSX);

                bufSz += images[i]->getImageDataSizeInBytes();
#ifndef _IRR_DEBUG
            }
#endif
        }
        auto imgDataBuf = core::make_smart_refctd_ptr<ICPUBuffer>(bufSz);
        for (uint32_t i = CMTLPipelineMetadata::SMtl::EMP_REFL_POSX, j = 0u; i < CMTLPipelineMetadata::SMtl::EMP_REFL_POSX + 6u; ++i)
        {
#ifndef _IRR_DEBUG
            if (images[i])
            {
#endif
                void* dst = reinterpret_cast<uint8_t*>(imgDataBuf->getPointer()) + regions_[j].bufferOffset;
                const void* src = reinterpret_cast<const uint8_t*>(images[i]->getBuffer()->getPointer()) + images[i]->getRegions().begin()[0].bufferOffset;
                const size_t sz = images[i]->getImageDataSizeInBytes();
                memcpy(dst, src, sz);

                ++j;
#ifndef _IRR_DEBUG
            }
#endif
        }

        //assuming all cubemap layer images are same size and same format
        ICPUImage::SCreationParams cubemapParams = images[CMTLPipelineMetadata::SMtl::EMP_REFL_POSX]->getCreationParameters();
        cubemapParams.arrayLayers = 6u;
        cubemapParams.type = IImage::ET_2D;

        auto cubemap = ICPUImage::create(std::move(cubemapParams));
        auto regions = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<ICPUImage::SBufferCopy>>(regions_);
        cubemap->setBufferAndRegions(std::move(imgDataBuf), regions);
        //new image goes to EMP_REFL_POSX index and other ones get nulled-out
        images[CMTLPipelineMetadata::SMtl::EMP_REFL_POSX] = std::move(cubemap);
        for (uint32_t i = CMTLPipelineMetadata::SMtl::EMP_REFL_POSX + 1u; i < CMTLPipelineMetadata::SMtl::EMP_REFL_POSX + 6u; ++i)
        {
            images[i] = nullptr;
        }
    }

    return images;
}

core::smart_refctd_ptr<ICPUDescriptorSet> COBJMeshFileLoader::makeDescSet(const images_set_t& _images, ICPUDescriptorSetLayout* _dsLayout)
{
    auto ds = core::make_smart_refctd_ptr<asset::ICPUDescriptorSet>(
        core::smart_refctd_ptr<ICPUDescriptorSetLayout>(_dsLayout)
    );
    //d starts from 1u because binding 0 is UBO
    for (uint32_t i = 0u, d = 1u; i <= CMTLPipelineMetadata::SMtl::EMP_REFL_POSX; ++i)
    {
        if (!_images[i])
            continue;

        constexpr IImageView<ICPUImage>::E_TYPE viewType[2]{ IImageView<ICPUImage>::ET_2D, IImageView<ICPUImage>::ET_CUBE_MAP };
        constexpr uint32_t layerCount[2]{ 1u, 6u };

        const bool isCubemap = (i == CMTLPipelineMetadata::SMtl::EMP_REFL_POSX);

        ICPUImageView::SCreationParams viewParams;
        viewParams.flags = static_cast<ICPUImageView::E_CREATE_FLAGS>(0u);
        viewParams.format = _images[i]->getCreationParameters().format;
        viewParams.image = _images[i];
        viewParams.viewType = viewType[isCubemap];
        viewParams.subresourceRange.baseArrayLayer = 0u;
        viewParams.subresourceRange.layerCount = layerCount[isCubemap];
        viewParams.subresourceRange.baseMipLevel = 0u;
        viewParams.subresourceRange.levelCount = 1u;

        auto desc = ds->getDescriptors(d).begin();
        desc->desc = core::make_smart_refctd_ptr<ICPUImageView>(std::move(viewParams));
        desc->image.imageLayout = EIL_UNDEFINED;
        desc->image.sampler = nullptr; //not needed, MTL loader puts immutable samplers into layout
        ++d;
    }
    auto desc = ds->getDescriptors(0u).begin();
    desc->desc = core::make_smart_refctd_ptr<ICPUBuffer>(16u*4u);//size of 4x4 matrix (MVP)
    desc->buffer.offset = 0ull;
    desc->buffer.size = 16u*4u;

    return ds;
}

//! Read RGB color
const char* COBJMeshFileLoader::readColor(const char* bufPtr, video::SColor& color, const char* const bufEnd)
{
	const uint32_t COLOR_BUFFER_LENGTH = 16;
	char colStr[COLOR_BUFFER_LENGTH];

	float tmp;

	color.setAlpha(255);
	bufPtr = goAndCopyNextWord(colStr, bufPtr, COLOR_BUFFER_LENGTH, bufEnd);
	sscanf(colStr,"%f",&tmp);
	color.setRed((int32_t)(tmp * 255.0f));
	bufPtr = goAndCopyNextWord(colStr,   bufPtr, COLOR_BUFFER_LENGTH, bufEnd);
	sscanf(colStr,"%f",&tmp);
	color.setGreen((int32_t)(tmp * 255.0f));
	bufPtr = goAndCopyNextWord(colStr,   bufPtr, COLOR_BUFFER_LENGTH, bufEnd);
	sscanf(colStr,"%f",&tmp);
	color.setBlue((int32_t)(tmp * 255.0f));
	return bufPtr;
}


//! Read 3d vector of floats
const char* COBJMeshFileLoader::readVec3(const char* bufPtr, float vec[3], const char* const bufEnd)
{
	const uint32_t WORD_BUFFER_LENGTH = 256;
	char wordBuffer[WORD_BUFFER_LENGTH];

	bufPtr = goAndCopyNextWord(wordBuffer, bufPtr, WORD_BUFFER_LENGTH, bufEnd);
	sscanf(wordBuffer,"%f",vec);
	bufPtr = goAndCopyNextWord(wordBuffer, bufPtr, WORD_BUFFER_LENGTH, bufEnd);
	sscanf(wordBuffer,"%f",vec+1);
	bufPtr = goAndCopyNextWord(wordBuffer, bufPtr, WORD_BUFFER_LENGTH, bufEnd);
	sscanf(wordBuffer,"%f",vec+2);

    vec[0] = -vec[0]; // change handedness
	return bufPtr;
}


//! Read 2d vector of floats
const char* COBJMeshFileLoader::readUV(const char* bufPtr, float vec[2], const char* const bufEnd)
{
	const uint32_t WORD_BUFFER_LENGTH = 256;
	char wordBuffer[WORD_BUFFER_LENGTH];

	bufPtr = goAndCopyNextWord(wordBuffer, bufPtr, WORD_BUFFER_LENGTH, bufEnd);
	sscanf(wordBuffer,"%f",vec);
	bufPtr = goAndCopyNextWord(wordBuffer, bufPtr, WORD_BUFFER_LENGTH, bufEnd);
	sscanf(wordBuffer,"%f",vec+1);

	vec[1] = 1.f-vec[1]; // change handedness
	return bufPtr;
}


//! Read boolean value represented as 'on' or 'off'
const char* COBJMeshFileLoader::readBool(const char* bufPtr, bool& tf, const char* const bufEnd)
{
	const uint32_t BUFFER_LENGTH = 8;
	char tfStr[BUFFER_LENGTH];

	bufPtr = goAndCopyNextWord(tfStr, bufPtr, BUFFER_LENGTH, bufEnd);
	tf = strcmp(tfStr, "off") != 0;
	return bufPtr;
}

//! skip space characters and stop on first non-space
const char* COBJMeshFileLoader::goFirstWord(const char* buf, const char* const bufEnd, bool acrossNewlines)
{
	// skip space characters
	if (acrossNewlines)
		while((buf != bufEnd) && core::isspace(*buf))
			++buf;
	else
		while((buf != bufEnd) && core::isspace(*buf) && (*buf != '\n'))
			++buf;

	return buf;
}


//! skip current word and stop at beginning of next one
const char* COBJMeshFileLoader::goNextWord(const char* buf, const char* const bufEnd, bool acrossNewlines)
{
	// skip current word
	while(( buf != bufEnd ) && !core::isspace(*buf))
		++buf;

	return goFirstWord(buf, bufEnd, acrossNewlines);
}


//! Read until line break is reached and stop at the next non-space character
const char* COBJMeshFileLoader::goNextLine(const char* buf, const char* const bufEnd)
{
	// look for newline characters
	while(buf != bufEnd)
	{
		// found it, so leave
		if (*buf=='\n' || *buf=='\r')
			break;
		++buf;
	}
	return goFirstWord(buf, bufEnd);
}


uint32_t COBJMeshFileLoader::copyWord(char* outBuf, const char* const inBuf, uint32_t outBufLength, const char* const bufEnd)
{
	if (!outBufLength)
		return 0;
	if (!inBuf)
	{
		*outBuf = 0;
		return 0;
	}

	uint32_t i = 0;
	while(inBuf[i])
	{
		if (core::isspace(inBuf[i]) || &(inBuf[i]) == bufEnd)
			break;
		++i;
	}

	uint32_t length = core::min(i, outBufLength-1);
	for (uint32_t j=0; j<length; ++j)
		outBuf[j] = inBuf[j];

	outBuf[length] = 0;
	return length;
}


core::stringc COBJMeshFileLoader::copyLine(const char* inBuf, const char* bufEnd)
{
	if (!inBuf)
		return core::stringc();

	const char* ptr = inBuf;
	while (ptr<bufEnd)
	{
		if (*ptr=='\n' || *ptr=='\r')
			break;
		++ptr;
	}
	// we must avoid the +1 in case the array is used up
	return core::stringc(inBuf, (uint32_t)(ptr-inBuf+((ptr < bufEnd) ? 1 : 0)));
}


const char* COBJMeshFileLoader::goAndCopyNextWord(char* outBuf, const char* inBuf, uint32_t outBufLength, const char* bufEnd)
{
	inBuf = goNextWord(inBuf, bufEnd, false);
	copyWord(outBuf, inBuf, outBufLength, bufEnd);
	return inBuf;
}


bool COBJMeshFileLoader::retrieveVertexIndices(char* vertexData, int32_t* idx, const char* bufEnd, uint32_t vbsize, uint32_t vtsize, uint32_t vnsize)
{
	char word[16] = "";
	const char* p = goFirstWord(vertexData, bufEnd);
	uint32_t idxType = 0;	// 0 = posIdx, 1 = texcoordIdx, 2 = normalIdx

	uint32_t i = 0;
	while ( p != bufEnd )
	{
		if ( ( core::isdigit(*p)) || (*p == '-') )
		{
			// build up the number
			word[i++] = *p;
		}
		else if ( *p == '/' || *p == ' ' || *p == '\0' )
		{
			// number is completed. Convert and store it
			word[i] = '\0';
			// if no number was found index will become 0 and later on -1 by decrement
			sscanf(word,"%d",idx+idxType);
			if (idx[idxType]<0)
			{
				switch (idxType)
				{
					case 0:
						idx[idxType] += vbsize;
						break;
					case 1:
						idx[idxType] += vtsize;
						break;
					case 2:
						idx[idxType] += vnsize;
						break;
				}
			}
			else
				idx[idxType]-=1;

			// reset the word
			word[0] = '\0';
			i = 0;

			// go to the next kind of index type
			if (*p == '/')
			{
				if ( ++idxType > 2 )
				{
					// error checking, shouldn't reach here unless file is wrong
					idxType = 0;
				}
			}
			else
			{
				// set all missing values to disable (=-1)
				while (++idxType < 3)
					idx[idxType]=-1;
				++p;
				break; // while
			}
		}

		// go to the next char
		++p;
	}

	return true;
}

std::string COBJMeshFileLoader::genKeyForMeshBuf(const SContext & _ctx, const std::string & _baseKey, const std::string & _mtlName, const std::string & _grpName) const
{
    if (_ctx.useMaterials)
    {   
        if (_ctx.useGroups)
            return _baseKey + "?" + _grpName + "?" + _mtlName;
        return _baseKey + "?" +  _mtlName;
    }
    return ""; // if nothing's broken this will never happen
}




} // end namespace scene
} // end namespace irr

#endif // _IRR_COMPILE_WITH_OBJ_LOADER_
