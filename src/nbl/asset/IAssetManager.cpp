// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "nbl/asset/asset.h"

#include "nbl/asset/CGeometryCreator.h"
#include "nbl/asset/CMeshManipulator.h"

#ifdef _NBL_COMPILE_WITH_MTL_LOADER_
#include "nbl/asset/CGraphicsPipelineLoaderMTL.h"
#endif

#ifdef _NBL_COMPILE_WITH_OBJ_LOADER_
#include "nbl/asset/COBJMeshFileLoader.h"
#endif

#ifdef _NBL_COMPILE_WITH_STL_LOADER_
#include "nbl/asset/CSTLMeshFileLoader.h"
#endif

#ifdef _NBL_COMPILE_WITH_PLY_LOADER_
#include "nbl/asset/CPLYMeshFileLoader.h"
#endif

#ifdef _NBL_COMPILE_WITH_BAW_LOADER_
#include "nbl/asset/CBAWMeshFileLoader.h"
#endif

#ifdef _NBL_COMPILE_WITH_GLTF_LOADER_
#include "nbl/asset/CGLTFLoader.h"
#endif

#ifdef _NBL_COMPILE_WITH_JPG_LOADER_
#include "nbl/asset/CImageLoaderJPG.h"
#endif

#ifdef _NBL_COMPILE_WITH_PNG_LOADER_
#include "nbl/asset/CImageLoaderPNG.h"
#endif

#ifdef _NBL_COMPILE_WITH_TGA_LOADER_
#include "nbl/asset/CImageLoaderTGA.h"
#endif

#ifdef _NBL_COMPILE_WITH_OPENEXR_LOADER_
#include "nbl/asset/CImageLoaderOpenEXR.h"
#endif

#ifdef _NBL_COMPILE_WITH_GLI_LOADER_
#include "nbl/asset/CGLILoader.h"
#endif

#ifdef _NBL_COMPILE_WITH_BIN_LOADER_
#include "nbl/asset/CBufferLoaderBIN.h"
#endif

#ifdef _NBL_COMPILE_WITH_STL_WRITER_
#include "nbl/asset/CSTLMeshWriter.h"
#endif

#ifdef _NBL_COMPILE_WITH_PLY_WRITER_
#include "nbl/asset/CPLYMeshWriter.h"
#endif

#ifdef _NBL_COMPILE_WITH_BAW_WRITER_
#include "nbl/asset/CBAWMeshWriter.h"
#endif

#ifdef _NBL_COMPILE_WITH_GLTF_WRITER_
#include "nbl/asset/CGLTFWriter.h"
#endif

#ifdef _NBL_COMPILE_WITH_TGA_WRITER_
#include "nbl/asset/CImageWriterTGA.h"
#endif

#ifdef _NBL_COMPILE_WITH_JPG_WRITER_
#include "nbl/asset/CImageWriterJPG.h"
#endif

#ifdef _NBL_COMPILE_WITH_PNG_WRITER_
#include "nbl/asset/CImageWriterPNG.h"
#endif

#ifdef _NBL_COMPILE_WITH_OPENEXR_WRITER_
#include "nbl/asset/CImageWriterOpenEXR.h"
#endif

#ifdef _NBL_COMPILE_WITH_GLI_WRITER_
#include "nbl/asset/CGLIWriter.h"
#endif

#include "nbl/core/core.h"
#include "nbl/asset/CGLSLLoader.h"
#include "nbl/asset/CSPVLoader.h"


using namespace nbl;
using namespace asset;

std::function<void(SAssetBundle&)> nbl::asset::makeAssetGreetFunc(const IAssetManager* const _mgr)
{
	return [_mgr](SAssetBundle& _asset) {
		_mgr->setAssetCached(_asset, true);
		auto rng = _asset.getContents();
		//assets being in the cache must be immutable
        //asset mutability is changed just before insertion by inserting methods of IAssetManager
        //for (auto ass : rng)
		//	_mgr->setAssetMutability(ass.get(), IAsset::EM_IMMUTABLE);
	};
}
std::function<void(SAssetBundle&)> nbl::asset::makeAssetDisposeFunc(const IAssetManager* const _mgr)
{
	return [_mgr](SAssetBundle& _asset) {
		_mgr->setAssetCached(_asset, false);
		auto rng = _asset.getContents();
        for (auto ass : rng)
			_mgr->setAssetMutability(ass.get(), IAsset::EM_MUTABLE);
	};
}

void IAssetManager::initializeMeshTools()
{
	m_meshManipulator = core::make_smart_refctd_ptr<CMeshManipulator>();
    m_geometryCreator = core::make_smart_refctd_ptr<CGeometryCreator>(m_meshManipulator.get());
	m_glslCompiler = core::make_smart_refctd_ptr<IGLSLCompiler>(m_fileSystem.get());
}

const IGeometryCreator* IAssetManager::getGeometryCreator() const
{
	return m_geometryCreator.get();
}

IMeshManipulator* IAssetManager::getMeshManipulator()
{
	return m_meshManipulator.get();
}


void IAssetManager::addLoadersAndWriters()
{
#ifdef _NBL_COMPILE_WITH_STL_LOADER_
	addAssetLoader(core::make_smart_refctd_ptr<asset::CSTLMeshFileLoader>(this));
#endif
#ifdef _NBL_COMPILE_WITH_PLY_LOADER_
	addAssetLoader(core::make_smart_refctd_ptr<asset::CPLYMeshFileLoader>(this));
#endif
#ifdef _NBL_COMPILE_WITH_OBJ_LOADER_
	addAssetLoader(core::make_smart_refctd_ptr<asset::COBJMeshFileLoader>(this));
#endif
#ifdef _NBL_COMPILE_WITH_BAW_LOADER_
	addAssetLoader(core::make_smart_refctd_ptr<asset::CBAWMeshFileLoader>(this));
#endif
#ifdef _NBL_COMPILE_WITH_GLTF_LOADER_
    addAssetLoader(core::make_smart_refctd_ptr<asset::CGLTFLoader>(this));
#endif
#ifdef _NBL_COMPILE_WITH_JPG_LOADER_
	addAssetLoader(core::make_smart_refctd_ptr<asset::CImageLoaderJPG>());
#endif
#ifdef _NBL_COMPILE_WITH_PNG_LOADER_
	addAssetLoader(core::make_smart_refctd_ptr<asset::CImageLoaderPng>());
#endif
#ifdef _NBL_COMPILE_WITH_OPENEXR_LOADER_
	addAssetLoader(core::make_smart_refctd_ptr<asset::CImageLoaderOpenEXR>(this));
#endif
#ifdef  _NBL_COMPILE_WITH_GLI_LOADER_
	addAssetLoader(core::make_smart_refctd_ptr<asset::CGLILoader>());
#endif 
#ifdef _NBL_COMPILE_WITH_TGA_LOADER_
	addAssetLoader(core::make_smart_refctd_ptr<asset::CImageLoaderTGA>());
#endif
#ifdef _NBL_COMPILE_WITH_BIN_LOADER_
    addAssetLoader(core::make_smart_refctd_ptr<asset::CBufferLoaderBIN>());
#endif
	addAssetLoader(core::make_smart_refctd_ptr<asset::CGLSLLoader>());
	addAssetLoader(core::make_smart_refctd_ptr<asset::CSPVLoader>());

#ifdef _NBL_COMPILE_WITH_BAW_WRITER_
	addAssetWriter(core::make_smart_refctd_ptr<asset::CBAWMeshWriter>(getFileSystem()));
#endif
#ifdef _NBL_COMPILE_WITH_GLTF_WRITER_
    addAssetWriter(core::make_smart_refctd_ptr<asset::CGLTFWriter>());
#endif
#ifdef _NBL_COMPILE_WITH_PLY_WRITER_
	addAssetWriter(core::make_smart_refctd_ptr<asset::CPLYMeshWriter>());
#endif
#ifdef _NBL_COMPILE_WITH_STL_WRITER_
	addAssetWriter(core::make_smart_refctd_ptr<asset::CSTLMeshWriter>());
#endif
#ifdef _NBL_COMPILE_WITH_TGA_WRITER_
	addAssetWriter(core::make_smart_refctd_ptr<asset::CImageWriterTGA>());
#endif
#ifdef _NBL_COMPILE_WITH_JPG_WRITER_
	addAssetWriter(core::make_smart_refctd_ptr<asset::CImageWriterJPG>());
#endif
#ifdef _NBL_COMPILE_WITH_PNG_WRITER_
	addAssetWriter(core::make_smart_refctd_ptr<asset::CImageWriterPNG>());
#endif
#ifdef _NBL_COMPILE_WITH_MTL_LOADER_
    addAssetLoader(core::make_smart_refctd_ptr<asset::CGraphicsPipelineLoaderMTL>(this));
#endif
#ifdef _NBL_COMPILE_WITH_OPENEXR_WRITER_
	addAssetWriter(core::make_smart_refctd_ptr<asset::CImageWriterOpenEXR>());
#endif
#ifdef _NBL_COMPILE_WITH_GLI_WRITER_
	addAssetWriter(core::make_smart_refctd_ptr<asset::CGLIWriter>());
#endif
}




void IAssetManager::insertBuiltinAssets()
{
	auto addBuiltInToCaches = [&](auto&& asset, const char* path) -> void
	{
		asset::SAssetBundle bundle({ asset });
		changeAssetKey(bundle, path);
		insertBuiltinAssetIntoCache(bundle);
	};

	// materials
	{
		//
		auto buildInGLSLShader = [&](	core::smart_refctd_ptr<asset::ICPUBuffer>&& data,
									asset::ISpecializedShader::E_SHADER_STAGE type,
									std::initializer_list<const char*> paths) -> void
		{
			auto unspecializedShader = core::make_smart_refctd_ptr<asset::ICPUShader>(std::move(data),asset::ICPUShader::buffer_contains_glsl);
			auto shader = core::make_smart_refctd_ptr<asset::ICPUSpecializedShader>(std::move(unspecializedShader), asset::ISpecializedShader::SInfo({}, nullptr, "main", type));
			for (auto& path : paths)
				addBuiltInToCaches(std::move(shader), path);
		};
		auto fileSystem = getFileSystem();

		buildInGLSLShader(fileSystem->loadBuiltinData<NBL_CORE_UNIQUE_STRING_LITERAL_TYPE("nbl/builtin/specializedshaders/fullscreentriangle.vert")>(),
			asset::ISpecializedShader::ESS_VERTEX,
			{"nbl/builtin/specializedshaders/fullscreentriangle.vert"});
		buildInGLSLShader(fileSystem->loadBuiltinData<NBL_CORE_UNIQUE_STRING_LITERAL_TYPE("nbl/builtin/materials/lambertian/singletexture/specializedshader.vert")>(),
			asset::ISpecializedShader::ESS_VERTEX,
			{"nbl/builtin/materials/lambertian/singletexture/specializedshader.vert","nbl/builtin/materials/debug/vertex_uv/specializedshader.vert"});
		buildInGLSLShader(fileSystem->loadBuiltinData<NBL_CORE_UNIQUE_STRING_LITERAL_TYPE("nbl/builtin/materials/lambertian/singletexture/specializedshader.frag")>(), // it somehow adds an extra "tt" raw string to the end of the returned value, beware
			asset::ISpecializedShader::ESS_FRAGMENT, 
			{"nbl/builtin/materials/lambertian/singletexture/specializedshader.frag"});

		// TODO
		buildInGLSLShader(fileSystem->loadBuiltinData<NBL_CORE_UNIQUE_STRING_LITERAL_TYPE("nbl/builtin/materials/debug/vertex_normal/specializedshader.vert")>(),
			asset::ISpecializedShader::ESS_VERTEX,
			{"nbl/builtin/materials/debug/vertex_normal/specializedshader.vert"});
		buildInGLSLShader(fileSystem->loadBuiltinData<NBL_CORE_UNIQUE_STRING_LITERAL_TYPE("nbl/builtin/materials/debug/vertex_color/specializedshader.vert")>(),
			asset::ISpecializedShader::ESS_VERTEX,
			{"nbl/builtin/materials/debug/vertex_color/specializedshader.vert"});
		buildInGLSLShader(fileSystem->loadBuiltinData<NBL_CORE_UNIQUE_STRING_LITERAL_TYPE("nbl/builtin/materials/debug/vertex_uv/specializedshader.frag")>(),
			asset::ISpecializedShader::ESS_FRAGMENT,
			{"nbl/builtin/materials/debug/vertex_uv/specializedshader.frag"});
		buildInGLSLShader(fileSystem->loadBuiltinData<NBL_CORE_UNIQUE_STRING_LITERAL_TYPE("nbl/builtin/materials/debug/vertex_normal/specializedshader.frag")>(),
			asset::ISpecializedShader::ESS_FRAGMENT,
			{"nbl/builtin/materials/debug/vertex_normal/specializedshader.frag","nbl/builtin/materials/debug/vertex_color/specializedshader.frag"});
	}

    /*
        SBinding for UBO - basic view parameters.
    */

    asset::ICPUDescriptorSetLayout::SBinding binding1;
    binding1.count = 1u;
    binding1.binding = 0u;
    binding1.stageFlags = static_cast<asset::ICPUSpecializedShader::E_SHADER_STAGE>(asset::ICPUSpecializedShader::ESS_VERTEX | asset::ICPUSpecializedShader::ESS_FRAGMENT);
    binding1.type = asset::EDT_UNIFORM_BUFFER;

    auto ds1Layout = core::make_smart_refctd_ptr<asset::ICPUDescriptorSetLayout>(&binding1, &binding1 + 1);
    addBuiltInToCaches(ds1Layout, "nbl/builtin/materials/lambertian/singletexture/descriptorsetlayout/1");

    /*
        SBinding for the texture (sampler).
    */

    asset::ICPUDescriptorSetLayout::SBinding binding3;
    binding3.binding = 0u;
    binding3.type = EDT_COMBINED_IMAGE_SAMPLER;
    binding3.count = 1u;
    binding3.stageFlags = static_cast<asset::ICPUSpecializedShader::E_SHADER_STAGE>(asset::ICPUSpecializedShader::ESS_FRAGMENT);
    binding3.samplers = nullptr;

    auto ds3Layout = core::make_smart_refctd_ptr<asset::ICPUDescriptorSetLayout>(&binding3, &binding3 + 1);
    addBuiltInToCaches(ds3Layout, "nbl/builtin/materials/lambertian/singletexture/descriptorsetlayout/3"); // TODO find everything what has been using it so far

	constexpr uint32_t pcCount = 1u;
	asset::SPushConstantRange pcRanges[pcCount] = {asset::ISpecializedShader::ESS_VERTEX,0u,sizeof(core::matrix4SIMD)};
	auto pLayout = core::make_smart_refctd_ptr<asset::ICPUPipelineLayout>(
			pcRanges,pcRanges+pcCount,
			nullptr, core::smart_refctd_ptr(ds1Layout),nullptr, core::smart_refctd_ptr(ds3Layout)
	);
	addBuiltInToCaches(pLayout,"nbl/builtin/materials/lambertian/singletexture/pipelinelayout"); // TODO find everything what has been using it so far

	// samplers
	{
		asset::ISampler::SParams params;
		params.TextureWrapU = asset::ISampler::ETC_REPEAT;
		params.TextureWrapV = asset::ISampler::ETC_REPEAT;
		params.TextureWrapW = asset::ISampler::ETC_REPEAT;
		params.BorderColor = asset::ISampler::ETBC_FLOAT_OPAQUE_BLACK;
		params.MinFilter = asset::ISampler::ETF_LINEAR;
		params.MaxFilter = asset::ISampler::ETF_LINEAR;
		params.MipmapMode = asset::ISampler::ESMM_LINEAR;
		params.CompareEnable = false;
		params.CompareFunc = asset::ISampler::ECO_ALWAYS;
		params.AnisotropicFilter = 4u;
		params.LodBias = 0.f;
		params.MinLod = -1000.f;
		params.MaxLod = 1000.f;
		auto sampler = core::make_smart_refctd_ptr<asset::ICPUSampler>(params);
		addBuiltInToCaches(sampler, "nbl/builtin/samplers/default");

		params.TextureWrapU = params.TextureWrapV = params.TextureWrapW = asset::ISampler::ETC_CLAMP_TO_BORDER;
		sampler = core::make_smart_refctd_ptr<asset::ICPUSampler>(params);
		addBuiltInToCaches(sampler, "nbl/builtin/samplers/default_clamp_to_border");
	}


    //images
    core::smart_refctd_ptr<asset::ICPUImage> dummy2dImage;
    {
        asset::ICPUImage::SCreationParams info;
        info.format = asset::EF_R8G8B8A8_UNORM;
        info.type = asset::ICPUImage::ET_2D;
        info.extent.width = 2u;
        info.extent.height = 2u;
        info.extent.depth = 1u;
        info.mipLevels = 1u;
        info.arrayLayers = 1u;
        info.samples = asset::ICPUImage::ESCF_1_BIT;
        info.flags = static_cast<asset::IImage::E_CREATE_FLAGS>(0u);
        auto buf = core::make_smart_refctd_ptr<asset::ICPUBuffer>(info.extent.width*info.extent.height*asset::getTexelOrBlockBytesize(info.format));
        memcpy(buf->getPointer(),
            //magenta-grey 2x2 chessboard
            std::array<uint8_t, 16>{{255, 0, 255, 255, 128, 128, 128, 255, 128, 128, 128, 255, 255, 0, 255, 255}}.data(),
            buf->getSize()
        );

        dummy2dImage = asset::ICPUImage::create(std::move(info));

        auto regions = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<asset::ICPUImage::SBufferCopy>>(1u);
        asset::ICPUImage::SBufferCopy& region = regions->front();
        region.imageSubresource.mipLevel = 0u;
        region.imageSubresource.baseArrayLayer = 0u;
        region.imageSubresource.layerCount = 1u;
        region.bufferOffset = 0u;
        region.bufferRowLength = 2u;
        region.bufferImageHeight = 0u;
        region.imageOffset = {0u, 0u, 0u};
        region.imageExtent = {2u, 2u, 1u};
        dummy2dImage->setBufferAndRegions(std::move(buf), regions);
    }
    
    //image views
    {
        asset::ICPUImageView::SCreationParams info;
        info.format = dummy2dImage->getCreationParameters().format;
        info.image = dummy2dImage;
        info.viewType = asset::IImageView<asset::ICPUImage>::ET_2D;
        info.flags = static_cast<asset::ICPUImageView::E_CREATE_FLAGS>(0u);
        info.subresourceRange.baseArrayLayer = 0u;
        info.subresourceRange.layerCount = 1u;
        info.subresourceRange.baseMipLevel = 0u;
        info.subresourceRange.levelCount = 1u;
        auto dummy2dImgView = core::make_smart_refctd_ptr<asset::ICPUImageView>(std::move(info));

        addBuiltInToCaches(dummy2dImgView, "nbl/builtin/image_views/dummy2d");
        addBuiltInToCaches(dummy2dImage, "nbl/builtin/images/dummy2d");
    }

    //ds layouts
    core::smart_refctd_ptr<asset::ICPUDescriptorSetLayout> defaultDs1Layout;
    {
        asset::ICPUDescriptorSetLayout::SBinding bnd;
        bnd.count = 1u;
        bnd.binding = 0u;
        //maybe even ESS_ALL_GRAPHICS?
        bnd.stageFlags = static_cast<asset::ICPUSpecializedShader::E_SHADER_STAGE>(asset::ICPUSpecializedShader::ESS_VERTEX | asset::ICPUSpecializedShader::ESS_FRAGMENT);
        bnd.type = asset::EDT_UNIFORM_BUFFER;
        defaultDs1Layout = core::make_smart_refctd_ptr<asset::ICPUDescriptorSetLayout>(&bnd, &bnd+1);
        //it's intentionally added to cache later, see comments below, dont touch this order of insertions
    }

    //desc sets
    {
        auto ds1 = core::make_smart_refctd_ptr<asset::ICPUDescriptorSet>(core::smart_refctd_ptr<asset::ICPUDescriptorSetLayout>(defaultDs1Layout.get()));
        {
            auto desc = ds1->getDescriptors(0u).begin();
            //for filling this UBO with actual data, one can use asset::SBasicViewParameters struct defined in nbl/asset/asset_utils.h
            constexpr size_t UBO_SZ = sizeof(asset::SBasicViewParameters);
            auto ubo = core::make_smart_refctd_ptr<asset::ICPUBuffer>(UBO_SZ);
            asset::fillBufferWithDeadBeef(ubo.get());
            desc->desc = std::move(ubo);
            desc->buffer.offset = 0ull;
            desc->buffer.size = UBO_SZ;
        }
        addBuiltInToCaches(ds1, "nbl/builtin/descriptor_set/basic_view_parameters");
        addBuiltInToCaches(defaultDs1Layout, "nbl/builtin/descriptor_set_layout/basic_view_parameters");
    }

    // pipeline layout
    core::smart_refctd_ptr<asset::ICPUPipelineLayout> pipelineLayout;
    {
        asset::ICPUDescriptorSetLayout::SBinding bnd;
        bnd.count = 1u;
        bnd.binding = 0u;
        bnd.stageFlags = static_cast<asset::ICPUSpecializedShader::E_SHADER_STAGE>(asset::ICPUSpecializedShader::ESS_VERTEX | asset::ICPUSpecializedShader::ESS_FRAGMENT);
        bnd.type = asset::EDT_UNIFORM_BUFFER;
        auto ds1Layout = core::make_smart_refctd_ptr<asset::ICPUDescriptorSetLayout>(&bnd, &bnd + 1);

        pipelineLayout = core::make_smart_refctd_ptr<asset::ICPUPipelineLayout>(nullptr, nullptr, nullptr, std::move(ds1Layout), nullptr, nullptr);
        auto paths = {"nbl/builtin/materials/lambertian/no_texture/pipelinelayout",
                      "nbl/builtin/loaders/PLY/pipelinelayout"};

        for(auto &path : paths)
            addBuiltInToCaches(pipelineLayout, path);
    }
}
