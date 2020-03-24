#include "irr/asset/asset.h"

#ifdef _IRR_COMPILE_WITH_MTL_LOADER_
#include "irr/asset/CGraphicsPipelineLoaderMTL.h"
#endif

#ifdef _IRR_COMPILE_WITH_OBJ_LOADER_
#include "irr/asset/COBJMeshFileLoader.h"
#endif

#ifdef _IRR_COMPILE_WITH_STL_LOADER_
#include "irr/asset/CSTLMeshFileLoader.h"
#endif

#ifdef _IRR_COMPILE_WITH_PLY_LOADER_
#include "irr/asset/CPLYMeshFileLoader.h"
#endif

#ifdef _IRR_COMPILE_WITH_BAW_LOADER_
#include "irr/asset/CBAWMeshFileLoader.h"
#endif

#ifdef _IRR_COMPILE_WITH_JPG_LOADER_
#include "irr/asset/CImageLoaderJPG.h"
#endif

#ifdef _IRR_COMPILE_WITH_PNG_LOADER_
#include "irr/asset/CImageLoaderPNG.h"
#endif

#ifdef _IRR_COMPILE_WITH_TGA_LOADER_
#include "irr/asset/CImageLoaderTGA.h"
#endif

#ifdef _IRR_COMPILE_WITH_OPENEXR_LOADER_
#include "irr/asset/CImageLoaderOpenEXR.h"
#endif

#ifdef _IRR_COMPILE_WITH_GLI_LOADER_
#include "irr/asset/CGLILoader.h"
#endif

#ifdef _IRR_COMPILE_WITH_STL_WRITER_
#include "irr/asset/CSTLMeshWriter.h"
#endif

#ifdef _IRR_COMPILE_WITH_PLY_WRITER_
#include "irr/asset/CPLYMeshWriter.h"
#endif

#ifdef _IRR_COMPILE_WITH_BAW_WRITER_
#include"irr/asset/CBAWMeshWriter.h"
#endif

#ifdef _IRR_COMPILE_WITH_TGA_WRITER_
#include "irr/asset/CImageWriterTGA.h"
#endif

#ifdef _IRR_COMPILE_WITH_JPG_WRITER_
#include "irr/asset/CImageWriterJPG.h"
#endif

#ifdef _IRR_COMPILE_WITH_PNG_WRITER_
#include "irr/asset/CImageWriterPNG.h"
#endif

#ifdef _IRR_COMPILE_WITH_OPENEXR_WRITER_
#include "irr/asset/CImageWriterOpenEXR.h"
#endif

#ifdef _IRR_COMPILE_WITH_GLI_WRITER_
#include "irr/asset/CGLIWriter.h"
#endif

#include "irr/asset/CGLSLLoader.h"
#include "irr/asset/CSPVLoader.h"


using namespace irr;
using namespace asset;

std::function<void(SAssetBundle&)> irr::asset::makeAssetGreetFunc(const IAssetManager* const _mgr)
{
    return [_mgr](SAssetBundle& _asset) {
        _mgr->setAssetCached(_asset, true); 
        auto rng = _asset.getContents();
        //assets being in the cache must be immutable
        for (auto it = rng.first; it != rng.second; ++it)
            _mgr->setAssetMutable(it->get(), false);
    };
}
std::function<void(SAssetBundle&)> irr::asset::makeAssetDisposeFunc(const IAssetManager* const _mgr)
{
    return [_mgr](SAssetBundle& _asset) { 
        _mgr->setAssetCached(_asset, false); 
        auto rng = _asset.getContents();
        for (auto it = rng.first; it != rng.second; ++it)
            _mgr->setAssetMutable(it->get(), true);
    };
}

void IAssetManager::initializeMeshTools()
{
    m_geometryCreator = core::make_smart_refctd_ptr<CGeometryCreator>();
    m_meshManipulator = core::make_smart_refctd_ptr<CMeshManipulator>();
    m_glslCompiler = core::make_smart_refctd_ptr<IGLSLCompiler>(m_fileSystem.get());
}

const IGeometryCreator* IAssetManager::getGeometryCreator() const
{
	return m_geometryCreator.get();
}

const IMeshManipulator* IAssetManager::getMeshManipulator() const
{
    return m_meshManipulator.get();
}


void IAssetManager::addLoadersAndWriters()
{
#ifdef _IRR_COMPILE_WITH_STL_LOADER_
	addAssetLoader(core::make_smart_refctd_ptr<asset::CSTLMeshFileLoader>());
#endif
#ifdef _IRR_COMPILE_WITH_PLY_LOADER_
	addAssetLoader(core::make_smart_refctd_ptr<asset::CPLYMeshFileLoader>(this));
#endif
#ifdef _IRR_COMPILE_WITH_MTL_LOADER_
    addAssetLoader(core::make_smart_refctd_ptr<asset::CGraphicsPipelineLoaderMTL>(this));
#endif
#ifdef _IRR_COMPILE_WITH_OBJ_LOADER_
	addAssetLoader(core::make_smart_refctd_ptr<asset::COBJMeshFileLoader>(this));
#endif
#ifdef _IRR_COMPILE_WITH_BAW_LOADER_
	addAssetLoader(core::make_smart_refctd_ptr<asset::CBAWMeshFileLoader>(this));
#endif
#ifdef _IRR_COMPILE_WITH_JPG_LOADER_
	addAssetLoader(core::make_smart_refctd_ptr<asset::CImageLoaderJPG>());
#endif
#ifdef _IRR_COMPILE_WITH_PNG_LOADER_
	addAssetLoader(core::make_smart_refctd_ptr<asset::CImageLoaderPng>());
#endif
#ifdef _IRR_COMPILE_WITH_OPENEXR_LOADER_
	addAssetLoader(core::make_smart_refctd_ptr<asset::CImageLoaderOpenEXR>());
#endif
#ifdef  _IRR_COMPILE_WITH_GLI_LOADER_
	addAssetLoader(core::make_smart_refctd_ptr<asset::CGLILoader>());
#endif 
#ifdef _IRR_COMPILE_WITH_TGA_LOADER_
	addAssetLoader(core::make_smart_refctd_ptr<asset::CImageLoaderTGA>());
#endif
	addAssetLoader(core::make_smart_refctd_ptr<asset::CGLSLLoader>());
	addAssetLoader(core::make_smart_refctd_ptr<asset::CSPVLoader>());

#ifdef _IRR_COMPILE_WITH_BAW_WRITER_
	addAssetWriter(core::make_smart_refctd_ptr<asset::CBAWMeshWriter>(getFileSystem()));
#endif
#ifdef _IRR_COMPILE_WITH_PLY_WRITER_
	addAssetWriter(core::make_smart_refctd_ptr<asset::CPLYMeshWriter>());
#endif
#ifdef _IRR_COMPILE_WITH_STL_WRITER_
	addAssetWriter(core::make_smart_refctd_ptr<asset::CSTLMeshWriter>());
#endif
#ifdef _IRR_COMPILE_WITH_TGA_WRITER_
	addAssetWriter(core::make_smart_refctd_ptr<asset::CImageWriterTGA>());
#endif
#ifdef _IRR_COMPILE_WITH_JPG_WRITER_
	addAssetWriter(core::make_smart_refctd_ptr<asset::CImageWriterJPG>());
#endif
#ifdef _IRR_COMPILE_WITH_PNG_WRITER_
	addAssetWriter(core::make_smart_refctd_ptr<asset::CImageWriterPNG>());
#endif
#ifdef _IRR_COMPILE_WITH_OPENEXR_WRITER_
	addAssetWriter(core::make_smart_refctd_ptr<asset::CImageWriterOpenEXR>());
#endif
#ifdef _IRR_COMPILE_WITH_GLI_WRITER_
	addAssetWriter(core::make_smart_refctd_ptr<asset::CGLIWriter>());
#endif
}

void IAssetManager::insertBuiltinAssets()
{
    auto addBuiltInToCaches = [&](auto asset, const char* path) -> void
	{
		asset::SAssetBundle bundle({asset});
		changeAssetKey(bundle,path);
		insertAssetIntoCache(bundle);
	};

	// materials
	{
		auto buildInShader = [&](const char* source, asset::ISpecializedShader::E_SHADER_STAGE type, std::initializer_list<const char*> paths) -> void
		{
			auto shader = core::make_smart_refctd_ptr<asset::ICPUSpecializedShader>(core::make_smart_refctd_ptr<asset::ICPUShader>(source),
																					asset::ISpecializedShader::SInfo({},nullptr,"main",type));
            for(auto &path : paths)
			    addBuiltInToCaches(shader,path);
		};

        // single texture lambertian shader
		buildInShader(R"===(
        #version 430 core

        layout(set = 0, binding = 0) uniform sampler2D albedo;

        layout(location = 0) in vec2 uv;

        layout(location = 0) out vec4 pixelColor;

        void main()
        {
            pixelColor = texture(albedo,uv);
        }
		)===", asset::ISpecializedShader::ESS_FRAGMENT, { "irr/builtin/materials/lambertian/singletexture/specializedshader" });

        // uv debug shader
        buildInShader(R"===(
        #version 430 core
        layout(location = 0) in vec3 vPos;
        layout(location = 2) in vec2 vTexCoord;

        #include <irr/builtin/glsl/vertex_utils/vertex_utils.glsl>
        #include <irr/builtin/glsl/broken_driver_workarounds/amd.glsl>

        layout (set = 1, binding = 0, row_major, std140) uniform UBO 
        {
            irr_glsl_SBasicViewParameters params;
        } CamData;
        
        layout(location = 0) out vec2 uv;

        void main()
        {
            gl_Position = irr_glsl_pseudoMul4x4with3x1(irr_builtin_glsl_workaround_AMD_broken_row_major_qualifier_mat4x4(CamData.params.MVP), vPos);
	        uv = vTexCoord;
        }
		)===", asset::ISpecializedShader::ESS_VERTEX, { "irr/builtin/materials/debug/uv_debug_shader/specializedshader",
                                                        "irr/builtin/materials/lambertian/singletexture/specializedshader" });
        buildInShader(R"===(
        #version 430 core

        layout(location = 0) in vec2 uv;

        layout(location = 0) out vec4 pixelColor;

        void main()
        {
            pixelColor = vec4(uv.x, uv.y, 1.0, 1.0);
        }
		)===", asset::ISpecializedShader::ESS_FRAGMENT, { "irr/builtin/materials/debug/uv_debug_shader/specializedshader" });

        // vertex color debug shader
        buildInShader(R"===(
        #version 430 core
        layout(location = 0) in vec3 vPos;
        layout(location = 1) in vec3 vCol;

        #include <irr/builtin/glsl/vertex_utils/vertex_utils.glsl>
        #include <irr/builtin/glsl/broken_driver_workarounds/amd.glsl>

        layout (set = 1, binding = 0, row_major, std140) uniform UBO 
        {
            irr_glsl_SBasicViewParameters params;
        } CamData;

        layout(location = 0) out vec3 color;

        void main()
        {
            gl_Position = irr_glsl_pseudoMul4x4with3x1(irr_builtin_glsl_workaround_AMD_broken_row_major_qualifier_mat4x4(CamData.params.MVP), vPos);
	        color = vCol;
        }
		)===", asset::ISpecializedShader::ESS_VERTEX, { "irr/builtin/materials/debug/vertex_color_debug_shader/specializedshader" });

        // normal debug shader
        buildInShader(R"===(
        #version 430 core
        layout(location = 0) in vec3 vPos; 
        layout(location = 3) in vec3 vNormal;

        #include <irr/builtin/glsl/vertex_utils/vertex_utils.glsl>
        #include <irr/builtin/glsl/broken_driver_workarounds/amd.glsl>

        layout (set = 1, binding = 0, row_major, std140) uniform UBO 
        {
            irr_glsl_SBasicViewParameters params;
        } CamData;

        layout(location = 0) out vec3 color;

        void main()
        {
            gl_Position = gl_Position = irr_glsl_pseudoMul4x4with3x1(irr_builtin_glsl_workaround_AMD_broken_row_major_qualifier_mat4x4(CamData.params.MVP), vPos);
            color = vNormal*0.5+vec3(0.5);
        }
        )===", asset::ISpecializedShader::ESS_VERTEX, { "irr/builtin/materials/debug/normal_debug_shader/specializedshader" });

        buildInShader(R"===(
        #version 430 core

        layout(location = 0) in vec3 color; 

        layout(location = 0) out vec4 pixelColor;

        void main()
        {
            pixelColor = vec4(color,1.0);
        }
        )===", asset::ISpecializedShader::ESS_FRAGMENT, { "irr/builtin/materials/debug/normal_debug_shader/specializedshader",
                                                          "irr/builtin/materials/debug/vertex_color_debug_shader/specializedshader" });

		constexpr uint32_t bindingCount = 1u;
		asset::ICPUDescriptorSetLayout::SBinding pBindings[bindingCount] = {0u,asset::EDT_COMBINED_IMAGE_SAMPLER,1u,asset::ISpecializedShader::ESS_FRAGMENT,nullptr};
		auto dsLayout = core::make_smart_refctd_ptr<asset::ICPUDescriptorSetLayout>(pBindings,pBindings+bindingCount);
		addBuiltInToCaches(dsLayout,"irr/builtin/materials/lambertian/singletexture/descriptorsetlayout/2");
		//
		constexpr uint32_t pcCount = 1u;
		asset::SPushConstantRange pcRanges[pcCount] = {asset::ISpecializedShader::ESS_VERTEX,0u,sizeof(core::matrix4SIMD)};
		auto pLayout = core::make_smart_refctd_ptr<asset::ICPUPipelineLayout>(pcRanges,pcRanges+pcCount,nullptr,nullptr,core::smart_refctd_ptr(dsLayout),nullptr);
		addBuiltInToCaches(pLayout,"irr/builtin/materials/lambertian/singletexture/pipelinelayout");
	}

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
		addBuiltInToCaches(sampler,"irr/builtin/samplers/default");

        params.TextureWrapU = params.TextureWrapV = params.TextureWrapW = asset::ISampler::ETC_CLAMP_TO_BORDER;
        sampler = core::make_smart_refctd_ptr<asset::ICPUSampler>(params);
        addBuiltInToCaches(sampler, "irr/builtin/samplers/default_clamp_to_border");
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

        addBuiltInToCaches(dummy2dImgView, "irr/builtin/image_views/dummy2d");
        addBuiltInToCaches(dummy2dImage, "irr/builtin/images/dummy2d");
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
            //for filling this UBO with actual data, one can use asset::SBasicViewParameters struct defined in irr/asset/asset_utils.h
            constexpr size_t UBO_SZ = sizeof(asset::SBasicViewParameters);
            auto ubo = core::make_smart_refctd_ptr<asset::ICPUBuffer>(UBO_SZ);
            asset::fillBufferWithDeadBeef(ubo.get());
            desc->desc = std::move(ubo);
            desc->buffer.offset = 0ull;
            desc->buffer.size = UBO_SZ;
        }
        addBuiltInToCaches(ds1, "irr/builtin/descriptor_set/basic_view_parameters");
        addBuiltInToCaches(defaultDs1Layout, "irr/builtin/descriptor_set_layout/basic_view_parameters");
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
        auto paths = {"irr/builtin/materials/lambertian/no_texture/pipelinelayout",
                      "irr/builtin/loaders/PLY/pipelinelayout"};

        for(auto &path : paths)
            addBuiltInToCaches(pipelineLayout, path);
    }
}
