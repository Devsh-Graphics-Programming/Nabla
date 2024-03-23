// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "nbl/asset/asset.h"
#include "nbl/asset/interchange/CGraphicsPipelineLoaderMTL.h"
#include "nbl/asset/utils/CDerivativeMapCreator.h"

#include <utility>
#include <regex>
#include <filesystem>

#include "nbl/system/CFileView.h"

#include "nbl/builtin/MTLdefaults.h"



using namespace nbl;
using namespace asset;

#define VERT_SHADER_NO_UV_CACHE_KEY "nbl/builtin/shader/loader/mtl/vertex_no_uv.vert"
#define VERT_SHADER_UV_CACHE_KEY "nbl/builtin/shader/loader/mtl/vertex_uv.vert"
#define FRAG_SHADER_NO_UV_CACHE_KEY "nbl/builtin/shader/loader/mtl/fragment_no_uv.frag"
#define FRAG_SHADER_UV_CACHE_KEY "nbl/builtin/shader/loader/mtl/fragment_uv.frag"

CGraphicsPipelineLoaderMTL::CGraphicsPipelineLoaderMTL(IAssetManager* _am, core::smart_refctd_ptr<system::ISystem>&& sys) : 
    IRenderpassIndependentPipelineLoader(_am), m_system(std::move(sys))
{
#if 0 // Remove IRenderpassIndependentPipelines and use MC for Mesh Loaders
    //create vertex shaders and insert them into cache
    auto registerShader = [&]<core::StringLiteral Path>(ICPUShader::E_SHADER_STAGE stage) -> void
    {
        auto fileSystem = m_assetMgr->getSystem();

        auto loadBuiltinData = [&](const std::string _path) -> core::smart_refctd_ptr<const nbl::system::IFile>
        {
            nbl::system::ISystem::future_t<core::smart_refctd_ptr<nbl::system::IFile>> future;
            fileSystem->createFile(future, system::path(_path), core::bitflag(nbl::system::IFileBase::ECF_READ) | nbl::system::IFileBase::ECF_MAPPABLE);
            if (future.wait())
                return future.copy();
            return nullptr;
        };

        core::smart_refctd_ptr<const system::IFile> data = loadBuiltinData(Path.value);
        auto buffer = core::make_smart_refctd_ptr<asset::ICPUBuffer>(data->getSize()+1u);
        char* bufferPtr = reinterpret_cast<char*>(buffer->getPointer());
        memcpy(bufferPtr, data->getMappedPointer(), data->getSize());
        bufferPtr[data->getSize()] = '\0';

        auto unspecializedShader = core::make_smart_refctd_ptr<asset::ICPUShader>(
            std::move(buffer),
            stage,
            asset::IShader::E_CONTENT_TYPE::ECT_GLSL,
            stage != ICPUShader::ESS_VERTEX
            ? "?Nabla PipelineLoaderMTL FragmentShader?"
            : "?Nabla PipelineLoaderMTL VertexShader?");
        
        ICPUSpecializedShader::SInfo specInfo({}, nullptr, "main");
		auto shader = core::make_smart_refctd_ptr<asset::ICPUSpecializedShader>(std::move(unspecializedShader),std::move(specInfo));
        const char* cacheKey = Path.value;
        auto assetbundle = SAssetBundle(nullptr,{ core::smart_refctd_ptr_static_cast<IAsset>(std::move(shader)) });
        insertBuiltinAssetIntoCache(m_assetMgr, assetbundle, cacheKey);
    };

    registerShader.operator()<core::StringLiteral(VERT_SHADER_NO_UV_CACHE_KEY)>(ICPUShader::ESS_VERTEX);
    registerShader.operator()<core::StringLiteral(VERT_SHADER_UV_CACHE_KEY)>(ICPUShader::ESS_VERTEX);
    registerShader.operator()<core::StringLiteral(FRAG_SHADER_NO_UV_CACHE_KEY)>(ICPUShader::ESS_FRAGMENT);
    registerShader.operator()<core::StringLiteral(FRAG_SHADER_UV_CACHE_KEY)>(ICPUShader::ESS_FRAGMENT);
#endif
}

void CGraphicsPipelineLoaderMTL::initialize()
{
    IRenderpassIndependentPipelineLoader::initialize();

    auto dfltOver = IAssetLoaderOverride(m_assetMgr);
    // need to do this first
    {
        const IAssetLoader::SAssetLoadContext fakeCtx(IAssetLoader::SAssetLoadParams{}, nullptr);

        // find ds1 layout
        auto ds1layout = dfltOver.findDefaultAsset<ICPUDescriptorSetLayout>("nbl/builtin/descriptor_set_layout/basic_view_parameters", fakeCtx, 0u).first;

        // precompute the no UV pipeline layout
        {
            SPushConstantRange pcRng;
            pcRng.stageFlags = ICPUShader::ESS_FRAGMENT;
            pcRng.offset = 0u;
            pcRng.size = sizeof(SMtl::params);
            //if intellisense shows error here, it's most likely intellisense's fault and it'll build fine anyway
            static_assert(sizeof(SMtl::params) <= ICPUMeshBuffer::MAX_PUSH_CONSTANT_BYTESIZE, "It must fit in push constants!");

            auto pplnLayout = core::make_smart_refctd_ptr<ICPUPipelineLayout>(std::span<const SPushConstantRange>(&pcRng,1),nullptr,core::smart_refctd_ptr(ds1layout),nullptr,nullptr);
            auto assetbundle = SAssetBundle(nullptr, { core::smart_refctd_ptr_static_cast<IAsset>(std::move(pplnLayout)) });
            insertBuiltinAssetIntoCache(m_assetMgr, assetbundle, "nbl/builtin/pipeline_layout/loader/mtl/no_uv");
        }
    }

    // wrong solution because `__TIME__` has a non-portable timezone definition, included only for illustrative purpose only
    std::tm tm = {
        .tm_sec = 6,
        .tm_min = 9,
        .tm_hour = 6,
        .tm_mday = 9,
        .tm_mon = 6,
        .tm_year = 69,
        .tm_isdst = 0
    };
    const auto tp = std::chrono::system_clock::from_time_t(std::mktime(&tm));

    // default pipelines
    auto default_mtl_file = core::make_smart_refctd_ptr<system::CFileView<system::CNullAllocator>>(
        system::path("Nabla default MTL material"),
        system::IFile::ECF_READ,
        std::chrono::clock_cast<system::IFile::time_point_t::clock>(tp),
        const_cast<char*>(DUMMY_MTL_CONTENT),
        strlen(DUMMY_MTL_CONTENT)
    );

    SAssetLoadParams assetLoadParams;
    auto bundle = loadAsset(default_mtl_file.get(), assetLoadParams, &dfltOver);

    insertBuiltinAssetIntoCache(m_assetMgr, bundle, "nbl/builtin/renderpass_independent_pipeline/loader/mtl/missing_material_pipeline");
}

bool CGraphicsPipelineLoaderMTL::isALoadableFileFormat(system::IFile* _file, const system::logger_opt_ptr logger) const
{
    if (!_file)
        return false;

    std::string mtl;
    mtl.resize(_file->getSize());
    system::IFile::success_t success;
    _file->read(success, mtl.data(), 0, _file->getSize());
    return success && mtl.find("newmtl")!=std::string::npos;
}

SAssetBundle CGraphicsPipelineLoaderMTL::loadAsset(system::IFile* _file, const IAssetLoader::SAssetLoadParams& _params, IAssetLoader::IAssetLoaderOverride* _override, uint32_t _hierarchyLevel)
{
    SContext ctx(
        asset::IAssetLoader::SAssetLoadContext{
            _params,
            _file
        },
        _hierarchyLevel,
        _override
    );

    const std::filesystem::path fullName = _file->getFileName();
	const std::string relPath = [&fullName]() -> std::string
	{
		auto dir = fullName.filename().string();
        return dir;
	}();

    auto materials = readMaterials(_file, _params.logger);

    // because one for UV and one without UV
    constexpr uint32_t PIPELINE_PERMUTATION_COUNT = 2u;
    const auto pipelineCount = materials.size()*PIPELINE_PERMUTATION_COUNT;

    auto retval = core::make_refctd_dynamic_array<SAssetBundle::contents_container_t>(pipelineCount);
    auto meta = core::make_smart_refctd_ptr<CMTLMetadata>(pipelineCount,core::smart_refctd_ptr(m_basicViewParamsSemantics));
    uint32_t offset = 0u;
    for (auto& material : materials)
    {
        auto createPplnDescAndMeta = [&](const bool hasUV) -> void
        {
            const uint32_t hash = hasUV ? 1u:0u;
            auto ppln = makePipelineFromMtl(ctx,material,hasUV);
            core::smart_refctd_ptr<ICPUDescriptorSet> ds3;
            if (hasUV)
            {
                const std::string dsCacheKey = fullName.string() + "?" + material.name + "?_ds";
                const uint32_t ds3HLevel = _hierarchyLevel+ICPUMesh::DESC_SET_HIERARCHYLEVELS_BELOW;
                ds3 = _override->findDefaultAsset<ICPUDescriptorSet>(dsCacheKey,ctx.inner,ds3HLevel).first;
                if (!ds3)
                {
                    ds3 = makeDescSet(loadImages(relPath,material,ctx), ppln->getLayout()->getDescriptorSetLayout(3u), ctx);
                    if (ds3)
                    {
                        SAssetBundle bundle(nullptr,{ ds3 });
                        _override->insertAssetIntoCache(bundle,dsCacheKey,ctx.inner,ds3HLevel);
                    }
                }
            }
            meta->placeMeta(offset,ppln.get(),std::move(ds3),material.params,std::string(material.name),hash);
            retval->operator[](offset) = std::move(ppln);
            offset++;
        };
        createPplnDescAndMeta(false);
        createPplnDescAndMeta(true);
    }
    
    if (materials.empty())
        return SAssetBundle(nullptr, {});
    return SAssetBundle(std::move(meta),std::move(retval));
}

core::smart_refctd_ptr<ICPURenderpassIndependentPipeline> CGraphicsPipelineLoaderMTL::makePipelineFromMtl(SContext& _ctx, const SMtl& _mtl, bool hasUV)
{
    return nullptr;
#if 0 // Remove IRenderpassIndependentPipelines and use MC for Mesh Loaders
    SBlendParams blendParams;

    std::string cacheKey("nbl/builtin/renderpass_independent_pipeline/loader/mtl/");
    {
        const uint32_t illum = _mtl.params.extra&0xfu;
        if (illum==4u || illum==6u || illum==7u || illum==9u)
        {
            cacheKey += "thindielectric";
            
            blendParams.blendParams[0].blendEnable = true;
            blendParams.blendParams[0].srcColorFactor = EBF_ONE;
            blendParams.blendParams[0].srcAlphaFactor = EBF_ONE;
            blendParams.blendParams[0].dstColorFactor = EBF_ONE_MINUS_SRC_ALPHA;
            blendParams.blendParams[0].dstAlphaFactor = EBF_ONE_MINUS_SRC_ALPHA;
        }
        else if (_mtl.maps[CMTLMetadata::CRenderpassIndependentPipeline::EMP_OPACITY].size() || _mtl.params.opacity!=1.f)
        {
            cacheKey += "opacitymapped";

            blendParams.blendParams[0].blendEnable = true;
            blendParams.blendParams[0].srcColorFactor = EBF_SRC_ALPHA;
            blendParams.blendParams[0].srcAlphaFactor = EBF_SRC_ALPHA;
            blendParams.blendParams[0].dstColorFactor = EBF_ONE_MINUS_SRC_ALPHA;
            blendParams.blendParams[0].dstAlphaFactor = EBF_ONE_MINUS_SRC_ALPHA;
        }
        else
            cacheKey += "opaque";

        if (hasUV)
            cacheKey += "/clamp/"+std::to_string(_mtl.clamp);
        else
            cacheKey += "/no_uv";
    }
    auto ppln = _ctx.loaderOverride->findDefaultAsset<ICPURenderpassIndependentPipeline>(cacheKey,_ctx.inner,_ctx.topHierarchyLevel).first;

    if (!ppln)
    {
        ICPUSpecializedShader* shaders[] =
        {
            _ctx.loaderOverride->findDefaultAsset<ICPUSpecializedShader>(hasUV ? VERT_SHADER_UV_CACHE_KEY:VERT_SHADER_NO_UV_CACHE_KEY,_ctx.inner,_ctx.topHierarchyLevel+ICPURenderpassIndependentPipeline::SPECIALIZED_SHADER_HIERARCHYLEVELS_BELOW).first.get(),
            _ctx.loaderOverride->findDefaultAsset<ICPUSpecializedShader>(hasUV ? FRAG_SHADER_UV_CACHE_KEY:FRAG_SHADER_NO_UV_CACHE_KEY,_ctx.inner,_ctx.topHierarchyLevel+ICPURenderpassIndependentPipeline::SPECIALIZED_SHADER_HIERARCHYLEVELS_BELOW).first.get()
        };

        const uint32_t pipelineHLevel = _ctx.topHierarchyLevel+ICPURenderpassIndependentPipeline::PIPELINE_LAYOUT_HIERARCHYLEVELS_BELOW;
        core::smart_refctd_ptr<ICPUPipelineLayout> layout = _ctx.loaderOverride->findDefaultAsset<ICPUPipelineLayout>("nbl/builtin/pipeline_layout/loader/mtl/no_uv",_ctx.inner,pipelineHLevel).first;
        if (hasUV)
        {
            auto noUVLayout = std::move(layout);

            std::string pplnLayoutCacheKey("nbl/builtin/pipeline_layout/loader/mtl/clamp/");
            pplnLayoutCacheKey += std::to_string(_mtl.clamp);
            layout = _ctx.loaderOverride->findDefaultAsset<ICPUPipelineLayout>(pplnLayoutCacheKey,_ctx.inner,pipelineHLevel).first;

            if (!layout)
            {
                core::smart_refctd_ptr<ICPUDescriptorSetLayout> ds3Layout;
                {
                    //assumes all supported textures are always present
                    //since vulkan doesnt support bindings with no/null descriptor, absent textures will be filled with dummy 2D texture (while creating desc set)
                    auto bindings = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<ICPUDescriptorSetLayout::SBinding>>(static_cast<size_t>(CMTLMetadata::CRenderpassIndependentPipeline::EMP_REFL_POSX) + 1ull);

                    ICPUDescriptorSetLayout::SBinding bnd;
                    bnd.count = 1u;
                    bnd.stageFlags = ICPUShader::ESS_FRAGMENT;
                    bnd.type = IDescriptor::E_TYPE::ET_COMBINED_IMAGE_SAMPLER;
                    bnd.binding = 0u;
                    std::fill(bindings->begin(), bindings->end(), bnd);

                    core::smart_refctd_ptr<ICPUSampler> samplers[] =
                    {
                        _ctx.loaderOverride->findDefaultAsset<ICPUSampler>("nbl/builtin/sampler/default",_ctx.inner,_ctx.topHierarchyLevel+ICPURenderpassIndependentPipeline::IMMUTABLE_SAMPLER_HIERARCHYLEVELS_BELOW).first,
                        _ctx.loaderOverride->findDefaultAsset<ICPUSampler>("nbl/builtin/sampler/default_clamp_to_border",_ctx.inner,_ctx.topHierarchyLevel+ICPURenderpassIndependentPipeline::IMMUTABLE_SAMPLER_HIERARCHYLEVELS_BELOW).first
                    };
                    for (uint32_t i = 0u; i <= CMTLMetadata::CRenderpassIndependentPipeline::EMP_REFL_POSX; ++i)
                    {
                        (*bindings)[i].binding = i;

                        const uint32_t clamp = (_mtl.clamp >> i) & 1u;
                        (*bindings)[i].samplers = samplers + clamp;
                    }
                    ds3Layout = core::make_smart_refctd_ptr<ICPUDescriptorSetLayout>(bindings->begin(), bindings->end());
                }
                layout = core::move_and_static_cast<ICPUPipelineLayout>(noUVLayout->clone(0u)); // clone at 0 depth
                layout->setDescriptorSetLayout(3u,std::move(ds3Layout));
                auto bundle = SAssetBundle(nullptr,{ layout });
                _ctx.loaderOverride->insertAssetIntoCache(bundle, pplnLayoutCacheKey, _ctx.inner, pipelineHLevel);
            }
        }

        constexpr uint32_t POSITION = 0u;
        constexpr uint32_t UV = 2u;
        constexpr uint32_t NORMAL = 3u;
        constexpr uint32_t BND_NUM = 0u;

        SVertexInputParams vtxParams;

        vtxParams.enabledAttribFlags = (1u << POSITION) | (1u << NORMAL);
        vtxParams.enabledBindingFlags = 1u << BND_NUM;
        vtxParams.bindings[BND_NUM].stride = 24u;
        vtxParams.bindings[BND_NUM].inputRate = EVIR_PER_VERTEX;
        //position
        vtxParams.attributes[POSITION].binding = BND_NUM;
        vtxParams.attributes[POSITION].format = EF_R32G32B32_SFLOAT;
        vtxParams.attributes[POSITION].relativeOffset = 0u;
        //normal
        vtxParams.attributes[NORMAL].binding = BND_NUM;
        vtxParams.attributes[NORMAL].format = EF_A2B10G10R10_SNORM_PACK32;
        vtxParams.attributes[NORMAL].relativeOffset = 20u;

        //uv
        if (hasUV)
        {
            vtxParams.enabledAttribFlags |= (1u << UV);
            vtxParams.attributes[UV].binding = BND_NUM;
            vtxParams.attributes[UV].format = EF_R32G32_SFLOAT;
            vtxParams.attributes[UV].relativeOffset = 12u;
        }

        ppln = core::make_smart_refctd_ptr<ICPURenderpassIndependentPipeline>(std::move(layout), shaders, shaders+2u, vtxParams, blendParams, SPrimitiveAssemblyParams{}, SRasterizationParams{});
    }
    return ppln;
#endif
}

namespace
{
    //! skip space characters and stop on first non-space
    const char* goFirstWord(const char* buf, const char* const _bufEnd, bool acrossNewlines = true)
    {
        // skip space characters
        if (acrossNewlines)
            while ((buf != _bufEnd) && core::isspace(*buf))
                ++buf;
        else
            while ((buf != _bufEnd) && core::isspace(*buf) && (*buf != '\n'))
                ++buf;

        return buf;
    }


    //! skip current word and stop at beginning of next one
    const char* goNextWord(const char* buf, const char* const _bufEnd, bool acrossNewlines = true)
    {
        // skip current word
        while ((buf != _bufEnd) && !core::isspace(*buf))
            ++buf;

        return goFirstWord(buf, _bufEnd, acrossNewlines);
    }


    //! Read until line break is reached and stop at the next non-space character
    const char* goNextLine(const char* buf, const char* const _bufEnd)
    {
        // look for newline characters
        while (buf != _bufEnd)
        {
            // found it, so leave
            if (*buf == '\n' || *buf == '\r')
                break;
            ++buf;
        }
        return goFirstWord(buf, _bufEnd);
    }


    uint32_t copyWord(char* outBuf, const char* const inBuf, uint32_t outBufLength, const char* const _bufEnd)
    {
        if (!outBufLength)
            return 0;
        if (!inBuf)
        {
            *outBuf = 0;
            return 0;
        }

        uint32_t i = 0;
        while (inBuf[i])
        {
            if (core::isspace(inBuf[i]) || &(inBuf[i]) == _bufEnd)
                break;
            ++i;
        }

        uint32_t length = core::min(i, outBufLength - 1u);
        for (uint32_t j = 0u; j < length; ++j)
            outBuf[j] = inBuf[j];

        outBuf[length] = 0;
        return length;
    }

    const char* goAndCopyNextWord(char* outBuf, const char* inBuf, uint32_t outBufLength, const char* _bufEnd)
    {
        inBuf = goNextWord(inBuf, _bufEnd, false);
        copyWord(outBuf, inBuf, outBufLength, _bufEnd);
        return inBuf;
    }
}

const char* CGraphicsPipelineLoaderMTL::readTexture(const char* _bufPtr, const char* const _bufEnd, SMtl* _currMaterial, const char* _mapType) const
{
    static const std::unordered_map<std::string, CMTLMetadata::CRenderpassIndependentPipeline::E_MAP_TYPE> str2type =
    {
        {"Ka", CMTLMetadata::CRenderpassIndependentPipeline::EMP_AMBIENT},
        {"Kd", CMTLMetadata::CRenderpassIndependentPipeline::EMP_DIFFUSE},
        {"Ke", CMTLMetadata::CRenderpassIndependentPipeline::EMP_EMISSIVE},
        {"Ks", CMTLMetadata::CRenderpassIndependentPipeline::EMP_SPECULAR},
        {"Ns", CMTLMetadata::CRenderpassIndependentPipeline::EMP_SHININESS},
        {"d", CMTLMetadata::CRenderpassIndependentPipeline::EMP_OPACITY},
        {"bump", CMTLMetadata::CRenderpassIndependentPipeline::EMP_BUMP},
        {"disp", CMTLMetadata::CRenderpassIndependentPipeline::EMP_DISPLACEMENT},
        {"refl", CMTLMetadata::CRenderpassIndependentPipeline::EMP_REFL_POSX},
        {"norm", CMTLMetadata::CRenderpassIndependentPipeline::EMP_NORMAL},
        {"Pr", CMTLMetadata::CRenderpassIndependentPipeline::EMP_ROUGHNESS},
        {"Pm", CMTLMetadata::CRenderpassIndependentPipeline::EMP_METALLIC},
        {"Ps", CMTLMetadata::CRenderpassIndependentPipeline::EMP_SHEEN}
    };
    static const std::unordered_map<std::string, CMTLMetadata::CRenderpassIndependentPipeline::E_MAP_TYPE> refl_str2type =
    {
        {"top", CMTLMetadata::CRenderpassIndependentPipeline::EMP_REFL_POSY},
        {"bottom", CMTLMetadata::CRenderpassIndependentPipeline::EMP_REFL_NEGY},
        {"front", CMTLMetadata::CRenderpassIndependentPipeline::EMP_REFL_NEGZ},
        {"back", CMTLMetadata::CRenderpassIndependentPipeline::EMP_REFL_POSZ},
        {"left", CMTLMetadata::CRenderpassIndependentPipeline::EMP_REFL_NEGX},
        {"right", CMTLMetadata::CRenderpassIndependentPipeline::EMP_REFL_POSX}
    };

    constexpr static size_t WORD_BUFFER_LENGTH = 512ull;
    char tmpbuf[WORD_BUFFER_LENGTH]{};

    std::string mapTypeStr = _mapType;
    if (mapTypeStr.compare(0ull, 4ull, "map_")==0)
        mapTypeStr.erase(0ull, 4ull);

    auto mapType = CMTLMetadata::CRenderpassIndependentPipeline::EMP_COUNT;
    auto found = str2type.find(mapTypeStr);
    if (found != str2type.end())
        mapType = found->second;

    constexpr uint32_t ILLUM_MODEL_BITS = 4u;
    _currMaterial->params.extra |= (1u << (ILLUM_MODEL_BITS + mapType));

    _bufPtr = goAndCopyNextWord(tmpbuf, _bufPtr, WORD_BUFFER_LENGTH, _bufEnd);
    while (tmpbuf[0]=='-')
    {
        if (mapType==CMTLMetadata::CRenderpassIndependentPipeline::EMP_REFL_POSX && strncmp(tmpbuf, "-type", 5)==0)
        {
            _bufPtr = goAndCopyNextWord(tmpbuf, _bufPtr, WORD_BUFFER_LENGTH, _bufEnd);
            if (strlen(tmpbuf) >= 8ull) //shortest one is "cube_top"
            {
                found = refl_str2type.find(tmpbuf+5); //skip "cube_"
                if (found != refl_str2type.end())
                    mapType = found->second;
            }
        }
        else if (strncmp(_bufPtr,"-bm",3)==0)
		{
			_bufPtr = goAndCopyNextWord(tmpbuf, _bufPtr, WORD_BUFFER_LENGTH, _bufEnd);
			sscanf(tmpbuf, "%f", &_currMaterial->params.bumpFactor);
		}
		else
		if (strncmp(_bufPtr,"-blendu",7)==0)
            _bufPtr = goNextWord(_bufPtr, _bufEnd);
		else
		if (strncmp(_bufPtr,"-blendv",7)==0)
            _bufPtr = goNextWord(_bufPtr, _bufEnd);
		else
		if (strncmp(_bufPtr,"-cc",3)==0)
            _bufPtr = goNextWord(_bufPtr, _bufEnd);
		else
		if (strncmp(_bufPtr,"-clamp",6)==0)
        {
            _bufPtr = goAndCopyNextWord(tmpbuf, _bufPtr, WORD_BUFFER_LENGTH, _bufEnd);
            if (mapType != CMTLMetadata::CRenderpassIndependentPipeline::EMP_COUNT)
            {
                uint32_t clamp = (strcmp("off", tmpbuf) != 0);
                _currMaterial->clamp |= (clamp<<mapType);
            }
        }
		else
		if (strncmp(_bufPtr,"-texres",7)==0)
			_bufPtr = goNextWord(_bufPtr, _bufEnd);
		else
		if (strncmp(_bufPtr,"-type",5)==0)
            _bufPtr = goNextWord(_bufPtr, _bufEnd);
		else
		if (strncmp(_bufPtr,"-mm",3)==0)
		{
            _bufPtr = goNextWord(_bufPtr, _bufEnd);
            _bufPtr = goNextWord(_bufPtr, _bufEnd);
		}
		else
		if (strncmp(_bufPtr,"-o",2)==0) // texture coord translation
		{
			_bufPtr = goAndCopyNextWord(tmpbuf, _bufPtr, WORD_BUFFER_LENGTH, _bufEnd);
			// next parameters are optional, so skip rest of loop if no number is found
			_bufPtr = goAndCopyNextWord(tmpbuf, _bufPtr, WORD_BUFFER_LENGTH, _bufEnd);
			if (!core::isdigit(tmpbuf[0]))
				continue;
			_bufPtr = goAndCopyNextWord(tmpbuf, _bufPtr, WORD_BUFFER_LENGTH, _bufEnd);
			if (!core::isdigit(tmpbuf[0]))
				continue;
		}
		else
		if (strncmp(_bufPtr,"-s",2)==0) // texture coord scale
		{
			_bufPtr = goAndCopyNextWord(tmpbuf, _bufPtr, WORD_BUFFER_LENGTH, _bufEnd);
			// next parameters are optional, so skip rest of loop if no number is found
			_bufPtr = goAndCopyNextWord(tmpbuf, _bufPtr, WORD_BUFFER_LENGTH, _bufEnd);
			if (!core::isdigit(tmpbuf[0]))
				continue;
			_bufPtr = goAndCopyNextWord(tmpbuf, _bufPtr, WORD_BUFFER_LENGTH, _bufEnd);
			if (!core::isdigit(tmpbuf[0]))
				continue;
		}
		else
		if (strncmp(_bufPtr,"-t",2)==0)
		{
			_bufPtr = goAndCopyNextWord(tmpbuf, _bufPtr, WORD_BUFFER_LENGTH, _bufEnd);
			// next parameters are optional, so skip rest of loop if no number is found
			_bufPtr = goAndCopyNextWord(tmpbuf, _bufPtr, WORD_BUFFER_LENGTH, _bufEnd);
			if (!core::isdigit(tmpbuf[0]))
				continue;
			_bufPtr = goAndCopyNextWord(tmpbuf, _bufPtr, WORD_BUFFER_LENGTH, _bufEnd);
			if (!core::isdigit(tmpbuf[0]))
				continue;
		}
		// get next word
		_bufPtr = goAndCopyNextWord(tmpbuf, _bufPtr, WORD_BUFFER_LENGTH, _bufEnd);
    }

    if (mapType != CMTLMetadata::CRenderpassIndependentPipeline::EMP_COUNT)
    {
        std::string path = tmpbuf;
        std::replace(path.begin(), path.end(), '\\', '/');
        _currMaterial->maps[mapType] = std::move(path);
    }

    return _bufPtr;
}

CGraphicsPipelineLoaderMTL::image_views_set_t CGraphicsPipelineLoaderMTL::loadImages(const std::string& relDir, SMtl& _mtl, SContext& _ctx)
{
    images_set_t images;
    image_views_set_t views;

    for (uint32_t i = 0u; i < images.size(); ++i)
    {
        SAssetLoadParams lp = _ctx.inner.params;
        if (_mtl.maps[i].size() )
        {
            const uint32_t hierarchyLevel = _ctx.topHierarchyLevel + ICPURenderpassIndependentPipeline::IMAGE_HIERARCHYLEVELS_BELOW; // this is weird actually, we're not sure if we're loading image or image view
            SAssetBundle bundle;
            if (i != CMTLMetadata::CRenderpassIndependentPipeline::EMP_BUMP)
                bundle = interm_getAssetInHierarchy(m_assetMgr, _mtl.maps[i], lp, hierarchyLevel, _ctx.loaderOverride);
            else // TODO: you should attempt to get derivative map FIRST, then restore and regenerate! (right now you're always restoring!)
            {
                // we need bumpmap restored to create derivative map from it
                const uint32_t restoreLevels = 3u; // 2 in case of image (image, texel buffer) and 3 in case of image view (view, image, texel buffer)
                lp.restoreLevels = std::max(lp.restoreLevels, hierarchyLevel + restoreLevels);
                bundle = interm_getAssetInHierarchy(m_assetMgr, _mtl.maps[i], lp, hierarchyLevel, _ctx.loaderOverride);
            }
            auto asset = _ctx.loaderOverride->chooseDefaultAsset(bundle,_ctx.inner);
            if (asset)
            switch (bundle.getAssetType())
            {
                case IAsset::ET_IMAGE:
                    images[i] = core::smart_refctd_ptr_static_cast<ICPUImage>(asset);
                    break;
                case IAsset::ET_IMAGE_VIEW:
                    views[i] = core::smart_refctd_ptr_static_cast<ICPUImageView>(asset);
                    break;
                default:
                    // TODO: log an error
                    break;
            }
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
    if (images[CMTLMetadata::CRenderpassIndependentPipeline::EMP_REFL_POSX])
    {
        assert(allCubemapFacesAreSameSizeAndFormat(images.data() + CMTLMetadata::CRenderpassIndependentPipeline::EMP_REFL_POSX));

        size_t bufSz = 0ull;
        //assuming all cubemap layer images are same size and same format
        const size_t alignment = 1u<<hlsl::findLSB(images[CMTLMetadata::CRenderpassIndependentPipeline::EMP_REFL_POSX]->getRegions().begin()->bufferRowLength);
        core::vector<ICPUImage::SBufferCopy> regions_;
        regions_.reserve(6ull);
        for (uint32_t i = CMTLMetadata::CRenderpassIndependentPipeline::EMP_REFL_POSX; i < CMTLMetadata::CRenderpassIndependentPipeline::EMP_REFL_POSX + 6u; ++i)
        {
            assert(images[i]);
#ifndef _NBL_DEBUG
            if (images[i])
            {
#endif
                //assuming each image has just 1 region
                assert(images[i]->getRegions().size()==1ull);

                regions_.push_back(images[i]->getRegions().begin()[0]);
                regions_.back().bufferOffset = core::roundUp(regions_.back().bufferOffset, alignment);
                regions_.back().imageSubresource.baseArrayLayer = (i - CMTLMetadata::CRenderpassIndependentPipeline::EMP_REFL_POSX);

                bufSz += images[i]->getImageDataSizeInBytes();
#ifndef _NBL_DEBUG
            }
#endif
        }
        auto imgDataBuf = core::make_smart_refctd_ptr<ICPUBuffer>(bufSz);
        for (uint32_t i = CMTLMetadata::CRenderpassIndependentPipeline::EMP_REFL_POSX, j = 0u; i < CMTLMetadata::CRenderpassIndependentPipeline::EMP_REFL_POSX + 6u; ++i)
        {
#ifndef _NBL_DEBUG
            if (images[i])
            {
#endif
                void* dst = reinterpret_cast<uint8_t*>(imgDataBuf->getPointer()) + regions_[j].bufferOffset;
                const void* src = reinterpret_cast<const uint8_t*>(images[i]->getBuffer()->getPointer()) + images[i]->getRegions().begin()[0].bufferOffset;
                const size_t sz = images[i]->getImageDataSizeInBytes();
                memcpy(dst, src, sz);

                ++j;
#ifndef _NBL_DEBUG
            }
#endif
        }

        //assuming all cubemap layer images are same size and same format
        ICPUImage::SCreationParams cubemapParams = images[CMTLMetadata::CRenderpassIndependentPipeline::EMP_REFL_POSX]->getCreationParameters();
        cubemapParams.arrayLayers = 6u;
        cubemapParams.type = IImage::ET_2D;
        auto cubemap = ICPUImage::create(std::move(cubemapParams));
        auto regions = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<ICPUImage::SBufferCopy>>(regions_);
        cubemap->setBufferAndRegions(std::move(imgDataBuf), regions);
        //new image goes to EMP_REFL_POSX index and other ones get nulled-out
        images[CMTLMetadata::CRenderpassIndependentPipeline::EMP_REFL_POSX] = std::move(cubemap);
        std::fill_n(images.begin()+CMTLMetadata::CRenderpassIndependentPipeline::EMP_REFL_POSX+1u,5u,nullptr);
    }

    for (uint32_t i = 0u; i < views.size(); ++i)
    {
        const bool isBumpmap = i==CMTLMetadata::CRenderpassIndependentPipeline::EMP_BUMP;

        core::smart_refctd_ptr<ICPUImage> image = images[i];
        if (isBumpmap && views[i])
            image = views[i]->getCreationParameters().image;
        if (!image)
            continue;

        const std::string viewCacheKey = _mtl.maps[i] + "?view";
        // try cache
        {
            auto view = _ctx.loaderOverride->findDefaultAsset<ICPUImageView>(viewCacheKey, _ctx.inner, _ctx.topHierarchyLevel+ICPURenderpassIndependentPipeline::IMAGEVIEW_HIERARCHYLEVELS_BELOW);
            if (view.first)
            {
                if (isBumpmap)
                {
                    auto meta = view.second->selfCast<CDerivativeMapMetadata>()->getAssetSpecificMetadata(view.first.get());
                    _mtl.params.bumpFactor *= static_cast<const CDerivativeMapMetadata::CImageView*>(meta)->scale[0];
                }
                views[i] = std::move(view.first);
                continue;
            }
        }

        float derivativeScale;
        if (isBumpmap)
        {
            const ISampler::E_TEXTURE_CLAMP wrap = _mtl.isClampToBorder(CMTLMetadata::CRenderpassIndependentPipeline::EMP_BUMP) ? ISampler::ETC_CLAMP_TO_BORDER : ISampler::ETC_REPEAT;
            image = CDerivativeMapCreator::createDerivativeMapFromHeightMap<true>(image.get(), wrap, wrap, ISampler::ETBC_FLOAT_OPAQUE_BLACK, &derivativeScale);
            _mtl.params.bumpFactor *= derivativeScale;
        }

        constexpr IImageView<ICPUImage>::E_TYPE viewType[2]{ IImageView<ICPUImage>::ET_2D, IImageView<ICPUImage>::ET_CUBE_MAP };
        constexpr uint32_t layerCount[2]{ 1u, 6u };

        const bool isCubemap = (i == CMTLMetadata::CRenderpassIndependentPipeline::EMP_REFL_POSX);

        ICPUImageView::SCreationParams viewParams = {};
        viewParams.flags = static_cast<ICPUImageView::E_CREATE_FLAGS>(0u);
        viewParams.format = image->getCreationParameters().format;
        viewParams.viewType = viewType[isCubemap];
        asset::IImage::E_ASPECT_FLAGS aspectFlags = asset::IImage::EAF_COLOR_BIT;
        if (isDepthOrStencilFormat(viewParams.format) && !isDepthOnlyFormat(viewParams.format))
        {
            if (isStencilOnlyFormat(viewParams.format))
                aspectFlags = asset::IImage::EAF_STENCIL_BIT;
            else
                aspectFlags = asset::IImage::EAF_DEPTH_BIT;
        }
        viewParams.subresourceRange.aspectMask = aspectFlags;
        viewParams.subresourceRange.baseArrayLayer = 0u;
        viewParams.subresourceRange.layerCount = layerCount[isCubemap];
        viewParams.subresourceRange.baseMipLevel = 0u;
        viewParams.subresourceRange.levelCount = 1u;
        viewParams.image = std::move(image);

        core::smart_refctd_ptr<IAssetMetadata> metaData;
        auto view = ICPUImageView::create(std::move(viewParams));
        if (isBumpmap)
            metaData = core::make_smart_refctd_ptr<CDerivativeMapMetadata>(view.get(),&derivativeScale,true);
        views[i] = view;
        auto assetBundle = SAssetBundle(std::move(metaData),{view});
        _ctx.loaderOverride->insertAssetIntoCache(assetBundle, viewCacheKey, _ctx.inner, _ctx.topHierarchyLevel+ICPURenderpassIndependentPipeline::IMAGEVIEW_HIERARCHYLEVELS_BELOW);
    }

    return views;
}

core::smart_refctd_ptr<ICPUDescriptorSet> CGraphicsPipelineLoaderMTL::makeDescSet(image_views_set_t&& _views, ICPUDescriptorSetLayout* _dsLayout, SContext& _ctx)
{
    if (!_dsLayout)
        return nullptr;

    auto ds = core::make_smart_refctd_ptr<asset::ICPUDescriptorSet>(core::smart_refctd_ptr<ICPUDescriptorSetLayout>(_dsLayout));
    auto dummy2d = _ctx.loaderOverride->findDefaultAsset<ICPUImageView>("nbl/builtin/image_view/dummy2d",_ctx.inner,_ctx.topHierarchyLevel+ICPURenderpassIndependentPipeline::IMAGEVIEW_HIERARCHYLEVELS_BELOW).first;
    for (uint32_t i = 0u; i <= CMTLMetadata::CRenderpassIndependentPipeline::EMP_REFL_POSX; ++i)
    {
        auto descriptorInfos = ds->getDescriptorInfos(i, IDescriptor::E_TYPE::ET_COMBINED_IMAGE_SAMPLER);
        descriptorInfos.begin()[0].desc = _views[i] ? std::move(_views[i]) : dummy2d;
        descriptorInfos.begin()[0].info.image.imageLayout = IImage::LAYOUT::READ_ONLY_OPTIMAL;
    }

    return ds;
}

auto CGraphicsPipelineLoaderMTL::readMaterials(system::IFile* _file, const system::logger_opt_ptr logger) const -> core::vector<SMtl>
{
    std::string mtl;
    size_t fileSize = _file->getSize();
    mtl.resize(fileSize);

    system::IFile::success_t success;
    _file->read(success, mtl.data(), 0, fileSize);
    if (!success)
        return {};

    const char* bufPtr = mtl.c_str();
    const char* const bufEnd = mtl.c_str()+mtl.size();

    constexpr static size_t WORD_BUFFER_LENGTH = 512ull;
    char tmpbuf[WORD_BUFFER_LENGTH]{};

    auto readFloat = [&tmpbuf, &bufPtr, bufEnd] {
        float f = 0.f;

        bufPtr = goAndCopyNextWord(tmpbuf, bufPtr, WORD_BUFFER_LENGTH, bufEnd);
        sscanf(tmpbuf, "%f", &f);

        return f;
    };
    auto readRGB = [&readFloat] {
        core::vector3df_SIMD rgb(1.f);

        rgb.r = readFloat();
        rgb.g = readFloat();
        rgb.b = readFloat();

        return rgb;
    };

    core::vector<SMtl> materials;
    SMtl* currMaterial = nullptr;

    while (bufPtr != bufEnd)
    {
        copyWord(tmpbuf, bufPtr, WORD_BUFFER_LENGTH, bufEnd);
        if (currMaterial && (strncmp("map_", tmpbuf, 4u)==0 || strcmp("refl", tmpbuf)==0 || strcmp("norm", tmpbuf)==0 || strcmp("bump", tmpbuf)==0 || strcmp("disp", tmpbuf)==0))
        {
            readTexture(bufPtr, bufEnd, currMaterial, tmpbuf);
        }

        switch (*bufPtr)
        {
        case 'n': // newmtl
        {
            materials.push_back({});
            currMaterial = &materials.back();

            // extract new material's name
            bufPtr = goAndCopyNextWord(tmpbuf, bufPtr, WORD_BUFFER_LENGTH, bufEnd);

            currMaterial->name = tmpbuf;
        }
        break;
        case 'a': // aniso, anisor
            if (currMaterial)
            {
                if (bufPtr[5] == 'r')
                    currMaterial->params.anisoRotation = readFloat();
                else
                    currMaterial->params.anisotropy = readFloat();
            }
        break;
        case 'i': // illum - illumination
            if (currMaterial)
            {
                bufPtr = goAndCopyNextWord(tmpbuf, bufPtr, WORD_BUFFER_LENGTH, bufEnd);
                currMaterial->params.extra |= (atol(tmpbuf)&0x0f);//illum values are in range [0;10]
            }
            break;
        case 'N':
            if (currMaterial)
            {
                switch (bufPtr[1])
                {
                case 's': // Ns - shininess
                    currMaterial->params.shininess = readFloat();
                    break;
                case 'i': // Ni - refraction index
                    currMaterial->params.IoR = readFloat();
                    break;
                }
            }
            break;
        case 'K':
            if (currMaterial)
            {
                switch (bufPtr[1])
                {
                case 'd':		// Kd = diffuse
                    currMaterial->params.diffuse = readRGB();
                    break;
                case 's':		// Ks = specular
                    currMaterial->params.specular = readRGB();
                    break;
                case 'a':		// Ka = ambience
                    currMaterial->params.ambient = readRGB();
                    break;
                case 'e':		// Ke = emissive
                    currMaterial->params.emissive = readRGB();
                    break;
                }	// end switch(bufPtr[1])
            }	// end case 'K': if (currMaterial)...
            break;
        case 'P':
            if (currMaterial)
            {
                switch (bufPtr[1])
                {
                case 'r':
                    currMaterial->params.roughness = readFloat();
                    break;
                case 'm':
                    currMaterial->params.metallic = readFloat();
                    break;
                case 's':
                    currMaterial->params.sheen = readFloat();
                    break;
                case 'c':
                    switch (bufPtr[2])
                    {
                    case 'r':
                        currMaterial->params.clearcoatRoughness = readFloat();
                        break;
                    case 0:
                        currMaterial->params.clearcoatThickness = readFloat();
                        break;
                    }
                    break;
                }
            }
            break;
        case 'd': // d - transparency
            if (currMaterial)
                currMaterial->params.opacity = readFloat();
            break;
        case 'T':
            if (currMaterial)
            {
                switch (bufPtr[1])
                {
                case 'f':		// Tf - Transmitivity
                    currMaterial->params.transmissionFilter = readRGB();
                    sprintf(tmpbuf, "%s, %s: Detected Tf parameter, it won't be used in generated shader - fallback to alpha=0.5 instead", _file->getFileName().string().c_str(), currMaterial->name.c_str());
                    logger.log(tmpbuf, system::ILogger::ELL_WARNING);
                    break;
                case 'r':       // Tr, transparency = 1.0-d
                    currMaterial->params.opacity = (1.f - readFloat());
                    break;
                }
            }
            break;
        default: // comments or not recognised
            break;
        } // end switch(bufPtr[0])
        // go to next line
        bufPtr = goNextLine(bufPtr, bufEnd);
    }	// end while (bufPtr)

    return materials;
}
