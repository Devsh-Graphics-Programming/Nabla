#include "irr/asset/CGraphicsPipelineLoaderMTL.h"

using namespace irr;
using namespace asset;

bool CGraphicsPipelineLoaderMTL::isALoadableFileFormat(io::IReadFile* _file) const
{
    if (!_file)
        return false;

    const size_t prevPos = _file->getPos();

    _file->seek(0ull);

    std::string mtl;
    mtl.resize(_file->getSize());
    _file->read(mtl.data(), _file->getSize());
    _file->seek(prevPos);

    return mtl.find("newmtl") != std::string::npos;
}


core::smart_refctd_ptr<ICPUPipelineLayout> CGraphicsPipelineLoaderMTL::makePipelineLayoutFromMtl(const CMTLPipelineMetadata::SMtl& _mtl)
{
    const size_t textureCnt = std::count_if(_mtl.maps, _mtl.maps + CMTLPipelineMetadata::SMtl::EMP_REFL_POSX + 1u, [](const std::string& _path) { return !_path.empty(); });
    auto bindings = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<ICPUDescriptorSetLayout::SBinding>>(textureCnt);

    ICPUDescriptorSetLayout::SBinding bnd;
    bnd.count = 1u;
    bnd.stageFlags = ICPUSpecializedShader::ESS_FRAGMENT;
    bnd.type = EDT_COMBINED_IMAGE_SAMPLER;
    std::fill(bindings->begin(), bindings->end(), bnd);

    auto getDefaultSampler = [this](const char* _key) {
        size_t storageSz = 1ull;
        asset::SAssetBundle bundle;
        const IAsset::E_TYPE types[]{ IAsset::ET_SAMPLER, static_cast<IAsset::E_TYPE>(0u) };

        m_assetMgr->findAssets(storageSz, &bundle, _key, types);
        auto assets = bundle.getContents();
        assert(assets.first != assets.second);

        return assets.first[0];
    };

    core::smart_refctd_ptr<ICPUSampler> samplers[2];
    samplers[0] = getDefaultSampler("irr/builtin/samplers/default");
    samplers[1] = getDefaultSampler("irr/builtin/samplers/default_clamp_to_border");
    for (uint32_t i = 0u, j = 0u; i <= CMTLPipelineMetadata::SMtl::EMP_REFL_POSX; ++i)
    {
        if (!_mtl.maps[i].empty())
        {
            (*bindings)[j].binding = j;

            const uint32_t clamp = (_mtl.clamp >> i) & 1u;
            (*bindings)[j].samplers = samplers + clamp;

            ++j;
        }
    }

    auto dsLayout = core::make_smart_refctd_ptr<ICPUDescriptorSetLayout>(bindings->begin(), bindings->end());
    //ds with textures for material goes to set=3
    auto layout = core::make_smart_refctd_ptr<ICPUPipelineLayout>(nullptr, nullptr, nullptr, nullptr, nullptr, std::move(dsLayout));

    return layout;
}

SAssetBundle CGraphicsPipelineLoaderMTL::loadAsset(io::IReadFile* _file, const IAssetLoader::SAssetLoadParams& _params, IAssetLoader::IAssetLoaderOverride* _override, uint32_t _hierarchyLevel)
{
    auto materials = readMaterials(_file);

    core::vector<core::smart_refctd_ptr<ICPURenderpassIndependentPipeline>> pipelines(materials.size());
    SVertexInputParams vtxParams;
    SBlendParams blendParams;
    SPrimitiveAssemblyParams primParams;
    SRasterizationParams rasterParams;
    for (size_t i = 0ull; i < pipelines.size(); ++i)
    {
        auto layout = makePipelineLayoutFromMtl(materials[i]);
        pipelines[i] = core::make_smart_refctd_ptr<ICPURenderpassIndependentPipeline>(nullptr, std::move(layout), nullptr, nullptr, vtxParams, blendParams, primParams, rasterParams);
        m_assetMgr->setAssetMetadata(pipelines[i].get(), core::make_smart_refctd_ptr<CMTLPipelineMetadata>(std::move(materials[i])));
    }
    materials.clear();

    return asset::SAssetBundle(std::move(pipelines));
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
    static const core::unordered_map<std::string, SMtl::E_MAP_TYPE> str2type =
    {
        {"Ka", SMtl::EMP_AMBIENT},
        {"Kd", SMtl::EMP_DIFFUSE},
        {"Ke", SMtl::EMP_EMISSIVE},
        {"Ks", SMtl::EMP_SPECULAR},
        {"Ns", SMtl::EMP_SHININESS},
        {"d", SMtl::EMP_OPACITY},
        {"bump", SMtl::EMP_BUMP},
        {"disp", SMtl::EMP_DISPLACEMENT},
        {"refl", SMtl::EMP_REFL_POSX},
        {"norm", SMtl::EMP_NORMAL},
        {"Pr", SMtl::EMP_ROUGHNESS},
        {"Pm", SMtl::EMP_METALLIC},
        {"Ps", SMtl::EMP_SHEEN}
    };
    static const core::unordered_map<std::string, SMtl::E_MAP_TYPE> refl_str2type =
    {
        {"top", SMtl::EMP_REFL_POSY},
        {"bottom", SMtl::EMP_REFL_NEGY},
        {"front", SMtl::EMP_REFL_NEGZ},
        {"back", SMtl::EMP_REFL_POSZ},
        {"left", SMtl::EMP_REFL_NEGX},
        {"right", SMtl::EMP_REFL_POSX}
    };

    constexpr static size_t WORD_BUFFER_LENGTH = 512ull;
    char tmpbuf[WORD_BUFFER_LENGTH]{};

    std::string mapTypeStr = _mapType;
    if (mapTypeStr.compare(0ull, 4ull, "map_")==0)
        mapTypeStr.erase(0ull, 4ull);

    SMtl::E_MAP_TYPE mapType = SMtl::EMP_COUNT;
    auto found = str2type.find(mapTypeStr);
    if (found != str2type.end())
        mapType = found->second;

    _bufPtr = goAndCopyNextWord(tmpbuf, _bufPtr, WORD_BUFFER_LENGTH, _bufEnd);
    while (tmpbuf[0]=='-')
    {
        if (mapType==SMtl::EMP_REFL_POSX && strncmp(tmpbuf, "-type", 5)==0)
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
			sscanf(tmpbuf, "%f", &_currMaterial->bumpFactor);
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
            if (mapType != SMtl::EMP_COUNT)
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

    if (mapType != SMtl::EMP_COUNT)
    {
        std::string path = tmpbuf;
        std::replace(path.begin(), path.end(), '\\', '/');
        _currMaterial->maps[mapType] = std::move(path);
    }

    return _bufPtr;
}

auto CGraphicsPipelineLoaderMTL::readMaterials(io::IReadFile* _file) const -> core::vector<SMtl>
{
    std::string mtl;
    mtl.resize(_file->getSize());
    _file->read(mtl.data(), _file->getSize());

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
                const float a = readFloat();
                if (bufPtr[5] == 'r')
                    currMaterial->anisoRotation = a;
                else
                    currMaterial->anisotropy = a;
            }
        break;
        case 'i': // illum - illumination
            if (currMaterial)
            {
                bufPtr = goAndCopyNextWord(tmpbuf, bufPtr, WORD_BUFFER_LENGTH, bufEnd);
                currMaterial->illumModel = atol(tmpbuf);
            }
            break;
        case 'N':
            if (currMaterial)
            {
                switch (bufPtr[1])
                {
                case 's': // Ns - shininess
                    currMaterial->shininess = readFloat();
                    break;
                case 'i': // Ni - refraction index
                    currMaterial->IoR = readFloat();
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
                    currMaterial->diffuse = readRGB();
                    break;
                case 's':		// Ks = specular
                    currMaterial->specular = readRGB();
                    break;
                case 'a':		// Ka = ambience
                    currMaterial->ambient = readRGB();
                    break;
                case 'e':		// Ke = emissive
                    currMaterial->emissive = readRGB();
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
                    currMaterial->roughness = readFloat();
                    break;
                case 'm':
                    currMaterial->metallic = readFloat();
                    break;
                case 's':
                    currMaterial->sheen = readFloat();
                    break;
                case 'c':
                    switch (bufPtr[2])
                    {
                    case 'r':
                        currMaterial->clearcoatRoughness = readFloat();
                        break;
                    case 0:
                        currMaterial->clearcoatThickness = readFloat();
                        break;
                    }
                    break;
                }
            }
            break;
        case 'd': // d - transparency
            if (currMaterial)
                currMaterial->opacity = readFloat();
            break;
        case 'T':
            if (currMaterial)
            {
                switch (bufPtr[1])
                {
                case 'f':		// Tf - Transmitivity
                    currMaterial->transmissionFilter = readRGB();
                    break;
                case 'r':       // Tr, transparency = 1.0-d
                    currMaterial->opacity = (1.f - readFloat());
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
