#include "nbl/asset/interchange/CIESProfileLoader.h"

using namespace nbl;
using namespace asset;

bool CIESProfileLoader::isALoadableFileFormat(system::IFile* _file, const system::logger_opt_ptr logger) const
{
    system::IFile::success_t success;
    std::string versionBuffer(0x45, ' ');
    const auto* fName = _file->getFileName().c_str();
    _file->read(success, versionBuffer.data(), 0, versionBuffer.size());

    if (success)
    {
        for (const auto& it : CIESProfileParser::VALID_SIGNATURES)
            if (versionBuffer.find(it.data()) != std::string::npos)
                return true;

        logger.log("%s: Invalid IES signature for \"%s\" file!", system::ILogger::ELL_ERROR, __FUNCTION__, fName);
    }
    else
        logger.log("%s: Failed to read \"%s\" file!", system::ILogger::ELL_ERROR, __FUNCTION__, fName);

    return false;
}

asset::SAssetBundle CIESProfileLoader::loadAsset(system::IFile* _file, const asset::IAssetLoader::SAssetLoadParams& _params, asset::IAssetLoader::IAssetLoaderOverride* _override, uint32_t _hierarchyLevel)
{
    if (not _file)
    {
        _params.logger.log("%s: Nullptr system::IFile pointer!", system::ILogger::ELL_ERROR, __FUNCTION__);
        return {};
    }

    IAssetLoader::SAssetLoadContext loadContex(_params, _file);
    core::vector<char> data(_file->getSize());
    system::IFile::success_t success;
    const auto* fName = _file->getFileName().c_str();
    _file->read(success, data.data(), 0, _file->getSize());

    if (not success)
    {
        _params.logger.log("%s: Failed to read \"%s\" file!", system::ILogger::ELL_ERROR, __FUNCTION__, fName);
        return {};
    }

    CIESProfileParser parser(data.data(), data.size());
    CIESProfile profile;

    if (not parser.parse(profile)) 
    {
        _params.logger.log("%s: Failed to parse emission profile for \"%s\" file!", system::ILogger::ELL_ERROR, __FUNCTION__, fName);
        return {};
    }

    auto meta = core::make_smart_refctd_ptr<CIESProfileMetadata>(profile);
    core::smart_refctd_ptr<asset::ICPUImageView> cpuImageView;

    if (_params.loaderFlags & E_LOADER_PARAMETER_FLAGS::ELPF_LOAD_METADATA_ONLY) // TODO: create dummy view with no regions for that case
        cpuImageView = _override->findDefaultAsset<ICPUImageView>("nbl/builtin/image_view/dummy2d", loadContex, _hierarchyLevel).first; // note: we could also pass empty content, but this would require adjusting IAssetLoader source to not attempt to use all loaders to find the asset
    else
    {
        const auto optimalResolution = profile.getAccessor().properties.optimalIESResolution;
        cpuImageView = profile.createIESTexture(0.f, false, optimalResolution.x, optimalResolution.y);
    }

    return asset::SAssetBundle(std::move(meta), { core::smart_refctd_ptr(cpuImageView) });
}
