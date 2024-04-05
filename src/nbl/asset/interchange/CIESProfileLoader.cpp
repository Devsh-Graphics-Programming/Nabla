#include "nbl/asset/interchange/CIESProfileLoader.h"

using namespace nbl;
using namespace asset;

asset::SAssetBundle
CIESProfileLoader::loadAsset(io::IReadFile* _file,
    const asset::IAssetLoader::SAssetLoadParams& _params,
    asset::IAssetLoader::IAssetLoaderOverride* _override,
    uint32_t _hierarchyLevel) {
    if (!_file)
        return {};

    IAssetLoader::SAssetLoadContext loadContex(_params, _file);
    core::vector<char> data(_file->getSize());
    _file->read(data.data(), _file->getSize());

    CIESProfileParser parser(data.data(), data.size());
    CIESProfile profile;

    if (!parser.parse(profile)) 
    {
        os::Printer::log("ERROR: Emission profile parsing error: " + std::string(parser.getErrorMsg()), ELL_ERROR);
        return {};
    }

    auto meta = core::make_smart_refctd_ptr<CIESProfileMetadata>(profile);
    core::smart_refctd_ptr<asset::ICPUImageView> cpuImageView;

    if (_params.loaderFlags & E_LOADER_PARAMETER_FLAGS::ELPF_LOAD_METADATA_ONLY)
        cpuImageView = _override->findDefaultAsset<ICPUImageView>("nbl/builtin/image_view/dummy2d", loadContex, _hierarchyLevel).first; // note: we could also pass empty content, but this would require adjusting IAssetLoader source to not attempt to use all loaders to find the asset
    else
        cpuImageView = profile.createIESTexture();

    return asset::SAssetBundle(std::move(meta), { core::smart_refctd_ptr(cpuImageView) });
}
