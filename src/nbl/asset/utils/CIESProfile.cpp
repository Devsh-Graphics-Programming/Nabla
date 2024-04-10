#include "CIESProfile.h"

#include <atomic>
#include "nbl/asset/filters/CBasicImageFilterCommon.h"

using namespace nbl;
using namespace asset;

const CIESProfile::IES_STORAGE_FORMAT CIESProfile::sample(IES_STORAGE_FORMAT theta, IES_STORAGE_FORMAT phi) const 
{
    auto wrapPhi = [&](const IES_STORAGE_FORMAT& _phi) -> IES_STORAGE_FORMAT
    {
        constexpr auto M_HALF_PI =core::HALF_PI<double>();
        constexpr auto M_TWICE_PI = core::PI<double>() * 2.0;

        switch (symmetry)
        {
            case ISOTROPIC: //! axial symmetry
                return 0.0;
            case QUAD_SYMETRIC: //! phi MIRROR_REPEAT wrap onto [0, 90] degrees range
            {
                float wrapPhi = abs(_phi); //! first MIRROR

                if (wrapPhi > M_HALF_PI) //! then REPEAT
                    wrapPhi = std::clamp<IES_STORAGE_FORMAT>(M_HALF_PI - (wrapPhi - M_HALF_PI), 0, M_HALF_PI);

                return wrapPhi; //! eg. maps (in degrees) 91,269,271 -> 89 and 179,181,359 -> 1
            }
            case HALF_SYMETRIC: //! phi MIRROR wrap onto [0, 180] degrees range
            case OTHER_HALF_SYMMETRIC:
                return abs(_phi); //! eg. maps (in degress) 181 -> 179 or 359 -> 1
            case NO_LATERAL_SYMMET: //! plot onto whole (in degress) [0, 360] range
            {
                if (_phi < 0)
                    return _phi + M_TWICE_PI;
                else
                    return _phi;
            }
            default:
                assert(false);
        }
    };

    const float vAngle = core::degrees(theta), hAngle = core::degrees(wrapPhi(phi));

    assert(vAngle >= 0.0 && vAngle <= 180.0);
    assert(hAngle >= 0.0 && hAngle <= 360.0);

    if (vAngle > vAngles.back())
        return 0.0;

    // bilinear interpolation
    auto lb = [](const core::vector<double>& angles, double angle) -> size_t
    {
        assert(!angles.empty());
        const size_t idx = std::upper_bound(std::begin(angles), std::end(angles), angle) - std::begin(angles);
        return (size_t)std::max((int64_t)idx - 1, (int64_t)0);
    };

    auto ub = [](const core::vector<double>& angles, double angle) -> size_t
    {
        assert(!angles.empty());
        const size_t idx = std::upper_bound(std::begin(angles), std::end(angles), angle) - std::begin(angles);
        return std::min<size_t>(idx, angles.size() - 1);
    };

    const size_t j0 = lb(vAngles, vAngle);
    const size_t j1 = ub(vAngles, vAngle);
    const size_t i0 = symmetry == ISOTROPIC ? 0 : lb(hAngles, hAngle);
    const size_t i1 = symmetry == ISOTROPIC ? 0 : ub(hAngles, hAngle);

    double uResp = i1 == i0 ? 1.0 : 1.0 / (hAngles[i1] - hAngles[i0]);
    double vResp = j1 == j0 ? 1.0 : 1.0 / (vAngles[j1] - vAngles[j0]);

    double u = (hAngle - hAngles[i0]) * uResp;
    double v = (vAngle - vAngles[j0]) * vResp;

    double s0 = getCandelaValue(i0, j0) * (1.0 - v) + getCandelaValue(i0, j1) * (v);
    double s1 = getCandelaValue(i1, j0) * (1.0 - v) + getCandelaValue(i1, j1) * (v);

    return s0 * (1.0 - u) + s1 * u;
}

inline core::vectorSIMDf CIESProfile::octahdronUVToDir(const float& u, const float& v)
{
    core::vectorSIMDf pos = core::vectorSIMDf(2 * (u - 0.5), 2 * (v - 0.5), 0.0);
    float abs_x = core::abs(pos.x), abs_y = core::abs(pos.y);
    pos.z = 1.0 - abs_x - abs_y;
    if (pos.z < 0.0) {
        pos.x = core::sign(pos.x) * (1.0 - abs_y);
        pos.y = core::sign(pos.y) * (1.0 - abs_x);
    }

    return core::normalize(pos);
}


inline std::pair<float, float> CIESProfile::sphericalDirToRadians(const core::vectorSIMDf& dir)
{
    const float theta = std::acos(std::clamp<float>(dir.z, -1.f, 1.f));
    const float phi = std::atan2(dir.y, dir.x);

    return { theta, phi };
}

template<class ExecutionPolicy>
core::smart_refctd_ptr<asset::ICPUImageView> CIESProfile::createIESTexture(ExecutionPolicy&& policy, asset::IAssetLoader::IAssetLoaderOverride* aOverride, const std::string key, const float flatten, const size_t width, const size_t height) const
{
    const bool inFlattenDomain = flatten >= 0.0 && flatten < 1.0; // [0, 1) range for blend equation, 1 is invalid
    
    assert(inFlattenDomain);
    assert(aOverride);

    asset::ICPUImage::SCreationParams imgInfo;
    imgInfo.type = asset::ICPUImage::ET_2D;
    imgInfo.extent.width = width;
    imgInfo.extent.height = height;
    imgInfo.extent.depth = 1u;
    imgInfo.mipLevels = 1u;
    imgInfo.arrayLayers = 1u;
    imgInfo.samples = asset::ICPUImage::ESCF_1_BIT;
    imgInfo.flags = static_cast<asset::IImage::E_CREATE_FLAGS>(0u);
    imgInfo.format = IES_TEXTURE_STORAGE_FORMAT;
    auto outImg = asset::ICPUImage::create(std::move(imgInfo));

    asset::ICPUImage::SBufferCopy region;
    constexpr auto texelBytesz = asset::getTexelOrBlockBytesize<IES_TEXTURE_STORAGE_FORMAT>();
    const size_t bufferRowLength = asset::IImageAssetHandlerBase::calcPitchInBlocks(width, texelBytesz);
    region.bufferRowLength = bufferRowLength;
    region.imageExtent = imgInfo.extent;
    region.imageSubresource.baseArrayLayer = 0u;
    region.imageSubresource.layerCount = 1u;
    region.imageSubresource.mipLevel = 0u;
    region.bufferImageHeight = 0u;
    region.bufferOffset = 0u;

    auto buffer = core::make_smart_refctd_ptr<asset::ICPUBuffer>(texelBytesz * bufferRowLength * height);

    if (!outImg->setBufferAndRegions(std::move(buffer), core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<asset::IImage::SBufferCopy>>(1ull, region)))
        return {};

    //! Generate 2D IES grid data CandelaPower Distribution Curve texture can be created from
    {
        const auto& creationParams = outImg->getCreationParameters();

        CFillImageFilter::state_type state;
        state.outImage = outImg.get();
        state.subresource.aspectMask = static_cast<IImage::E_ASPECT_FLAGS>(0);
        state.subresource.baseArrayLayer = 0u;
        state.subresource.layerCount = 1u;
        state.outRange.extent = creationParams.extent;

        const IImageFilter::IState::ColorValue::WriteMemoryInfo wInfo(creationParams.format, outImg->getBuffer()->getPointer());

        const double maxValue = getMaxCandelaValue();
        const double maxValueRecip = 1.0 / maxValue;

        const double vertInv = 1.0 / height;
        const double horiInv = 1.0 / width;

        auto fill = [&](uint32_t blockArrayOffset, core::vectorSIMDu32 position) -> void
        {
            const auto dir = octahdronUVToDir(((float)position.x + 0.5) * vertInv, ((float)position.y + 0.5) * horiInv);
            const auto [theta, phi] = sphericalDirToRadians(dir);
            const auto intensity = sample(theta, phi);
            const auto value = intensity * maxValueRecip;
            const auto blendV = value * (1.0 - flatten) + avgEmmision * flatten; //! blend the IES texture with "flatten"

            asset::IImageFilter::IState::ColorValue color;
            //asset::encodePixels<CIESProfile::IES_TEXTURE_STORAGE_FORMAT>(color.asDouble, &blendV); TODO: FIX THIS ENCODE, GIVES ARTIFACTS
            const uint16_t encodeV = static_cast<uint16_t>(std::clamp(blendV * UI16_MAX_D, 0.0, UI16_MAX_D));
            *color.asUShort = encodeV;
            color.writeMemory(wInfo, blockArrayOffset);
        };

        CBasicImageFilterCommon::clip_region_functor_t clip(state.subresource, state.outRange, creationParams.format);
        const auto& regions = outImg->getRegions(state.subresource.mipLevel);
        CBasicImageFilterCommon::executePerRegion(std::forward<ExecutionPolicy>(policy), outImg.get(), fill, regions.begin(), regions.end(), clip);
    }

    ICPUImageView::SCreationParams viewParams = {};
    viewParams.image = outImg;
    viewParams.flags = static_cast<ICPUImageView::E_CREATE_FLAGS>(0);
    viewParams.viewType = IImageView<ICPUImage>::ET_2D;
    viewParams.format = viewParams.image->getCreationParameters().format;
    viewParams.subresourceRange.aspectMask = static_cast<IImage::E_ASPECT_FLAGS>(0);
    viewParams.subresourceRange.levelCount = viewParams.image->getCreationParameters().mipLevels;
    viewParams.subresourceRange.layerCount = 1u;

    auto imageView = ICPUImageView::create(std::move(viewParams));
    const auto aHash = key + "?flatten=" + std::to_string(flatten);
    asset::IAssetLoader::SAssetLoadContext ctx = { {}, nullptr };
    SAssetBundle bundle = asset::SAssetBundle(nullptr, { core::smart_refctd_ptr(imageView) });
    aOverride->insertAssetIntoCache(bundle, aHash, ctx, 0u);

    return core::smart_refctd_ptr(imageView);
}

//! Explicit instantiations
template core::smart_refctd_ptr<asset::ICPUImageView> CIESProfile::createIESTexture(const std::execution::sequenced_policy&, asset::IAssetLoader::IAssetLoaderOverride*, const std::string, const float, const size_t, const size_t) const;
template core::smart_refctd_ptr<asset::ICPUImageView> CIESProfile::createIESTexture(const std::execution::parallel_policy&, asset::IAssetLoader::IAssetLoaderOverride*, const std::string, const float, const size_t, const size_t) const;
template core::smart_refctd_ptr<asset::ICPUImageView> CIESProfile::createIESTexture(const std::execution::parallel_unsequenced_policy&, asset::IAssetLoader::IAssetLoaderOverride*, const std::string, const float, const size_t, const size_t) const;

core::smart_refctd_ptr<asset::ICPUImageView> CIESProfile::createIESTexture(asset::IAssetLoader::IAssetLoaderOverride* aOverride, const std::string key, const float flatten, const size_t width, const size_t height) const
{
    return createIESTexture(std::execution::seq, aOverride, key, flatten, width, height);
}