#include "CIESProfile.h"

using namespace nbl;
using namespace asset;

const double& CIESProfile::getIntegral() const 
{
    size_t numHSamples = 2 * getHoriSize();
    size_t numVSamples = 2 * getVertSize();
    double dTheta = core::PI<double>() / numVSamples;
    double dPhi = 2 * core::PI<double>() / numHSamples;
    double dThetaInAngle = MAX_VANGLE / numVSamples;
    double dPhiInAngle = MAX_HANGLE / numHSamples;
    double res = 0;
    for (size_t i = 0; i < numVSamples; i++) {
        for (size_t j = 0; j < numHSamples; j++) {
            res += dPhi * dTheta * sin(dTheta * i) * sample(i * dThetaInAngle, j * dPhiInAngle);
        }
    }
    return res;
}

const double& CIESProfile::sample(double vAngle, double hAngle) const {
    if (vAngle > vAngles.back()) return 0.0;
    hAngle = hAngles.back() == 0.0
        ? 0.0
        : fmod(hAngle,
            hAngles.back()); // when last horizontal angle is zero

    // it's symmetric across all planes

    // bilinear interpolation
    auto lb = [](const core::vector<double>& angles, double angle) -> int {
        int idx = std::upper_bound(std::begin(angles), std::end(angles), angle) -
            std::begin(angles);
        return std::max(idx - 1, 0);
        };
    auto ub = [](const core::vector<double>& angles, double angle) -> int {
        int idx = std::upper_bound(std::begin(angles), std::end(angles), angle) -
            std::begin(angles);
        return std::min(idx, (int)angles.size() - 1);
        };
    int j0 = lb(vAngles, vAngle);
    int j1 = ub(vAngles, vAngle);
    int i0 = lb(hAngles, hAngle);
    int i1 = ub(hAngles, hAngle);
    double uResp = i1 == i0 ? 1.0 : 1.0 / (hAngles[i1] - hAngles[i0]);
    double vResp = j1 == j0 ? 1.0 : 1.0 / (vAngles[j1] - vAngles[j0]);
    double u = (hAngle - hAngles[i0]) * uResp;
    double v = (vAngle - vAngles[j0]) * vResp;
    double s0 = getValue(i0, j0) * (1.0 - v) + getValue(i0, j1) * (v);
    double s1 = getValue(i1, j0) * (1.0 - v) + getValue(i1, j1) * (v);
    return s0 * (1.0 - u) + s1 * u;
}

inline core::vectorSIMDf CIESProfile::octahdronUVToDir(const float& u, const float& v, const float& zAngleDegrees)
{
    core::vectorSIMDf pos = core::vectorSIMDf(2 * (u - 0.5), 2 * (v - 0.5), 0.0);
    float abs_x = core::abs(pos.x), abs_y = core::abs(pos.y);
    pos.z = 1.0 - abs_x - abs_y;
    if (pos.z < 0.0) {
        pos.x = core::sign(pos.x) * (1.0 - abs_y);
        pos.y = core::sign(pos.y) * (1.0 - abs_x);
    }

    // rotate position vector around Z-axis with "zAngleDegrees"
    auto rotateAroundZ = [&]() -> void
    {
        const auto& zAngleRadians = zAngleDegrees * (core::PI<float>() / 180.0f);
        const auto& cosineAngle = std::cos(zAngleRadians);
        const auto& sineAngle = std::sin(zAngleRadians);
 
        pos = core::vectorSIMDf(
            cosineAngle * pos.x - cosineAngle * pos.y,
            sineAngle * pos.x + sineAngle * pos.y,
            pos.z
        );
    };

    if(zAngleDegrees != 0.f)
        rotateAroundZ();

    return core::normalize(pos);
}

inline std::pair<float, float> CIESProfile::sphericalDirToAngles(const core::vectorSIMDf& dir)
{
    float theta = std::acos(dir.z);
    float phi = std::atan2(dir.y, dir.x);
    return { theta, phi < 0 ? phi + 2 * core::PI<float>() : phi };
}

core::smart_refctd_ptr<asset::ICPUImageView> CIESProfile::createCDCTexture(const float& zAngleDegreeRotation, const size_t& width, const size_t& height) const
{
    asset::ICPUImage::SCreationParams imgInfo;
    imgInfo.type = asset::ICPUImage::ET_2D;
    imgInfo.extent.width = width;
    imgInfo.extent.height = height;
    imgInfo.extent.depth = 1u;
    imgInfo.mipLevels = 1u;
    imgInfo.arrayLayers = 1u;
    imgInfo.samples = asset::ICPUImage::ESCF_1_BIT;
    imgInfo.flags = static_cast<asset::IImage::E_CREATE_FLAGS>(0u);
    imgInfo.format = asset::EF_R16_UNORM;
    auto outImg = asset::ICPUImage::create(std::move(imgInfo));

    asset::ICPUImage::SBufferCopy region;
    size_t texelBytesz = asset::getTexelOrBlockBytesize(imgInfo.format);
    size_t bufferRowLength =
        asset::IImageAssetHandlerBase::calcPitchInBlocks(width, texelBytesz);
    region.bufferRowLength = bufferRowLength;
    region.imageExtent = imgInfo.extent;
    region.imageSubresource.baseArrayLayer = 0u;
    region.imageSubresource.layerCount = 1u;
    region.imageSubresource.mipLevel = 0u;
    region.bufferImageHeight = 0u;
    region.bufferOffset = 0u;
    auto buffer = core::make_smart_refctd_ptr<asset::ICPUBuffer>(
        texelBytesz * bufferRowLength * height);

    double maxValue = getMaxValue();
    double maxValueRecip = 1.0 / maxValue;

    double vertInv = 1.0 / height;
    double horiInv = 1.0 / width;
    char* bufferPtr = reinterpret_cast<char*>(buffer->getPointer());
    for (size_t i = 0; i < height; i++) {
        for (size_t j = 0; j < width; j++) {
            auto dir = octahdronUVToDir(j * vertInv, i * horiInv, zAngleDegreeRotation);
            auto [theta, phi] = sphericalDirToAngles(dir);
            double I = sample(core::degrees(theta), core::degrees(phi));
            uint16_t value = static_cast<uint16_t>(
                std::clamp(I * maxValueRecip * 65535.0, 0.0, 65535.0));
            *reinterpret_cast<uint16_t*>(
                bufferPtr + i * bufferRowLength * texelBytesz + j * texelBytesz) =
                value;
        }
    }

    if (!outImg->setBufferAndRegions(std::move(buffer), core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<asset::IImage::SBufferCopy>>(1ull, region)))
        return {};

    ICPUImageView::SCreationParams viewParams = {};
    viewParams.image = outImg;
    viewParams.flags = static_cast<ICPUImageView::E_CREATE_FLAGS>(0);
    viewParams.viewType = IImageView<ICPUImage>::ET_2D;
    viewParams.format = viewParams.image->getCreationParameters().format;
    viewParams.subresourceRange.aspectMask = static_cast<IImage::E_ASPECT_FLAGS>(0);
    viewParams.subresourceRange.levelCount = viewParams.image->getCreationParameters().mipLevels;
    viewParams.subresourceRange.layerCount = 1u;

    return ICPUImageView::create(std::move(viewParams));
}