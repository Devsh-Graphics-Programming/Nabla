#include "CIESProfileParser.h"

using namespace nbl;
using namespace asset;

int CIESProfileParser::getInt(const char* errorMsg)
{
    return readStream<int>(errorMsg);
}

double CIESProfileParser::getDouble(const char* errorMsg)
{
    return readStream<double>(errorMsg);
}

bool CIESProfileParser::parse(CIESProfile& result) 
{
    auto removeTrailingWhiteChars = [](std::string& str) -> void
    {
        if (std::isspace(str.back()))
        {
            auto it = str.rbegin();
            while (it != str.rend() && std::isspace(static_cast<unsigned char>(*it)))
                ++it;

            str.erase(it.base(), str.end());
        }
    };

    // skip metadata
    std::string line;

    std::getline(ss, line);
    removeTrailingWhiteChars(line);

    CIESProfile::Version iesVersion;

    if (line.find(SIG_LM63_1995.data()) != std::string::npos)
        iesVersion = CIESProfile::V_1995;
    else if (line.find(SIG_LM63_2002.data()) != std::string::npos)
        iesVersion = CIESProfile::V_2002;
    else if (line.find(SIG_IESNA91.data()) != std::string::npos)
        iesVersion = CIESProfile::V_1995;
    else if (line.find(SIG_ERCO_LG.data()) != std::string::npos)
        iesVersion = CIESProfile::V_1995;
    else
    {
        errorMsg = "Unknown IESNA:LM-63 version, the IES input being parsed is invalid!";
        return false;
    }

     while (std::getline(ss, line)) 
    {
        removeTrailingWhiteChars(line);
        if (line == "TILT=INCLUDE" || line == "TILT=NONE")
            break;
    }

    if (line == "TILT=INCLUDE") {
        double lampToLuminaire = getDouble("lampToLuminaire truncated");
        int numTilt = getDouble("numTilt truncated");
        for (int i = 0; i < numTilt; i++)
            getDouble("tilt angle truncated");
        for (int i = 0; i < numTilt; i++)
            getDouble("tilt multiplying factor truncated");
    }
    else if (line != "TILT=NONE") {
        errorMsg = "TILT not specified";
        return false;
    }

    int numLamps = getInt("numLamps truncated");
    double lumensPerLamp = getDouble("lumensPerLamp truncated");
    double candelaMultiplier = getDouble("candelaMultiplier truncated");
    int vSize = getInt("vSize truncated");
    int hSize = getInt("hSize truncated");

    int type_ = getInt("type truncated");
    if (error)
        return false;
    if (type_ <= 0 || type_ > 3) {
        errorMsg = "unrecognized type";
        return false;
    }
    CIESProfile::PhotometricType type =
        static_cast<CIESProfile::PhotometricType>(type_);
    if (type != CIESProfile::PhotometricType::TYPE_C) {
        errorMsg = "Only type C is supported for now";
        return false;
    }

    int unitsType = getInt("unitsType truncated");
    double width = getDouble("width truncated"),
    length = getDouble("length truncated"),
    height = getDouble("height truncated"),
    ballastFactor = getDouble("ballastFactor truncated"),
    reserved = getDouble("reserved truncated"),
    inputWatts = getDouble("inputWatts truncated");

    if (error)
        return false;

    result = CIESProfile(type, hSize, vSize);
    result.version = iesVersion;

    if (vSize < 2)
        return false;

    {
        const uint32_t maxDimMeasureSize = core::max(hSize, vSize);
        result.optimalIESResolution = decltype(result.optimalIESResolution){ maxDimMeasureSize, maxDimMeasureSize };
        result.optimalIESResolution *= 2u; // safe bias for our bilinear interpolation to work nicely and increase resolution of a profile
    }

    auto& vAngles = result.vAngles;
    for (int i = 0; i < vSize; i++) {
        vAngles[i] = getDouble("vertical angle truncated");
    }
    if (!std::is_sorted(vAngles.begin(), vAngles.end())) {
        errorMsg = "Vertical angles should be sorted";
        return false;
    }
    if (vAngles[0] != 0.0 && vAngles[0] != 90.0) {
        errorMsg = "First vertical angle must be 0 or 90 in type C";
        return false;
    }
    if (vAngles[vSize - 1] != 90.0 && vAngles[vSize - 1] != 180.0) {
        errorMsg = "Last vertical angle must be 90 or 180 in type C";
        return false;
    }

    auto& hAngles = result.hAngles;
    for (int i = 0; i < hSize; i++) {
        hAngles[i] = getDouble("horizontal angle truncated");
        if (i != 0 && hAngles[i - 1] > hAngles[i])
            return false; // Angles should be sorted
    }

    float fluxMultiplier = 1.0f;
    {
        const auto firstHAngle = hAngles.front();
        const auto lastHAngle = hAngles.back();

        if (lastHAngle == 0)
            result.symmetry = CIESProfile::ISOTROPIC;
        else if (lastHAngle == 90)
        {
            result.symmetry = CIESProfile::QUAD_SYMETRIC;
            fluxMultiplier = 4.0;
        }
        else if (lastHAngle == 180)
        {
            result.symmetry = CIESProfile::HALF_SYMETRIC;
            fluxMultiplier = 2.0;
        }
        else if (lastHAngle == 360)
            result.symmetry = CIESProfile::NO_LATERAL_SYMMET;
        else
        {
            if (firstHAngle == 90 && lastHAngle == 270 && result.version == CIESProfile::V_1995)
            {
                result.symmetry = CIESProfile::OTHER_HALF_SYMMETRIC;
                fluxMultiplier = 2.0;

                for (auto& angle : hAngles)
                    angle -= firstHAngle; // patch the profile to HALF_SYMETRIC by shifting [90,270] range to [0, 180]
            }
            else
                return false;
        }
    }

    {
        const double factor = ballastFactor * candelaMultiplier;
        for (int i = 0; i < hSize; i++)
            for (int j = 0; j < vSize; j++)
                result.setCandelaValue(i, j, factor * getDouble("intensity value truncated"));
    }

    float totalEmissionIntegral = 0.0, nonZeroEmissionDomainSize = 0.0;
    constexpr auto FULL_SOLID_ANGLE = 4.0f * core::PI<float>();

    const auto H_ANGLES_I_RANGE = result.symmetry != CIESProfile::ISOTROPIC ? result.hAngles.size() - 1 : 1;
    const auto V_ANGLES_I_RANGE = result.vAngles.size() - 1;

    for (size_t i = 0; i < H_ANGLES_I_RANGE; i++)
    {
        const float dPhiRad = result.symmetry != CIESProfile::ISOTROPIC ? (hAngles[i + 1] - hAngles[i]) : core::PI<float>() * 2.0f;

        for (size_t j = 0; j < V_ANGLES_I_RANGE; j++)
        {
            const auto candelaValue = result.getCandelaValue(i, j);

            // interpolate candela value spanned onto a solid angle
            const auto candelaAverage = result.symmetry != CIESProfile::ISOTROPIC ? 
                  0.25f * (candelaValue + result.getCandelaValue(i + 1, j) + result.getCandelaValue(i, j + 1) + result.getCandelaValue(i + 1, j + 1))
                : 0.5f * (candelaValue + result.getCandelaValue(i, j + 1));

            if (result.maxCandelaValue < candelaValue)
                result.maxCandelaValue = candelaValue;

            const float thetaRad = core::radians<float>(result.vAngles[j]);
            const float cosLo = std::cos(core::radians<float>(result.vAngles[j]));
            const float cosHi = std::cos(core::radians<float>(result.vAngles[j + 1]));

            const auto differentialSolidAngle = dPhiRad*(cosLo - cosHi);
            const auto integralV = candelaAverage * differentialSolidAngle;

            if (integralV > 0.0)
            {
                totalEmissionIntegral += integralV;
                nonZeroEmissionDomainSize += differentialSolidAngle;
            }
        }
    }

    nonZeroEmissionDomainSize = std::clamp<float>(nonZeroEmissionDomainSize, 0.0, FULL_SOLID_ANGLE);
    if (nonZeroEmissionDomainSize <= 0) // protect us from division by 0 (just in case, we should never hit it)
        return false;

    result.avgEmmision = totalEmissionIntegral / static_cast<decltype(totalEmissionIntegral)>(nonZeroEmissionDomainSize);
    result.totalEmissionIntegral = totalEmissionIntegral * fluxMultiplier; // we use fluxMultiplier to calculate final total emission for case where we have some symmetry between planes (fluxMultiplier is 1.0f if ISOTROPIC or NO_LATERAL_SYMMET because they already have correct total emission integral calculated), also note it doesn't affect average emission at all

    return !error;
}