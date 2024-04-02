#include "CIESProfileParser.h"

using namespace nbl;
using namespace asset;

int CIESProfileParser::getInt(const char* errorMsg)
{
    int in;
    if (ss >> in)
        return in;
    error = true;
    if (!this->errorMsg)
        this->errorMsg = errorMsg;
    return 0;
}

double CIESProfileParser::getDouble(const char* errorMsg)
{
    double in;
    if (ss >> in)
        return in;
    error = true;
    if (!this->errorMsg)
        this->errorMsg = errorMsg;
    return -1.0;
}

bool CIESProfileParser::parse(CIESProfile& result) 
{
    // skip metadata
    std::string line;

    std::getline(ss, line);
    if (line.back() == '\r')
        line.pop_back();

    CIESProfile::Version iesVersion;
    if (line == "IESNA:LM-63-1995")
        iesVersion = CIESProfile::V_1995;
    else if (line == "IESNA:LM-63-2002")
        iesVersion = CIESProfile::V_2002;
    else
    {
        errorMsg = "Unknown IESNA:LM-63 version, the IES input being parsed is invalid!";
        return false;
    }

    while (std::getline(ss, line)) 
    {
        if (line.back() == '\r')
            line.pop_back();
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

    // validate horizontal angles
    {
        const auto& firstHAngle = result.hAnglesOffset = hAngles.front();
        const auto& lastHAngle = hAngles.back();

        if (lastHAngle == 0)
            result.symmetry = CIESProfile::ISOTROPIC;
        else if (lastHAngle == 90)
            result.symmetry = CIESProfile::QUAD_SYMETRIC;
        else if (lastHAngle == 180)
            result.symmetry = CIESProfile::HALF_SYMETRIC;
        else if (lastHAngle == 360)
            result.symmetry = CIESProfile::NO_LATERAL_SYMMET;
        else
        {
            if (firstHAngle == 90 && lastHAngle == 270 && result.version == CIESProfile::V_1995)
            {
                result.symmetry = CIESProfile::OTHER_HALF_SYMMETRIC;

                for (auto& angle : hAngles)
                    angle -= firstHAngle; // patch the profile to HALF_SYMETRIC by shifting [90,270] range to [0, 180]
            }
            else
                return false;
        }
    }

    const double factor = ballastFactor * candelaMultiplier;
    for (int i = 0; i < hSize; i++)
        for (int j = 0; j < vSize; j++)
            result.setValue(i, j, factor * getDouble("intensity value truncated"));
    
    result.maxValue = *std::max_element(std::begin(result.data), std::end(result.data));

    return !error;
}