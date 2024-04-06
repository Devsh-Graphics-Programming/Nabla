// Copyright (C) 2018-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_C_IES_PROFILE_PARSER_H_INCLUDED__
#define __NBL_ASSET_C_IES_PROFILE_PARSER_H_INCLUDED__

#include "nbl/asset/utils/CIESProfile.h"

namespace nbl
{
    namespace asset
    {
        class CIESProfileParser 
        {
            public:
                CIESProfileParser(char* buf, size_t size) { ss << std::string(buf, size); }

                const char* getErrorMsg() const { return errorMsg; }

                /*
                    An IES file comes with following data

                    ############################
                    ● IESNA:LM-63-[2002 or 1995]
                    ● [Keyword 1] Keyword data 
                    ● [Keyword 2] Keyword data 
                    ● [Keyword 3] Keyword data 
                    ● : 
                    ● [Keyword n] Keyword data 
                    ● TILT=<filename> or INCLUDE or NONE

                    if(TILT == INCLUDE)
                        ● <lamp to luminaire geometry>
                        ● <number of tilt angles> 
                        ● <angles> 
                        ● <multiplying factors> 

                    ● <number of lamps> <lumens per lamp> <candelamultiplier> <number of vertical angles> <number of horizontal angles> <photometric type> <units type> <width> <length> <height> 
                    ● <ballast factor> <future use> <input watts> 
                    ● <vertical angles> 
                    ● <horizontal angles> 
                    ● <candela values for all vertical angles at 1st horizontal angle> 
                    ● <candela values for all vertical angles as 2nd horizontal angle>
                    ● :
                    ● <candela values for all vertical angles at last horizontal>
                    #############################################################

                    The IES (Illuminating Engineering Society) format provides photometric data for light fixtures.
                    Candela values along with proper vertical and horizontal angles describe the intensity 
                    and distribution of light emitted by the fixture in specific directions.

                    Candela value represents the luminous intensity in a particular direction. 
                    It tells how much light is emitted by the fixture in that direction. 
                    The horizontal and vertical angles provided by IES in degrees specify 
                    the spread of light in those respective planes.

                    To measure these values, photometric testing is conducted using specialized equipment called a goniophotometer. 
                    A goniophotometer allows precise measurement of light distribution characteristics, 
                    including candela values at various angles.
                    
                    One can think of candela values provided by IES as 2D texture grid where single texel 
                    in the grid represents a candela value at position [<index of horizontal angle>][<index of vertical angle>]
                    
                    Therefore to obtain illuminant target candela value for vertical angle θ and horizontal angle φ you need to access 
                    
                    ● <2D texture grid of candela values pointer>[<<θ angles vector>.size()> * <index of θ angle> + <index of φ angle>]

                    For more details see @CIESProfile class implementation

                    Also note that

                    ● a point (cos(φ), sin(φ)) on XY IES plane is denoted in LH cartesian coordinate system
                    ● Nabla IES loader & plotter perform bi-linear interpolation as an extra feature to find candela value for an angle being in-between 2 angles in the measured domain range provided by IES
                */

                bool parse(CIESProfile& result);

            private:
                int getInt(const char* errorMsg);
                double getDouble(const char* errorMsg);

                bool error{ false };
                const char* errorMsg{ nullptr };
                std::stringstream ss;
        };
    }
}

#endif // __NBL_ASSET_C_IES_PROFILE_PARSER_H_INCLUDED__