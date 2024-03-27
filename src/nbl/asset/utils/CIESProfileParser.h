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