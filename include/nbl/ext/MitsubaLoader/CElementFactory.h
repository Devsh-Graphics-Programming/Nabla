// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __I_ELEMENT_FACTORY_H_INCLUDED__
#define __I_ELEMENT_FACTORY_H_INCLUDED__

#include "nbl/ext/MitsubaLoader/CElementSensor.h"
#include "nbl/ext/MitsubaLoader/CElementIntegrator.h"
#include "nbl/ext/MitsubaLoader/CElementShape.h"

namespace nbl
{
namespace ext
{
namespace MitsubaLoader
{
class ParserManager;

class CElementFactory
{
public:
    using return_type = std::pair<IElement*, std::string>;
    using element_creation_func = return_type (*)(const char**, ParserManager*);
    const static core::unordered_map<std::string, std::pair<element_creation_func, bool>, core::CaseInsensitiveHash, core::CaseInsensitiveEquals> createElementTable;

    //constructs certain elements based on element's name and its attributes
    template<class element_type>
    static return_type createElement(const char** _atts, ParserManager* _util);
    //
    static return_type processAlias(const char** _atts, ParserManager* _util);
    static return_type processRef(const char** _atts, ParserManager* _util);
};

}
}
}

#endif