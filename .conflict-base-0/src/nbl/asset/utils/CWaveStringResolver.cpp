// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

//! wave token preprocessing with isolated translation unit optimization
/*
    the source has separate optimization flags because of how boost spirit (header only dep) the wave relies on is slow in Debug builds, ABI related 
    options remain and there is no mismatch, we force agressive inlining and optimizations mostly regardless build configuration by default
*/

/*
    Arek leaving thoughts, TODO:
    
    in NBL_WAVE_STRING_RESOLVER_TU_DEBUG_OPTIMISATION mode enabled -> here in this TU do

    #define _ITERATOR_DEBUG_LEVEL 0
    #define _HAS_ITERATOR_DEBUGGING 0

    and allow Nabla to mismatch debug iterator *on purpose* by

    #define _ALLOW_ITERATOR_DEBUG_LEVEL_MISMATCH 

    in Debug/RWDI

    then make preprocess full C API with raw in/out pointers and bytes out pointer,
    with mismtach we must be very careful about memory ownership as STL stuff will have
    different struct layouts and its easy to make a crash, we will have extra memcpy and
    deallocation but as a trade each config will have almost the same preprocessing perf
    which matters for our NSC integration

    then we can think to make use of existing shader cache and maybe consider HLSL PCH
    which NSC would inject into each input

    NOTE: this approach allows to do all in single Nabla module, no extra proxy/fake shared DLL needed!
    NOTE: yep I know I have currently a callback for which context size will differ accross TUs afterwards but will think about it

    or ignore it and take care of NSC special target creating global HLSL PCH injected into each registered input
*/

#include "nabla.h"

using namespace nbl;
using namespace nbl::asset;

#include "nbl/asset/utils/waveContext.h"

namespace nbl::wave
{
    std::string preprocess(std::string& code, const nbl::asset::IShaderCompiler::SPreprocessorOptions& preprocessOptions, bool withCaching, std::function<void(nbl::wave::context&)> post)
    {
        nbl::wave::context context(code.begin(), code.end(), preprocessOptions.sourceIdentifier.data(), { preprocessOptions });
        context.set_caching(withCaching);
        context.add_macro_definition("__HLSL_VERSION");
        context.add_macro_definition("__SPIRV_MAJOR_VERSION__=" + std::to_string(IShaderCompiler::getSpirvMajor(preprocessOptions.targetSpirvVersion)));
        context.add_macro_definition("__SPIRV_MINOR_VERSION__=" + std::to_string(IShaderCompiler::getSpirvMinor(preprocessOptions.targetSpirvVersion)));

        // instead of defining extraDefines as "NBL_GLSL_LIMIT_MAX_IMAGE_DIMENSION_1D 32768", 
        // now define them as "NBL_GLSL_LIMIT_MAX_IMAGE_DIMENSION_1D=32768" 
        // to match boost wave syntax
        // https://www.boost.org/doc/libs/1_82_0/libs/wave/doc/class_reference_context.html#:~:text=Maintain%20defined%20macros-,add_macro_definition,-bool%20add_macro_definition

        // preprocess
        core::string resolvedString;
        try
        {
            for (const auto& define : preprocessOptions.extraDefines)
            {
                std::string macroDefinition(define.identifier);
                macroDefinition.push_back('=');
                macroDefinition.append(define.definition);
                context.add_macro_definition(macroDefinition);
            }

            auto stream = std::stringstream();
            for (auto i = context.begin(); i != context.end(); i++)
                stream << i->get_value();
            resolvedString = stream.str();
        }
        catch (boost::wave::preprocess_exception& e)
        {
            preprocessOptions.logger.log("%s exception caught. %s [%s:%d:%d]",system::ILogger::ELL_ERROR,e.what(),e.description(),e.file_name(),e.line_no(),e.column_no());
            return {};
        }
        catch (...)
        {
            preprocessOptions.logger.log("Unknown exception caught!",system::ILogger::ELL_ERROR);
            return {};
        }

        post(context);

        return resolvedString;
    }
}
