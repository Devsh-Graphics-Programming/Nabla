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
#include <boost/exception/diagnostic_information.hpp>

using namespace nbl;
using namespace nbl::asset;

#include "nbl/asset/utils/waveContext.h"

namespace
{
std::string getLineSnippet(std::string_view text, const int lineNo)
{
    if (lineNo <= 0)
        return {};

    int currentLine = 1;
    size_t lineStart = 0ull;
    while (lineStart <= text.size())
    {
        const auto lineEnd = text.find('\n', lineStart);
        if (currentLine == lineNo)
        {
            const auto count = lineEnd == std::string_view::npos ? text.size() - lineStart : lineEnd - lineStart;
            auto line = std::string(text.substr(lineStart, count));
            if (!line.empty() && line.back() == '\r')
                line.pop_back();
            return line;
        }

        if (lineEnd == std::string_view::npos)
            break;
        lineStart = lineEnd + 1ull;
        currentLine++;
    }

    return {};
}

std::string makeCaretLine(const int columnNo)
{
    if (columnNo <= 0)
        return {};

    return std::string(static_cast<size_t>(columnNo - 1), ' ') + '^';
}

std::string makeWaveFailureContext(
    const nbl::asset::IShaderCompiler::SPreprocessorOptions& preprocessOptions,
    const std::string_view code,
    const char* const phase,
    const std::string_view activeMacroDefinition,
    const char* const fileName,
    const int lineNo,
    const int columnNo)
{
    std::ostringstream stream;
    stream << "Wave preprocessing context:";
    if (!preprocessOptions.sourceIdentifier.empty())
        stream << "\n  source: " << preprocessOptions.sourceIdentifier;
    stream << "\n  phase: " << phase;
    stream << "\n  extra_define_count: " << preprocessOptions.extraDefines.size();
    stream << "\n  source_has_trailing_newline: " << ((!code.empty() && code.back() == '\n') ? "yes" : "no");
    if (!activeMacroDefinition.empty())
        stream << "\n  active_macro_definition: " << activeMacroDefinition;
    if (fileName && fileName[0] != '\0')
        stream << "\n  location: " << fileName << ':' << lineNo << ':' << columnNo;

    const auto snippet = getLineSnippet(code, lineNo);
    if (!snippet.empty() && fileName && preprocessOptions.sourceIdentifier == fileName)
    {
        stream << "\n  snippet: " << snippet;
        const auto caret = makeCaretLine(columnNo);
        if (!caret.empty())
            stream << "\n           " << caret;
    }

    return stream.str();
}
}

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
        const char* phase = "registering built-in macros";
        std::string activeMacroDefinition;
        try
        {
            phase = "registering extra macro definitions";
            for (const auto& define : preprocessOptions.extraDefines)
            {
                activeMacroDefinition = define.identifier;
                activeMacroDefinition.push_back('=');
                activeMacroDefinition.append(define.definition);
                context.add_macro_definition(activeMacroDefinition);
            }
            activeMacroDefinition.clear();

            phase = "expanding translation unit";
            auto stream = std::stringstream();
            for (auto i = context.begin(); i != context.end(); i++)
                stream << i->get_value();
            resolvedString = stream.str();
        }
        catch (boost::wave::preprocess_exception& e)
        {
            const auto failureContext = makeWaveFailureContext(preprocessOptions, code, phase, activeMacroDefinition, e.file_name(), e.line_no(), e.column_no());
            preprocessOptions.logger.log("%s exception caught. %s [%s:%d:%d]\n%s", system::ILogger::ELL_ERROR, e.what(), e.description(), e.file_name(), e.line_no(), e.column_no(), failureContext.c_str());
            preprocessOptions.logger.log("Boost diagnostic information:\n%s", system::ILogger::ELL_ERROR, boost::diagnostic_information(e).c_str());
            return {};
        }
        catch (const boost::exception& e)
        {
            const auto failureContext = makeWaveFailureContext(preprocessOptions, code, phase, activeMacroDefinition, preprocessOptions.sourceIdentifier.data(), 0, 0);
            preprocessOptions.logger.log("Boost exception caught during Wave preprocessing.\n%s", system::ILogger::ELL_ERROR, failureContext.c_str());
            preprocessOptions.logger.log("Boost diagnostic information:\n%s", system::ILogger::ELL_ERROR, boost::diagnostic_information(e).c_str());
            return {};
        }
        catch (const std::exception& e)
        {
            const auto failureContext = makeWaveFailureContext(preprocessOptions, code, phase, activeMacroDefinition, preprocessOptions.sourceIdentifier.data(), 0, 0);
            preprocessOptions.logger.log("std::exception caught during Wave preprocessing. %s\n%s", system::ILogger::ELL_ERROR, e.what(), failureContext.c_str());
            return {};
        }
        catch (...)
        {
            const auto failureContext = makeWaveFailureContext(preprocessOptions, code, phase, activeMacroDefinition, preprocessOptions.sourceIdentifier.data(), 0, 0);
            preprocessOptions.logger.log("Unknown exception caught during Wave preprocessing.\n%s", system::ILogger::ELL_ERROR, failureContext.c_str());
            preprocessOptions.logger.log("Current exception diagnostic information:\n%s", system::ILogger::ELL_ERROR, boost::current_exception_diagnostic_information().c_str());
            return {};
        }

        post(context);

        return resolvedString;
    }
}
