// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

//! wave token preprocessing with isolated translation unit optimization
/*
    the source has separate optimization flags because of how boost spirit (header only dep) the wave relies on is slow in Debug builds, ABI related 
    options remain and there is no mismatch, we force agressive inlining and optimizations mostly regardless build configuration by default
*/

#include "nabla.h"
#include <boost/exception/diagnostic_information.hpp>
#include <boost/wave/util/insert_whitespace_detection.hpp>
#include <algorithm>
#include <optional>

using namespace nbl;
using namespace nbl::asset;

#include "nbl/asset/utils/waveContext.h"

namespace
{
constexpr size_t kWaveFailureLogOutputTailMaxChars = 4096ull;
constexpr size_t kWaveFailureLogOutputTailMaxLines = 16ull;
constexpr size_t kWaveFailureLogTokenPreviewMaxChars = 160ull;

struct WaveRenderProgress
{
    core::string output;
    std::string previousFile;
    int previousLine = 0;
    bool hasPreviousToken = false;
    bool previousWasExplicitWhitespace = false;
    size_t emittedTokenCount = 0ull;
};

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

size_t countLogicalLines(const std::string_view text)
{
    if (text.empty())
        return 0ull;

    size_t lines = static_cast<size_t>(std::count(text.begin(), text.end(), '\n'));
    if (text.back() != '\n')
        ++lines;
    return lines;
}

std::string truncateEscapedPreview(std::string value, const size_t maxChars)
{
    if (value.size() <= maxChars)
        return value;

    if (maxChars <= 3ull)
        return value.substr(0ull, maxChars);

    value.resize(maxChars - 3ull);
    value += "...";
    return value;
}

std::string indentMultiline(std::string_view text, std::string_view indent)
{
    if (text.empty())
        return {};

    std::string out;
    out.reserve(text.size() + indent.size() * 4ull);

    size_t lineStart = 0ull;
    while (lineStart < text.size())
    {
        out.append(indent.data(), indent.size());

        const auto lineEnd = text.find('\n', lineStart);
        if (lineEnd == std::string_view::npos)
        {
            out.append(text.data() + lineStart, text.size() - lineStart);
            break;
        }

        out.append(text.data() + lineStart, lineEnd - lineStart + 1ull);
        lineStart = lineEnd + 1ull;
    }

    return out;
}

std::string makeFailureLogOutputTail(std::string_view text)
{
    if (text.empty())
        return {};

    size_t start = text.size();
    size_t chars = 0ull;
    size_t newlines = 0ull;
    while (start > 0ull)
    {
        --start;
        ++chars;
        if (text[start] == '\n')
        {
            ++newlines;
            if (newlines > kWaveFailureLogOutputTailMaxLines)
            {
                ++start;
                break;
            }
        }

        if (chars >= kWaveFailureLogOutputTailMaxChars)
            break;
    }

    std::string tail;
    if (start > 0ull)
        tail = "[truncated]\n";
    tail.append(text.data() + start, text.size() - start);
    return tail;
}

std::string makeWaveFailureContext(
    const nbl::asset::IShaderCompiler::SPreprocessorOptions& preprocessOptions,
    const nbl::wave::context& context,
    const WaveRenderProgress& renderProgress,
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
        stream << "\n  source: " << nbl::wave::detail::escape_control_chars(preprocessOptions.sourceIdentifier);
    stream << "\n  phase: " << phase;
    stream << "\n  extra_define_count: " << preprocessOptions.extraDefines.size();
    stream << "\n  source_length_bytes: " << code.size();
    stream << "\n  source_has_trailing_newline: " << ((!code.empty() && code.back() == '\n') ? "yes" : "no");
    stream << "\n  include_depth: " << context.get_iteration_depth();
    stream << "\n  current_include_spelling: " << nbl::wave::detail::escape_control_chars(context.get_current_relative_filename());
    stream << "\n  current_directory: " << nbl::wave::detail::escape_control_chars(context.get_current_directory().string());
#if BOOST_WAVE_SUPPORT_PRAGMA_ONCE != 0
    stream << "\n  current_include_absolute_path: " << nbl::wave::detail::escape_control_chars(context.get_current_filename());
#endif
    if (!activeMacroDefinition.empty())
        stream << "\n  active_macro_definition: " << nbl::wave::detail::escape_control_chars(activeMacroDefinition);
    if (fileName && fileName[0] != '\0')
        stream << "\n  location: " << nbl::wave::detail::escape_control_chars(fileName) << ':' << lineNo << ':' << columnNo;
    stream << "\n  emitted_output_bytes: " << renderProgress.output.size();
    stream << "\n  emitted_output_lines: " << countLogicalLines(renderProgress.output);
    stream << "\n  emitted_token_count: " << renderProgress.emittedTokenCount;
    const auto snippet = getLineSnippet(code, lineNo);
    if (!snippet.empty() && fileName && preprocessOptions.sourceIdentifier == fileName)
    {
        stream << "\n  snippet: " << snippet;
        const auto caret = makeCaretLine(columnNo);
        if (!caret.empty())
            stream << "\n           " << caret;
    }

    const auto outputTail = makeFailureLogOutputTail(renderProgress.output);
    if (!outputTail.empty())
        stream << "\n  partial_output_tail:\n" << indentMultiline(outputTail, "    ");

    return stream.str();
}

template<typename TokenT>
bool isWhitespaceLikeToken(const TokenT& token)
{
    using namespace boost::wave;

    const auto id = token_id(token);
    return id == T_NEWLINE || id == T_GENERATEDNEWLINE || id == T_CONTLINE || IS_CATEGORY(token, WhiteSpaceTokenType);
}

void renderPreprocessedOutput(nbl::wave::context& context, WaveRenderProgress& renderProgress)
{
    using namespace boost::wave;

    util::insert_whitespace_detection whitespace(true);
    auto& perfStats = nbl::wave::detail::perf_stats();
    auto it = context.begin();
    const auto end = context.end();
    while (it != end)
    {
        std::optional<nbl::wave::detail::ScopedPerfTimer> loopBodyTimer;
        if (perfStats.enabled)
            loopBodyTimer.emplace(perfStats.loopBodyTime);

        const auto& token = *it;
        const auto id = token_id(token);
        if (id != T_EOF && id != T_EOI)
        {
            std::optional<nbl::wave::detail::ScopedPerfTimer> tokenTimer;
            if (perfStats.enabled)
                tokenTimer.emplace(perfStats.tokenHandlingTime);

            const auto explicitWhitespace = isWhitespaceLikeToken(token);
            const auto& position = token.get_position();
            const auto& value = token.get_value();

            const auto currentLine = position.get_line();
            const auto& currentFile = position.get_file();

            if (renderProgress.hasPreviousToken && !explicitWhitespace)
            {
                bool movedToNewLogicalLine = currentLine > renderProgress.previousLine;
                if (!movedToNewLogicalLine)
                {
                    movedToNewLogicalLine =
                        renderProgress.previousFile.size() != currentFile.size() ||
                        !std::equal(currentFile.begin(), currentFile.end(), renderProgress.previousFile.begin());
                }

                if (movedToNewLogicalLine)
                {
                    if (renderProgress.output.empty() || renderProgress.output.back() != '\n')
                    {
                        renderProgress.output.push_back('\n');
                        whitespace.shift_tokens(T_NEWLINE);
                    }
                }
                else if (!renderProgress.previousWasExplicitWhitespace && whitespace.must_insert(id, value))
                {
                    if (renderProgress.output.empty() || (renderProgress.output.back() != ' ' && renderProgress.output.back() != '\n' && renderProgress.output.back() != '\r' && renderProgress.output.back() != '\t'))
                    {
                        renderProgress.output.push_back(' ');
                        whitespace.shift_tokens(T_SPACE);
                    }
                }
            }

            renderProgress.output.append(value.data(), value.size());
            whitespace.shift_tokens(id);
            if (!renderProgress.hasPreviousToken ||
                renderProgress.previousFile.size() != currentFile.size() ||
                !std::equal(currentFile.begin(), currentFile.end(), renderProgress.previousFile.begin()))
            {
                renderProgress.previousFile.assign(currentFile.c_str(), currentFile.size());
            }
            renderProgress.previousLine = currentLine;
            renderProgress.hasPreviousToken = true;
            renderProgress.previousWasExplicitWhitespace = explicitWhitespace;
            ++renderProgress.emittedTokenCount;

            if (tokenTimer.has_value())
                tokenTimer.reset();
        }

        if (loopBodyTimer.has_value())
            loopBodyTimer.reset();

        if (perfStats.enabled)
        {
            nbl::wave::detail::ScopedPerfTimer iteratorAdvanceTimer(perfStats.iteratorAdvanceTime);
            ++it;
        }
        else
            ++it;
    }
}

std::string preprocessImpl(
    std::string& code,
    const nbl::asset::IShaderCompiler::SPreprocessorOptions& preprocessOptions,
    const bool withCaching,
    std::function<void(nbl::wave::context&)> post)
{
    nbl::wave::context context(code.begin(), code.end(), preprocessOptions.sourceIdentifier.data(), { preprocessOptions });

    WaveRenderProgress renderProgress;
    const char* phase = "registering built-in macros";
    std::string activeMacroDefinition;
    const auto reportPartialOutputOnFailure = [&]()
    {
        if (preprocessOptions.onPartialOutputOnFailure)
            preprocessOptions.onPartialOutputOnFailure(renderProgress.output);
    };
    const auto makeFailureContext = [&](const char* const fileName, const int lineNo, const int columnNo)
    {
        return makeWaveFailureContext(preprocessOptions, context, renderProgress, code, phase, activeMacroDefinition, fileName, lineNo, columnNo);
    };
    try
    {
        const auto totalBegin = std::chrono::steady_clock::now();
        nbl::wave::detail::reset_perf_stats();
        context.set_caching(withCaching);
        context.add_macro_definition("__HLSL_VERSION");
        context.add_macro_definition("__SPIRV_MAJOR_VERSION__=" + std::to_string(IShaderCompiler::getSpirvMajor(preprocessOptions.targetSpirvVersion)));
        context.add_macro_definition("__SPIRV_MINOR_VERSION__=" + std::to_string(IShaderCompiler::getSpirvMinor(preprocessOptions.targetSpirvVersion)));

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
        {
            nbl::wave::detail::ScopedPerfTimer renderTimer(nbl::wave::detail::perf_stats().renderTime);
            renderPreprocessedOutput(context, renderProgress);
        }
        auto& perfStats = nbl::wave::detail::perf_stats();
        perfStats.outputBytes = renderProgress.output.size();
        perfStats.emittedTokenCount = renderProgress.emittedTokenCount;
        perfStats.totalPreprocessTime = std::chrono::steady_clock::now() - totalBegin;
    }
    catch (boost::wave::preprocess_exception& e)
    {
        reportPartialOutputOnFailure();
        const auto escapedDescription = nbl::wave::detail::escape_control_chars(e.description());
        const auto escapedFileName = nbl::wave::detail::escape_control_chars(e.file_name());
        const auto failureContext = makeFailureContext(e.file_name(), e.line_no(), e.column_no());
        preprocessOptions.logger.log("%s exception caught. %s [%s:%d:%d]\n%s", system::ILogger::ELL_ERROR, e.what(), escapedDescription.c_str(), escapedFileName.c_str(), e.line_no(), e.column_no(), failureContext.c_str());
        preprocessOptions.logger.log("Boost diagnostic information:\n%s", system::ILogger::ELL_ERROR, boost::diagnostic_information(e).c_str());
        return {};
    }
    catch (const boost::exception& e)
    {
        reportPartialOutputOnFailure();
        const auto failureContext = makeFailureContext(preprocessOptions.sourceIdentifier.data(), 0, 0);
        preprocessOptions.logger.log("Boost exception caught during Wave preprocessing.\n%s", system::ILogger::ELL_ERROR, failureContext.c_str());
        preprocessOptions.logger.log("Boost diagnostic information:\n%s", system::ILogger::ELL_ERROR, boost::diagnostic_information(e).c_str());
        return {};
    }
    catch (const std::exception& e)
    {
        reportPartialOutputOnFailure();
        const auto failureContext = makeFailureContext(preprocessOptions.sourceIdentifier.data(), 0, 0);
        preprocessOptions.logger.log("std::exception caught during Wave preprocessing. %s\n%s", system::ILogger::ELL_ERROR, e.what(), failureContext.c_str());
        return {};
    }
    catch (...)
    {
        reportPartialOutputOnFailure();
        const auto failureContext = makeFailureContext(preprocessOptions.sourceIdentifier.data(), 0, 0);
        preprocessOptions.logger.log("Unknown exception caught during Wave preprocessing.\n%s", system::ILogger::ELL_ERROR, failureContext.c_str());
        preprocessOptions.logger.log("Current exception diagnostic information:\n%s", system::ILogger::ELL_ERROR, boost::current_exception_diagnostic_information().c_str());
        return {};
    }

    post(context);
    nbl::wave::detail::dump_perf_stats();

    return std::move(renderProgress.output);
}
}

namespace nbl::wave
{
    std::string preprocess(std::string& code, const nbl::asset::IShaderCompiler::SPreprocessorOptions& preprocessOptions, bool withCaching, std::function<void(nbl::wave::context&)> post)
    {
        return preprocessImpl(code, preprocessOptions, withCaching, std::move(post));
    }
}
