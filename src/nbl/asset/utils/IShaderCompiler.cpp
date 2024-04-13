// Copyright (C) 2018-2022 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#include "nbl/asset/utils/IShaderCompiler.h"
#include "nbl/asset/utils/shadercUtils.h"
#include "nbl/asset/utils/CGLSLVirtualTexturingBuiltinIncludeGenerator.h"

#include <sstream>
#include <regex>
#include <iterator>

using namespace nbl;
using namespace nbl::asset;

std::string IShaderCompiler::escapeFilename(std::string&& code)
{
    std::string dest;
    dest.reserve(code.size() * 2);
    for (char c : code)
    {
        if (c == '\\')
            dest.append("\\" "\\");
        else 
            dest.push_back(c);
    }
    return dest;
}

//all "#", except those in "#include"/"#version"/"#pragma shader_stage(...)", replaced with `PREPROC_DIRECTIVE_DISABLER`
void IShaderCompiler::disableAllDirectivesExceptIncludes(std::string& _code)
{
    // TODO: replace this with a proper-ish proprocessor and includer one day
    std::regex directive("#(?!(( |\t|\r|\v|\f)*(include|version|pragma shader_stage)))");//all # not followed by "include" nor "version" nor "pragma shader_stage"
    //`#pragma shader_stage(...)` is needed for determining shader stage when `_stage` param of IShaderCompiler functions is set to ESS_UNKNOWN
    _code = std::regex_replace(_code, directive, IShaderCompiler::PREPROC_DIRECTIVE_DISABLER);
}

void IShaderCompiler::reenableDirectives(std::string& _code)
{
    std::regex directive(IShaderCompiler::PREPROC_DIRECTIVE_ENABLER);
    _code = std::regex_replace(_code, directive, "#");
}

std::string IShaderCompiler::encloseWithinExtraInclGuards(std::string&& _code, uint32_t _maxInclusions, const char* _identifier)
{
    assert(_maxInclusions != 0u);

    using namespace std::string_literals;
    std::string defBase_ = "_GENERATED_INCLUDE_GUARD_"s + _identifier + "_";
    std::replace_if(defBase_.begin(), defBase_.end(), [](char c) ->bool { return !::isalpha(c) && !::isdigit(c); }, '_');

    auto genDefs = [&defBase_, _maxInclusions, _identifier] {
        auto defBase = [&defBase_](uint32_t n) { return defBase_ + std::to_string(n); };
        std::string defs = "#ifndef " + defBase(0) + "\n\t#define " + defBase(0) + "\n";
        for (uint32_t i = 1u; i <= _maxInclusions; ++i) {
            const std::string defname = defBase(i);
            defs += "#elif !defined(" + defname + ")\n\t#define " + defname + "\n";
        }
        defs += "#endif\n";
        return defs;
    };
    auto genUndefs = [&defBase_, _maxInclusions, _identifier] {
        auto defBase = [&defBase_](int32_t n) { return defBase_ + std::to_string(n); };
        std::string undefs = "#ifdef " + defBase(_maxInclusions) + "\n\t#undef " + defBase(_maxInclusions) + "\n";
        for (int32_t i = _maxInclusions - 1; i >= 0; --i) {
            const std::string defname = defBase(i);
            undefs += "#elif defined(" + defname + ")\n\t#undef " + defname + "\n";
        }
        undefs += "#endif\n";
        return undefs;
    };

    std::string identifier = IShaderCompiler::escapeFilename(_identifier);
    return
        genDefs() +
        "\n"
        "#ifndef " + defBase_ + std::to_string(_maxInclusions) +
        "\n" +
        // This will get turned back into #line after the directives get re-enabled
        IShaderCompiler::PREPROC_DIRECTIVE_DISABLER + "line 1 \"" + identifier.c_str() + "\"\n" +
        _code +
        "\n"
        "#endif"
        "\n\n" +
        genUndefs();
}

// Amount of lines before the #line after having run encloseWithinExtraInclGuards
uint32_t IShaderCompiler::encloseWithinExtraInclGuardsLeadingLines(uint32_t _maxInclusions)
{
    auto lineDirectiveString = std::string(IShaderCompiler::PREPROC_DIRECTIVE_DISABLER) + "line";
    std::string str = IShaderCompiler::encloseWithinExtraInclGuards(std::string(""), _maxInclusions, "encloseWithinExtraInclGuardsLeadingLines");
    size_t lineDirectivePos = str.find(lineDirectiveString);
    auto substr = str.substr(0, lineDirectivePos - lineDirectiveString.length());

    return std::count(substr.begin(), substr.end(), '\n');
}

IShaderCompiler::IShaderCompiler(core::smart_refctd_ptr<system::ISystem>&& system)
    : m_system(std::move(system))
{
    m_defaultIncludeFinder = core::make_smart_refctd_ptr<CIncludeFinder>(core::smart_refctd_ptr(m_system));
    m_defaultIncludeFinder->addGenerator(core::make_smart_refctd_ptr<asset::CGLSLVirtualTexturingBuiltinIncludeGenerator>());
    m_defaultIncludeFinder->getIncludeStandard("", "nbl/builtin/glsl/utils/common.glsl");
}

std::string IShaderCompiler::preprocessShader(
    system::IFile* sourcefile,
    IShader::E_SHADER_STAGE stage,
    const SPreprocessorOptions& preprocessOptions,
    std::vector<CCache::SEntry::SPreprocessingDependency>* dependencies) const
{
    std::string code(sourcefile->getSize(), '\0');

    system::IFile::success_t success;
    sourcefile->read(success, code.data(), 0, sourcefile->getSize());
    if (!success)
        return nullptr;

    return preprocessShader(std::move(code), stage, preprocessOptions, dependencies);
}

auto IShaderCompiler::IIncludeGenerator::getInclude(const std::string& includeName) const -> IIncludeLoader::found_t
{
    core::vector<std::pair<std::regex, HandleFunc_t>> builtinNames = getBuiltinNamesToFunctionMapping();
    for (const auto& pattern : builtinNames)
    if (std::regex_match(includeName,pattern.first))
    {
        if (auto contents=pattern.second(includeName); !contents.empty())
        {
            // Welcome, you've came to a very disused piece of code, please check the first parameter (path) makes sense!
            _NBL_DEBUG_BREAK_IF(true);
            return {includeName,contents};
        }
    }

    return {};
}

core::vector<std::string> IShaderCompiler::IIncludeGenerator::parseArgumentsFromPath(const std::string& _path)
{
    core::vector<std::string> args;

    std::stringstream ss{ _path };
    std::string arg;
    while (std::getline(ss, arg, '/'))
        args.push_back(std::move(arg));

    return args;
}

IShaderCompiler::CFileSystemIncludeLoader::CFileSystemIncludeLoader(core::smart_refctd_ptr<system::ISystem>&& system) : m_system(std::move(system))
{}

auto IShaderCompiler::CFileSystemIncludeLoader::getInclude(const system::path& searchPath, const std::string& includeName) const -> found_t
{
    system::path path = searchPath / includeName;
    if (std::filesystem::exists(path))
        path = std::filesystem::canonical(path);

    core::smart_refctd_ptr<system::IFile> f;
    {
        system::ISystem::future_t<core::smart_refctd_ptr<system::IFile>> future;
        m_system->createFile(future, path.c_str(), system::IFile::ECF_READ);
        if (!future.wait())
            return {};
        future.acquire().move_into(f);
    }
    if (!f)
        return {};
    const size_t size = f->getSize();

    std::string contents(size, '\0');
    system::IFile::success_t succ;
    f->read(succ, contents.data(), 0, size);
    const bool success = bool(succ);
    assert(success);

    return {f->getFileName(),std::move(contents)};
}

IShaderCompiler::CIncludeFinder::CIncludeFinder(core::smart_refctd_ptr<system::ISystem>&& system) 
    : m_defaultFileSystemLoader(core::make_smart_refctd_ptr<CFileSystemIncludeLoader>(std::move(system)))
{
    addSearchPath("", m_defaultFileSystemLoader);
}

// ! includes within <>
// @param requestingSourceDir: the directory where the incude was requested
// @param includeName: the string within <> of the include preprocessing directive
auto IShaderCompiler::CIncludeFinder::getIncludeStandard(const system::path& requestingSourceDir, const std::string& includeName) const -> IIncludeLoader::found_t
{
    if (auto contents = tryIncludeGenerators(includeName)) 
        return contents;
    if (auto contents = trySearchPaths(includeName)) 
        return contents;
    return m_defaultFileSystemLoader->getInclude(requestingSourceDir.string(),includeName);
}

// ! includes within ""
// @param requestingSourceDir: the directory where the incude was requested
// @param includeName: the string within "" of the include preprocessing directive
auto IShaderCompiler::CIncludeFinder::getIncludeRelative(const system::path& requestingSourceDir, const std::string& includeName) const -> IIncludeLoader::found_t
{
    if (auto contents = m_defaultFileSystemLoader->getInclude(requestingSourceDir.string(),includeName))
        return contents;
    return trySearchPaths(includeName);
}

void IShaderCompiler::CIncludeFinder::addSearchPath(const std::string& searchPath, const core::smart_refctd_ptr<IIncludeLoader>& loader)
{
    if (!loader)
        return;
    m_loaders.push_back(LoaderSearchPath{ loader, searchPath });
}

void IShaderCompiler::CIncludeFinder::addGenerator(const core::smart_refctd_ptr<IIncludeGenerator>& generatorToAdd)
{
    if (!generatorToAdd)
        return;

    // this will find the place of first generator with prefix <= generatorToAdd or end
    auto found = std::lower_bound(m_generators.begin(), m_generators.end(), generatorToAdd->getPrefix(),
        [](const core::smart_refctd_ptr<IIncludeGenerator>& generator, const std::string_view& value)
        {
            auto element = generator->getPrefix();
            return element.compare(value) > 0; // first to return false is lower_bound -> first element that is <= value
        });

    m_generators.insert(found, generatorToAdd);
}

auto IShaderCompiler::CIncludeFinder::trySearchPaths(const std::string& includeName) const -> IIncludeLoader::found_t
{
    for (const auto& itr : m_loaders)
    if (auto contents = itr.loader->getInclude(itr.searchPath,includeName))
        return contents;
    return {};
}

auto IShaderCompiler::CIncludeFinder::tryIncludeGenerators(const std::string& includeName) const -> IIncludeLoader::found_t
{
    // Need custom function because std::filesystem doesn't consider the parameters we use after the extension like CustomShader.hlsl/512/64
    auto removeExtension = [](const std::string& str)
    {
        return str.substr(0, str.find_last_of('.'));
    };

    auto standardizePrefix = [](const std::string_view& prefix) -> std::string
    {
        std::string ret(prefix);
        // Remove Trailing '/' if any, to compare to filesystem paths
        if (*ret.rbegin() == '/' && ret.size() > 1u)
            ret.resize(ret.size() - 1u);
        return ret;
    };

    auto extension_removed_path = system::path(removeExtension(includeName));
    system::path path = extension_removed_path.parent_path();

    // Try Generators with Matching Prefixes:
    // Uses a "Path Peeling" method which goes one level up the directory tree until it finds a suitable generator
    auto end = m_generators.begin();
    while (!path.empty() && path.root_name().empty() && end != m_generators.end())
    {
        auto begin = std::lower_bound(end, m_generators.end(), path.string(),
            [&standardizePrefix](const core::smart_refctd_ptr<IIncludeGenerator>& generator, const std::string& value)
            {
                const auto element = standardizePrefix(generator->getPrefix());
                return element.compare(value) > 0; // first to return false is lower_bound -> first element that is <= value
            });

        // search from new beginning to real end
        end = std::upper_bound(begin, m_generators.end(), path.string(),
            [&standardizePrefix](const std::string& value, const core::smart_refctd_ptr<IIncludeGenerator>& generator)
            {
                const auto element = standardizePrefix(generator->getPrefix());
                return value.compare(element) > 0; // first to return true is upper_bound -> first element that is < value
            });

        for (auto generatorIt = begin; generatorIt != end; generatorIt++)
        {
            if (auto contents = (*generatorIt)->getInclude(includeName))
                return contents;
        }

        path = path.parent_path();
    }

    return {};
}
