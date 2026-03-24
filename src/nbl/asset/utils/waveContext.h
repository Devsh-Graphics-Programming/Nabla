// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_ASSET_WAVE_CONTEXT_H_INCLUDED_
#define _NBL_ASSET_WAVE_CONTEXT_H_INCLUDED_
//! This file is not supposed to be included in user-accesible header files

#include <boost/wave.hpp>
#include <boost/wave/cpplexer/cpp_lex_token.hpp>
#include <boost/wave/cpplexer/cpp_lex_iterator.hpp>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <string_view>
#include <unordered_map>
#include <unordered_set>

#include "nbl/asset/utils/IShaderCompiler.h"
#include "includeResolutionCommon.h"

namespace nbl::wave
{
using namespace boost;
using namespace boost::wave;
using namespace boost::wave::util;

namespace detail
{
struct PerfStats
{
    bool enabled = false;
    bool includeDetailsEnabled = false;
    uint64_t includeRequests = 0ull;
    uint64_t includeLookupCount = 0ull;
    uint64_t includeResolutionCacheSkips = 0ull;
    uint64_t postLoadPragmaSkips = 0ull;
    std::chrono::nanoseconds includeLookupTime = std::chrono::nanoseconds::zero();
    std::chrono::nanoseconds tokenHandlingTime = std::chrono::nanoseconds::zero();
    std::chrono::nanoseconds iteratorAdvanceTime = std::chrono::nanoseconds::zero();
    std::chrono::nanoseconds loopBodyTime = std::chrono::nanoseconds::zero();
    std::chrono::nanoseconds renderTime = std::chrono::nanoseconds::zero();
    std::chrono::nanoseconds totalPreprocessTime = std::chrono::nanoseconds::zero();
    size_t outputBytes = 0ull;
    uint64_t emittedTokenCount = 0ull;
    std::unordered_map<std::string, uint64_t> requestedIncludeSpellingCounts;
    std::unordered_map<std::string, uint64_t> resolvedIncludePathCounts;
};

inline PerfStats& perf_stats()
{
    static PerfStats stats = []()
    {
        PerfStats value;
        value.enabled = std::getenv("NBL_WAVE_PROFILE") != nullptr;
        value.includeDetailsEnabled = std::getenv("NBL_WAVE_PROFILE_INCLUDES") != nullptr;
        return value;
    }();
    return stats;
}

inline void reset_perf_stats()
{
    auto& stats = perf_stats();
    const bool enabled = stats.enabled;
    const bool includeDetailsEnabled = stats.includeDetailsEnabled;
    stats = {};
    stats.enabled = enabled;
    stats.includeDetailsEnabled = includeDetailsEnabled;
}

class ScopedPerfTimer
{
    public:
        explicit ScopedPerfTimer(std::chrono::nanoseconds& target) : m_target(target), m_begin(std::chrono::steady_clock::now()) {}
        ~ScopedPerfTimer()
        {
            m_target += std::chrono::steady_clock::now() - m_begin;
        }

    private:
        std::chrono::nanoseconds& m_target;
        std::chrono::steady_clock::time_point m_begin;
};

inline void dump_perf_stats()
{
    const auto& stats = perf_stats();
    if (!stats.enabled)
        return;

    const auto to_ms = [](const std::chrono::nanoseconds value) -> double
    {
        return std::chrono::duration<double, std::milli>(value).count();
    };

    std::fprintf(
        stderr,
        "[wave-profile] total_ms=%.3f include_lookup_ms=%.3f token_handling_ms=%.3f iterator_advance_ms=%.3f loop_body_ms=%.3f render_ms=%.3f include_requests=%llu include_lookups=%llu resolution_cache_skips=%llu postload_pragma_skips=%llu emitted_tokens=%llu output_bytes=%zu\n",
        to_ms(stats.totalPreprocessTime),
        to_ms(stats.includeLookupTime),
        to_ms(stats.tokenHandlingTime),
        to_ms(stats.iteratorAdvanceTime),
        to_ms(stats.loopBodyTime),
        to_ms(stats.renderTime),
        static_cast<unsigned long long>(stats.includeRequests),
        static_cast<unsigned long long>(stats.includeLookupCount),
        static_cast<unsigned long long>(stats.includeResolutionCacheSkips),
        static_cast<unsigned long long>(stats.postLoadPragmaSkips),
        static_cast<unsigned long long>(stats.emittedTokenCount),
        stats.outputBytes
    );

    if (!stats.includeDetailsEnabled)
        return;

    auto dumpTopCounts = [](const char* const label, const std::unordered_map<std::string, uint64_t>& counts)
    {
        if (counts.empty())
            return;

        std::vector<std::pair<std::string, uint64_t>> entries;
        entries.reserve(counts.size());
        for (const auto& [name, count] : counts)
            entries.emplace_back(name, count);

        std::sort(entries.begin(), entries.end(), [](const auto& lhs, const auto& rhs)
        {
            if (lhs.second != rhs.second)
                return lhs.second > rhs.second;
            return lhs.first < rhs.first;
        });

        constexpr size_t kMaxEntries = 24ull;
        const auto limit = std::min(entries.size(), kMaxEntries);
        for (size_t i = 0ull; i < limit; ++i)
        {
            const auto& entry = entries[i];
            std::fprintf(stderr, "[wave-profile] %s[%zu]=%llu %s\n", label, i, static_cast<unsigned long long>(entry.second), entry.first.c_str());
        }
    };

    dumpTopCounts("requested_include", stats.requestedIncludeSpellingCounts);
    dumpTopCounts("resolved_include", stats.resolvedIncludePathCounts);
}

struct LanguageFlagConfig
{
    bool preserveComments = false;
    bool enableCpp20 = true;
    bool preferPpNumbers = true;
    bool emitLineDirectives = true;
    bool includeGuardDetection = true;
    bool emitPragmaDirectives = true;
};

inline boost::wave::language_support make_language_flags(const LanguageFlagConfig& config)
{
    auto flags = boost::wave::language_support();
    if (config.enableCpp20)
        flags = boost::wave::language_support(flags | support_cpp20); // C++20 lexer mode. https://github.com/Devsh-Graphics-Programming/wave/blob/e02cda69e4d070fd9b16a39282d6b5c717cb3da4/include/boost/wave/language_support.hpp#L56-L59
    if (config.preferPpNumbers)
        flags = boost::wave::language_support(flags | support_option_prefer_pp_numbers); // Prefer pp-number lexing before retokenization. https://github.com/Devsh-Graphics-Programming/wave/blob/e02cda69e4d070fd9b16a39282d6b5c717cb3da4/include/boost/wave/language_support.hpp#L71
    if (config.preserveComments)
        flags = boost::wave::language_support(flags | support_option_preserve_comments); // Keep comments in the token stream. https://github.com/Devsh-Graphics-Programming/wave/blob/e02cda69e4d070fd9b16a39282d6b5c717cb3da4/include/boost/wave/language_support.hpp#L67
    if (config.emitLineDirectives)
        flags = boost::wave::language_support(flags | support_option_emit_line_directives); // Emit #line directives in the output. https://github.com/Devsh-Graphics-Programming/wave/blob/e02cda69e4d070fd9b16a39282d6b5c717cb3da4/include/boost/wave/language_support.hpp#L72
    if (config.includeGuardDetection)
        flags = boost::wave::language_support(flags | support_option_include_guard_detection); // Let Wave short-circuit classic include guards. https://github.com/Devsh-Graphics-Programming/wave/blob/e02cda69e4d070fd9b16a39282d6b5c717cb3da4/libs/wave/include/boost/wave/language_support.hpp#L239
    if (config.emitPragmaDirectives)
        flags = boost::wave::language_support(flags | support_option_emit_pragma_directives); // Keep pragma directives in the output. https://github.com/Devsh-Graphics-Programming/wave/blob/e02cda69e4d070fd9b16a39282d6b5c717cb3da4/include/boost/wave/language_support.hpp#L74
    // support_option_emit_contnewlines // Emit escaped line continuations. https://github.com/Devsh-Graphics-Programming/wave/blob/e02cda69e4d070fd9b16a39282d6b5c717cb3da4/include/boost/wave/language_support.hpp#L65
    // support_option_insert_whitespace // Let Wave inject separator whitespace. https://github.com/Devsh-Graphics-Programming/wave/blob/e02cda69e4d070fd9b16a39282d6b5c717cb3da4/include/boost/wave/language_support.hpp#L66
    return flags;
}

inline std::string escape_control_chars(std::string_view text)
{
    static constexpr char hex[] = "0123456789ABCDEF";

    std::string out;
    out.reserve(text.size());

    for (const auto ch : text)
    {
        switch (ch)
        {
            case '\n':
                out += "\\n";
                break;
            case '\r':
                out += "\\r";
                break;
            case '\t':
                out += "\\t";
                break;
            case '\0':
                out += "\\0";
                break;
            default:
            {
                const auto uch = static_cast<unsigned char>(ch);
                if (uch < 0x20u || uch == 0x7Fu)
                {
                    out += "\\x";
                    out.push_back(hex[uch >> 4u]);
                    out.push_back(hex[uch & 0x0Fu]);
                }
                else
                    out.push_back(ch);
                break;
            }
        }
    }

    return out;
}

inline std::string escape_control_chars(const char* text)
{
    if (!text)
        return {};
    return escape_control_chars(std::string_view(text));
}
}

// for including builtins 
struct load_to_string final
{
    template <typename IterContextT>
    class inner
    {
        public:
            template <typename PositionT>
            static void init_iterators(IterContextT& iter_ctx, PositionT const& act_pos, boost::wave::language_support language)
            {
                iter_ctx.instring = iter_ctx.ctx.take_located_include_content();
                if (!iter_ctx.instring.empty() && iter_ctx.instring.back() != '\n' && iter_ctx.instring.back() != '\r')
                    iter_ctx.instring.push_back('\n');

                using iterator_type = IterContextT::iterator_type;
                iter_ctx.first = iterator_type(iter_ctx.instring.begin(),iter_ctx.instring.end(),PositionT(iter_ctx.filename),language);
                iter_ctx.last = iterator_type();
            }

        private:
            std::string instring;
    };
};


struct preprocessing_hooks final : public boost::wave::context_policies::default_preprocessing_hooks
{
    preprocessing_hooks(const nbl::asset::IShaderCompiler::SPreprocessorOptions& _preprocessOptions)
        : m_includeFinder(_preprocessOptions.includeFinder), m_logger(_preprocessOptions.logger), m_preserveComments(_preprocessOptions.preserveComments), m_pragmaStage(nbl::asset::IShader::E_SHADER_STAGE::ESS_UNKNOWN), m_dxc_compile_flags_override()
    {
        hash_token_occurences = 0;
    }

    template <typename ContextT, typename TokenT>
    bool found_directive(ContextT const& ctx, TokenT const& directive)
    {
        hash_token_occurences++;
        return false;
    }

    template <typename ContextT>
    bool locate_include_file(ContextT& ctx, std::string& file_path, bool is_system, char const* current_name, std::string& dir_path, std::string& native_name)
    {
        assert(false); // should never be called
        return false;
    }


    // interpretation of #pragma's of the form 'wave option[(value)]'
    template <typename ContextT, typename ContainerT>
    bool interpret_pragma(
        ContextT const& ctx, ContainerT& pending,
        typename ContextT::token_type const& option, ContainerT const& values,
        typename ContextT::token_type const& act_token
    )
    {
        auto optionStr = option.get_value().c_str();
        if (strcmp(optionStr,"shader_stage")==0) 
        {
            auto valueIter = values.begin();
            if (valueIter == values.end())
            {
                m_logger.log("Pre-processor error:\nMalformed shader_stage pragma. No shaderstage option given", nbl::system::ILogger::ELL_ERROR);
                return false;
            }
            auto shaderStageIdentifier = std::string(valueIter->get_value().c_str());
            const static core::unordered_map<std::string,IShader::E_SHADER_STAGE> stageFromIdent =
            {
                { "vertex", IShader::E_SHADER_STAGE::ESS_VERTEX },
                { "fragment", IShader::E_SHADER_STAGE::ESS_FRAGMENT },
                { "tesscontrol", IShader::E_SHADER_STAGE::ESS_TESSELLATION_CONTROL },
                { "tesseval", IShader::E_SHADER_STAGE::ESS_TESSELLATION_EVALUATION },
                { "geometry", IShader::E_SHADER_STAGE::ESS_GEOMETRY },
                { "compute", IShader::E_SHADER_STAGE::ESS_COMPUTE }
            };
            auto found = stageFromIdent.find(shaderStageIdentifier);
            if (found == stageFromIdent.end())
            {
                m_logger.log("Pre-processor error:\nMalformed shader_stage pragma. Unknown stage '%s'", nbl::system::ILogger::ELL_ERROR, shaderStageIdentifier.c_str());
                return false;
            }
            valueIter++;
            if (valueIter != values.end()) {
                m_logger.log("Pre-processor error:\nMalformed shader_stage pragma. Too many arguments", nbl::system::ILogger::ELL_ERROR);
                return false;
            }
            m_pragmaStage = found->second;
            return true;
        }
        
        if (strcmp(optionStr, "dxc_compile_flags") == 0) {
            if (hash_token_occurences != 1) {
                m_logger.log("Pre-processor error: Encountered a \"#pragma wave dxc_compile_flags\" but it is not the first preprocessor directive.", system::ILogger::ELL_ERROR);
                return false;
            }
            m_dxc_compile_flags_override.clear();
            std::string arg = "";
            for (auto valueIter = values.begin(); valueIter != values.end(); valueIter++) {
                std::string compiler_option_s = std::string(valueIter->get_value().c_str());
                // the compiler_option_s is a token thus can be only part of the actual argument, i.e. "-spirv" will be split into tokens [ "-", "spirv" ]
                // for dxc_compile_flags just join the strings until it finds a whitespace or end of args
                if (IS_CATEGORY(*valueIter, WhiteSpaceTokenType))
                {
                    if (!arg.empty())
                    {
                        m_dxc_compile_flags_override.push_back(arg);
                        arg.clear();
                    }
                }
                else 
                {
                    // append string
                    arg += compiler_option_s;
                }
            }
            if(arg.size() > 0)
                m_dxc_compile_flags_override.push_back(arg);
        
            return true;
        }

        return false;
    }

    template <typename ContextT, typename ContainerT>
    bool found_error_directive(ContextT const& ctx, ContainerT const& message)
    {
        std::ostringstream stream;
        for (const auto& token : message)
            stream << token.get_value();
        m_logger.log("Pre-processor encountered error directive:\n%s", nbl::system::ILogger::ELL_ERROR, stream.str().c_str());
        return false;
    }

    const asset::IShaderCompiler::CIncludeFinder* m_includeFinder;
    system::logger_opt_ptr m_logger;
    bool m_preserveComments;
    asset::IShader::E_SHADER_STAGE m_pragmaStage;
    int hash_token_occurences;
    std::vector<std::string> m_dxc_compile_flags_override;

};

class context : private boost::noncopyable
{
    private:
        using actual_context_type = context;

    public:
        using token_type = boost::wave::cpplexer::lex_token<>;
        using string_type = token_type::string_type;

        using target_iterator_type = core::string::iterator;
        using lexer_type = boost::wave::cpplexer::lex_iterator<token_type>;
        typedef pp_iterator<context>                    iterator_type;

        using input_policy_type = load_to_string;
        using position_type = token_type::position_type;

        // type of a token sequence
        using token_sequence_type = std::list<token_type,boost::fast_pool_allocator<token_type>>;

    private:
        // stack of shared_ptr's to the pending iteration contexts
        typedef boost::shared_ptr<base_iteration_context<context, lexer_type> >
            iteration_ptr_type;
        typedef boost::wave::util::iteration_context_stack<iteration_ptr_type>
            iteration_context_stack_type;
        typedef typename iteration_context_stack_type::size_type iter_size_type;

        context* this_() { return this; }           // avoid warning in constructor

    public:
        context(target_iterator_type const& first_, target_iterator_type const& last_, char const* fname, preprocessing_hooks const& hooks_)
            : first(first_), last(last_), filename(fname)
            , has_been_initialized(false)
            , current_filename(fname)
            , current_relative_filename(fname)
            , macros(*this_())
            , language([&hooks_]
                {
                    auto config = detail::LanguageFlagConfig{};
                    config.preserveComments = hooks_.m_preserveComments;
                    return detail::make_language_flags(config);
                }())
            , hooks(hooks_)
        {
            macros.init_predefined_macros(fname);
        }

        // default copy constructor
        // default assignment operator
        // default destructor

        // iterator interface
        iterator_type begin()
        {
            return iterator_type(*this, first, last, position_type(filename.c_str()));
        }
        iterator_type begin(target_iterator_type const& first_, target_iterator_type const& last_)
        {
            return iterator_type(*this, first_, last_, position_type(filename.c_str()));
        }
        iterator_type end() const
        {
            return iterator_type();
        }

        // maintain include paths
        typename iteration_context_stack_type::size_type get_iteration_depth() const
        {
            return iter_ctxs.size();
        }

        // maintain defined macros
        template <typename StringT>
        bool add_macro_definition(StringT macrostring, bool is_predefined = false)
        {
            return boost::wave::util::add_macro_definition(*this,
                util::to_string<std::string>(macrostring), is_predefined,
                get_language());
        }

        // Define and undefine macros, macro introspection
        template <typename StringT>
        bool add_macro_definition(StringT const& name, position_type const& pos,
            bool has_params, std::vector<token_type>& parameters,
            token_sequence_type& definition, bool is_predefined = false)
        {
            return macros.add_macro(
                token_type(T_IDENTIFIER, util::to_string<string_type>(name), pos),
                has_params, parameters, definition, is_predefined);
        }
        template <typename StringT>
        bool is_defined_macro(StringT const& str) const
        {
            return macros.is_defined(util::to_string<string_type>(str));
        }
        template <typename StringT>
        bool get_macro_definition(StringT const& name,
            bool& has_params, bool& is_predefined, position_type& pos,
            std::vector<token_type>& parameters,
            token_sequence_type& definition) const
        {
            return macros.get_macro(util::to_string<string_type>(name),
                has_params, is_predefined, pos, parameters, definition);
        }
        template <typename StringT>
        bool remove_macro_definition(StringT const& undefname, bool even_predefined = false)
        {
            // strip leading and trailing whitespace
            string_type name = util::to_string<string_type>(undefname);
            typename string_type::size_type pos = name.find_first_not_of(" \t");
            if (pos != string_type::npos) {
                typename string_type::size_type endpos = name.find_last_not_of(" \t");
                name = name.substr(pos, endpos - pos + 1);
            }

            return macros.remove_macro(name, macros.get_main_pos(), even_predefined);
        }
        void reset_macro_definitions()
        {
            macros.reset_macromap(); macros.init_predefined_macros();
        }

        // Iterate over names of defined macros
        typedef boost::wave::util::macromap<context> macromap_type;
        typedef typename macromap_type::name_iterator name_iterator;
        typedef typename macromap_type::const_name_iterator const_name_iterator;

        name_iterator macro_names_begin() { return macros.begin(); }
        name_iterator macro_names_end() { return macros.end(); }
        const_name_iterator macro_names_begin() const { return macros.begin(); }
        const_name_iterator macro_names_end() const { return macros.end(); }

        // This version now is used internally mainly, but since it was a documented
        // API function we leave it in the public interface.
        bool add_macro_definition(token_type const& name, bool has_params,
            std::vector<token_type>& parameters, token_sequence_type& definition,
            bool is_predefined = false)
        {
            return macros.add_macro(name, has_params, parameters, definition,
                is_predefined);
        }

        // get the Wave version information
        static std::string get_version()
        {
            boost::wave::util::predefined_macros p;
            return util::to_string<std::string>(p.get_fullversion());
        }
        static std::string get_version_string()
        {
            boost::wave::util::predefined_macros p;
            return util::to_string<std::string>(p.get_versionstr());
        }

        // access current language options
        boost::wave::language_support get_language() const { return language; }

        position_type& get_main_pos() { return macros.get_main_pos(); }
        position_type const& get_main_pos() const { return macros.get_main_pos(); }

        // change and ask for maximal possible include nesting depth
        void set_max_include_nesting_depth(iter_size_type new_depth)
        {
            iter_ctxs.set_max_include_nesting_depth(new_depth);
        }
        iter_size_type get_max_include_nesting_depth() const
        {
            return iter_ctxs.get_max_include_nesting_depth();
        }

        // access the policies
        preprocessing_hooks& get_hooks() { return hooks; }
        preprocessing_hooks const& get_hooks() const { return hooks; }

        // return type of actually used context type (might be the derived type)
        actual_context_type& derived()
        {
            return *static_cast<actual_context_type*>(this);
        }
        actual_context_type const& derived() const
        {
            return *static_cast<actual_context_type const*>(this);
        }

        // Nabla Additions Start
        const system::path& get_current_directory() const
        {
            return current_dir;
        }
        void set_current_directory(const system::path& filepath)
        {
            current_dir = filepath.parent_path();
        }
        const core::string& get_located_include_content() const
        {
            return located_include_content;
        }
        core::string take_located_include_content()
        {
            return std::move(located_include_content);
        }
        // Nabla Additions End

#if !defined(BOOST_NO_MEMBER_TEMPLATE_FRIENDS)
    protected:
        friend class boost::wave::pp_iterator<context>;
        friend class boost::wave::impl::pp_iterator_functor<context>;
        friend class boost::wave::util::macromap<context>;
#endif

        // make sure the context has been initialized
        void init_context()
        {
            if (has_been_initialized)
                return;

            set_current_directory(system::path(filename));
            has_been_initialized = true;  // execute once
        }

        template <typename IteratorT2>
        bool is_defined_macro(IteratorT2 const& begin, IteratorT2 const& end) const
        {
            return macros.is_defined(begin, end);
        }

        // conditional compilation contexts
        bool get_if_block_status() const { return ifblocks.get_status(); }
        bool get_if_block_some_part_status() const
        {
            return ifblocks.get_some_part_status();
        }
        bool get_enclosing_if_block_status() const
        {
            return ifblocks.get_enclosing_status();
        }
        void enter_if_block(bool new_status)
        {
            ifblocks.enter_if_block(new_status);
        }
        bool enter_elif_block(bool new_status)
        {
            return ifblocks.enter_elif_block(new_status);
        }
        bool enter_else_block() { return ifblocks.enter_else_block(); }
        bool exit_if_block() { return ifblocks.exit_if_block(); }
        typename boost::wave::util::if_block_stack::size_type get_if_block_depth() const
        {
            return ifblocks.get_if_block_depth();
        }

        // stack of iteration contexts
        iteration_ptr_type pop_iteration_context()
        {
            iteration_ptr_type top = iter_ctxs.top(); iter_ctxs.pop(); return top;
        }
        void push_iteration_context(position_type const& act_pos, iteration_ptr_type iter_ctx)
        {
            iter_ctxs.push(*this, act_pos, iter_ctx);
        }

        template <typename IteratorT2>
        token_type expand_tokensequence(IteratorT2& first_, IteratorT2 const& last_,
            token_sequence_type& pending, token_sequence_type& expanded,
            bool& seen_newline, bool expand_defined = false,
            bool expand_has_include = false)
        {
            return macros.expand_tokensequence(first_, last_, pending, expanded,
                seen_newline, expand_defined, expand_has_include);
        }

        template <typename IteratorT2>
        void expand_whole_tokensequence(IteratorT2& first_, IteratorT2 const& last_,
            token_sequence_type& expanded, bool expand_defined = true,
            bool expand_has_include = true)
        {
            macros.expand_whole_tokensequence(
                expanded, first_, last_,
                expand_defined, expand_has_include);

            // remove any contained placeholder
            boost::wave::util::impl::remove_placeholders(expanded);
        }

    public:
#if BOOST_WAVE_SUPPORT_PRAGMA_ONCE != 0
        void set_current_filename(char const* real_name)
        {
            current_filename = real_name;
        }
        std::string const& get_current_filename() const
        {
            return current_filename;
        }

        bool has_pragma_once(std::string const& filename_) const
        {
            return pragma_once_headers.contains(filename_);
        }
        bool has_cached_include_resolution(std::string_view includeName, bool is_system, std::string& absolutePath) const
        {
            const auto found = include_resolution_cache.find(make_include_resolution_key(includeName, is_system));
            if (found == include_resolution_cache.end())
                return false;

            absolutePath = found->second;
            return true;
        }
        void cache_include_resolution(std::string_view includeName, bool is_system, const std::string& absolutePath)
        {
            include_resolution_cache.insert_or_assign(make_include_resolution_key(includeName, is_system), absolutePath);
        }
        bool add_pragma_once_header(std::string const& filename_, std::string const& guard_name)
        {
            get_hooks().detected_include_guard(derived(), filename_, guard_name);
            return pragma_once_headers.emplace(filename_).second;
        }
        bool add_pragma_once_header(token_type const& pragma_, std::string const& filename_)
        {
            get_hooks().detected_pragma_once(derived(), pragma_, filename_);
            return pragma_once_headers.emplace(filename_).second;
        }
#endif

        void set_current_relative_filename(char const* real_name)
        {
            current_relative_filename = real_name;
        }
        std::string const& get_current_relative_filename() const
        {
            return current_relative_filename;
        }

        void set_caching(bool b) {
            cachingRequested = b;
        }

        std::vector<IShaderCompiler::CCache::SEntry::SPreprocessingDependency>&& get_dependencies() {
            return std::move(dependencies);
        }

    private:
        // the main input stream
        target_iterator_type first;         // underlying input stream
        target_iterator_type last;
        const std::string filename;               // associated main filename
        bool has_been_initialized;          // set cwd once

        std::string current_filename;              // real name of current preprocessed file
        std::string current_relative_filename;        // real relative name of current preprocessed file

        // Nabla Additions Start
        // these are temporaries!
        system::path current_dir;
        core::string located_include_content;
#if BOOST_WAVE_SUPPORT_PRAGMA_ONCE != 0
        std::unordered_set<std::string> pragma_once_headers;
#endif
        std::unordered_map<std::string, std::string> include_resolution_cache;
        // Cache Additions 
        bool cachingRequested = false;
        std::vector<IShaderCompiler::CCache::SEntry::SPreprocessingDependency> dependencies = {};
        // Nabla Additions End

        boost::wave::util::if_block_stack ifblocks;   // conditional compilation contexts
        iteration_context_stack_type iter_ctxs;       // iteration contexts
        macromap_type macros;                         // map of defined macros
        const boost::wave::language_support language;       // supported language/extensions
        preprocessing_hooks hooks;                    // hook policy instance

        std::string make_include_resolution_key(std::string_view includeName, bool is_system) const
        {
            std::string key;
            const bool globallyResolved = is_system || asset::detail::isGloballyResolvedIncludeName(includeName);
            if (!globallyResolved)
            {
                const auto currentDirString = current_dir.generic_string();
                key.reserve(currentDirString.size() + includeName.size() + 3ull);
                key.append(currentDirString);
                key.push_back('\n');
            }
            else
                key.reserve(includeName.size() + 2ull);
            key.push_back(globallyResolved ? 'G' : 'R');
            key.push_back('\n');
            key.append(includeName.data(), includeName.size());
            return key;
        }
};

}


template<> inline bool boost::wave::impl::pp_iterator_functor<nbl::wave::context>::on_include_helper(char const* f, char const* s, bool is_system, bool include_next)
{
    assert(!include_next);
    namespace fs = boost::filesystem;

    // try to locate the given file, searching through the include path lists
    std::string file_path(s);

    // call the 'found_include_directive' hook function
    if (ctx.get_hooks().found_include_directive(ctx.derived(),f,false))
        return true;    // client returned false: skip file to include

    IShaderCompiler::IIncludeLoader::found_t result;
    auto* includeFinder = ctx.get_hooks().m_includeFinder;
    bool standardInclude;
    std::string cachedAbsolutePath;
    const bool needHash = ctx.cachingRequested;
    auto& perfStats = nbl::wave::detail::perf_stats();
    if (perfStats.enabled)
    {
        ++perfStats.includeRequests;
        if (perfStats.includeDetailsEnabled)
            ++perfStats.requestedIncludeSpellingCounts[file_path];
    }
    if (ctx.has_cached_include_resolution(file_path, is_system, cachedAbsolutePath))
    {
        if (ctx.has_pragma_once(cachedAbsolutePath))
        {
            if (perfStats.enabled)
                ++perfStats.includeResolutionCacheSkips;
            return true;
        }

        if (includeFinder && nbl::system::path(cachedAbsolutePath).is_absolute())
        {
            nbl::wave::detail::ScopedPerfTimer lookupTimer(perfStats.includeLookupTime);
            if (perfStats.enabled)
                ++perfStats.includeLookupCount;
            result = includeFinder->getDefaultFileSystemLoader()->getInclude(nbl::system::path{}, cachedAbsolutePath, needHash);
            standardInclude = is_system;
        }
    }

    if (!result && includeFinder)
    {
        nbl::wave::detail::ScopedPerfTimer lookupTimer(perfStats.includeLookupTime);
        if (perfStats.enabled)
            ++perfStats.includeLookupCount;
        if (is_system) {
            result = includeFinder->getIncludeStandard(ctx.get_current_directory(), file_path, needHash);
            standardInclude = true;
        }
        else {
            result = includeFinder->getIncludeRelative(ctx.get_current_directory(), file_path, needHash);
            standardInclude = false;
        }
    }
    else if (!result) {
        const auto escapedPath = nbl::wave::detail::escape_control_chars(file_path);
        ctx.get_hooks().m_logger.log("Pre-processor error: Include finder not assigned, preprocessor will not include file %s", nbl::system::ILogger::ELL_ERROR, escapedPath.c_str());
        return false;
    }

    if (!result)
    {
        const auto escapedPath = nbl::wave::detail::escape_control_chars(file_path);
        const auto escapedSource = nbl::wave::detail::escape_control_chars(ctx.get_current_relative_filename());
        const auto escapedDirectory = nbl::wave::detail::escape_control_chars(ctx.get_current_directory().string());
        ctx.get_hooks().m_logger.log("Pre-processor error: Bad include file.\n  requested_include: %s\n  requesting_source: %s\n  requesting_directory: %s", nbl::system::ILogger::ELL_ERROR, escapedPath.c_str(), escapedSource.c_str(), escapedDirectory.c_str());
        BOOST_WAVE_THROW_CTX(ctx, preprocess_exception, bad_include_file, file_path.c_str(), act_pos);
        return false;
    }

#if BOOST_WAVE_SUPPORT_PRAGMA_ONCE != 0
    if (ctx.has_pragma_once(result.absolutePath.string()))
    {
        ctx.cache_include_resolution(file_path, is_system, result.absolutePath.string());
        if (perfStats.enabled)
            ++perfStats.postLoadPragmaSkips;
        return true;
    }
#endif

    ctx.cache_include_resolution(file_path, is_system, result.absolutePath.string());
    if (perfStats.enabled && perfStats.includeDetailsEnabled)
        ++perfStats.resolvedIncludePathCounts[result.absolutePath.generic_string()];

    // If caching was requested, push a new SDependency onto dependencies
    if (ctx.cachingRequested) {
        ctx.dependencies.emplace_back(ctx.get_current_directory(), file_path, standardInclude, std::move(result.hash));
    }

    ctx.located_include_content = std::move(result.contents);
    // the new include file determines the actual current directory
    ctx.set_current_directory(result.absolutePath);

    {
        // preprocess the opened file
        boost::shared_ptr<base_iteration_context_type> new_iter_ctx(
            new iteration_context_type(ctx,result.absolutePath.string().c_str(),act_pos,
                boost::wave::enable_prefer_pp_numbers(ctx.get_language()),
                is_system ? base_iteration_context_type::system_header :
                base_iteration_context_type::user_header));

        // call the include policy trace function
        ctx.get_hooks().opened_include_file(ctx.derived(),file_path,result.absolutePath.string(),is_system);

        // store current file position
        iter_ctx->real_relative_filename = ctx.get_current_relative_filename().c_str();
        iter_ctx->filename = act_pos.get_file();
        iter_ctx->line = act_pos.get_line();
        iter_ctx->if_block_depth = ctx.get_if_block_depth();
        iter_ctx->emitted_lines = (unsigned int)(-1);   // force #line directive

        // push the old iteration context onto the stack and continue with the new
        ctx.push_iteration_context(act_pos, iter_ctx);
        iter_ctx = new_iter_ctx;
        seen_newline = true;        // fake a newline to trigger pp_directive
        must_emit_line_directive = true;

        act_pos.set_file(iter_ctx->filename);  // initialize file position
#if BOOST_WAVE_SUPPORT_PRAGMA_ONCE != 0
        ctx.set_current_filename(result.absolutePath.string().c_str());
#endif

        ctx.set_current_relative_filename(file_path.c_str());
        iter_ctx->real_relative_filename = file_path.c_str();

        act_pos.set_line(iter_ctx->line);
        act_pos.set_column(0);
    }

    return true;
}

#endif
