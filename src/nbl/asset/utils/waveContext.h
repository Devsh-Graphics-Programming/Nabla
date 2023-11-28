// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_ASSET_WAVE_CONTEXT_H_INCLUDED_
#define _NBL_ASSET_WAVE_CONTEXT_H_INCLUDED_
//! This file is not supposed to be included in user-accesible header files

#define BOOST_WAVE_ENABLE_COMMANDLINE_MACROS 1
#define BOOST_WAVE_SUPPORT_PRAGMA_ONCE 0
#define BOOST_WAVE_EMIT_PRAGMA_DIRECTIVES 1
#define BOOST_WAVE_SERIALIZATION 0
#define BOOST_WAVE_SUPPORT_INCLUDE_NEXT 0
#include <boost/wave.hpp>
#include <boost/wave/cpplexer/cpp_lex_token.hpp>
#include <boost/wave/cpplexer/cpp_lex_iterator.hpp>


namespace nbl::wave
{
using namespace boost;
using namespace boost::wave;
using namespace boost::wave::util;

// for including builtins 
struct load_file_or_builtin_to_string
{
    template <typename IterContextT>
    class inner
    {
        public:
            template <typename PositionT>
            static void init_iterators(IterContextT& iter_ctx, PositionT const& act_pos, boost::wave::language_support language)
            {
                using iterator_type = typename IterContextT::iterator_type;

                std::string filepath(iter_ctx.filename.begin(), iter_ctx.filename.end());
                auto inclFinder = iter_ctx.ctx.get_hooks().m_includeFinder;
                if (inclFinder) 
                {
                    std::optional<std::string> result;
                    system::path requestingSourceDir(iter_ctx.ctx.get_current_directory().string());
                    if (iter_ctx.type == IterContextT::base_type::file_type::system_header) // is it a sys include (#include <...>)?
                        result = inclFinder->getIncludeStandard(requestingSourceDir, filepath);
                    else // regular #include "..."
                        result = inclFinder->getIncludeRelative(requestingSourceDir, filepath);

                    if (!result)
                        BOOST_WAVE_THROW_CTX(iter_ctx.ctx, boost::wave::preprocess_exception,
                            bad_include_file, iter_ctx.filename.c_str(), act_pos);
                    auto& res_str = *result;
                    iter_ctx.instring = res_str;
                }
                iter_ctx.first = iterator_type(
                    iter_ctx.instring.begin(), iter_ctx.instring.end(),
                    PositionT(iter_ctx.filename), language);
                iter_ctx.last = iterator_type();
            }

        private:
            std::string instring;
    };
};


struct custom_preprocessing_hooks : public boost::wave::context_policies::default_preprocessing_hooks
{
    custom_preprocessing_hooks(const IShaderCompiler::SPreprocessorOptions& _preprocessOptions) 
        : m_includeFinder(_preprocessOptions.includeFinder), m_logger(_preprocessOptions.logger), m_pragmaStage(IShader::ESS_UNKNOWN) {}

    const IShaderCompiler::CIncludeFinder* m_includeFinder;
    system::logger_opt_ptr m_logger;
    IShader::E_SHADER_STAGE m_pragmaStage;


    template <typename ContextT>
    bool locate_include_file(ContextT& ctx, std::string& file_path, bool is_system, char const* current_name, std::string& dir_path, std::string& native_name) 
    {
        //on builtin return true
        dir_path = ctx.get_current_directory().string();
        std::optional<std::string> result;
        if (is_system) {
            result = m_includeFinder->getIncludeStandard(dir_path, file_path);
            dir_path = "";
        }
        else
            result = m_includeFinder->getIncludeRelative(dir_path, file_path);
        if (!result)
        {
            m_logger.log("Pre-processor error: Bad include file.\n'%s' does not exist.", nbl::system::ILogger::ELL_ERROR, file_path.c_str());
            return false;
        }
        native_name = file_path;
        return true;
    }


    // interpretation of #pragma's of the form 'wave option[(value)]'
    template <typename ContextT, typename ContainerT>
    bool
        interpret_pragma(ContextT const& ctx, ContainerT& pending,
            typename ContextT::token_type const& option, ContainerT const& values,
            typename ContextT::token_type const& act_token)
    {
        auto optionStr = option.get_value().c_str();
        if (strcmp(optionStr, "shader_stage") == 0) 
        {
            auto valueIter = values.begin();
            if (valueIter == values.end()) {
                m_logger.log("Pre-processor error:\nMalformed shader_stage pragma. No shaderstage option given", nbl::system::ILogger::ELL_ERROR);
                return false;
            }
            auto shaderStageIdentifier = std::string(valueIter->get_value().c_str());
            core::unordered_map<std::string, IShader::E_SHADER_STAGE> stageFromIdent = {
                { "vertex", IShader::ESS_VERTEX },
                { "fragment", IShader::ESS_FRAGMENT },
                { "tesscontrol", IShader::ESS_TESSELLATION_CONTROL },
                { "tesseval", IShader::ESS_TESSELLATION_EVALUATION },
                { "geometry", IShader::ESS_GEOMETRY },
                { "compute", IShader::ESS_COMPUTE }
            };
            auto found = stageFromIdent.find(shaderStageIdentifier);
            if (found == stageFromIdent.end())
            {
                m_logger.log("Pre-processor error:\nMalformed shader_stage pragma. Unknown stage '%s'", nbl::system::ILogger::ELL_ERROR, shaderStageIdentifier);
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
        return false;
    }


    template <typename ContextT, typename ContainerT>
    bool found_error_directive(ContextT const& ctx, ContainerT const& message) {
        m_logger.log("Pre-processor error:\n%s", nbl::system::ILogger::ELL_ERROR, message);
        return true;
    }
};

class include_paths
{
    private:
        typedef std::list<std::pair<system::path, std::string> >
            include_list_type;
        typedef include_list_type::value_type include_value_type;

    public:
        inline include_paths() : was_sys_include_path(false), current_dir(), current_rel_dir() {}

        bool add_include_path(char const* path_, bool is_system = false)
        {
            return add_include_path(path_, (is_system || was_sys_include_path) ?
                system_include_paths : user_include_paths);
        }

        void set_sys_include_delimiter() { was_sys_include_path = true; }

        bool find_include_file(std::string& s, std::string& dir, bool is_system, char const* current_file) const;

        system::path get_current_directory() const
        {
            return current_dir;
        }
        void set_current_directory(char const* path_)
        {
            namespace fs = nbl::system;
            fs::path filepath(path_);
            fs::path filename = current_dir.is_absolute() ? filepath : (current_dir / filepath);
            current_rel_dir.clear();
            current_rel_dir = filepath.parent_path();
            current_dir = filename.parent_path();
        }

    protected:
        bool find_include_file(std::string& s, std::string& dir,
            include_list_type const& pathes, char const*) const;
        bool add_include_path(char const* path_, include_list_type& pathes_);

    private:
        include_list_type user_include_paths;
        include_list_type system_include_paths;
        bool was_sys_include_path;          // saw a set_sys_include_delimiter()
        system::path current_dir;
        system::path current_rel_dir;
};

template <
    typename IteratorT,
    typename LexIteratorT = boost::wave::cpplexer::lex_iterator<boost::wave::cpplexer::lex_token<>>,
    typename InputPolicyT = load_file_or_builtin_to_string,
    typename HooksT = custom_preprocessing_hooks,
    typename DerivedT = boost::wave::this_type
>
class context : private boost::noncopyable
{
    private:
        typedef typename mpl::if_<
            boost::is_same<DerivedT, this_type>, context, DerivedT
        >::type actual_context_type;

    public:
        // public typedefs
        typedef typename LexIteratorT::token_type       token_type;
        typedef typename token_type::string_type        string_type;

        typedef IteratorT                               target_iterator_type;
        typedef LexIteratorT                            lexer_type;
        typedef pp_iterator<context>                    iterator_type;

        typedef InputPolicyT                            input_policy_type;
        typedef typename token_type::position_type      position_type;

        // type of a token sequence
        typedef std::list<token_type, boost::fast_pool_allocator<token_type> >
            token_sequence_type;
        // type of the policies
        typedef HooksT                                  hook_policy_type;

    private:
        // stack of shared_ptr's to the pending iteration contexts
        typedef boost::shared_ptr<base_iteration_context<context, lexer_type> >
            iteration_ptr_type;
        typedef boost::wave::util::iteration_context_stack<iteration_ptr_type>
            iteration_context_stack_type;
        typedef typename iteration_context_stack_type::size_type iter_size_type;

        context* this_() { return this; }           // avoid warning in constructor

    public:
        context(target_iterator_type const& first_, target_iterator_type const& last_,
            char const* fname = "<Unknown>", HooksT const& hooks_ = HooksT())
            : first(first_), last(last_), filename(fname)
            , has_been_initialized(false)
            , current_relative_filename(fname)
            , macros(*this_())
            , language(language_support(
                support_cpp
                | support_option_convert_trigraphs
                | support_option_emit_line_directives
                | support_option_emit_pragma_directives
                | support_option_insert_whitespace
            ))
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
        bool add_include_path(char const* path_)
        {
            return includes.add_include_path(path_, false);
        }
        bool add_sysinclude_path(char const* path_)
        {
            return includes.add_include_path(path_, true);
        }
        void set_sysinclude_delimiter() { includes.set_sys_include_delimiter(); }
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
        void set_language(boost::wave::language_support language_,
            bool reset_macros = true)
        {
            language = language_;
            if (reset_macros)
                reset_macro_definitions();
        }
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
        hook_policy_type& get_hooks() { return hooks; }
        hook_policy_type const& get_hooks() const { return hooks; }

        // return type of actually used context type (might be the derived type)
        actual_context_type& derived()
        {
            return *static_cast<actual_context_type*>(this);
        }
        actual_context_type const& derived() const
        {
            return *static_cast<actual_context_type const*>(this);
        }

        // return the directory of the currently preprocessed file
        nbl::system::path get_current_directory() const
        {
            return includes.get_current_directory();
        }

#if !defined(BOOST_NO_MEMBER_TEMPLATE_FRIENDS)
    protected:
        friend class boost::wave::pp_iterator<context>;
        friend class boost::wave::impl::pp_iterator_functor<context>;
        friend class boost::wave::util::macromap<context>;
#endif

        // make sure the context has been initialized
        void init_context()
        {
            if (!has_been_initialized) {
                std::string fname(filename);
                if (filename != "<Unknown>" && filename != "<stdin>") {
                    nbl::system::path fpath(filename);
                    fname = fpath.string();
                    includes.set_current_directory(fname.c_str());
                }
                has_been_initialized = true;  // execute once
            }
        }

        template <typename IteratorT2>
        bool is_defined_macro(IteratorT2 const& begin, IteratorT2 const& end) const
        {
            return macros.is_defined(begin, end);
        }

        // maintain include paths (helper functions)
        void set_current_directory(char const* path_)
        {
            includes.set_current_directory(path_);
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
        void set_current_relative_filename(char const* real_name)
        {
            current_relative_filename = real_name;
        }
        std::string const& get_current_relative_filename() const
        {
            return current_relative_filename;
        }

        bool find_include_file(std::string& s, std::string& d, bool is_system,
            char const* current_file) const
        {
            return includes.find_include_file(s, d, is_system, current_file);
        }

    private:
        // the main input stream
        target_iterator_type first;         // underlying input stream
        target_iterator_type last;
        std::string filename;               // associated main filename
        bool has_been_initialized;          // set cwd once
        std::string current_relative_filename;        // real relative name of current preprocessed file

        boost::wave::util::if_block_stack ifblocks;   // conditional compilation contexts
        include_paths includes;    // lists of include directories to search
        iteration_context_stack_type iter_ctxs;       // iteration contexts
        macromap_type macros;                         // map of defined macros
        boost::wave::language_support language;       // supported language/extensions
        hook_policy_type hooks;                       // hook policy instance
};

}


template<> inline bool boost::wave::impl::pp_iterator_functor<nbl::wave::context<std::string::iterator>>::on_include_helper(char const* f, char const* s, bool is_system, bool include_next)
{
    namespace fs = boost::filesystem;

    // try to locate the given file, searching through the include path lists
    std::string file_path(s);
    std::string dir_path;
    char const* current_name = 0; // never try to match current file name

    // call the 'found_include_directive' hook function
    if (ctx.get_hooks().found_include_directive(ctx.derived(), f, include_next))
        return true;    // client returned false: skip file to include

    file_path = util::impl::unescape_lit(file_path);
    std::string native_path_str;

    if (!ctx.get_hooks().locate_include_file(ctx, file_path, is_system,
        current_name, dir_path, native_path_str))
    {
        BOOST_WAVE_THROW_CTX(ctx, preprocess_exception, bad_include_file,
            file_path.c_str(), act_pos);
        return false;
    }

    // test, if this file is known through a #pragma once directive
    {
        // the new include file determines the actual current directory
        ctx.set_current_directory(native_path_str.c_str());

        // preprocess the opened file
        boost::shared_ptr<base_iteration_context_type> new_iter_ctx(
            new iteration_context_type(ctx, native_path_str.c_str(), act_pos,
                boost::wave::enable_prefer_pp_numbers(ctx.get_language()),
                is_system ? base_iteration_context_type::system_header :
                base_iteration_context_type::user_header));

        // call the include policy trace function
        ctx.get_hooks().opened_include_file(ctx.derived(), dir_path, file_path,
            is_system);

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

        ctx.set_current_relative_filename(dir_path.c_str());
        iter_ctx->real_relative_filename = dir_path.c_str();

        act_pos.set_line(iter_ctx->line);
        act_pos.set_column(0);
    }
    return true;
}
#endif