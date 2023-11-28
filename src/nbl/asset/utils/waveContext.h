// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_ASSET_WAVE_CONTEXT_H_INCLUDED_
#define _NBL_ASSET_WAVE_CONTEXT_H_INCLUDED_
//! This file is not supposed to be included in user-accesible header files

#include <boost/wave.hpp>
#include <boost/wave/cpplexer/cpp_lex_token.hpp>
#include <boost/wave/cpplexer/cpp_lex_iterator.hpp>

namespace nbl::wave
{
using namespace boost;
using namespace boost::wave;
using namespace boost::wave::util;


class include_paths
{
    private:
        typedef std::list<std::pair<system::path, std::string> >
            include_list_type;
        typedef include_list_type::value_type include_value_type;

#if BOOST_WAVE_SUPPORT_PRAGMA_ONCE != 0
        typedef boost::wave::util::bidirectional_map<std::string, std::string>::type
            pragma_once_set_type;
#endif

    public:
        include_paths()
            : was_sys_include_path(false),
            current_dir(),
            current_rel_dir()
        {}

        bool add_include_path(char const* path_, bool is_system = false)
        {
            return add_include_path(path_, (is_system || was_sys_include_path) ?
                system_include_paths : user_include_paths);
        }

        void set_sys_include_delimiter() { was_sys_include_path = true; }

        bool find_include_file(std::string& s, std::string& dir, bool is_system,
            char const* current_file) const;
        system::path get_current_directory() const
        {
            return current_dir;
        }
        void set_current_directory(char const* path_) {
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

#if BOOST_WAVE_SUPPORT_PRAGMA_ONCE != 0
    public:
        bool has_pragma_once(std::string const& filename)
        {
            using boost::multi_index::get;
            return get<boost::wave::util::from>(pragma_once_files).find(filename) != pragma_once_files.end();
        }
        bool add_pragma_once_header(std::string const& filename,
            std::string const& guard_name)
        {
            typedef pragma_once_set_type::value_type value_type;
            return pragma_once_files.insert(value_type(filename, guard_name)).second;
        }
        bool remove_pragma_once_header(std::string const& guard_name)
        {
            typedef pragma_once_set_type::index_iterator<boost::wave::util::to>::type to_iterator;
            typedef std::pair<to_iterator, to_iterator> range_type;

            range_type r = pragma_once_files.get<boost::wave::util::to>().equal_range(guard_name);
            if (r.first != r.second) {
                using boost::multi_index::get;
                get<boost::wave::util::to>(pragma_once_files).erase(r.first, r.second);
                return true;
            }
            return false;
        }

    private:
        pragma_once_set_type pragma_once_files;
#endif
    };

template <
    typename IteratorT,
    typename LexIteratorT = lex_iterator_t,
    typename InputPolicyT = nbl::asset::impl::load_file_or_builtin_to_string,
    typename HooksT = nbl::asset::impl::custom_preprocessing_hooks,
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
#if BOOST_WAVE_SUPPORT_PRAGMA_ONCE != 0
            , current_filename(fname)
#endif
            , current_relative_filename(fname)
            , macros(*this_())
            , language(language_support(
                support_cpp
                | support_option_convert_trigraphs
                | support_option_emit_line_directives
#if BOOST_WAVE_SUPPORT_PRAGMA_ONCE != 0
                | support_option_include_guard_detection
#endif
#if BOOST_WAVE_EMIT_PRAGMA_DIRECTIVES != 0
                | support_option_emit_pragma_directives
#endif
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
#if BOOST_WAVE_ENABLE_COMMANDLINE_MACROS != 0
        template <typename StringT>
        bool add_macro_definition(StringT macrostring, bool is_predefined = false)
        {
            return boost::wave::util::add_macro_definition(*this,
                util::to_string<std::string>(macrostring), is_predefined,
                get_language());
        }
#endif
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

#if BOOST_WAVE_SUPPORT_PRAGMA_ONCE != 0
            // ensure this gets removed from the list of include guards as well
            includes.remove_pragma_once_header(
                util::to_string<std::string>(name));
#endif
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
#if BOOST_WAVE_SUPPORT_PRAGMA_ONCE != 0
        // support for #pragma once
        // maintain the real name of the current preprocessed file
        void set_current_filename(char const* real_name)
        {
            current_filename = real_name;
        }
        std::string const& get_current_filename() const
        {
            return current_filename;
        }

        // maintain the list of known headers containing #pragma once
        bool has_pragma_once(std::string const& filename_)
        {
            return includes.has_pragma_once(filename_);
        }
        bool add_pragma_once_header(std::string const& filename_,
            std::string const& guard_name)
        {
            get_hooks().detected_include_guard(derived(), filename_, guard_name);
            return includes.add_pragma_once_header(filename_, guard_name);
        }
        bool add_pragma_once_header(token_type const& pragma_,
            std::string const& filename_)
        {
            get_hooks().detected_pragma_once(derived(), pragma_, filename_);
            return includes.add_pragma_once_header(filename_,
                "__BOOST_WAVE_PRAGMA_ONCE__");
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

        bool find_include_file(std::string& s, std::string& d, bool is_system,
            char const* current_file) const
        {
            return includes.find_include_file(s, d, is_system, current_file);
        }

#if BOOST_WAVE_SERIALIZATION != 0
    public:
        BOOST_STATIC_CONSTANT(unsigned int, version = 0x10);
        BOOST_STATIC_CONSTANT(unsigned int, version_mask = 0x0f);

    private:
        friend class boost::serialization::access;
        template<class Archive>
        void save(Archive& ar, const unsigned int version) const
        {
            using namespace boost::serialization;

            string_type cfg(BOOST_PP_STRINGIZE(BOOST_WAVE_CONFIG));
            string_type kwd(BOOST_WAVE_PRAGMA_KEYWORD);
            string_type strtype(BOOST_PP_STRINGIZE((BOOST_WAVE_STRINGTYPE)));
            ar& make_nvp("config", cfg);
            ar& make_nvp("pragma_keyword", kwd);
            ar& make_nvp("string_type", strtype);

            ar& make_nvp("language_options", language);
            ar& make_nvp("macro_definitions", macros);
            ar& make_nvp("include_settings", includes);
        }
        template<class Archive>
        void load(Archive& ar, const unsigned int loaded_version)
        {
            using namespace boost::serialization;
            if (version != (loaded_version & ~version_mask)) {
                BOOST_WAVE_THROW_CTX((*this), preprocess_exception,
                    incompatible_config, "cpp_context state version",
                    get_main_pos());
                return;
            }

            // check compatibility of the stored information
            string_type config, pragma_keyword, string_type_str;

            // BOOST_PP_STRINGIZE(BOOST_WAVE_CONFIG)
            ar& make_nvp("config", config);
            if (config != BOOST_PP_STRINGIZE(BOOST_WAVE_CONFIG)) {
                BOOST_WAVE_THROW_CTX((*this), preprocess_exception,
                    incompatible_config, "BOOST_WAVE_CONFIG", get_main_pos());
                return;
            }

            // BOOST_WAVE_PRAGMA_KEYWORD
            ar& make_nvp("pragma_keyword", pragma_keyword);
            if (pragma_keyword != BOOST_WAVE_PRAGMA_KEYWORD) {
                BOOST_WAVE_THROW_CTX((*this), preprocess_exception,
                    incompatible_config, "BOOST_WAVE_PRAGMA_KEYWORD",
                    get_main_pos());
                return;
            }

            // BOOST_PP_STRINGIZE((BOOST_WAVE_STRINGTYPE))
            ar& make_nvp("string_type", string_type_str);
            if (string_type_str != BOOST_PP_STRINGIZE((BOOST_WAVE_STRINGTYPE))) {
                BOOST_WAVE_THROW_CTX((*this), preprocess_exception,
                    incompatible_config, "BOOST_WAVE_STRINGTYPE", get_main_pos());
                return;
            }

            try {
                // read in the useful bits
                ar& make_nvp("language_options", language);
                ar& make_nvp("macro_definitions", macros);
                ar& make_nvp("include_settings", includes);
            }
            catch (boost::wave::preprocess_exception const& e) {
                // catch version mismatch exceptions and call error handler
                get_hooks().throw_exception(derived(), e);
            }
        }
        BOOST_SERIALIZATION_SPLIT_MEMBER()
#endif

    private:
        // the main input stream
        target_iterator_type first;         // underlying input stream
        target_iterator_type last;
        std::string filename;               // associated main filename
        bool has_been_initialized;          // set cwd once
#if BOOST_WAVE_SUPPORT_PRAGMA_ONCE != 0
        std::string current_filename;       // real name of current preprocessed file
#endif
        std::string current_relative_filename;        // real relative name of current preprocessed file

        boost::wave::util::if_block_stack ifblocks;   // conditional compilation contexts
        include_paths includes;    // lists of include directories to search
        iteration_context_stack_type iter_ctxs;       // iteration contexts
        macromap_type macros;                         // map of defined macros
        boost::wave::language_support language;       // supported language/extensions
        hook_policy_type hooks;                       // hook policy instance
    };

}
#endif