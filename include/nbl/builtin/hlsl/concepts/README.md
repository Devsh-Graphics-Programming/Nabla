# Constraints: one source, two compilers

Part of the [Nabla HLSL builtin library](../README.md).

Every header under `nbl/builtin/hlsl` is compiled twice: by dxc as HLSL 202x[[17]](#r17)[[18]](#r18), and by a C++20 host compiler when the same header is included from C++. C++20 has `concept` and `requires`. HLSL has neither, only templates, so the only way to constrain a template there is SFINAE[[8]](#r8) with `enable_if`[[9]](#r9) and a spare `void` template parameter. The macros in [concepts.hlsl](../concepts.hlsl) and [impl/base.hlsl](impl/base.hlsl) let you write a constraint once and get the right one of the two.

The selector is `__cpp_concepts`: defined by the C++20 compiler, never by dxc.

- [Prerequisites](#prerequisites)
- [The example, as you write it](#the-example-as-you-write-it)
- [What the C++20 compiler sees](#what-the-c20-compiler-sees)
- [What dxc sees](#what-dxc-sees)
- [Why the two paths differ](#why-the-two-paths-differ)
- [Macro reference](#macro-reference)
- [Rules](#rules)
- [Errors you will see](#errors-you-will-see)
- [Expression concepts](#expression-concepts)
- [Resources](#resources)
- [TODO: forms not explained yet](#todo-forms-not-explained-yet)

## Prerequisites

The page assumes you already know:

- class and function templates, and what a primary template is [[1]](#r1)[[6]](#r6)
- default template arguments are filled in at the use site [[2]](#r2)
- partial specialization and how the compiler picks one [[3]](#r3)
- function templates overload and cannot be partially specialized [[4]](#r4)[[7]](#r7)
- SFINAE and `enable_if` [[8]](#r8)[[9]](#r9)[[14]](#r14)
- the detection idiom: `void_t`, `decltype`, `declval` [[10]](#r10)[[11]](#r11)[[12]](#r12)[[13]](#r13)[[15]](#r15)
- C++20 `requires` and `concept` [[16]](#r16)

The numbers point into the [resources](#resources) at the end.

## The example, as you write it

Three things that come up constantly: a struct that only makes sense for some arguments, a struct with alternative bodies chosen by a property of the type, and a constrained function. This compiles as is with nsc, live here: [godbolt.devsh.eu/z/68ddn3](https://godbolt.devsh.eu/z/68ddn3).

```cpp
#include <nbl/builtin/hlsl/cpp_compat.hlsl>
#include <nbl/builtin/hlsl/concepts.hlsl>
#include <nbl/builtin/hlsl/type_traits.hlsl>
using namespace nbl::hlsl;


// a simple config struct
struct Config
{
    NBL_CONSTEXPR_STATIC_INLINE uint32_t WorkgroupSize = 64;
};

// primary template: value = false for anything
template<class C> struct is_configuration : false_type {};

// specialize it for Config, so is_configuration<Config>::value is true
template<>        struct is_configuration<Config> : true_type {};

// the usual _v shorthand so the conditions below read nicely
template<class C> NBL_CONSTEXPR bool is_configuration_v = is_configuration<C>::value;


// 1. a primary that only accepts a configuration
template<class Cfg, class BinOp NBL_PRIMARY_REQUIRES(is_configuration_v<Cfg>)
struct reduction
{
    static uint32_t run(uint32_t x) { return x * Cfg::WorkgroupSize; }
};


// 2. alternative bodies picked by a property, not a shape
// primary is declared only, NBL_STRUCT_CONSTRAINABLE is what lets the specializations carry a condition
template<class T NBL_STRUCT_CONSTRAINABLE>
struct length_helper;

// the same condition goes in TOP and in BOT
template<class T> NBL_PARTIAL_REQ_TOP(is_vector_v<T>)
struct length_helper<T NBL_PARTIAL_REQ_BOT(is_vector_v<T>) >
{
    static float call(T v) { return sqrt(dot(v, v)); }
};

template<class T> NBL_PARTIAL_REQ_TOP(!is_vector_v<T>)
struct length_helper<T NBL_PARTIAL_REQ_BOT(!is_vector_v<T>) >
{
    static float call(T v) { return abs(v); }
};


// 3. a constrained function
template<class T NBL_FUNC_REQUIRES(is_integral_v<T>)
T roundUpToPoT(T v)
{
    T p = 1;
    while (p < v)
        p <<= 1;
    return p;
}
```

Use:

```cpp
// happy path
reduction<Config, plus_t>::run(2);                 // 128
length_helper<float32_t2>::call(float32_t2(3, 4)); // 5, the vector specialization
length_helper<float>::call(-7.0f);                 // 7, the scalar specialization
roundUpToPoT(37u);                                 // 64

// constraint violated: compile error on both paths, pointing at these lines
reduction<float, plus_t>::run(2);                  // float is not a configuration
roundUpToPoT(1.5f);                                // float is not integral
```

Note the odd punctuation: `NBL_PRIMARY_REQUIRES` and `NBL_FUNC_REQUIRES` replace the closing `>` of the template parameter list, and `NBL_PARTIAL_REQ_BOT` sits inside the pattern before its closing `>`. The macros have to own those brackets because the two expansions put different text around them.

## What the C++20 compiler sees

`__cpp_concepts` is defined, so the macros expand to `requires` clauses and nothing else. `NBL_STRUCT_CONSTRAINABLE` and `NBL_PARTIAL_REQ_BOT` vanish. This form compiles with MSVC `/std:c++20` and gives the same four results.

```cpp
template<class Cfg, class BinOp> requires (is_configuration_v<Cfg>)
struct reduction { ... };


// NBL_STRUCT_CONSTRAINABLE and NBL_PARTIAL_REQ_BOT are gone, the pattern is just <T>
template<class T>
struct length_helper;

template<class T> requires (is_vector_v<T>)
struct length_helper<T> { ... };

template<class T> requires (!is_vector_v<T>)
struct length_helper<T> { ... };


template<class T> requires (is_integral_v<T>)
T roundUpToPoT(T v) { ... }
```

Two partial specializations with the identical pattern `<T>` are allowed because their constraints differ; the constraint is part of the specialization's identity[[16]](#r16).

## What dxc sees

`__cpp_concepts` is not defined, so the macros expand to `enable_if` in the only places HLSL can evaluate it. `NBL_PARTIAL_REQ_TOP` vanishes.

```cpp
template<class Cfg, class BinOp, typename __requires = enable_if_t<(is_configuration_v<Cfg>), void> >
struct reduction { ... };


// the `void` slot: length_helper<float> is really length_helper<float, void>
template<class T, typename __requires = void>
struct length_helper;

// NBL_PARTIAL_REQ_TOP is gone, the condition is a second pattern argument that is `void` when true and nothing when false
template<class T>
struct length_helper<T, enable_if_t<(is_vector_v<T>), void> > { ... };

template<class T>
struct length_helper<T, enable_if_t<(!is_vector_v<T>), void> > { ... };


// a non-type parameter whose type only exists when the condition is true, `= true` is filler
template<class T, enable_if_t<(is_integral_v<T>), bool> = true>
T roundUpToPoT(T v) { ... }
```

How each one works:

- **`reduction`**: the extra parameter's default is `enable_if_t<cond, void>`. When the condition is false that type does not exist, so forming `reduction<float, plus_t>` fails while filling in the default argument. When true, the parameter is `void` and nobody notices it.
- **`length_helper`**: the primary gets a `void` slot. `length_helper<float32_t2>` means `length_helper<float32_t2, void>`; defaults are filled in before anything else happens[[2]](#r2). Each specialization is a pattern on that slot: `enable_if_t<cond, void>` is `void` when the condition holds, so the pattern matches, and does not exist otherwise, so the pattern is silently dropped[[8]](#r8).
- **`roundUpToPoT`**: functions cannot be partially specialized[[3]](#r3)[[7]](#r7) and a defaulted `typename` parameter is not part of a function's signature[[4]](#r4), so the condition goes into the *type* of a non-type parameter. `enable_if_t<cond, bool>` either is `bool` (and `= true` fills it) or does not exist (and the overload drops out of resolution[[5]](#r5)[[8]](#r8)).

## Why the two paths differ

The constraint ends up in a different place for partial specializations, and that is the whole reason there are five macros instead of one.

| construct | C++20 condition sits | HLSL condition sits | same place? |
|---|---|---|---|
| primary struct | after `template<...>`, before the name | last slot inside `template<...>` | yes |
| partial specialization | after `template<...>`, before the name | inside the pattern, after the name | **no** |
| function | after `template<...>`, before the return type | last slot inside `template<...>` | yes |

Where both land in the same place, one macro covers both paths (`NBL_PRIMARY_REQUIRES`, `NBL_FUNC_REQUIRES`). For partial specializations the C++20 clause is above the name and the HLSL `enable_if` is below it; a macro cannot move text across the name, so one macro expands on each side and the condition is written twice.

Underneath that are three ordinary language rules, which hold for plain code as much as for templates:

1. A class name is defined once. A class template therefore has one primary, and partial specializations are its only way to provide alternative bodies[[3]](#r3). That is why the primary needs a slot: there is nothing else for the specializations to match against.
2. A function name can be defined many times. Function templates therefore use overloading and have no partial specializations[[4]](#r4)[[7]](#r7).
3. Two functions are different only if their signatures differ, and a default argument is not part of a signature[[4]](#r4). A `typename = enable_if_t<...>` slot on a function is invisible to overloading, so two such overloads are a redefinition. A parameter's *type* is part of the signature, hence `enable_if_t<cond, bool> = true`.

## Macro reference

Defined in [concepts.hlsl](../concepts.hlsl) and [impl/base.hlsl](impl/base.hlsl).

| macro | put it | C++20 expansion | HLSL expansion |
|---|---|---|---|
| `NBL_STRUCT_CONSTRAINABLE` | before the closing `>` of a primary you will specialize with `NBL_PARTIAL_REQ_BOT` | nothing | `,typename __requires=void` |
| `NBL_PRIMARY_REQUIRES(c)` | in place of the closing `>` of a constrained primary | `> requires (c)` | `,typename __requires=enable_if_t<(c),void> >` |
| `NBL_PARTIAL_REQ_TOP(c)` | after `template<...>` of a partial specialization | `requires (c)` | nothing |
| `NBL_PARTIAL_REQ_BOT(c)` | before the closing `>` of the specialization's pattern | nothing | `,enable_if_t<(c),void>` |
| `NBL_FUNC_REQUIRES(c)` | in place of the closing `>` of a function template | `> requires (c)` | `,enable_if_t<(c),bool> = true>` |
| `NBL_BOOL_CONCEPT` | where you would write `concept` | `concept` | `NBL_CONSTEXPR bool` |

`NBL_BOOL_CONCEPT` is how the simple named concepts in [core.hlsl](core.hlsl) are written, for example `template<typename T> NBL_BOOL_CONCEPT Integral = is_integral_v<T>;`. On the C++20 path it is a real concept, on the HLSL path a `bool` variable template, and either can be used as the argument of the `*_REQUIRES` macros.

## Rules

- **TOP and BOT conditions must be identical.** Each path only sees one of them, so a mismatch gives you different behaviour in C++ and in the shader and no diagnostic.
- **A primary you intend to specialize with `NBL_PARTIAL_REQ_BOT` must carry `NBL_STRUCT_CONSTRAINABLE` (or `NBL_PRIMARY_REQUIRES`).** Otherwise the HLSL pattern has one argument more than the primary has parameters.
- **Keep the conditions of one specialization set mutually exclusive.** `enable_if` does not take part in "more specialized" ordering[[3]](#r3), so a type that satisfies two conditions is an ambiguity error, and nothing warns you until such a type shows up. The same holds on the C++20 path for bare boolean conditions; only concepts that subsume each other are ordered[[16]](#r16).
- **A shape beats a condition.** A specialization on `vector<T,N>` or a full specialization on `float` is more specialized than `<T, enable_if_t<...>>` and wins without ambiguity[[3]](#r3). Use a shape when one exists and a condition only for properties a shape cannot express.
- **Never spell the slot yourself.** Do not name `__requires`, do not pass a second argument, do not change the default from `void`. Every specialization in the codebase is written against `void`.
- **The slot is a real template argument.** `plus<uint32_t>` *is* `plus<uint32_t, void>` on the HLSL path. It shows up anywhere template arguments are visible, and it is why an HLSL instantiation has one more argument than the same C++ one. It also currently breaks `-fspv-debug=vulkan`: dxc asserts while emitting debug type information for a `void` template argument, see [DirectXShaderCompiler#9082](https://github.com/microsoft/DirectXShaderCompiler/issues/9082).

## Errors you will see

| message (dxc) | meaning |
|---|---|
| `no type named 'type' in 'nbl::hlsl::enable_if<false, void>'` followed by `in instantiation of default argument for 'X<...>'` | a `NBL_PRIMARY_REQUIRES` condition failed for that instantiation; the note names the use site |
| `no matching function for call to 'f'` with `candidate template ignored: disabled by 'enable_if' [with T = ...]` | a `NBL_FUNC_REQUIRES` condition failed, and no other overload accepts the type |
| `implicit instantiation of undefined template 'X<T, void>'` | the primary is declared only and no partial specialization's condition held for `T` |
| `ambiguous partial specializations of 'X<T, void>'` | two specialization conditions are both true for `T` |
| `too many template arguments for class template 'X'` at a `NBL_PARTIAL_REQ_BOT` site | the primary is missing `NBL_STRUCT_CONSTRAINABLE` |

## Expression concepts

Say you want a concept `Loadable<A>` that is true when `A` has a nested `value_type` and `a.load(ix)` returns it. In plain C++20 you would write (compiles with MSVC, `/std:c++20`):

```cpp
template<typename A>
concept Loadable = requires(A a, uint32_t ix)
{
    typename A::value_type;
    { a.load(ix) } -> std::same_as<typename A::value_type>;
};

// In simple words: Loadable<A> is true if `A::value_type` exists and `a.load(ix)` would compile
// and return exactly that type. Pure compile-time check, `a` and `ix` are never real objects.
```

HLSL has no `requires` expression, so Nabla has to build the same check for dxc out of `decltype`, `declval` and the `void` slot. That is what the `NBL_CONCEPT_BEGIN` / `NBL_CONCEPT_END` block does. The same concept in Nabla, compiles with nsc, live here: [godbolt.devsh.eu/z/6o4fMn](https://godbolt.devsh.eu/z/6o4fMn):

```cpp
#define NBL_CONCEPT_NAME Loadable
#define NBL_CONCEPT_TPLT_PRM_KINDS (typename)           // kinds and names of the concept's template parameters,
#define NBL_CONCEPT_TPLT_PRM_NAMES (A)                  // as Boost.PP sequences, one entry per parameter
#define NBL_CONCEPT_PARAM_0 (a,A)                       // the `requires(A a, uint32_t ix)` part: (name, type)
#define NBL_CONCEPT_PARAM_1 (ix,uint32_t)
NBL_CONCEPT_BEGIN(2)                                    // 2 = number of NBL_CONCEPT_PARAM_n above
#define a NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_0       // after BEGIN, make `a` and `ix` usable in expressions
#define ix NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_1
NBL_CONCEPT_END(
    ((NBL_CONCEPT_REQ_TYPE)(A::value_type))                                                       // typename A::value_type;
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((a.load(ix)), ::nbl::hlsl::is_same_v, typename A::value_type)) // { a.load(ix) } -> same_as<value_type>
);
#undef ix
#undef a
#include <nbl/builtin/hlsl/concepts/__end.hlsl>         // undefines NBL_CONCEPT_NAME & co for the next concept

// use it like any other bool concept
template<typename A NBL_FUNC_REQUIRES(Loadable<A>)
float first(A a) { return a.load(0u); }
```

Checked against three types:

```cpp
struct Good        { using value_type = float; float    load(uint32_t ix) { return 1.0f; } };
struct WrongReturn { using value_type = float; uint32_t load(uint32_t ix) { return 1u; } };
struct NoLoad      { using value_type = float; };

Loadable<Good>          // 1
Loadable<WrongReturn>   // 0
Loadable<NoLoad>        // 0
```

**What the C++20 compiler sees.** `NBL_CONCEPT_BEGIN` opens a real concept with a `requires` expression, each `NBL_CONCEPT_REQ_*` becomes one line of its body, `NBL_CONCEPT_PARAM_T` expands to the parameter's name:

```cpp
template<typename A>
concept Loadable = requires (A a, uint32_t ix)
{
    typename A::value_type;
    { (a.load(ix)) };
    requires ::nbl::hlsl::is_same_v<decltype (a.load(ix)), typename A::value_type>;
};
```

The last line is a *nested* requirement.

**What dxc sees.** `NBL_CONCEPT_BEGIN` opens a namespace. For every requirement `NBL_CONCEPT_END` emits one detector struct with the `void` slot: a primary deriving from `false_type`, and a partial specialization deriving from `true_type` whose slot pattern only exists when the requirement holds. `NBL_CONCEPT_PARAM_T` expands to `declval<Type>()`, which is how `a.load(ix)` is spelled without a real object. The concept itself is the `&&` of all detectors. This expansion compiles with nsc and gives the same three results:

```cpp
namespace __concept__Loadable
{
// NBL_CONCEPT_REQ_TYPE: the slot pattern is make_void_t<the type>, which is void if the type exists
template<typename A, typename = void>
struct __requirement0 : false_type {};
template<typename A>
struct __requirement0<A, make_void_t<typename A::value_type> > : true_type {};

// NBL_CONCEPT_REQ_EXPR_RET_TYPE: the slot pattern is enable_if_t<Trait<decltype(expr), Args...>>
template<typename A, typename = void>
struct __requirement1 : false_type {};
template<typename A>
struct __requirement1<A, enable_if_t<is_same_v<decltype(declval<A>().load(declval<uint32_t>())), typename A::value_type> > > : true_type {};
}

template<typename A>
NBL_CONSTEXPR bool Loadable = __concept__Loadable::__requirement0<A>::value && __concept__Loadable::__requirement1<A>::value;
```

So an expression concept on the HLSL path is the `length_helper` pattern from the main example with the condition replaced by "does this `decltype` exist". The four requirement kinds and their slot patterns on the dxc path:

| kind | C++20 line | dxc slot pattern |
|---|---|---|
| `NBL_CONCEPT_REQ_TYPE` `(X)` | `typename X;` | `make_void_t<typename X>` |
| `NBL_CONCEPT_REQ_EXPR` `(E)` | `E;` | `make_void_t<decltype(E)>` |
| `NBL_CONCEPT_REQ_EXPR_RET_TYPE` `((E), Trait, Args...)` | `{E}; requires Trait<decltype(E), Args...>;` | `enable_if_t<Trait<decltype(E), Args...> >` |
| `NBL_CONCEPT_REQ_TYPE_ALIAS_CONCEPT` `(Concept, Args...)` | `requires Concept<Args...>;` | `enable_if_t<Concept<Args...> >` |

A real multi-parameter one to compare against is `LoadableImage` in [accessors/loadable_image.hlsl](accessors/loadable_image.hlsl). For a one-expression concept without named parameters, `NBL_VALID_EXPRESSION(Name, (T)(U), expr)` in [concepts.hlsl](../concepts.hlsl) does the same with less ceremony. The sequence syntax is Boost.Preprocessor[[19]](#r19).

## Resources

Templates and specialization
1. <a id="r1"></a>[cppreference: Templates](https://en.cppreference.com/w/cpp/language/templates)
2. <a id="r2"></a>[cppreference: Template parameters, "Default template arguments"](https://en.cppreference.com/w/cpp/language/template_parameters#Default_template_arguments)
3. <a id="r3"></a>[cppreference: Partial template specialization](https://en.cppreference.com/w/cpp/language/partial_specialization)
4. <a id="r4"></a>[cppreference: Function template](https://en.cppreference.com/w/cpp/language/function_template), sections "signature" and "overloading"
5. <a id="r5"></a>[cppreference: Overload resolution](https://en.cppreference.com/w/cpp/language/overload_resolution)
6. <a id="r6"></a>Arthur O'Dwyer, *Template Normal Programming*, CppCon 2016: [part 1](https://www.youtube.com/watch?v=vwrXHznaYLA), [part 2](https://www.youtube.com/watch?v=VIz6xBvwYd8)
7. <a id="r7"></a>Herb Sutter, [*Why Not Specialize Function Templates?*](http://www.gotw.ca/publications/mill17.htm)

SFINAE and detection

8. <a id="r8"></a>[cppreference: SFINAE](https://en.cppreference.com/w/cpp/language/sfinae)
9. <a id="r9"></a>[std::enable_if](https://en.cppreference.com/w/cpp/types/enable_if)
10. <a id="r10"></a>[std::void_t](https://en.cppreference.com/w/cpp/types/void_t)
11. <a id="r11"></a>[std::declval](https://en.cppreference.com/w/cpp/utility/declval)
12. <a id="r12"></a>[decltype](https://en.cppreference.com/w/cpp/language/decltype)
13. <a id="r13"></a>[std::experimental::is_detected](https://en.cppreference.com/w/cpp/experimental/is_detected)
14. <a id="r14"></a>Jonathan Boccara, [*How to Make SFINAE Pretty, part 1*](https://www.fluentcpp.com/2018/05/15/make-sfinae-pretty-1-what-value-sfinae-brings-to-code/)
15. <a id="r15"></a>Walter E. Brown, *Modern Template Metaprogramming: A Compendium*, CppCon 2014: [part I](https://www.youtube.com/watch?v=Am2is2QCvxY), [part II](https://www.youtube.com/watch?v=a0FliKwcwXE)

C++20 concepts

16. <a id="r16"></a>[cppreference: Constraints and concepts](https://en.cppreference.com/w/cpp/language/constraints)

HLSL and tooling

17. <a id="r17"></a>[DXC wiki: HLSL 2021](https://github.com/microsoft/DirectXShaderCompiler/wiki/HLSL-2021)
18. <a id="r18"></a>[Announcing HLSL 2021](https://devblogs.microsoft.com/directx/announcing-hlsl-2021/)
19. <a id="r19"></a>[Boost.Preprocessor](https://www.boost.org/doc/libs/release/libs/preprocessor/doc/index.html), for reading the `NBL_CONCEPT_*` expansions

Try snippets with `tools/nsc/bin/nsc.exe -no-nbl-builtins -T cs_6_8 -E main -spirv -HV 202x -Fo out.spv file.hlsl`; drop `-no-nbl-builtins` to use Nabla headers.

## TODO: forms not explained yet

Other shapes found in the headers, each still to get its own section. Descriptions are from reading the code, not from compiling it (unverified).

- **Return-type `enable_if_t<cond, R>` on a function template** (unverified): [bit.hlsl:36-55](../bit.hlsl#L36-L55), [spirv_intrinsics/core.hlsl:159-178](../spirv_intrinsics/core.hlsl#L159-L178), [glsl_compat/core.hlsl:81](../glsl_compat/core.hlsl#L81), [glsl_compat/subgroup_arithmetic.hlsl:86-127](../glsl_compat/subgroup_arithmetic.hlsl#L86-L127).
- **Member function enabled through a dummy defaulted parameter** `template<typename S = T>` / `template<typename C = bool_constant<Flag>>` (unverified, the trick is explained in [[14]](#r14)): [bda/bda_accessor.hlsl:64](../bda/bda_accessor.hlsl#L64), [memory_accessor.hlsl:192](../memory_accessor.hlsl#L192), [bxdf/base/lambertian.hlsl:44-53](../bxdf/base/lambertian.hlsl#L44-L53), [bxdf/ndf/ggx.hlsl:187](../bxdf/ndf/ggx.hlsl#L187).
- **Inheriting from `enable_if_t<cond, Base>` as a `static_assert` stand-in**, a hard error not SFINAE (unverified): [functional.hlsl:69](../functional.hlsl#L69), [format/octahedral.hlsl:16](../format/octahedral.hlsl#L16), [format/shared_exp.hlsl:17](../format/shared_exp.hlsl#L17).
- **C++-only `requires` specializations behind `#ifndef __HLSL_VERSION`**, no `NBL_PARTIAL_REQ_BOT` needed (unverified): [cpp_compat/impl/intrinsics_impl.hlsl:300-480](../cpp_compat/impl/intrinsics_impl.hlsl#L300-L480) with `AUTO_SPECIALIZE_TRIVIAL_CASE_HELPER` at line 308, [tgmath/impl.hlsl:225-373](../tgmath/impl.hlsl#L225-L373), [functional.hlsl:153](../functional.hlsl#L153) next to its HLSL twin at line 165, [cpp_compat/matrix.hlsl:19](../cpp_compat/matrix.hlsl#L19).
- **`NBL_VALID_EXPRESSION` in practice** (unverified): [cpp_compat/impl/intrinsics_impl.hlsl:45](../cpp_compat/impl/intrinsics_impl.hlsl#L45), [spirv_intrinsics/core.hlsl:433](../spirv_intrinsics/core.hlsl#L433).
- **`NBL_CONCEPT_REQ_EXPR` and `NBL_CONCEPT_REQ_TYPE_ALIAS_CONCEPT` worked examples**, the table in [Expression concepts](#expression-concepts) is read off the macros for these two (unverified): [concepts.hlsl](../concepts.hlsl), users in [accessors/](accessors/) and [../bxdf/](../bxdf/).
- **`conditional_t` / `NBL_IF_CONSTEXPR` as dispatch that removes nothing** (unverified): [subgroup2/ballot.hlsl:35](../subgroup2/ballot.hlsl#L35), [bxdf/common.hlsl:470](../bxdf/common.hlsl#L470).
