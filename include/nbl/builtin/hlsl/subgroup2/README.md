# subgroup2 arithmetic

Not every device supports subgroup arithmetic (`VK_SUBGROUP_FEATURE_ARITHMETIC_BIT`). So `subgroup2::reduction`, `subgroup2::inclusive_scan` and `subgroup2::exclusive_scan` either use the native subgroup instructions or emulate them with subgroup shuffles. The choice is made at compile time from `device_capabilities::shaderSubgroupArithmetic`, which you set from the device's limits on the host.

Both paths combine values across one subgroup, and the rest of the configuration comes from the `ArithmeticParams` you pass in.

- [ArithmeticParams](#arithmeticparams)
- [What ItemsPerInvocation is](#what-itemsperinvocation-is)
- [How the specializations are chosen](#how-the-specializations-are-chosen)
- [Seeing it in SPIR-V](#seeing-it-in-spir-v)

## ArithmeticParams

```cpp
using config_t = subgroup2::Configuration<5>;                                            // SubgroupSize 32
using params_t = subgroup2::ArithmeticParams<config_t, plus<uint32_t>, 1, device_capabilities>;
subgroup2::inclusive_scan<params_t> scan;
const params_t::type_t result = scan(value);
```

| `ArithmeticParams` member | Comes from |
|---|---|
| `config_t` | `Configuration<SubgroupSizeLog2>`. Must match the subgroup size the pipeline runs with. |
| `binop_t`, `scalar_t` | the binop, e.g. `plus<uint32_t>`, and its `type_t` |
| `type_t` | `vector<scalar_t, ItemsPerInvocation>`: each invocation holds `ItemsPerInvocation` consecutive elements |
| `UseNativeIntrinsics` | `device_capabilities::shaderSubgroupArithmetic`, `false` when the member is missing or `device_capabilities` is `void` |

## What ItemsPerInvocation is

**Each thread holds N consecutive elements in one `vector<T, N>`, instead of one element. That's all it means.** (N is `ItemsPerInvocation`.)

With a subgroup of 32 and `ItemsPerInvocation = 2`, one subgroup covers 32 × 2 = 64 elements, and lane `k` holds elements `2k` and `2k+1`:

| lane | holds | element indices |
|---|---|---|
| 0 | `(e0, e1)` | 0, 1 |
| 1 | `(e2, e3)` | 2, 3 |
| ... | ... | ... |
| 31 | `(e62, e63)` | 62, 63 |

The subgroup still has 32 lanes, and an operation still needs only one cross-lane step. The extra items are combined serially inside each invocation.

For example, take an inclusive `plus` scan of 64 ones. The steps are the numbered comments in the [primary `inclusive_scan`](arithmetic_portability_impl.hlsl#L39-L72):

| | lane 0 | lane 1 | lane 2 | ... | lane 31 |
|---|---|---|---|---|---|
| holds | (1, 1) | (1, 1) | (1, 1) | ... | (1, 1) |
| (1) own items scanned | (1, **2**) | (1, **2**) | (1, **2**) | ... | (1, **2**) |
| (2) subgroup exclusive scan of the totals | 0 | 2 | 4 | ... | 62 |
| (3) combined with every item | (1, 2) | (3, 4) | (5, 6) | ... | (63, 64) |

`workgroup2::ArithmeticConfiguration`'s `ItemsPerInvocation_0` is this same value. It's the vector type workgroup data accessors read and write.

## How the specializations are chosen

The public structs in `arithmetic_portability.hlsl` inherit from `impl::<op><Params, BinOp, ItemsPerInvocation, UseNativeIntrinsics>`, defined in `arithmetic_portability_impl.hlsl`. **`ItemsPerInvocation` is what selects the primary template, not native support:**

| `ItemsPerInvocation` | native | Selected | What it does |
|---|---|---|---|
| not 1 | either | primary template | Combines the invocation's own items serially, then makes one `ItemsPerInvocation = 1` call (the exclusive scan for both scans, the reduction for `reduction`) on each invocation's total. See [What ItemsPerInvocation is](#what-itemsperinvocation-is). |
| 1 | `true` | partial specialization on `BinOp<T>`, written out by the `SPECIALIZE_ALL` macro for `bit_and`, `bit_or`, `bit_xor`, `plus`, `multiplies`, `minimum`, `maximum` | one `glsl::subgroup{Inclusive,Exclusive,}{And,Or,Xor,Add,Mul,Min,Max}` call |
| 1 | `false` | partial specialization on `<Params, BinOp, 1, false>`, for any binop | Emulation with `SizeLog2` shuffle steps. Inclusive scan: Hillis-Steele with `subgroupShuffleUp`. Exclusive scan: shift up by one lane, then inclusive. Reduction: `subgroupShuffleXor` butterfly. |

So native support only decides which of the two `ItemsPerInvocation = 1` versions does the cross-lane step, whether it's called directly or from inside the primary template.

A binop without a `SPECIALIZE_ALL` line has no `<..., 1, true>` specialization. With native on, even `ItemsPerInvocation = 1` falls back to the primary template, which then calls itself, and DXC rejects it with `recursive functions are not allowed`. Use custom binops with `shaderSubgroupArithmetic = false`.

## Seeing it in SPIR-V

The shader below compiles one inclusive `plus` scan over a subgroup of 32. Change the three macros to pick a different specialization.

```cpp
#include "nbl/builtin/hlsl/functional.hlsl"
#include "nbl/builtin/hlsl/workgroup/basic.hlsl"
#include "nbl/builtin/hlsl/subgroup2/arithmetic_params.hlsl"
#include "nbl/builtin/hlsl/subgroup2/arithmetic_portability.hlsl"

// change these to pick a different specialization
#ifndef SUBGROUP_SIZE_LOG2
#define SUBGROUP_SIZE_LOG2 5 // subgroup of 32
#endif
#ifndef ITEMS_PER_INVOCATION
#define ITEMS_PER_INVOCATION 1
#endif
#ifndef NATIVE
#define NATIVE 1
#endif

struct device_capabilities
{
    NBL_CONSTEXPR_STATIC_INLINE bool shaderSubgroupArithmetic = bool(NATIVE);
};

using namespace nbl::hlsl;
using config_t = subgroup2::Configuration<SUBGROUP_SIZE_LOG2>;
using params_t = subgroup2::ArithmeticParams<config_t, plus<uint32_t>, ITEMS_PER_INVOCATION, device_capabilities>;

[[vk::binding(0,0)]] RWStructuredBuffer<params_t::type_t> data;

[numthreads(64,1,1)]
void main()
{
    const uint32_t ix = workgroup::SubgroupContiguousIndex();
    subgroup2::inclusive_scan<params_t> scan;
    data[ix] = scan(data[ix]);
}
```

```
nsc -spirv -HV 202x -T cs_6_7 -E main -fspv-target-env=vulkan1.3 -enable-16bit-types -DNATIVE=1 -DITEMS_PER_INVOCATION=1 -Fc scan.spv scan.hlsl
```

On Compiler Explorer (the release `nsc`, not necessarily your branch):

| Variant | Link |
|---|---|
| native, 1 item | https://godbolt.devsh.eu/z/Waoao1 |
| emulated, 1 item | https://godbolt.devsh.eu/z/Yexhc4 |
| native, 4 items | https://godbolt.devsh.eu/z/z5rbno |
| emulated, 4 items | https://godbolt.devsh.eu/z/Y67zWK |
| all four side by side | https://godbolt.devsh.eu/z/abM995 |

What each variant compiles to, with SubgroupSize 32:

| Variant | Subgroup instructions | `OpIAdd` |
|---|---|---|
| native, 1 item | 1 `OpGroupNonUniformIAdd InclusiveScan` | 1: the `SubgroupContiguousIndex` |
| emulated, 1 item | 5 × (`OpControlBarrier`, `OpGroupNonUniformShuffleUp`, `OpSelect`), one per shuffle step | 6: 5 steps + index |
| native, 4 items | 1 `OpGroupNonUniformIAdd ExclusiveScan` (the primary template asks for the exclusive scan) | 8: 3 to scan the 4 items + 4 to add the exclusive result + index |
| emulated, 4 items | 6 × (`OpControlBarrier`, `OpGroupNonUniformShuffleUp`, `OpSelect`): 1 to shift up a lane + 5 inclusive steps | 13: 3 + 5 + 4 + index |
