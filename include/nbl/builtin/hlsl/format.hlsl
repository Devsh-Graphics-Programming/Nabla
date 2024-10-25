#ifndef _NBL_HLSL_FORMAT_HLSL_
#define _NBL_HLSL_FORMAT_HLSL_

#include "nbl/builtin/hlsl/format/shared_exp.hlsl"
#include "nbl/builtin/hlsl/format/octahedral.hlsl"

namespace nbl
{
namespace hlsl
{
namespace format
{

enum TexelBlockFormat : uint32_t
{
    // depth
    D16_UNORM,
    X8_D24_UNORM_PACK32,
    D32_SFLOAT,
    S8_UINT,
    D16_UNORM_S8_UINT,
    D24_UNORM_S8_UINT,
    D32_SFLOAT_S8_UINT,

    // color
    R4G4_UNORM_PACK8,
    R4G4B4A4_UNORM_PACK16,
    B4G4R4A4_UNORM_PACK16,
    R5G6B5_UNORM_PACK16,
    B5G6R5_UNORM_PACK16,
    R5G5B5A1_UNORM_PACK16,
    B5G5R5A1_UNORM_PACK16,
    A1R5G5B5_UNORM_PACK16,
    R8_UNORM,
    R8_SNORM,
    R8_USCALED,
    R8_SSCALED,
    R8_UINT,
    R8_SINT,
    R8_SRGB,
    R8G8_UNORM,
    R8G8_SNORM,
    R8G8_USCALED,
    R8G8_SSCALED,
    R8G8_UINT,
    R8G8_SINT,
    R8G8_SRGB,
    R8G8B8_UNORM,
    R8G8B8_SNORM,
    R8G8B8_USCALED,
    R8G8B8_SSCALED,
    R8G8B8_UINT,
    R8G8B8_SINT,
    R8G8B8_SRGB,
    B8G8R8_UNORM,
    B8G8R8_SNORM,
    B8G8R8_USCALED,
    B8G8R8_SSCALED,
    B8G8R8_UINT,
    B8G8R8_SINT,
    B8G8R8_SRGB,
    R8G8B8A8_UNORM,
    R8G8B8A8_SNORM,
    R8G8B8A8_USCALED,
    R8G8B8A8_SSCALED,
    R8G8B8A8_UINT,
    R8G8B8A8_SINT,
    R8G8B8A8_SRGB,
    B8G8R8A8_UNORM,
    B8G8R8A8_SNORM,
    B8G8R8A8_USCALED,
    B8G8R8A8_SSCALED,
    B8G8R8A8_UINT,
    B8G8R8A8_SINT,
    B8G8R8A8_SRGB,
    A8B8G8R8_UNORM_PACK32,
    A8B8G8R8_SNORM_PACK32,
    A8B8G8R8_USCALED_PACK32,
    A8B8G8R8_SSCALED_PACK32,
    A8B8G8R8_UINT_PACK32,
    A8B8G8R8_SINT_PACK32,
    A8B8G8R8_SRGB_PACK32,
    A2R10G10B10_UNORM_PACK32,
    A2R10G10B10_SNORM_PACK32,
    A2R10G10B10_USCALED_PACK32,
    A2R10G10B10_SSCALED_PACK32,
    A2R10G10B10_UINT_PACK32,
    A2R10G10B10_SINT_PACK32,
    A2B10G10R10_UNORM_PACK32,
    A2B10G10R10_SNORM_PACK32,
    A2B10G10R10_USCALED_PACK32,
    A2B10G10R10_SSCALED_PACK32,
    A2B10G10R10_UINT_PACK32,
    A2B10G10R10_SINT_PACK32,
    R16_UNORM,
    R16_SNORM,
    R16_USCALED,
    R16_SSCALED,
    R16_UINT,
    R16_SINT,
    R16_SFLOAT,
    R16G16_UNORM,
    R16G16_SNORM,
    R16G16_USCALED,
    R16G16_SSCALED,
    R16G16_UINT,
    R16G16_SINT,
    R16G16_SFLOAT,
    R16G16B16_UNORM,
    R16G16B16_SNORM,
    R16G16B16_USCALED,
    R16G16B16_SSCALED,
    R16G16B16_UINT,
    R16G16B16_SINT,
    R16G16B16_SFLOAT,
    R16G16B16A16_UNORM,
    R16G16B16A16_SNORM,
    R16G16B16A16_USCALED,
    R16G16B16A16_SSCALED,
    R16G16B16A16_UINT,
    R16G16B16A16_SINT,
    R16G16B16A16_SFLOAT,
    R32_UINT,
    R32_SINT,
    R32_SFLOAT,
    R32G32_UINT,
    R32G32_SINT,
    R32G32_SFLOAT,
    R32G32B32_UINT,
    R32G32B32_SINT,
    R32G32B32_SFLOAT,
    R32G32B32A32_UINT,
    R32G32B32A32_SINT,
    R32G32B32A32_SFLOAT,
    R64_UINT,
    R64_SINT,
    R64_SFLOAT,
    R64G64_UINT,
    R64G64_SINT,
    R64G64_SFLOAT,
    R64G64B64_UINT,
    R64G64B64_SINT,
    R64G64B64_SFLOAT,
    R64G64B64A64_UINT,
    R64G64B64A64_SINT,
    R64G64B64A64_SFLOAT,
    B10G11R11_UFLOAT_PACK32,
    E5B9G9R9_UFLOAT_PACK32,

    //! Block Compression Formats!
    BC1_RGB_UNORM_BLOCK,
    BC1_RGB_SRGB_BLOCK,
    BC1_RGBA_UNORM_BLOCK,
    BC1_RGBA_SRGB_BLOCK,
    BC2_UNORM_BLOCK,
    BC2_SRGB_BLOCK,
    BC3_UNORM_BLOCK,
    BC3_SRGB_BLOCK,
    BC4_UNORM_BLOCK,
    BC4_SNORM_BLOCK,
    BC5_UNORM_BLOCK,
    BC5_SNORM_BLOCK,
    BC6H_UFLOAT_BLOCK,
    BC6H_SFLOAT_BLOCK,
    BC7_UNORM_BLOCK,
    BC7_SRGB_BLOCK,
    ASTC_4x4_UNORM_BLOCK,
    ASTC_4x4_SRGB_BLOCK,
    ASTC_5x4_UNORM_BLOCK,
    ASTC_5x4_SRGB_BLOCK,
    ASTC_5x5_UNORM_BLOCK,
    ASTC_5x5_SRGB_BLOCK,
    ASTC_6x5_UNORM_BLOCK,
    ASTC_6x5_SRGB_BLOCK,
    ASTC_6x6_UNORM_BLOCK,
    ASTC_6x6_SRGB_BLOCK,
    ASTC_8x5_UNORM_BLOCK,
    ASTC_8x5_SRGB_BLOCK,
    ASTC_8x6_UNORM_BLOCK,
    ASTC_8x6_SRGB_BLOCK,
    ASTC_8x8_UNORM_BLOCK,
    ASTC_8x8_SRGB_BLOCK,
    ASTC_10x5_UNORM_BLOCK,
    ASTC_10x5_SRGB_BLOCK,
    ASTC_10x6_UNORM_BLOCK,
    ASTC_10x6_SRGB_BLOCK,
    ASTC_10x8_UNORM_BLOCK,
    ASTC_10x8_SRGB_BLOCK,
    ASTC_10x10_UNORM_BLOCK,
    ASTC_10x10_SRGB_BLOCK,
    ASTC_12x10_UNORM_BLOCK,
    ASTC_12x10_SRGB_BLOCK,
    ASTC_12x12_UNORM_BLOCK,
    ASTC_12x12_SRGB_BLOCK,
    ETC2_R8G8B8_UNORM_BLOCK,
    ETC2_R8G8B8_SRGB_BLOCK,
    ETC2_R8G8B8A1_UNORM_BLOCK,
    ETC2_R8G8B8A1_SRGB_BLOCK,
    ETC2_R8G8B8A8_UNORM_BLOCK,
    ETC2_R8G8B8A8_SRGB_BLOCK,
    EAC_R11_UNORM_BLOCK,
    EAC_R11_SNORM_BLOCK,
    EAC_R11G11_UNORM_BLOCK,
    EAC_R11G11_SNORM_BLOCK,
    PVRTC1_2BPP_UNORM_BLOCK_IMG,
    PVRTC1_4BPP_UNORM_BLOCK_IMG,
    PVRTC2_2BPP_UNORM_BLOCK_IMG,
    PVRTC2_4BPP_UNORM_BLOCK_IMG,
    PVRTC1_2BPP_SRGB_BLOCK_IMG,
    PVRTC1_4BPP_SRGB_BLOCK_IMG,
    PVRTC2_2BPP_SRGB_BLOCK_IMG,
    PVRTC2_4BPP_SRGB_BLOCK_IMG,
#if 0 // later
    //! Planar formats
    G8_B8_R8_3PLANE_420_UNORM,
    G8_B8R8_2PLANE_420_UNORM,
    G8_B8_R8_3PLANE_422_UNORM,
    G8_B8R8_2PLANE_422_UNORM,
    G8_B8_R8_3PLANE_444_UNORM,
#endif
    //! Unknown color format:
    TBF_UNKNOWN,
    TBF_COUNT = TBF_UNKNOWN
};

enum BlockViewClass : uint32_t
{
    SIZE_8_BIT,
    SIZE_16_BIT,
    SIZE_24_BIT,
    SIZE_32_BIT,
    SIZE_48_BIT,
    SIZE_64_BIT,
    SIZE_96_BIT,
    SIZE_128_BIT,
    SIZE_192_BIT,
    SIZE_256_BIT,

    BC1_RGB,
    BC1_RGBA,
    BC2,
    BC3,
    BC4,
    BC5,
    BC6,
    BC7,

    ETC2_RGB,
    ETC2_RGBA,
    ETC2_EAC_RGBA,
    ETC2_EAC_R,
    ETC2_EAC_RG,

    ASTC_4X4,
    ASTC_5X4,
    ASTC_5X5,
    ASTC_6X5,
    ASTC_6X6,
    ASTC_8X5,
    ASTC_8X6,
    ASTC_8X8,
    ASTC_10X5,
    ASTC_10X6,
    ASTC_10X8,
    ASTC_10X10,
    ASTC_12X10,
    ASTC_12X12,

    // [TODO] there are still more format classes; https://registry.khronos.org/vulkan/specs/1.2-extensions/html/chap43.html#formats-compatibility-classes
    BVC_UNKNOWN,
    BVC_COUNT = BVC_UNKNOWN
};

// default is invalid/runtime dynamic
template<BlockViewClass _Class=BlockViewClass::BVC_UNKNOWN>
struct view_class_traits
{
    BlockViewClass Class;
    TexelBlockFormat RawAccessViewFormat;
    uint32_t ByteSize;
    uint32_t Alignment;
};
#define SPECIALIZE_CLASS(CLASS,RAWFMT,SIZE,ALIGNMENT) template<> \
struct view_class_traits<CLASS> \
{ \
    NBL_CONSTEXPR_STATIC_INLINE BlockViewClass Class = CLASS; \
    NBL_CONSTEXPR_STATIC_INLINE TexelBlockFormat RawAccessViewFormat = RAWFMT; \
    NBL_CONSTEXPR_STATIC_INLINE uint32_t ByteSize = SIZE; \
    NBL_CONSTEXPR_STATIC_INLINE uint32_t Alignment = ALIGNMENT; \
}
SPECIALIZE_CLASS(BlockViewClass::SIZE_8_BIT,TexelBlockFormat::R8_UINT,1,1);
SPECIALIZE_CLASS(BlockViewClass::SIZE_16_BIT,TexelBlockFormat::R16_UINT,2,2);
SPECIALIZE_CLASS(BlockViewClass::SIZE_32_BIT,TexelBlockFormat::R32_UINT,4,4);
SPECIALIZE_CLASS(BlockViewClass::SIZE_64_BIT,TexelBlockFormat::R32G32_UINT,8,4);
SPECIALIZE_CLASS(BlockViewClass::SIZE_96_BIT,TexelBlockFormat::R32G32B32_UINT,12,4);
SPECIALIZE_CLASS(BlockViewClass::SIZE_128_BIT,TexelBlockFormat::R32G32B32A32_UINT,16,4);
SPECIALIZE_CLASS(BlockViewClass::SIZE_192_BIT,TexelBlockFormat::R64G64B64_UINT,24,8);
SPECIALIZE_CLASS(BlockViewClass::SIZE_256_BIT,TexelBlockFormat::R64G64B64A32_UINT,32,8);
// TODO: block compressed
#undef SPECIALIZE_CLASS

enum TexelKind
{
    Float = 0,
    Integer,
    SRGB,
    Normalized,
    Scaled
};

enum TexelAttributes : uint32_t
{
    HasDepthBit,
    HasStencilBit,
    BGRABit,
    SignedBit,
    CompressedBit
};

// default is invalid/runtime dynamic
template<TexelBlockFormat Format=TexelBlockFormat::TBF_UNKNOWN>
struct block_traits
{
    view_class_traits<> ClassTraits;
    // float int uint
#ifndef __HLSL_VERSION
    std::type_info* DecodeTypeID;
#else
    uint32_t DecodeTypeID;
#endif
    TexelKind Kind;
    TexelAttributes Attributes;
    // size stuff
    uint32_t BlockSizeX;
    uint32_t BlockSizeY;
    uint32_t BlockSizeZ;
    uint32_t Channels;
    // bits per pixel
    // TODO: rational in HLSL
    uint32_t BPPNum;
    uint32_t BPPDen;
};

// TODO: turn into a macro so we can fast-define this
template<>
struct block_traits<TexelBlockFormat::B8G8R8A8_SRGB>
{
    NBL_CONSTEXPR_STATIC_INLINE BlockViewClass Class = BlockViewClass::SIZE_32_BIT;
    using class_traits_t = view_class_traits<Class>;
#ifndef __HLSL_VERSION
    NBL_CONSTEXPR_STATIC_INLINE std::type_info* DecodeTypeID = &typeid(float);
#else
    NBL_CONSTEXPR_STATIC_INLINE uint32_t DecodeTypeID = impl::typeid_t<float>::value;
#endif
    NBL_CONSTEXPR_STATIC_INLINE uint32_t BlockSizeX = 1;
    NBL_CONSTEXPR_STATIC_INLINE uint32_t BlockSizeY = 1;
    NBL_CONSTEXPR_STATIC_INLINE uint32_t BlockSizeZ = 1;
    NBL_CONSTEXPR_STATIC_INLINE uint32_t Channels = 4;

    NBL_CONSTEXPR_STATIC_INLINE uint32_t BPPNum = 32;
    NBL_CONSTEXPR_STATIC_INLINE uint32_t BPPDen = 1;
};

/*
template<TexelBlockFormat Format>
struct texel_block
{
    unsigned_of_size<traits<Format>::alignment> storage[traits<Format>::ByteSize/traits<Format>::Alignment];
};
*/
}

// conversion ops for constant to runtime traits
namespace impl
{


template<format::BlockViewClass ConstexprT>
struct _static_cast_helper<format::view_class_traits<format::BlockViewClass::BVC_UNKNOWN>,format::view_class_traits<ConstexprT> >
{
    using T = format::view_class_traits<format::BlockViewClass::BVC_UNKNOWN>;
    using U = format::view_class_traits<ConstexprT>;

    T operator()(U val)
    {
        T retval;
        retval.Class = U::Class;
        retval.RawAccessViewFormat = U::RawAccessViewFormat;
        retval.ByteSize = U::ByteSize;
        retval.Alignment = U::Alignment;
        return retval;
    }
};

// TODO: specialization for the BlockTraits

}
}
}
#endif