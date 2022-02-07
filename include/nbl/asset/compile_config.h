// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_COMPILE_CONFIG_H_INCLUDED__
#define __NBL_ASSET_COMPILE_CONFIG_H_INCLUDED__

// TODO: @Anastazluk review the `_NBL_COMPILE_WITH` which are present here

//! Define _NBL_COMPILE_WITH_OPENSSL_ to enable compiling the engine using libssl.
/** This enables the engine to read and write encrypted BAW format files.
If you comment this out, the engine will no longer read or write encrypted .baw files! */
#define _NBL_COMPILE_WITH_OPENSSL_

//! Define _NBL_COMPILE_WITH_JPEGLIB_ to enable compiling the engine using libjpeg.
/** This enables the engine to read jpeg images. If you comment this out,
the engine will no longer read .jpeg images. */
#define _NBL_COMPILE_WITH_LIBJPEG_

//! Define _NBL_COMPILE_WITH_LIBPNG_ to enable compiling the engine using libpng.
/** This enables the engine to read png images. If you comment this out,
the engine will no longer read .png images. */
#define _NBL_COMPILE_WITH_LIBPNG_

//! Uncomment the following line if you want to ignore the deprecated warnings
//#define IGNORE_DEPRECATED_WARNING

//! Define __NBL_COMPILE_WITH_ZIP_ARCHIVE_LOADER_ if you want to open ZIP and GZIP archives
/** ZIP reading has several more options below to configure. */
#define __NBL_COMPILE_WITH_ZIP_ARCHIVE_LOADER_

//#ifdef __NBL_COMPILE_WITH_ZIP_ARCHIVE_LOADER_

//! Define _NBL_COMPILE_WITH_ZLIB_ to enable compiling the engine using zlib.
/** This enables the engine to read from compressed .zip archives. If you
disable this feature, the engine can still read archives, but only uncompressed
ones. */
#define _NBL_COMPILE_WITH_ZLIB_

//! Define _NBL_COMPILE_WITH_ZIP_ENCRYPTION_ if you want to read AES-encrypted ZIP archives
#define _NBL_COMPILE_WITH_ZIP_ENCRYPTION_

//! Define _NBL_COMPILE_WITH_BZIP2_ if you want to support bzip2 compressed zip archives
/** bzip2 is superior to the original zip file compression modes, but requires
a certain amount of memory for decompression and adds several files to the
library. */
#define _NBL_COMPILE_WITH_BZIP2_

//! Define _NBL_COMPILE_WITH_LZMA_ if you want to use LZMA compressed zip files.
/** LZMA is a very efficient compression code, known from 7zip. Irrlicht
currently only supports zip archives, though. */
//#define _NBL_COMPILE_WITH_LZMA_

//#endif

//! Define __NBL_COMPILE_WITH_MOUNT_ARCHIVE_LOADER_ if you want to mount folders as archives
#define __NBL_COMPILE_WITH_MOUNT_ARCHIVE_LOADER_

//! Define __NBL_COMPILE_WITH_PAK_ARCHIVE_LOADER_ if you want to open ID software PAK archives
#define __NBL_COMPILE_WITH_PAK_ARCHIVE_LOADER_

//! Define __NBL_COMPILE_WITH_TAR_ARCHIVE_LOADER_ if you want to open TAR archives
#define __NBL_COMPILE_WITH_TAR_ARCHIVE_LOADER_

#define _NBL_FORMAT_VERSION 3

//! @see @ref CBlobsLoadingManager
#define _NBL_ADD_BLOB_SUPPORT(BlobClassName, EnumValue, Function, ...) \
    case asset::Blob::EnumValue:                                       \
        return asset::BlobClassName::Function(__VA_ARGS__);

//! Used inside CBlobsLoadingManager. Adds support of given blob types.
#ifdef OLD_SHADERS
// @crisspl / @Anastazluk fix this shit for new pipeline !!!
#define _NBL_SUPPORTED_BLOBS(Function, ...)                                                        \
    _NBL_ADD_BLOB_SUPPORT(RawBufferBlobV3, EBT_RAW_DATA_BUFFER, Function, __VA_ARGS__)             \
    _NBL_ADD_BLOB_SUPPORT(TexturePathBlobV3, EBT_TEXTURE_PATH, Function, __VA_ARGS__)              \
    _NBL_ADD_BLOB_SUPPORT(MeshBlobV3, EBT_MESH, Function, __VA_ARGS__)                             \
    _NBL_ADD_BLOB_SUPPORT(SkinnedMeshBlobV3, EBT_SKINNED_MESH, Function, __VA_ARGS__)              \
    _NBL_ADD_BLOB_SUPPORT(MeshBufferBlobV3, EBT_MESH_BUFFER, Function, __VA_ARGS__)                \
    _NBL_ADD_BLOB_SUPPORT(SkinnedMeshBufferBlobV3, EBT_SKINNED_MESH_BUFFER, Function, __VA_ARGS__) \
    _NBL_ADD_BLOB_SUPPORT(MeshDataFormatDescBlobV3, EBT_DATA_FORMAT_DESC, Function, __VA_ARGS__)   \
    _NBL_ADD_BLOB_SUPPORT(FinalBoneHierarchyBlobV3, EBT_FINAL_BONE_HIERARCHY, Function, __VA_ARGS__)
#else
#define _NBL_SUPPORTED_BLOBS(Function, ...)
#endif

#endif
