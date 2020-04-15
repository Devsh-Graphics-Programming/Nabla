#include "irr/asset/IBuiltinIncludeLoader.h"

#include <string>
#include <sstream>
#include <cctype>

#include "irr/asset/ITexturePacker.h"

namespace irr {
namespace asset
{

static core::vector<std::string> parseArgumentsFromPath(const std::string& _path)
{
	core::vector<std::string> args;

	std::stringstream ss{_path};
	std::string arg;
	while (std::getline(ss, arg, '/'))
		args.push_back(std::move(arg));

	return args;
}

class CGLSLTexturePackerBuiltinIncludeLoader : public irr::asset::IBuiltinIncludeLoader
{
public:
    const char* getVirtualDirectoryName() const override { return "glsl/texture_packer/"; }

private:
	static std::string getDefinitions(const std::string& _path)
	{
		auto args = parseArgumentsFromPath(_path.substr(_path.find_first_of('/') + 1, _path.npos));
		if (args.size() < 4u)
			return {};

		constexpr uint32_t 
			ix_addr_x_bits	= 0u,
			ix_addr_y_bits	= 1u,
			ix_pg_sz_log2	= 2u,
			ix_tile_padding	= 3u;

		const uint32_t addr_x_bits = std::atoi(args[ix_addr_x_bits].c_str());
		const uint32_t addr_y_bits = std::atoi(args[ix_addr_y_bits].c_str());
		const uint32_t pg_sz_log2 = std::atoi(args[ix_pg_sz_log2].c_str());
		const uint32_t tile_padding = std::atoi(args[ix_tile_padding].c_str());

		ICPUTexturePacker::SMiptailPacker::rect tilePacking[ICPUTexturePacker::MAX_PHYSICAL_PAGE_SIZE_LOG2];
		//this could be cached..
		ICPUTexturePacker::SMiptailPacker::computeMiptailOffsets(tilePacking, pg_sz_log2, tile_padding);

		auto tilePackingOffsetsStr = [&] {
			std::string offsets;
			for (uint32_t i = 0u; i < pg_sz_log2; ++i)
				offsets += "uvec2(" + std::to_string(tilePacking[i].x) + "," + std::to_string(tilePacking[i].x) + ")" + (i==(pg_sz_log2-1u) ? "" : ",");
			return offsets;
		};

		const std::string inclguard = "_IRR_TEXTURE_PACKER_DEFINITIONS_" + args[ix_addr_x_bits] + "_" + args[ix_addr_y_bits] + "_" + args[ix_pg_sz_log2] + "_" + args[ix_tile_padding] + "_INCLUDED_";

		using namespace std::string_literals;
		return
			"\n#ifndef " + inclguard +
			"\n#define " + inclguard +
			"\n#define ADDR_Y_SHIFT "s + args[ix_addr_x_bits] + "u"
			"\n#define ADDR_LAYER_SHIFT " + std::to_string(addr_x_bits+addr_y_bits) + "u" +
			R"(
#define ADDR_X_MASK uint((0x1u<<ADDR_Y_SHIFT)-1u)
#define ADDR_Y_MASK uint((0x1u<<(ADDR_LAYER_SHIFT-ADDR_Y_SHIFT))-1u))" +

			"\n\n#define PAGE_SZ " + std::to_string(1u<<pg_sz_log2) + "u" +
			"\n#define PAGE_SZ_LOG2 " + args[ix_pg_sz_log2] + "u" +
			"\n#define TILE_PADDING " + args[ix_tile_padding] + "u" +
			"\n#define PADDED_TILE_SIZE uint(PAGE_SZ+2*TILE_PADDING)" +
			"\n\nconst uvec2 packingOffsets[] = uvec2[PAGE_SZ_LOG2+1]( uvec2(0u,0u)," + tilePackingOffsetsStr() + ");" + 
			"\n#endif\n";
	}

protected:
	core::vector<std::pair<std::regex, HandleFunc_t>> getBuiltinNamesToFunctionMapping() const override
	{
		return {
			//definitions.glsl/addr_x_bits/addr_y_bits/pg_sz_log2/tile_padding
			{ std::regex{"definitions\\.glsl/[0-9]+/[0-9]+/[0-9]+/[0-9]+"}, &getDefinitions },
		};
	}
};

}}