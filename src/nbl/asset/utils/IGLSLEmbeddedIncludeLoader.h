// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_ASSET_I_GLSL_EMBEDDED_INCLUDE_LOADER_H_INCLUDED__
#define __NBL_ASSET_ASSET_I_GLSL_EMBEDDED_INCLUDE_LOADER_H_INCLUDED__

#include "nbl/system/declarations.h"

#include "nbl/asset/utils/IBuiltinIncludeLoader.h"

namespace nbl::asset
{

class IGLSLEmbeddedIncludeLoader : public IBuiltinIncludeLoader
{
	protected:
		virtual ~IGLSLEmbeddedIncludeLoader() = default;

		inline core::vector<std::pair<std::regex,HandleFunc_t>> getBuiltinNamesToFunctionMapping() const override
		{
			std::string pattern(getVirtualDirectoryName());
			pattern += ".*";
			HandleFunc_t tmp = [this](const std::string& _name) -> std::string {return getFromDiskOrEmbedding(_name);};
			return {{std::regex{pattern},std::move(tmp)}};
		}
		
		static core::vector<std::string> parseArgumentsFromPath(const std::string& _path)
		{
			core::vector<std::string> args;

			std::stringstream ss{ _path };
			std::string arg;
			while (std::getline(ss, arg, '/'))
				args.push_back(std::move(arg));

			return args;
		}
		
		system::ISystem* s;

	public:
		IGLSLEmbeddedIncludeLoader(system::ISystem* system) : s(system) {}

		//
		const char* getVirtualDirectoryName() const override { return ""; }

		//
		inline std::string getFromDiskOrEmbedding(const std::string& _name) const
		{
			auto path = "nbl/builtin/" + _name;
			system::ISystem::future_t<core::smart_refctd_ptr<system::IFile>> future;
			bool validInput = s->createFile(future, path, system::IFile::ECF_READ);
			assert(validInput);
			core::smart_refctd_ptr<system::IFile> file = future.get();

			size_t fileSize = file->getSize();
			std::string content(fileSize, '/0');
			system::future<size_t> read_future;
			file->read(read_future, content.data(), 0, fileSize);
			assert(validInput);
			read_future.get();

			return content;
		}
};

}

#endif
