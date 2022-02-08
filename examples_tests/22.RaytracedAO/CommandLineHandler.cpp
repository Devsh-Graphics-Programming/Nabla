// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "CommandLineHandler.hpp"

#include <algorithm>
#include <filesystem>
#include <iostream>

using namespace nbl;
using namespace asset;
using namespace core;

CommandLineHandler::CommandLineHandler(const std::vector<std::string>& argv)
{
	if(argv.size() > MaxRayTracerCommandLineArgs)
	{
		std::cout << helpMessage.data() << std::endl;
		return;
	}
	
	auto logError = [&](const std::string message)
	{
		std::cout << "ERROR (" + std::to_string(__LINE__) + " line): " + message << std::endl;
	};

	auto arguments = argv;
	
	auto ltrim = [](const std::string &s)
	{
		const std::string WHITESPACE = " \n\r\t\f\v";
		size_t start = s.find_first_not_of(WHITESPACE);
		return (start == std::string::npos) ? "" : s.substr(start);
	};
 
	auto rtrim = [](const std::string &s)
	{
		const std::string WHITESPACE = " \n\r\t\f\v";
		size_t end = s.find_last_not_of(WHITESPACE);
		return (end == std::string::npos) ? "" : s.substr(0, end + 1);
	};

	auto getSerializedValues = [&](const auto& variablesStream, const std::regex& separator=std::regex{"[[:s:]]"})
	{
		std::sregex_token_iterator it{ variablesStream.begin(), variablesStream.end(), separator, -1 };
		std::vector<std::string> variablesHandle = { it,{} };

		// remove any accidental whitespace only vars
		variablesHandle.erase(
			std::remove_if(
				variablesHandle.begin(),variablesHandle.end(),
				[](const std::string& x) {return !std::regex_search(x,std::regex{"[^[:s:]]"}); }
			),
			variablesHandle.end()
		);

		return variablesHandle;
	};

	initializeMatchingMap();

	RaytracerExampleArguments previousArg = REA_COUNT;

	bool success = true;

	for (auto i = 0; i < arguments.size(); ++i)
	{
		std::string rawFetchedCmdArgument = arguments[i];
		
		bool addToPreviousOption = false;

		const auto firstHyphen = rawFetchedCmdArgument.find_first_of("-");
		if(firstHyphen != 0)
			addToPreviousOption = true;

		if(addToPreviousOption)
		{
			if(REA_COUNT != previousArg)
			{
				if(!rawVariables[previousArg].has_value())
					rawVariables[previousArg].emplace(std::vector<std::string>());
				
				auto & outVector = rawVariables[previousArg].value();
				std::vector<std::string> toAdd = getSerializedValues(rawFetchedCmdArgument);
				outVector.insert(outVector.end(), toAdd.begin(), toAdd.end());
			}
			else
			{
				logError("Unexcepted argument!, command options should start with '-' character");
				success = false;
				break;
			}
		}
		else
		{
			const auto offset = firstHyphen + 1;
			const auto endOfFetchedVariableName = rawFetchedCmdArgument.find_first_of("=");
			const auto count = endOfFetchedVariableName - offset;
			const auto cmdFetchedVariable = rawFetchedCmdArgument.substr(offset, count);
			std::string variable = cmdFetchedVariable;
			auto arg = getMatchedVariableMapID(variable);
			
			if(arg == REA_COUNT)
			{
				logError("Unexcepted argument!!");
				success = false;
				break;
			}

			if(rawVariables[arg].has_value())
			{
				logError("Variable used previously!");
				success = false;
				break;
			}
			
			if(endOfFetchedVariableName != std::string::npos)
			{
				auto value = rawFetchedCmdArgument.substr(endOfFetchedVariableName + 1);
				auto zipExtensionPos = value.find(".zip");
				if(zipExtensionPos == std::string::npos)
					zipExtensionPos = value.find(".ZIP");

				if(zipExtensionPos != std::string::npos)
				{
					auto endOfZip = zipExtensionPos + 4;
					std::vector<std::string> toAdd;
					// found .zip, add .zip path + the rest
					auto zip = value.substr(0, endOfZip);
					auto remaining = value.substr(endOfZip);
					toAdd.push_back(zip);
					if(!remaining.empty()) {
						// remove starting spaces
						remaining = ltrim(remaining);
						toAdd.push_back(remaining);
					}
					rawVariables[arg].emplace(toAdd);
				}
				else
				{
					std::vector<std::string> toAdd;
					// no .zip, push back all the variable to SCENE
					toAdd.push_back(value);
					rawVariables[arg].emplace(toAdd);
				}

			}
			else
			{
				std::vector<std::string> emptyVec;
				rawVariables[arg].emplace(emptyVec);
			}

			previousArg = arg;
		}

	}

	if (!validateParameters() || !success)
		return;

	performFinalAssignmentStepForUsefulVariables();
}

bool CommandLineHandler::validateParameters()
{
	auto logError = [&](const std::string message)
	{
		std::cout << "ERROR (" + std::to_string(__LINE__) + " line): " + message << std::endl;
	};

	if(rawVariables[REA_SCENE].has_value())
	{
		auto sceneDirectory = rawVariables[REA_SCENE].value();
		if(sceneDirectory.empty())
		{
			logError("Expected at least one value for SCENE");
			return false;
		}
	}

	return true;
}