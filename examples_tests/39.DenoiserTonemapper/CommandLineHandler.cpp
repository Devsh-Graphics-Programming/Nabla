#include "CommandLineHandler.hpp"

CommandLineHandler::CommandLineHandler(const int argc, core::vector<std::string> argv, IAssetManager* am)
{
	rawVariables[DTEA_OPENEXR_FILE].first = OPENEXR_FILE;
	rawVariables[DTEA_CHANNEL_NAMES].first = CHANNEL_NAMES;
	rawVariables[DTEA_EXPOSURE_BIAS].first = EXPOSURE_BIAS;
	rawVariables[DTEA_DENOISER_BLEND_FACTOR].first = DENOISER_BLEND_FACTOR;
	rawVariables[DTEA_BLOOM_SIZE].first = BLOOM_SIZE;
	rawVariables[DTEA_REINHARD].first = REINHARD;
	rawVariables[DTEA_ACES].first = ACES;
	rawVariables[DTEA_OUTPUT].first = OUTPUT;

	std::array<std::string, PROPER_CMD_ARGUMENTS_AMOUNT> argvMappedList;

	auto fillArgvList = [&](auto argvStream, auto variableCount)
	{
		for (auto i = 0; i < variableCount; ++i)
			argvMappedList[i] = argvStream[i];
	};

	auto getSerializedValues = [&](auto variablesStream, auto supposedArgumentsAmout, bool onlyEntireArgvArgument = false)
	{
		core::vector<std::string> variablesHandle;
		variablesHandle.reserve(supposedArgumentsAmout);

		std::string tmpStream;
		for (auto x = 0ul; x < variablesStream.size(); ++x)
		{
			const auto character = variablesStream.at(x);
			if (character == '\n')
				continue;

			if (onlyEntireArgvArgument ? (character == '\r') : (character == ','))
			{
				variablesHandle.push_back(tmpStream);
				tmpStream.clear();
			}
			else if (x == variablesStream.size() - 1)
			{
				tmpStream.push_back(character);
				variablesHandle.push_back(tmpStream);
				tmpStream.clear();
			}
			else
				tmpStream.push_back(character);
		}

		return variablesHandle;
	};

	if (argc == PROPER_CMD_ARGUMENTS_AMOUNT)
		mode = CLM_CMD_LIST;
	else if (argc == PROPER_BATCH_FILE_ARGUMENTS_AMOUNT)
		mode = CLM_FILE;
	else
		mode = CLM_UNKNOWN;

	if (mode == CLM_UNKNOWN)
		os::Printer::log(requiredArgumentsMessage.data(), ELL_ERROR);
	else if (mode == CLM_CMD_LIST || mode == CLM_FILE)
	{
		if (std::string(argv[1]) == "-batch")
		{
			auto file = am->getFileSystem()->createAndOpenFile(argv[2].c_str());
			std::string argvStream;
			argvStream.resize(file->getSize(), ' ');
			file->read(argvStream.data(), file->getSize());

			auto arguments = getSerializedValues(argvStream, PROPER_CMD_ARGUMENTS_AMOUNT, true);

			if (arguments.size() != PROPER_CMD_ARGUMENTS_AMOUNT)
			{
				os::Printer::log("The file is incorrect!", ELL_ERROR);
				os::Printer::log(requiredArgumentsMessage.data(), ELL_INFORMATION);
				status = false;
			}

			fillArgvList(arguments, PROPER_CMD_ARGUMENTS_AMOUNT);
		}
		else if (argc == PROPER_CMD_ARGUMENTS_AMOUNT)
			fillArgvList(argv, argc);
		else
		{
			os::Printer::log("Invalid syntax!", ELL_ERROR);
			os::Printer::log(requiredArgumentsMessage.data(), ELL_INFORMATION);
			status = false;
		}

		// read from argv list to map and put variables to appropiate places in a cache
		for (auto i = 0; i < DTEA_COUNT; ++i)
		{
			auto& referenceVariableMap = rawVariables[i];

			for (auto z = 0; z < PROPER_CMD_ARGUMENTS_AMOUNT; ++z)
			{
				std::string rawFetchedCmdArgument = argvMappedList[z];
				const auto offset = rawFetchedCmdArgument.find_last_of("-") + 1;
				const auto endOfFetchedVariableName = rawFetchedCmdArgument.find_first_of("=");
				const auto count = endOfFetchedVariableName - offset;
				const auto cmdFetchedVariable = rawFetchedCmdArgument.substr(offset, count);

				auto isTonemapperDetected = [&]()
				{
					if (referenceVariableMap.first == ACES || referenceVariableMap.first == REINHARD)
						if (cmdFetchedVariable == TONEMAPPER)
							return true;
					
					return false;
				};

				const auto tonemapperDetected = isTonemapperDetected();
				const auto matchedVariables = ((referenceVariableMap.first == cmdFetchedVariable) || tonemapperDetected);

				if (matchedVariables)
				{
					std::string variable = cmdFetchedVariable;
					const auto beginningOfVariables = rawFetchedCmdArgument.find_last_of("=") + 1;
					auto variablesStream = rawFetchedCmdArgument.substr(beginningOfVariables);

					if (tonemapperDetected)
					{
						auto foundAces = rawFetchedCmdArgument.find(ACES) != std::string::npos;
						auto foundReinhard = rawFetchedCmdArgument.find(REINHARD) != std::string::npos;

						if (foundAces)
							variable = ACES;
						else if (foundReinhard)
							variable = REINHARD;
						else 
							variable = REINHARD;
					}

					if (variable == ACES)
					{
						// 5 values according with the syntax
						auto variablesHandle = getSerializedValues(variablesStream, AA_COUNT);
						auto& reference = rawVariables[DTEA_ACES];
						reference.second = variablesHandle;

						if (variablesHandle.size() != AA_COUNT)
							variablesHandle.resize(AA_COUNT);

						reference.second[AA_ARG1] = variablesHandle[AA_ARG1].empty() ? std::string("2.51") : variablesHandle[AA_ARG1];
						reference.second[AA_ARG2] = variablesHandle[AA_ARG2].empty() ? std::string("0.03") : variablesHandle[AA_ARG2];
						reference.second[AA_ARG3] = variablesHandle[AA_ARG3].empty() ? std::string("2.43") : variablesHandle[AA_ARG3];
						reference.second[AA_ARG4] = variablesHandle[AA_ARG4].empty() ? std::string("0.59") : variablesHandle[AA_ARG4];
						reference.second[AA_ARG5] = variablesHandle[AA_ARG5].empty() ? std::string("0.14") : variablesHandle[AA_ARG5];
					}
					else if (variable == REINHARD)
					{
						// at the moment there is no variables for REINHARD
						auto variablesHandle = getSerializedValues(variablesStream, 0);
						auto& reference = rawVariables[DTEA_REINHARD];
						reference.second = variablesHandle;
					}
					else if (variable == CHANNEL_NAMES)
					{
						// various amount of values allowed
						auto variablesHandle = getSerializedValues(variablesStream, 3);
							referenceVariableMap.second = variablesHandle;
					}
					else  
					{
						// always one value
						auto variablesHandle = getSerializedValues(variablesStream, 1);
						referenceVariableMap.second = variablesHandle;
					}
				}
				else
					continue;
			}
		}
	}
	else if (argc > 1 && argc < 7)
	{
		os::Printer::log("Single argument assumptions aren't allowed - too less arguments!", ELL_ERROR);
		os::Printer::log(requiredArgumentsMessage.data(), ELL_INFORMATION);
		status = false;
	}
	else if (argc > 7)
	{
		os::Printer::log("Too many arguments!", ELL_ERROR);
		os::Printer::log(requiredArgumentsMessage.data(), ELL_INFORMATION);
		status = false;
	}
}