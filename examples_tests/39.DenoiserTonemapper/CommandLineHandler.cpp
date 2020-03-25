#include "CommandLineHandler.hpp"

using namespace irr;
using namespace asset;
using namespace core;

CommandLineHandler::CommandLineHandler(core::vector<std::string> argv, IAssetManager* am) : status(false)
{
	core::vector<std::array<std::string, PROPER_CMD_ARGUMENTS_AMOUNT>> argvMappedList(1);

	auto fillArgvList = [&](auto argvStream, auto variableCount, uint64_t batchFileID)
	{
		for (auto i = 0; i < variableCount; ++i)
			argvMappedList[batchFileID][i] = argvStream[i];
	};

	auto getBatchFilesArgvStream = [&](std::string& fileStream)
	{
		core::vector<std::string> argvsStream;
		size_t offset = {};
		while (true)
		{
			const auto previousOffset = offset;
			offset = fileStream.find("\r\n", previousOffset);
			if (offset == std::string::npos)
				break;
			else
				offset += 2;

			argvsStream.push_back(fileStream.substr(previousOffset, offset));
		}

		return argvsStream;
	};

	auto getSerializedValues = [&](const auto& variablesStream, auto supposedArgumentsAmout, bool onlyEntireArgvArgument = false)
	{
		core::vector<std::string> variablesHandle;
		variablesHandle.reserve(supposedArgumentsAmout);

		std::string tmpStream;
		for (auto x = 0ul; x < variablesStream.size(); ++x)
		{
			const auto character = variablesStream.at(x);

			if (onlyEntireArgvArgument ? (character == ' ') : (character == ','))
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
			else if (character == '\r' || character == '\n')
			{
				variablesHandle.push_back(tmpStream);
				break;
			}
			else
				tmpStream.push_back(character);
		}

		return variablesHandle;
	};

	if (argv.size() == PROPER_CMD_ARGUMENTS_AMOUNT)
		mode = CLM_CMD_LIST;
	else if (argv.size() == PROPER_BATCH_FILE_ARGUMENTS_AMOUNT)
		mode = CLM_BATCH_INPUT;
	else if (argv.size() > 1 && argv.size() < 7)
	{
		os::Printer::log("Single argument assumptions aren't allowed - too few arguments!", ELL_ERROR);
		os::Printer::log(requiredArgumentsMessage.data(), ELL_INFORMATION);
		return;
	}
	else if (argv.size() > 7)
	{
		os::Printer::log("Too many arguments!", ELL_ERROR);
		os::Printer::log(requiredArgumentsMessage.data(), ELL_INFORMATION);
		return;
	}
	else
	{
		mode = CLM_UNKNOWN;
		os::Printer::log(requiredArgumentsMessage.data(), ELL_INFORMATION);
		return;
	}

	if (std::string(argv[1]) == "-batch")
	{
		auto file = am->getFileSystem()->createAndOpenFile(argv[2].c_str());
		std::string fileStream;
		fileStream.resize(file->getSize(), ' ');
		file->read(fileStream.data(), file->getSize());
		fileStream += "\r\n";

		bool log = false;
		const auto batchInputStream = getBatchFilesArgvStream(fileStream);
		argvMappedList.resize(batchInputStream.size());

		for (auto i = 0ul; i < batchInputStream.size(); ++i)
		{
			const auto argvStream = *(batchInputStream.begin() + i);
			const auto arguments = getSerializedValues(argvStream, PROPER_CMD_ARGUMENTS_AMOUNT, true);

			if (arguments.size() != PROPER_CMD_ARGUMENTS_AMOUNT)
			{
				os::Printer::log("The input batch file command with id: " + std::to_string(i) + " is incorrect - skipping it!", ELL_WARNING);
				log = true;
			}

			fillArgvList(arguments, PROPER_CMD_ARGUMENTS_AMOUNT, i);
		}

		if(log)
			os::Printer::log(requiredArgumentsMessage.data(), ELL_INFORMATION);
	}
	else if (argv.size() == PROPER_CMD_ARGUMENTS_AMOUNT)
		fillArgvList(argv, argv.size(), 0);
	else
	{
		os::Printer::log("Invalid syntax!", ELL_ERROR);
		os::Printer::log(requiredArgumentsMessage.data(), ELL_INFORMATION);
	}

	// read from argv list to map and put variables to appropiate places in a cache
	rawVariables.resize(argvMappedList.size());

	for (auto c = 0ul; c < argvMappedList.size(); ++c)
	{
		const auto cmdArgumentsPerFile = *(argvMappedList.begin() + c);
		initializeMatchingMap(rawVariables[c]);

		for (auto i = 0; i < DTEA_COUNT; ++i)
		{
			auto& referenceVariableMap = rawVariables[c][i];

			for (auto z = 0; z < PROPER_CMD_ARGUMENTS_AMOUNT; ++z)
			{
				std::string rawFetchedCmdArgument = cmdArgumentsPerFile[z];
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
						auto& reference = rawVariables[c][DTEA_ACES];
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
						auto& reference = rawVariables[c][DTEA_REINHARD];
						reference.second = variablesHandle;
					}
					else if (variable == CHANNEL_NAMES)
					{
						// various amount of values allowed
						auto variablesHandle = getSerializedValues(variablesStream, 3);
						referenceVariableMap.second = variablesHandle;
					}
					else if (variable == CAMERA_TRANSFORM)
					{
						// various amount of values allowed, but useful is first 9 values
						auto variablesHandle = getSerializedValues(variablesStream, 9);
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

	performFInalStepForUsefulVariables();
	status = true;
}