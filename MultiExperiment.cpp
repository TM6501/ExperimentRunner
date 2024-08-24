#include "pch.h"
#include "MultiExperiment.h"
#include "Experiment.h"
#include <fstream>
#include <iostream>
#include <vector>
#include <filesystem>
#include <unordered_map>
#include <stdexcept>
#include <sstream>
#include <thread>

namespace Experiment
{
	// The map of YAML pairs that must remain in a min-max form:
	std::unordered_map<std::string, std::string> MultiExperiment::m_yamlMinMaxPairs = {		
		{"minWeight", "maxWeight"},
		{"minNeuronInputCount", "maxNeuronInputCount"},
		{"minPreBias", "maxPreBias"},
		{"minPostBias", "maxPostBias"},
		{"minConstraint", "maxConstraint"},
		{"RunsToGrade", "NumberOfRuns"},
		{"ParentsToKeepPerGeneration", "EpochPopulationSize"}
	};

	std::string MultiExperiment::getCurrentTimeString()
	{
		time_t time_ptr;
		time_ptr = time(NULL);

		// Get the localtime
		tm* tm_local = localtime(&time_ptr);

		std::stringstream ss;
		ss << std::setfill('0');
		ss << std::setw(4) << tm_local->tm_year + 1900 << "-"
			<< std::setw(2) << tm_local->tm_mon + 1 << "-"
			<< std::setw(2) << tm_local->tm_mday << "_"
			<< std::setw(2) << tm_local->tm_hour << "-"
			<< std::setw(2) << tm_local->tm_min << "-"
			<< std::setw(2) << tm_local->tm_sec;

		return ss.str();
	}

	MultiExperiment::MultiExperiment()
	{
	}

	std::string MultiExperiment::setupExperiments(std::string yamlFilename)
	{
		YAML::Node fullConfig = YAML::LoadFile(yamlFilename);
		std::string experimentDir = fullConfig["MultiExperimentVariables"]["PrimaryDataDirectory"].as<std::string>();
		experimentDir.append("/");
		experimentDir.append(fullConfig["MultiExperimentVariables"]["MultiExperimentName"].as<std::string>());		
		experimentDir.append("_");
		experimentDir.append(getCurrentTimeString());
		std::string yamlDir = experimentDir + "/experimentYaml/";

		// Create directory to hold the output yaml files:
		std::filesystem::create_directories(yamlDir);

		// Remove the multi-experiment stuff before duplicating:
		YAML::Node singleExperiment = YAML::Clone(fullConfig);
		singleExperiment.remove("MultiExperiment");
		singleExperiment.remove("MultiExperimentVariables");

		// Get our 1+ copies:
		std::vector<YAML::Node> allConfigs = separateYamlConfigs(singleExperiment);

		// Make sure all variable combinations make sense:
		allConfigs = getNumericallyValidYaml(allConfigs);

		// Go through each copy and change/add their data directories. Also, turn
		// off screen output for each to ensure they don't output to screen when
		// being run as secondary processes:
		for (unsigned int i = 0; i < allConfigs.size(); ++i)
		{
			allConfigs[i]["Experiment"]["Logging"]["PrimaryDataDirectory"] = experimentDir;
			allConfigs[i]["Experiment"]["Logging"]["ExperimentName"] = i;
			allConfigs[i]["Experiment"]["Logging"]["ScreenLogging"]["OutputMod"] = -1;
		}
		
		// Save them all to our yaml config directory:
		saveMultipleYaml(allConfigs, yamlDir);

		// Return the directory filled with properly prepared YAML files:
		return yamlDir;
	}

	void MultiExperiment::copyExample(std::string yamlFilename)
	{
		YAML::Node fullFile = YAML::LoadFile(yamlFilename);
		YAML::Node copy1 = YAML::Clone(fullFile);
		YAML::Node copy2 = YAML::Clone(fullFile);

		copy1["Configuration"]["TestValue"] = 1;
		copy2["Configuration"]["TestValue"] = 7;

		std::cout << "Original: " << fullFile << std::endl;
		std::cout << "Copy 1: " << copy1 << std::endl;
		std::cout << "Copy 2: " << copy2 << std::endl;
	}

	void MultiExperiment::stringVectorOut(const std::vector<std::string>& in)
	{
		for (unsigned int i = 0; i < in.size() - 1; ++i)
			std::cout << in[i] << ", ";
		
		std::cout << in[in.size() - 1];
	}

	YAML::Node MultiExperiment::getNodeAtDepth(const YAML::Node& top, const std::vector<std::string>& depth)
	{
		YAML::Node retVal = YAML::Clone(top);
		for (unsigned int i = 0; i < depth.size(); ++i)
			retVal = retVal[depth[i]];
		
		return retVal;
	}

	void MultiExperiment::setValue(YAML::Node& top, const std::vector<std::string>& depth, const std::string& value)
	{
		// Step through the progression to our node. Pointers don't work; need to
		// actually store each node step as we move through.
		std::vector<YAML::Node> steps{ top };
		for (unsigned int i = 0; i < depth.size() - 1; ++i)
		{
			steps.push_back(steps[steps.size() - 1][depth[i]]);
		}

		// Set the value:
		steps[steps.size() - 1][depth[depth.size() - 1]] = value;
	}

	void MultiExperiment::setValue(YAML::Node& top, const std::vector<std::string>& depth, const YAML::Node& value)
	{
		// Step through the progression to our node. Pointers don't work; need to
		// actually store each node step as we move through.
		std::vector<YAML::Node> steps{ top };
		for (unsigned int i = 0; i < depth.size() - 1; ++i)
		{
			steps.push_back(steps[steps.size() - 1][depth[i]]);
		}

		// Set the value:
		steps[steps.size() - 1][depth[depth.size() - 1]] = value;
	}

	void MultiExperiment::addCopies(std::vector<YAML::Node>& currentList, unsigned int copies)
	{
		unsigned int startLen = static_cast<unsigned int>(currentList.size());

		for (unsigned int i = 0; i < copies - 1; ++i)
		{
			for (unsigned int j = 0; j < startLen; ++j)
			{
				currentList.push_back(YAML::Clone(currentList[j]));
			}
		}
	}

	void MultiExperiment::recursiveSeparate(const YAML::Node& fullFile,
		std::vector<YAML::Node>& outputCopies, std::vector<std::string>& currentDepth)
	{
		YAML::Node deepNode = getNodeAtDepth(fullFile, currentDepth);		

		for (const auto& kv : deepNode)
		{
			if (kv.second.IsMap())
			{
				currentDepth.push_back(kv.first.as<std::string>());
				recursiveSeparate(fullFile, outputCopies, currentDepth);
				currentDepth.erase(currentDepth.end() - 1);
			}
			else if (kv.second.IsSequence())
			{
				// Make as many copies as we have items in the sequence:
				unsigned int startLen = static_cast<unsigned int>(outputCopies.size());
				unsigned int seqLen = static_cast<unsigned int>(kv.second.size());
				addCopies(outputCopies, seqLen);

				// Set the sequence values into our copies:
				currentDepth.push_back(kv.first.as<std::string>());
				for (unsigned int i = 0; i < seqLen; ++i)
				{
					for (unsigned int j = 0; j < startLen; ++j)
					{
						if (kv.second[i].IsSequence())
							setValue(outputCopies[(i * startLen) + j], currentDepth, kv.second[i]);
						else	
							setValue(outputCopies[(i * startLen) + j], currentDepth, kv.second[i].as<std::string>());
					}
				}

				currentDepth.erase(currentDepth.end() - 1);	
			}
		}
	}

	std::vector<YAML::Node> MultiExperiment::separateYamlConfigs(std::string yamlFilename)
	{
		YAML::Node fullFile = YAML::LoadFile(yamlFilename);

		return separateYamlConfigs(fullFile);
	}

	std::vector<YAML::Node> MultiExperiment::separateYamlConfigs(YAML::Node fullFile)
	{	
		// Start with a copy:
		std::vector<YAML::Node> outputCopies{ YAML::Clone(fullFile) };
		std::vector<std::string> currentDepth;

		for (const auto& kv : fullFile)
		{
			// Run once for every top-level item:
			currentDepth.clear();
			currentDepth.push_back(kv.first.as<std::string>());
			recursiveSeparate(fullFile, outputCopies, currentDepth);
		}		

		return outputCopies;
	}

	void MultiExperiment::saveMultipleYaml(const std::vector<YAML::Node>& manyYaml, const std::string& directory)
	{
		// Make sure the directory exists:
		std::filesystem::create_directories(directory);

		std::string filename;
		std::ofstream out;

		for (unsigned int i = 0; i < manyYaml.size(); ++i)
		{
			filename = directory + std::to_string(i) + ".yaml";
			out.open(filename.c_str());
			out << manyYaml[i];
			out.close();
		}
	}

	std::vector<YAML::Node> MultiExperiment::getNumericallyValidYaml(const std::vector<YAML::Node>& manyYaml)
	{
		std::vector<YAML::Node> retVal;

		for (unsigned int i = 0; i < manyYaml.size(); ++i)
		{
			if (getIsNumericallyValid(manyYaml[i]))
				retVal.push_back(manyYaml[i]);
		}

		return retVal;
	}

	bool MultiExperiment::getIsNumericallyValid(const YAML::Node& testYaml)
	{
		std::unordered_map<std::string, double> namedVals;
		std::vector<std::string> currentDepth;

		// Determine all of the name-value pairs:
		recursiveGetNamedValuePairs(testYaml, namedVals, currentDepth);

		// For each value in the map of min-max pairs, check to see if the pairs
		// are in our named values. If they are, the min must be less than or
		// equal to the max:
		bool retValue = true;
		for (auto i = m_yamlMinMaxPairs.begin(); i != m_yamlMinMaxPairs.end(); ++i)
		{
			// Both are there, they must be in order:
			if (namedVals.find(i->first) != namedVals.end() &&
				namedVals.find(i->second) != namedVals.end())
			{
				if (namedVals.at(i->second) < namedVals.at(i->first))
				{
					// std::cout << "Removing due to " << i->first << ", " << i->second << " combination." << std::endl;
					retValue = false;
					break;
				}
			}
		}

		return retValue;
	}

	void MultiExperiment::recursiveGetNamedValuePairs(const YAML::Node& fullYaml,
		std::unordered_map<std::string, double>& namedVals,
		std::vector<std::string>& currentDepth)
	{
		YAML::Node deepNode = getNodeAtDepth(fullYaml, currentDepth);
		double tempVal;

		for (const auto& kv : deepNode)
		{
			if (kv.second.IsMap())
			{
				currentDepth.push_back(kv.first.as<std::string>());
				recursiveGetNamedValuePairs(fullYaml, namedVals, currentDepth);
				currentDepth.erase(currentDepth.end() - 1);
			}
			else if (kv.second.IsScalar())
			{
				// Store the name-value:
				try
				{
					tempVal = std::stod(kv.second.as<std::string>());

					// If we made it here, we got a valid string:
					namedVals[kv.first.as<std::string>()] = tempVal;
				}
				catch (const std::invalid_argument& ia)
				{					
					ia;  // Prevent variable-not-used warnings.
					// Don't need to do anything. Non-double isn't an error, just
					// a case we don't need to handle.
					// std::cout << "Found - " << kv.first.as<std::string>() << ": " << kv.second.as<std::string>() << std::endl;
				}
			}
		}
	}

	/******************************************
	* Running multiple processes requireds different code for different
	* operating systems.  All OS-specific code will be below this comment.
	******************************************/
#ifdef __linux__
#include <unistd.h>
#include <sys/wait.h>
	void MultiExperiment::runDirectoryOfExperiments(std::string directory, unsigned int maxConcurrentChildren)
	{
		// Get the list of files in the directory:
		std::vector<std::string> allFiles;
		for (const auto& entry : std::filesystem::directory_iterator(directory))
			allFiles.push_back(entry.path().string());

		// Our process ID:
		pid_t id = 1;

		// Track how many processes have started and not yet finished:
		unsigned int currentConcurrentChildren = 0;
		unsigned int processesComplete = 0;

		// Start as many children as we have concurrent children:
		std::cout << "Starting up to " << maxConcurrentChildren 
			<< " processes to run the " << allFiles.size() << " experiments." << std::endl;
		while (currentConcurrentChildren < maxConcurrentChildren && id > 0 && allFiles.size() > 0)
		{
			id = startSingleChild(allFiles);
			if (id == 0) // Child process
			{
				doWork(allFiles);
			}			
			std::cout << "[" << getCurrentTimeString() << "]: Process "
				<< currentConcurrentChildren << " started." << std::endl;
			currentConcurrentChildren += 1;
		}

		// Wait on each child. When it finished, start another if needed.
		while (allFiles.size() > 0)
		{
			wait(NULL);  // Infinite wait until a process ends.
			std::cout << "[" << getCurrentTimeString() << "]: Child "
				<< processesComplete << " complete. Starting another." << std::endl;
			++processesComplete;
			id = startSingleChild(allFiles);
			if (id == 0)
				doWork(allFiles);
		}

		// Wait until all children are done:
		std::cout << "[" << getCurrentTimeString() 
			<< "]: Done creating children. Waiting for all to finish." << std::endl;
		while (currentConcurrentChildren > 0)
		{
			wait(NULL);
			currentConcurrentChildren -= 1;			
			std::cout << "[" << getCurrentTimeString() << "]: Child " 
				<< processesComplete << " complete; " << currentConcurrentChildren << " still running." << std::endl;
			++processesComplete;
		}

		std::cout << "[" << getCurrentTimeString() 
			<< "]: All children complete. Exiting." << std::endl;
	}

	pid_t MultiExperiment::startSingleChild(std::vector<std::string>& yamlVector)
	{
		pid_t id = fork();
		if (id > 0)
		{
			yamlVector.erase(
				yamlVector.begin(), yamlVector.begin() +
				std::min(1, static_cast<int>(yamlVector.size())));
		}
		return id;
	}

	void MultiExperiment::doWork(const std::vector<std::string>& yamlVector)
	{
		// Try running from the system in order to gather output:
		// Full Command: testApp_r.out single "yamlFile" > "yamlFile.logOut"
		char fullCommand[1000] = "./testApp_r.out single \"";
		strcat(fullCommand, yamlVector[0].c_str());
		strcat(fullCommand, "\" > \"");
		strcat(fullCommand, yamlVector[0].c_str());
		strcat(fullCommand, ".logOut\"");
		system(fullCommand);
		exit(0);
		
		// The old (preferred) way, when debug is not required:
		// Always take the first available, run the experiment, then quit:
		// Experiment singleExperiment(yamlVector[0]);
		// singleExperiment.runExperiment();
		// exit(0);
	}

#elif _WIN32
#include <windows.h>

	void MultiExperiment::runDirectoryOfExperiments(std::string directory, unsigned int maxConcurrentChildren)
	{	
		std::vector<STARTUPINFOA> startInfos;
		std::vector<PROCESS_INFORMATION> processInfos;
		std::vector<std::string> yamlFilenames;
		std::string temp;
		unsigned int loopWaitMilliseconds = 500;

		for (auto const& dir_entry : std::filesystem::directory_iterator{ directory })
		{
			temp = dir_entry.path().string();

			// Make sure it ends with ".yaml":
			if (temp.length() > 5 &&
				0 == temp.compare(temp.length() - 5, 5, ".yaml"))
			{
				yamlFilenames.push_back(temp);
			}			
		}
		
		unsigned int totalToRun = static_cast<unsigned int>(yamlFilenames.size());
		unsigned int totalStarted = 0;
		unsigned int totalFinished = 0;
		unsigned int currentConcurrent = 0;
		unsigned int totalErrors = 0;

		std::cout << "Running " << totalToRun << " experiments with "
			<< maxConcurrentChildren << " parallel processes." << std::endl;

		for (currentConcurrent = 0;
			currentConcurrent < maxConcurrentChildren && yamlFilenames.size() > 0;
			++currentConcurrent)
		{
			if (!startSingleChild(startInfos, processInfos, yamlFilenames))
			{
				// Maybe only one failed and we should try the others?
				--currentConcurrent;
				--totalToRun;
			}
			else
			{
				std::cout << "Child " << totalStarted << " of " << totalToRun 
					      << " started." << std::endl;
				++totalStarted;
			}
		}

		while (yamlFilenames.size() > 0)
		{
			// Sleep to not spin this loop too quickly:
			std::this_thread::sleep_for(std::chrono::milliseconds(loopWaitMilliseconds));

			// Start children if we need them:
			while (currentConcurrent < maxConcurrentChildren && yamlFilenames.size() > 0)
			{
				if (!startSingleChild(startInfos, processInfos, yamlFilenames))
				{
					--totalToRun;					
				}
				else
				{
					std::cout << "Child " << totalStarted << " of " << totalToRun
						<< " started." << std::endl;
					++totalStarted;
					++currentConcurrent;
				}
			}

			// Check to see if each is still running:
			for (unsigned int i = 0; i < processInfos.size(); ++i)
			{
				// Process is done? Remove it and decrement the number of
				// children currently running:
				if (!checkRunningAndRemove(startInfos, processInfos, i))
				{	
					--currentConcurrent;
					std::cout << "Child " << totalFinished << " of " << totalToRun
						<< " finished." << std::endl;					
					++totalFinished;
				}
			}
		}

		std::cout << "All children created. Waiting for them to finish." << std::endl;
		while (currentConcurrent > 0)
		{
			for (unsigned int i = 0; i < processInfos.size(); ++i)
			{
				// Process is done? Remove it and decrement the number of
				// children currently running:
				if (!checkRunningAndRemove(startInfos, processInfos, i))
				{
					--currentConcurrent;
					std::cout << "Child " << totalFinished << " of " << totalToRun
						<< " finished." << std::endl;
					++totalFinished;
				}
			}
		}

		std::cout << "All experiments complete." << std::endl;
		if (totalErrors > 0)
		{
			std::cout << totalErrors << " experiments hit errors starting and weren't run." << std::endl;
		}
		else
		{
			std::cout << "All experiments ran successfully." << std::endl;
		}
	}

	bool MultiExperiment::startSingleChild(std::vector<STARTUPINFOA>& startInfos,
		std::vector<PROCESS_INFORMATION>& processInfos,
		std::vector<std::string>& workToDo)
	{
		// Create and zero-out the Windows structs:
		STARTUPINFOA tempSi;
		PROCESS_INFORMATION tempPi;
		ZeroMemory(&tempSi, sizeof(tempSi));
		tempSi.cb = sizeof(tempSi);
		ZeroMemory(&tempPi, sizeof(tempPi));

		// Lay out the work to be done:
		char exeBuffer[1000] = "Starter.exe single ";
		strcat_s(exeBuffer, 1000, workToDo[0].c_str());
		workToDo.erase(workToDo.begin(), workToDo.begin() + 1);

		// Add the infos to the vectors BEFORE starting the process:
		startInfos.push_back(tempSi);
		processInfos.push_back(tempPi);

		// Start the process:
		if (CreateProcessA(
			NULL, // No module name
			exeBuffer, // Command line to execute
			NULL, // Process handle not inheritable
			NULL, // Thread handle not inheritable
			FALSE, // Handle inheritance is false
			0, // No creation flags
			NULL, // Use parent's environment block
			NULL, // Use parent's starting directory
			&startInfos[startInfos.size() - 1],  // pointer to startup info
			&processInfos[processInfos.size() - 1] // pointer to process_information
		))
		{
			return true;
		}
		else  // Failed to start the process:
		{
			std::cout << "Failed to start a process with the following command line: " << std::endl;
			std::cout << exeBuffer << std::endl;
			startInfos.pop_back();
			processInfos.pop_back();
			return false;
		}
	}

	bool MultiExperiment::checkRunningAndRemove(std::vector<STARTUPINFOA>& startInfos,
		std::vector<PROCESS_INFORMATION>& processInfos,
		unsigned int processToCheck)
	{
		DWORD exitCode;
		GetExitCodeProcess(processInfos[processToCheck].hProcess, &exitCode);

		// Process is still running:
		if (exitCode == STILL_ACTIVE)
		{
			return true;
		}
		// Not running. Clean it up and remove it:
		else
		{
			CloseHandle(processInfos[processToCheck].hProcess);
			CloseHandle(processInfos[processToCheck].hThread);
			startInfos.erase(startInfos.begin() + processToCheck);
			processInfos.erase(processInfos.begin() + processToCheck);
			return false;
		}
	}

#endif

}
