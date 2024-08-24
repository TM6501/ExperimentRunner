// ExperimentRunner.cpp : Defines the functions for the static library.
//
#include "pch.h"
#include "framework.h"

#include "Experiment.h"
#include "IAgent.h"

#include "CGPGenerator.h"
#include "AbstractCGPIndividual.h"
#include "GymWorldGenerator.h"

#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <random>
#include <filesystem>
#include <iomanip>
#include <ctime>
#include <sstream>

#include "yaml-cpp/yaml.h"

extern unsigned int DEBUG_LEVEL;

namespace Experiment
{
	Experiment::Experiment(std::string yamlConfigFileName) :
		m_agentGenerator(nullptr), m_instanceWorld(nullptr), m_persistentWorld(nullptr),
		m_experimentConfig(YAML::Null), m_worldConfig(YAML::Null),
		m_agentConfig(YAML::Null), m_experimentFileName(yamlConfigFileName),
		m_dataDirectory(""), m_finalAgent(nullptr),
		m_finalAgentReleased(false)
	{
		if (!readInConfig(m_experimentFileName))
		{
			std::cout << "Failed to read in experiment yaml file: "
				<< m_experimentFileName << ". Exiting." << std::endl;
			exit(-1);
		}
		if (DEBUG_LEVEL > 3)
			DEBUG_printConfig(m_experimentConfig);
	}

	Experiment::~Experiment()
	{
		if (DEBUG_LEVEL > 4)
			std::cout << "At top of \"" << classname() << ":" << __func__ << "\"" << std::endl;

		if (m_agentGenerator != nullptr)
			delete m_agentGenerator;
		
		if (m_instanceWorld != nullptr)
			delete m_instanceWorld;

		if (m_persistentWorld != nullptr)
			delete m_persistentWorld;
	}

	void Experiment::deleteFinalAgent()
	{
		if (DEBUG_LEVEL > 4)
			std::cout << "At top of \"" << classname() << ":" << __func__ << "\"" << std::endl;

		if (m_finalAgent != nullptr && !m_finalAgentReleased)
			delete m_finalAgent;

		m_finalAgent = nullptr;
		m_finalAgentReleased = false;
	}

	bool Experiment::readInConfig(std::string yamlConfigFileName)
	{
		if (DEBUG_LEVEL > 4)
			std::cout << "At top of \"" << classname() << ":" << __func__ << "\"" << std::endl;

		YAML::Node fullFile = YAML::LoadFile(yamlConfigFileName);
		bool retVal = true;
		m_experimentConfig = fullFile["Experiment"];
				
		if (fullFile["World"])
			m_worldConfig = fullFile["World"];
		else
		{
			std::cout << "World must be defined in experiment yaml." << std::endl;
			retVal = false;
		}
		
		if (fullFile["AgentGeneration"])
			m_agentConfig = fullFile["AgentGeneration"];
		else
		{
			std::cout << "AgentGeneration must be defined in experiment yaml." << std::endl;
			retVal = false;
		}

		return retVal;
	}

	std::string Experiment::getFilenameFromFullPath(std::string fullPath)
	{
		if (DEBUG_LEVEL > 4)
			std::cout << "At top of \"" << classname() << ":" << __func__ << "\"" << std::endl;

		// Get the position of the last slash or backslash:
		auto slashPos = fullPath.find_last_of("/\\");

		// If it didn't exist, return the full path:
		if (slashPos == std::string::npos)
			return fullPath;
		// Otherwise only return the filename at the end:
		else
			return fullPath.substr(slashPos + 1);			
	}

	std::string Experiment::getCurrentTimeString()
	{
		if (DEBUG_LEVEL > 4)
			std::cout << "At top of \"" << classname() << ":" << __func__ << "\"" << std::endl;

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

	bool Experiment::createDataDirectories()
	{
		if (DEBUG_LEVEL > 4)
			std::cout << "At top of \"" << classname() << ":" << __func__ << "\"" << std::endl;

		m_dataDirectory = m_experimentConfig["Logging"]["PrimaryDataDirectory"].as<std::string>();
		m_dataDirectory.append("/");
		m_dataDirectory.append(m_experimentConfig["Logging"]["ExperimentName"].as<std::string>());

		// The same experiment may be run multiple times. Add a date-time to make it unique:
		m_dataDirectory += "_" + getCurrentTimeString();
		
		std::filesystem::create_directories(m_dataDirectory);

		bool directoryCreationSuccess = true;
		bool fileCopySuccess = false;

		// Create a directory to store the agent data files:
		directoryCreationSuccess = directoryCreationSuccess && 
			std::filesystem::create_directories(m_dataDirectory + "/" + "Agents");

		// Create a directory to store the world data files:
		directoryCreationSuccess = directoryCreationSuccess && 
			std::filesystem::create_directories(m_dataDirectory + "/" + "World");

		// Create a directory to store the initial configuration files:
		directoryCreationSuccess = directoryCreationSuccess && 
			std::filesystem::create_directories(m_dataDirectory + "/" + "Configuration");

		// Copy the configuration file to the configuration directory to aid in repeatability
		// of the experiments:
		std::string expConfigFname = getFilenameFromFullPath(m_experimentFileName);
			
		if (directoryCreationSuccess)
		{
			fileCopySuccess = std::filesystem::copy_file(m_experimentFileName, m_dataDirectory + "/Configuration/" + expConfigFname);
		}

		// If we didn't succeed entirely, let the user know:
		if (!directoryCreationSuccess)
			std::cout << "Failed to create all of the data directories." << std::endl;
		else if (!fileCopySuccess)
			std::cout << "Failed to copy over the configuration file." << std::endl;

		return (directoryCreationSuccess && fileCopySuccess);
	}

	bool Experiment::createCSVOfstream()
	{
		if (DEBUG_LEVEL > 4)
			std::cout << "At top of \"" << classname() << ":" << __func__ << "\"" << std::endl;

		// If we're logging to CSV, open the corresponding file:
		if (m_experimentConfig["Logging"]["CSVFileLogging"]["OutputMod"].as<int>() > 0)
		{			
			m_csvFileOutput.open(m_dataDirectory + "/epochOutput.csv");
			if (m_csvFileOutput.is_open())
			{
				// Write the top line (column titles):
				m_csvFileOutput << "epoch,bestFitness" << std::endl;
				return true;
			}
			else
			{
				return false;
			}
		}
		else
		{
			// Success if it didn't need to be opened:
			return true;
		}
	}

	bool Experiment::createWorld()
	{
		if (DEBUG_LEVEL > 4)
			std::cout << "At top of \"" << classname() << ":" << __func__ << "\"" << std::endl;

		bool retVal = true;

		if (m_worldConfig.IsNull())
		{
			std::cout << "World config not provided." << std::endl;
			retVal = false;
		}
		else if (m_worldConfig["Type"].as<std::string>() != "Instance")
		{
			std::cout << "Instance is the only currently implemented World Type." << std::endl;
			retVal = false;
		}

		// We have an instance world type.  Create it:
		else
		{
			m_instanceWorld = new CGP::GymWorldGenerator();
			if (m_instanceWorld->loadConfigurationFromYaml(m_worldConfig))
			{
				if (!m_instanceWorld->createTestWorld())
				{
					std::cout << "Failed to create the test world." << std::endl;
					retVal = false;
				}
			}
			else
			{
				std::cout << "Failed to read in the world's YAML configuration." << std::endl;
				retVal = false;
			}
		}
		return retVal;
	}

	bool Experiment::createActorGenerator()
	{
		if (DEBUG_LEVEL > 4)
			std::cout << "At top of \"" << classname() << ":" << __func__ << "\"" << std::endl;

		bool retVal = true;

		if (m_agentConfig.IsNull())
		{
			std::cout << "AgentGeneration must be defined in the experiment yaml." << std::endl;
			retVal = false;
		}

		if (m_agentConfig["Type"].as<std::string>() == "CGP")
		{
			m_agentGenerator = new CGP::CGPGenerator();
			if (!m_agentGenerator->loadConfigurationFromYaml(m_agentConfig))
			{
				std::cout << "Failed to read in AgentGeneration yaml information." << std::endl;
				retVal = false;
			}
		}
		else
		{
			std::cout << "Non 'CGP' AgentGeneration types not yet implemented." << std::endl;
			return retVal = false;
		}

		return retVal;
	}

	void Experiment::runExperiment()
	{
		if (DEBUG_LEVEL > 4)
			std::cout << "At top of \"" << classname() << ":" << __func__ << "\"" << std::endl;

		bool canProceed = true;

		if (!createDataDirectories())
		{
			std::cout << "Failed to create the logging directories." << std::endl;
			canProceed = false;
		}

		if (!createWorld())
		{
			std::cout << "Failed to create the world." << std::endl;
			canProceed = false;
		}

		if (!createActorGenerator())
		{
			std::cout << "Failed to create actor generator." << std::endl;
			canProceed = false;
		}

		if (!createCSVOfstream())
		{
			std::cout << "Failed to open CSV output file." << std::endl;
			canProceed = false;
		}

		if (!canProceed)
		{
			std::cout << "Errors prevent the experiment from proceeding." << std::endl;
			return;
		}

		if (m_worldConfig["Type"].as<std::string>() == "Instance")
			doRunInstanceExperiment();
		else
			std::cout << "Only Instance-world experiments are implemented, so far." << std::endl;
	}

	std::vector<IAgent*> Experiment::getParentSet(const std::vector<IAgent*>& parents, int parentsInSet)
	{
		if (DEBUG_LEVEL > 4)
			std::cout << "At top of \"" << classname() << ":" << __func__ << "\"" << std::endl;

		// Make them static so they aren't recreated every time:
		static std::random_device rd;
		static std::mt19937 rng(rd());
		
		// Simple, typical solution:
		if (parents.size() == static_cast<unsigned int>(parentsInSet))
			return parents;

		if (static_cast<unsigned int>(parentsInSet) > parents.size())
		{
			std::cout << "Cannot return more parents than exist." << std::endl;
			exit(-1);
		}

		std::uniform_int_distribution<int> parDist(0, static_cast<int>(parents.size() - 1));

		if (parentsInSet == 1)
		{
			std::vector<IAgent*> retVal{ parents[parDist(rng)] };
			return retVal;
		}
		// More than one parent in a set:
		else
		{
			std::vector<int> used;
			std::vector<IAgent*> retVal;
			int selection;
			while (retVal.size() < static_cast<unsigned int>(parentsInSet))
			{
				// Keep choosing random values until we hit one we haven't used yet:
				selection = parDist(rng);
				while (std::find(used.begin(), used.end(), selection) != used.end())
					selection = parDist(rng);

				// Add to our used array and return parents:
				used.push_back(selection);
				retVal.push_back(parents[selection]);
			}

			return retVal;
		}		
	}

	IAgent* Experiment::getAgentFromJson(std::string filename)
	{
		if (DEBUG_LEVEL > 4)
			std::cout << "At top of \"" << classname() << ":" << __func__ << "\"" << std::endl;

		IAgent* retVal = nullptr;

		// Make sure we have an agent generator:
		if (m_agentGenerator == nullptr && !createActorGenerator())
		{
			std::cout << "Need an agent generator to load agent from Json." << std::endl;
		}
		else
		{
			std::ifstream inFile(filename.c_str());			
			if (inFile.is_open())
			{
				json j = json::parse(inFile);
				inFile.close();
				retVal = m_agentGenerator->getRandomIndividual();
				retVal->readSelfFromJson(j);
			}
			else
			{
				std::cout << "Failed to open " << filename << std::endl;
			}
		}

		return retVal;			
	}

	void Experiment::doRunInstanceExperiment()
	{
		if (DEBUG_LEVEL > 4)
			std::cout << "At top of \"" << classname() << ":" << __func__ << "\"" << std::endl;

		// Delete any final agent from a previous experiment:
		deleteFinalAgent();

		// Assuming all parameters are properly provided:
		int startingPop = m_experimentConfig["Population"]["StartingPopulationSize"].as<int>();
		int epochPop = m_experimentConfig["Population"]["EpochPopulationSize"].as<int>();
		int parentsPerGeneration = m_experimentConfig["Population"]["ParentsToKeepPerGeneration"].as<int>();
		int parentsPerChild = m_experimentConfig["Population"]["ParentsToUseForEachChildCreation"].as<int>();
		bool parentsRepeatTest = m_experimentConfig["Population"]["ParentsRepeatExamination"].as<bool>();

		double maxScore = m_experimentConfig["Ending"]["MaxScore"].as<double>();
		int maxEpochs = m_experimentConfig["Ending"]["MaxEpochs"].as<int>();

		int screenLogMod = m_experimentConfig["Logging"]["ScreenLogging"]["OutputMod"].as<int>();
		int csvLogMod = m_experimentConfig["Logging"]["CSVFileLogging"]["OutputMod"].as<int>();

		std::vector<double> parentScores;
		std::vector<IAgent*> parents;
		std::vector<IAgent*> population;
		std::vector<double> scores;
		std::vector<IAgent*> tempParents;
		std::vector<IAgent*> tempChildren;
		m_finalAgent = nullptr;
		
		// Generate our starting population:
		for (int i = 0; i < startingPop; ++i)
			population.push_back(m_agentGenerator->getRandomIndividual());

		for (int epoch = 0; epoch < maxEpochs; ++epoch)
		{
			scores = m_instanceWorld->getPopulationFitnesses(population);

			// If the parents were held out of this test, put them and their scores back in at
			// the end so that better children will be selected ahead of them:
			if (!parentsRepeatTest)
			{
				population.insert(population.end(), parents.begin(), parents.end());
				scores.insert(scores.end(), parentScores.begin(), parentScores.end());
			}
	
			// Fill up our parents vector with the best:
			parents.clear();
			parentScores.clear();
			while (parents.size() < static_cast<unsigned int>(parentsPerGeneration))
			{
				// Get the index of the highest score:
				auto it = std::minmax_element(scores.begin(), scores.end());
				int max_idx = static_cast<int>(std::distance(scores.begin(), it.second));

				// Copy that score and individual over:
				parents.push_back(population[max_idx]);
				parentScores.push_back(scores[max_idx]);

				// Remove them from the population:
				population.erase(population.begin() + max_idx);
				scores.erase(scores.begin() + max_idx);
			}

			// Log to CSV if it is enabled and available:
			if (csvLogMod > 0 && epoch % csvLogMod == 0 && m_csvFileOutput.is_open())
			{
				// Log the epoch and the best score we found so far.
				m_csvFileOutput << epoch << "," << parentScores[0] << std::endl;
			}

			// The best is now sitting in population/scores [0]:
			if (parentScores[0] >= maxScore)
			{
				if (screenLogMod > 0)
					std::cout << "Epoch " << epoch << " solution found: " << parentScores[0] << std::endl;				
				break;
			}

			// Print the epoch's scores:
			if (screenLogMod != -1 && epoch % screenLogMod == 0)
			{
				std::cout << "Epoch " << epoch << " best score: " << parentScores[0] << std::endl;
			}

			// Go through the rest of the population and free each individual's memory:
			for (unsigned int i = 0; i < population.size(); ++i)
				delete population[i];

			population.clear();
			scores.clear();

			// If parents have to "re-prove" themselves, put them back into the population:
			if (parentsRepeatTest)
				population.insert(population.end(), parents.begin(), parents.end());

			while (population.size() < static_cast<unsigned int>(epochPop))
			{
				tempParents = getParentSet(parents, parentsPerChild);
				tempChildren = m_agentGenerator->getMutatedChildren(tempParents);

				population.insert(population.end(), tempChildren.begin(), tempChildren.end());
			}
		}

		// Gather the final agent, delete the rest of the parents:
		m_finalAgent = parents[0];
		
		for (unsigned int i = 1; i < parents.size(); ++i)
			delete parents[i];

		parents.clear();
	}

	IAgent* Experiment::getFinalAgent()
	{
		if (DEBUG_LEVEL > 4)
			std::cout << "At top of \"" << classname() << ":" << __func__ << "\"" << std::endl;

		return m_finalAgent;
	}

	double Experiment::testSingleAgent(IAgent* testAgent)
	{
		if (DEBUG_LEVEL > 4)
			std::cout << "At top of \"" << classname() << ":" << __func__ << "\"" << std::endl;

		double retVal = -1.0;
		// Worlds test full populations; create one:
		std::vector<IAgent*> population{ testAgent };

		// If we created a persistent world, we can't test a single agent:
		if (m_persistentWorld != nullptr && m_instanceWorld == nullptr)
		{
			std::cout << "Can't test a single agent in a persistent world." << std::endl;
		}
		// Maybe we just need to create the world?
		else if (m_instanceWorld == nullptr)
		{
			if (createWorld())
			{ 
				retVal = m_instanceWorld->getPopulationFitnesses(population)[0];
			}
		}
		// World already exists, just test:
		else
		{
			retVal = m_instanceWorld->getPopulationFitnesses(population)[0];
		}

		return retVal;
	}

	void Experiment::doRunWorldExperiment()
	{
		if (DEBUG_LEVEL > 4)
			std::cout << "At top of \"" << classname() << ":" << __func__ << "\"" << std::endl;

		std::cout << "World experiments not yet implemented." << std::endl;
	}

	void Experiment::DEBUG_testAgentGeneration()
	{
		if (m_agentGenerator == nullptr)
		{
			std::cout << "Agent generator not available." << std::endl;
			return;
		}

		// For now, we just test CGP individuals:
		IAgent* testAgent = m_agentGenerator->getRandomIndividual();

		std::cout << "Random agent's genotype: " << std::endl;
		static_cast<CGP::AbstractCGPIndividual*>(testAgent)->printGenotype();

		std::vector<IAgent*> parents{ testAgent };
		std::vector<IAgent*> children = m_agentGenerator->getMutatedChildren(parents);

		std::cout << "\n\nMutated child's genotype: " << std::endl;
		static_cast<CGP::AbstractCGPIndividual*>(children[0])->printGenotype();
	}

	void Experiment::DEBUG_printConfig(YAML::Node node, int tabLevel)
	{
		// Recursively print out everything in the given node:
		switch (node.Type())
		{
			case YAML::NodeType::Undefined:
				std::cout << "UNDEFINED" << std::endl;
			break;
			
			case YAML::NodeType::Null:
				std::cout << "NULL" << std::endl;
			break;
			
			case YAML::NodeType::Scalar:
				std::cout << node.as<std::string>() << std::endl;
			break;
			
			case YAML::NodeType::Sequence:
				std::cout << std::endl;
				for (unsigned int i = 0; i < node.size(); ++i)
				{
					// Add the needed tabs:
					for (int t = 0; t < tabLevel; ++t)
						std::cout << "  ";
					std::cout << i << ": ";
					DEBUG_printConfig(node[i], tabLevel + 1);
				}
			break;
			
			case YAML::NodeType::Map:
				std::cout << std::endl;
				for (YAML::const_iterator it = node.begin(); it != node.end(); ++it)
				{
					// Add the needed tabs:
					for (int t = 0; t < tabLevel; ++t)
						std::cout << "  ";
					std::cout << it->first << ": ";
					DEBUG_printConfig(it->second, tabLevel + 1);
				}
			break;
		}
	}
}