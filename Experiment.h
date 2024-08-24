#pragma once
#include <string>
#include "IAgentGenerator.h"
#include "IInstanceWorld.h"
#include "IPersistentWorld.h"

#include <fstream>
#include <string>

#include "yaml-cpp/yaml.h"

namespace Experiment
{
	// This class will read in all experiment details, allocate the
	// appropriate classes, and kick off the experiment. Most of the
	// heavy lifting will be done by the worlds and agents.
	class Experiment
	{
	public:
		Experiment(std::string yamlConfigFileName);
		~Experiment();

		void runExperiment();

		IAgent* getFinalAgent();
		double testSingleAgent(IAgent* testAgent);
		IAgent* getAgentFromJson(std::string filename);

	protected:
		// After runExperiment() does all of the necessary allocations and tests,
		// if everything is in order, doRunXXXExperiment() is called.
		void doRunInstanceExperiment();
		void doRunWorldExperiment();

		std::vector<IAgent*> getParentSet(const std::vector<IAgent*>& parents, int parentsInSet);

		bool createCSVOfstream();
		bool createDataDirectories();
		bool createWorld();
		bool createActorGenerator();
		
		bool readInConfig(std::string yamlConfigFileName);

		void DEBUG_printConfig(YAML::Node node, int tabLevel=0);
		void DEBUG_testAgentGeneration();

		IAgentGenerator* m_agentGenerator;
		IInstanceWorld* m_instanceWorld;
		IPersistentWorld* m_persistentWorld;

		YAML::Node m_experimentConfig;
		YAML::Node m_worldConfig;
		YAML::Node m_agentConfig;
		std::string m_experimentFileName;

		std::string m_dataDirectory;
		std::ofstream m_csvFileOutput;

		std::string getFilenameFromFullPath(std::string fullPath);

		std::string getCurrentTimeString();

		void deleteFinalAgent();
		IAgent* m_finalAgent;
		// A boolean to track if we've handed over a pointer to the final agent.
		// If we have, it is no longer our responsibility to delete it:
		bool m_finalAgentReleased;
	
	private:
		virtual const std::string classname() { return "Experiment"; }
	};
}