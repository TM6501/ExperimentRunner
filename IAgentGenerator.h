#pragma once
#include "IAgent.h"
#include <string>

#include "yaml-cpp/yaml.h"

namespace Experiment
{
	class IAgentGenerator
	{
	public:
		virtual ~IAgentGenerator() {}
		
		// Instantiate self. Return true the file was valid yaml
		// and included all of the parameters needed.
		virtual bool loadConfigurationFromYaml(YAML::Node node) = 0;

		// Generate a random starting agent.
		virtual IAgent* getRandomIndividual() = 0;

		// Create mutated children based on the parents passed in.
		// In most situations, only a single agent will be passed in and
		// only a single agent will be returned, but the interface is
		// many-to-many to allow for all cases.
		virtual std::vector<IAgent*> getMutatedChildren(const std::vector<IAgent*>& parents) = 0;

		// Set the path where data will be store for all agents:
		virtual void setDataPath(std::string dataPath) { m_dataPath = dataPath; }

	protected:
		// The path to the top-level directory where all agents should store data:
		std::string m_dataPath;
	};
}