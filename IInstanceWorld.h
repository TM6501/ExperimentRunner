#pragma once
#include <string>
#include <vector>
#include "IAgent.h"

#include "yaml-cpp/yaml.h"

namespace Experiment
{
	// This interface defines what interactions must be made available
	// for an instance world. An instance world is one in which each
	// agent acts inside their own bubble universe. Generally, this
	// universe is created for the express purposed of testing a single
	// agent and then it is destroyed. This is the more-typical reinforcement
	// learning procedure.
	class IInstanceWorld
	{
	public:
		virtual ~IInstanceWorld() {}

		// Load all of what we need from a Yaml file. Return
		// True if we are successful and found all data we needed:
		virtual bool loadConfigurationFromYaml(YAML::Node node) = 0;

		// Create the world, return true if successful.
		virtual bool createTestWorld() = 0;

		// Test a population of individuals, return their fitnesses:
		virtual std::vector<double> getPopulationFitnesses(const std::vector<IAgent*>& population) = 0;
	};
}