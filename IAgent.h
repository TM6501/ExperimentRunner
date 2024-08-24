#pragma once
#include <vector>
#include "json.hpp"
using json = nlohmann::json;

namespace Experiment
{
	class IAgent
	{
	public:
		virtual ~IAgent() {}

		virtual std::vector<double> calculateOutputs(const std::vector<double>& inputs) = 0;		

		virtual void writeSelfToJson(json& j) = 0;
		virtual void readSelfFromJson(json& j) = 0;
	};
}