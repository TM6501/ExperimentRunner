#pragma once
/* The Gym Sage Runner is a class designed to allow any C++ agent, in this case
* a JBrain, to run test or train itself on a Python Gymnasium environment while
* also getting input from a "sage". This class will load up a trained agent
* and return its selected action along with the observation to the newly training
* or testing agent.
* 
* Because this class is being built for this project's specific purposes, it will
* make a few assumptions.
*  - The 'sage' is a trained PPO agent from stable_baselines3.* 
*/
#include <string>
#include <vector>
#include "Python.h"

#include <mutex>
#include <thread>
#include <queue>

namespace Experiment
{
	class GymSageRunner
	{
	private:
		// Private constructor to enforce the singleton nature.
		// This class shouldn't be instantiated with any other using a 
		// Python interface.
		GymSageRunner();

	protected:
		// To simplify Python-interface usage, we'll use and load a python
		// file specifically written for us:
		PyObject* m_sageRunnerModule;		
		PyObject* m_resetFunc;
		PyObject* m_loadFunc;		
		PyObject* m_stepFunc;
		PyObject* m_renderFunc;
		PyObject* m_sageRunnerInstance;
				
		PyObject* m_gcModule;
		PyObject* m_gcCollectFunction;

		int m_observationSize;
		int m_actionSize;
		bool m_useArgMax;
		bool m_initialized;
		float m_currentReward;
		bool m_environmentDone;
		std::vector<double> m_mostRecentObs;
		std::vector<double> m_mostRecentAgentAction;

		bool initializePython();
		void finalizePython();
		void addLocalDirectory();

	public:
		static GymSageRunner* getInstance();

		void initialize(
			bool useArgMax,
			int observationSize,
			int actionSize,
			std::string envName = "CartPole-v1",
			std::string sageName = "ppo_sage",
			std::string renderMode = "rgb_array"
		);

		~GymSageRunner();

		virtual std::string classname() { return "GymSageRunner"; }

		// Reset the environment. Set m_mostRecentObs & m_mostRecentAgentAction
		bool reset();

		// Ask the environment to render itself. Set nothing.
		void render();

		// Take a step forward in the environment. Set m_mostRecentObs,
		// m_mostRecentAgentAction, m_environmentDone, and m_currentReward.
		bool step(std::vector<double> action);

		// Get recent data:
		std::vector<double> getRecentObs();
		std::vector<double> getRecentAgentAction();
		bool getEnvironmentDone();
		float getCurrentReward();
		void getAllStatus(std::vector<double>& recentObs,
			std::vector<double>& recentAction, bool& envDone, float& reward);

		bool testStringArguments(std::string stringArg1, std::string stringArg2);
	}; // End class GymSageRunner
} // End namespace Experiment

