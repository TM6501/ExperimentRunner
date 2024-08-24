#include "pch.h"
#include "GymSageRunner.h"
#include <algorithm>
#include <numeric>
#include <iostream>  // Error printing

namespace Experiment
{
	GymSageRunner* GymSageRunner::getInstance()
	{
		// Not built to handle multiple threads:
		static GymSageRunner* instance = new GymSageRunner();

		return instance;
	}

	GymSageRunner::GymSageRunner()
		: m_sageRunnerModule(nullptr),
		m_resetFunc(nullptr),
		m_loadFunc(nullptr),
		m_stepFunc(nullptr),
		m_renderFunc(nullptr),
		m_sageRunnerInstance(nullptr),
		m_gcModule(nullptr),
		m_gcCollectFunction(nullptr),
		m_observationSize(-1),
		m_actionSize(-1),
		m_useArgMax(false),
		m_initialized(false),
		m_currentReward(0.0),
		m_environmentDone(false)
	{}

	GymSageRunner::~GymSageRunner()
	{
		finalizePython();
	}

	void GymSageRunner::finalizePython()
	{
		// Clear the SageRunner environment pointers:
		Py_CLEAR(m_stepFunc);
		Py_CLEAR(m_loadFunc);
		Py_CLEAR(m_resetFunc);
		Py_CLEAR(m_sageRunnerModule);

		// Clear the garbage collection:
		// Py_CLEAR(m_gcCollectFunction);
		// Py_CLEAR(m_gcModule);
	}

	void GymSageRunner::initialize(bool useArgMax, int observationSize, int actionSize)
	{
		// Don't repeat the initialization process:
		if (!m_initialized)
		{
			m_useArgMax = useArgMax;
			m_observationSize = observationSize;
			m_actionSize = actionSize;
			m_initialized = true;

			if (!initializePython())
			{
				std::cout << "Python initialization error." << std::endl;
				exit(-1);
			}

			// Load up the environment and agent:
			PyObject_CallNoArgs(m_loadFunc);
		}
	}

	// Add the local directory to the Python search path:
	void GymSageRunner::addLocalDirectory()
	{
		PyConfig config;
		PyConfig_InitPythonConfig(&config);
		config.module_search_paths_set = 1;
		PyWideStringList_Append(&config.module_search_paths, L".");
		Py_InitializeFromConfig(&config);
	}

	bool GymSageRunner::initializePython()
	{
		Py_SetProgramName(L"CGP_CPP_TestApp");
		Py_Initialize();
		// addLocalDirectory();

		// PyRun_SimpleString("import sys");
		// PyRun_SimpleString("print(sys.version)");
		// PyRun_SimpleString("print(f'C++: {sys.path}')");

		// Get the gym module:
		PyObject* pSageRunnerName = PyUnicode_FromString("SageRunner");
		m_sageRunnerModule = PyImport_Import(pSageRunnerName);
		if (m_sageRunnerModule == nullptr)
		{
			PyErr_Print();
			return false;
		}
		Py_XDECREF(pSageRunnerName);

		// Get our instance of SageRunner:
		PyObject* getInstanceFunction = PyObject_GetAttrString(m_sageRunnerModule, "getInstance");
		if (getInstanceFunction == nullptr)
		{
			PyErr_Print();
			return false;
		}

		m_sageRunnerInstance = PyObject_CallNoArgs(getInstanceFunction);
		if (m_sageRunnerInstance == nullptr)
		{
			PyErr_Print();
			return false;
		}

		// No longer need a reference to the function:
		Py_XDECREF(getInstanceFunction);

		// Get each function:
		m_resetFunc = PyObject_GetAttrString(m_sageRunnerInstance, "reset");
		if (m_resetFunc == nullptr)
		{
			PyErr_Print();
			return false;
		}

		m_loadFunc = PyObject_GetAttrString(m_sageRunnerInstance, "load");
		if (m_loadFunc == nullptr)
		{
			PyErr_Print();
			return false;
		}

		m_stepFunc = PyObject_GetAttrString(m_sageRunnerInstance, "step");
		if (m_stepFunc == nullptr)
		{
			PyErr_Print();
			return false;
		}

		m_renderFunc = PyObject_GetAttrString(m_sageRunnerInstance, "render");
		if (m_renderFunc == nullptr)
		{
			PyErr_Print();
			return false;
		}

		return true;
	}

	bool GymSageRunner::step(std::vector<double> action)
	{
		// Create a tuple for our input action. Python API requires
		// tuples as arguments:
		PyObject* pActionTuple = nullptr;
		PyObject* pStepReturn = nullptr;

		// ArgMax would mean that we want only the index of the max value.
		// !ArgMax means we want the whole action list.
		
		// Some code must be duplicated because PyFloat_FromDouble implicitly
		// allocates memory that goes out of scope after the closing if brace:
		if (!m_useArgMax)
		{			
			pActionTuple = PyTuple_New(m_actionSize);
			for (int i = 0; i < m_actionSize; ++i)
				PyTuple_SetItem(pActionTuple, i, PyFloat_FromDouble(action[i]));

			// Call the step function:
			pStepReturn = PyObject_CallObject(m_stepFunc, pActionTuple);
			Py_XDECREF(pActionTuple);
		}
		/***********************
		Someday, we need to figure out why m_useArgMax results in a crash when
		calling the step function. It will come back and bite us on the ass.
		*********************/
		else
		{
			pActionTuple = PyTuple_New(1);
			int maxElemIdx = static_cast<int>(std::max_element(action.begin(), action.end())
				- action.begin());
			PyList_SetItem(pActionTuple, 0, PyFloat_FromDouble(double(maxElemIdx)));
			
			// Call the step function:
			pStepReturn = PyObject_CallObject(m_stepFunc, pActionTuple);
			Py_XDECREF(pActionTuple);
		}
				
		if (pStepReturn == nullptr)
		{
			std::cout << "pStepReturn == nullptr" << std::endl;
			std::cout << "useArgMax: " << m_useArgMax << std::endl;
			
			for (unsigned int i = 0; i < action.size(); ++i)
				std::cout << action[i] << ", ";
			std::cout << std::endl;
			
			PyErr_Print();
			return false;
		}

		// Step returns 4 values: obs (list), reward (double), done (bool),
		// and action (list). The action is the "sage action", meaning the action
		// that WOULD be taken by the trained agent as its next step given the
		// current action:
		PyObject* pObs = PyTuple_GetItem(pStepReturn, 0);
		PyObject* pReward = PyTuple_GetItem(pStepReturn, 1);
		PyObject* pDone = PyTuple_GetItem(pStepReturn, 2);
		PyObject* pAction = PyTuple_GetItem(pStepReturn, 3);

		// Fill the observation and action:
		// Fill the observation:
		m_mostRecentObs.clear();
		for (int i = 0; i < m_observationSize; ++i)
			m_mostRecentObs.push_back(PyFloat_AsDouble(PyList_GetItem(pObs, i)));

		m_mostRecentAgentAction.clear();
		for (int i = 0; i < m_actionSize; ++i)
			m_mostRecentAgentAction.push_back(PyFloat_AsDouble(PyList_GetItem(pAction, i)));

		// Get Done:
		if (PyBool_Check(pDone))
		{
			m_environmentDone = PyObject_IsTrue(pDone);
		}
		else
		{
			PyErr_Print();
			std::cout << "Return value 2 wasn't a boolean." << std::endl;
			return false;
		}

		m_currentReward += static_cast<float>(PyFloat_AsDouble(pReward));

		// PyTuple_GetItem returns references that are BORROWED from the tuple
		// above them. They are only valid as long as we hold the parent object
		// reference. We only need to decref the parent object:
		Py_XDECREF(pStepReturn);

		// Making it here means we successfully called the step function.
		return true;
	}

	bool GymSageRunner::reset()
	{
		// Reset returns the observation and agent's action:
		PyObject* pObsAction = PyObject_CallNoArgs(m_resetFunc);
		if (pObsAction == nullptr)
		{
			PyErr_Print();
			return false;
		}

		PyObject* pObs = PyTuple_GetItem(pObsAction, 0);
		if (pObs == nullptr)
		{
			PyErr_Print();
			return false;
		}
		
		PyObject* pAction = PyTuple_GetItem(pObsAction, 1);
		if (pAction == nullptr)
		{
			PyErr_Print();
			return false;
		}

		// Fill the observation:
		m_mostRecentObs.clear();
		for (int i = 0; i < m_observationSize; ++i)
			m_mostRecentObs.push_back(PyFloat_AsDouble(PyList_GetItem(pObs, i)));

		m_mostRecentAgentAction.clear();
		for (int i = 0; i < m_actionSize; ++i)
			m_mostRecentAgentAction.push_back(PyFloat_AsDouble(PyList_GetItem(pAction, i)));

		// PyTuple_GetItem returns references that are BORROWED from the tuple
		// above them. They are only valid as long as we hold the parent object
		// reference. We only need to decref the parent object:
		Py_XDECREF(pObsAction);

		// Reward resets to zero:
		m_currentReward = 0.0;
		m_environmentDone = false;

		return true;
	}

	void GymSageRunner::render()
	{
		PyObject_CallNoArgs(m_renderFunc);
	}

	std::vector<double> GymSageRunner::getRecentObs()
	{
		return m_mostRecentObs;
	}

	std::vector<double> GymSageRunner::getRecentAgentAction()
	{
		return m_mostRecentAgentAction;
	}

	bool GymSageRunner::getEnvironmentDone()
	{
		return m_environmentDone;
	}

	float GymSageRunner::getCurrentReward()
	{
		return m_currentReward;
	}

	void GymSageRunner::getAllStatus(std::vector<double>& recentObs,
		std::vector<double>& recentAction, bool& envDone, float& reward)
	{
		recentObs = m_mostRecentObs;
		recentAction = m_mostRecentAgentAction;
		envDone = m_environmentDone;
		reward = m_currentReward;
	}




} // End namespace Experiment