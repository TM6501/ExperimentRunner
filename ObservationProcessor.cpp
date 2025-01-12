#include "pch.h"
#include "ObservationProcessor.h"
#include <iostream>
#include <cassert>

namespace JBrain
{
  ObservationProcessor::ObservationProcessor(CGP::INPUT_PREPROCESSING procType,
    const unsigned int& obsSize,
    const std::vector<std::vector<double> >& obsRanges,
    const std::vector<unsigned int>& obsBucketsPerInput,
    const unsigned int& actionSize,
    const std::vector<std::vector<double> >& actionRanges,
    const std::vector<unsigned int>& actionBucketsPerInput)
    : m_procType(procType),
    m_obsSize(obsSize),
    m_obsRanges(obsRanges),
    m_actSize(actionSize),
    m_actRanges(actionRanges)
  {
    // Default output size is the observation size:
    m_obsOutputSize = obsSize;

    if (procType == CGP::INPUT_PREPROCESSING::BUCKETS)
    {
      if (m_obsRanges.size() != m_obsSize || m_obsRanges.size() != obsBucketsPerInput.size())
      {
        std::cout << "ERROR: Observation ranges do not match observation and bucketsPerInput size." << std::endl;
      }

      if (m_actRanges.size() != m_actSize || m_actRanges.size() != actionBucketsPerInput.size())
      {
        std::cout << "ERROR: Action ranges do not match action and bucketsPerInput size." << std::endl;
      }

      // Fill in the expected output size:
      m_obsOutputSize = 0;
      for (unsigned int idx = 0; idx < m_obsRanges.size(); ++idx)
      {
        m_obsOutputSize += obsBucketsPerInput[idx];
        m_individualObsSizes.push_back(obsBucketsPerInput[idx]);
      }

      m_actOutputSize = 0;
      for (unsigned int idx = 0; idx < m_actRanges.size(); ++idx)
      {
        m_actOutputSize += actionBucketsPerInput[idx];
        m_individualActSizes.push_back(actionBucketsPerInput[idx]);
      }

      // Fill with the bucket maximums:
      for (unsigned int i = 0; i < m_obsSize; ++i)
      {
        m_obsBucketCutoffs.push_back(createSingleInputBucketCutoff(m_obsRanges[i][0], m_obsRanges[i][1], obsBucketsPerInput[i]));
      }

      for (unsigned int i = 0; i < m_actSize; ++i)
      {
        m_actBucketCutoffs.push_back(createSingleInputBucketCutoff(m_actRanges[i][0], m_actRanges[i][1], actionBucketsPerInput[i]));
      }
    }
    else if (procType == CGP::INPUT_PREPROCESSING::NEGATIVE_VALUE_ADD)
      m_obsOutputSize = 2 * obsSize;  // 2 output values per observation value

    else if (procType == CGP::INPUT_PREPROCESSING::UNDEFINED)
      std::cout << "ERROR: Undefined preprocessing type." << std::endl;
  }
  
  std::vector<double> ObservationProcessor::processInput(const std::vector<double>& envObs)
  {
    switch (m_procType)
    {
    case CGP::INPUT_PREPROCESSING::NO_CHANGE:
      return processInput_NoChange(envObs);
      break;

    case CGP::INPUT_PREPROCESSING::NEGATIVE_VALUE_ADD:
      return processInput_NegativeValueAdd(envObs);
      break;

    case CGP::INPUT_PREPROCESSING::BUCKETS:
      return processInput_Bucket(envObs);
      break;

    default:  // Should never hit this.
      std::cout << "ERROR: Undefined preprocessing type (processInput)." << std::endl;
      return std::vector<double> {};        
    }
  }

  std::vector<double> ObservationProcessor::processInput_NoChange(const std::vector<double>& envObs)
  {
    std::vector<double> retVal(envObs);
    return retVal;
  }

  std::vector<double> ObservationProcessor::processInput_Bucket(const std::vector<double>& envObs)
  {
    std::vector<double> retVal;

    // Go through each input value, find the bucket where it belongs, and mark it as 1.
    // All other values are zero. Combine all of these bucket values into a single input
    // vector:
    m_separatedInputs_simplified.clear();
    m_separatedInputs.clear();
    std::vector<double> singleSepVal;
    for (unsigned int i = 0; i < envObs.size(); ++i)
    {
      singleSepVal.clear();
      bool foundSpot = false;
      for (unsigned int j = 0; j < m_obsBucketCutoffs[i].size(); ++j)
      {
        // If we're at the last spot, mark it as a 1.0:
        if (!foundSpot && j == m_obsBucketCutoffs[i].size() - 1)
        {
          retVal.push_back(1.0);
          singleSepVal.push_back(1.0);
          m_separatedInputs_simplified.push_back(j);
        }
        
        // Value is lower than this spot in the bucket cutoff:
        else if (!foundSpot && envObs[i] <= m_obsBucketCutoffs[i][j])
        {
          retVal.push_back(1.0);
          singleSepVal.push_back(1.0);
          m_separatedInputs_simplified.push_back(j);
          foundSpot = true;
        }
        // 0.0 by default:
        else
        {
          retVal.push_back(0.0);
          singleSepVal.push_back(0.0);
        }
      }
      m_separatedInputs.push_back(singleSepVal);
    }

    return retVal;
  }

  // Process an agent's selected output value in the same manner as an input value,
  // just with different buckets.
  std::vector<unsigned int> ObservationProcessor::getSeparatedOutputs_simplified(const std::vector<double>& outputs)
  {
    std::vector<unsigned int> retVal;

    // Go through each input value, find the bucket where it belongs, and mark it as 1.
    for (unsigned int i = 0; i < outputs.size(); ++i)
    {
      bool foundSpot = false;
      for (unsigned int j = 0; j < m_actBucketCutoffs[i].size(); ++j)
      {
        // If we're at the last spot, mark it as a 1.0:
        if (!foundSpot && j == m_actBucketCutoffs[i].size() - 1)
        {
          retVal.push_back(j);
        }
        // Value is lower than this spot in the bucket cutoff:
        else if (!foundSpot && outputs[i] <= m_actBucketCutoffs[i][j])
        {
          retVal.push_back(j);
          foundSpot = true;
        }
      }
    }

    return retVal;
  }

  std::vector<double> ObservationProcessor::processOutput(const std::vector<unsigned int>& brainAct)
  {
    // We don't have the other processing modes defined, yet:
    assert(("Undefined ObservationProcessor processing type" && m_procType == CGP::INPUT_PREPROCESSING::BUCKETS));

    return processOutput_Bucket(brainAct);
  }

  std::vector<double> ObservationProcessor::processOutput_Bucket(const std::vector<unsigned int>& brainAct)
  {
    std::vector<double> retVal;
    
    // For every selected bucket, convert to the appropriate output value:
    double tmpMin, tmpMax;
    unsigned int bucketNum;
    for (unsigned int idx = 0; idx < brainAct.size(); ++idx)
    {
      bucketNum = brainAct[idx];      

      // Special case first: 
      if (bucketNum == 0)
      {
        tmpMin = m_actRanges[idx][0];
        tmpMax = m_actBucketCutoffs[idx][0];
      }
      else
      {
        tmpMin = m_actBucketCutoffs[idx][bucketNum - 1];
        tmpMax = m_actBucketCutoffs[idx][bucketNum];        
      }

      retVal.push_back((tmpMin + tmpMax) / 2.0);
    }
    return retVal;
  }

  std::vector<double> ObservationProcessor::processInput_NegativeValueAdd(const std::vector<double>& envObs)
  {
    std::vector<double> retVal;

    // Go through each input and insert 2 values into the return vector:
    // The abs(val) if negative, or 0.0 otherwise AND
    // The val if positive, or 0.0 otherwise.
    for (unsigned int i = 0; i < envObs.size(); ++i)
    {
      if (envObs[i] < 0.0)
      {
        retVal.push_back(fabs(envObs[i]));
        retVal.push_back(0.0);
      }
      else
      {
        retVal.push_back(0.0);
        retVal.push_back(envObs[i]);
      }
    }

    return retVal;
  }

  std::vector<double> ObservationProcessor::createSingleInputBucketCutoff(
    const double& minVal, const double& maxVal, const unsigned int& buckets)
  {
    double fullRange = maxVal - minVal;
    double stepVal = fullRange / double(buckets);

    std::vector<double> retVal;

    // Create the maximum value for each bucket:
    for (unsigned int i = 0; i < buckets; ++i)
    {
      retVal.push_back(minVal + (stepVal * (i + 1)));
    }

    return retVal;
  }

}  // End namespace JBrain