#include "pch.h"
#include "ObservationProcessor.h"
#include <iostream>

namespace JBrain
{
  ObservationProcessor::ObservationProcessor(CGP::INPUT_PREPROCESSING procType, const unsigned int& obsSize,
    const std::vector<std::vector<double> >& ranges, const unsigned int& bucketsPerInput)
    : m_procType(procType),
    m_obsSize(obsSize),
    m_ranges(ranges)
  {
    // Default output size is the observation size:
    m_obsOutputSize = obsSize;

    if (procType == CGP::INPUT_PREPROCESSING::BUCKETS)
    {
      if (m_ranges.size() != m_obsSize)
      {
        std::cout << "ERROR: Ranges does not match observation size." << std::endl;
      }

      // Fill in the expected output size:
      m_obsOutputSize = obsSize * bucketsPerInput;

      // Fill with the bucket maximums:
      for (unsigned int i = 0; i < m_obsSize; ++i)
      {
        m_bucketCutoffs.push_back(createSingleInputBucketCutoff(m_ranges[i][0], m_ranges[i][1], bucketsPerInput));
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
    for (unsigned int i = 0; i < envObs.size(); ++i)
    {
      bool foundSpot = false;
      for (unsigned int j = 0; j < m_bucketCutoffs[i].size(); ++j)
      {
        // If we're at the last spot, mark it as a 1.0:
        if (!foundSpot && j == m_bucketCutoffs[i].size() - 1)
          retVal.push_back(1.0);
        
        // Value is lower than this spot in the bucket cutoff:
        else if (!foundSpot && envObs[i] <= m_bucketCutoffs[i][j])
        {
          retVal.push_back(1.0);
          foundSpot = true;
        }
        // 0.0 by default:
        else
          retVal.push_back(0.0);
      }
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