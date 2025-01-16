import numpy as np
from estimators.common import noisers
from estimators import base


class LaplaceEstimateNoiser(base.EstimateNoiserBase):
  """A noiser that adds Laplace noise to a cardinality estimate."""

  def __init__(self, epsilon, random_state=None):
    """Instantiates a LaplaceEstimateNoiser object.

    Args:
      epsilon:  The differential privacy level.
      random_state:  Optional instance of numpy.random.RandomState that is used
        to seed the random number generator.
    """
    # Note that any cardinality estimator will have sensitivity (delta_f) of 1.
    self._noiser = noisers.LaplaceMechanism(lambda x: x, 1.0, epsilon,
                                            random_state=random_state)

  def __call__(self, cardinality_estimate):
    """Returns a cardinality estimate with Laplace noise."""
    if type(cardinality_estimate) == float:
      return self._noiser(np.array([cardinality_estimate]))[0]
    else:
      return self._noiser(cardinality_estimate)


class GeometricEstimateNoiser(base.EstimateNoiserBase):
  """A noiser that adds discrete Laplace noise to a cardinality estimate."""

  def __init__(self, epsilon, random_state=None):
    """Instantiates a GeometricEstimateNoiser object.

    Args:
      epsilon:  The differential privacy level.
      random_state:  Optional instance of numpy.random.RandomState that is used
        to seed the random number generator.
    """
    # Note that any cardinality estimator will have sensitivity (delta_f) of 1.
    self._noiser = noisers.GeometricMechanism(lambda x: x, 1.0, epsilon,
                                              random_state=random_state)

  def __call__(self, cardinality_estimate):
    """Returns a cardinality estimate with discrete Laplace noise."""
    if type(cardinality_estimate) == float:
      return self._noiser(np.array([cardinality_estimate]))[0]
    else:
      return self._noiser(cardinality_estimate)


class GaussianEstimateNoiser(base.EstimateNoiserBase):
  """A noiser that adds Gaussian noise to a cardinality estimate."""

  def __init__(self, epsilon, delta, num_queries=1, random_state=None):
    """Instantiates a GaussianEstimateNoiser object.

    Args:
      epsilon:  The differential privacy level.
      delta:  The differential privacy level.
      num_queries: The number of queries for which the noiser is used. Note
        that the constructed noiser will be (epsilon, delta)-differentially
        private when answering (no more than) num_queries queries.
      random_state:  Optional instance of numpy.random.RandomState that is used
        to seed the random number generator.
    """
    # Note that any cardinality estimator will have sensitivity (delta_f) of 1.
    self._noiser = noisers.GaussianMechanism(
      lambda x: x, 1.0, epsilon, delta, num_queries=num_queries,
      random_state=random_state)

  def __call__(self, cardinality_estimate):
    """Returns a cardinality estimate with Gaussian noise."""
    if type(cardinality_estimate) == float:
      return self._noiser(np.array([cardinality_estimate]))[0]
    else:
      return self._noiser(cardinality_estimate)


class DiscreteGaussianEstimateNoiser(base.EstimateNoiserBase):
  """A noiser that adds discrete Gaussian noise to a cardinality estimate."""

  def __init__(self, epsilon, delta, num_queries=1, random_state=None):
    """Instantiates a DiscreteGaussianEstimateNoiser object.

    Args:
      epsilon:  The differential privacy level.
      delta:  The differential privacy level.
      num_queries: The number of queries for which the noiser is used. Note
        that the constructed noiser will be (epsilon, delta)-differentially
        private when answering (no more than) num_queries queries.
      random_state:  Optional instance of numpy.random.RandomState that is used
        to seed the random number generator.
    """
    # Note that any cardinality estimator will have sensitivity (delta_f) of 1.
    self._noiser = noisers.DiscreteGaussianMechanism(
      lambda x: x, 1.0, epsilon, delta, num_queries=num_queries,
      random_state=random_state)

  def __call__(self, cardinality_estimate):
    """Returns a cardinality estimate with Gaussian noise."""
    if type(cardinality_estimate) == float:
      return self._noiser(np.array([cardinality_estimate]))[0]
    else:
      return self._noiser(cardinality_estimate)