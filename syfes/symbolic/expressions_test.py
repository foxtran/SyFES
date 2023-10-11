# coding=utf-8
# Copyright 2023 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for symbolic.expressions."""

import copy

from absl.testing import absltest
from absl.testing import parameterized
import jax
import numpy as np
import sympy

from syfes.symbolic import expressions
from syfes.symbolic import instructions

jax.config.update('jax_enable_x64', True)


class ExpressionTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()

    self.num_features = 2
    self.num_shared_parameters = 2
    self.num_variables = 3

    self.features = {
        f'feature_{i}': np.random.rand(5) for i in range(self.num_features)
    }
    self.shared_parameters = {
        f'shared_parameter_{i}': np.random.rand()
        for i in range(self.num_shared_parameters)
    }
    self.bound_parameters = {'gamma_utransform': np.random.rand()}
    self.parameters = {**self.shared_parameters, **self.bound_parameters}
    self.variables = {
        f'variable_{i}': np.zeros(5) for i in range(self.num_variables - 1)
    }
    self.variables.update({'final': np.zeros(5)})

    self.expression = expressions.Expression(
        feature_names=list(self.features.keys()),
        shared_parameter_names=list(self.shared_parameters.keys()),
        variable_names=list(self.variables.keys()),
        instruction_list=[
            instructions.MultiplicationInstruction(
                'variable_0', 'feature_0', 'shared_parameter_0'),
            instructions.AdditionInstruction(
                'variable_1', 'feature_1', 'shared_parameter_1'),
            instructions.AdditionInstruction(
                'variable_1', 'variable_1', 'variable_0'),
            instructions.Power2Instruction('final', 'variable_1'),
            instructions.UTransformInstruction('final', 'final')
        ])

  def test_constructor(self):
    self.assertEqual(self.expression.num_features, self.num_features)
    self.assertEqual(self.expression.num_parameters,
                     self.num_shared_parameters + 1)  # 1 from UTransform
    self.assertEqual(self.expression.num_variables, self.num_variables)

  def test_constructor_without_expression_in_variable_names(self):
    with self.assertRaisesRegex(
        ValueError, '"final" not found in variable_names.'):
      expressions.Expression(
          feature_names=[],
          shared_parameter_names=[],
          variable_names=[],
          instruction_list=[])

  def test_constructor_with_repeated_name(self):
    with self.assertRaisesRegex(ValueError, 'Repeated names found in input.'):
      expressions.Expression(
          feature_names=['var'],
          shared_parameter_names=['var'],
          variable_names=['final'],
          instruction_list=[])

  def test_constructor_with_wrong_instruction_type(self):
    with self.assertRaisesRegex(
        TypeError, r"1 is of type <class 'int'>, not an "
                   'instance of instructions.Instruction'):
      expressions.Expression(
          feature_names=list(self.features.keys()),
          shared_parameter_names=list(self.shared_parameters.keys()),
          variable_names=list(self.variables.keys()),
          instruction_list=[1])

  @parameterized.parameters(
      (instructions.Power2Instruction('variable_0', 'var'),
       (r'Instruction variable_0 = var \*\* 2 contains invalid input argument '
        'var')),
      (instructions.AdditionInstruction('variable_0', 'shared_parameter_1',
                                        'gamma_utransform'),
       (r'Instruction variable_0 = shared_parameter_1 \+ gamma_utransform '
        'contains invalid input argument gamma_utransform')),
  )
  def test_constructor_with_invalid_input(self, instruction, error_message):
    with self.assertRaisesRegex(ValueError, error_message):
      expressions.Expression(
          feature_names=list(self.features.keys()),
          shared_parameter_names=list(self.shared_parameters.keys()),
          variable_names=list(self.variables.keys()),
          instruction_list=[instruction])

  @parameterized.parameters(
      (instructions.Power2Instruction('feature_0', 'shared_parameter_0'),
       (r'Instruction feature_0 = shared_parameter_0 \*\* 2 contains '
        'invalid output argument feature_0')),
      (instructions.AdditionInstruction(
          'feature_1', 'shared_parameter_1', 'variable_1'),
       (r'Instruction feature_1 = shared_parameter_1 \+ variable_1 contains '
        'invalid output argument feature_1')
       ),
      (instructions.Power4Instruction(
          'bound_parameter_1', 'shared_parameter_1'),
       (r'Instruction bound_parameter_1 = shared_parameter_1 \*\* 4 contains '
        'invalid output argument bound_parameter_1')
       ),
  )
  def test_constructor_with_invalid_output(self, instruction, error_message):
    with self.assertRaisesRegex(ValueError, error_message):
      expressions.Expression(
          feature_names=list(self.features.keys()),
          shared_parameter_names=list(self.shared_parameters.keys()),
          variable_names=list(self.variables.keys()),
          instruction_list=[instruction])

  @parameterized.parameters(False, True)
  def test_eval(self, use_jax):
    tmp = (
        (self.features['feature_0'] * self.parameters['shared_parameter_0']) +
        (self.features['feature_1'] + self.parameters['shared_parameter_1']))
    tmp = self.parameters['gamma_utransform'] * tmp ** 2
    expected_f = tmp / (1. + tmp)

    f = self.expression.eval(
        self.features, self.parameters, use_jax=use_jax)

    np.testing.assert_allclose(f, expected_f)


  def test_convert_expression_to_and_from_dict(self):
    self.assertEqual(
        self.expression,
        expressions.Expression.from_dict(
            self.expression.to_dict()))

  @parameterized.parameters(
      expressions.f_empty,
      expressions.f_empty,
  )
  def test_make_isomorphic_copy(self, expression):
    features = {
        feature_name: np.random.rand(5)
        for feature_name in expression.feature_names
    }
    shared_parameters = {
        parameter_name: np.random.rand()
        for parameter_name in expression.shared_parameter_names
    }
    renamed_shared_parameters = {
        (expression._isomorphic_copy_shared_parameter_prefix
         + str(index)): value
        for index, value in enumerate(shared_parameters.values())
    }
    bound_parameters = {
        parameter_name: np.random.rand()
        for parameter_name in expression.bound_parameter_names
    }

    expression_copy = expression.make_isomorphic_copy()

    np.testing.assert_allclose(
        expression.eval(
            features=features, parameters={
                **shared_parameters, **bound_parameters}),
        expression_copy.eval(
            features=features, parameters={
                **renamed_shared_parameters, **bound_parameters})
        )


if __name__ == '__main__':
  absltest.main()
