# Copyright (c) 2021 Robert Bosch GmbH
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import tensorflow as tf


class SingleTaskLayer(tf.keras.layers.Layer):
    """Class representing the single task for the TMM/THMM model defined in TODO: add paper inforamtion
    """

    def __init__(self, config, output_dim_head=2, label_activation="softmax"):
        """Constructor for initializing a class object

        Args:
            config (dict): configuration for the model
            output_dim_head (int, optional): the dimension for the output of a class. Defaults to 2.
            label_activation (str, optional): specifies whether the label activation is sigmoid or softmax. Defaults to "softmax".
        """
        super(SingleTaskLayer, self).__init__()
        self.config = config
        self.hidden_layer_1 = tf.keras.layers.Dense(
            self.config["dense_layer_size"], activation='relu')
        self.drop_hidden_layer_1 = tf.keras.layers.Dropout(
            self.config["dropout_rate"])
        self.hidden_layer_2 = tf.keras.layers.Dense(
            self.config["dense_layer_size"], activation='relu')
        self.drop_hidden_layer_2 = tf.keras.layers.Dropout(
            self.config["dropout_rate"])
        self.label_activation = tf.keras.layers.Dense(
            output_dim_head, activation=label_activation)

    def call(self, inputs, **kwargs):
        x = self.drop_hidden_layer_1(self.hidden_layer_1(inputs))
        x = self.drop_hidden_layer_2(self.hidden_layer_2(x))
        return self.label_activation(x), x
