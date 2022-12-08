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

import numpy as np
import pandas as pd


def get_multitask_output(y):
    """Convert [[1,0,0], [0,1,0],] to [[[1,0], [0, 1]], [[0,1], [1, 0]], [[0,1], [0,1]]]

    Args:
        y (array): input array

    Returns:
        _type_: _description_
    """

    y_multitask = list()

    for index in range(y.shape[1]):
        y_multitask.append(list())

    for labels in y:
        for index, label in enumerate(list(labels)):
            if label == 1:
                y_multitask[index].append([1, 0])
            else:
                y_multitask[index].append([0, 1])
    y_multitask = [np.asarray(item) for item in y_multitask]
    return y_multitask


def get_mlb_output(y):
    """Convert  [[[1,0], [0, 1]], [[0,1], [1, 0]], [[0,1], [0,1]]]  to [[1,0,0], [0,1,0],]

    Args:
        y (array): input array

    Returns:
        _type_: _description_
    """

    y_bin = list()
    for index in [item for item in range(y[0].shape[0])]:
        labels = list()
        for label_index in [item for item in range(len(y))]:
            labels.append(y[label_index][index][0])
        y_bin.append(labels)
    return np.asarray(y_bin)


# def get_class_metrics(y, y_pred, labels):

#     y_pred = 1 * y_pred
#     label_metrics_list = list()
#     for index, label in enumerate(labels):
#         slice_org = y[:, index]
#         slice_pred = y_pred[:, index]
#         tp = np.sum(np.logical_and(slice_org, slice_pred))
#         fp = np.sum(np.logical_and(np.logical_not(slice_org), slice_pred))
#         fn = np.sum(np.logical_and(slice_org, np.logical_not(slice_pred)))

#         if tp + fp:
#             precision = tp / (tp + fp)
#         else:
#             precision = 0

#         if tp + fn:
#             recall = tp / (tp + fn)
#         else:
#             recall = 0

#         if precision + recall:
#             f1 = (2 * precision * recall) / (precision + recall)
#         else:
#             f1 = 0

#         label_metrics_list.append(
#             {
#                 "precision": precision,
#                 "recall": recall,
#                 "f1": f1,
#                 "label": label
#             }
#         )

#     return pd.DataFrame(label_metrics_list)
