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

from collections import Counter
from itertools import permutations

import pandas as pd


def get_CPC_IPC_list(cpcs_list, ipcs_list):
    """Combine CPCs and IPCs into a single list.

    Args:
        cpcs_list (list): 
        ipcs_list (list): 

    Returns:
        list: combined list containing both CPCs and IPCs
    """

    codes_list = list()
    for cpcs, ipcs in zip(cpcs_list, ipcs_list):
        codes = set()
        if type(cpcs) is str and len(cpcs) > 1:
            codes.update(eval(cpcs))

        if type(ipcs) is str and len(ipcs) > 1:
            codes.update(eval(ipcs))
        codes_list.append(codes)
    return codes_list
