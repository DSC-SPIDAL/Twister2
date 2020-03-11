#  // Licensed under the Apache License, Version 2.0 (the "License");
#  // you may not use this file except in compliance with the License.
#  // You may obtain a copy of the License at
#  //
#  // http://www.apache.org/licenses/LICENSE-2.0
#  //
#  // Unless required by applicable law or agreed to in writing, software
#  // distributed under the License is distributed on an "AS IS" BASIS,
#  // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  // See the License for the specific language governing permissions and
#  // limitations under the License.

# MPI IMPORTS

import os
# TWISTER2 IMPORTS
from twister2.Twister2Environment import Twister2Environment

## FOR LINKING MPI PROCESSES: NOT NEEDED FOR THE BASIC EXAMPLE
#import mpi4py
#mpi4py.rc(initialize=False, finalize=False)
# NUMPY IMPORTS

from twister2deepnet.deepnet.examples.MnistDistributed import MnistDistributed
from twister2deepnet.deepnet.data.UtilPanda import UtilPanda
from twister2deepnet.deepnet.io.FileUtils import FileUtils
from twister2deepnet.deepnet.io.ArrowUtils import ArrowUtils

# MPI.Init()
# device = torch.device("cpu")

###############################################
########## Twister2 Data Processing ##########
##############################################

env = Twister2Environment(resources=[{"cpu": 1, "ram": 512, "instances": 4}])

# TODO: replace this with WorkerConfig

world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
world_rank = int(os.environ['OMPI_COMM_WORLD_RANK'])

__data_folder = '/tmp/twister2deepnet/mnist/'

train_data_save_path = "/tmp/parquet/train/"
test_data_save_path = "/tmp/parquet/test/"

train_data_file = str(world_rank) + ".data"
test_data_file = str(world_rank) + ".data"
train_target_file = str(world_rank) + ".target"
test_target_file = str(world_rank) + ".target"

if world_rank == 0:
    FileUtils.mkdir_branch_with_access(train_data_save_path)
    FileUtils.mkdir_branch_with_access(test_data_save_path)


def save_to_disk(data_set=None, save_path=None, save_file=None):
    # TODO use os.path.join and refactor
    if data_set is None or save_path is None or save_file is None:
        raise Exception("Input Cannot be None")
    elif not os.path.exists(save_path):
        raise Exception("Save Path doesn't exist")
    elif os.path.exists(save_path + save_file):
        pass
    else:
        utilPanda = UtilPanda()
        dataframe = utilPanda.convert_partition_to_pandas(data_set)
        table = ArrowUtils.create_to_table(dataFrame=dataframe)
        ArrowUtils.write_to_table(table=table, save_path=save_path + save_file)


mniste = MnistDistributed(source_dir=__data_folder, parallelism=world_size, world_rank=world_rank)

train_set_data, train_set_target, bsz = mniste.load_train_data()
test_set_data, test_set_target, bsz = mniste.load_test_data()

# Extracting Dataset from the partition
train_dataset = train_set_data.dataset
test_dataset = test_set_data.dataset
train_targetset = train_set_target.dataset
test_targetset = test_set_target.dataset



# Creating Pandas Data Frame and write to disk with Arrow in Parquet Format
## save train data
save_to_disk(data_set=train_dataset, save_path=train_data_save_path, save_file=train_data_file)
## save train labels
save_to_disk(data_set=train_targetset, save_path=train_data_save_path, save_file=train_target_file)
## save test data
save_to_disk(data_set=test_dataset, save_path=test_data_save_path, save_file=test_data_file)
## save test labels
save_to_disk(data_set=test_targetset, save_path=test_data_save_path, save_file=test_target_file)

#print(type(train_dataset), len(train_dataset), world_rank, world_size)
