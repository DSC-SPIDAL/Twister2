//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
package edu.iu.dsc.tws.executor.api;

import java.util.Set;
import java.util.logging.Logger;

import edu.iu.dsc.tws.common.config.Config;
import edu.iu.dsc.tws.comms.core.TaskPlan;
import edu.iu.dsc.tws.comms.op.Communicator;
import edu.iu.dsc.tws.executor.comm.operations.batch.AllReduceBatchOperation;
import edu.iu.dsc.tws.executor.comm.operations.batch.BroadcastBatchOperation;
import edu.iu.dsc.tws.executor.comm.operations.batch.GatherBatchOperation;
import edu.iu.dsc.tws.executor.comm.operations.batch.KeyedReduceBatchOperation;
import edu.iu.dsc.tws.executor.comm.operations.batch.PartitionBatchOperation;
import edu.iu.dsc.tws.executor.comm.operations.batch.ReduceBatchOperation;
import edu.iu.dsc.tws.executor.comm.operations.streaming.AllReduceStreamingOperation;
import edu.iu.dsc.tws.executor.comm.operations.streaming.BroadcastStreamingOperation;
import edu.iu.dsc.tws.executor.comm.operations.streaming.GatherStreamingOperation;
import edu.iu.dsc.tws.executor.comm.operations.streaming.KeyedReduceStreamingOperation;
import edu.iu.dsc.tws.executor.comm.operations.streaming.PartitionStreamingOperation;
import edu.iu.dsc.tws.executor.comm.operations.streaming.ReduceStreamingOperation;
import edu.iu.dsc.tws.executor.core.CommunicationOperationType;
import edu.iu.dsc.tws.task.api.Operations;
import edu.iu.dsc.tws.task.graph.Edge;
import edu.iu.dsc.tws.task.graph.OperationMode;

public class ParallelOperationFactory {
  private static final Logger LOG = Logger.getLogger(ParallelOperationFactory.class.getName());
  private Communicator channel;

  private Config config;

  private TaskPlan taskPlan;

  private EdgeGenerator edgeGenerator;

  public ParallelOperationFactory(Config cfg, Communicator network,
                                  TaskPlan plan, EdgeGenerator e) {
    this.channel = network;
    this.config = cfg;
    this.taskPlan = plan;
    this.edgeGenerator = e;
  }

  /**
   * Building the parallel operation based on the batch or streaming tasks. And also
   * the sub cateogories depends on the communication used for each edge in the task graph.
   * ***/
  public IParallelOperation build(Edge edge, Set<Integer> sources, Set<Integer> dests,
                                  OperationMode operationMode) {

    if (operationMode.equals(OperationMode.BATCH)) {
      //LOG.info("Batch Job Building ...");
      if (!edge.isKeyed()) {
        if (CommunicationOperationType.BATCH_PARTITION.equals(edge.getOperation())) {
          PartitionBatchOperation partitionOp
              = new PartitionBatchOperation(config, channel, taskPlan);
          partitionOp.prepare(sources, dests, edgeGenerator, edge.getDataType(), edge.getName());
          return partitionOp;
        } else if (CommunicationOperationType.BATCH_BROADCAST.equals(edge.getOperation())) {
          BroadcastBatchOperation bcastOp = new BroadcastBatchOperation(config, channel, taskPlan);
          // get the first as the source
          bcastOp.prepare(sources.iterator().next(), dests, edgeGenerator, edge.getDataType(),
              edge.getName());
          return bcastOp;
        } else if (CommunicationOperationType.BATCH_GATHER.equals(edge.getOperation())) {
          System.out.println("Gather Ops");
          GatherBatchOperation gatherOp = new GatherBatchOperation(config, channel, taskPlan);
          gatherOp.prepare(sources, dests.iterator().next(), edgeGenerator, edge.getDataType(),
              edge.getName(), config, taskPlan);
          return gatherOp;
        } else if (CommunicationOperationType.BATCH_REDUCE.equals(edge.getOperation())) {
          ReduceBatchOperation reduceBatchOperation = new ReduceBatchOperation(config, channel,
              taskPlan);
          reduceBatchOperation.prepare(sources, dests.iterator().next(), edgeGenerator,
              edge.getDataType(), edge.getName());
          return reduceBatchOperation;
        } else if (CommunicationOperationType.BATCH_ALLREDUCE.equals(edge.getOperation())) {
          AllReduceBatchOperation allReduceBatchOperation = new AllReduceBatchOperation(config,
              channel, taskPlan);
          allReduceBatchOperation.prepare(sources, dests, edgeGenerator, edge.getDataType(),
              edge.getName());
          return allReduceBatchOperation;
        } else if (CommunicationOperationType.BATCH_KEYED_REDUCE.equals(edge.getOperation())) {
          KeyedReduceBatchOperation keyedReduceBatchOperation
              = new KeyedReduceBatchOperation(config, channel, taskPlan);
          keyedReduceBatchOperation.prepare(sources, dests, edgeGenerator, edge.getDataType(),
              edge.getName());
          return keyedReduceBatchOperation;
        }
      } else {
        if (CommunicationOperationType.BATCH_BROADCAST.equals(edge.getOperation())) {
          BroadcastBatchOperation broadcastOp = new BroadcastBatchOperation(config, channel,
              taskPlan);
          broadcastOp.prepare(sources.iterator().next(), dests, edgeGenerator, edge.getDataType(),
              edge.getName());
          return broadcastOp;
        } else if (CommunicationOperationType.BATCH_REDUCE.equals(edge.getOperation())) {
          ReduceBatchOperation reduceBatchOperation = new ReduceBatchOperation(config, channel,
              taskPlan);
          reduceBatchOperation.prepare(sources, dests.iterator().next(), edgeGenerator,
              edge.getDataType(), edge.getName());
          return reduceBatchOperation;
        } else if (CommunicationOperationType.BATCH_ALLREDUCE.equals(edge.getOperation())) {
          AllReduceBatchOperation allReduceBatchOperation = new AllReduceBatchOperation(config,
              channel, taskPlan);
          allReduceBatchOperation.prepare(sources, dests, edgeGenerator, edge.getDataType(),
              edge.getName());
          return allReduceBatchOperation;
        } else if (CommunicationOperationType.BATCH_KEYED_REDUCE.equals(edge.getOperation())) {
          KeyedReduceBatchOperation keyedReduceBatchOperation
              = new KeyedReduceBatchOperation(config, channel, taskPlan);
          keyedReduceBatchOperation.prepare(sources, dests, edgeGenerator, edge.getDataType(),
              edge.getName());
          return keyedReduceBatchOperation;
        }
      }

    } else if (operationMode.equals(OperationMode.STREAMING)) {
      //LOG.info("Streaming Job Building ...");
      if (!edge.isKeyed()) {
        if (CommunicationOperationType.STREAMING_PARTITION.equals(edge.getOperation())) {
          PartitionStreamingOperation partitionOp = new PartitionStreamingOperation(config, channel,
              taskPlan);
          partitionOp.prepare(sources, dests, edgeGenerator, edge.getDataType(), edge.getName());
          return partitionOp;
        } else if (CommunicationOperationType.STREAMING_BROADCAST.equals(edge.getOperation())) {
          BroadcastStreamingOperation bcastOp = new BroadcastStreamingOperation(config, channel,
              taskPlan);
          // get the first as the source
          bcastOp.prepare(sources.iterator().next(), dests, edgeGenerator, edge.getDataType(),
              edge.getName());
          return bcastOp;
        } else if (CommunicationOperationType.STREAMING_GATHER.equals(edge.getOperation())) {
          GatherStreamingOperation gatherOp = new GatherStreamingOperation(config, channel,
              taskPlan);
          gatherOp.prepare(sources, dests.iterator().next(), edgeGenerator, edge.getDataType(),
              edge.getName(), config, taskPlan);
          return gatherOp;
        } else if (CommunicationOperationType.STREAMING_REDUCE.equals(edge.getOperation())) {
          ReduceStreamingOperation reduceStreamingOperation = new ReduceStreamingOperation(config,
              channel, taskPlan);
          reduceStreamingOperation.prepare(sources, dests.iterator().next(), edgeGenerator,
              edge.getDataType(), edge.getName());
          return reduceStreamingOperation;
        } else if (CommunicationOperationType.STREAMING_ALLREDUCE.equals(edge.getOperation())) {
          AllReduceStreamingOperation allReduceStreamingOperation
              = new AllReduceStreamingOperation(config, channel, taskPlan);
          allReduceStreamingOperation.prepare(sources, dests, edgeGenerator, edge.getDataType(),
              edge.getName());
          return allReduceStreamingOperation;
        } else if (CommunicationOperationType.STREAMING_KEYED_REDUCE.equals(edge.getOperation())) {
          KeyedReduceStreamingOperation keyedReduceStreamingOperation
              = new KeyedReduceStreamingOperation(config, channel, taskPlan);
          keyedReduceStreamingOperation.prepare(sources, dests, edgeGenerator, edge.getDataType(),
              edge.getName());
          return keyedReduceStreamingOperation;
        }
      } else {
        if (CommunicationOperationType.STREAMING_BROADCAST.equals(edge.getOperation())) {
          BroadcastStreamingOperation broadcastOp = new BroadcastStreamingOperation(config,
              channel, taskPlan);
          broadcastOp.prepare(sources.iterator().next(), dests, edgeGenerator, edge.getDataType(),
              edge.getName());
          return broadcastOp;
        } else if (CommunicationOperationType.STREAMING_REDUCE.equals(edge.getOperation())) {
          ReduceStreamingOperation reduceStreamingOperation = new ReduceStreamingOperation(config,
              channel, taskPlan);
          reduceStreamingOperation.prepare(sources, dests.iterator().next(), edgeGenerator,
              edge.getDataType(), edge.getName());
          return reduceStreamingOperation;
        } else if (CommunicationOperationType.STREAMING_ALLREDUCE.equals(edge.getOperation())) {
          AllReduceStreamingOperation allReduceStreamingOperation
              = new AllReduceStreamingOperation(config, channel, taskPlan);
          allReduceStreamingOperation.prepare(sources, dests, edgeGenerator, edge.getDataType(),
              edge.getName());
          return allReduceStreamingOperation;
        } else if (CommunicationOperationType.STREAMING_KEYED_REDUCE.equals(edge.getOperation())) {
          KeyedReduceStreamingOperation keyedReduceStreamingOperation
              = new KeyedReduceStreamingOperation(config, channel, taskPlan);
          keyedReduceStreamingOperation.prepare(sources, dests, edgeGenerator, edge.getDataType(),
              edge.getName());
          return keyedReduceStreamingOperation;
        }
      }
    }

    return null;
  }


  /**
   * Factory Format to Choose the Corresponding Operation
   */
  public IParallelOperation build(Edge edge, Set<Integer> sources, Set<Integer> dests) {
    if (!edge.isKeyed()) {
      if (Operations.PARTITION.equals(edge.getOperation())) {
        PartitionStreamingOperation partitionOp = new PartitionStreamingOperation(config,
            channel, taskPlan);
        partitionOp.prepare(sources, dests, edgeGenerator, edge.getDataType(), edge.getName());
        return partitionOp;
      } else if (Operations.BROADCAST.equals(edge.getOperation())) {
        BroadcastStreamingOperation bcastOp = new BroadcastStreamingOperation(config,
            channel, taskPlan);
        // get the first as the source
        bcastOp.prepare(sources.iterator().next(), dests, edgeGenerator, edge.getDataType(),
            edge.getName());
        return bcastOp;
      } else if (Operations.GATHER.equals(edge.getOperation())) {
        GatherStreamingOperation gatherOp = new GatherStreamingOperation(config, channel, taskPlan);
        gatherOp.prepare(sources, dests.iterator().next(), edgeGenerator, edge.getDataType(),
            edge.getName(), config, taskPlan);
        return gatherOp;
      } else if (Operations.REDUCE.equals(edge.getOperation())) {
        ReduceStreamingOperation reduceStreamingOperation = new ReduceStreamingOperation(config,
            channel, taskPlan);
        reduceStreamingOperation.prepare(sources, dests.iterator().next(), edgeGenerator,
            edge.getDataType(), edge.getName());
        return reduceStreamingOperation;
      } else if (Operations.ALL_REDUCE.equals(edge.getOperation())) {
        AllReduceStreamingOperation allReduceStreamingOperation
            = new AllReduceStreamingOperation(config, channel, taskPlan);
        allReduceStreamingOperation.prepare(sources, dests, edgeGenerator, edge.getDataType(),
            edge.getName());
        return allReduceStreamingOperation;
      } else if (Operations.KEYED_REDUCE.equals(edge.getOperation())) {
        KeyedReduceStreamingOperation keyedReduceStreamingOperation
            = new KeyedReduceStreamingOperation(config, channel, taskPlan);
        keyedReduceStreamingOperation.prepare(sources, dests, edgeGenerator, edge.getDataType(),
            edge.getName());
        return keyedReduceStreamingOperation;
      }
    } else {
      if (Operations.BROADCAST.equals(edge.getOperation())) {
        BroadcastStreamingOperation broadcastOp = new BroadcastStreamingOperation(config,
            channel, taskPlan);
        broadcastOp.prepare(sources.iterator().next(), dests, edgeGenerator, edge.getDataType(),
            edge.getName());
        return broadcastOp;
      } else if (Operations.REDUCE.equals(edge.getOperation())) {
        ReduceStreamingOperation reduceStreamingOperation = new ReduceStreamingOperation(config,
            channel, taskPlan);
        reduceStreamingOperation.prepare(sources, dests.iterator().next(), edgeGenerator,
            edge.getDataType(), edge.getName());
        return reduceStreamingOperation;
      } else if (Operations.ALL_REDUCE.equals(edge.getOperation())) {
        AllReduceStreamingOperation allReduceStreamingOperation
            = new AllReduceStreamingOperation(config, channel, taskPlan);
        allReduceStreamingOperation.prepare(sources, dests, edgeGenerator, edge.getDataType(),
            edge.getName());
        return allReduceStreamingOperation;
      } else if (Operations.KEYED_REDUCE.equals(edge.getOperation())) {
        KeyedReduceStreamingOperation keyedReduceStreamingOperation
            = new KeyedReduceStreamingOperation(config, channel, taskPlan);
        keyedReduceStreamingOperation.prepare(sources, dests, edgeGenerator, edge.getDataType(),
            edge.getName());
        return keyedReduceStreamingOperation;
      }
    }
    return null;
  }
}
