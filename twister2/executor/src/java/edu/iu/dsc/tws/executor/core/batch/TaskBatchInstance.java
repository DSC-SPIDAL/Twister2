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
package edu.iu.dsc.tws.executor.core.batch;


import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.BlockingQueue;

import edu.iu.dsc.tws.api.checkpointing.CheckpointingClient;
import edu.iu.dsc.tws.api.compute.IMessage;
import edu.iu.dsc.tws.api.compute.OutputCollection;
import edu.iu.dsc.tws.api.compute.TaskContext;
import edu.iu.dsc.tws.api.compute.executor.ExecutorContext;
import edu.iu.dsc.tws.api.compute.executor.INodeInstance;
import edu.iu.dsc.tws.api.compute.executor.IParallelOperation;
import edu.iu.dsc.tws.api.compute.executor.ISync;
import edu.iu.dsc.tws.api.compute.graph.OperationMode;
import edu.iu.dsc.tws.api.compute.modifiers.Closable;
import edu.iu.dsc.tws.api.compute.nodes.ICompute;
import edu.iu.dsc.tws.api.compute.nodes.INode;
import edu.iu.dsc.tws.api.compute.schedule.elements.TaskSchedulePlan;
import edu.iu.dsc.tws.api.config.Config;
import edu.iu.dsc.tws.executor.core.DefaultOutputCollection;
import edu.iu.dsc.tws.executor.core.TaskContextImpl;

/**
 * The class represents the instance of the executing task
 */
public class TaskBatchInstance implements INodeInstance, ISync {
  /**
   * The actual task executing
   */
  private ICompute task;

  /**
   * All the inputs will come through a single queue, otherwise we need to look
   * at different queues for messages
   */
  private BlockingQueue<IMessage> inQueue;

  /**
   * Output will go throuh a single queue
   */
  private BlockingQueue<IMessage> outQueue;

  /**
   * The configuration
   */
  private Config config;

  /**
   * The output collection to be used
   */
  private OutputCollection outputCollection;

  /**
   * The globally unique task id
   */
  private int globalTaskId;

  /**
   * Task id
   */
  private int taskId;

  /**
   * Task index that goes from 0 to parallism - 1
   */
  private int taskIndex;

  /**
   * Number of parallel tasks
   */
  private int parallelism;

  /**
   * Name of the task
   */
  private String taskName;

  /**
   * Node configurations
   */
  private Map<String, Object> nodeConfigs;

  /**
   * Parallel operations
   */
  private Map<String, IParallelOperation> outParOps = new HashMap<>();

  /**
   * Inward parallel operations
   */
  private Map<String, IParallelOperation> inParOps = new HashMap<>();

  /**
   * The worker id
   */
  private int workerId;

  /**
   * The instance state
   */
  private InstanceState state = new InstanceState(InstanceState.INIT);

  /**
   * Output edges
   */
  private Map<String, String> outputEdges;

  /**
   * Input edges
   */
  private Map<String, Set<String>> inputEdges;

  /**
   * Task context
   */
  private TaskContext taskContext;

  /**
   * The low watermark for queued messages
   */
  private int lowWaterMark;

  /**
   * The high water mark for messages
   */
  private int highWaterMark;

  /**
   * The task schedule
   */
  private TaskSchedulePlan taskSchedule;

  /**
   * Keep track of syncs received
   */
  private Set<String> syncReceived = new HashSet<>();

  /**
   * Keep an array for iteration
   */
  private IParallelOperation[] intOpArray;

  /**
   * Keep an array out out edges for iteration
   */
  private String[] inEdgeArray;

  /**
   * Keep an array for iteration
   */
  private IParallelOperation[] outOpArray;

  /**
   * Keep an array out out edges for iteration
   */
  private String[] outEdgeArray;

  public TaskBatchInstance(ICompute task, BlockingQueue<IMessage> inQueue,
                           BlockingQueue<IMessage> outQueue, Config config, String tName,
                           int taskId, int globalTaskId, int tIndex, int parallel,
                           int wId, Map<String, Object> cfgs,
                           Map<String, Set<String>> inEdges, Map<String, String> outEdges,
                           TaskSchedulePlan taskSchedule, CheckpointingClient checkpointingClient,
                           String taskGraphName, long tasksVersion) {
    this.task = task;
    this.inQueue = inQueue;
    this.outQueue = outQueue;
    this.config = config;
    this.globalTaskId = globalTaskId;
    this.taskId = taskId;
    this.taskIndex = tIndex;
    this.parallelism = parallel;
    this.taskName = tName;
    this.nodeConfigs = cfgs;
    this.workerId = wId;
    this.inputEdges = inEdges;
    this.outputEdges = outEdges;
    this.lowWaterMark = ExecutorContext.instanceQueueLowWaterMark(config);
    this.highWaterMark = ExecutorContext.instanceQueueHighWaterMark(config);
    this.taskSchedule = taskSchedule;
  }

  public void prepare(Config cfg) {
    outputCollection = new DefaultOutputCollection(outQueue);
    taskContext = new TaskContextImpl(taskIndex, taskId, globalTaskId, taskName, parallelism,
        workerId, outputCollection, nodeConfigs, inputEdges, outputEdges, taskSchedule,
        OperationMode.BATCH);
    task.prepare(cfg, taskContext);

    /// we will use this array for iteration
    this.outOpArray = new IParallelOperation[outParOps.size()];
    int index = 0;
    for (Map.Entry<String, IParallelOperation> e : outParOps.entrySet()) {
      this.outOpArray[index++] = e.getValue();
    }

    this.outEdgeArray = new String[outputEdges.size()];
    index = 0;
    for (String e : outputEdges.keySet()) {
      this.outEdgeArray[index++] = e;
    }

    /// we will use this array for iteration
    this.intOpArray = new IParallelOperation[inParOps.size()];
    index = 0;
    for (Map.Entry<String, IParallelOperation> e : inParOps.entrySet()) {
      this.intOpArray[index++] = e.getValue();
    }

    this.inEdgeArray = new String[inputEdges.size()];
    index = 0;
    for (String e : inputEdges.keySet()) {
      this.inEdgeArray[index++] = e;
    }
  }

  public void registerOutParallelOperation(String edge, IParallelOperation op) {
    outParOps.put(edge, op);
  }

  public void registerInParallelOperation(String edge, IParallelOperation op) {
    inParOps.put(edge, op);
  }

  @Override
  public boolean execute() {
    // we started the executio
    if (state.isSet(InstanceState.INIT) && state.isNotSet(InstanceState.EXECUTION_DONE)) {
      while (!inQueue.isEmpty() && outQueue.size() < lowWaterMark) {
        IMessage m = inQueue.poll();
        task.execute(m);
        state.addState(InstanceState.EXECUTING);
      }

      // for compute we don't have to have the context done as when the inputs finish and execution
      // is done, we are done executing
      // progress in communication
      boolean complete = isComplete(intOpArray);
      // if we no longer needs to progress comm and input is empty
      if (inQueue.isEmpty() && state.isSet(InstanceState.SYNCED) && complete) {
        task.endExecute();
        state.addState(InstanceState.EXECUTION_DONE);
      }
    }

    // now check the output queue
    while (!outQueue.isEmpty()) {
      IMessage message = outQueue.peek();
      if (message != null) {
        String edge = message.edge();

        // invoke the communication operation
        IParallelOperation op = outParOps.get(edge);
        int flags = 0;
        if (op.send(globalTaskId, message, flags)) {
          outQueue.poll();
        } else {
          // no point progressing further
          break;
        }
      }
    }

    // if execution is done and outqueue is emput, we have put everything to communication
    if (state.isSet(InstanceState.EXECUTION_DONE) && outQueue.isEmpty()
        && state.isNotSet(InstanceState.OUT_COMPLETE)) {
      for (IParallelOperation op : outParOps.values()) {
        op.finish(globalTaskId);
      }
      state.addState(InstanceState.OUT_COMPLETE);
    }

    // lets progress the communication
    boolean complete = isComplete(outOpArray);
    // after we have put everything to communication and no progress is required, lets finish
    if (state.isSet(InstanceState.OUT_COMPLETE) && complete) {
      state.addState(InstanceState.SENDING_DONE);
    }
    return !state.isSet(InstanceState.SENDING_DONE);
  }

  public boolean sync(String edge, byte[] value) {
    syncReceived.add(edge);
    if (syncReceived.equals(inParOps.keySet())) {
      state.addState(InstanceState.SYNCED);
      syncReceived.clear();
    }
    return true;
  }

  /**
   * Progress the communication and return weather we need to further progress
   *
   * @return true if further progress is needed
   */
  private boolean isComplete(IParallelOperation[] ops) {
    boolean allDone = true;
    for (int i = 0; i < ops.length; i++) {
      ops[i].progress();
      if (!ops[i].isComplete()) {
        allDone = false;
      }
    }
    return allDone;
  }

  @Override
  public boolean isComplete() {
    boolean complete = true;
    for (int i = 0; i < outOpArray.length; i++) {
      if (!outOpArray[i].isComplete()) {
        complete = false;
      }
    }

    for (int i = 0; i < intOpArray.length; i++) {
      if (!intOpArray[i].isComplete()) {
        complete = false;
      }
    }

    return complete;
  }

  @Override
  public int getId() {
    return globalTaskId;
  }

  @Override
  public int getIndex() {
    return this.taskIndex;
  }

  @Override
  public INode getNode() {
    return task;
  }

  @Override
  public void close() {
    if (task instanceof Closable) {
      ((Closable) task).close();
    }
  }

  @Override
  public void reset() {
    this.taskContext.reset();
    if (task instanceof Closable) {
      ((Closable) task).reset();
    }
    state = new InstanceState(InstanceState.INIT);
  }

  public BlockingQueue<IMessage> getInQueue() {
    return inQueue;
  }
}
