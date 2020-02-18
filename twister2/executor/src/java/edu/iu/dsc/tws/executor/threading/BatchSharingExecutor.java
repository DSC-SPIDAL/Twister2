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
package edu.iu.dsc.tws.executor.threading;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.logging.Logger;

import edu.iu.dsc.tws.api.comms.channel.TWSChannel;
import edu.iu.dsc.tws.api.compute.executor.ExecutionPlan;
import edu.iu.dsc.tws.api.compute.executor.ExecutionState;
import edu.iu.dsc.tws.api.compute.executor.IExecution;
import edu.iu.dsc.tws.api.compute.executor.IExecutionHook;
import edu.iu.dsc.tws.api.compute.executor.INodeInstance;
import edu.iu.dsc.tws.api.compute.executor.IParallelOperation;
import edu.iu.dsc.tws.api.config.Config;

public class BatchSharingExecutor extends ThreadSharingExecutor {
  private static final Logger LOG = Logger.getLogger(BatchSharingExecutor.class.getName());

  // keep track of finished executions
  //private Map<Integer, Boolean> finishedInstances = new ConcurrentHashMap<>();
  private AtomicInteger finishedInstances = new AtomicInteger(0);

  // worker id
  private int workerId;

  // not stopped
  protected boolean notStopped = true;

  // clean up is called
  private boolean cleanUpCalled = false;

  /**
   * Wait for threads to finsih
   */
  private CountDownLatch doneSignal;

  public BatchSharingExecutor(Config cfg, int workerId, TWSChannel channel, ExecutionPlan plan,
                              IExecutionHook hook) {
    super(cfg, channel, plan, hook);
    this.workerId = workerId;
  }

  /**
   * Execution Method for Batch Tasks
   */
  public boolean runExecution() {
    Map<Integer, INodeInstance> nodes = executionPlan.getNodes();

    if (nodes.size() == 0) {
      LOG.warning(String.format("Worker %d has zero assigned tasks, you may "
          + "have more workers than tasks", workerId));
      return true;
    }

    // if this is a previously executed plan we have to reset the nodes
    if (executionPlan.getExecutionState() == ExecutionState.EXECUTED) {
      resetNodes(executionPlan.getNodes(), executionPlan.getParallelOperations());
    }

    scheduleExecution(nodes);

    // we progress until all the channel finish
    while (finishedInstances.get() != nodes.size()) {
      channel.progress();
    }

    cleanUp(nodes);
    return true;
  }

  @Override
  public boolean execute(boolean close) {
    boolean e = execute();
    if (close) {
      closeExecution();
    }
    return e;
  }

  public boolean isNotStopped() {
    return notStopped;
  }

  private void scheduleExecution(Map<Integer, INodeInstance> nodes) {
    // initialize finished
    // initFinishedInstances();
    List<INodeInstance> tasks = new ArrayList<>(nodes.values());

    // prepare the tasks
    for (INodeInstance node : tasks) {
      node.prepare(config);
    }

    boolean bindTaskToThread = numThreads >= tasks.size();

    if (bindTaskToThread) {
      doneSignal = new CountDownLatch(tasks.size());
      for (INodeInstance task : tasks) {
        threads.execute(new BatchWorker(task));
      }
    } else {

      final AtomicBoolean[] taskStatus = new AtomicBoolean[tasks.size()];
      for (int i = 0; i < tasks.size(); i++) {
        taskStatus[i] = new AtomicBoolean(false);
      }
      doneSignal = new CountDownLatch(numThreads);
      for (int i = 0; i < numThreads; i++) {
        threads.execute(new BatchWorker(tasks, taskStatus));
      }
    }
  }

  private void cleanUp(Map<Integer, INodeInstance> nodes) {
    // lets wait for thread to finish
    try {
      doneSignal.await();
    } catch (InterruptedException e) {
      throw new RuntimeException("Interrupted", e);
    }

    executionPlan.setExecutionState(ExecutionState.EXECUTED);

    // clear the finished instances
    finishedInstances.set(0);
    cleanUpCalled = true;
    // execute the hook
    executionHook.afterExecution();
  }

  @Override
  public IExecution runIExecution() {
    Map<Integer, INodeInstance> nodes = executionPlan.getNodes();

    if (nodes.size() == 0) {
      LOG.warning(String.format("Worker %d has zero assigned tasks, you may "
          + "have more workers than tasks", workerId));
      return new NullExecutor();
    }

    // if this is a previously executed plan we have to reset the nodes
    if (executionPlan.getExecutionState() == ExecutionState.EXECUTED) {
      resetNodes(executionPlan.getNodes(), executionPlan.getParallelOperations());
    }

    scheduleExecution(nodes);
    return new BatchExecution(executionPlan, nodes);
  }

  @Override
  public boolean closeExecution() {
    Map<Integer, INodeInstance> nodes = executionPlan.getNodes();

    if (nodes.size() == 0) {
      LOG.warning(String.format("Worker %d has zero assigned tasks, you may "
          + "have more workers than tasks", workerId));
      return true;
    }

    scheduleWaitFor(nodes);

    // we progress until all the channel finish
    while (isNotStopped() && finishedInstances.get() != nodes.size()) {
      channel.progress();
    }

    close(executionPlan, nodes);
    return true;
  }

  private void scheduleWaitFor(Map<Integer, INodeInstance> nodes) {
    BlockingQueue<INodeInstance> tasks;

    tasks = new ArrayBlockingQueue<>(nodes.size() * 2);
    tasks.addAll(nodes.values());

    int curTaskSize = tasks.size();
    CommunicationWorker[] workers = new CommunicationWorker[curTaskSize];

    doneSignal = new CountDownLatch(curTaskSize);
    for (int i = 0; i < curTaskSize; i++) {
      workers[i] = new CommunicationWorker(tasks);
      threads.execute(workers[i]);
    }
  }

  private void resetNodes(Map<Integer, INodeInstance> nodes, List<IParallelOperation> ops) {
    // clean up the instances
    for (INodeInstance node : nodes.values()) {
      node.reset();
    }

    // lets close the operations
    for (IParallelOperation op : ops) {
      op.reset();
    }
  }

  private void close(ExecutionPlan executionPlan, Map<Integer, INodeInstance> nodes) {
    // lets wait for thread to finish
    try {
      doneSignal.await();
    } catch (InterruptedException e) {
      throw new RuntimeException("Interrupted", e);
    }

    List<IParallelOperation> ops = executionPlan.getParallelOperations();
    resetNodes(nodes, ops);

    // clean up the instances
    for (INodeInstance node : nodes.values()) {
      node.close();
    }

    // lets close the operations
    for (IParallelOperation op : ops) {
      op.close();
    }
    executionHook.onClose(this);
    // clear the finished instances
    finishedInstances.set(0);
    cleanUpCalled = true;
  }

  protected class CommunicationWorker implements Runnable {
    private BlockingQueue<INodeInstance> tasks;

    public CommunicationWorker(BlockingQueue<INodeInstance> tasks) {
      this.tasks = tasks;
    }

    @Override
    public void run() {
      while (isNotStopped()) {
//        try {
        INodeInstance nodeInstance = tasks.poll();
        if (nodeInstance != null) {
          boolean complete = nodeInstance.isComplete();
          if (complete) {
            finishedInstances.incrementAndGet(); //(nodeInstance.getId(), true);
          } else {
            // we need to further execute this task
            tasks.offer(nodeInstance);
          }
        } else {
          break;
        }
//        } catch (Throwable t) {
//          LOG.log(Level.SEVERE, String.format("%d Error in executor", workerId), t);
//          throw new RuntimeException("Error occurred in execution of task", t);
//        }
      }
      doneSignal.countDown();
    }
  }

  protected class BatchWorker implements Runnable {

    //round robin mode
    private List<INodeInstance> tasks;
    private AtomicBoolean[] ignoreIndex;
    private int lastIndex;

    //dedicated mode
    private boolean bindTaskToThread;
    private INodeInstance task;

    public BatchWorker(List<INodeInstance> tasks, AtomicBoolean[] ignoreIndex) {
      this.tasks = tasks;
      this.ignoreIndex = ignoreIndex;
    }

    public BatchWorker(INodeInstance task) {
      this.bindTaskToThread = true;
      this.task = task;
    }

    private int getNext() {
      if (this.lastIndex == tasks.size()) {
        this.lastIndex = 0;
      }

      if (ignoreIndex[this.lastIndex].compareAndSet(false, true)) {
        return this.lastIndex++;
      }
      this.lastIndex++;
      return -1;
    }

    @Override
    public void run() {
      if (this.bindTaskToThread) {
        while (isNotStopped()) {
          boolean needsFurther = this.task.execute();
          if (!needsFurther) {
            finishedInstances.incrementAndGet(); //(task.getId(), true);
            break;
          }
        }
      } else {
        while (isNotStopped() && finishedInstances.get() != tasks.size()) {
//          try {
          int nodeInstanceIndex = this.getNext();
          if (nodeInstanceIndex != -1) {
            INodeInstance nodeInstance = this.tasks.get(nodeInstanceIndex);
            boolean needsFurther = nodeInstance.execute();
            if (!needsFurther) {
              finishedInstances.incrementAndGet(); //(nodeInstance.getId(), true);
            } else {
              //need further execution
              this.ignoreIndex[nodeInstanceIndex].set(false);
            }
          }
//          } catch (Throwable t) {
//            LOG.log(Level.SEVERE, String.format("%d Error in executor", workerId), t);
//            throw new RuntimeException("Error occurred in execution of task", t);
//          }
        }
      }
      doneSignal.countDown();
    }
  }

  private class BatchExecution implements IExecution {
    private Map<Integer, INodeInstance> nodeMap;

    private ExecutionPlan executionPlan;

    private BlockingQueue<INodeInstance> tasks;

    private boolean taskExecution = true;

    BatchExecution(ExecutionPlan executionPlan, Map<Integer, INodeInstance> nodeMap) {
      this.nodeMap = nodeMap;
      this.executionPlan = executionPlan;

      tasks = new ArrayBlockingQueue<>(nodeMap.size() * 2);
      tasks.addAll(nodeMap.values());
    }

    @Override
    public boolean waitForCompletion() {
      // we progress until all the channel finish
      while (isNotStopped() && finishedInstances.get() != nodeMap.size()) {
        channel.progress();
      }
      // we are going to set to executed
      executionPlan.setExecutionState(ExecutionState.EXECUTED);

      cleanUp(nodeMap);

      // now wait for it
      closeExecution();
      return true;
    }

    @Override
    public boolean progress() {
      if (taskExecution) {
        // we progress until all the channel finish
        if (finishedInstances.get() != nodeMap.size()) {
          channel.progress();
          return true;
        }
        // we are going to set to executed
        executionPlan.setExecutionState(ExecutionState.EXECUTED);
        // clean up
        cleanUp(nodeMap);
        cleanUpCalled = false;
        // if we finish, lets schedule
        scheduleWaitFor(nodeMap);
        taskExecution = false;
      }

      // we progress until all the channel finish
      if (isNotStopped() && finishedInstances.get() != nodeMap.size()) {
        channel.progress();
        return true;
      }

      return false;
    }

    public void close() {
      if (isNotStopped()) {
        throw new RuntimeException("We need to stop the execution before close");
      }

      if (!cleanUpCalled) {
        BatchSharingExecutor.this.close(executionPlan, nodeMap);
        executionHook.onClose(BatchSharingExecutor.this);
        cleanUpCalled = true;
      } else {
        throw new RuntimeException("Close is called on a already closed execution");
      }
    }

    @Override
    public void stop() {
      notStopped = false;
    }
  }
}
