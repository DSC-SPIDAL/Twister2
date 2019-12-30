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
package edu.iu.dsc.tws.tsched.userdefined;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;
import java.util.logging.Logger;

import edu.iu.dsc.tws.api.compute.exceptions.TaskSchedulerException;
import edu.iu.dsc.tws.api.compute.graph.ComputeGraph;
import edu.iu.dsc.tws.api.compute.graph.Vertex;
import edu.iu.dsc.tws.api.compute.schedule.ITaskScheduler;
import edu.iu.dsc.tws.api.compute.schedule.elements.Resource;
import edu.iu.dsc.tws.api.compute.schedule.elements.TaskInstanceId;
import edu.iu.dsc.tws.api.compute.schedule.elements.TaskInstancePlan;
import edu.iu.dsc.tws.api.compute.schedule.elements.TaskSchedulePlan;
import edu.iu.dsc.tws.api.compute.schedule.elements.Worker;
import edu.iu.dsc.tws.api.compute.schedule.elements.WorkerPlan;
import edu.iu.dsc.tws.api.compute.schedule.elements.WorkerSchedulePlan;
import edu.iu.dsc.tws.api.config.Config;
import edu.iu.dsc.tws.tsched.spi.common.TaskSchedulerContext;
import edu.iu.dsc.tws.tsched.spi.taskschedule.TaskInstanceMapCalculation;
import edu.iu.dsc.tws.tsched.utils.TaskAttributes;

/**
 * This class allocate the task instances into the container in a round robin manner. First, it will
 * allocate the task instances into the logical container values and then it will calculate the
 * required ram, disk, and cpu values for task instances and the containers which is based
 * on the task configuration values and the allocated worker values respectively.
 * <p>
 * {@literal For example, if there are two tasks with parallelism value of 2, 1st task -> instance 0 will
 * go to container 0, 2nd task -> instance 0 will go to container 1, 1st task -> instance 1 will
 * go to container 1, 2nd task -> instance 1 will go to container 1.}
 */
public class UserDefinedTaskScheduler implements ITaskScheduler {

  private static final Logger LOG = Logger.getLogger(UserDefinedTaskScheduler.class.getName());

  //Represents the task instance ram
  private Double instanceRAM;

  //Represents the task instance disk
  private Double instanceDisk;

  //Represents the task instance cpu value
  private Double instanceCPU;

  //Config object
  private Config config;

  private int workerId;

  /**
   * This method initialize the task instance values with the values specified in the task config
   * object.
   */
  @Override
  public void initialize(Config cfg) {
    this.config = cfg;
    this.instanceRAM = TaskSchedulerContext.taskInstanceRam(config);
    this.instanceDisk = TaskSchedulerContext.taskInstanceDisk(config);
    this.instanceCPU = TaskSchedulerContext.taskInstanceCpu(config);
  }

  @Override
  public void initialize(Config cfg, int workerid) {
    this.initialize(cfg);
    this.workerId = workerid;
  }
  /**
   * This is the base method which receives the dataflow taskgraph and the worker plan to allocate
   * the task instances to the appropriate workers with their required ram, disk, and cpu values.
   *
   * @return TaskSchedulePlan
   */
  @Override
  public TaskSchedulePlan schedule(ComputeGraph graph, WorkerPlan workerPlan) {

    int taskSchedulePlanId = 0;

    //Allocate the task instances into the containers/workers
    Set<WorkerSchedulePlan> workerSchedulePlans = new LinkedHashSet<>();

    //To get the vertex set from the taskgraph
    Set<Vertex> taskVertexSet = graph.getTaskVertexSet();

    //Allocate the task instances into the logical containers.
    Map<Integer, List<TaskInstanceId>> userDefinedContainerInstanceMap =
        userDefinedSchedulingAlgorithm(graph, workerPlan.getNumberOfWorkers());

    TaskInstanceMapCalculation instanceMapCalculation = new TaskInstanceMapCalculation(
        this.instanceRAM, this.instanceCPU, this.instanceDisk);

    Map<Integer, Map<TaskInstanceId, Double>> instancesRamMap =
        instanceMapCalculation.getInstancesRamMapInContainer(userDefinedContainerInstanceMap,
            taskVertexSet);

    Map<Integer, Map<TaskInstanceId, Double>> instancesDiskMap =
        instanceMapCalculation.getInstancesDiskMapInContainer(userDefinedContainerInstanceMap,
            taskVertexSet);

    Map<Integer, Map<TaskInstanceId, Double>> instancesCPUMap =
        instanceMapCalculation.getInstancesCPUMapInContainer(userDefinedContainerInstanceMap,
            taskVertexSet);

    for (int containerId : userDefinedContainerInstanceMap.keySet()) {

      double containerRAMValue = TaskSchedulerContext.containerRamPadding(config);
      double containerDiskValue = TaskSchedulerContext.containerDiskPadding(config);
      double containerCpuValue = TaskSchedulerContext.containerCpuPadding(config);

      List<TaskInstanceId> taskTaskInstanceIds = userDefinedContainerInstanceMap.get(containerId);
      Map<TaskInstanceId, TaskInstancePlan> taskInstancePlanMap = new HashMap<>();

      for (TaskInstanceId id : taskTaskInstanceIds) {

        double instanceRAMValue = instancesRamMap.get(containerId).get(id);
        double instanceDiskValue = instancesDiskMap.get(containerId).get(id);
        double instanceCPUValue = instancesCPUMap.get(containerId).get(id);

        Resource instanceResource = new Resource(instanceRAMValue,
            instanceDiskValue, instanceCPUValue);

        taskInstancePlanMap.put(id, new TaskInstancePlan(id.getTaskName(),
            id.getTaskId(), id.getTaskIndex(), instanceResource));

        containerRAMValue += instanceRAMValue;
        containerDiskValue += instanceDiskValue;
        containerCpuValue += instanceDiskValue;
      }

      Worker worker = workerPlan.getWorker(containerId);
      Resource containerResource;

      //Create the container resource value based on the worker plan
      if (worker != null && worker.getCpu() > 0 && worker.getDisk() > 0 && worker.getRam() > 0) {
        containerResource = new Resource((double) worker.getRam(),
            (double) worker.getDisk(), (double) worker.getCpu());
        LOG.fine("Worker (if loop):" + containerId + "\tRam:" + worker.getRam()
            + "\tDisk:" + worker.getDisk() + "\tCpu:" + worker.getCpu());
      } else {
        containerResource = new Resource(containerRAMValue, containerDiskValue, containerCpuValue);
        LOG.fine("Worker (else loop):" + containerId + "\tRam:" + containerRAMValue
            + "\tDisk:" + containerDiskValue + "\tCpu:" + containerCpuValue);
      }

      //Schedule the task instance plan into the task container plan.
      WorkerSchedulePlan taskWorkerSchedulePlan =
          new WorkerSchedulePlan(containerId,
              new HashSet<>(taskInstancePlanMap.values()), containerResource);
      workerSchedulePlans.add(taskWorkerSchedulePlan);
    }
    return new TaskSchedulePlan(taskSchedulePlanId, workerSchedulePlans);
  }

  /**
   * This method retrieves the parallel task map and the total number of task instances for the task
   * vertex set. Then, it will allocate the instances into the number of containers allocated for
   * the task in a round robin fashion.
   *
   * The user could write their own type of allocations into the available workers using their
   * own scheduling algorithm.
   */
  private static Map<Integer, List<TaskInstanceId>> userDefinedSchedulingAlgorithm(
      ComputeGraph graph, int numberOfContainers) {

    Map<Integer, List<TaskInstanceId>> userDefinedAllocation = new LinkedHashMap<>();
    for (int i = 0; i < numberOfContainers; i++) {
      userDefinedAllocation.put(i, new ArrayList<>());
    }

    Set<Vertex> taskVertexSet = new LinkedHashSet<>(graph.getTaskVertexSet());
    TreeSet<Vertex> orderedTaskSet = new TreeSet<>(new VertexComparator());
    orderedTaskSet.addAll(taskVertexSet);

    TaskAttributes taskAttributes = new TaskAttributes();
    int globalTaskIndex = 0;
    for (Vertex vertex : taskVertexSet) {
      int totalTaskInstances;
      if (!graph.getNodeConstraints().isEmpty()) {
        totalTaskInstances = taskAttributes.getTotalNumberOfInstances(vertex,
            graph.getNodeConstraints());
      } else {
        totalTaskInstances = taskAttributes.getTotalNumberOfInstances(vertex);
      }

      if (!graph.getNodeConstraints().isEmpty()) {
        int instancesPerWorker = taskAttributes.getInstancesPerWorker(graph.getGraphConstraints());
        int maxTaskInstancesPerContainer = 0;
        int containerIndex;
        for (int i = 0; i < totalTaskInstances; i++) {
          containerIndex = i % numberOfContainers;
          if (maxTaskInstancesPerContainer < instancesPerWorker) {
            userDefinedAllocation.get(containerIndex).add(
                new TaskInstanceId(vertex.getName(), globalTaskIndex, i));
            ++maxTaskInstancesPerContainer;
          } else {
            throw new TaskSchedulerException("Task Scheduling couldn't be possible for the present"
                + "configuration, please check the number of workers, "
                + "maximum instances per worker");
          }
        }
      } else {
        String task = vertex.getName();
        int containerIndex;
        for (int i = 0; i < totalTaskInstances; i++) {
          containerIndex = i % numberOfContainers;
          userDefinedAllocation.get(containerIndex).add(
              new TaskInstanceId(task, globalTaskIndex, i));
        }
      }
      globalTaskIndex++;
    }
    return userDefinedAllocation;
  }

  private static class VertexComparator implements Comparator<Vertex> {
    @Override
    public int compare(Vertex o1, Vertex o2) {
      return o1.getName().compareTo(o2.getName());
    }
  }
}
