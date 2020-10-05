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
package edu.iu.dsc.tws.rsched.schedulers.mesos.driver;

//import java.io.File;

import java.util.logging.Logger;

import edu.iu.dsc.tws.api.config.Config;
import edu.iu.dsc.tws.api.config.SchedulerContext;
import edu.iu.dsc.tws.api.driver.IScalerPerCluster;
import edu.iu.dsc.tws.proto.system.job.JobAPI;
import edu.iu.dsc.tws.rsched.schedulers.mesos.MesosController;

//import edu.iu.dsc.tws.rsched.schedulers.mesos.MesosContext;
//import edu.iu.dsc.tws.rsched.utils.ProcessUtils;

public class MesosScaler implements IScalerPerCluster {

  private static final Logger LOG = Logger.getLogger(MesosScaler.class.getName());

  private JobAPI.Job job;
  private Config config;
  private MesosController mesosController;

  // replicas and workersPerPod values for scalable compute resource (scalable statefulSet)
  private String scalableSSName;
  private int replicas;
  private int workersPerPod;
  private int computeResourceIndex;
  private String frameworkId;


  public MesosScaler(Config config, JobAPI.Job job, MesosController mesosController) {
    this.mesosController = mesosController;
    this.job = job;
    this.config = config;

    computeResourceIndex = job.getComputeResourceCount() - 1;

    this.replicas = job.getComputeResource(computeResourceIndex).getInstances();
    this.workersPerPod = job.getComputeResource(computeResourceIndex).getWorkersPerPod();

//    scalableSSName = KubernetesUtils.createWorkersStatefulSetName(
//        job.getJobName(), job.getComputeResourceCount() - 1);
  }

  @Override
  public boolean isScalable() {
    // if there is no scalable compute resource in the job, can not be scalable
    boolean computeResourceScalable =
        job.getComputeResource(job.getComputeResourceCount() - 1).getScalable();
    if (!computeResourceScalable) {
      return false;
    }

    // if it is an OpenMPI job, it is not scalable
    if (SchedulerContext.usingOpenMPI(config)) {
      return false;
    }

    return true;
  }

  /**
   * add new workers to the scalable compute resource
   * @return
   */
  @Override
  public boolean scaleUpWorkers(int instancesToAdd) {

//    if (instancesToAdd % workersPerPod != 0) {
//      LOG.severe("instancesToAdd has to be a multiple of workersPerPod=" + workersPerPod);
//      return false;
//    }
//
//    int podsToAdd = instancesToAdd / workersPerPod;
//
//    boolean scaledUp = mesosController.patchWorkers(scalableSSName, replicas + podsToAdd);
//    if (!scaledUp) {
//      return false;
//    }
//
//    replicas = replicas + podsToAdd;

    return true;
  }

  /**
   * remove workers from the scalable compute resource
   * @param instancesToRemove
   * @return
   */
  @Override
  public boolean scaleDownWorkers(int instancesToRemove, int numberOfWorkers) {

//    if (instancesToRemove % workersPerPod != 0) {
//      LOG.severe("instancesToRemove has to be a multiple of workersPerPod=" + workersPerPod);
//      return false;
//    }
//
//    int podsToRemove = instancesToRemove / workersPerPod;
//
//    if (podsToRemove > replicas) {
//      LOG.severe(String.format("There are %d instances of scalable ComputeResource, "
//          + "and %d instances requested to be removed", replicas, podsToRemove));
//      return false;
//    }
//
//    boolean scaledDown = mesosController.patchWorkers(scalableSSName, replicas - podsToRemove);
//    if (!scaledDown) {
//      return false;
//    }
//
//    // update replicas
//    replicas = replicas - podsToRemove;

//    String frameworkKillCommand = "curl -XPOST http://"
//        + MesosContext.getMesosMasterHost(config) + ":5050/master/teardown -d frameworkId="
//        + frameworkId;
//    System.out.println("kill command:" + frameworkKillCommand);
//
//    ProcessUtils.runSyncProcess(false,
//        frameworkKillCommand.split(" "), new StringBuilder(),
//        new File("."), true);
    return true;
  }

  public void setFrameWorkId(String id) {
    frameworkId = id;
  }
}
