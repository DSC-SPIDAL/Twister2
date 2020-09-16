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
package edu.iu.dsc.tws.examples.internal.jobmaster;

import java.util.LinkedList;
import java.util.List;
import java.util.logging.Logger;

import edu.iu.dsc.tws.api.Twister2Job;
import edu.iu.dsc.tws.examples.basic.HelloWorld;
import edu.iu.dsc.tws.master.dashclient.DashboardClient;
import edu.iu.dsc.tws.proto.jobmaster.JobMasterAPI;
import edu.iu.dsc.tws.proto.system.job.JobAPI;
import edu.iu.dsc.tws.proto.utils.NodeInfoUtils;
import edu.iu.dsc.tws.proto.utils.WorkerInfoUtils;

public final class DashboardClientExample {
  private static final Logger LOG = Logger.getLogger(DashboardClientExample.class.getName());

  private DashboardClientExample() {
  }

  public static void main(String[] args) {

    if (args.length < 2) {
      printUsage();
      return;
    }

    String dashAddress = args[0];
    String jobID = args[1] + (int) (Math.random() * 100000);
    LOG.info("jobID: " + jobID);
    DashboardClient dashClient = new DashboardClient(dashAddress, jobID, 2);

    // if number of args is 3, kill the job
    if (args.length == 3) {
      dashClient.jobStateChange(JobAPI.JobState.KILLED);
      return;
    }

    // job has one type of ComputeResource
    int computeResourceInstances = 3;
    boolean scalable = true;

    JobAPI.Job job = Twister2Job.newBuilder()
        .setJobName("job-1")
        .setWorkerClass(HelloWorld.class)
        .addComputeResource(2, 1024, computeResourceInstances, scalable)
        .build()
        .serialize();

    // all workers on the same node
    // just to make things simple
    JobMasterAPI.NodeInfo nodeInfo = NodeInfoUtils.createNodeInfo("123.123.123", "rack-0", "dc-0");

//    DashboardClient dashClient = new DashboardClient("http://localhost:8080", "job-1");

    testRegisterJob(dashClient, job, nodeInfo);
    testRegisterWorker(dashClient, 0, job.getComputeResource(0), nodeInfo);
    testRegisterWorker(dashClient, 1, job.getComputeResource(0), nodeInfo);
//    testRegisterWorker(dashClient, 2, job.getComputeResource(0), nodeInfo);

    // test scaling up
    testScalingUp(dashClient, job);
//    testRegisterWorker(dashClient, 3, job.getComputeResource(0), nodeInfo);
//    testRegisterWorker(dashClient, 4, job.getComputeResource(0), nodeInfo);

    // test scaling down
//    testScalingDown(dashClient, job);

    // test state change
    dashClient.workerStateChange(0, JobMasterAPI.WorkerState.COMPLETED);
    dashClient.jobStateChange(JobAPI.JobState.STARTED);

    // job killed
    dashClient.jobStateChange(JobAPI.JobState.KILLED);

    dashClient.close();
  }

  public static JobAPI.Job testRegisterJob(DashboardClient dashClient,
                                           JobAPI.Job job,
                                           JobMasterAPI.NodeInfo nodeInfo) {
    dashClient.registerJob(job, nodeInfo);
    return job;
  }

  public static void testRegisterWorker(DashboardClient dashClient,
                                        int workerID,
                                        JobAPI.ComputeResource computeResource,
                                        JobMasterAPI.NodeInfo nodeInfo) {

    JobMasterAPI.WorkerInfo workerInfo =
        WorkerInfoUtils.createWorkerInfo(workerID, "123.456.789", 9009, nodeInfo, computeResource);

    JobMasterAPI.WorkerState initialState = JobMasterAPI.WorkerState.STARTED;
    dashClient.registerWorker(workerInfo, initialState);
  }

  public static boolean testScalingUp(DashboardClient dashClient, JobAPI.Job job) {

    int addedWorkers = 2;
    int updatedNumberOfWorkers = job.getNumberOfWorkers() + addedWorkers;
    LOG.info("change: " + addedWorkers + " updatedNumberOfWorkers: " + updatedNumberOfWorkers);

    List<Integer> workerList = new LinkedList<>();
    return dashClient.scaledWorkers(
        addedWorkers, updatedNumberOfWorkers, workerList);
  }

  public static boolean testScalingDown(DashboardClient dashClient, JobAPI.Job job) {

    int removedWorkers = 2;
    int change = 0 - removedWorkers;
    int updatedNumberOfWorkers = job.getNumberOfWorkers() - removedWorkers;
    LOG.info("change: " + change + " updatedNumberOfWorkers: " + updatedNumberOfWorkers);

    List<Integer> killedWorkers = new LinkedList<>();
    killedWorkers.add(job.getNumberOfWorkers() - 1);
    killedWorkers.add(job.getNumberOfWorkers() - 2);
    return dashClient.scaledWorkers(change, updatedNumberOfWorkers, killedWorkers);
  }

  public static void printUsage() {
    LOG.info("Usage: java edu.iu.dsc.tws.examples.internal.jobmaster.DashboardClientExample "
        + "dashAddress jobID"
        + "\n sends job KILLED message to Dashboard for this job.");
  }
}
