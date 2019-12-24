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
package edu.iu.dsc.tws.rsched.schedulers.k8s.worker;

import java.io.IOException;
import java.net.InetAddress;
import java.net.UnknownHostException;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.logging.Level;
import java.util.logging.Logger;

import com.google.gson.reflect.TypeToken;

import edu.iu.dsc.tws.api.config.Config;
import edu.iu.dsc.tws.api.exceptions.TimeoutException;
import edu.iu.dsc.tws.api.resource.ControllerContext;
import edu.iu.dsc.tws.api.resource.IWorkerController;
import edu.iu.dsc.tws.api.scheduler.SchedulerContext;
import edu.iu.dsc.tws.proto.jobmaster.JobMasterAPI;
import edu.iu.dsc.tws.proto.utils.WorkerInfoUtils;
import edu.iu.dsc.tws.rsched.schedulers.k8s.KubernetesContext;
import edu.iu.dsc.tws.rsched.schedulers.k8s.KubernetesUtils;

import io.kubernetes.client.ApiClient;
import io.kubernetes.client.ApiException;
import io.kubernetes.client.Configuration;
import io.kubernetes.client.apis.CoreV1Api;
import io.kubernetes.client.models.V1Pod;
import io.kubernetes.client.models.V1PodList;
import io.kubernetes.client.util.Watch;

public class K8sWorkerController implements IWorkerController {
  private static final Logger LOG = Logger.getLogger(K8sWorkerController.class.getName());

  private Config config;
  private String jobID;
  private int numberOfPods;
  private int numberOfWorkers;
  private int workersPerPod;
  private static CoreV1Api coreApi;
  private static ApiClient apiClient;
  private ArrayList<JobMasterAPI.WorkerInfo> workerList;
  private JobMasterAPI.WorkerInfo thisWorker;

  public K8sWorkerController(Config config, String podName, String podIpStr, String containerName,
                             String jobID, int workersPerPod) {
    this.config = config;
    numberOfWorkers = SchedulerContext.workerInstances(config);
    this.workersPerPod = workersPerPod;
    numberOfPods = numberOfWorkers / workersPerPod;
    workerList = new ArrayList<JobMasterAPI.WorkerInfo>();
    this.jobID = jobID;

    int containerIndex = KubernetesUtils.indexFromName(containerName);
    int workerID = calculateWorkerID(podName, containerIndex);
    int basePort = KubernetesContext.workerBasePort(config);
    InetAddress podIP = convertStringToIP(podIpStr);
    thisWorker =
        WorkerInfoUtils.createWorkerInfo(workerID, podIpStr, basePort + containerIndex, null);

    createApiInstances();
  }

  public static void createApiInstances() {

    try {
      apiClient = io.kubernetes.client.util.Config.defaultClient();
      apiClient.getHttpClient().setReadTimeout(0, TimeUnit.MILLISECONDS);
    } catch (IOException e) {
      LOG.log(Level.SEVERE, "Exception when creating ApiClient: ", e);
      throw new RuntimeException(e);
    }
    Configuration.setDefaultApiClient(apiClient);

    coreApi = new CoreV1Api(apiClient);
  }

  public static CoreV1Api getCoreApi() {
    if (coreApi == null) {
      createApiInstances();
    }

    return coreApi;
  }

  /**
   * return WorkerNetworkInfo object for this worker
   */
  @Override
  public JobMasterAPI.WorkerInfo getWorkerInfo() {
    return thisWorker;
  }

  /**
   * return the WorkerNetworkInfo object for the given id
   * @return
   */
  public JobMasterAPI.WorkerInfo getWorkerInfoForID(int id) {
    for (JobMasterAPI.WorkerInfo info: workerList) {
      if (info.getWorkerID() == id) {
        return info;
      }
    }

    return null;
  }

  /**
   * return the total number of workers in this job,
   * not the currently active ones
   * @return
   */
  @Override
  public int getNumberOfWorkers() {
    return numberOfWorkers;
  }

  /**
   * return all workers in the job
   * @return
   */
  @Override
  public ArrayList<JobMasterAPI.WorkerInfo> getJoinedWorkers() {
    return workerList;
  }

  /**
   * build worker list and if the list is incomplete,
   * resend the query and try until all received
   */
  public boolean buildWorkerListWaitForAll(long timeLimit) {

    long startTime = System.currentTimeMillis();
    long sleepInterval = 300;

    // when waiting, it will print log message at least after this much time
    long logMessageInterval = 1000;
    //this count is restarted after each log message
    long waitTimeCountForLog = 0;

    while (true) {
      buildWorkerList();
      if (numberOfWorkers == workerList.size()) {
        LOG.info("Received data about all pods. ");
        printWorkers(workerList);
        return true;
      } else if (waitTimeCountForLog >= logMessageInterval) {
        LOG.info("Data is not received for some pods. Number of received workers: "
            + workerList.size() + ". Will try again. Waiting " + logMessageInterval + "ms");
        waitTimeCountForLog = 0;
      }

      long duration = System.currentTimeMillis() - startTime;
      if (duration > timeLimit) {
        LOG.log(Level.SEVERE, "Time limit has been reached when trying to build worker list. "
            + "Given Time limit: " + timeLimit + "ms.");
        return false;
      }

      try {
        Thread.sleep(sleepInterval);
        waitTimeCountForLog += sleepInterval;
      } catch (InterruptedException e) {
        LOG.log(Level.WARNING, "Thread sleep interrupted.", e);
      }
    }
  }

  /**
   * covert the given string to ip address object
   * @param ipStr
   * @return
   */
  private InetAddress convertStringToIP(String ipStr) {
    try {
      return InetAddress.getByName(ipStr);
    } catch (UnknownHostException e) {
      LOG.log(Level.SEVERE, "Can not convert the pod IP to InetAddress: " + ipStr, e);
      throw new RuntimeException(e);
    }
  }

  /**
   * build worker list by getting the pod list from the kubernetes master
   */
  private void buildWorkerList() {
    String namespace = KubernetesContext.namespace(config);
    String servicelabel = KubernetesUtils.createServiceLabelWithKey(jobID);
    int basePort = KubernetesContext.workerBasePort(config);

    V1PodList list = null;
    try {
      list = coreApi.listNamespacedPod(
          namespace, null, null, null, servicelabel, null, null, null, null);
    } catch (ApiException e) {
      String logMessage = "Exception when getting the pod list for the job: " + jobID + "\n"
          + "exCode: " + e.getCode() + "\n"
          + "responseBody: " + e.getResponseBody();
      LOG.log(Level.SEVERE, logMessage, e);
      throw new RuntimeException(e);
    }

    workerList.clear();

    for (V1Pod pod : list.getItems()) {
      String podName = pod.getMetadata().getName();
      if (!podName.startsWith(jobID)) {
        LOG.warning("A pod received that does not belong to this job. PodName: " + podName);
        continue;
      }

      InetAddress podIP = convertStringToIP(pod.getStatus().getPodIP());

      for (int i = 0; i < workersPerPod; i++) {
        int containerIndex = i;
        int workerID = calculateWorkerID(podName, containerIndex);
        JobMasterAPI.WorkerInfo workerNetworkInfo =
            WorkerInfoUtils.createWorkerInfo(workerID, pod.getStatus().getPodIP(),
                basePort + containerIndex, null);
        workerList.add(workerNetworkInfo);
      }
    }
  }

  /**
   * calculate a unique id for the worker using podName and containerName
   * worker ids start from 0 and go up sequentially
   * @param podName
   * @param containerIndex
   * @return
   */
  public int calculateWorkerID(String podName, int containerIndex) {
    int podNo = KubernetesUtils.indexFromName(podName);

    return podNo * workersPerPod + containerIndex;
  }

  public static void printWorkers(ArrayList<JobMasterAPI.WorkerInfo> workers) {

    StringBuffer buffer = new StringBuffer();
    buffer.append("Number of workers: " + workers.size() + "\n");
    int i = 0;
    for (JobMasterAPI.WorkerInfo worker: workers) {
      buffer.append(String.format("%d: workerID[%d] %s:%d\n",
          i++, worker.getWorkerID(), worker.getWorkerIP(), worker.getPort()));
    }

    LOG.info(buffer.toString());
  }

  /**
   * wait for all pods to run
   * @return
   */
  @Override
  public List<JobMasterAPI.WorkerInfo> getAllWorkers() throws TimeoutException {
    // first make sure all workers are in the list
    long timeLimitMilliSec = ControllerContext.maxWaitTimeForAllToJoin(config);
    long startTime = System.currentTimeMillis();
    if (workerList.size() < numberOfWorkers) {
      boolean listBuilt = buildWorkerListWaitForAll(timeLimitMilliSec);
      if (!listBuilt) {
        throw
            new TimeoutException("All workers have not joined the job on the time limit: "
            + timeLimitMilliSec + "ms.");
      }
    }

    ArrayList<String> podNameList = constructPodNameList();

    long duration = System.currentTimeMillis() - startTime;
    long remainingTimeLimit = timeLimitMilliSec - duration;

    boolean allRunning = waitUntilAllPodsRunning(podNameList, remainingTimeLimit);
    if (allRunning) {
      return workerList;
    } else {
      throw
          new TimeoutException("All workers have not joined the job on the specified time limit: "
              + timeLimitMilliSec + "ms.");
    }
  }

  /**
   * construct the list of pod names for this job
   */
  private ArrayList<String> constructPodNameList() {

    ArrayList<String> podNameList = new ArrayList<>();
    for (int i = 0; i < numberOfPods; i++) {
      String podName = jobID + "-" + i;
      podNameList.add(podName);
    }
    return podNameList;
  }

  /**
   * watch events until getting the Running event for all the pods in the list
   */
  public boolean waitUntilAllPodsRunning(ArrayList<String> podList, long timeoutMiliSec) {

    /** Pod Phases: Pending, Running, Succeeded, Failed, Unknown
     * ref: https://kubernetes.io/docs/concepts/workloads/pods/pod-lifecycle/#pod-phase */

    String phase = "Running";
    String namespace = KubernetesContext.namespace(config);
    String servicelabel = KubernetesUtils.createServiceLabelWithKey(jobID);
    Integer timeoutSeconds = (int) (timeoutMiliSec / 1000);
    Watch<V1Pod> watch = null;

    try {
      watch = Watch.createWatch(
          apiClient,
          coreApi.listNamespacedPodCall(namespace, null, null, null, servicelabel,
              null, null, timeoutSeconds, Boolean.TRUE, null, null),
          new TypeToken<Watch.Response<V1Pod>>() {
          }.getType());

    } catch (ApiException e) {
      String logMessage = "Exception when watching the pods for the job: " + jobID + "\n"
          + "exCode: " + e.getCode() + "\n"
          + "responseBody: " + e.getResponseBody();
      LOG.log(Level.SEVERE, logMessage, e);
      throw new RuntimeException(e);
    }

    boolean result = false;

    for (Watch.Response<V1Pod> item : watch) {
      if (item.object != null
          && podList.contains(item.object.getMetadata().getName())
          && phase.equals(item.object.getStatus().getPhase())) {

        // remove the pod from the list
        podList.remove(item.object.getMetadata().getName());
        LOG.log(Level.INFO, "Received pod Running event for the pod: "
            + item.object.getMetadata().getName());

        if (podList.size() == 0) {
          result = true;
          break;
        }
      }
    }

    try {
      watch.close();
    } catch (IOException e) {
      LOG.log(Level.SEVERE, "Exception closing watcher.", e);
    }

    return result;
  }

  /**
   * not implemented
   * @return
   */
  @Override
  public void waitOnBarrier() throws TimeoutException {
  }

}
