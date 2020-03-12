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
package edu.iu.dsc.tws.common.zk;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import java.util.concurrent.TimeUnit;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.stream.Collectors;

import com.google.protobuf.InvalidProtocolBufferException;

import org.apache.curator.framework.CuratorFramework;
import org.apache.curator.framework.recipes.cache.NodeCache;
import org.apache.curator.framework.recipes.cache.NodeCacheListener;
import org.apache.curator.framework.recipes.cache.PathChildrenCache;
import org.apache.curator.framework.recipes.cache.PathChildrenCacheEvent;
import org.apache.curator.framework.recipes.cache.PathChildrenCacheListener;
import org.apache.curator.framework.recipes.nodes.PersistentNode;
import org.apache.curator.utils.CloseableUtils;

import edu.iu.dsc.tws.api.config.Config;
import edu.iu.dsc.tws.api.exceptions.TimeoutException;
import edu.iu.dsc.tws.api.exceptions.Twister2Exception;
import edu.iu.dsc.tws.api.faulttolerance.FaultToleranceContext;
import edu.iu.dsc.tws.api.resource.ControllerContext;
import edu.iu.dsc.tws.api.resource.IAllJoinedListener;
import edu.iu.dsc.tws.api.resource.IJobMasterFailureListener;
import edu.iu.dsc.tws.api.resource.IScalerListener;
import edu.iu.dsc.tws.api.resource.IWorkerController;
import edu.iu.dsc.tws.api.resource.IWorkerFailureListener;
import edu.iu.dsc.tws.api.resource.IWorkerStatusUpdater;
import edu.iu.dsc.tws.proto.jobmaster.JobMasterAPI;
import edu.iu.dsc.tws.proto.jobmaster.JobMasterAPI.WorkerInfo;
import edu.iu.dsc.tws.proto.jobmaster.JobMasterAPI.WorkerState;
import edu.iu.dsc.tws.proto.system.job.JobAPI;

/**
 * we assume each worker is assigned a unique ID outside of this class.
 * <p>
 * We create following directories for a job by the submitting client.
 *   persistent state znode (directory)
 *   ephemeral state znode (directory)
 *   barrier znode (directory)
 *   events znode (directory)
 *
 * We create a persistent znode for each worker under the job persistent state znode.
 *   It has WorkerInfo object and the latest WorkerState
 *   Job Master watches the children of this znode for worker state changes.
 *   We also use these znodes for checking whether workers are coming from failure.
 *
 * We create an ephemeral znode for each worker under the job ephemeral state znode.
 *   The children of this znode is used for watching for worker failures and joins.
 *   Job Master watches the children of this znode.
 *
 * We create a barrier znode for each worker under the job barrier znode,
 *   when a barrier operation started.
 *   When all workers created their znodes under this directory, all workers arrived on the barrier.
 *   Job master watches the children of this znode and informs all workers by publishing an event.
 *   Each worker is responsible for deletion of their znodes under the barrier directory,
 *   when they proceed through the barrier.
 *   Job Master deletes the worker barrier znodes in case of scaling down or job termination.
 *
 * <p>
 * When the job completes, job terminator deletes all job znodes.
 * Sometimes, job terminator may not be invoked or it may fail before cleaning up job resources.
 * Therefore, when a job is being submitted,
 * we check whether the directories that will be used by the job exists,
 * if so, job submission will fail.
 * User need to first run job termination or should use another job name.
 *
 * <p>
 * Events
 * All events except the job scaling event is published on the events znode (directory)
 * as new child znodes with sequential numerical names
 * Workers get job scaling events by watching persistent job znode
 */

public class ZKWorkerController implements IWorkerController, IWorkerStatusUpdater {

  public static final Logger LOG = Logger.getLogger(ZKWorkerController.class.getName());

  // number of workers in this job
  private int numberOfWorkers;

  // name of this job
  private String jobID;

  // WorkerInfo object for this worker
  private WorkerInfo workerInfo;

  // initial state of this worker
  private WorkerState initialState;

  // the client to connect to ZK server
  private CuratorFramework client;

  // persistent ephemeral znode for this worker
  private PersistentNode workerEphemZNode;

  // children cache for events znode
  private PathChildrenCache eventsChildrenCache;

  // job znode cache for watching scaling events
  private NodeCache jobZnodeCache;

  // config object
  private Config config;
  private String rootPath;

  // all workers in the job, including completed and failed ones
  private List<WorkerInfo> workers;

  // a flag to show whether all workers joined the job
  // Initially it is false. It becomes true when all workers joined the job.
  // it becomes false when the job is scaled up,
  // it because true when all newly added workers joined
  private boolean allJoined = false;

  // last event index at the time of restart, if restarted
  private int numberOfPastEvents;

  // put past events into this sorted map
  // process only the latest allWorkersJoined event from the past events, if exist
  private TreeMap<Integer, JobMasterAPI.JobEvent> pastEvents;

  // synchronization object for waiting all workers to join the job
  private Object waitObjectAllJoined = new Object();

  // synchronization object for waiting all workers to arrive on the barrier
  private Object waitObjectBarrier = new Object();

  // it is set, when all workers arrived on the barrier
  // it is set to false, when the worker starts to wait on a new barrier
  private boolean barrierProceeded = false;

  // Inform worker failure events
  private IWorkerFailureListener failureListener;

  // Inform when all workers joined the job
  private IAllJoinedListener allJoinedListener;

  // Inform events related to the job master
  private List<IJobMasterFailureListener> jmFailureListeners = new LinkedList<>();

  // Inform scaling events
  private IScalerListener scalerListener;

  // some events may arrive after initializing the workerController but before
  // registering the listener,
  // we keep the last AllWorkersJoined and JobMasterRestarted events
  // to deliver when there is no proper listener
  private JobMasterAPI.AllWorkersJoined allJoinedEventCache;
  private JobMasterAPI.JobMasterRestarted jmRestartedCache;

  // we keep many fail events in the buffer to deliver later on
  private List<JobMasterAPI.JobEvent> failEventBuffer = new LinkedList<>();
  private List<Integer> scalingEventBuffer = new LinkedList<>();

  /**
   * Construct ZKWorkerController but do not initialize yet
   */
  public ZKWorkerController(Config config,
                            String jobID,
                            int numberOfWorkers,
                            WorkerInfo workerInfo) {
    this.config = config;
    this.jobID = jobID;
    this.numberOfWorkers = numberOfWorkers;
    this.workerInfo = workerInfo;

    this.rootPath = ZKContext.rootNode(config);
    workers = new ArrayList<>(numberOfWorkers);
  }

  /**
   * Initialize this ZKWorkerController
   * Connect to the ZK server
   * create a persistent znode for this worker
   *   set this WorkerInfo and its status in the body of that znode
   * create an ephemeral znode for this worker
   * create a cache for the persistent znode of the job for watching job scaling events
   *
   * initialState has to be either: WorkerState.STARTED or WorkerState.RESTARTED
   * <p>
   * The body of the persistent worker znode will be updated as the status of worker changes
   * from STARTED, COMPLETED,
   */
  public void initialize(WorkerState initState) throws Exception {

    this.initialState = initState;

    if (!(initState == WorkerState.STARTED || initState == WorkerState.RESTARTED)) {
      throw new Exception("initialState has to be either WorkerState.STARTED or "
          + "WorkerState.RESTARTED. Supplied value: " + initState);
    }

    try {
      String zkServerAddresses = ZKContext.serverAddresses(config);
      int sessionTimeoutMs = FaultToleranceContext.sessionTimeout(config);
      client = ZKUtils.connectToServer(zkServerAddresses, sessionTimeoutMs);

      // if this worker is started with scaling up, or
      // if it is coming from failure
      // we need to handle past events
      numberOfPastEvents = ZKEventsManager.getNumberOfPastEvents(client, rootPath, jobID);
      pastEvents = new TreeMap<>(Collections.reverseOrder());

      int wID = workerInfo.getWorkerID();

      if (isRestarted()) {
        // delete ephemeral worker znode from previous run, if there is any
        ZKEphemStateManager.removeEphemZNode(client, rootPath, jobID, wID);

        // if the worker barrier znode exist from the previous run, delete it
        // worker may have crashed during barrier operation
        if (ZKBarrierManager.existWorkerZNode(client, rootPath, jobID, wID)) {
          ZKBarrierManager.deleteWorkerZNode(client, rootPath, jobID, wID);
        }
      }

      String eventsDir = ZKUtils.eventsDir(rootPath, jobID);
      eventsChildrenCache = new PathChildrenCache(client, eventsDir, true);
      addEventsChildrenCacheListener(eventsChildrenCache);
      eventsChildrenCache.start();

      // We cache job znode data
      // So we will listen job scaling up/down
      String jobDir = ZKUtils.jobDir(rootPath, jobID);
      jobZnodeCache = new NodeCache(client, jobDir);
      addJobZnodeCacheListener();
      // start the cache and wait for it to get current data from zk server
      jobZnodeCache.start(true);

      // update numberOfWorkers from jobZnode
      // with scaling up/down, it may have been changed
      JobAPI.Job job = JobWithState.decode(jobZnodeCache.getCurrentData().getData()).getJob();
      if (numberOfWorkers != job.getNumberOfWorkers()) {
        numberOfWorkers = job.getNumberOfWorkers();
        LOG.info("numberOfWorkers updated from persJobZnode as: " + numberOfWorkers);
      }

      createWorkerZnode();

      LOG.info("This worker: " + workerInfo.getWorkerID() + " initialized successfully.");

    } catch (Exception e) {
      LOG.log(Level.SEVERE, "Exception when initializing ZKWorkerController", e);
      throw e;
    }
  }

  /**
   * create the znode for this worker
   */
  private void createWorkerZnode() throws Exception {
    String workersEphemDir = ZKUtils.ephemDir(rootPath, jobID);
    String workerPath = ZKUtils.workerPath(workersEphemDir, workerInfo.getWorkerID());

    workerEphemZNode =
        ZKEphemStateManager.createWorkerZnode(client, rootPath, jobID, workerInfo.getWorkerID());
    workerEphemZNode.start();
    try {
      workerEphemZNode.waitForInitialCreate(10000, TimeUnit.MILLISECONDS);
    } catch (InterruptedException e) {
      LOG.log(Level.SEVERE,
          "Could not create worker ephemeral znode: " + workerPath, e);
      throw new Exception("Could not create worker znode: " + workerPath, e);
    }

    String fullWorkerPath = workerEphemZNode.getActualPath();
    LOG.info("An ephemeral znode is created for this worker: " + fullWorkerPath);
  }

  public boolean isRestarted() {
    return initialState == WorkerState.RESTARTED;
  }

  /**
   * Update worker status with new state
   * return true if successful
   * <p>
   * Initially worker status is set as STARTED or RESTARTED.
   * Therefore, there is no need to call this method after starting this IWorkerController
   * This method should be called to change worker status to COMPLETED, FAILED, etc.
   */
  @Override
  public boolean updateWorkerStatus(WorkerState newStatus) {

    try {
      return ZKPersStateManager.updateWorkerStatus(
          client, rootPath, jobID, workerInfo, newStatus);

    } catch (Twister2Exception e) {
      LOG.log(Level.SEVERE, e.getMessage(), e);
      return false;
    }
  }

  @Override
  public WorkerState getWorkerStatusForID(int id) {
    WorkerWithState workerWS = null;
    try {
      workerWS = ZKPersStateManager.getWorkerWithState(client, rootPath, jobID, id);
    } catch (Twister2Exception e) {
      LOG.log(Level.SEVERE, e.getMessage(), e);
    }
    if (workerWS != null) {
      return workerWS.getState();
    }

    return null;
  }

  @Override
  public WorkerInfo getWorkerInfo() {
    return workerInfo;
  }

  @Override
  public WorkerInfo getWorkerInfoForID(int id) {

    // get it from local cache if allJoined,
    if (allJoined) {
      return workers.stream()
          .filter(wInfo -> wInfo.getWorkerID() == id)
          .findFirst()
          .orElse(null);
    }

    try {
      WorkerWithState workerWS =
          ZKPersStateManager.getWorkerWithState(client, rootPath, jobID, id);
      if (workerWS != null) {
        return workerWS.getInfo();
      }

    } catch (Twister2Exception e) {
      LOG.log(Level.SEVERE, e.getMessage(), e);
    }

    return null;
  }

  @Override
  public int getNumberOfWorkers() {
    return numberOfWorkers;
  }

  @Override
  public List<WorkerInfo> getJoinedWorkers() {

    List<WorkerWithState> workersWithState = null;
    try {
      workersWithState = ZKPersStateManager.getWorkers(client, rootPath, jobID);
    } catch (Twister2Exception e) {
      LOG.log(Level.SEVERE, e.getMessage(), e);
      return null;
    }

    return workersWithState
        .stream()
        .filter(workerWithState -> workerWithState.running())
        .map(workerWithState -> workerWithState.getInfo())
        .collect(Collectors.toList());
  }

  @Override
  public List<WorkerInfo> getAllWorkers() throws TimeoutException {
    // if all workers already joined, return the workers list
    if (allJoined) {
      return cloneWorkers();
    }

    // wait until all workers joined or time limit is reached
    long timeLimit = ControllerContext.maxWaitTimeForAllToJoin(config);
    long startTime = System.currentTimeMillis();

    long delay = 0;
    while (delay < timeLimit) {
      synchronized (waitObjectAllJoined) {
        try {
          waitObjectAllJoined.wait(timeLimit - delay);

          // proceeding with notification or timeout
          if (allJoined) {
            return cloneWorkers();
          } else {
            throw new TimeoutException("Not all workers joined the job on the given time limit: "
                + timeLimit + "ms.");
          }

        } catch (InterruptedException e) {
          delay = System.currentTimeMillis() - startTime;
        }
      }
    }

    if (allJoined) {
      return cloneWorkers();
    } else {
      throw new TimeoutException("Not all workers joined the job on the given time limit: "
          + timeLimit + "ms.");
    }
  }

  protected List<WorkerInfo> cloneWorkers() {

    List<WorkerInfo> clone = new LinkedList<>();
    for (WorkerInfo info : workers) {
      clone.add(info);
    }
    return clone;
  }

  /**
   * remove the workerInfo for the given ID if it exists, and add the given one
   */
  private void updateWorkerInfo(WorkerInfo info) {
    workers.removeIf(wInfo -> wInfo.getWorkerID() == info.getWorkerID());
    workers.add(info);
  }

  /**
   * add a single IWorkerFailureListener
   * if additional IWorkerFailureListener tried to be added,
   * do not add and return false
   */
  public boolean addFailureListener(IWorkerFailureListener iWorkerFailureListener) {
    if (this.failureListener != null) {
      return false;
    }

    this.failureListener = iWorkerFailureListener;

    // if failEventBuffer is not null, deliver all messages in a new thread
    if (!failEventBuffer.isEmpty()) {
      new Thread("Twister2-FailedEventSupplier") {
        @Override
        public void run() {
          for (JobMasterAPI.JobEvent jobEvent: failEventBuffer) {
            if (jobEvent.hasRestarted()) {
              failureListener.restarted(jobEvent.getRestarted().getWorkerInfo());
              LOG.fine("FAILED event delivered from cache.");
            } else if (jobEvent.hasFailed()) {
              failureListener.failed(jobEvent.getFailed().getWorkerID());
              LOG.fine("RESTARTED event delivered from cache.");
            }
          }
        }
      }.start();
    }

    return true;
  }

  /**
   * add a single IAllJoinedListener
   * if additional IAllJoinedListener tried to be added,
   * do not add and return false
   */
  public boolean addAllJoinedListener(IAllJoinedListener iAllJoinedListener) {
    if (allJoinedListener != null) {
      return false;
    }

    allJoinedListener = iAllJoinedListener;

    // if allJoinedEventToDeliver is not null, deliver that message in a new thread
    if (allJoinedEventCache != null) {
      new Thread("Twister2-AllJoinedEventSupplier") {
        @Override
        public void run() {
          allJoinedListener.allWorkersJoined(allJoinedEventCache.getWorkerInfoList());
          LOG.fine("AllWorkersJoined event delivered from cache.");
        }
      }.start();
    }

    return true;
  }

  /**
   * add a single IScalerListener
   * if additional IScalerListener tried to be added,
   * do not add and return false
   */
  public boolean addScalerListener(IScalerListener iScalerListener) {
    if (scalerListener != null) {
      return false;
    }

    scalerListener = iScalerListener;

    // if scalingEventBuffer is not null, deliver that message in a new thread
    if (!scalingEventBuffer.isEmpty()) {
      new Thread("Twister2-ScalingEventSupplier") {
        @Override
        public void run() {
          for (Integer change: scalingEventBuffer) {
            if (change > 0) {
              scalerListener.workersScaledUp(change);
              LOG.info("workersScaledUp event delivered from cache.");
            } else {
              scalerListener.workersScaledDown(Math.abs(change));
              LOG.info("workersScaledDown event delivered from cache.");
            }
          }
        }
      }.start();
    }

    return true;
  }

  /**
   * TODO: jm restarted implemented, but jm failed not implemented yet
   * Supports multiple IJobMasterFailureListeners
   */
  public boolean addJMFailureListener(IJobMasterFailureListener iJobMasterFailureListener) {

    jmFailureListeners.add(iJobMasterFailureListener);
    // if allJoinedEventToDeliver is not null, deliver that message in a new thread
    if (jmRestartedCache != null) {
      new Thread("Twister2-JMFailedEventSupplier") {
        @Override
        public void run() {
          jmFailureListeners.forEach(l -> l.restarted(jmRestartedCache.getJmAddress()));
          LOG.fine("JobMasterRestarted event delivered from cache.");
        }
      }.start();
    }
    return true;
  }

  /**
   * Get current list of workers
   * This list does not have the workers that have failed or already completed
   */
  public List<WorkerInfo> getCurrentWorkers() {

    List<WorkerWithState> workersWithState = null;
    try {
      workersWithState = ZKPersStateManager.getWorkers(client, rootPath, jobID);
    } catch (Twister2Exception e) {
      LOG.log(Level.SEVERE, e.getMessage(), e);
      return null;
    }

    return workersWithState
        .stream()
        .filter(workerWithState -> workerWithState.startedOrCompleted())
        .map(workerWithState -> workerWithState.getInfo())
        .collect(Collectors.toList());
  }

  /**
   * get number of current workers in the job as seen from this worker
   */
  public int getNumberOfCurrentWorkers() throws Exception {

    String workersEphemDir = ZKUtils.ephemDir(rootPath, jobID);
    int size = -1;
    try {
      size = client.getChildren().forPath(workersEphemDir).size();
    } catch (Exception e) {
      throw e;
    }
    return size;
  }

  /**
   * listen for additions to the job events directory in zk server
   * @param cache
   */
  private void addEventsChildrenCacheListener(PathChildrenCache cache) {
    PathChildrenCacheListener listener = new PathChildrenCacheListener() {

      public void childEvent(CuratorFramework clientOfEvent, PathChildrenCacheEvent event) {

        switch (event.getType()) {
          case CHILD_ADDED:
            eventPublished(event);
            break;

          default:
            // nothing to do
        }
      }
    };
    cache.getListenable().addListener(listener);
  }

  /**
   * listen for content changes in the persistent job znode body for scaling events
   */
  private void addJobZnodeCacheListener() {
    NodeCacheListener listener = new NodeCacheListener() {

      @Override
      public void nodeChanged() throws Exception {
        byte[] jobZnodeBodyBytes = jobZnodeCache.getCurrentData().getData();
        JobAPI.Job job = JobWithState.decode(jobZnodeBodyBytes).getJob();

        int change = job.getNumberOfWorkers() - numberOfWorkers;
        numberOfWorkers = job.getNumberOfWorkers();

        // job scaled up
        if (change > 0) {
          allJoined = false;
          if (scalerListener != null) {
            scalerListener.workersScaledUp(change);
          } else {
            scalingEventBuffer.add(change);
          }
          LOG.info("Job scaled up. new numberOfWorkers: " + numberOfWorkers);

          // job scaled down
        } else if (change < 0) {
          // remove scaled down workers from worker list
          workers.removeIf(wInfo -> wInfo.getWorkerID() >= job.getNumberOfWorkers());
          if (scalerListener != null) {
            scalerListener.workersScaledDown(Math.abs(change));
          } else {
            scalingEventBuffer.add(change);
          }
          LOG.info("Job scaled down. new numberOfWorkers: " + numberOfWorkers);
        }

      }
    };
    jobZnodeCache.getListenable().addListener(listener);
  }

  /**
   * when a new event is published on the events directory by the job master
   * take necessary actions
   */
  private void eventPublished(PathChildrenCacheEvent event) {

    String eventPath = event.getData().getPath();
    int eventIndex = ZKUtils.getWorkerIDFromPersPath(eventPath);

    byte[] eventData = event.getData().getData();
    JobMasterAPI.JobEvent jobEvent;
    try {
      jobEvent = ZKEventsManager.decodeJobEvent(eventData);
    } catch (InvalidProtocolBufferException e) {
      LOG.log(Level.SEVERE, "Could not decode received JobEvent from the path: " + eventPath, e);
      return;
    }

    // if there are already events when the worker has started, handle those
    if (eventIndex < numberOfPastEvents) {

      // ignore all past events for a worker that is not restarted
      if (!isRestarted()) {
        return;
      }

      // if the worker is restarted, put all events into the map object
      // we will use the latest allJoined event only from the past events
      pastEvents.put(eventIndex, jobEvent);
      if (pastEvents.size() == numberOfPastEvents) {
        processPastEvents();
      }

      return;
    }

    if (jobEvent.hasAllJoined()) {
      processAllJoinedEvent(jobEvent);
      return;
    }

    if (jobEvent.hasFailed()) {
      JobMasterAPI.WorkerFailed workerFailed = jobEvent.getFailed();
      // if the event is about this worker, ignore it
      if (workerFailed.getWorkerID() == workerInfo.getWorkerID()) {
        return;
      }

      LOG.info(String.format("Worker[%s] FAILED. ", workerFailed.getWorkerID()));

      if (failureListener != null) {
        failureListener.failed(workerFailed.getWorkerID());
      } else {
        failEventBuffer.add(jobEvent);
      }
    }

    if (jobEvent.hasRestarted()) {

      JobMasterAPI.WorkerRestarted workerRestarted = jobEvent.getRestarted();
      // if the event is about this worker, ignore it
      if (workerRestarted.getWorkerInfo().getWorkerID() == workerInfo.getWorkerID()) {
        return;
      }

      LOG.info(String.format("Worker[%s] RESTARTED.",
          workerRestarted.getWorkerInfo().getWorkerID()));

      updateWorkerInfo(workerRestarted.getWorkerInfo());

      if (failureListener != null) {
        failureListener.restarted(workerRestarted.getWorkerInfo());
      } else {
        failEventBuffer.add(jobEvent);
      }
    }

    if (jobEvent.hasAllArrived()) {
      barrierProceeded = true;
      synchronized (waitObjectBarrier) {
        waitObjectBarrier.notify();
      }

      LOG.info("AllArrivedOnBarrier event received. Current numberOfWorkers: " + numberOfWorkers);
    }

    if (jobEvent.hasJmRestarted()) {
      JobMasterAPI.JobMasterRestarted jmRestarted = jobEvent.getJmRestarted();
      LOG.info("JobMasterRestarted event received. JM Address: " + jmRestarted.getJmAddress());

      if (jmFailureListeners.size() != 0) {
        jmFailureListeners.forEach(l -> l.restarted(jmRestarted.getJmAddress()));
      } else {
        jmRestartedCache = jmRestarted;
      }
    }
  }

  /**
   * look for an AllJoined event among past events,
   * if there is, process the one with highest index
   * ignore all other past events
   * PastEvents are sorted in reverse order, so we process the first one we find
   */
  private void processPastEvents() {
    for (Map.Entry<Integer, JobMasterAPI.JobEvent> entry: pastEvents.entrySet()) {
      if (entry.getValue().hasAllJoined()) {
        LOG.info("AllWorkersJoined event from past events. Event index: " + entry.getKey());
        processAllJoinedEvent(entry.getValue());
        return;
      }
    }
  }

  private void processAllJoinedEvent(JobMasterAPI.JobEvent jobEvent) {

    JobMasterAPI.AllWorkersJoined allJoinedEvent = jobEvent.getAllJoined();
    workers.clear();
    workers.addAll(allJoinedEvent.getWorkerInfoList());
    allJoined = true;

    synchronized (waitObjectAllJoined) {
      waitObjectAllJoined.notify();
    }

    if (allJoinedListener != null) {
      allJoinedListener.allWorkersJoined(allJoinedEvent.getWorkerInfoList());
    } else {
      allJoinedEventCache = allJoinedEvent;
    }
  }

  /**
   * All workers create a znode on the job barrier directory
   * Job master watches for directory creations/removals
   * when the number of znodes on that directory reaches the number of workers in thr job,
   * Job master publishes AllArrivedOnBarrier event
   * Workers proceed when they get this event or when they time out
   * <p>
   * Worker remove their znodes after they proceed through the barrier
   * so that they can wait on the barrier again
   * Workers are responsible for creating and removing znodes on the barrier
   * Job master removes barrier znode after the job completion or scale down.
   *
   * if timeout is reached, throws TimeoutException.
   */
  @Override
  public void waitOnBarrier() throws TimeoutException {

    barrierProceeded = false;

    try {
      ZKBarrierManager.createWorkerZNode(client, rootPath, jobID, workerInfo.getWorkerID());
    } catch (Twister2Exception e) {
      LOG.log(Level.SEVERE, e.getMessage(), e);
      return;
    }

    // wait until all workers joined or time limit is reached
    long timeLimit = ControllerContext.maxWaitTimeOnBarrier(config);
    long startTime = System.currentTimeMillis();

    long delay = 0;
    while (delay < timeLimit) {
      synchronized (waitObjectBarrier) {
        try {
          if (!barrierProceeded) {
            waitObjectBarrier.wait(timeLimit - delay);
            break;
          }

        } catch (InterruptedException e) {
          delay = System.currentTimeMillis() - startTime;
        }
      }
    }

    // delete barrier znode in any case
    try {
      ZKBarrierManager.deleteWorkerZNode(client, rootPath, jobID, workerInfo.getWorkerID());
    } catch (Twister2Exception e) {
      LOG.log(Level.SEVERE, e.getMessage(), e);
    }

    if (barrierProceeded) {
      return;
    } else {
      throw new TimeoutException("Barrier timed out. Not all workers arrived on time limit: "
          + timeLimit + "ms.");
    }
  }

  @Override
  public IWorkerFailureListener getFailureListener() {
    return failureListener;
  }

  /**
   * close all local entities.
   */
  public void close() {
    CloseableUtils.closeQuietly(workerEphemZNode);
    CloseableUtils.closeQuietly(eventsChildrenCache);
    CloseableUtils.closeQuietly(jobZnodeCache);
  }

}
