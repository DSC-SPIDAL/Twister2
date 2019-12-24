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
package edu.iu.dsc.tws.rsched.schedulers.nomad;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.logging.Level;
import java.util.logging.Logger;

import com.hashicorp.nomad.apimodel.Job;
import com.hashicorp.nomad.apimodel.JobListStub;
import com.hashicorp.nomad.apimodel.NetworkResource;
import com.hashicorp.nomad.apimodel.Port;
import com.hashicorp.nomad.apimodel.Resources;
import com.hashicorp.nomad.apimodel.Task;
import com.hashicorp.nomad.apimodel.TaskGroup;
import com.hashicorp.nomad.apimodel.Template;
import com.hashicorp.nomad.javasdk.EvaluationResponse;
import com.hashicorp.nomad.javasdk.NomadApiClient;
import com.hashicorp.nomad.javasdk.NomadApiConfiguration;
import com.hashicorp.nomad.javasdk.NomadException;
import com.hashicorp.nomad.javasdk.ServerQueryResponse;

import edu.iu.dsc.tws.api.config.Config;
import edu.iu.dsc.tws.api.scheduler.IController;
import edu.iu.dsc.tws.api.scheduler.SchedulerContext;
import edu.iu.dsc.tws.master.JobMasterContext;
import edu.iu.dsc.tws.proto.system.job.JobAPI;
import edu.iu.dsc.tws.rsched.uploaders.scp.ScpContext;
import edu.iu.dsc.tws.rsched.utils.JobUtils;

public class NomadController implements IController {
  private static final Logger LOG = Logger.getLogger(NomadController.class.getName());

  private Config config;

  private boolean isVerbose;

  public NomadController(boolean isVerbose) {
    this.isVerbose = isVerbose;
  }

  @Override
  public void initialize(Config cfg) {
    this.config = cfg;
  }

  @Override
  public boolean start(JobAPI.Job job) {
    String uri = NomadContext.nomadSchedulerUri(config);
    NomadApiClient nomadApiClient = new NomadApiClient(
        new NomadApiConfiguration.Builder().setAddress(uri).build());

    Job nomadJob = getJob(job);
    try {
      EvaluationResponse response = nomadApiClient.getJobsApi().register(nomadJob);
      LOG.log(Level.INFO, "Submitted job to nomad: " + response);
    } catch (IOException | NomadException e) {
      LOG.log(Level.SEVERE, "Failed to submit the job: ", e);
    } finally {
      closeClient(nomadApiClient);
    }
    return false;
  }

  @Override
  public void close() {
  }

  @Override
  public boolean kill(JobAPI.Job job) {
    String jobID = job.getJobId();

    String uri = NomadContext.nomadSchedulerUri(config);
    LOG.log(Level.INFO, "Killing Job " + jobID);
    NomadApiClient nomadApiClient = new NomadApiClient(
        new NomadApiConfiguration.Builder().setAddress(uri).build());
    try {
      Job nomadJob = getRunningJob(nomadApiClient, job.getJobId());
      if (nomadJob == null) {
        LOG.log(Level.INFO, "Cannot find the running job: " + job.getJobId());
        return false;
      }
      nomadApiClient.getJobsApi().deregister(nomadJob.getId());
    } catch (RuntimeException | IOException | NomadException e) {
      LOG.log(Level.SEVERE, "Failed to terminate job " + jobID
          + " with error: " + e.getMessage(), e);
      return false;
    } finally {
      closeClient(nomadApiClient);
    }
    return true;
  }

  private void closeClient(NomadApiClient nomadApiClient) {
    try {
      if (nomadApiClient != null) {
        nomadApiClient.close();
      }
    } catch (IOException e) {
      LOG.log(Level.SEVERE, String.format("Error closing client: %s", e.getMessage()), e);
    }
  }

  private Job getJob(JobAPI.Job job) {
    String jobID = job.getJobId();
    Job nomadJob = new Job();
    nomadJob.setId(jobID);
    nomadJob.setName(jobID);
    nomadJob.setType("batch");
    nomadJob.addTaskGroups(getTaskGroup(job));
    nomadJob.setDatacenters(Arrays.asList(NomadContext.NOMAD_DEFAULT_DATACENTER));
    nomadJob.setMeta(getMetaData(job));
    return nomadJob;
  }

  private static List<JobListStub> getRunningJobList(NomadApiClient apiClient) {
    ServerQueryResponse<List<JobListStub>> response;
    try {
      response = apiClient.getJobsApi().list();
    } catch (IOException | NomadException e) {
      LOG.log(Level.SEVERE, "Error when attempting to fetch job list", e);
      throw new RuntimeException(e);
    }
    return response.getValue();
  }

  private static Job getRunningJob(NomadApiClient apiClient, String jobID) {
    List<JobListStub> jobs = getRunningJobList(apiClient);
    for (JobListStub job : jobs) {
      Job jobActual;
      try {
        jobActual = apiClient.getJobsApi().info(job.getId()).getValue();
      } catch (IOException | NomadException e) {
        String msg = "Failed to retrieve job info for job " + job.getId()
            + " part of job " + jobID;
        LOG.log(Level.SEVERE, msg, e);
        throw new RuntimeException(msg, e);
      }
      if (jobID.equals(jobActual.getName())) {
        return jobActual;
      }
    }
    return null;
  }

  private TaskGroup getTaskGroup(JobAPI.Job job) {
    TaskGroup taskGroup = new TaskGroup();
    if (JobMasterContext.jobMasterRunsInClient(config)) {
      taskGroup.setCount(job.getNumberOfWorkers());
    } else {
      taskGroup.setCount(job.getNumberOfWorkers() + 1);
    }
    taskGroup.setName(job.getJobId());
    taskGroup.addTasks(getShellDriver(job));
    return taskGroup;
  }

  private static Map<String, String> getMetaData(JobAPI.Job job) {
    String jobID = job.getJobId();
    Map<String, String> metaData = new HashMap<>();
    metaData.put(NomadContext.NOMAD_JOB_NAME, jobID);
    return metaData;
  }

  private Task getShellDriver(JobAPI.Job job) {
    String taskName = job.getJobId();
    Task task = new Task();
    // get the job working directory
    String workingDirectory = NomadContext.workingDirectory(config);
    String jobWorkingDirectory = Paths.get(workingDirectory, job.getJobId()).toString();
    String configDirectoryName = Paths.get(workingDirectory,
        job.getJobId(), SchedulerContext.clusterType(config)).toString();

    String corePackageFile = SchedulerContext.temporaryPackagesPath(config) + "/"
        + SchedulerContext.corePackageFileName(config);
    String jobPackageFile = SchedulerContext.temporaryPackagesPath(config) + "/"
        + SchedulerContext.jobPackageFileName(config);

    String nomadScriptContent = getNomadScriptContent(config, configDirectoryName);

    task.setName(taskName);
    task.setDriver("raw_exec");
    task.addConfig(NomadContext.NOMAD_TASK_COMMAND, NomadContext.SHELL_CMD);
    String[] args = workerProcessCommand(workingDirectory, job);
    task.addConfig(NomadContext.NOMAD_TASK_COMMAND_ARGS, args);
    Template template = new Template();
    template.setEmbeddedTmpl(nomadScriptContent);
    template.setDestPath(NomadContext.NOMAD_SCRIPT_NAME);
    task.addTemplates(template);

    Resources resourceReqs = new Resources();
    String portNamesConfig = NomadContext.networkPortNames(config);
    String[] portNames = portNamesConfig.split(",");
    // configure nomad to allocate dynamic ports
    Port[] ports = new Port[portNames.length];
    int i = 0;
    for (String p : portNames) {
      ports[i] = new Port().setLabel(p);
      i++;
    }
    NetworkResource networkResource = new NetworkResource();
    networkResource.addDynamicPorts(ports);
    resourceReqs.addNetworks(networkResource);
    JobAPI.ComputeResource computeResource = JobUtils.getComputeResource(job, 0);
    if (computeResource == null) {
      LOG.log(Level.SEVERE, "Error: there is no compute resource");
      return null;
    }
    int  cpu = (int) computeResource.getCpu();
    int  disk = (int) computeResource.getDiskGigaBytes();
    int memory = computeResource.getRamMegaBytes();

    resourceReqs.setCpu(cpu * 200);
    resourceReqs.setMemoryMb(memory);
    resourceReqs.setDiskMb(disk * 1024);

    LOG.log(Level.INFO, "Compute resources are " + cpu + " " + memory + " " + disk);
    Map<String, String> envVars = new HashMap<>();
    envVars.put(NomadContext.WORKING_DIRECTORY_ENV,
        NomadContext.workingDirectory(config));

    if (!NomadContext.sharedFileSystem(config)) {
      envVars.put(NomadContext.DOWNLOAD_PACKAGE_ENV, "false");
    } else {
      envVars.put(NomadContext.DOWNLOAD_PACKAGE_ENV, "true");
    }
    // we are putting the core packages as env variable
    envVars.put(NomadContext.CORE_PACKAGE_ENV, corePackageFile);
    envVars.put(NomadContext.JOB_PACKAGE_ENV, jobPackageFile);

    task.setEnv(envVars);
    task.setResources(resourceReqs);
    return task;
  }

  private String getScriptPath(Config cfg, String jWorkingDirectory) {
    String shellScriptName = NomadContext.shellScriptName(cfg);
    return Paths.get(jWorkingDirectory, shellScriptName).toString();
  }

  private String getNomadScriptContent(Config cfg, String jConfigDir) {
    String shellDirectoryPath = getScriptPath(cfg, jConfigDir);
    try {
      return new String(Files.readAllBytes(Paths.get(
          shellDirectoryPath)), StandardCharsets.UTF_8);
    } catch (IOException e) {
      String msg = "Failed to read nomad script from "
          + NomadContext.shellScriptName(cfg) + " . Please check file path! - "
          + shellDirectoryPath;
      LOG.log(Level.SEVERE, msg, e);
      throw new RuntimeException(msg, e);
    }
  }

  private String[] workerProcessCommand(String workingDirectory, JobAPI.Job job) {

    String twister2Home = Paths.get(workingDirectory, job.getJobId()).toString();
    //String configDirectoryName = Paths.get(workingDirectory,
    //    job.getJobId(), SchedulerContext.clusterType(config)).toString();
    String configDirectoryName = "";
    // lets construct the mpi command to launch
    List<String> mpiCommand = workerProcessCommand(getScriptPath(config, configDirectoryName));
    Map<String, Object> map = workerCommandArguments(config, workingDirectory, job);
    String jobId = job.getJobId();
    String runIncLient = null;
    if (JobMasterContext.jobMasterRunsInClient(config)) {
      runIncLient = "true";
    } else {
      runIncLient = "false";
    }
    //mpiCommand.add(map.get("procs").toString());
    mpiCommand.add(runIncLient);
    mpiCommand.add(map.get("java_props").toString());
    mpiCommand.add(map.get("classpath").toString());
    mpiCommand.add(map.get("container_class").toString());
    mpiCommand.add(job.getJobId());
    mpiCommand.add(twister2Home);
    mpiCommand.add(jobId);

    mpiCommand.add(SchedulerContext.jobPackageUrl(config));
    mpiCommand.add(SchedulerContext.corePackageUrl(config));
    mpiCommand.add(SchedulerContext.downloadMethod(config));
    mpiCommand.add(ScpContext.uploaderJobDirectory(config));


    LOG.log(Level.FINE, String.format("Command %s", mpiCommand));

    String[] array = new String[mpiCommand.size()];
    for (int i = 0; i < array.length; i++) {
      array[i] = mpiCommand.get(i);
    }
    return array;
  }

  private Map<String, Object> workerCommandArguments(Config cfg, String workingDirectory,
                                                     JobAPI.Job job) {
    Map<String, Object> commands = new HashMap<>();
    // lets get the configurations
    commands.put("procs", job.getNumberOfWorkers());

    String jobClassPath = JobUtils.jobClassPath(cfg, job, workingDirectory);
    LOG.log(Level.FINE, "Job class path: " + jobClassPath);
    String systemClassPath = JobUtils.systemClassPath(cfg);
    String classPath = jobClassPath + ":" + systemClassPath;
    commands.put("classpath", classPath);
    commands.put("java_props", "");
    commands.put("container_class", job.getWorkerClassName());

    return commands;
  }

  private List<String> workerProcessCommand(String mpiScript) {
    List<String> slurmCmd;
    slurmCmd = new ArrayList<>(Collections.singletonList(mpiScript));
    return slurmCmd;
  }
  public String createPersistentJobDirName(String jobID) {
    return SchedulerContext.nfsServerPath(config) + "/" + jobID;
  }
}
